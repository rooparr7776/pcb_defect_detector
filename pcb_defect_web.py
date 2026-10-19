"""Interactive Web Dashboard for PCB Defect Detection & Classification.

Features:
- Automatic Image Registration (ORB, SIFT, ECC) with interactive misalignment simulation
- Illumination-Invariant Differencing (Sobel Gradient, SSIM Structural, Hybrid, CLAHE) with lighting simulation
- Two-Stage Hybrid Detection: Classical Candidate RoIs + Stage-2 CNN Defect Classification
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from pcb_defect_detection import (
    DEFAULT_DATA_ROOT,
    calculate_metrics,
    detect_defects,
    discover_samples,
    draw_results,
    load_annotations,
)
from pcb_registration import (
    draw_registration_matches,
    simulate_misalignment,
)
from pcb_differencing import (
    compute_difference_map,
    simulate_illumination_change,
)
from pcb_classifier import (
    DEFECT_CLASSES,
    DEFECT_COLORS,
    load_defect_classifier,
)


st.set_page_config(page_title="PCB Defect Detection & Classification", layout="wide")


@st.cache_data(show_spinner="Scanning dataset image pairs and annotations...")
def get_samples(data_root):
    return discover_samples(Path(data_root))


@st.cache_resource(show_spinner="Loading Stage-2 CNN Defect Classifier...")
def get_cached_classifier():
    return load_defect_classifier()


st.title("PCB Surface Defect Detection & Classification")
st.caption("Classical Image Processing + Auto-Registration + Illumination Invariance + Stage-2 CNN")

# Sidebar Configuration
with st.sidebar:
    st.header("1. Dataset Selection")
    data_root = st.text_input("PCBData folder", str(DEFAULT_DATA_ROOT))
    samples = get_samples(data_root)
    if not samples:
        st.error("No complete image pairs and annotations found in this folder.")
        st.stop()

    sample_id = st.selectbox("Sample ID", list(samples))

    st.divider()
    st.header("2. Auto-Registration")
    enable_reg = st.toggle("Enable Auto-Registration", value=False, help="Align test image to template before differencing.")
    reg_method = "none"
    if enable_reg:
        reg_method = st.selectbox("Registration Method", ["orb", "sift", "ecc"], index=0)

    with st.expander("Misalignment Simulator (Test Registration)"):
        sim_shift_x = st.slider("Shift X (pixels)", -30, 30, 0, 1)
        sim_shift_y = st.slider("Shift Y (pixels)", -30, 30, 0, 1)
        sim_rot = st.slider("Rotation (degrees)", -10.0, 10.0, 0.0, 0.5)

    st.divider()
    st.header("3. Differencing Engine")
    diff_method_name = st.selectbox(
        "Algorithm",
        [
            "Raw Absolute Difference",
            "Sobel Gradient (Illumination-Invariant)",
            "SSIM Structural (Contrast & Lighting Invariant)",
            "Hybrid (Gradient + SSIM Fusion)",
        ],
        index=0,
    )
    diff_key_map = {
        "Raw Absolute Difference": "raw",
        "Sobel Gradient (Illumination-Invariant)": "gradient",
        "SSIM Structural (Contrast & Lighting Invariant)": "ssim",
        "Hybrid (Gradient + SSIM Fusion)": "hybrid",
    }
    diff_method = diff_key_map[diff_method_name]
    use_clahe = st.checkbox("Apply CLAHE Equalization", value=False, help="Normalizes local PCB contrast variations.")

    with st.expander("Illumination Simulator (Test Lighting Invariance)"):
        sim_brightness = st.slider("Brightness Offset", -80, 80, 0, 5)
        sim_contrast = st.slider("Contrast Factor", 0.5, 2.0, 1.0, 0.05)

    st.divider()
    st.header("4. Stage-2 CNN Classifier")
    enable_cnn = st.toggle("Enable CNN Classification", value=True, help="Classifies candidate boxes into 6 DeepPCB defect types.")
    conf_filter = st.slider("Minimum Confidence", 0.0, 1.0, 0.25, 0.05) if enable_cnn else 0.0

    st.divider()
    st.header("5. Detection Thresholds")
    min_area = st.slider("Minimum contour area", 0, 300, 12, 1, help="Discard small speckles smaller than this pixel area.")
    iou_threshold = st.slider("IoU match threshold", 0.10, 0.90, 0.33, 0.01, help="Overlap needed to match a ground-truth label.")


# Load Images
template_path, test_path, annotation_path = samples[sample_id]
template_bgr = cv2.imread(str(template_path))
test_bgr = cv2.imread(str(test_path))

if template_bgr is None or test_bgr is None:
    st.error(f"Could not read the image pair for sample {sample_id}.")
    st.stop()

# Apply Misalignment Simulation if set
test_input_bgr = test_bgr.copy()
if sim_shift_x != 0 or sim_shift_y != 0 or sim_rot != 0.0:
    test_input_bgr, _ = simulate_misalignment(
        test_input_bgr, shift_x=sim_shift_x, shift_y=sim_shift_y, angle_deg=sim_rot
    )

# Apply Illumination Simulation if set
if sim_brightness != 0 or sim_contrast != 1.0:
    test_input_bgr = simulate_illumination_change(
        test_input_bgr, brightness_offset=sim_brightness, contrast_factor=sim_contrast
    )

# Load Annotations & Model
ground_truth = load_annotations(annotation_path)
classifier_model = get_cached_classifier() if enable_cnn else None

# Run Detection Pipeline
(
    template_smooth,
    test_smooth,
    difference,
    thresholded,
    cleaned,
    detections,
    aligned_test,
    reg_info,
    classified_results,
) = detect_defects(
    template_bgr=template_bgr,
    test_bgr=test_input_bgr,
    min_area=min_area,
    registration=reg_method,
    diff_method=diff_method,
    use_clahe=use_clahe,
    classifier_model=classifier_model,
)

# Filter low-confidence predictions if CNN enabled
if enable_cnn and classified_results and conf_filter > 0.0:
    filtered_results = [res for res in classified_results if res["confidence"] >= conf_filter]
else:
    filtered_results = classified_results

# Compute Metrics
metrics = calculate_metrics(
    detections=detections,
    ground_truth=ground_truth,
    iou_threshold=iou_threshold,
    classified_results=filtered_results if enable_cnn else None,
)

# Main View: Tabs for clean organization
tab_overview, tab_pipeline, tab_analytics, tab_diagnostics = st.tabs([
    "Inspection Overview", "Pipeline Stages", "Defect Table", "Novelty Demonstrator"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tab_overview:
    st.subheader(f"Inspection Result: Sample {sample_id}")

    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6, col_m7 = st.columns(7)
    col_m1.metric("Candidates", len(filtered_results) if enable_cnn else len(detections))
    col_m2.metric("Correct (TP)", metrics["true_positives"])
    col_m3.metric("Extra (FP)", metrics["false_positives"])
    col_m4.metric("Missed (FN)", metrics["false_negatives"])
    col_m5.metric("Precision", f"{metrics['precision']:.1%}")
    col_m6.metric("Recall", f"{metrics['recall']:.1%}")
    if metrics["class_accuracy"] is not None:
        col_m7.metric("Class Accuracy", f"{metrics['class_accuracy']:.1%}")
    else:
        col_m7.metric("Mean IoU", f"{metrics['mean_iou']:.2f}")

    if reg_info.get("method") != "none":
        st.success(f"Auto-Registration ({reg_info['method'].upper()}) active: {reg_info.get('inliers', 0)} inlier keypoints matched.")
    if diff_method != "raw":
        st.info(f"Illumination-invariant differencing active ({diff_method_name}).")

    result_rgb = draw_results(
        aligned_test,
        detections=detections,
        ground_truth=ground_truth,
        classified_results=filtered_results if enable_cnn else None,
    )

    ov_col1, ov_col2 = st.columns(2)
    with ov_col1:
        st.markdown("**Clean Reference Template**")
        st.image(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with ov_col2:
        st.markdown("**Test Board with Detected Defects**")
        st.image(result_rgb, use_container_width=True)
        st.caption("Green boxes = Ground truth annotations. Colored boxes = Detected & Classified defects.")

    # Defect Class Legend
    st.markdown("##### Defect Class Color Legend:")
    legend_html = " ".join([
        f"<span style='background-color:rgb{DEFECT_COLORS[cid][2]},{DEFECT_COLORS[cid][1]},{DEFECT_COLORS[cid][0]}; "
        f"color:white; padding:3px 8px; border-radius:4px; margin-right:6px; font-weight:600; font-size:12px;'>"
        f"{cname.upper()}</span>"
        for cid, cname in DEFECT_CLASSES.items() if cid != 0
    ])
    st.markdown(legend_html, unsafe_allow_html=True)


# ----------------- TAB 2: PIPELINE STAGES -----------------
with tab_pipeline:
    st.subheader("5-Stage Visual Processing Pipeline")

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.markdown("**1. Test Board (Before Alignment)**")
        st.image(cv2.cvtColor(test_input_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Input board (may have vibration shift or lighting changes).")

    with pcol2:
        st.markdown(f"**2. Aligned Test Board ({reg_method.upper()})**")
        st.image(cv2.cvtColor(aligned_test, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Warped into exact pixel alignment using Homography.")

    with pcol3:
        st.markdown(f"**3. Difference Heatmap ({diff_method.upper()})**")
        diff_vis = cv2.applyColorMap(difference, cv2.COLORMAP_INFERNO)
        st.image(cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Bright regions signify localized changes.")

    pcol4, pcol5, pcol6 = st.columns(3)
    with pcol4:
        st.markdown("**4. Otsu Threshold Mask**")
        st.image(thresholded, use_container_width=True)
        st.caption("Binarized mask separating changed pixels from background.")

    with pcol5:
        st.markdown("**5. Cleaned Morphology Mask**")
        st.image(cleaned, use_container_width=True)
        st.caption("Opening removed speckles; closing bridged fragments.")

    with pcol6:
        st.markdown("**6. Final Detections & Class Labels**")
        st.image(result_rgb, use_container_width=True)
        st.caption("Candidate RoIs classified by CNN with confidence.")


# ----------------- TAB 3: DEFECT TABLE -----------------
with tab_analytics:
    st.subheader("Detailed Defect Catalog")
    if filtered_results:
        table_rows = []
        for idx, item in enumerate(filtered_results, start=1):
            x1, y1, x2, y2 = item["box"]
            table_rows.append({
                "Defect ID": f"D{idx}",
                "Class Name": item["class_name"].capitalize(),
                "Confidence": f"{item['confidence']:.1%}",
                "Bounding Box (x1, y1, x2, y2)": f"({x1}, {y1}, {x2}, {y2})",
                "Width (px)": x2 - x1,
                "Height (px)": y2 - y1,
                "Area (px²)": (x2 - x1) * (y2 - y1),
            })
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No defect candidates passed the current area and confidence thresholds.")


# ----------------- TAB 4: NOVELTY DEMONSTRATOR -----------------
with tab_diagnostics:
    st.subheader("Interactive Innovation Lab")

    st.markdown("#### 1. Auto-Registration Feature Matches")
    if reg_info.get("matches") and len(reg_info["matches"]) > 0:
        match_vis = draw_registration_matches(
            test_input_bgr,
            reg_info["kp_test"],
            template_bgr,
            reg_info["kp_temp"],
            reg_info["matches"],
            max_draw=35,
        )
        st.image(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Green lines depict top inlier feature matches between Test (Left) and Template (Right) used to solve Homography H ({reg_info['inliers']} inliers).")
    else:
        st.info("Enable Auto-Registration in the sidebar (and optionally add a test shift/rotation) to view inlier feature matching lines.")

    st.divider()
    st.markdown("#### 2. Illumination Invariance: Raw vs. Invariant Differencing")
    dcol1, dcol2 = st.columns(2)
    raw_diff = compute_difference_map(template_smooth, test_smooth, method="raw")
    inv_diff = compute_difference_map(template_smooth, test_smooth, method=diff_method if diff_method != "raw" else "ssim")

    with dcol1:
        st.markdown("**Standard Raw Differencing**")
        st.image(cv2.cvtColor(cv2.applyColorMap(raw_diff, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Mean error: {raw_diff.mean():.2f}. (Vulnerable to lighting fluctuations).")
    with dcol2:
        st.markdown(f"**{diff_method.capitalize() if diff_method != 'raw' else 'SSIM'} Differencing (Invariant)**")
        st.image(cv2.cvtColor(cv2.applyColorMap(inv_diff, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Mean error: {inv_diff.mean():.2f}. (Isolates true structural/gradient defects).")