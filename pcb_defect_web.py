"""Industrial Automated Optical Inspection (AOI) Dashboard for PCB Defect Detection & Classification.

Features:
- Automatic Image Registration (ORB, SIFT, ECC) with interactive misalignment simulation
- Illumination-Invariant Differencing (Sobel Gradient, SSIM Structural, Hybrid, CLAHE) with lighting simulation
- Two-Stage Hybrid Detection: Classical Candidate RoIs + Stage-2 CNN Defect Classification
- Interactive Defect Deep-Dive Zoom Inspector with Electrical Severity Diagnostics
- ISO-Compliant Quality Inspection Audit Report Generator
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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


# Page Configuration
st.set_page_config(
    page_title="PCB AOI System Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Industrial Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top banner */
    .aoi-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.25);
    }
    .aoi-title {
        color: #f8fafc;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .aoi-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 12px;
    }

    /* Status Pills */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-pass {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-fail {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .pill-info {
        background-color: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }
    .pill-purple {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Metric Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }
    @media (max-width: 900px) {
        .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }

    /* Defect deep-dive cards */
    .diagnosis-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
    }
    .severity-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .sev-critical { background: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }
    .sev-moderate { background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }
    .sev-minor { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Scanning dataset image pairs and annotations...")
def get_samples(data_root):
    return discover_samples(Path(data_root))


@st.cache_resource(show_spinner="Loading Stage-2 CNN Defect Classifier...")
def get_cached_classifier():
    return load_defect_classifier()


# ===================== SIDEBAR CONTROLS =====================
with st.sidebar:
    st.markdown("### ⚙️ Inspection Configuration")

    # 1. Dataset Selection
    with st.expander("📁 1. Board & Dataset", expanded=True):
        data_root = st.text_input("PCBData folder", str(DEFAULT_DATA_ROOT))
        samples = get_samples(data_root)
        if not samples:
            st.error("No complete image pairs and annotations found in this folder.")
            st.stop()
        sample_id = st.selectbox("Sample ID", list(samples))

    # 2. Auto-Registration
    with st.expander("🎯 2. Auto-Registration (Homography)", expanded=True):
        enable_reg = st.toggle("Enable Auto-Registration", value=True, help="Aligns test board onto template space using keypoint homography.")
        reg_method = "orb"
        if enable_reg:
            reg_method = st.selectbox("Algorithm", ["orb", "sift", "ecc"], index=0, help="ORB is fast; SIFT handles large rotations; ECC does subpixel optimization.")
        else:
            reg_method = "none"

        st.caption("Misalignment Simulator:")
        sim_shift_x = st.slider("Simulate Shift X (px)", -30, 30, 0, 1)
        sim_shift_y = st.slider("Simulate Shift Y (px)", -30, 30, 0, 1)
        sim_rot = st.slider("Simulate Rotation (°)", -10.0, 10.0, 0.0, 0.5)

    # 3. Differencing Engine
    with st.expander("💡 3. Illumination Differencing", expanded=True):
        diff_method_name = st.selectbox(
            "Differencing Engine",
            [
                "SSIM Structural (Contrast & Lighting Invariant)",
                "Sobel Gradient (Illumination-Invariant)",
                "Hybrid (Gradient + SSIM Fusion)",
                "Raw Absolute Difference",
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
        use_clahe = st.checkbox("CLAHE Contrast Equalization", value=True, help="Equalizes local PCB illumination variations.")

        st.caption("Illumination Simulator:")
        sim_brightness = st.slider("Brightness Offset", -80, 80, 0, 5)
        sim_contrast = st.slider("Contrast Factor", 0.5, 2.0, 1.0, 0.05)

    # 4. Stage-2 CNN Classifier
    with st.expander("🧠 4. Stage-2 CNN Classifier", expanded=True):
        enable_cnn = st.toggle("Enable CNN Classification", value=True, help="Classifies candidate boxes into 6 DeepPCB defect types.")
        conf_filter = st.slider("Confidence Cutoff", 0.0, 1.0, 0.25, 0.05) if enable_cnn else 0.0

    # 5. Detection Thresholds
    with st.expander("📏 5. Tuning Thresholds", expanded=False):
        min_area = st.slider("Min Contour Area (px)", 0, 300, 12, 1, help="Small speckle noise cutoff.")
        iou_threshold = st.slider("IoU Match Threshold", 0.10, 0.90, 0.33, 0.01, help="Minimum overlap to count a True Positive.")


# ===================== LOAD & PROCESS IMAGES =====================
template_path, test_path, annotation_path = samples[sample_id]
template_bgr = cv2.imread(str(template_path))
test_bgr = cv2.imread(str(test_path))

if template_bgr is None or test_bgr is None:
    st.error(f"Could not read the image pair for sample {sample_id}.")
    st.stop()

# Apply Misalignment Simulator if non-zero
test_input_bgr = test_bgr.copy()
if sim_shift_x != 0 or sim_shift_y != 0 or sim_rot != 0.0:
    test_input_bgr, _ = simulate_misalignment(
        test_input_bgr, shift_x=sim_shift_x, shift_y=sim_shift_y, angle_deg=sim_rot
    )

# Apply Illumination Simulator if non-zero
if sim_brightness != 0 or sim_contrast != 1.0:
    test_input_bgr = simulate_illumination_change(
        test_input_bgr, brightness_offset=sim_brightness, contrast_factor=sim_contrast
    )

ground_truth = load_annotations(annotation_path)
classifier_model = get_cached_classifier() if enable_cnn else None

# Run Complete Pipeline
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

# Confidence Filtering
if enable_cnn and classified_results and conf_filter > 0.0:
    filtered_results = [res for res in classified_results if res["confidence"] >= conf_filter]
else:
    filtered_results = classified_results

metrics = calculate_metrics(
    detections=detections,
    ground_truth=ground_truth,
    iou_threshold=iou_threshold,
    classified_results=filtered_results if enable_cnn else None,
)

is_passed = (len(filtered_results if enable_cnn else detections) == 0)

# Severity Classification helper
def get_defect_severity(class_name):
    cname = class_name.lower()
    if cname in ["short", "open"]:
        return "Critical (Electrical Failure)", "sev-critical"
    elif cname in ["mousebite", "spur"]:
        return "Moderate (Signal Integrity)", "sev-moderate"
    else:
        return "Minor (Surface Contaminant)", "sev-minor"


# ===================== TOP INDUSTRIAL HEADER =====================
pass_fail_pill = f"""<span class="status-pill {'pill-pass' if is_passed else 'pill-fail'}">● {'PASSED (Zero Defects)' if is_passed else f'FLAGGED ({len(filtered_results if enable_cnn else detections)} Defects Found)'}</span>"""
reg_pill = f"""<span class="status-pill pill-info">⚡ Alignment: {reg_method.upper() if enable_reg else 'OFF'} ({reg_info.get('inliers', 0)} inliers)</span>"""
diff_pill = f"""<span class="status-pill pill-purple">💡 Differencing: {diff_method.upper()}</span>"""
cnn_pill = f"""<span class="status-pill pill-info">🧠 CNN Classifier: {'ON' if enable_cnn else 'OFF'}</span>"""

st.markdown(
    f"""
    <div class="aoi-header">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
            <div>
                <h1 class="aoi-title">⚡ PCB Automated Optical Inspection (AOI) Pro</h1>
                <div class="aoi-subtitle">Production Quality Control & Automated Defect Classification System</div>
                <div class="badge-container">
                    {pass_fail_pill}
                    {reg_pill}
                    {diff_pill}
                    {cnn_pill}
                </div>
            </div>
            <div style="text-align:right; color:#94a3b8; font-size:0.85rem;">
                <div><b>Board ID:</b> <code>{sample_id}</code></div>
                <div><b>Timestamp:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                <div><b>Inspection Mode:</b> Inline High-Precision</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===================== KPI METRIC CARDS =====================
st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Candidates</div>
            <div class="kpi-value">{len(filtered_results if enable_cnn else detections)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Verified (TP)</div>
            <div class="kpi-value" style="color:#10b981;">{metrics['true_positives']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">False Alarms (FP)</div>
            <div class="kpi-value" style="color:#ef4444;">{metrics['false_positives']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Precision</div>
            <div class="kpi-value">{metrics['precision']:.1%}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Recall</div>
            <div class="kpi-value">{metrics['recall']:.1%}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{'Class Accuracy' if metrics['class_accuracy'] is not None else 'Mean IoU'}</div>
            <div class="kpi-value" style="color:#6366f1;">
                {f"{metrics['class_accuracy']:.1%}" if metrics['class_accuracy'] is not None else f"{metrics['mean_iou']:.2f}"}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===================== NAVIGATION TABS =====================
tab_overview, tab_inspector, tab_pipeline, tab_lab, tab_report = st.tabs([
    "🔍 Inspection Console",
    "🔬 Defect Deep-Dive",
    "⚙️ 5-Stage Vision Pipeline",
    "🧪 Innovation & Simulators",
    "📋 Quality Audit Report",
])


# ----------------- TAB 1: INSPECTION CONSOLE -----------------
with tab_overview:
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        st.markdown("#### Live Optical Inspection View")
    with col_ctrl2:
        show_gt = st.checkbox("Show Ground Truth (Green)", value=True)

    result_rgb = draw_results(
        aligned_test,
        detections=detections,
        ground_truth=ground_truth if show_gt else [],
        classified_results=filtered_results if enable_cnn else None,
    )

    col_view1, col_view2 = st.columns(2)
    with col_view1:
        st.markdown("**Golden Reference Template (Clean Board)**")
        st.image(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_view2:
        st.markdown("**Inspected Board with Annotated Defect Boxes**")
        st.image(result_rgb, use_container_width=True)

    st.markdown("##### Defect Class Palette & Classification Legend")
    legend_html = " ".join([
        f"<span style='background-color:rgb({DEFECT_COLORS[cid][2]},{DEFECT_COLORS[cid][1]},{DEFECT_COLORS[cid][0]}); "
        f"color:white; padding:4px 10px; border-radius:6px; margin-right:8px; font-weight:700; font-size:12px; display:inline-block; margin-bottom:6px;'>"
        f"{cname.upper()}</span>"
        for cid, cname in DEFECT_CLASSES.items() if cid != 0
    ])
    st.markdown(legend_html, unsafe_allow_html=True)

    # Class distribution chart if defects found
    if filtered_results:
        st.markdown("##### Detected Defect Breakdown")
        class_series = pd.Series([item["class_name"].capitalize() for item in filtered_results]).value_counts()
        chart_col, _ = st.columns([2, 1])
        with chart_col:
            st.bar_chart(class_series, color="#6366f1", height=220)


# ----------------- TAB 2: DEFECT DEEP-DIVE INSPECTOR -----------------
with tab_inspector:
    st.markdown("#### Microscopic Defect Analysis (AOI Zoom)")

    if not filtered_results:
        st.success("No defects detected under current thresholds! Board is completely clean.")
    else:
        defect_options = [
            f"Defect D{idx}: {item['class_name'].upper()} ({item['confidence']:.1%})"
            for idx, item in enumerate(filtered_results, start=1)
        ]
        selected_defect_label = st.selectbox("Select Defect to Inspect:", defect_options, index=0)
        selected_idx = int(selected_defect_label.split(":")[0].replace("Defect D", "")) - 1
        item = filtered_results[selected_idx]

        bx1, by1, bx2, by2 = item["box"]
        cname = item["class_name"]
        conf = item["confidence"]
        sev_label, sev_class = get_defect_severity(cname)

        # Crop patch with context margin
        margin = 25
        h_img, w_img = template_bgr.shape[:2]
        crop_x1 = max(0, bx1 - margin)
        crop_y1 = max(0, by1 - margin)
        crop_x2 = min(w_img, bx2 + margin)
        crop_y2 = min(h_img, by2 + margin)

        crop_temp = template_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_test = aligned_test[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_diff = difference[crop_y1:crop_y2, crop_x1:crop_x2]

        crop_col1, crop_col2, crop_col3, crop_col4 = st.columns([1, 1, 1, 1.4])
        with crop_col1:
            st.markdown("**1. Reference Crop**")
            st.image(cv2.cvtColor(crop_temp, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Clean design trace")

        with crop_col2:
            st.markdown("**2. Manufactured Crop**")
            st.image(cv2.cvtColor(crop_test, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Physical manufactured trace")

        with crop_col3:
            st.markdown("**3. Difference Anomaly**")
            diff_colored = cv2.applyColorMap(crop_diff, cv2.COLORMAP_INFERNO)
            st.image(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Isolated flaw gradient")

        with crop_col4:
            st.markdown("**4. AI Diagnostic Verdict**")
            st.markdown(
                f"""
                <div class="diagnosis-card">
                    <div style="font-size:1.3rem; font-weight:800; color:#0f172a; margin-bottom:6px;">
                        {cname.upper()}
                    </div>
                    <div style="margin-bottom:12px;">
                        <span class="severity-badge {sev_class}">{sev_label}</span>
                    </div>
                    <div style="font-size:0.9rem; color:#475569; line-height:1.6;">
                        <div><b>Confidence:</b> {conf:.1%}</div>
                        <div><b>Coordinates:</b> ({bx1}, {by1}) to ({bx2}, {by2})</div>
                        <div><b>Dimensions:</b> {bx2-bx1} px × {by2-by1} px ({ (bx2-bx1)*(by2-by1) } px²)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("##### Complete Defect Manifest Catalog")
        table_rows = []
        for idx, it in enumerate(filtered_results, start=1):
            x1, y1, x2, y2 = it["box"]
            sev, _ = get_defect_severity(it["class_name"])
            table_rows.append({
                "ID": f"D{idx}",
                "Classification": it["class_name"].upper(),
                "Confidence": f"{it['confidence']:.1%}",
                "Severity": sev,
                "Bounding Box": f"({x1}, {y1}) -> ({x2}, {y2})",
                "Width (px)": x2 - x1,
                "Height (px)": y2 - y1,
                "Area (px²)": (x2 - x1) * (y2 - y1),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ----------------- TAB 3: 5-STAGE VISION PIPELINE -----------------
with tab_pipeline:
    st.markdown("#### End-to-End Computer Vision Pipeline")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("**Stage 01: Raw Test Board**")
        st.image(cv2.cvtColor(test_input_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Camera capture (includes any vibration or lighting shift).")

    with p2:
        st.markdown(f"**Stage 02: Homography Aligned ({reg_method.upper()})**")
        st.image(cv2.cvtColor(aligned_test, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Warped into pixel-perfect alignment with reference template.")

    with p3:
        st.markdown(f"**Stage 03: {diff_method.upper()} Differencing**")
        diff_colored = cv2.applyColorMap(difference, cv2.COLORMAP_INFERNO)
        st.image(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Illumination-invariant difference map.")

    p4, p5, p6 = st.columns(3)
    with p4:
        st.markdown("**Stage 04: Otsu Binarization**")
        st.image(thresholded, use_container_width=True)
        st.caption("Automatic histogram threshold separates defect pixels.")

    with p5:
        st.markdown("**Stage 05: Morphological Clean**")
        st.image(cleaned, use_container_width=True)
        st.caption("Opening erases speckles; closing bridges fragmented regions.")

    with p6:
        st.markdown("**Stage 06: CNN Defect Classification**")
        st.image(result_rgb, use_container_width=True)
        st.caption("Candidate boxes classified into 6 DeepPCB defect classes.")


# ----------------- TAB 4: INNOVATION & SIMULATORS -----------------
with tab_lab:
    st.markdown("#### Diagnostic Innovation Laboratory")

    st.markdown("##### 1. Auto-Registration: Feature Matches & Alignment")
    if reg_info.get("matches") and len(reg_info["matches"]) > 0:
        match_vis = draw_registration_matches(
            test_input_bgr,
            reg_info["kp_test"],
            template_bgr,
            reg_info["kp_temp"],
            reg_info["matches"],
            max_draw=40,
        )
        st.image(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Top 40 inlier feature matches between Test Board (left) and Clean Template (right). Inliers: {reg_info['inliers']}.")
    else:
        st.info("Enable Auto-Registration in the sidebar to inspect inlier feature matching lines.")

    st.divider()
    st.markdown("##### 2. Illumination Invariance: Raw Subtraction vs Invariant Differencing")
    raw_diff = compute_difference_map(template_smooth, test_smooth, method="raw")
    inv_diff = compute_difference_map(template_smooth, test_smooth, method=diff_method if diff_method != "raw" else "ssim")

    col_raw, col_inv = st.columns(2)
    with col_raw:
        st.markdown("**Classical Raw Pixel Differencing**")
        st.image(cv2.cvtColor(cv2.applyColorMap(raw_diff, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Mean residual noise: {raw_diff.mean():.2f}. (Vulnerable to brightness and exposure shifts).")

    with col_inv:
        st.markdown(f"**{diff_method.upper() if diff_method != 'raw' else 'SSIM'} Invariant Differencing**")
        st.image(cv2.cvtColor(cv2.applyColorMap(inv_diff, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"Mean residual noise: {inv_diff.mean():.2f}. (Mathematically isolates genuine circuit geometry changes).")


# ----------------- TAB 5: QUALITY AUDIT REPORT -----------------
with tab_report:
    st.markdown("#### Quality Assurance Inspection Certificate")

    audit_status = "REJECT / REWORK" if not is_passed else "ACCEPT / PASS"
    audit_color = "#ef4444" if not is_passed else "#10b981"

    st.markdown(
        f"""
        <div style="background:#ffffff; border:2px solid #e2e8f0; border-radius:12px; padding:24px; color:#1e293b;">
            <div style="display:flex; justify-content:space-between; border-bottom:2px solid #f1f5f9; padding-bottom:16px;">
                <div>
                    <h2 style="margin:0; font-size:1.5rem; color:#0f172a;">PCB AUTOMATED INSPECTION CERTIFICATE</h2>
                    <div style="color:#64748b; font-size:0.85rem;">Compliance Standard: IPC-A-610 Class 3 / ISO 9001:2015</div>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:1.4rem; font-weight:800; color:{audit_color}; border:2px solid {audit_color}; padding:6px 16px; border-radius:8px;">
                        {audit_status}
                    </span>
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin:20px 0; font-size:0.9rem;">
                <div><b>Board Serial:</b> <code>{sample_id}</code></div>
                <div><b>Date & Time:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                <div><b>Inspector:</b> Auto-AOI Engine v2.0</div>
                <div><b>Alignment Algorithm:</b> {reg_method.upper()}</div>
                <div><b>Differencing Mode:</b> {diff_method.upper()}</div>
                <div><b>CNN Classifier:</b> {'Active (7-Class PyTorch)' if enable_cnn else 'Disabled'}</div>
            </div>
            <div style="margin-top:20px;">
                <b>Inspection Summary:</b>
                <ul>
                    <li>Total Candidate Regions Detected: <b>{len(filtered_results if enable_cnn else detections)}</b></li>
                    <li>Verified Ground Truth Matches (TP): <b>{metrics['true_positives']}</b></li>
                    <li>False Alarms (FP): <b>{metrics['false_positives']}</b></li>
                    <li>System Precision: <b>{metrics['precision']:.1%}</b> | System Recall: <b>{metrics['recall']:.1%}</b></li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )