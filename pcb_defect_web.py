"""Browser UI for the classical DeepPCB defect detection demo."""

from pathlib import Path

import cv2
import streamlit as st

from pcb_defect_detection import (
    DEFAULT_DATA_ROOT,
    calculate_metrics,
    detect_defects,
    discover_samples,
    draw_results,
    load_annotations,
)


st.set_page_config(page_title="PCB Defect Detection", layout="wide")


@st.cache_data(show_spinner="Finding image pairs and annotations...")
def get_samples(data_root):
    return discover_samples(Path(data_root))


st.title("PCB Surface Defect Detection")
st.caption("Classical image processing on aligned DeepPCB reference and test images")

with st.sidebar:
    st.header("Analysis")
    data_root = st.text_input("PCBData folder", str(DEFAULT_DATA_ROOT))
    samples = get_samples(data_root)
    if not samples:
        st.error("No complete image pairs and annotations found in this folder.")
        st.stop()

    sample_id = st.selectbox("Dataset sample", list(samples))
    min_area = st.slider(
        "Minimum contour area",
        min_value=0,
        max_value=300,
        value=12,
        help="Discard smaller connected regions from the threshold mask.",
    )
    iou_threshold = st.slider(
        "IoU match threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.33,
        step=0.01,
        help="Minimum overlap required to count a detection as a true positive.",
    )

template_path, test_path, annotation_path = samples[sample_id]
template_bgr = cv2.imread(str(template_path))
test_bgr = cv2.imread(str(test_path))
if template_bgr is None or test_bgr is None:
    st.error(f"Could not read the image pair for sample {sample_id}.")
    st.stop()
if template_bgr.shape[:2] != test_bgr.shape[:2]:
    st.error("The selected reference and test image dimensions do not match.")
    st.stop()

ground_truth = load_annotations(annotation_path)
reference_gray, test_gray, difference, thresholded, cleaned, detections = detect_defects(
    template_bgr, test_bgr, min_area
)
tp, fp, fn, precision, recall, mean_iou = calculate_metrics(
    detections, ground_truth, iou_threshold
)

def show_image(column, title, explanation, image):
    with column:
        st.markdown(f"**{title}**")
        st.image(image, use_container_width=True)
        st.caption(explanation)


st.subheader(f"Sample {sample_id}")
st.caption(f"Reference: `{template_path.name}`  |  Test: `{test_path.name}`")
st.info(
    "The clean board is compared with the test board. Bright changed pixels are "
    "cleaned up, grouped into boxes, then checked against the green dataset labels."
)

st.markdown(
    """
    <style>
    .pipeline {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin: 8px 0 22px;
    }
    .pipeline-step {
        min-height: 88px;
        padding: 12px;
        border-left: 3px solid #db6d3b;
        background: #f2f5f1;
        color: #263a36;
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    .pipeline-step span { color: #5c6e68; font-size: 0.9rem; }
    .pipeline-step b { color: #db6d3b; font-size: 0.8rem; }
    @media (max-width: 800px) {
        .pipeline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    <div class="pipeline">
      <div class="pipeline-step"><b>01 / INPUT</b><strong>Aligned pair</strong><span>Clean board + test board</span></div>
      <div class="pipeline-step"><b>02 / PREPARE</b><strong>Grayscale + blur</strong><span>Reduce color and tiny noise</span></div>
      <div class="pipeline-step"><b>03 / COMPARE</b><strong>Pixel difference</strong><span>Show where the boards changed</span></div>
      <div class="pipeline-step"><b>04 / CLEAN</b><strong>Threshold + morphology</strong><span>Keep connected changed regions</span></div>
      <div class="pipeline-step"><b>05 / CHECK</b><strong>Boxes + score</strong><span>Compare boxes with labels</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 1. Start with the aligned image pair")
input_columns = st.columns(2)
show_image(
    input_columns[0], "Reference PCB", "The clean template the test image is compared with.",
    cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB),
)
show_image(
    input_columns[1], "Defective PCB", "The aligned test image that may contain defects.",
    cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB),
)

st.markdown("### 2. Simplify the images and compare pixels")
heatmap = cv2.cvtColor(
    cv2.applyColorMap(difference, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB
)
prep_columns = st.columns(3)
show_image(
    prep_columns[0], "Reference: grayscale + blur",
    "Removes color detail and smooths tiny pixel noise.",
    cv2.cvtColor(reference_gray, cv2.COLOR_GRAY2RGB),
)
show_image(
    prep_columns[1], "Test: grayscale + blur",
    "The same preparation is applied to the test board.",
    cv2.cvtColor(test_gray, cv2.COLOR_GRAY2RGB),
)
show_image(
    prep_columns[2], "Difference heatmap",
    "Bright yellow areas changed most; dark areas are more alike.", heatmap,
)

st.markdown("### 3. Turn changes into clean regions")
mask_columns = st.columns(2)
show_image(
    mask_columns[0], "Threshold mask",
    "Otsu thresholding makes changed pixels white and the rest black.",
    cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB),
)
show_image(
    mask_columns[1], "Cleaned mask",
    "Morphology removes specks and joins nearby white pixels into regions.",
    cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB),
)

st.markdown("### 4. Box candidate defects and compare with labels")
prediction_only = draw_results(test_bgr, detections, [])
result_rgb = draw_results(test_bgr, detections, ground_truth)
result_columns = st.columns(2)
show_image(
    result_columns[0], "Detected regions",
    f"Red boxes are changed regions that passed the {min_area}-pixel area cutoff.",
    prediction_only,
)
show_image(
    result_columns[1], "Detections vs. dataset labels",
    "Red = detected region. Green = annotated ground-truth defect.", result_rgb,
)

st.markdown("### How did it do?")
st.info(
    f"The image processing found **{len(detections)} candidate regions**. "
    f"**{tp}** matched an annotation, **{fp}** did not match one, and "
    f"**{fn} annotated defects were missed** at IoU {iou_threshold:.2f}."
)
metric_columns = st.columns(6)
metric_values = [
    ("Candidates", str(len(detections))),
    ("Correct matches", str(tp)),
    ("Extra boxes", str(fp)),
    ("Missed labels", str(fn)),
    ("Precision", f"{precision:.1%}"),
    ("Recall", f"{recall:.1%}"),
]
for column, (label, value) in zip(metric_columns, metric_values):
    column.metric(label, value)
st.caption(f"Mean IoU of matched boxes: {mean_iou:.3f}")
with st.expander("What do these scores mean?"):
    st.markdown(
        "- **Correct match:** a detected box overlaps a dataset box by at least the selected IoU threshold.\n"
        "- **Precision:** of all detected boxes, the fraction that matched a label.\n"
        "- **Recall:** of all labelled defects, the fraction the image processing found.\n"
        "- **Mean IoU:** average overlap for the correctly matched boxes."
    )