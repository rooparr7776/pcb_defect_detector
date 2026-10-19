# FILE: pcb_defect_detector/nanonets_web.py

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import pandas as pd

from skimage.metrics import structural_similarity


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "NanoNetsPCBChecker"
    / "DLAssignment"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

st.set_page_config(
    page_title="NanoNets PCB Component Inspection",
    page_icon="🔧",
    layout="wide",
)

# ============================================================
# DATASET UTILITIES
# ============================================================


def find_dataset_images(dataset_root):
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        return []

    images = []

    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Ignore red-marked visualization images.
        if "errors" in [part.lower() for part in path.parts]:
            continue

        if path.stem.endswith("_annotated"):
            continue

        xml_path = path.with_suffix(".xml")

        if xml_path.exists():
            images.append(path)

    return sorted(images, key=lambda p: p.name.lower())


@st.cache_data(show_spinner=False)
def read_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def parse_annotation(xml_path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    components = []

    for obj in root.findall("object"):
        name = obj.findtext("name", default="unknown")
        rotated = int(obj.findtext("rotated", default="0"))

        box = obj.find("bndbox")

        if box is None:
            continue

        x1 = int(float(box.findtext("xmin", default="0")))
        y1 = int(float(box.findtext("ymin", default="0")))
        x2 = int(float(box.findtext("xmax", default="0")))
        y2 = int(float(box.findtext("ymax", default="0")))

        if x2 <= x1 or y2 <= y1:
            continue

        # Component number is the portion before "_".
        component_number = name.split("_")[0]

        components.append(
            {
                "name": name,
                "component_number": component_number,
                "bbox": (x1, y1, x2, y2),
                "rotated": rotated,
                "center": (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                ),
            }
        )

    return components


def load_component_mapping(dataset_root):
    mapping_path = Path(dataset_root) / "number_to_component.json"

    if not mapping_path.exists():
        return {}

    try:
        with open(mapping_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


# ============================================================
# IMAGE UTILITIES
# ============================================================


def rgb_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def crop_roi(image, bbox, padding=5):
    height, width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def resize_like(image, target_shape):
    target_height, target_width = target_shape[:2]

    if image.shape[:2] == (target_height, target_width):
        return image

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def normalize_roi(image):
    gray = rgb_to_gray(image)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return gray


def calculate_ssim(reference_roi, test_roi):
    reference_gray = normalize_roi(reference_roi)
    test_gray = normalize_roi(test_roi)

    test_gray = resize_like(
        test_gray,
        reference_gray.shape,
    )

    if min(reference_gray.shape) < 7:
        return 0.0

    score = structural_similarity(
        reference_gray,
        test_gray,
        data_range=255,
    )

    return float(score)


def calculate_difference_ratio(
    reference_roi,
    test_roi,
    threshold=25,
):
    reference_gray = normalize_roi(reference_roi)
    test_gray = normalize_roi(test_roi)

    test_gray = resize_like(
        test_gray,
        reference_gray.shape,
    )

    difference = cv2.absdiff(
        reference_gray,
        test_gray,
    )

    _, mask = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    changed_pixels = np.count_nonzero(mask)
    total_pixels = mask.size

    if total_pixels == 0:
        return 0.0, difference, mask

    return (
        changed_pixels / total_pixels,
        difference,
        mask,
    )


def calculate_edge_difference_ratio(
    reference_roi,
    test_roi,
):
    reference_gray = normalize_roi(reference_roi)
    test_gray = normalize_roi(test_roi)

    test_gray = resize_like(
        test_gray,
        reference_gray.shape,
    )

    reference_edges = cv2.Canny(
        reference_gray,
        50,
        150,
    )

    test_edges = cv2.Canny(
        test_gray,
        50,
        150,
    )

    edge_difference = cv2.absdiff(
        reference_edges,
        test_edges,
    )

    ratio = (
        np.count_nonzero(edge_difference)
        / edge_difference.size
    )

    return float(ratio), edge_difference


# ============================================================
# IMAGE REGISTRATION
# ============================================================


def align_test_image(reference_rgb, test_rgb):
    reference_gray = rgb_to_gray(reference_rgb)
    test_gray = rgb_to_gray(test_rgb)

    if reference_gray.shape != test_gray.shape:
        test_rgb = cv2.resize(
            test_rgb,
            (
                reference_gray.shape[1],
                reference_gray.shape[0],
            ),
        )

        test_gray = rgb_to_gray(test_rgb)

    reference_float = (
        reference_gray.astype(np.float32) / 255.0
    )

    test_float = (
        test_gray.astype(np.float32) / 255.0
    )

    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-6,
    )

    try:
        correlation, warp_matrix = cv2.findTransformECC(
            reference_float,
            test_float,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
        )

        height, width = reference_rgb.shape[:2]

        aligned = cv2.warpAffine(
            test_rgb,
            warp_matrix,
            (width, height),
            flags=(
                cv2.INTER_LINEAR
                | cv2.WARP_INVERSE_MAP
            ),
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned, True, float(correlation)

    except cv2.error:
        return test_rgb, False, 0.0


# ============================================================
# COMPONENT ROTATION ANALYSIS
# ============================================================


def estimate_orientation(roi):
    gray = normalize_roi(roi)

    if min(gray.shape) < 5:
        return None

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(contour) < 5:
        return None

    rectangle = cv2.minAreaRect(contour)

    width, height = rectangle[1]
    angle = rectangle[2]

    if width < height:
        angle += 90.0

    return float(angle % 180.0)


def angle_difference(angle1, angle2):
    difference = abs(angle1 - angle2) % 180.0

    return min(
        difference,
        180.0 - difference,
    )


# ============================================================
# CLASSICAL COMPONENT INSPECTION
# ============================================================


def inspect_components(
    reference_image,
    test_image,
    reference_components,
    ssim_threshold,
    difference_threshold,
    edge_threshold,
    rotation_threshold,
    padding,
):
    results = []

    overlay = test_image.copy()

    for component in reference_components:
        name = component["name"]
        bbox = component["bbox"]

        reference_roi = crop_roi(
            reference_image,
            bbox,
            padding,
        )

        test_roi = crop_roi(
            test_image,
            bbox,
            padding,
        )

        if (
            reference_roi is None
            or test_roi is None
            or reference_roi.size == 0
            or test_roi.size == 0
        ):
            continue

        test_roi = resize_like(
            test_roi,
            reference_roi.shape,
        )

        ssim_score = calculate_ssim(
            reference_roi,
            test_roi,
        )

        difference_ratio, difference, mask = (
            calculate_difference_ratio(
                reference_roi,
                test_roi,
            )
        )

        edge_ratio, edge_difference = (
            calculate_edge_difference_ratio(
                reference_roi,
                test_roi,
            )
        )

        reference_angle = estimate_orientation(
            reference_roi,
        )

        test_angle = estimate_orientation(
            test_roi,
        )

        if (
            reference_angle is not None
            and test_angle is not None
        ):
            angle_delta = angle_difference(
                reference_angle,
                test_angle,
            )
        else:
            angle_delta = None

        suspicious_appearance = (
            ssim_score < ssim_threshold
            and difference_ratio > difference_threshold
            and edge_ratio > edge_threshold
        )

        suspected_rotation = (
            angle_delta is not None
            and angle_delta >= rotation_threshold
        )

        if suspected_rotation:
            status = "ROTATION CANDIDATE"

        elif suspicious_appearance:
            status = "SUSPECTED DEFECT"

        else:
            status = "NORMAL"

        x1, y1, x2, y2 = bbox

        if status == "NORMAL":
            color = (0, 255, 0)

        elif status == "ROTATION CANDIDATE":
            color = (255, 165, 0)

        else:
            color = (255, 0, 0)

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            overlay,
            name,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        results.append(
            {
                "Component": name,
                "Status": status,
                "SSIM": round(ssim_score, 4),
                "Difference Ratio": round(
                    difference_ratio,
                    4,
                ),
                "Edge Difference": round(
                    edge_ratio,
                    4,
                ),
                "Reference Angle": (
                    round(reference_angle, 2)
                    if reference_angle is not None
                    else None
                ),
                "Test Angle": (
                    round(test_angle, 2)
                    if test_angle is not None
                    else None
                ),
                "Angle Difference": (
                    round(angle_delta, 2)
                    if angle_delta is not None
                    else None
                ),
                "bbox": bbox,
                "difference_image": difference,
                "threshold_mask": mask,
                "edge_difference": edge_difference,
            }
        )

    return results, overlay


# ============================================================
# GROUND-TRUTH EVALUATION
# ============================================================


def evaluate_results(
    results,
    test_components,
    reference_components,
):
    reference_names = {
        component["name"]
        for component in reference_components
    }

    test_by_name = {
        component["name"]: component
        for component in test_components
    }

    actual_missing = (
        reference_names - set(test_by_name.keys())
    )

    actual_rotated = {
        component["name"]
        for component in test_components
        if component["rotated"] == 1
    }

    predictions = {
        result["Component"]: result["Status"]
        for result in results
    }

    predicted_missing = {
        name
        for name, status in predictions.items()
        if status == "SUSPECTED DEFECT"
    }

    predicted_rotated = {
        name
        for name, status in predictions.items()
        if status == "ROTATION CANDIDATE"
    }

    return {
        "actual_missing": actual_missing,
        "actual_rotated": actual_rotated,
        "predicted_missing": predicted_missing,
        "predicted_rotated": predicted_rotated,
    }


def binary_metrics(actual, predicted):
    tp = len(actual & predicted)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


# ============================================================
# STREAMLIT APPLICATION
# ============================================================


st.title("PCB Component Inspection — NanoNets")
st.caption(
    "Classical image processing: ROI comparison, "
    "SSIM, edge differences, and contour geometry. "
    "No ML/DL model is used."
)

with st.sidebar:
    st.header("Dataset")

    dataset_input = st.text_input(
        "NanoNets dataset folder",
        value=str(DEFAULT_DATASET),
    )

dataset_root = Path(dataset_input).expanduser()

images = find_dataset_images(dataset_root)

if not images:
    st.error(
        "No images with matching XML annotations were found. "
        "Check the dataset folder path and ensure the original "
        "images and XML files are present."
    )

    st.stop()

image_lookup = {
    str(path): path
    for path in images
}

with st.sidebar:
    st.header("Reference PCB")

    reference_path_string = st.selectbox(
        "Select a known defect-free reference image",
        options=list(image_lookup.keys()),
        format_func=lambda value: Path(value).name,
    )

    st.header("Test PCB")

    test_options = [
        str(path)
        for path in images
        if str(path) != reference_path_string
    ]

    if not test_options:
        st.error("At least two annotated images are required.")
        st.stop()

    test_path_string = st.selectbox(
        "Select a PCB to inspect",
        options=test_options,
        format_func=lambda value: Path(value).name,
    )

    st.header("Detection Parameters")

    ssim_threshold = st.slider(
        "SSIM threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
        help="Lower SSIM indicates greater structural difference.",
    )

    difference_threshold = st.slider(
        "Difference ratio threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01,
    )

    edge_threshold = st.slider(
        "Edge difference threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.08,
        step=0.01,
    )

    rotation_threshold = st.slider(
        "Rotation angle threshold (degrees)",
        min_value=5,
        max_value=90,
        value=25,
        step=1,
    )

    padding = st.slider(
        "ROI padding (pixels)",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
    )

    run_button = st.button(
        "Run Inspection",
        type="primary",
        use_container_width=True,
    )

reference_path = image_lookup[reference_path_string]
test_path = image_lookup[test_path_string]

reference_image = read_image(str(reference_path))
test_image = read_image(str(test_path))

if reference_image is None or test_image is None:
    st.error("An image could not be loaded.")
    st.stop()

try:
    reference_components = parse_annotation(
        reference_path.with_suffix(".xml")
    )

    test_components = parse_annotation(
        test_path.with_suffix(".xml")
    )

except (ET.ParseError, OSError, ValueError) as error:
    st.error(f"Unable to parse annotations: {error}")
    st.stop()

if not reference_components:
    st.error("The reference image has no valid component annotations.")
    st.stop()

if reference_image.shape[:2] != test_image.shape[:2]:
    test_image = cv2.resize(
        test_image,
        (
            reference_image.shape[1],
            reference_image.shape[0],
        ),
    )

if run_button:
    with st.spinner("Aligning images and inspecting components..."):
        aligned_image, alignment_success, correlation = (
            align_test_image(
                reference_image,
                test_image,
            )
        )

        results, overlay = inspect_components(
            reference_image=reference_image,
            test_image=aligned_image,
            reference_components=reference_components,
            ssim_threshold=ssim_threshold,
            difference_threshold=difference_threshold,
            edge_threshold=edge_threshold,
            rotation_threshold=rotation_threshold,
            padding=padding,
        )

        evaluation = evaluate_results(
            results,
            test_components,
            reference_components,
        )

    st.subheader("1. Input Images")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            reference_image,
            caption="Reference PCB",
            use_container_width=True,
        )

    with col2:
        st.image(
            test_image,
            caption="Original Test PCB",
            use_container_width=True,
        )

    st.subheader("2. Image Alignment")

    if alignment_success:
        st.success(
            f"ECC alignment completed. "
            f"Correlation: {correlation:.4f}"
        )
    else:
        st.warning(
            "ECC alignment failed. The original test image "
            "was used; localization results may be unreliable."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            aligned_image,
            caption="Aligned Test PCB",
            use_container_width=True,
        )

    with col2:
        st.image(
            cv2.absdiff(
                rgb_to_gray(reference_image),
                rgb_to_gray(aligned_image),
            ),
            caption="Whole-image absolute difference",
            use_container_width=True,
            clamp=True,
        )

    st.subheader("3. Component Inspection Results")

    st.image(
        overlay,
        caption=(
            "Green = normal; orange = rotation candidate; "
            "red = suspected defect"
        ),
        use_container_width=True,
    )

    if results:
        results_df = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key not in {
                        "bbox",
                        "difference_image",
                        "threshold_mask",
                        "edge_difference",
                    }
                }
                for result in results
            ]
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Inspection Results CSV",
            data=csv_data,
            file_name="nanonets_inspection_results.csv",
            mime="text/csv",
        )

        st.subheader("4. Inspect Individual Components")

        component_names = [
            result["Component"]
            for result in results
        ]

        selected_name = st.selectbox(
            "Component",
            component_names,
        )

        selected_result = next(
            result
            for result in results
            if result["Component"] == selected_name
        )

        reference_component = next(
            component
            for component in reference_components
            if component["name"] == selected_name
        )

        reference_roi = crop_roi(
            reference_image,
            reference_component["bbox"],
            padding,
        )

        test_roi = crop_roi(
            aligned_image,
            reference_component["bbox"],
            padding,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                reference_roi,
                caption="Reference ROI",
                use_container_width=True,
            )

        with col2:
            st.image(
                test_roi,
                caption="Test ROI",
                use_container_width=True,
            )

        with col3:
            st.image(
                selected_result["difference_image"],
                caption="Absolute difference",
                use_container_width=True,
                clamp=True,
            )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                selected_result["threshold_mask"],
                caption="Threshold mask",
                use_container_width=True,
                clamp=True,
            )

        with col2:
            st.image(
                selected_result["edge_difference"],
                caption="Edge difference",
                use_container_width=True,
                clamp=True,
            )

    else:
        st.warning("No component regions could be inspected.")

    st.subheader("5. Ground-Truth Evaluation")

    st.warning(
        "The following metrics compare image-based predictions "
        "with XML annotations. Ground-truth annotations are used "
        "for evaluation only, not to generate predictions. "
        "The reference image must be verified as defect-free."
    )

    missing_metrics = binary_metrics(
        evaluation["actual_missing"],
        evaluation["predicted_missing"],
    )

    rotation_metrics = binary_metrics(
        evaluation["actual_rotated"],
        evaluation["predicted_rotated"],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Missing Components")

        st.metric(
            "Ground-truth missing",
            len(evaluation["actual_missing"]),
        )

        st.metric(
            "Predicted missing candidates",
            len(evaluation["predicted_missing"]),
        )

        st.metric(
            "Precision",
            f"{missing_metrics['Precision']:.1%}",
        )

        st.metric(
            "Recall",
            f"{missing_metrics['Recall']:.1%}",
        )

        st.metric(
            "F1 score",
            f"{missing_metrics['F1']:.1%}",
        )

    with col2:
        st.markdown("#### Rotated Components")

        st.metric(
            "Ground-truth rotated",
            len(evaluation["actual_rotated"]),
        )

        st.metric(
            "Predicted rotation candidates",
            len(evaluation["predicted_rotated"]),
        )

        st.metric(
            "Precision",
            f"{rotation_metrics['Precision']:.1%}",
        )

        st.metric(
            "Recall",
            f"{rotation_metrics['Recall']:.1%}",
        )

        st.metric(
            "F1 score",
            f"{rotation_metrics['F1']:.1%}",
        )

    st.subheader("6. Ground-Truth Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Actual missing components")

        st.write(
            sorted(
                evaluation["actual_missing"],
                key=lambda value: int(value)
                if value.isdigit()
                else value,
            )
        )

        st.write("Predicted missing candidates")

        st.write(
            sorted(
                evaluation["predicted_missing"],
                key=lambda value: int(value)
                if value.isdigit()
                else value,
            )
        )

    with col2:
        st.write("Actual rotated components")

        st.write(
            sorted(evaluation["actual_rotated"])
        )

        st.write("Predicted rotation candidates")

        st.write(
            sorted(evaluation["predicted_rotated"])
        )

    st.caption(
        "Important: this is an initial classical-image-processing "
        "baseline. ROI comparison can confuse missing components, "
        "rotation, illumination changes, and alignment errors. "
        "Contour orientation is unreliable for symmetric components. "
        "Tune parameters on development images and evaluate on "
        "held-out images before reporting final performance."
    )
else:
    st.info(
        "Select a known defect-free reference image and a test image, "
        "adjust the parameters, and click Run Inspection."
    )