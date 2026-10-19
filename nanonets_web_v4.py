"""
PCB Component Inspection V4 — classical image processing only.
Dataset: NanoNetsPCBChecker/Dataset
No neural network training/inference. XML annotations are used to define
reference component ROIs and for evaluation only, never to make predictions.

Run:
    pip install streamlit opencv-python-headless numpy pandas scikit-image
    streamlit run nanonets_web_v4.py
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.metrics import structural_similarity

st.set_page_config(page_title="PCB Inspection V4", page_icon="🔧", layout="wide")

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "NanoNetsPCBChecker" / "Dataset"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------- Dataset ---------------------------------
@st.cache_data(show_spinner=False)
def discover_images(folder_text):
    root = Path(folder_text).expanduser()
    if not root.exists():
        return []
    found = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
            continue
        if "errors" in {x.lower() for x in p.parts} or p.stem.endswith("_annotated"):
            continue
        if p.with_suffix(".xml").exists():
            found.append(str(p))
    return sorted(found, key=lambda s: (Path(s).stem.split(".")[0], Path(s).name.lower()))


@st.cache_data(show_spinner=False)
def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


@st.cache_data(show_spinner=False)
def read_annotations(image_path):
    xml_path = Path(image_path).with_suffix(".xml")
    root = ET.parse(str(xml_path)).getroot()
    items = {}
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        try:
            x1, y1, x2, y2 = [
                int(round(float(box.findtext(k, "0"))))
                for k in ("xmin", "ymin", "xmax", "ymax")
            ]
            rotated = int(float(obj.findtext("rotated", "0") or "0"))
        except (ValueError, TypeError):
            continue
        if x2 > x1 and y2 > y1:
            items[name] = {"box": (x1, y1, x2, y2), "rotated": rotated}
    return items


def board_id(path):
    # 10.jpg and 10.10.jpg belong to board 10.
    return Path(path).stem.split(".")[0]


def crop(image, box, pad=0):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    result = image[y1:y2, x1:x2]
    return result if result.size else None


# ----------------------- Registration / features -----------------------
def gray(image):
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(image, (3, 3), 0)


def register(reference, test):
    """ECC global affine registration; safely returns the original on failure."""
    h, w = reference.shape[:2]
    if test.shape[:2] != (h, w):
        test = cv2.resize(test, (w, h), interpolation=cv2.INTER_AREA)

    scale = min(1.0, 1200.0 / max(h, w))
    sw, sh = max(64, int(w * scale)), max(64, int(h * scale))
    rg = cv2.resize(gray(reference), (sw, sh)).astype(np.float32) / 255.0
    tg = cv2.resize(gray(test), (sw, sh)).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    try:
        corr, warp = cv2.findTransformECC(
            rg, tg, warp, cv2.MOTION_AFFINE, criteria, None, 5
        )
        # Convert translation from downscaled to full-resolution coordinates.
        warp[0, 2] /= scale
        warp[1, 2] /= scale
        aligned = cv2.warpAffine(
            test, warp, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, float(corr), True
    except cv2.error:
        return test, 0.0, False


def local_roi(reference, test, box, search=6, padding=3):
    """
    Compare the reference component crop with a nearby test crop.
    A local search is accepted only when correlation is meaningful.
    The returned correlation is a registration diagnostic, not a defect label.
    """
    x1, y1, x2, y2 = map(int, box)
    template = crop(reference, box, padding)
    if template is None:
        return None, None, 0.0

    h, w = reference.shape[:2]
    tw, th = template.shape[1], template.shape[0]
    # Search around the padded box while keeping candidate crops same-sized.
    sx, sy = max(0, x1 - padding - search), max(0, y1 - padding - search)
    ex, ey = min(w, x2 + padding + search), min(h, y2 + padding + search)
    area = test[sy:ey, sx:ex]
    if area.shape[0] < th or area.shape[1] < tw:
        return template, crop(test, box, padding), 0.0

    tg = gray(template)
    ag = gray(area)
    if tg is None or ag is None or float(np.std(tg)) < 2.0:
        return template, crop(test, box, padding), 0.0

    try:
        corr_map = cv2.matchTemplate(ag, tg, cv2.TM_CCOEFF_NORMED)
        _, best, _, loc = cv2.minMaxLoc(corr_map)
        lx, ly = loc
        matched = area[ly:ly + th, lx:lx + tw]
        if matched.shape[:2] != template.shape[:2]:
            matched = crop(test, box, padding)
        return template, matched, float(best)
    except cv2.error:
        return template, crop(test, box, padding), 0.0


def feature_vector(reference_roi, test_roi):
    if reference_roi is None or test_roi is None:
        return None
    if reference_roi.size == 0 or test_roi.size == 0:
        return None

    rh, rw = reference_roi.shape[:2]
    test_roi = cv2.resize(test_roi, (rw, rh), interpolation=cv2.INTER_AREA)
    r = gray(reference_roi)
    t = gray(test_roi)

    # CLAHE mitigates mild lighting changes but is not used as the only signal.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    rn, tn = clahe.apply(r), clahe.apply(t)

    win = min(7, min(rn.shape))
    if win % 2 == 0:
        win -= 1
    similarity = (
        float(structural_similarity(rn, tn, data_range=255, win_size=win))
        if win >= 3 else 1.0
    )
    diff = cv2.absdiff(rn, tn)
    # Ignore tiny intensity fluctuations; ratio remains normalized [0,1].
    changed = float(np.mean(diff > 32))

    re = cv2.Canny(rn, 50, 140)
    te = cv2.Canny(tn, 50, 140)
    r_count = int(np.count_nonzero(re))
    t_count = int(np.count_nonzero(te))
    edge_loss = float(np.count_nonzero((re > 0) & (te == 0)) / max(r_count, 1))
    edge_gain = float(np.count_nonzero((re == 0) & (te > 0)) / max(r_count, 1))
    r_texture, t_texture = float(np.std(rn)), float(np.std(tn))
    texture_loss = float(max(0.0, (r_texture - t_texture) / max(r_texture, 1.0)))
    edge_density_ratio = float(t_count / max(r_count, 1))

    # Multi-signal anomaly score. SSIM is mapped from [-1,1] to [0,1].
    # Missing-component evidence emphasizes lost structure, not just any change.
    structural_change = np.clip((1.0 - similarity) / 2.0, 0.0, 1.0)
    score = (
        0.25 * structural_change
        + 0.20 * changed
        + 0.25 * np.clip(edge_loss, 0.0, 1.0)
        + 0.20 * np.clip(texture_loss, 0.0, 1.0)
        + 0.10 * np.clip(max(0.0, 1.0 - edge_density_ratio), 0.0, 1.0)
    )
    return {
        "ssim": similarity, "changed": changed, "edge_loss": edge_loss,
        "edge_gain": edge_gain, "texture_loss": texture_loss,
        "edge_density_ratio": edge_density_ratio, "score": float(score),
        "diff": diff,
    }


def inspect(reference, aligned_test, reference_objects, search, padding,
            threshold, rotation_enabled, rotation_gain):
    rows, overlay, regions = [], aligned_test.copy(), {}
    for name, meta in reference_objects.items():
        box = meta["box"]
        r, t, local_corr = local_roi(reference, aligned_test, box, search, padding)
        f = feature_vector(r, t)
        if f is None:
            continue

        angle, gain = (rotation_comparison(r, t) if rotation_enabled else (None, None))
        is_rotation = bool(
            rotation_enabled and angle in (90, 180, 270)
            and gain is not None and gain >= rotation_gain
            and f["score"] >= threshold
        )
        # Require corroborating structural-loss evidence to avoid flagging every
        # harmless colour/brightness change as a missing part.
        structural_loss = (
            f["ssim"] < 0.78
            and (f["edge_loss"] >= 0.22 or f["texture_loss"] >= 0.18)
        )
        is_change = bool(f["score"] >= threshold and structural_loss and not is_rotation)
        status = "ROTATION CANDIDATE (EXPERIMENTAL)" if is_rotation else (
            "CHANGE / POSSIBLE MISSING" if is_change else "NO STRONG CHANGE"
        )
        color = (0, 165, 255) if is_rotation else ((0, 0, 255) if is_change else (0, 190, 0))
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, name, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        rows.append({
            "Component": name, "Status": status,
            "Anomaly score": round(f["score"], 4),
            "SSIM": round(f["ssim"], 4),
            "Changed pixels": round(f["changed"], 4),
            "Edge loss": round(f["edge_loss"], 4),
            "Texture loss": round(f["texture_loss"], 4),
            "Edge density ratio": round(f["edge_density_ratio"], 4),
            "Local match": round(local_corr, 4),
            "Rotation angle candidate": angle,
            "Rotation advantage": round(gain, 4) if gain is not None else None,
        })
        regions[name] = (r, t, f["diff"])
    return pd.DataFrame(rows), overlay, regions


def rotation_comparison(r, t):
    """Experimental quarter-turn appearance comparison; not reliable for symmetric parts."""
    rg, tg = gray(r), gray(t)
    if min(rg.shape) < 12 or min(tg.shape) < 12 or np.std(rg) < 8:
        return None, None
    target = cv2.resize(tg, (64, 64))
    variants = {
        0: rg,
        90: cv2.rotate(rg, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(rg, cv2.ROTATE_180),
        270: cv2.rotate(rg, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    scores = {}
    for angle, variant in variants.items():
        variant = cv2.resize(variant, (64, 64))
        scores[angle] = float(structural_similarity(variant, target, data_range=255))
    best_angle = max(scores, key=scores.get)
    return best_angle, scores[best_angle] - scores[0]


def metric_values(actual, predicted, universe):
    actual, predicted, universe = set(actual), set(predicted), set(universe)
    tp, fp, fn = len(actual & predicted), len(predicted - actual), len(actual - predicted)
    tn = len(universe - (actual | predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1}


def fmt(value):
    return "N/A" if value is None else f"{value:.1%}"


# -------------------------------- UI ------------------------------------
st.title("PCB Component Inspection — Classical Image Processing V4")
st.caption(
    "Global ECC registration + local ROI matching + SSIM, changed-pixel, edge-loss "
    "and texture-loss features. No deep learning or model training."
)
with st.sidebar:
    st.header("Dataset")
    dataset_text = st.text_input("Annotated dataset directory", str(DEFAULT_DATASET))

image_paths = discover_images(dataset_text)
if len(image_paths) < 2:
    st.error("No usable image/XML pairs found. Select the Dataset folder inside NanoNetsPCBChecker.")
    st.stop()

# Prefer full-board, non-rotated images as likely references, but ask the user to verify.
clean_candidates = []
for path in image_paths:
    try:
        ann = read_annotations(path)
        if len(ann) == 60 and not any(v["rotated"] for v in ann.values()):
            clean_candidates.append(path)
    except (ET.ParseError, OSError, ValueError):
        pass

st.info(f"Found {len(image_paths)} annotated image/XML pairs. "
        f"{len(clean_candidates)} look like clean-reference candidates by XML; verify visually.")

with st.sidebar:
    reference_options = clean_candidates or image_paths
    reference_path = st.selectbox(
        "Reference PCB (known-good)", reference_options,
        format_func=lambda p: Path(p).name
    )
    same_board = [p for p in image_paths if p != reference_path and board_id(p) == board_id(reference_path)]
    test_options = same_board or [p for p in image_paths if p != reference_path]
    test_path = st.selectbox("Test PCB", test_options, format_func=lambda p: Path(p).name)
    st.caption("Preferred pairing: 10.jpg → 10.10.jpg, 1.jpg → 1.1.jpg, etc.")
    threshold = st.slider("Anomaly score threshold", 0.10, 0.75, 0.42, 0.01)
    search = st.slider("Local alignment search (pixels)", 0, 20, 6, 1)
    padding = st.slider("ROI padding (pixels)", 0, 15, 3, 1)
    rotation_enabled = st.checkbox("Experimental 90°/180°/270° rotation comparison", False)
    rotation_gain = st.slider("Rotation similarity advantage", 0.05, 0.60, 0.18, 0.01,
                              disabled=not rotation_enabled)
    run = st.button("Run Inspection", type="primary", use_container_width=True)

if board_id(reference_path) != board_id(test_path):
    st.warning("These images have different board IDs. Component coordinates may not correspond; evaluation may be invalid.")
if not run:
    st.stop()

reference = load_image(reference_path)
test_original = load_image(test_path)
if reference is None or test_original is None:
    st.error("Could not load one of the selected images.")
    st.stop()

try:
    reference_objects = read_annotations(reference_path)
    test_objects = read_annotations(test_path)
except (ET.ParseError, OSError, ValueError) as exc:
    st.error(f"Could not parse XML annotations: {exc}")
    st.stop()

with st.spinner("Registering board and comparing component regions..."):
    aligned, ecc_score, ecc_ok = register(reference, test_original)
    frame, overlay, regions = inspect(
        reference, aligned, reference_objects, search, padding,
        threshold, rotation_enabled, rotation_gain
    )

if frame.empty:
    st.error("No component regions could be processed.")
    st.stop()

st.subheader("1. Input images and registration")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB), caption="Reference PCB", use_container_width=True)
with c2:
    st.image(cv2.cvtColor(test_original, cv2.COLOR_BGR2RGB), caption="Original test PCB", use_container_width=True)
with c3:
    st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB),
             caption=f"Aligned test (ECC {ecc_score:.4f})" if ecc_ok else "Alignment fallback (ECC failed)",
             use_container_width=True)
if not ecc_ok:
    st.warning("ECC registration failed; predictions may be unreliable. Check that the images show the same board.")
elif ecc_score < 0.75:
    st.warning("Registration correlation is low. Review the aligned image before trusting the predictions.")

st.subheader("2. Component-level predictions")
st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
         caption="Green: no strong change; red: change/possible missing; orange: experimental rotation",
         use_container_width=True)
st.dataframe(frame, use_container_width=True, hide_index=True)
st.download_button(
    "Download predictions CSV", frame.to_csv(index=False).encode("utf-8"),
    file_name=f"pcb_inspection_{Path(test_path).stem}.csv", mime="text/csv"
)

st.subheader("3. Inspect one component")
selected = st.selectbox("Component", frame["Component"].tolist())
roi_r, roi_t, roi_diff = regions[selected]
a, b, c = st.columns(3)
with a:
    st.image(cv2.cvtColor(roi_r, cv2.COLOR_BGR2RGB), caption="Reference ROI", use_container_width=True)
with b:
    st.image(cv2.cvtColor(roi_t, cv2.COLOR_BGR2RGB), caption="Matched test ROI", use_container_width=True)
with c:
    st.image(roi_diff, caption="Contrast-normalized absolute difference", use_container_width=True, clamp=True)

st.subheader("4. XML ground-truth evaluation")
st.warning(
    "XML annotations are used here only after predictions are produced. "
    "A red region is a candidate, not proof of a missing component. "
    "Do not report performance from a single image pair."
)
actual_missing = set(reference_objects) - set(test_objects)
actual_rotated = {n for n, v in test_objects.items() if v["rotated"] == 1}
pred_missing = set(frame.loc[frame["Status"] == "CHANGE / POSSIBLE MISSING", "Component"])
pred_rotated = set(frame.loc[frame["Status"] == "ROTATION CANDIDATE (EXPERIMENTAL)", "Component"])

for title, actual, predicted in [
    ("Missing-component candidates", actual_missing, pred_missing),
    ("Rotation candidates", actual_rotated, pred_rotated),
]:
    m = metric_values(actual, predicted, reference_objects)
    st.markdown(f"#### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual", len(actual))
    c2.metric("Predicted", len(predicted))
    c3.metric("Precision", fmt(m["Precision"]))
    c4.metric("Recall", fmt(m["Recall"]))
    st.caption(f"F1={fmt(m['F1'])} | TP={m['TP']} | FP={m['FP']} | FN={m['FN']} | TN={m['TN']}")
    left, right = st.columns(2)
    left.write("XML ground truth: " + (", ".join(sorted(actual)) if actual else "None"))
    right.write("Predicted candidates: " + (", ".join(sorted(predicted)) if predicted else "None"))

st.subheader("5. Interpretation")
st.write(
    "A low precision means too many intact components are being flagged. Increase the anomaly "
    "threshold gradually and inspect false positives. A low recall means real defects are being "
    "missed; lower the threshold cautiously. Tune on development pairs, then evaluate on different "
    "held-out pairs. Do not tune on the same pairs used for final reported results."
)
st.caption(
    "This is a classical computer-vision baseline, not a guaranteed perfect detector. Performance "
    "depends on image registration, lighting, annotation quality, and whether the reference is truly "
    "defect-free. Quarter-turn detection is experimental and disabled by default."
)
