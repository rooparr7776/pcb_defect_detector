"""Classical PCB defect localization with aligned DeepPCB image pairs.

Requires: opencv-python, numpy, matplotlib
Run from the workspace root with: python pcb_defect_detection.py --list
"""

import argparse
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "DeepPCB-master" / "PCBData"
TEST_SUFFIX = "_test.jpg"


def discover_samples(data_root):
    """Find real test/template pairs and matching annotations in the dataset."""
    samples = {}
    for test_path in sorted(data_root.rglob(f"*{TEST_SUFFIX}")):
        sample_id = test_path.name[: -len(TEST_SUFFIX)]
        template_path = test_path.with_name(f"{sample_id}_temp.jpg")
        annotation_path = test_path.parent.parent / f"{test_path.parent.name}_not" / f"{sample_id}.txt"
        if template_path.is_file() and annotation_path.is_file():
            samples[sample_id] = (template_path, test_path, annotation_path)
    return samples


def load_annotations(annotation_path):
    """Read DeepPCB boxes; accept whitespace or comma separated rows."""
    boxes = []
    with annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line in annotation_file:
            values = re.split(r"[\s,]+", line.strip())
            if not values or not values[0]:
                continue
            if len(values) < 4:
                raise ValueError(f"Malformed annotation row in {annotation_path}: {line.strip()}")
            x1, y1, x2, y2 = map(int, values[:4])
            boxes.append((x1, y1, x2, y2))
    return boxes


def preprocess_images(template_bgr, test_bgr):
    """Convert an aligned pair to lightly blurred grayscale images."""
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    # A light blur suppresses isolated pixel noise without heavily softening PCB details.
    template_smooth = cv2.GaussianBlur(template_gray, (3, 3), 0)
    test_smooth = cv2.GaussianBlur(test_gray, (3, 3), 0)
    return template_smooth, test_smooth


def detect_defects(template_bgr, test_bgr, min_area):
    """Difference aligned grayscale images and turn persistent regions into boxes."""
    template_smooth, test_smooth = preprocess_images(template_bgr, test_bgr)

    # The provided pair is aligned, so absolute pixel difference localizes changed regions.
    difference = cv2.absdiff(template_smooth, test_smooth)

    # Otsu chooses a threshold from this pair's difference histogram automatically.
    _, thresholded = cv2.threshold(
        difference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Opening removes tiny specks; closing joins nearby fragments of the same defect.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # External contours provide one axis-aligned box for each connected difference region.
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        detections.append((x, y, x + width, y + height))

    return template_smooth, test_smooth, difference, thresholded, cleaned, detections


def box_iou(first, second):
    """Calculate intersection over union for two x1,y1,x2,y2 boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def calculate_metrics(detections, ground_truth, iou_threshold):
    """Greedily match each ground-truth box to at most one detection by IoU."""
    candidates = []
    for detection_index, detection in enumerate(detections):
        for truth_index, truth in enumerate(ground_truth):
            overlap = box_iou(detection, truth)
            if overlap >= iou_threshold:
                candidates.append((overlap, detection_index, truth_index))

    matched_detections = set()
    matched_truth = set()
    matched_ious = []
    for overlap, detection_index, truth_index in sorted(candidates, reverse=True):
        if detection_index not in matched_detections and truth_index not in matched_truth:
            matched_detections.add(detection_index)
            matched_truth.add(truth_index)
            matched_ious.append(overlap)

    true_positives = len(matched_detections)
    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = true_positives / (true_positives + false_positives) if detections else 0.0
    recall = true_positives / (true_positives + false_negatives) if ground_truth else 0.0
    mean_iou = sum(matched_ious) / true_positives if true_positives else 0.0
    return true_positives, false_positives, false_negatives, precision, recall, mean_iou


def draw_results(test_bgr, detections, ground_truth):
    """Overlay detected boxes in red and annotation boxes in green."""
    result = test_bgr.copy()
    for x1, y1, x2, y2 in ground_truth:
        cv2.rectangle(result, (x1, y1), (x2, y2), (50, 200, 50), 1)
    for index, (x1, y1, x2, y2) in enumerate(detections, start=1):
        cv2.rectangle(result, (x1, y1), (x2, y2), (30, 30, 230), 2)
        cv2.putText(result, f"D{index}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 230), 1, cv2.LINE_AA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def display_results(template_bgr, test_bgr, difference, thresholded, cleaned, result_rgb):
    """Show each requested processing stage together for easy comparison."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB), "Reference PCB", None),
        (cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB), "Defective PCB", None),
        (difference, "Absolute difference", "gray"),
        (thresholded, "Otsu threshold", "gray"),
        (cleaned, "After morphology", "gray"),
        (result_rgb, "Detections (red) and labels (green)", None),
    ]
    for axis, (image, title, color_map) in zip(axes.flat, panels):
        axis.imshow(image, cmap=color_map)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Detect PCB differences using classical image processing only."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Path to the extracted DeepPCB PCBData directory.")
    parser.add_argument("--sample", help="Sample ID, for example an ID shown by --list.")
    parser.add_argument("--list", action="store_true", help="List discovered samples and exit.")
    parser.add_argument("--min-area", type=float, default=12.0,
                        help="Ignore contours with a smaller pixel area (default: 12).")
    parser.add_argument("--iou-threshold", type=float, default=0.33,
                        help="IoU required for a match with an annotation (default: 0.33).")
    args = parser.parse_args()

    if not args.data_root.is_dir():
        parser.error(f"Dataset folder not found: {args.data_root}")
    samples = discover_samples(args.data_root)
    if not samples:
        parser.error(f"No complete test/template/annotation pairs found under {args.data_root}")

    if args.list:
        print(f"Found {len(samples)} complete samples under {args.data_root}")
        for sample_id in list(samples)[:20]:
            print(sample_id)
        if len(samples) > 20:
            print("... use --sample with any discovered ID")
        return

    sample_id = args.sample or next(iter(samples))
    if sample_id not in samples:
        parser.error(f"Sample {sample_id!r} not found. Run with --list to see available IDs.")

    template_path, test_path, annotation_path = samples[sample_id]
    template_bgr = cv2.imread(str(template_path))
    test_bgr = cv2.imread(str(test_path))
    if template_bgr is None or test_bgr is None:
        parser.error(f"Could not read the image pair for sample {sample_id}.")
    if template_bgr.shape[:2] != test_bgr.shape[:2]:
        parser.error(f"Image dimensions do not match for sample {sample_id}; expected aligned pairs.")

    ground_truth = load_annotations(annotation_path)
    _, _, difference, thresholded, cleaned, detections = detect_defects(
        template_bgr, test_bgr, args.min_area
    )
    tp, fp, fn, precision, recall, mean_iou = calculate_metrics(
        detections, ground_truth, args.iou_threshold
    )

    print(f"Sample: {sample_id}")
    print(f"Reference: {template_path}")
    print(f"Test image: {test_path}")
    print(f"Annotations: {annotation_path}")
    print(f"Detections: {len(detections)} | Ground truth: {len(ground_truth)}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | IoU threshold: {args.iou_threshold:.2f}")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | Mean matched IoU: {mean_iou:.3f}")

    result_rgb = draw_results(test_bgr, detections, ground_truth)
    display_results(template_bgr, test_bgr, difference, thresholded, cleaned, result_rgb)


if __name__ == "__main__":
    main()