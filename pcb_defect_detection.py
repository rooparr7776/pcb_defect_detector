"""PCB defect localization and classification with classical vision and deep learning.

Features:
1. Automatic Image Registration (ORB, SIFT, ECC)
2. Illumination-Invariant Differencing (Sobel Gradient, SSIM Structural, Hybrid, CLAHE)
3. Two-Stage Hybrid Detection (Classical Candidate Proposals + Stage-2 CNN Classification)
"""

import argparse
from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pcb_registration import (
    auto_register_pair,
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
    classify_detections,
    load_defect_classifier,
    train_classifier,
)


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
    """Read DeepPCB boxes; returns list of (x1, y1, x2, y2, class_id)."""
    boxes = []
    with annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line in annotation_file:
            values = re.split(r"[\s,]+", line.strip())
            if not values or not values[0]:
                continue
            if len(values) < 4:
                raise ValueError(f"Malformed annotation row in {annotation_path}: {line.strip()}")
            x1, y1, x2, y2 = map(int, values[:4])
            class_id = int(values[4]) if len(values) >= 5 else 1
            boxes.append((x1, y1, x2, y2, class_id))
    return boxes


def preprocess_images(template_bgr, test_bgr):
    """Convert an aligned pair to lightly blurred grayscale images."""
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY) if template_bgr.ndim == 3 else template_bgr
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY) if test_bgr.ndim == 3 else test_bgr

    template_smooth = cv2.GaussianBlur(template_gray, (3, 3), 0)
    test_smooth = cv2.GaussianBlur(test_gray, (3, 3), 0)
    return template_smooth, test_smooth


def detect_defects(
    template_bgr,
    test_bgr,
    min_area=12.0,
    registration="none",
    diff_method="raw",
    use_clahe=False,
    classifier_model=None,
):
    """Full PCB defect detection pipeline.
    
    1. Automatic image registration (ORB/SIFT/ECC) if requested.
    2. Light smoothing and optional CLAHE.
    3. Illumination-invariant differencing (raw, gradient, SSIM, hybrid).
    4. Otsu thresholding + morphological opening/closing.
    5. External contour extraction -> candidate RoI bounding boxes.
    6. Stage-2 CNN classification into specific defect types.
    """
    # Step 1: Automatic Registration
    if registration and registration.lower() != "none":
        aligned_test, homography, inliers, matches, kp_test, kp_temp = auto_register_pair(
            test_bgr, template_bgr, method=registration
        )
        reg_info = {
            "method": registration,
            "homography": homography,
            "inliers": inliers,
            "matches": matches,
            "kp_test": kp_test,
            "kp_temp": kp_temp,
        }
    else:
        aligned_test = test_bgr.copy()
        reg_info = {"method": "none", "homography": np.eye(3), "inliers": 0, "matches": []}

    # Step 2: Grayscale and Smoothing
    template_smooth, test_smooth = preprocess_images(template_bgr, aligned_test)

    # Step 3: Illumination-Invariant Differencing
    difference = compute_difference_map(
        template_smooth, test_smooth, method=diff_method, use_clahe=use_clahe
    )

    # Step 4: Otsu Binarization and Morphology
    _, thresholded = cv2.threshold(
        difference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Step 5: External Contours -> Bounding Boxes
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        detections.append((x, y, x + width, y + height))

    # Step 6: Stage 2 CNN Classification
    classified_results = classify_detections(
        test_smooth, template_smooth, difference, detections, model=classifier_model
    )

    return (
        template_smooth,
        test_smooth,
        difference,
        thresholded,
        cleaned,
        detections,
        aligned_test,
        reg_info,
        classified_results,
    )


def box_iou(first, second):
    """Calculate intersection over union for two (x1, y1, x2, y2) boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def calculate_metrics(detections, ground_truth, iou_threshold=0.33, classified_results=None):
    """Match detections to ground truth and calculate precision, recall, mean IoU, and class accuracy."""
    gt_boxes = [g[:4] for g in ground_truth]
    gt_classes = [g[4] if len(g) >= 5 else None for g in ground_truth]

    det_boxes = []
    det_classes = []
    if classified_results:
        for res in classified_results:
            det_boxes.append(res["box"])
            det_classes.append(res["class_id"])
    else:
        det_boxes = [d[:4] for d in detections]
        det_classes = [None] * len(det_boxes)

    candidates = []
    for d_idx, d_box in enumerate(det_boxes):
        for t_idx, t_box in enumerate(gt_boxes):
            overlap = box_iou(d_box, t_box)
            if overlap >= iou_threshold:
                candidates.append((overlap, d_idx, t_idx))

    matched_detections = set()
    matched_truth = set()
    matched_ious = []
    class_correct = 0

    for overlap, d_idx, t_idx in sorted(candidates, reverse=True):
        if d_idx not in matched_detections and t_idx not in matched_truth:
            matched_detections.add(d_idx)
            matched_truth.add(t_idx)
            matched_ious.append(overlap)
            if det_classes[d_idx] is not None and gt_classes[t_idx] is not None:
                if det_classes[d_idx] == gt_classes[t_idx]:
                    class_correct += 1

    true_positives = len(matched_detections)
    false_positives = len(det_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives
    precision = true_positives / (true_positives + false_positives) if det_boxes else 0.0
    recall = true_positives / (true_positives + false_negatives) if gt_boxes else 0.0
    mean_iou = sum(matched_ious) / true_positives if true_positives else 0.0
    class_accuracy = class_correct / true_positives if true_positives and any(c is not None for c in det_classes) else None

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "class_accuracy": class_accuracy,
        "matched_detections": matched_detections,
        "matched_truth": matched_truth,
    }


def draw_results(image_bgr, detections, ground_truth, classified_results=None):
    """Draw ground-truth labels (green) and detected defects color-coded by class."""
    result = image_bgr.copy()

    # Draw ground truth (green)
    for gt in ground_truth:
        x1, y1, x2, y2 = gt[:4]
        gt_class = gt[4] if len(gt) >= 5 else None
        label = f"GT: {DEFECT_CLASSES.get(gt_class, 'defect')}" if gt_class is not None else "GT"
        cv2.rectangle(result, (x1, y1), (x2, y2), (50, 200, 50), 1)
        cv2.putText(result, label, (x1, min(result.shape[0] - 4, y2 + 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (50, 200, 50), 1, cv2.LINE_AA)

    # Draw classified or generic detections
    if classified_results:
        for index, item in enumerate(classified_results, start=1):
            x1, y1, x2, y2 = item["box"]
            cname = item["class_name"]
            conf = item["confidence"]
            bgr_color = item["color"]
            cv2.rectangle(result, (x1, y1), (x2, y2), bgr_color, 2)
            badge = f"D{index}: {cname} ({conf:.0%})"
            cv2.putText(result, badge, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr_color, 1, cv2.LINE_AA)
    else:
        for index, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det[:4]
            cv2.rectangle(result, (x1, y1), (x2, y2), (30, 30, 230), 2)
            cv2.putText(result, f"D{index}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 230), 1, cv2.LINE_AA)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def display_results(template_bgr, test_bgr, difference, thresholded, cleaned, result_rgb, reg_info=None, diff_method="raw"):
    """Show each processing stage together in a clean 2x3 Matplotlib grid."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    figure.canvas.manager.set_window_title(f"PCB Defect Detection ({diff_method.upper()} Differencing)")

    reg_title = "Reference PCB (Clean)"
    test_title = "Defective PCB (Aligned)" if (reg_info and reg_info.get("method") != "none") else "Defective PCB"
    diff_title = f"{diff_method.capitalize()} Difference"

    panels = [
        (cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB), reg_title, None),
        (cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB), test_title, None),
        (difference, diff_title, "inferno" if diff_method != "raw" else "gray"),
        (thresholded, "Otsu Threshold Mask", "gray"),
        (cleaned, "After Morphology Clean", "gray"),
        (result_rgb, "Detections & Classified Types", None),
    ]

    for axis, (image, title, color_map) in zip(axes.flat, panels):
        axis.imshow(image, cmap=color_map)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="PCB Defect Detection with Auto-Registration, Illumination Invariance, and CNN Classification."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Path to extracted DeepPCB PCBData directory.")
    parser.add_argument("--sample", help="Sample ID, for example an ID shown by --list.")
    parser.add_argument("--list", action="store_true", help="List discovered samples and exit.")
    parser.add_argument("--min-area", type=float, default=12.0,
                        help="Ignore contours with smaller pixel area (default: 12).")
    parser.add_argument("--iou-threshold", type=float, default=0.33,
                        help="IoU required for match with annotation (default: 0.33).")
    parser.add_argument("--registration", choices=["none", "orb", "sift", "ecc"], default="none",
                        help="Auto-registration algorithm to align images (default: none).")
    parser.add_argument("--diff-method", choices=["raw", "gradient", "ssim", "hybrid"], default="raw",
                        help="Differencing method (default: raw).")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast equalization before differencing.")
    parser.add_argument("--classify", action="store_true", help="Enable Stage-2 lightweight CNN defect classification.")
    parser.add_argument("--train-classifier", action="store_true", help="Train the CNN classifier on dataset patches.")
    parser.add_argument("--simulate-misalignment", action="store_true",
                        help="Add synthetic shift and rotation to test auto-registration.")
    parser.add_argument("--simulate-lighting", action="store_true",
                        help="Add synthetic brightness and contrast changes to test illumination invariance.")

    args = parser.parse_args()

    if args.train_classifier:
        print("Training LightweightDefectCNN on dataset...")
        train_classifier(args.data_root)
        return

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
        parser.error(f"Could not read image pair for sample {sample_id}.")

    # Optional simulations for testing novelty
    if args.simulate_misalignment:
        print("[Simulation] Adding 12px shift and 2.5 deg rotation to test board...")
        test_bgr, _ = simulate_misalignment(test_bgr, shift_x=12, shift_y=-8, angle_deg=2.5)

    if args.simulate_lighting:
        print("[Simulation] Adding lighting shift (+45 brightness, 1.25x contrast)...")
        test_bgr = simulate_illumination_change(test_bgr, brightness_offset=45, contrast_factor=1.25)

    # Optional CNN classifier
    classifier_model = None
    if args.classify:
        classifier_model = load_defect_classifier()
        if classifier_model is None:
            print("[Warning] Trained classifier weights not found. Run with --train-classifier first.")

    ground_truth = load_annotations(annotation_path)

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
        test_bgr=test_bgr,
        min_area=args.min_area,
        registration=args.registration,
        diff_method=args.diff_method,
        use_clahe=args.clahe,
        classifier_model=classifier_model,
    )

    metrics = calculate_metrics(
        detections=detections,
        ground_truth=ground_truth,
        iou_threshold=args.iou_threshold,
        classified_results=classified_results if args.classify else None,
    )

    print("\n" + "=" * 55)
    print(f"Sample ID:              {sample_id}")
    print(f"Registration Method:    {args.registration.upper()} (Inliers: {reg_info.get('inliers', 0)})")
    print(f"Differencing Engine:    {args.diff_method.upper()} (CLAHE: {args.clahe})")
    print(f"CNN Classifier Enabled: {args.classify}")
    print(f"Detected Regions:       {len(detections)}")
    print(f"Ground Truth Labels:    {len(ground_truth)}")
    print(f"True Positives (TP):    {metrics['true_positives']}")
    print(f"False Positives (FP):   {metrics['false_positives']}")
    print(f"False Negatives (FN):   {metrics['false_negatives']}")
    print(f"Precision:              {metrics['precision']:.1%}")
    print(f"Recall:                 {metrics['recall']:.1%}")
    print(f"Mean IoU:               {metrics['mean_iou']:.3f}")
    if metrics["class_accuracy"] is not None:
        print(f"Classification Acc:     {metrics['class_accuracy']:.1%}")
    print("=" * 55 + "\n")

    if args.classify and classified_results:
        print("Classified Defects:")
        for idx, item in enumerate(classified_results, start=1):
            x1, y1, x2, y2 = item["box"]
            print(f"  [D{idx}] {item['class_name'].upper():<12} | Conf: {item['confidence']:.1%} | Bounding Box: ({x1}, {y1}) -> ({x2}, {y2})")

    result_rgb = draw_results(
        aligned_test,
        detections=detections,
        ground_truth=ground_truth,
        classified_results=classified_results if args.classify else None,
    )

    display_results(
        template_bgr=template_bgr,
        test_bgr=aligned_test,
        difference=difference,
        thresholded=thresholded,
        cleaned=cleaned,
        result_rgb=result_rgb,
        reg_info=reg_info,
        diff_method=args.diff_method,
    )


if __name__ == "__main__":
    main()