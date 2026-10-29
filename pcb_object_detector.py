"""Paired-image Faster R-CNN training and inference for the DeepPCB dataset."""

from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pcb_classifier import DEFECT_CLASSES, DEFECT_COLORS


DEFAULT_DETECTOR_PATH = Path(__file__).resolve().parent / "models" / "deeppcb_fasterrcnn.pth"
NUM_DETECTOR_CLASSES = len(DEFECT_CLASSES)
INPUT_SIZE = 640


def _read_boxes(annotation_path):
    boxes = []
    with annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line in annotation_file:
            values = line.split()
            if not values:
                continue
            if len(values) < 4:
                raise ValueError(f"Malformed annotation row in {annotation_path}: {line.strip()}")
            x1, y1, x2, y2 = map(int, values[:4])
            class_id = int(values[4]) if len(values) >= 5 else 1
            if class_id not in range(1, NUM_DETECTOR_CLASSES):
                raise ValueError(f"Invalid DeepPCB class ID {class_id} in {annotation_path}")
            boxes.append((x1, y1, x2, y2, class_id))
    return boxes


def _paired_image(template_gray, test_gray):
    if template_gray.shape != test_gray.shape:
        raise ValueError("DeepPCB template and test images must have matching dimensions.")
    difference = cv2.absdiff(test_gray, template_gray)
    paired = np.stack((test_gray, template_gray, difference), axis=2)
    return torch.from_numpy(paired.transpose(2, 0, 1).copy()).float().div_(255.0)


class DeepPCBDetectionDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = list(samples.items())
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, (template_path, test_path, annotation_path) = self.samples[index]
        template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        test_gray = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)
        if template_gray is None or test_gray is None:
            raise OSError(f"Could not read DeepPCB images for sample {sample_id}.")

        height, width = test_gray.shape[:2]
        boxes = []
        labels = []
        for x1, y1, x2, y2, class_id in _read_boxes(annotation_path):
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
                labels.append(class_id)

        image = _paired_image(template_gray, test_gray)
        box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.as_tensor(labels, dtype=torch.int64)

        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
                if box_tensor.numel():
                    old_x1 = box_tensor[:, 0].clone()
                    box_tensor[:, 0] = width - box_tensor[:, 2]
                    box_tensor[:, 2] = width - old_x1
            if random.random() < 0.5:
                image = image.flip(-2)
                if box_tensor.numel():
                    old_y1 = box_tensor[:, 1].clone()
                    box_tensor[:, 1] = height - box_tensor[:, 3]
                    box_tensor[:, 3] = height - old_y1

        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0])
            * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": torch.zeros((len(box_tensor),), dtype=torch.int64),
        }
        return image, target


def _collate_detection_batch(batch):
    return tuple(zip(*batch))


def create_detector(pretrained=False):
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model_options = {
        "rpn_anchor_generator": anchor_generator,
        "min_size": INPUT_SIZE,
        "max_size": INPUT_SIZE,
        "image_mean": (0.5, 0.5, 0.1),
        "image_std": (0.25, 0.25, 0.15),
        "box_detections_per_img": 300,
        "box_score_thresh": 0.001,
        "rpn_pre_nms_top_n_train": 2000,
        "rpn_post_nms_top_n_train": 1000,
        "rpn_pre_nms_top_n_test": 1000,
        "rpn_post_nms_top_n_test": 500,
    }
    if pretrained:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=None,
            **model_options,
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=NUM_DETECTOR_CLASSES,
            **model_options,
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_DETECTOR_CLASSES)
    return model


def load_paired_detector(weights_path=DEFAULT_DETECTOR_PATH, device="cpu"):
    checkpoint_path = Path(weights_path)
    if not checkpoint_path.is_file():
        return None, None, None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported detector checkpoint format: {checkpoint_path}")
    model = create_detector()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, float(checkpoint["confidence_threshold"]), checkpoint


def predict_paired_image(
    model,
    template_gray,
    test_gray,
    confidence_threshold=0.25,
    device=None,
):
    device = torch.device(device) if device else next(model.parameters()).device
    model.eval()
    image = _paired_image(template_gray, test_gray).to(device)
    with torch.inference_mode():
        prediction = model([image])[0]

    height, width = test_gray.shape[:2]
    detections = []
    for box, class_id, confidence in zip(
        prediction["boxes"].cpu().numpy(),
        prediction["labels"].cpu().numpy(),
        prediction["scores"].cpu().numpy(),
    ):
        if confidence < confidence_threshold:
            continue
        x1, y1, x2, y2 = box
        clipped_box = (
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(0, min(width, int(round(x2)))),
            max(0, min(height, int(round(y2)))),
        )
        if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
            continue
        label = int(class_id)
        detections.append({
            "box": clipped_box,
            "class_id": label,
            "class_name": DEFECT_CLASSES.get(label, "unknown"),
            "confidence": float(confidence),
            "color": DEFECT_COLORS.get(label, (200, 200, 200)),
        })
    return detections


def _training_and_validation_samples(data_root, samples, validation_fraction=0.2):
    manifest_path = Path(data_root) / "trainval.txt"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DeepPCB training manifest not found: {manifest_path}")

    train_ids = set()
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line in manifest:
            values = line.split()
            if values:
                train_ids.add(Path(values[0]).stem)
    trainval_samples = {
        sample_id: samples[sample_id]
        for sample_id in sorted(train_ids)
        if sample_id in samples
    }
    if not trainval_samples:
        raise ValueError(f"No discovered samples match {manifest_path}")

    groups = {}
    for sample_id, sample in trainval_samples.items():
        group_name = sample[1].parent.parent.name
        groups.setdefault(group_name, {})[sample_id] = sample
    if len(groups) < 2:
        raise ValueError("At least two DeepPCB board groups are required for group-wise validation.")

    group_names = sorted(groups)
    random.Random(42).shuffle(group_names)
    validation_group_count = max(1, round(len(group_names) * validation_fraction))
    validation_group_count = min(validation_group_count, len(group_names) - 1)
    validation_groups = set(group_names[:validation_group_count])
    train_samples = {
        sample_id: sample
        for group_name, group_samples in groups.items()
        if group_name not in validation_groups
        for sample_id, sample in group_samples.items()
    }
    validation_samples = {
        sample_id: sample
        for group_name, group_samples in groups.items()
        if group_name in validation_groups
        for sample_id, sample in group_samples.items()
    }
    return train_samples, validation_samples, sorted(validation_groups)


def _evaluate_predictions(predictions, samples, iou_threshold=0.33):
    from pcb_defect_detection import calculate_metrics, load_annotations

    ground_truth_by_sample = {
        sample_id: load_annotations(samples[sample_id][2])
        for sample_id in predictions
    }
    totals = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
    for sample_id, results in predictions.items():
        metrics = calculate_metrics(
            detections=[result["box"] for result in results],
            ground_truth=ground_truth_by_sample[sample_id],
            iou_threshold=iou_threshold,
            classified_results=results,
        )
        for key in totals:
            totals[key] += metrics[key]
    precision = totals["true_positives"] / max(
        1, totals["true_positives"] + totals["false_positives"]
    )
    recall = totals["true_positives"] / max(
        1, totals["true_positives"] + totals["false_negatives"]
    )
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {**totals, "precision": precision, "recall": recall, "f1": f1}


def calculate_map50(predictions_by_sample, ground_truth_by_sample):
    """Calculate per-class AP and mean AP at an IoU threshold of 0.5."""
    from pcb_defect_detection import box_iou

    average_precisions = {}
    for class_id, class_name in DEFECT_CLASSES.items():
        if class_id == 0:
            continue
        total_ground_truth = sum(
            sum(1 for box in boxes if len(box) >= 5 and box[4] == class_id)
            for boxes in ground_truth_by_sample.values()
        )
        if not total_ground_truth:
            average_precisions[class_name] = None
            continue

        candidates = []
        matched_ground_truth = {
            sample_id: set() for sample_id in ground_truth_by_sample
        }
        for sample_id, detections in predictions_by_sample.items():
            for detection in detections:
                if detection["class_id"] == class_id:
                    candidates.append((
                        detection["confidence"],
                        sample_id,
                        detection["box"],
                    ))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        true_positives = []
        false_positives = []
        for _, sample_id, detection_box in candidates:
            ground_truth = ground_truth_by_sample[sample_id]
            eligible = [
                (index, box)
                for index, box in enumerate(ground_truth)
                if len(box) >= 5 and box[4] == class_id
            ]
            best_match = max(
                (
                    (box_iou(detection_box, box[:4]), index)
                    for index, box in eligible
                    if index not in matched_ground_truth[sample_id]
                ),
                default=(0.0, None),
            )
            if best_match[0] >= 0.5:
                matched_ground_truth[sample_id].add(best_match[1])
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)

        cumulative_true_positives = np.cumsum(true_positives)
        cumulative_false_positives = np.cumsum(false_positives)
        recalls = cumulative_true_positives / total_ground_truth
        precisions = cumulative_true_positives / np.maximum(
            cumulative_true_positives + cumulative_false_positives, 1
        )
        recall_points = np.linspace(0.0, 1.0, 101)
        interpolated_precision = [
            np.max(precisions[recalls >= recall]) if np.any(recalls >= recall) else 0.0
            for recall in recall_points
        ]
        average_precisions[class_name] = float(np.mean(interpolated_precision))

    valid_aps = [
        average_precision
        for average_precision in average_precisions.values()
        if average_precision is not None
    ]
    mean_average_precision = float(np.mean(valid_aps)) if valid_aps else 0.0
    return mean_average_precision, average_precisions


def train_paired_detector(
    data_root,
    epochs=10,
    batch_size=2,
    learning_rate=0.002,
    max_pairs=None,
    max_validation_pairs=None,
    save_path=DEFAULT_DETECTOR_PATH,
    device=None,
):
    from pcb_defect_detection import discover_samples

    data_root = Path(data_root)
    samples = discover_samples(data_root)
    if not samples:
        raise ValueError(f"No complete DeepPCB samples found under {data_root}")
    train_samples, validation_samples, validation_groups = _training_and_validation_samples(
        data_root, samples
    )
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be at least 1.")
        if max_pairs < len(train_samples):
            selected_ids = sorted(train_samples)
            random.Random(42).shuffle(selected_ids)
            train_samples = {sample_id: train_samples[sample_id] for sample_id in selected_ids[:max_pairs]}
    if max_validation_pairs is not None:
        if max_validation_pairs < 1:
            raise ValueError("max_validation_pairs must be at least 1.")
        if max_validation_pairs < len(validation_samples):
            selected_ids = sorted(validation_samples)
            random.Random(42).shuffle(selected_ids)
            validation_samples = {
                sample_id: validation_samples[sample_id]
                for sample_id in selected_ids[:max_validation_pairs]
            }

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    print(
        f"Preparing paired-image detector on {device}; "
        f"{len(train_samples)} training pairs, {len(validation_samples)} validation pairs, "
        f"{epochs} epochs, batch size {batch_size}.",
        flush=True,
    )
    train_dataset = DeepPCBDetectionDataset(train_samples, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_detection_batch,
    )
    print("Loading COCO-pretrained Faster R-CNN weights...", flush=True)
    model = create_detector(pretrained=True).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 3), gamma=0.3
    )

    total_steps = len(train_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_started = time.perf_counter()
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            loss_values = model(images, targets)
            loss = sum(loss_values.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite detector loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_steps:
                elapsed = time.perf_counter() - epoch_started
                seconds_per_step = elapsed / batch_index
                remaining = seconds_per_step * (total_steps - batch_index)
                print(
                    f"Epoch {epoch}/{epochs} | batch {batch_index}/{total_steps} "
                    f"| loss {float(loss.detach()):.4f} "
                    f"| elapsed {elapsed / 60:.1f} min "
                    f"| estimated epoch remaining {remaining / 60:.1f} min",
                    flush=True,
                )
        scheduler.step()
        average_loss = epoch_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch}/{epochs}: training loss={average_loss:.4f} "
            f"({len(train_samples)} train pairs; validation groups: "
            f"{', '.join(validation_groups)})",
            flush=True,
        )

    model.eval()
    validation_predictions = {}
    print(
        f"Running validation inference on {len(validation_samples)} pairs...",
        flush=True,
    )
    validation_started = time.perf_counter()
    for index, (sample_id, (template_path, test_path, _)) in enumerate(
        validation_samples.items(), start=1
    ):
        template_gray = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        test_gray = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)
        if template_gray is None or test_gray is None:
            raise OSError(f"Could not read validation images for sample {sample_id}.")
        validation_predictions[sample_id] = predict_paired_image(
            model, template_gray, test_gray, confidence_threshold=0.0, device=device
        )
        if index == 1 or index % 10 == 0 or index == len(validation_samples):
            elapsed = time.perf_counter() - validation_started
            remaining = elapsed / index * (len(validation_samples) - index)
            print(
                f"Validation {index}/{len(validation_samples)} "
                f"| elapsed {elapsed / 60:.1f} min "
                f"| estimated remaining {remaining / 60:.1f} min",
                flush=True,
            )

    threshold_results = []
    for threshold in np.arange(0.05, 0.901, 0.05):
        filtered_predictions = {
            sample_id: [
                result for result in results if result["confidence"] >= threshold
            ]
            for sample_id, results in validation_predictions.items()
        }
        metrics = _evaluate_predictions(filtered_predictions, validation_samples)
        threshold_results.append({"threshold": float(threshold), **metrics})
    best_validation = max(
        threshold_results,
        key=lambda result: (
            result["f1"],
            result["precision"],
            result["threshold"],
        ),
    )
    ground_truth_by_sample = {
        sample_id: _read_boxes(paths[2])
        for sample_id, paths in validation_samples.items()
    }
    best_validation["map50"], best_validation["ap50_per_class"] = calculate_map50(
        validation_predictions, ground_truth_by_sample
    )

    checkpoint_path = Path(save_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.cpu().state_dict(),
        "confidence_threshold": best_validation["threshold"],
        "validation_metrics": best_validation,
        "validation_groups": validation_groups,
        "train_pairs": len(train_samples),
        "validation_pairs": len(validation_samples),
        "input_size": INPUT_SIZE,
        "class_names": DEFECT_CLASSES,
    }
    torch.save(checkpoint, checkpoint_path)
    model.to(device)
    print(
        f"Saved detector: {checkpoint_path}\n"
        f"Validation threshold={best_validation['threshold']:.2f}, "
        f"precision={best_validation['precision']:.1%}, "
        f"recall={best_validation['recall']:.1%}, "
        f"F1={best_validation['f1']:.1%}",
        flush=True,
    )
    return model, best_validation
