"""Two-stage hybrid PCB defect detector: Stage 2 Lightweight CNN classifier.

Takes candidate Regions of Interest (RoIs) from classical image processing,
crops local multi-channel patches (test, template, difference), and classifies
each defect into the DeepPCB classes:
0: normal / background (false-positive rejection)
1: open
2: short
3: mousebite
4: spur
5: copper (spurious copper)
6: pin-hole
"""

from pathlib import Path
import random
import re
import cv2
import numpy as np
import torch
import torch.nn as nn


DEFECT_CLASSES = {
    0: "background",
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

# Distinct, high-visibility colors (BGR for OpenCV, RGB for Streamlit/Matplotlib)
DEFECT_COLORS = {
    0: (140, 140, 140),   # Gray
    1: (30, 30, 230),     # Red (Open)
    2: (0, 140, 255),     # Orange (Short)
    3: (200, 30, 200),    # Magenta (Mousebite)
    4: (230, 180, 20),    # Cyan (Spur)
    5: (30, 210, 230),    # Yellow/Amber (Copper)
    6: (50, 200, 50),     # Green (Pin-hole)
}

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "defect_classifier.pth"


class LightweightDefectCNN(nn.Module):
    """Ultra-lightweight CNN for classifying 64x64 multi-channel RoI patches (~100k params)."""

    def __init__(self, in_channels=3, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)


def crop_roi_patch(test_gray, temp_gray, diff_gray, bbox, patch_size=(64, 64), margin=10):
    """Extract a 3-channel (test, template, difference) patch around a bounding box."""
    h, w = test_gray.shape[:2]
    x1, y1, x2, y2 = bbox

    # Expand box slightly to give the CNN spatial circuit context
    x1_exp = max(0, x1 - margin)
    y1_exp = max(0, y1 - margin)
    x2_exp = min(w, x2 + margin)
    y2_exp = min(h, y2 + margin)

    crop_test = test_gray[y1_exp:y2_exp, x1_exp:x2_exp]
    crop_temp = temp_gray[y1_exp:y2_exp, x1_exp:x2_exp]
    crop_diff = diff_gray[y1_exp:y2_exp, x1_exp:x2_exp]

    if crop_test.size == 0 or crop_temp.size == 0 or crop_diff.size == 0:
        patch = np.zeros((3, patch_size[1], patch_size[0]), dtype=np.float32)
        return patch

    res_test = cv2.resize(crop_test, patch_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    res_temp = cv2.resize(crop_temp, patch_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    res_diff = cv2.resize(crop_diff, patch_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    patch = np.stack([res_test, res_temp, res_diff], axis=0)
    return patch


def load_defect_classifier(weights_path=None, device="cpu"):
    """Load trained CNN classifier weights."""
    model_path = Path(weights_path) if weights_path else DEFAULT_MODEL_PATH
    model = LightweightDefectCNN(in_channels=3, num_classes=7)
    if model_path.is_file():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    return None


def classify_detections(test_gray, temp_gray, diff_gray, detections, model=None, device="cpu"):
    """Classify candidate detection boxes into specific DeepPCB defect categories."""
    if not detections:
        return []

    if model is None:
        # Fallback: return generic detection without classification
        return [
            {
                "box": box,
                "class_id": 1,
                "class_name": "defect",
                "confidence": 1.0,
                "color": DEFECT_COLORS[1],
            }
            for box in detections
        ]

    patches = []
    for box in detections:
        p = crop_roi_patch(test_gray, temp_gray, diff_gray, box)
        patches.append(p)

    batch = torch.tensor(np.array(patches), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

    results = []
    for box, pred_id, conf in zip(detections, preds.cpu().numpy(), confidences.cpu().numpy()):
        cid = int(pred_id)
        results.append({
            "box": box,
            "class_id": cid,
            "class_name": DEFECT_CLASSES.get(cid, "unknown"),
            "confidence": float(conf),
            "color": DEFECT_COLORS.get(cid, (200, 200, 200)),
        })
    return results


def extract_training_dataset(samples, max_pairs=300):
    """Extract multi-channel patches and ground truth labels from DeepPCB samples."""
    patches = []
    labels = []

    pair_list = list(samples.items())
    random.seed(42)
    random.shuffle(pair_list)
    selected = pair_list[:max_pairs]

    for sample_id, (temp_path, test_path, ann_path) in selected:
        temp_img = cv2.imread(str(temp_path), cv2.IMREAD_GRAYSCALE)
        test_img = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)
        if temp_img is None or test_img is None:
            continue

        diff_img = cv2.absdiff(temp_img, test_img)
        h, w = test_img.shape[:2]

        # Read annotations
        defect_boxes = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = re.split(r"[\s,]+", line.strip())
                if len(parts) >= 5:
                    x1, y1, x2, y2, cid = map(int, parts[:5])
                    defect_boxes.append((x1, y1, x2, y2, cid))

        # Extract positive defect crops
        for x1, y1, x2, y2, cid in defect_boxes:
            if cid in DEFECT_CLASSES:
                patch = crop_roi_patch(test_img, temp_img, diff_img, (x1, y1, x2, y2))
                patches.append(patch)
                labels.append(cid)

        # Extract negative normal/background crops
        for _ in range(max(1, len(defect_boxes) // 2)):
            rx = random.randint(30, max(31, w - 70))
            ry = random.randint(30, max(31, h - 70))
            bg_box = (rx, ry, rx + 40, ry + 40)
            # Ensure no overlap with any defect box
            overlap = False
            for dx1, dy1, dx2, dy2, _ in defect_boxes:
                if not (bg_box[2] < dx1 or bg_box[0] > dx2 or bg_box[3] < dy1 or bg_box[1] > dy2):
                    overlap = True
                    break
            if not overlap:
                patch = crop_roi_patch(test_img, temp_img, diff_img, bg_box)
                patches.append(patch)
                labels.append(0)

    return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int64)


def train_classifier(data_root, max_pairs=350, epochs=8, batch_size=64, lr=1e-3, save_path=None):
    """Train the lightweight CNN classifier on DeepPCB patches."""
    from pcb_defect_detection import discover_samples
    samples = discover_samples(Path(data_root))
    if not samples:
        raise ValueError(f"No samples found in {data_root}")

    print(f"Extracting patches from {min(len(samples), max_pairs)} image pairs...")
    x_data, y_data = extract_training_dataset(samples, max_pairs=max_pairs)
    print(f"Dataset extracted: {len(x_data)} patches total.")

    # Class balance weighting
    class_counts = np.bincount(y_data, minlength=7)
    weights = np.zeros(7, dtype=np.float32)
    for c in range(7):
        weights[c] = (len(y_data) / (7.0 * max(1, class_counts[c])))
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    # Train / Val split
    indices = np.arange(len(x_data))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    x_train, y_train = torch.tensor(x_data[train_idx]), torch.tensor(y_data[train_idx])
    x_val, y_val = torch.tensor(x_data[val_idx]), torch.tensor(y_data[val_idx])

    model = LightweightDefectCNN(in_channels=3, num_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Beginning training for {epochs} epochs...")
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = torch.argmax(out, dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(x_val)
            val_preds = torch.argmax(val_out, dim=1)
            val_acc = (val_preds == y_val).sum().item() / max(1, len(y_val))

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")
        if val_acc > best_acc:
            best_acc = val_acc

    # Save weights
    out_file = Path(save_path) if save_path else DEFAULT_MODEL_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_file)
    print(f"Model saved to: {out_file} (Best Val Acc: {best_acc:.1%})")
    return model
