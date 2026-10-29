# PCB Surface Defect Detection & Classification

An advanced PCB defect inspection system featuring **Automatic Image Registration**, **Illumination-Invariant Differencing**, and a **Two-Stage Hybrid Detector** that pairs classical candidate localization with a lightweight PyTorch CNN classifier for the DeepPCB benchmark.

---

## Key Features & Novelties

### 1. Automatic Image Registration (Homography Alignment)
- Automatically aligns test boards to reference boards even if misaligned due to camera vibration or conveyor movement.
- Implements:
  - **ORB**: Fast, robust binary keypoint matching with RANSAC homography.
  - **SIFT**: Scale and rotation-invariant feature matching.
  - **ECC**: Enhanced Correlation Coefficient maximization for subpixel intensity fine-alignment.
- Visual inlier match lines displayable in the UI.

### 2. Illumination-Invariant & Structural Differencing
- Resolves the major vulnerability of traditional pixel subtraction under factory lighting shifts.
- Implements:
  - **Sobel Gradient Magnitude Differencing**: Mathematically invariant to uniform lighting and exposure changes ($\nabla(I + C) = \nabla I$).
  - **SSIM (Structural Similarity Index) Differencing**: Invariant to both illumination shifts and contrast scaling, isolating structural defects.
  - **Hybrid Fusion**: Blends gradient edge transitions with local structural similarity.
  - **CLAHE**: Contrast Limited Adaptive Histogram Equalization for local contrast normalization.

### 3. Two-Stage Hybrid Detector (Classical RoIs + Stage-2 CNN)
- **Stage 1 (Rapid Candidate Proposals):** High-speed morphological difference contour detection extracts bounding boxes in milliseconds without heavy object-detection anchors.
- **Stage 2 (Lightweight CNN Classifier):** A compact PyTorch convolutional neural network (~100k parameters) ingests 3-channel context patches (`[test, template, difference]`) and classifies defects into DeepPCB categories:
  - `open` (broken circuit trace)
  - `short` (accidental bridge between traces)
  - `mousebite` (nick/notch along a copper edge)
  - `spur` (unwanted pointed protrusion)
  - `copper` (spurious isolated metal speck)
  - `pin-hole` (void or drill anomaly)
  - `background` (false-positive rejection)

---

## Installation

Install required dependencies:

```powershell
python -m pip install opencv-python numpy matplotlib streamlit torch torchvision
```

Ensure `PCBData` is placed in `DeepPCB-master/PCBData/`.

---

## Interactive Web Dashboard

Launch the Streamlit web dashboard:

```powershell
python -m streamlit run pcb_defect_web.py
```

### Dashboard Capabilities:
- **Sample Selector:** Choose from 1,500 DeepPCB pairs.
- **Auto-Registration:** Toggle ORB, SIFT, or ECC alignment.
- **Misalignment Simulator:** Adjust Shift X, Shift Y, and Rotation sliders to watch Auto-Registration restore alignment in real time.
- **Differencing Engine:** Switch between Raw, Sobel Gradient, SSIM, and Hybrid differencing.
- **Illumination Simulator:** Adjust Brightness and Contrast sliders to evaluate illumination invariance.
- **CNN Classification:** View color-coded bounding boxes, confidence badges, and an interactive defect data table.
- **Paired-Image Faster R-CNN:** Train a pretrained-backbone object detector using the test board, its template, and their absolute difference; small-object anchors help retain DeepPCB's small defects.
- **Diagnostic Labs:** Inspect feature matching lines and side-by-side error heatmaps.
- **Batch Evaluation:** Run precision, recall, F1, mAP@0.5, per-class AP@0.5, mean IoU, and class accuracy on the official `test.txt` split and download per-board CSV results. The dashboard defaults to Hybrid differencing, 200 px minimum contour area, and a 0.85 CNN confidence cutoff. A separately labeled reference run reports the previously measured official test results for those settings; rerun the evaluator to measure the current selection.

### Train the paired-image DeepPCB detector

The Faster R-CNN trains only on pairs listed in `trainval.txt`. It holds out complete board groups from that split for validation, chooses its confidence cutoff by validation F1, and saves the checkpoint to `models/deeppcb_fasterrcnn.pth`. The official `test.txt` split is not used for training or threshold selection.

```powershell
python pcb_defect_detection.py --train-detector --detector-epochs 10 --detector-batch-size 2
```

Training starts from COCO-pretrained Faster R-CNN ResNet-50 weights, so an internet connection is needed the first time unless those weights are already cached. Training on CPU is slow; CUDA is preferred. To check the training flow quickly on CPU, run one epoch on a small subset and save it under a separate diagnostic filename:

```powershell
python -u pcb_defect_detection.py --train-detector --detector-epochs 1 --detector-batch-size 2 --max-training-pairs 8 --max-validation-pairs 8 --detector-output models/deeppcb_fasterrcnn_smoke.pth
```

This diagnostic model and its small validation sample are not suitable for reporting benchmark performance. For a real run, omit both pair limits and use the normal output path. Training now prints batch and validation progress with elapsed-time estimates. After training, select **Use paired-image Faster R-CNN** in the dashboard or run:

```powershell
python pcb_defect_detection.py --sample 00041000 --detector
```

Use the dashboard's **Batch Evaluation** tab to compare the trained detector with the contour-based baseline on the official test split. The new detector reports precision, recall, F1, mAP@0.5, per-class AP@0.5, and localization metrics; results depend on training and should not be assumed to meet a target before evaluation. Detection metrics are more informative than image-level accuracy because true-negative counts are not well-defined for bounding-box detection.

---

## Command-Line Usage

Run with all advanced features enabled:

```powershell
python pcb_defect_detection.py --sample 00041000 --registration orb --diff-method ssim --classify
```

### CLI Arguments:
- `--registration`: Auto-registration method (`none`, `orb`, `sift`, `ecc`).
- `--diff-method`: Differencing engine (`raw`, `gradient`, `ssim`, `hybrid`).
- `--clahe`: Enable CLAHE contrast equalization.
- `--classify`: Enable Stage-2 CNN classification.
- `--min-area`: Minimum contour area cutoff in pixels (default: `12`).
- `--iou-threshold`: IoU threshold for matching against dataset labels (default: `0.33`).
- `--simulate-misalignment`: Adds synthetic shift and rotation to test auto-registration.
- `--simulate-lighting`: Adds synthetic brightness and contrast changes to test illumination invariance.
- `--train-classifier`: Retrains the lightweight CNN classifier on dataset patches.