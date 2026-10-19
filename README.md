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
python -m pip install opencv-python numpy matplotlib streamlit torch
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
- **Diagnostic Labs:** Inspect feature matching lines and side-by-side error heatmaps.

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