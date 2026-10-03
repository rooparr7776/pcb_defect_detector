# PCB Surface Defect Detection

A simple PCB defect-localization project using classical image processing with Python and OpenCV. It compares each defective PCB image with its aligned defect-free reference, highlights changed regions, and compares detected boxes with the dataset annotations. No machine-learning or deep-learning model is used.

## Get the Project

Clone this repository and enter the project folder:

```powershell
git clone https://github.com/rooparr7776/pcb_defect_detector.git
cd pcb_defect_detector
```

## Dataset

The dataset is required to run the detector and is **not included in this repository**. Obtain DeepPCB from the [original project](https://github.com/tangsanli5201/DeepPCB) and, following its terms, extract or copy its `PCBData` folder into this project's `DeepPCB-master` folder. The default location should be:

```text
pcb_defect_detector/
  DeepPCB-master/
    PCBData/
```

Each sample inside `PCBData` is organized as:

```text
PCBData/
  groupNNNNN/
    NNNNN/
      <sample_id>_temp.jpg  # defect-free reference
      <sample_id>_test.jpg  # defective test image
    NNNNN_not/
      <sample_id>.txt       # annotated defect boxes
```

The annotation rows contain `x1 y1 x2 y2 class_id`. The scripts discover image pairs and matching annotation files from the extracted folders.

The dataset's documentation restricts its use to research purposes. Do not commit or push the dataset; this repository ignores `DeepPCB-master/PCBData/`.

## Install

From the project folder, install the required packages:

```powershell
python -m pip install opencv-python numpy matplotlib streamlit
```

If you put `PCBData` somewhere else, provide its path to the command-line program with `--data-root`. In the web app, enter the path in the **PCBData folder** field in the sidebar.

## View Results in the Browser

Start the local web app:

```powershell
python -m streamlit run pcb_defect_web.py
```

Open [http://localhost:8501](http://localhost:8501). Use **Dataset sample** in the sidebar to choose an image pair. The page shows the reference image, test image, absolute difference, threshold mask, cleaned mask, and final overlay. Detected boxes are red; annotation boxes are green. You can adjust the minimum contour area and IoU match threshold in the sidebar.

Keep the terminal running while using the page. Press `Ctrl+C` in that terminal to stop the server.

## Run the Matplotlib Version

List available sample IDs:

```powershell
python pcb_defect_detection.py --list
```

Run one sample by ID:

```powershell
python pcb_defect_detection.py --sample 00041000
```

The image panels open in a Matplotlib window, and the detection metrics print in the terminal. If `--sample` is omitted, the first discovered sample is used.

## Processing Steps

1. Convert the aligned reference and test images to grayscale and apply a light Gaussian blur to reduce pixel noise.
2. Calculate their absolute difference.
3. Apply Otsu thresholding to separate changed pixels from the background.
4. Apply morphological opening to remove small specks and closing to connect nearby changed pixels.
5. Find external contours and draw a bounding box around each region above the minimum-area setting.
6. Compare detections with ground-truth boxes using one-to-one matching at the selected IoU threshold.

The evaluation is class-agnostic: the image-processing method finds changed regions but does not identify defect types. Precision, recall, and mean matched IoU therefore measure localization only. Results depend on the thresholding and contour-area settings; this is a simple educational baseline, not a tuned production detector.