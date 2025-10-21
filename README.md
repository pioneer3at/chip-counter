## Chip Counter

Helpers and a simple pipeline to:

- Press one button
- Turn on an LED
- Capture an image from a CSI camera (with desktop fallback)
- Detect quantity of chips using a single-stage multi-class detector (YOLOv8)

### TL;DR

```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run a one-shot capture + detect (desktop/cam fallback)
python -m chip_counter --weights path/to/best.pt --save-image out.jpg

# On Raspberry Pi with button+LED (BCM pins)
python -m chip_counter --weights path/to/best.pt --button-pin 17 --led-pin 27
```

---

## Overview

This repo provides a minimal, production-friendly scaffold to run a poker chip counting flow on embedded hardware (e.g. Raspberry Pi) with a CSI camera and GPIO-connected button/LED, while remaining runnable on a laptop for development.

- `chip_counter.hardware` contains button and LED helpers with Raspberry Pi GPIO support and desktop mocks.
- `chip_counter.camera` provides a CSI camera capture helper using Picamera2 when available and falls back to OpenCV.
- `chip_counter.detection` wraps a YOLOv8 detector for multi-class chip count detection.
- `chip_counter.cli` ties it all together: press button → LED on → capture → detect → LED off.

## Installation

Requirements:
- Python 3.10+
- macOS/Linux supported. Raspberry Pi recommended for CSI camera and GPIO.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raspberry Pi extras:
- Picamera2 (libcamera) is typically installed via apt on Pi OS:

```bash
sudo apt update
sudo apt install -y python3-picamera2 libcamera-apps
```

- On Raspberry Pi OS Bookworm, use the rpicam-* tools (e.g., `rpicam-hello`, `rpicam-still`) instead of the older `libcamera-*` names.

- Recommended venv on Pi (so apt-installed Picamera2 is visible):

```bash
# If you just added your user to video/render groups, reboot first
# sudo usermod -aG video,render $USER && sudo reboot

cd /home/pi/chip-counter
deactivate 2>/dev/null || true
rm -rf .venv
/usr/bin/python3 --version  # should print 3.11.x on Pi OS Bookworm
/usr/bin/python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt

# Sanity checks
python -c "from picamera2 import Picamera2; print('Picamera2 OK')"
rpicam-hello -t 2000

# Try a capture
python -m chip_counter.capture --output outputs/test.jpg --no-wait
```

- GPIO libraries (often preinstalled or available via apt):
  - RPi.GPIO or gpiozero. The code dynamically falls back to mock GPIO on non-Pi platforms.

Note: We intentionally do not pin Pi-specific packages in `requirements.txt` to keep desktop installs simple.

## Single-stage multi-class detection (YOLOv8)

We use a single-stage multi-class detector. Each detection box is labeled with a class that represents the count of chips within the box.

- Classes: `count_1, count_2, ..., count_K`.
- The detector both localizes stacks and predicts the count per stack.
- Inference sums the per-box counts to produce a total.

### Dataset layout (YOLO format)

```
datasets/
  chips/
    images/
      train/
        img_001.jpg
        ...
      val/
        img_101.jpg
        ...
    labels/
      train/
        img_001.txt   # YOLO boxes: class cx cy w h (normalized)
      val/
        img_101.txt
```

Each line in a label file is a YOLO-format annotation. The class index corresponds to the integer count class for that box.

Example class mapping for K=20 (1..20) is provided in `configs/data_counts.yaml`.

### Data config (template)

See `configs/data_counts.yaml`. Adjust `path`, `train`, `val`, and the `names` list to match your dataset and K.

### Training (CLI)

We use Ultralytics YOLOv8. With your dataset prepared and a data config ready:

```bash
# Small model to start (change to yolov8s/ m/ l as desired)
yolo detect train \
  data=configs/data_counts.yaml \
  model=yolov8n.pt \
  imgsz=640 \
  epochs=100 \
  project=runs/chips \
  name=yolov8n_counts
```

This trains a model that predicts class `count_i` for each detected stack. Export weights will be in `runs/chips/yolov8n_counts/weights/best.pt`.

### Training (Python API)

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='configs/data_counts.yaml',
    imgsz=640,
    epochs=100,
    project='runs/chips',
    name='yolov8n_counts'
)
```

### Inference

CLI one-shot (desktop or Pi):

```bash
python -m chip_counter --weights runs/chips/yolov8n_counts/weights/best.pt \
  --save-image out.jpg

# Raspberry Pi with button + LED (BCM pin numbers)
python -m chip_counter --weights runs/chips/yolov8n_counts/weights/best.pt \
  --button-pin 17 --led-pin 27 --save-image out.jpg
```

Python API:

```python
from chip_counter.detection.yolo_detector import YOLODetector
from chip_counter.camera.csi import CameraCapture

detector = YOLODetector(weights_path='runs/chips/yolov8n_counts/weights/best.pt')
camera = CameraCapture()

image = camera.capture_image()
total_count, detections = detector.predict_total_count(image)
print('Total chips:', total_count)
for det in detections:
    print(det)
```

### Denomination mapping (sum a monetary value)

If your model’s classes are chip colors or names (e.g., `purple`, `black`, `orange`, `yellow`, `blue`, `red`, `green`), you can sum a monetary total by providing a YAML mapping of `class_name -> value` to the CLI:

```yaml
# configs/denominations.yaml
purple: 1000
black: 100
orange: -2000
yellow: 500
blue: 10
red: 50
green: 20
```

Run the end-to-end CLI and include the mapping:

```bash
python -m chip_counter \
  --weights runs/chips/yolov8n_counts/weights/best.pt \
  --denom-yaml configs/denominations.yaml \
  --save-image out.jpg
```

The CLI will print per-detection class names, the total chip count (if using `count_*` classes), and the `Total value` derived from the YAML mapping.

## Two-stage pipeline (detect → color classify)

When classes are color-dependent, better results often come from a two-stage approach: detect stacks first, then classify color on cropped images.

### 1) Stage 1: Detector (YOLOv8 detect)
- Train a detector to find chip stacks (single class like `stack`) or keep `count_*` if you also need per-stack counts.
- For color-sensitive workflows, disable HSV color jitter during training: `hsv_h=0 hsv_s=0 hsv_v=0`.

### 2) Generate classifier crops from YOLO labels
Use our tool to create a color-classification dataset of crops from your labeled images:

```bash
python -m chip_counter.tools.crops_from_yolo \
  --images datasets/chips/images/train \
  --labels datasets/chips/labels/train \
  --out datasets/colors \
  --names-yaml datasets/chips/dataset.yaml  # uses its names order

# Result: datasets/colors/images/{train,val}/{purple,black,orange,...}/*.jpg
```

### 3) Stage 2: Train a classifier (YOLOv8 classify)
Create a simple classifier data YAML (or pass directories directly):

```bash
yolo classify train \
  data=datasets/colors/images \
  model=yolov8n-cls.pt imgsz=224 epochs=50 batch=32 device=cpu \
  hsv_h=0 hsv_s=0 hsv_v=0

# Best classifier weights at runs/classify/train/weights/best.pt
```

### 4) Run end-to-end

```bash
python -m chip_counter \
  --weights runs/detect/train/weights/best.pt \
  --color-weights runs/classify/train/weights/best.pt \
  --denom-yaml configs/denominations.yaml \
  --conf 0.2 --imgsz 640 \
  --save-overlay outputs/overlay.jpg
```

The overlay includes detector class/conf and the classifier’s color prediction per box. The total monetary value uses the color mapping.

## Pipeline details

The `chip_counter` CLI performs the following steps:
1. Optionally wait for a physical button press (or press Enter on desktop).
2. Turn on an LED to improve exposure/visibility (if configured).
3. Capture an image from the CSI camera (Picamera2) or fall back to OpenCV.
4. Run YOLOv8 detection and sum counts from detected boxes.
5. Turn off the LED and print the total.

## Notes and tips

- Lighting: Consistent illumination and a plain background greatly improve detection quality. Use the LED during capture if possible.
- Classes: Ensure `names` in `configs/data_counts.yaml` exactly matches your intended count classes. E.g., for K=20 use `[count_1, ..., count_20]`.
- Fine-grained classes: More K requires more labeled data per class. Consider starting with a small K.
- Calibration: Fix the camera position and distance to keep stack size consistent.

## License

MIT

## Dataset capture (single-shot)

Use the capture-only CLI to save images for dataset collection. It can save to a directory (auto timestamped filenames) or a specific file path.

```bash
# Save to directory with timestamped name
python -m chip_counter.capture --output datasets/chips/images/train

# Save to a specific file path
python -m chip_counter.capture --output datasets/chips/images/train/img_001.jpg

# Raspberry Pi: wait for physical button and use LED
python -m chip_counter.capture --output datasets/chips/images/train \
  --button-pin 17 --led-pin 27

# Capture immediately without waiting for input/button
python -m chip_counter.capture --output datasets/chips/images/train --no-wait
```

After capturing, create corresponding YOLO label files under `datasets/chips/labels/train/` with the same stem name and `.txt` extension.

## Convert LabelMe JSONs to YOLOv8 labels

If you labeled with LabelMe (e.g., `test.json`, `test2.json`), convert to YOLO labels:

```bash
# Example: convert two files into labels/train
python -m chip_counter.tools.labelme_to_yolo \
  test.json test2.json \
  --out-labels datasets/chips/labels/train \
  --max-k 20

# Or convert an entire directory of JSONs
python -m chip_counter.tools.labelme_to_yolo \
  datasets/chips/labelme-jsons \
  --out-labels datasets/chips/labels/train \
  --max-k 20
```

Notes:
- Supported labels include `count_7`, `7`, or any label ending with digits like `chips_7`.
- Classes are 0-indexed in YOLO: `count_1 -> class 0`, `count_2 -> class 1`, ...
- The converter creates 1 label file per image (same stem) with box lines `class cx cy w h`.

If your labels don't include an explicit count in the text, use mapping/defaults:

```bash
# Map label text to counts and provide a default if parsing fails
python -m chip_counter.tools.labelme_to_yolo \
  test.json test2.json \
  --out-labels datasets/chips/labels/train \
  --map chips_stack=5 --map stack10=10 \
  --default-count 1
```

Troubleshooting empty .txt files:
- Ensure LabelMe shapes are polygons/rectangles with points.
- Ensure `imageWidth`/`imageHeight` exist in JSON or the `imagePath` is valid so the script can infer size. If missing, re-save in LabelMe or keep `imagePath` next to JSON.
- Confirm your labels actually parse to counts (use `--map` or `--default-count`).

### Alternative: Convert with labelme2yolo

You can also use the community `labelme2yolo` converter.

```bash
pip install labelme2yolo

# Convert a LabelMe folder. Use bbox for detection; polygon is for segmentation.
labelme2yolo --json_dir labelme \
             --output_format bbox \
             --val_size 0.1 \
             --seed 42

# The tool creates a YOLO dataset folder inside the json_dir (often labelme/YOLODataset)
ls -l labelme

# Move/copy into this repo's dataset location
mkdir -p datasets/chips
rsync -a labelme/YOLODataset/ datasets/chips/

# Train using the generated dataset.yaml
yolo task=detect mode=train \
  data=datasets/chips/dataset.yaml \
  model=yolov8n.pt imgsz=512 epochs=50 batch=4
```

Notes:
- Ensure your LabelMe labels match your intended classes (e.g., `count_1..count_K`). If needed, edit `datasets/chips/dataset.yaml` `names:` to reflect your classes while preserving the ID order.
- If no validation split was created, either re-run with a larger `--val_size` or manually move some pairs into `images/val` and `labels/val`.
