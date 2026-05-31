# Face Mask Detection — YOLOv8

Real-time face mask detection with YOLOv8. Detects three classes: **with mask**, **without mask**, and **incorrectly worn mask**.

## Classes

| ID | Label |
|----|-------|
| 0 | `mask_weared_incorrect` |
| 1 | `with_mask` |
| 2 | `without_mask` |

## Project Structure

```
├── annotations/          # Pascal VOC XML annotations
├── images/               # Raw dataset images
├── output/               # YOLO-formatted dataset + data.yaml
├── runs/detect/          # Training runs (best weights in train8/weights/best.pt)
├── yolo_mask_detection.py        # Data prep + YOLOv8 training pipeline
├── ann_cnn_transfer_learning.py  # Baseline ANN / CNN / MobileNetV2 experiments
├── real_time_inference.py        # Webcam inference
├── requirements.txt
├── yolov8n.pt            # YOLOv8n base weights
└── yolo11n.pt            # YOLOv11n base weights
```

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, Ultralytics, OpenCV, TensorFlow, W&B

## Usage

### 1. Prepare data and train

Update paths in `yolo_mask_detection.py` to point to your local `annotations/` and `images/` directories, then run:

```bash
python yolo_mask_detection.py
```

This will:
- Parse Pascal VOC XML annotations
- Convert bounding boxes to YOLO format
- Split 80/20 train/val
- Train YOLOv8n for 80 epochs (imgsz=416, batch=16)
- Log metrics to Weights & Biases

### 2. Real-time webcam inference

Update `model_path` in `real_time_inference.py` to point to your trained weights, then run:

```bash
python real_time_inference.py
```

Press `q` to quit.

### 3. Baseline model experiments

```bash
python ann_cnn_transfer_learning.py
```

Trains and compares:
- ANN (basic + improved with BatchNorm)
- CNN (3-block conv)
- MobileNetV2 transfer learning

## Training Config

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n (pretrained) |
| Epochs | 80 |
| Image size | 416 |
| Batch size | 16 |
| Optimizer | Auto |
| IoU threshold | 0.7 |
| Augmentation | Mosaic, RandAugment, HSV, fliplr |

Best weights: `runs/detect/train8/weights/best.pt`

## Dataset

Pascal VOC format XML annotations with three mask-related labels. Images are split 80/20 for training and validation.

## Monitoring

Training metrics logged to [Weights & Biases](https://wandb.ai). Set your own API key in `yolo_mask_detection.py` before running.
