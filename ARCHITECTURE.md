# Architecture

## Overview

Two parallel ML approaches for 3-class face mask detection:

1. **YOLOv8 object detection** — primary pipeline, produces bounding boxes + class labels on raw frames
2. **Classification baselines** — ANN / CNN / MobileNetV2 on cropped/resized images for comparison

---

## Data Flow

```
annotations/*.xml  (Pascal VOC)
        │
        ▼
yolo_mask_detection.py
  ├─ parse XML → (filename, [(class, xmin, ymin, xmax, ymax)])
  ├─ LabelEncoder: {mask_weared_incorrect→0, with_mask→1, without_mask→2}
  ├─ convert_to_yolo_format: abs bbox → normalized (cx, cy, w, h)
  ├─ 80/20 split (sequential, no shuffle)
  ├─ write output/images/{train,val}/  +  output/labels/{train,val}/
  ├─ write output/data.yaml
  └─ YOLO('yolov8n.pt').train(...)  →  runs/detect/trainN/weights/best.pt

ann_cnn_transfer_learning.py
  ├─ parse XML → image-level label (last object wins)
  ├─ cv2.imread → resize 128×128 → flatten (49152-d) for ANN
  │                              → keep (128,128,3) for CNN/Transfer
  ├─ LabelBinarizer + to_categorical → one-hot
  ├─ 80/20 train_test_split (random_state=42)
  └─ train ANN → improved ANN → CNN → MobileNetV2
```

---

## Model Details

### YOLOv8 (primary)

| Component | Detail |
|-----------|--------|
| Base weights | `yolov8n.pt` (nano, pretrained COCO) |
| Input size | 416 × 416 |
| Epochs | 80 |
| Batch | 16 |
| Classes | 3 |
| Experiment tracking | Weights & Biases |
| Output | `runs/detect/trainN/weights/best.pt` |

### ANN (baseline)

- Input: 49152-d flattened vector (128×128×3)
- Basic: `Flatten → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.5) → Dense(3, softmax)`
- Improved: 4 dense blocks (1024→512→256→128) with BatchNorm + Dropout(0.3), Adam lr=5e-4

### CNN (baseline)

- 3 conv blocks: Conv2D(32) → Conv2D(64) → Conv2D(128), each with BatchNorm + MaxPool + Dropout
- Head: `Dense(128) + BatchNorm + Dropout(0.5) → Dense(3, softmax)`
- Adam lr=1e-4

### MobileNetV2 Transfer (baseline)

- Frozen `MobileNetV2(imagenet, include_top=False, input_shape=(128,128,3))`
- Head: `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(3, softmax)`
- Adam lr=1e-4

All baselines: EarlyStopping(patience=10) + ReduceLROnPlateau(factor=0.5, patience=3)

---

## Inference

```
real_time_inference.py
  cv2.VideoCapture(0)
      │  frame
      ▼
  YOLO(best.pt)(frame)
      │  results[0]
      ▼
  results[0].plot()  →  annotated frame
      │
  cv2.imshow(...)
```

Press `q` to exit loop; releases capture and destroys windows.

---

## File Map

```
yolo_mask_detection.py        data prep + YOLOv8 training
ann_cnn_transfer_learning.py  ANN / CNN / MobileNetV2 baselines
real_time_inference.py        webcam inference
requirements.txt              Python deps
yolov8n.pt                    YOLOv8n base weights
yolo11n.pt                    YOLOv11n base weights (unused)
images/                       raw dataset images (PNG)
annotations/                  Pascal VOC XML (not in repo root, external path)
output/                       generated YOLO dataset + data.yaml (gitignored)
runs/                         Ultralytics training outputs (gitignored)
```
