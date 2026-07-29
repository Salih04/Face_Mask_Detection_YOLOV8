# Project Context

## Goal

Train and deploy a real-time face mask detector that classifies each detected face into one of three states:

| Class | Meaning |
|-------|---------|
| `with_mask` | Mask worn correctly |
| `without_mask` | No mask |
| `mask_weared_incorrect` | Mask worn improperly (chin, nose exposed, etc.) |

Primary deliverable: a YOLOv8 model that runs on a webcam feed in real time.

## Motivation

COVID-era mask compliance monitoring. Detecting *incorrect* wearing (not just presence/absence) is the distinguishing requirement — most public datasets only have binary mask/no-mask labels.

## Dataset

- Format: Pascal VOC XML bounding box annotations
- Images: PNG files in `images/` (~850+ samples based on filenames observed)
- Labels: 3 classes above
- Naming convention: `maksssksksss{N}.png` / `.xml`
- Split: 80% train / 20% val (sequential split in training script — no shuffle)

## Approach Rationale

YOLOv8 chosen as primary because:
- Single-stage detector → real-time capable on CPU/GPU
- Pretrained COCO weights → fast convergence on small dataset
- Ultralytics API handles augmentation (mosaic, HSV, fliplr) without manual config

ANN / CNN / MobileNetV2 baselines in `ann_cnn_transfer_learning.py` serve as academic comparison — demonstrate why object detection outperforms image-level classification for this task.

## Experiment Tracking

The current training script does not initialize W&B. If experiment tracking is
added later, credentials must be supplied through secure environment
configuration and must never be committed.

## Environment

Originally developed on Windows (`C:\Users\salih\Desktop\...`). Path handling needs update for cross-platform use.

## Current State (as of 2026-05-31)

- Training pipeline complete and functional (with path fix)
- Best weights from `runs/detect/train8/weights/best.pt` (per README)
- Real-time webcam inference works; model path arg still hardcoded to `train5`
- No deployment packaging or export to ONNX/TensorRT yet
