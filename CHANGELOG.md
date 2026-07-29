# Changelog

## [Unreleased]

### Security
- Removed an unknown hard-coded W&B credential without testing or using it.
- Removed the unused W&B import and direct dependency.
- Added credential-handling guidance in `SECURITY.md`.

### Known Fixes Needed
- Cross-platform paths (hardcoded Windows `C:\Users\salih\...`)
- `data.yaml` written as single line (invalid YAML)
- `real_time_inference.py` model path points to `train5`, not `train8`

---

## 2026-05-31 — `704df02`

### Changed
- `real_time_inference.py`: updated webcam inference script
- `README.md`: full rewrite with project structure, class table, training config table, usage instructions for all three scripts, W&B monitoring note

---

## 2025-01-05 — `ac9176e`

### Changed
- `README.md`: minor update (1 line change)

---

## 2025-01-04 — `f188541`

### Added
- Initial commit
- `yolo_mask_detection.py`: Pascal VOC XML parser, YOLO format converter, 80/20 split, `data.yaml` generator, YOLOv8n training loop, W&B integration, validation set visualization
- `ann_cnn_transfer_learning.py`: ANN baseline, improved ANN (BatchNorm + LR scheduler), CNN (3-block), MobileNetV2 transfer learning
- `real_time_inference.py`: webcam inference loop
- `requirements.txt`: numpy, pandas, matplotlib, opencv-python, scikit-learn, torch, tensorflow, wandb, ultralytics
- `yolov8n.pt`: YOLOv8n pretrained base weights
- `yolo11n.pt`: YOLOv11n pretrained base weights
- `images/`: dataset images (PNG)
