# Task State

## Completed

- [x] Parse Pascal VOC XML annotations → YOLO format
- [x] 80/20 train/val split with `LabelEncoder`
- [x] `data.yaml` generation for Ultralytics trainer
- [x] YOLOv8n training pipeline (80 epochs, 416px, batch 16)
- [x] W&B logging integration
- [x] Real-time webcam inference (`real_time_inference.py`)
- [x] Baseline ANN (basic + improved with BatchNorm + LR scheduler)
- [x] CNN (3-block conv) training + evaluation
- [x] MobileNetV2 transfer learning baseline
- [x] README with full usage docs and training config table

## Open / Known Issues

- [ ] Hardcoded Windows paths in `yolo_mask_detection.py` (lines 17-19) and `ann_cnn_transfer_learning.py` (lines 44-45) — break on non-Windows machines
- [ ] `real_time_inference.py` hardcodes `runs/detect/train5/weights/best.pt` but README says best weights are in `train8/`
- [ ] W&B API key hardcoded in `yolo_mask_detection.py` line 15 — should be env var
- [ ] `data.yaml` written as single line (line 85 in training script) — invalid YAML for Ultralytics
- [ ] No inference script for static images (only webcam)
- [ ] No model export step (ONNX / TensorRT) for deployment
- [ ] `ann_cnn_transfer_learning.py` has duplicate imports (e.g. `Sequential`, `Adam`, `LabelBinarizer` imported multiple times)
- [ ] YOLOv11 weights (`yolo11n.pt`) present but unused in any script

## Next Priorities

1. Fix cross-platform paths (use `pathlib.Path` or argparse)
2. Move W&B API key to `.env` / environment variable
3. Fix `data.yaml` formatting bug
4. Add `--model-path` CLI arg to `real_time_inference.py`
