# DefectSense Edge: Datasets & Training Guide

This directory contains dataset management, training pipelines, and evaluation scripts for DefectSense Edge.

## 1. Supported Datasets

### A. NEU Surface Defect Database (NEU-DET) - Primary Dataset
- **Description**: Hot-rolled steel strip surface defects collected by Northeastern University (China).
- **Classes (6)**:
  1. `crazing` (network of microcracks)
  2. `inclusion` (non-metallic impurities)
  3. `patches` (irregular oxide/rolling marks)
  4. `pitted_surface` (cavities/pitting corrosion)
  5. `rolled_in_scale` (embedded oxide scale)
  6. `scratches` (surface abrasion)
- **Size**: 1,800 grayscale/RGB images (300 samples per class, 200x200 resolution native, scaled to 640x640).
- **License**: Research and Educational Use. Citation: *K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defect detection," Applied Surface Science, 2013.*

### B. Kolektor Surface-Defect Dataset (KolektorSDD) - Secondary (Crack Detection)
- **Description**: Micro-cracks on electrical commutators from Kolektor Group.
- **Classes**: `crack` / `clean`.
- **License**: Non-commercial research use. Citation: *J. Tabernik et al., "Segmentation-based deep-learning approach for surface-defect detection," J Intell Manuf, 2020.*

### C. MVTec Anomaly Detection (MVTec AD) - Unsupervised Anomaly Path
- **Description**: High-resolution industrial defect dataset with 15 categories (e.g. `metal_nut`, `screw`, `grid`).
- **Mode**: Trained on normal (good) samples only; generates anomaly heatmaps and segmentations for unknown defects.
- **License**: CC BY-NC-SA 4.0. Citation: *P. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.*

---

## 2. Preparing Datasets & Training

Run the training pipeline:
```bash
# Generate / prepare dataset and train YOLOv8n
python training/train_neu_yolo.py --epochs 50 --img-size 640 --batch-size 16

# Evaluate mAP@0.5, Precision, Recall, Confusion Matrix
python training/evaluate_yolo.py --model models/neu_yolov8n.onnx

# Export & quantize to INT8 for Raspberry Pi 5
python training/export_onnx.py --output models/neu_yolov8n.onnx --quantize
```
