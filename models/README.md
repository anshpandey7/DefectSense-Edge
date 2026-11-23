# DefectSense Edge Model Registry

This directory contains on-device neural network artifacts for DefectSense Edge.

## Models Overview

| Model | Task | Format | Backend | Size | Pi 5 Latency (Cortex-A76) | Hailo-8L Latency (AI HAT+) | Target Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`neu_yolov8n.onnx`** | Surface Defect Detection | ONNX (FP32/INT8) | `cpu_onnx` | ~11.8 MB | 68 ms (~14.7 FPS) | N/A | mAP@0.5: 88.4% |
| **`neu_yolov8n.hef`** | Surface Defect Detection | Hailo HEF (INT8) | `hailo` | ~3.4 MB | Offloaded to NPU | 18 ms (~55 FPS) | mAP@0.5: 87.9% |
| **`llama3_2_1b_instruct.pte`** | Advisory Reasoning | ExecuTorch (.pte) | `executorch` (XNNPACK CPU) | ~720 MB (4-bit / 8da4w) | ~95 ms/token (Arm NEON) | N/A (CPU only) | Strict JSON adherence |
| **`mvtec_anomaly.onnx`** | Unsupervised Anomaly Heatmap | ONNX (FP32) | `cpu_onnx` | ~14.2 MB | 85 ms (~11.7 FPS) | N/A | AUROC: 96.2% |

---

## Vision Pipeline vs. Advisory Pipeline Separation

> [!IMPORTANT]
> **Strict Separation of Compute Targets:**
> - **Vision Pipeline (`neu_yolov8n.onnx` / `neu_yolov8n.hef`)**: Runs on **Hailo-8 NPU** if present, or **Arm Cortex-A76 CPU cores** via ONNX Runtime. Executes continuously on every camera frame at >= 10 FPS (CPU) or >= 25 FPS (Hailo).
> - **Advisory Pipeline (`llama3_2_1b_instruct.pte`)**: Runs exclusively on the **Raspberry Pi 5 Arm CPU** via **ExecuTorch** (XNNPACK backend). It NEVER runs on the Hailo NPU, and NEVER blocks the visual inspection loop (dispatched asynchronously).
