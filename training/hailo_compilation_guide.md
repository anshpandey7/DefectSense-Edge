# Hailo-8 / Hailo-8L (.hef) Compilation Guide for DefectSense Edge

This guide details compiling the NEU-DET defect detection model to a Hailo Executable Format (`.hef`) file for hardware acceleration on the **Raspberry Pi AI HAT+** (Hailo-8 / Hailo-8L NPU) delivering **>= 25 FPS**.

---

## 1. Prerequisites (on x86_64 host / development PC)
Hailo Dataflow Compiler (DFC) requires Ubuntu 22.04 LTS (x86_64).

```bash
# Install Hailo Dataflow Compiler
pip install hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl
```

---

## 2. Model Translation (ONNX to Hailo HAR)
Translate the exported `neu_yolov8n.onnx` to Hailo Archive format (`.har`):

```bash
hailo parser onnx models/neu_yolov8n.onnx \
    --hw-arch hailo8l \
    --start-node-names "images" \
    --end-node-names "/model.22/cv2.0/cv2.0.2/Conv" "/model.22/cv3.0/cv3.0.2/Conv" \
    --har-path models/neu_yolov8n.har
```

---

## 3. Model Optimization & Quantization (Full INT8)
Create a quantization calibration dataset of representative factory steel plate images (at least 128 images from `data/datasets/neu_det/images/val`):

```python
# scripts/calibrate_hailo.py
import numpy as np
import glob, cv2

calib_images = []
for p in glob.glob("data/datasets/neu_det/images/val/*.jpg")[:128]:
    im = cv2.imread(p)
    im = cv2.resize(im, (640, 640))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    calib_images.append(im)

np.save("data/calib_images.npy", np.array(calib_images, dtype=np.uint8))
```

Run Hailo optimization script:
```bash
hailo optimize models/neu_yolov8n.har \
    --hw-arch hailo8l \
    --calib-data-path data/calib_images.npy \
    --har-path models/neu_yolov8n_quantized.har
```

---

## 4. Compile to `.hef` Binary
Compile the quantized `.har` for the Raspberry Pi AI HAT+ (Hailo-8L):

```bash
hailo compiler models/neu_yolov8n_quantized.har \
    --hw-arch hailo8l \
    --output-path models/neu_yolov8n.hef
```

---

## 5. Deployment on Raspberry Pi 5
1. Install Hailo drivers and runtime on Raspberry Pi OS Bookworm:
   ```bash
   sudo apt install hailo-all
   ```
2. Check that the Hailo NPU is detected:
   ```bash
   hailortcli fw-control identify
   ```
3. Copy `models/neu_yolov8n.hef` to the Pi 5.
4. Run DefectSense Edge with Hailo backend:
   ```bash
   python app/main.py --backend hailo
   ```
   DefectSense Edge automatically offloads the vision pipeline to the Hailo NPU at **>= 25 FPS**, leaving all 4 Arm Cortex-A76 cores free for the ExecuTorch Llama advisor.
