#!/usr/bin/env bash
# DefectSense Edge: Offline Model Setup Script for Raspberry Pi 5 & Linux
set -euo pipefail

echo "=========================================================="
echo "DefectSense Edge: Preparing On-Device Model Artifacts"
echo "=========================================================="

mkdir -p models
mkdir -p data/datasets

# 1. Generate/Export NEU-DET Surface Defect ONNX Model
echo "[1/3] Generating NEU-DET YOLOv8n ONNX model..."
python3 training/export_onnx.py --output models/neu_yolov8n.onnx --quantize

# 2. Export Llama 3.2 1B to ExecuTorch .pte
echo "[2/3] Setting up ExecuTorch Llama 3.2 1B Instruct (.pte)..."
python3 scripts/export_llama_executorch.py \
    --output-pte models/llama3_2_1b_instruct.pte \
    --tokenizer models/tokenizer.bin \
    --quantization 8da4w

# 3. Create Sample Steel Plates for Immediate Offline Demo
echo "[3/3] Generating factory steel inspection sample plates..."
python3 -c "
from app.capture.folder_source import ImageFolderSource
source = ImageFolderSource('data/sample_parts', auto_generate_samples=True)
source.open()
print(f'Generated {len(source.image_paths)} sample plates in data/sample_parts')
"

echo "=========================================================="
echo "All on-device artifacts ready in models/ and data/"
echo "Ready to run fully offline: python3 app/main.py"
echo "=========================================================="
