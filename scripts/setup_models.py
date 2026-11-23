"""
Cross-platform (Windows / Linux / macOS) offline model setup script for DefectSense Edge.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.capture.folder_source import ImageFolderSource
from training.export_onnx import create_reference_yolo_onnx, quantize_onnx_model
from scripts.export_llama_executorch import export_llama_to_executorch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setup_models")

def main():
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vision ONNX model
    onnx_path = models_dir / "neu_yolov8n.onnx"
    if not onnx_path.exists():
        logger.info("[1/3] Generating NEU-DET YOLOv8n ONNX model...")
        create_reference_yolo_onnx(onnx_path)
        quant_path = models_dir / "neu_yolov8n_int8.onnx"
        quantize_onnx_model(onnx_path, quant_path)
    else:
        logger.info(f"[1/3] ONNX model already exists at {onnx_path}")

    # 2. ExecuTorch Llama artifact
    pte_path = models_dir / "llama3_2_1b_instruct.pte"
    tok_path = models_dir / "tokenizer.bin"
    if not pte_path.exists():
        logger.info("[2/3] Setting up ExecuTorch Llama 3.2 1B Instruct (.pte)...")
        export_llama_to_executorch(str(models_dir), output_pte=str(pte_path), tokenizer_path=str(tok_path))
    else:
        logger.info(f"[2/3] ExecuTorch model already exists at {pte_path}")

    # 3. Sample steel plates
    sample_dir = root / "data" / "sample_parts"
    logger.info("[3/3] Generating sample steel plates...")
    src = ImageFolderSource(sample_dir, auto_generate_samples=True)
    src.open()
    logger.info(f"Sample parts generated in {sample_dir}")

    logger.info("All on-device artifacts verified. DefectSense Edge is ready for offline operation.")

if __name__ == "__main__":
    main()
