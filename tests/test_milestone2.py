"""
Unit and integration tests for Milestone 2:
ONNX export, ONNX Runtime CPU detector, letterbox preprocessing,
inference latency, NMS, area calculation, and evaluation pipeline.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pytest

# Add project root to sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.onnx_cpu import ONNXCPUDetector, letterbox
from app.vision.backend_base import Detection
from training.export_onnx import create_reference_yolo_onnx, quantize_onnx_model
from training.dataset_manager import generate_synthetic_neu_yolo_dataset
from training.evaluate_yolo import generate_evaluation_report


def test_letterbox_preprocessing():
    """Verify letterbox maintains aspect ratio and pads to 640x640."""
    # Test non-square input (800x600)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    padded, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
    assert padded.shape == (640, 640, 3)
    assert ratio == 640.0 / 800.0  # scaled by width
    assert dw == 0.0
    assert dh == (640 - (600 * ratio)) / 2.0


def test_onnx_export_and_validation():
    """Verify reference ONNX model generation and graph validity."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        onnx_file = Path(tmpdir) / "test_neu.onnx"
        create_reference_yolo_onnx(onnx_file)
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 1000

        # Verify with ONNX checker
        import onnx
        model = onnx.load(str(onnx_file))
        onnx.checker.check_model(model)
        assert len(model.graph.input) == 1
        assert model.graph.input[0].name == "images"
        assert len(model.graph.output) == 1
        assert model.graph.output[0].name == "output0"


def test_onnx_cpu_inference():
    """Verify ONNX Runtime CPU inference executes and returns valid detections."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        onnx_file = Path(tmpdir) / "test_neu.onnx"
        create_reference_yolo_onnx(onnx_file)

        detector = ONNXCPUDetector(
            model_path=str(onnx_file),
            confidence_threshold=0.01,  # Low threshold to capture any head output
            iou_threshold=0.45,
            num_threads=2,
        )
        detector.load()
        assert detector.is_loaded is True
        assert detector.backend_name == "cpu_onnx"

        # Test frame
        frame = np.full((640, 640, 3), 120, dtype=np.uint8)
        # Draw a simulated crack
        cv2.line(frame, (50, 50), (200, 70), (20, 20, 25), 3)

        detections, inf_ms = detector.predict(frame)
        assert inf_ms > 0.0
        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert len(det.bbox) == 4
            assert det.area_pct >= 0.0
            assert isinstance(det.is_edge, bool)


def test_evaluation_metrics_generation():
    """Verify evaluation report generation."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        eval_dir = Path(tmpdir) / "eval"
        metrics = generate_evaluation_report("models/neu_yolov8n.onnx", "data/data.yaml", eval_dir)
        assert metrics["overall"]["mAP_50"] > 0.80
        assert "crack" in metrics["per_class"]
        assert (eval_dir / "evaluation_metrics.json").exists()
