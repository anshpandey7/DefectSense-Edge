"""
Unit and integration tests for Milestone 5:
Hailo NPU detector backend, unsupervised anomaly detection (MVTec AD style),
and benchmark telemetry suite.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.hailo_backend import HailoDetector
from app.vision.anomaly import AnomalyDetector
from app.capture.folder_source import generate_synthetic_plate
from scripts.benchmark import get_cpu_temperature, benchmark_vision_backend, benchmark_llama, run_full_benchmark


def test_hailo_detector_backend():
    """Verify Hailo detector initializes and returns high-throughput predictions."""
    detector = HailoDetector()
    detector.load()
    assert detector.is_loaded is True
    assert detector.backend_name == "hailo"

    frame = np.full((640, 640, 3), 120, dtype=np.uint8)
    dets, ms = detector.predict(frame)
    assert ms > 0.0
    assert isinstance(dets, list)


def test_unsupervised_anomaly_detector():
    """Verify unsupervised anomaly detector on normal vs anomalous samples."""
    detector = AnomalyDetector(anomaly_threshold=0.60)

    # 1. Calibrate on normal plates
    normal_samples = [generate_synthetic_plate(None) for _ in range(3)]
    detector.fit_normal_samples(normal_samples)

    # 2. Test on clean plate
    clean_plate = generate_synthetic_plate(None)
    score_clean, overlay_clean, det_clean = detector.evaluate_frame(clean_plate)
    assert 0.0 <= score_clean <= 1.0
    assert overlay_clean.shape == clean_plate.shape

    # 3. Test on anomalous plate (plate with jagged fracture crack)
    crack_plate = generate_synthetic_plate("crack")
    score_crack, overlay_crack, det_crack = detector.evaluate_frame(crack_plate)
    assert score_crack >= score_clean
    assert overlay_crack.shape == crack_plate.shape
    if det_crack:
        assert det_crack.class_name == "unknown_anomaly"


def test_benchmark_suite_execution():
    """Verify benchmark suite produces accurate metrics and saves report."""
    temp_c = get_cpu_temperature()
    assert 10.0 <= temp_c <= 110.0

    report = run_full_benchmark(num_frames=5)
    assert "hardware" in report
    assert "vision_benchmarks" in report
    assert "advisor_benchmarks" in report
    assert report["vision_benchmarks"]["cpu_onnx"]["fps"] > 0.0
    assert report["vision_benchmarks"]["hailo_npu"]["fps"] >= 25.0
    assert (PROJECT_ROOT / "docs" / "benchmark_report.json").exists()
