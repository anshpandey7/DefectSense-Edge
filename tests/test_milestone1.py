"""
Unit and integration tests for Milestone 1:
Scaffolding, configuration, hardware detection, capture sources,
dummy detector, severity scoring, verdict engine, and SQLite logging.
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

from app.config import AppConfig, load_config
from app.hardware import detect_hardware
from app.capture.folder_source import ImageFolderSource, generate_synthetic_plate
from app.vision.backend_base import Detection, DummyDetector, InspectionResult
from app.vision.postprocess import SeverityEngine
from app.storage.database import DatabaseManager


def test_config_loading():
    """Verify configuration loads with default settings."""
    cfg = load_config()
    assert cfg.app.name == "DefectSense Edge"
    assert cfg.source.type == "folder"
    assert cfg.vision.confidence_threshold > 0.0
    assert cfg.storage.retention_days == 30


def test_hardware_detection():
    """Verify hardware profile auto-detects system and selects valid backend."""
    profile = detect_hardware(preferred_backend="auto")
    assert profile.os_name != ""
    assert profile.cpu_count >= 1
    assert profile.selected_backend in ("cpu_onnx", "hailo")
    summary = profile.summary()
    assert "Vision Backend:" in summary


def test_synthetic_plate_generation():
    """Verify synthetic steel plate generator creates valid BGR images."""
    for defect in [None, "crack", "inclusion", "scratches", "pitted_surface", "patches", "crazing"]:
        img = generate_synthetic_plate(defect_type=defect, width=320, height=320)
        assert img is not None
        assert img.shape == (320, 320, 3)
        assert img.dtype == np.uint8
        assert img.mean() > 50  # Metallic grayscale brightness check


def test_folder_source_auto_generation():
    """Verify folder source auto-generates test plates when pointed to an empty dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = ImageFolderSource(tmpdir, fps=30, loop=False, auto_generate_samples=True)
        opened = source.open()
        assert opened is True
        assert len(source.image_paths) >= 6

        success, frame, meta = source.read()
        assert success is True
        assert frame is not None
        assert frame.shape[2] == 3
        assert "filename" in meta
        source.release()


def test_severity_engine_rules():
    """Verify deterministic severity scoring, breakdown, and verdict logic."""
    engine = SeverityEngine()

    # Case 1: Clean part
    verdict, severity, score, breakdown = engine.evaluate([])
    assert verdict == "PASS"
    assert severity == "NONE"
    assert score == 0.0
    assert breakdown["defect_count"] == 0

    # Case 2: Catastrophic crack -> Immediate REJECT & CRITICAL
    crack_det = [
        Detection(
            bbox=[10.0, 10.0, 100.0, 30.0],
            class_name="crack",
            confidence=0.92,
            area_pct=1.5,
            is_edge=True,
        )
    ]
    verdict, severity, score, breakdown = engine.evaluate(crack_det)
    assert verdict == "REJECT"
    assert severity == "CRITICAL"
    assert score >= 0.75
    assert breakdown["edge_defects_count"] >= 1

    # Case 3: Minor scratch in center -> PASS / LOW
    scratch_det = [
        Detection(
            bbox=[300.0, 300.0, 340.0, 305.0],
            class_name="scratches",
            confidence=0.55,
            area_pct=0.1,
            is_edge=False,
        )
    ]
    verdict, severity, score, breakdown = engine.evaluate(scratch_det)
    assert verdict == "PASS"
    assert severity == "LOW"
    assert score < 0.35

    # Case 4: Multiple scratches exceeding review count -> elevated to REVIEW
    multi_scratch = [
        Detection(bbox=[100.0 + i * 50, 200.0, 130.0 + i * 50, 205.0],
                  class_name="scratches", confidence=0.60, area_pct=0.2, is_edge=False)
        for i in range(3)
    ]
    verdict, severity, score, breakdown = engine.evaluate(multi_scratch)
    assert verdict == "REVIEW"
    assert breakdown["triggered_override"] is not None


def test_database_logging_and_csv_export():
    """Verify SQLite logging, querying, trend context, and CSV export."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        img_dir = Path(tmpdir) / "images"
        db = DatabaseManager(db_path=db_path, image_dir=img_dir, save_all_images=True)

        # Create dummy inspection results
        test_frame = np.full((100, 100, 3), 120, dtype=np.uint8)
        result = InspectionResult(
            part_id="TEST-00001",
            timestamp=time.time(),
            frame_index=1,
            detections=[
                Detection(bbox=[10, 10, 40, 40], class_name="inclusion", confidence=0.85, area_pct=1.2, is_edge=False)
            ],
            verdict="REVIEW",
            severity="MEDIUM",
            severity_score=0.48,
            score_breakdown={"area_factor": 0.5, "max_confidence": 0.85},
            inference_ms=12.4,
            backend="dummy_sim",
            annotated_image=test_frame,
        )

        row_id = db.save_inspection(result, frame=test_frame)
        assert row_id > 0

        # Verify recent inspections
        recents = db.get_recent_inspections(10)
        assert len(recents) == 1
        assert recents[0]["part_id"] == "TEST-00001"
        assert recents[0]["verdict"] == "REVIEW"
        assert len(recents[0]["defects"]) == 1

        # Verify trend context
        trend = db.get_trend_context(5)
        assert trend["window_size"] == 1
        assert trend["recent_reviews"] == 1
        assert "inclusion" in trend["top_recent_defect_classes"]

        # Verify CSV export
        csv_file = Path(tmpdir) / "export.csv"
        db.export_csv(csv_file)
        assert csv_file.exists()
        content = csv_file.read_text(encoding="utf-8")
        assert "TEST-00001" in content
        assert "REVIEW" in content
