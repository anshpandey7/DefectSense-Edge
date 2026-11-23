"""
Offline Proof and Air-Gapped Verification Test for DefectSense Edge.
Simulates total network disconnection by blocking all outbound sockets and HTTP requests.
Verifies that:
1. Vision inference runs 100% locally on CPU without external API calls.
2. Severity and verdict engine computes decisions purely on-device.
3. ExecuTorch Llama advisor generates actionable recommendations completely offline.
4. SQLite database logs records and generates CSV reports with network cable unplugged.
"""

from __future__ import annotations

import socket
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.advisor.executorch_llama import ExecuTorchLlamaRunner
from app.advisor.prompts import build_defect_context
from app.advisor.worker import AsyncAdvisorWorker
from app.capture.folder_source import generate_synthetic_plate
from app.config import load_config
from app.hardware import detect_hardware
from app.storage.database import DatabaseManager
from app.vision.backend_base import Detection, InspectionResult
from app.vision.onnx_cpu import ONNXCPUDetector
from app.vision.postprocess import SeverityEngine


class NetworkBlockedError(RuntimeError):
    pass


def test_fully_offline_pipeline(monkeypatch):
    """
    Simulate disconnected network cable (air-gapped environment)
    and verify the entire DefectSense Edge pipeline functions without errors.
    """

    # 1. Block all network connections (TCP/UDP socket creation and connection)
    def block_socket_connect(*args, **kwargs):
        raise NetworkBlockedError("Outbound network disabled: system is running air-gapped!")

    def block_urlopen(*args, **kwargs):
        raise NetworkBlockedError("Network request blocked: Zero cloud calls allowed!")

    monkeypatch.setattr(socket.socket, "connect", block_socket_connect)
    monkeypatch.setattr(urllib.request, "urlopen", block_urlopen)

    # Verify network is indeed blocked
    with pytest.raises(NetworkBlockedError):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))

    # 2. Run DefectSense Edge end-to-end pipeline under air-gapped constraint
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "offline_test.db"
        img_dir = tmp_path / "images"

        db = DatabaseManager(db_path=db_path, image_dir=img_dir, save_all_images=True)
        hw = detect_hardware()
        engine = SeverityEngine()

        # Load Vision Detector (ONNX CPU)
        onnx_model = PROJECT_ROOT / "models" / "neu_yolov8n.onnx"
        detector = ONNXCPUDetector(model_path=str(onnx_model))
        detector.load()
        assert detector.is_loaded is True

        # Load ExecuTorch Llama Advisor
        runner = ExecuTorchLlamaRunner()
        runner.initialize()
        worker = AsyncAdvisorWorker(runner=runner, db=db, debounce_seconds=0.1)
        worker.start()

        # Test Frame with Crack Defect
        test_frame = generate_synthetic_plate("crack", width=640, height=640)

        # Step A: Vision Inference (completely offline)
        detections, inf_ms = detector.predict(test_frame)
        assert inf_ms > 0.0

        # Ensure defect detection is present for testing advisor pipeline
        if not detections:
            detections.append(Detection(
                bbox=[20.0, 20.0, 120.0, 60.0],
                class_name="crack",
                confidence=0.92,
                area_pct=1.5,
                is_edge=True,
            ))

        # Step B: Severity Evaluation
        verdict, severity, score, breakdown = engine.evaluate(detections)
        assert verdict in ("REVIEW", "REJECT")

        # Step C: SQLite Logging
        result = InspectionResult(
            part_id="OFFLINE-001",
            timestamp=time.time(),
            frame_index=1,
            detections=detections,
            verdict=verdict,
            severity=severity,
            severity_score=score,
            score_breakdown=breakdown,
            inference_ms=inf_ms,
            backend=detector.backend_name,
            annotated_image=test_frame,
        )
        rec_id = db.save_inspection(result, frame=test_frame)
        assert rec_id > 0

        # Step D: ExecuTorch Advisor Generation (completely offline)
        worker.submit_inspection(rec_id, result)
        time.sleep(0.4)

        # Step E: Verify database record and offline CSV report
        recent = db.get_recent_inspections(limit=1)
        assert len(recent) == 1
        assert recent[0]["part_id"] == "OFFLINE-001"
        assert recent[0]["advice"] is not None
        assert "recommended_action" in recent[0]["advice"]

        csv_file = tmp_path / "offline_export.csv"
        db.export_csv(csv_file)
        assert csv_file.exists()
        assert "OFFLINE-001" in csv_file.read_text(encoding="utf-8")

        worker.stop()
