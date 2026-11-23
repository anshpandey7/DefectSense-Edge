"""
Unit and integration tests for Milestone 4:
FastAPI operator web portal, video stream, REST endpoints,
operator Q&A, and GPIO/buzzer alert controller.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.alerts.buzzer_gpio import GPIOAlertController
from app.advisor.executorch_llama import ExecuTorchLlamaRunner
from app.config import AppConfig, load_config
from app.hardware import detect_hardware
from app.storage.database import DatabaseManager
from app.vision.backend_base import Detection, InspectionResult
from app.web.server import AppState, create_app


def test_gpio_alert_controller():
    """Verify GPIO controller handles PASS/REJECT triggers without throwing in simulated mode."""
    controller = GPIOAlertController(enable_gpio=False, enable_buzzer=False)
    # Should not throw
    controller.trigger_verdict_alert("REJECT", "CRITICAL")
    controller.trigger_verdict_alert("PASS", "NONE")
    controller.cleanup()


def test_web_portal_endpoints():
    """Verify FastAPI routes: dashboard HTML, status, stats, recent list, CSV export, and Q&A."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_web.db"
        img_dir = Path(tmpdir) / "images"
        db = DatabaseManager(db_path=db_path, image_dir=img_dir)

        # Seed sample inspection
        res = InspectionResult(
            part_id="PART-WEB-001",
            timestamp=time.time(),
            frame_index=1,
            detections=[Detection(bbox=[20, 20, 60, 60], class_name="crack", confidence=0.91)],
            verdict="REJECT",
            severity="CRITICAL",
            severity_score=0.85,
            score_breakdown={"area_factor": 0.5},
            inference_ms=15.2,
            backend="cpu_onnx",
        )
        db.save_inspection(res)

        config = load_config()
        hw = detect_hardware()
        runner = ExecuTorchLlamaRunner()
        runner.initialize()

        state = AppState(config=config, hw_profile=hw, db=db, advisor_runner=runner)
        state.update_frame(np.zeros((100, 100, 3), dtype=np.uint8), res, fps=15.0)

        app = create_app(state)
        client = TestClient(app)

        # 1. Dashboard HTML
        r_home = client.get("/")
        assert r_home.status_code == 200
        assert "DefectSense" in r_home.text

        # 2. Status API
        r_status = client.get("/api/status")
        assert r_status.status_code == 200
        status_data = r_status.json()
        assert status_data["app_name"] == "DefectSense Edge"
        assert status_data["fps"] == 15.0

        # 3. Recent inspections
        r_recent = client.get("/api/inspections/recent")
        assert r_recent.status_code == 200
        recent_data = r_recent.json()
        assert len(recent_data) >= 1
        assert recent_data[0]["part_id"] == "PART-WEB-001"

        # 4. Statistics API
        r_stats = client.get("/api/statistics")
        assert r_stats.status_code == 200
        stats_data = r_stats.json()
        assert stats_data["total_inspections"] >= 1
        assert "crack" in stats_data["defect_classes"]

        # 5. CSV export
        r_csv = client.get("/api/export/csv")
        assert r_csv.status_code == 200
        assert "PART-WEB-001" in r_csv.text

        # 6. Operator Q&A
        r_qa = client.post("/api/operator/query", json={"question": "Why was this part rejected?"})
        assert r_qa.status_code == 200
        qa_data = r_qa.json()
        assert "answer" in qa_data
        assert "crack" in qa_data["answer"].lower() or "reject" in qa_data["answer"].lower()

        # 7. Settings update
        r_sett = client.post("/api/settings", json={"confidence_threshold": 0.42, "enable_buzzer": True})
        assert r_sett.status_code == 200
        assert r_sett.json()["status"] == "ok"
