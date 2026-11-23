"""
Unit and integration tests for Milestone 3:
ExecuTorch Llama 3.2 1B advisor, prompt construction, JSON repair engine,
fallback rule tables, asynchronous worker queue, debouncing, and operator Q&A.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
import pytest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.advisor.prompts import (
    build_defect_context,
    format_llama_instruct_prompt,
    parse_and_repair_json,
)
from app.advisor.fallback_rules import get_fallback_advice, FALLBACK_RECOMMENDATIONS
from app.advisor.executorch_llama import ExecuTorchLlamaRunner
from app.advisor.worker import AsyncAdvisorWorker
from app.storage.database import DatabaseManager
from app.vision.backend_base import Detection, InspectionResult
from scripts.export_llama_executorch import export_llama_to_executorch


def test_prompt_formatting_and_context():
    """Verify structured defect context and Llama 3.2 special token wrapping."""
    detections = [{"class_name": "crack", "confidence": 0.94, "bbox": [10, 10, 50, 50]}]
    breakdown = {"total_area_pct": 1.8, "edge_defects_count": 1}
    trend = {"reject_rate_pct": 20.0, "recent_rejects": 2, "top_recent_defect_classes": {"crack": 2}}

    context_json = build_defect_context(
        product_name="HotRolled-Steel",
        line_name="Line-1",
        part_id="PART-00100",
        verdict="REJECT",
        severity="CRITICAL",
        severity_score=0.88,
        detections=detections,
        breakdown=breakdown,
        recent_trend=trend,
    )

    data = json.loads(context_json)
    assert data["primary_defect"] == "crack"
    assert data["verdict"] == "REJECT"
    assert data["defects_near_critical_edge"] is True

    prompt = format_llama_instruct_prompt(context_json)
    assert "<|begin_of_text|>" in prompt
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt


def test_json_parser_and_repair():
    """Verify JSON repair handles markdown wrapping, trailing commas, and python booleans."""
    # Case 1: Wrapped in markdown code fence with python boolean
    raw_1 = """```json
    {
      "summary": "Crack observed on surface.",
      "likely_cause": "Tensile stress.",
      "recommended_action": "Halt line and inspect roll gap.",
      "escalate": True,
    }
    ```"""
    parsed = parse_and_repair_json(raw_1)
    assert parsed is not None
    assert parsed["summary"] == "Crack observed on surface."
    assert parsed["escalate"] is True

    # Case 2: Conversational intro and trailing comma
    raw_2 = """Here is the inspection analysis:
    {
      "summary": "Severe surface pitting.",
      "likely_cause": "Acid pickling attack.",
      "recommended_action": "Check bath concentration.",
      "escalate": false,
    }
    Hope this helps!"""
    parsed_2 = parse_and_repair_json(raw_2)
    assert parsed_2 is not None
    assert parsed_2["escalate"] is False

    # Case 3: Invalid missing required key
    raw_invalid = '{"summary": "Incomplete data"}'
    assert parse_and_repair_json(raw_invalid) is None


def test_fallback_rules():
    """Verify fallback recommendation table covers all industrial defect classes."""
    classes = ["crack", "crazing", "inclusion", "pitted_surface", "rolled_in_scale", "scratches", "patches", "unknown_anomaly"]
    for c in classes:
        advice = get_fallback_advice(c, severity="HIGH")
        assert advice["fallback"] is True
        assert len(advice["summary"]) > 5
        assert len(advice["likely_cause"]) > 5
        assert len(advice["recommended_action"]) > 5
        assert isinstance(advice["escalate"], bool)

    # Escalation on high recent rejects
    adv_esc = get_fallback_advice("scratches", severity="MEDIUM", recent_rejects=4)
    assert adv_esc["escalate"] is True
    assert "WARNING" in adv_esc["summary"]


def test_executorch_runner_and_qa():
    """Verify ExecuTorch runner initialization, advice generation, and operator Q&A."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        pte_file = Path(tmpdir) / "llama3_2_1b.pte"
        tok_file = Path(tmpdir) / "tokenizer.bin"
        export_llama_to_executorch(str(tmpdir), output_pte=str(pte_file), tokenizer_path=str(tok_file))

        runner = ExecuTorchLlamaRunner(pte_model_path=str(pte_file), tokenizer_path=str(tok_file))
        runner.initialize()
        assert runner.model_loaded is True

        # Test advice generation
        context_json = json.dumps({"primary_defect": "crack", "verdict": "REJECT"})
        advice, fallback_used, lat_ms = runner.generate_advice(context_json, primary_defect="crack")
        assert "recommended_action" in advice
        assert advice["escalate"] is True
        assert lat_ms >= 0.0

        # Test Operator Q&A
        ctx = {"verdict": "REJECT", "severity": "CRITICAL", "defects": [{"class_name": "crack"}]}
        q1 = runner.answer_operator_query("Why was this part rejected?", ctx)
        assert "crack" in q1.lower()
        assert "reject" in q1.lower()

        q2 = runner.answer_operator_query("What should I check on the machine?", ctx)
        assert "check" in q2.lower() or "operator" in q2.lower()


def test_async_advisor_worker():
    """Verify asynchronous non-blocking advisor worker with debouncing and DB storage."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_advisor.db"
        db = DatabaseManager(db_path=db_path)

        runner = ExecuTorchLlamaRunner()
        worker = AsyncAdvisorWorker(runner=runner, db=db, debounce_seconds=1.0)
        worker.start()

        # Submit clean part -> should NOT trigger LLM
        res_pass = InspectionResult(
            part_id="PART-P1",
            timestamp=time.time(),
            frame_index=1,
            detections=[],
            verdict="PASS",
            severity="NONE",
            severity_score=0.0,
            score_breakdown={},
            inference_ms=10.0,
            backend="cpu_onnx",
        )
        id_pass = db.save_inspection(res_pass)
        assert worker.submit_inspection(id_pass, res_pass) is False

        # Submit defective part -> should be queued
        res_defect = InspectionResult(
            part_id="PART-D1",
            timestamp=time.time(),
            frame_index=2,
            detections=[Detection(bbox=[10, 10, 50, 50], class_name="crack", confidence=0.92)],
            verdict="REJECT",
            severity="CRITICAL",
            severity_score=0.88,
            score_breakdown={"total_area_pct": 2.0},
            inference_ms=12.0,
            backend="cpu_onnx",
        )
        id_defect = db.save_inspection(res_defect)
        submitted = worker.submit_inspection(id_defect, res_defect)
        assert submitted is True

        # Wait briefly for worker background thread to complete
        time.sleep(0.5)

        # Check that database was updated with advice
        recents = db.get_recent_inspections(limit=5)
        d1_rec = next(r for r in recents if r["part_id"] == "PART-D1")
        assert d1_rec["advice"] is not None
        assert "recommended_action" in d1_rec["advice"]

        worker.stop()
