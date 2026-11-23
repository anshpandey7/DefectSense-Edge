"""
Asynchronous Advisor Worker with priority queue, defect event debouncing,
timeout guardrails, and non-blocking video loop integration.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from app.advisor.executorch_llama import ExecuTorchLlamaRunner
from app.advisor.fallback_rules import get_fallback_advice
from app.advisor.prompts import build_defect_context
from app.storage.database import DatabaseManager
from app.vision.backend_base import InspectionResult

logger = logging.getLogger("advisor_worker")


@dataclass
class AdvisorTask:
    inspection_id: int
    result: InspectionResult
    trend_context: Dict[str, Any]
    line_name: str
    product_name: str
    timestamp: float


class AsyncAdvisorWorker:
    """
    Background worker thread running ExecuTorch Llama advisory generation.
    Completely decoupled from the main video loop to maintain >= 10 FPS real-time inspection.
    """

    def __init__(
        self,
        runner: ExecuTorchLlamaRunner,
        db: DatabaseManager,
        debounce_seconds: float = 3.0,
        timeout_seconds: float = 8.0,
        line_name: str = "Line-1",
        product_name: str = "HotRolled-Steel",
    ):
        self.runner = runner
        self.db = db
        self.debounce_seconds = debounce_seconds
        self.timeout_seconds = timeout_seconds
        self.line_name = line_name
        self.product_name = product_name

        self._queue: queue.Queue[AdvisorTask] = queue.Queue(maxsize=10)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_defect_time = 0.0
        self._callbacks: list[Callable[[int, Dict[str, Any], bool], None]] = []

    def register_callback(self, callback: Callable[[int, Dict[str, Any], bool], None]) -> None:
        """Register listener for new completed advice (e.g., UI WebSocket, GPIO alert)."""
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.runner.initialize()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="AdvisorWorker")
        self._thread.start()
        logger.info("Async Advisor worker started in background thread.")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("Async Advisor worker stopped.")

    def submit_inspection(self, inspection_id: int, result: InspectionResult) -> bool:
        """
        Submit inspection to advisor.
        Only triggers on defects (REVIEW / REJECT), with debouncing to prevent queue saturation.
        """
        if result.verdict == "PASS":
            return False  # Do not trigger LLM for clean passing parts

        now = time.time()
        # Debounce check
        if now - self._last_defect_time < self.debounce_seconds:
            logger.debug(f"Defect on part {result.part_id} debounced (interval < {self.debounce_seconds}s)")
            # Still provide immediate rule-based fallback without burdening the queue
            fallback_adv = get_fallback_advice(
                result.detections[0].class_name if result.detections else "unknown",
                result.severity,
            )
            self.db.update_advice(inspection_id, fallback_adv, fallback_used=True)
            return False

        self._last_defect_time = now
        trend = self.db.get_trend_context(n_recent=10)

        task = AdvisorTask(
            inspection_id=inspection_id,
            result=result,
            trend_context=trend,
            line_name=self.line_name,
            product_name=self.product_name,
            timestamp=now,
        )

        try:
            self._queue.put_nowait(task)
            logger.info(f"Dispatched advisory task for {result.part_id} (Verdict: {result.verdict}) to background queue.")
            return True
        except queue.Full:
            logger.warning("Advisor queue full; generating immediate fallback advice.")
            fallback_adv = get_fallback_advice(
                result.detections[0].class_name if result.detections else "unknown",
                result.severity,
            )
            self.db.update_advice(inspection_id, fallback_adv, fallback_used=True)
            return False

    def _worker_loop(self) -> None:
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._process_task(task)
            except Exception as e:
                logger.error(f"Error processing advisor task for part {task.result.part_id}: {e}")
                # Emergency fallback
                primary = task.result.detections[0].class_name if task.result.detections else "unknown"
                adv = get_fallback_advice(primary, task.result.severity)
                self.db.update_advice(task.inspection_id, adv, fallback_used=True)
            finally:
                self._queue.task_done()

    def _process_task(self, task: AdvisorTask) -> None:
        res = task.result
        detections_dicts = [d.to_dict() for d in res.detections]
        primary_defect = res.detections[0].class_name if res.detections else "unknown"

        context_json = build_defect_context(
            product_name=task.product_name,
            line_name=task.line_name,
            part_id=res.part_id,
            verdict=res.verdict,
            severity=res.severity,
            severity_score=res.severity_score,
            detections=detections_dicts,
            breakdown=res.score_breakdown,
            recent_trend=task.trend_context,
        )

        recent_rejects = task.trend_context.get("recent_rejects", 0)

        # Run inference through runner
        advice, fallback_used, lat_ms = self.runner.generate_advice(
            defect_context_json=context_json,
            primary_defect=primary_defect,
            severity=res.severity,
            recent_rejects=recent_rejects,
            timeout_seconds=self.timeout_seconds,
        )

        # Update database with resulting advice
        self.db.update_advice(task.inspection_id, advice, fallback_used=fallback_used)
        res.advice = advice

        logger.info(
            f"[Advisor Complete] Part: {res.part_id} | Fallback: {fallback_used} | Latency: {lat_ms:.1f}ms | "
            f"Summary: {advice.get('summary', '')[:60]}... | Escalate: {advice.get('escalate', False)}"
        )

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(task.inspection_id, advice, fallback_used)
            except Exception as e:
                logger.error(f"Error in advisor callback: {e}")
