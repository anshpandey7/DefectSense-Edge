"""
FastAPI Operator Web Service for DefectSense Edge.
Delivers real-time MJPEG visual streaming, WebSocket alert broadcast,
historical analysis, CSV exports, settings management, and operator Q&A.
Zero cloud dependence: 100% locally hosted.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.advisor.executorch_llama import ExecuTorchLlamaRunner
from app.config import AppConfig
from app.hardware import HardwareProfile
from app.storage.database import DatabaseManager
from app.vision.backend_base import InspectionResult

logger = logging.getLogger("web.server")

STATIC_DIR = Path(__file__).parent / "static"


class OperatorQuestion(BaseModel):
    question: str


class SettingsUpdate(BaseModel):
    confidence_threshold: Optional[float] = None
    enable_buzzer: Optional[bool] = None
    enable_gpio: Optional[bool] = None


class AppState:
    """Shared application state accessible across web routes and the inspection thread."""

    def __init__(
        self,
        config: AppConfig,
        hw_profile: HardwareProfile,
        db: DatabaseManager,
        advisor_runner: ExecuTorchLlamaRunner,
    ):
        self.config = config
        self.hw_profile = hw_profile
        self.db = db
        self.advisor_runner = advisor_runner

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_result: Optional[InspectionResult] = None
        self.latest_advice: Optional[Dict[str, Any]] = None
        self.current_fps: float = 0.0
        self.active_websockets: Set[WebSocket] = set()
        self.lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None

    def update_frame(self, frame: np.ndarray, result: InspectionResult, fps: float) -> None:
        self.latest_frame = frame
        self.latest_result = result
        self.current_fps = fps

    def update_advice(self, advice: Dict[str, Any]) -> None:
        self.latest_advice = advice


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="DefectSense Edge Operator Portal", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def get_index() -> HTMLResponse:
        html_file = STATIC_DIR / "index.html"
        if not html_file.exists():
            return HTMLResponse("<h1>DefectSense Edge Portal</h1><p>UI loading...</p>")
        return HTMLResponse(html_file.read_text(encoding="utf-8"))

    @app.get("/video_feed")
    def video_feed() -> StreamingResponse:
        """Multipart MJPEG video stream for low-latency live camera view."""

        def frame_generator():
            while True:
                if state.latest_frame is not None:
                    # Encode frame to JPEG
                    ret, jpeg = cv2.imencode(".jpg", state.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = jpeg.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                time_sleep = 1.0 / max(5.0, state.config.source.fps)
                import time
                time.sleep(time_sleep)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        """System health and operational telemetry."""
        return {
            "app_name": state.config.app.name,
            "line_name": state.config.app.line_name,
            "product_name": state.config.app.product_name,
            "fps": round(state.current_fps, 1),
            "backend": state.latest_result.backend if state.latest_result else state.hw_profile.selected_backend,
            "hardware": {
                "os": state.hw_profile.os_name,
                "machine": state.hw_profile.machine,
                "cpu_cores": state.hw_profile.cpu_count,
                "is_raspberry_pi": state.hw_profile.is_raspberry_pi,
                "pi_model": state.hw_profile.pi_model,
                "hailo_detected": state.hw_profile.hailo_detected,
                "ram_gb": round(state.hw_profile.ram_gb, 1),
            },
            "advisor": {
                "runtime": state.advisor_runner.runtime_type,
                "model_loaded": state.advisor_runner.model_loaded,
                "pte_path": str(state.advisor_runner.pte_model_path),
            },
            "latest_inspection": state.latest_result.to_dict() if state.latest_result else None,
            "latest_advice": state.latest_advice,
        }

    @app.get("/api/inspections/recent")
    def get_recent_inspections(limit: int = 50) -> List[Dict[str, Any]]:
        return state.db.get_recent_inspections(limit=limit)

    @app.get("/api/statistics")
    def get_statistics() -> Dict[str, Any]:
        return state.db.get_statistics()

    @app.get("/api/export/csv")
    def export_csv() -> FileResponse:
        csv_path = Path("data/defectsense_export.csv")
        state.db.export_csv(csv_path)
        return FileResponse(
            path=str(csv_path),
            media_type="text/csv",
            filename=f"defectsense_inspections_{int(cv2.getTickCount())}.csv",
        )

    @app.post("/api/operator/query")
    def handle_operator_query(payload: OperatorQuestion) -> Dict[str, str]:
        """Answer operator questions using on-device Llama reasoning."""
        last_ctx = state.latest_result.to_dict() if state.latest_result else {}
        if state.latest_advice:
            last_ctx["advice"] = state.latest_advice
        answer = state.advisor_runner.answer_operator_query(payload.question, last_ctx)
        return {"question": payload.question, "answer": answer}

    @app.post("/api/settings")
    def update_settings(payload: SettingsUpdate) -> Dict[str, Any]:
        if payload.confidence_threshold is not None:
            state.config.vision.confidence_threshold = payload.confidence_threshold
        if payload.enable_buzzer is not None:
            state.config.alerts.enable_buzzer = payload.enable_buzzer
        if payload.enable_gpio is not None:
            state.config.alerts.enable_gpio = payload.enable_gpio
        return {"status": "ok", "settings": state.config.model_dump()}

    @app.websocket("/ws/live")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Push real-time inspection events and Llama advice to the UI."""
        await websocket.accept()
        state.active_websockets.add(websocket)
        try:
            # Send initial state snapshot
            init_msg = {
                "type": "init",
                "status": get_status(),
                "recent": state.db.get_recent_inspections(limit=10),
                "stats": state.db.get_statistics(),
            }
            await websocket.send_text(json.dumps(init_msg))

            # Keep connection open for incoming pings or queries
            while True:
                data = await websocket.receive_text()
                # Operators can also query over websocket
                try:
                    req = json.loads(data)
                    if req.get("action") == "query":
                        q_text = req.get("question", "")
                        last_ctx = state.latest_result.to_dict() if state.latest_result else {}
                        ans = state.advisor_runner.answer_operator_query(q_text, last_ctx)
                        await websocket.send_text(json.dumps({"type": "answer", "answer": ans}))
                except Exception:
                    pass
        except WebSocketDisconnect:
            state.active_websockets.discard(websocket)
        except Exception:
            state.active_websockets.discard(websocket)

    return app


async def broadcast_ws_message(state: AppState, message: Dict[str, Any]) -> None:
    """Helper to broadcast JSON event to all connected UI clients."""
    msg_str = json.dumps(message)
    dead_sockets = set()
    for ws in state.active_websockets:
        try:
            await ws.send_text(msg_str)
        except Exception:
            dead_sockets.add(ws)
    state.active_websockets.difference_update(dead_sockets)
