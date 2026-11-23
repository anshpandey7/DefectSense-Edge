"""
DefectSense Edge: Real-time visual inspection and quality-control assistant.
Main entry point and inspection loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import uvicorn

# Ensure root package is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import AppConfig, load_config
from app.hardware import HardwareProfile, detect_hardware
from app.capture.base import FrameSource
from app.capture.folder_source import ImageFolderSource
from app.capture.camera_source import CameraSource, VideoSource
from app.vision.backend_base import BaseDetector, DummyDetector, InspectionResult
from app.vision.postprocess import SeverityEngine
from app.storage.database import DatabaseManager
from app.advisor.executorch_llama import ExecuTorchLlamaRunner
from app.advisor.worker import AsyncAdvisorWorker
from app.alerts.buzzer_gpio import GPIOAlertController
from app.web.server import AppState, create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("defectsense.main")


def annotate_frame(
    frame: np.ndarray,
    result: InspectionResult,
    fps: float = 0.0,
    hardware_info: str = "CPU",
) -> np.ndarray:
    """Draw professional industrial HUD overlay with bounding boxes, verdict badge, and FPS."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Colors (BGR)
    verdict_colors = {
        "PASS": (46, 139, 87),     # Sea Green
        "REVIEW": (0, 165, 255),    # Orange
        "REJECT": (34, 34, 205),    # Crimson Red
    }
    badge_color = verdict_colors.get(result.verdict, (128, 128, 128))

    # Draw defect bounding boxes
    for det in result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        # Box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), badge_color, 2)

        # Label background
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(18, y1)), badge_color, -1)
        cv2.putText(canvas, label, (x1 + 3, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Top banner background (semi-transparent)
    header_overlay = canvas.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, 52), (25, 25, 25), -1)
    cv2.addWeighted(header_overlay, 0.8, canvas, 0.2, 0, canvas)

    # Big Verdict Badge
    cv2.rectangle(canvas, (10, 8), (140, 44), badge_color, -1)
    cv2.putText(canvas, result.verdict, (22, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Top Status Info
    info_text = f"Part: {result.part_id} | Sev: {result.severity} ({result.severity_score:.2f}) | Def: {len(result.detections)}"
    cv2.putText(canvas, info_text, (155, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    perf_text = f"FPS: {fps:.1f} | Latency: {result.inference_ms:.1f}ms | Backend: {hardware_info}"
    cv2.putText(canvas, perf_text, (155, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 220, 160), 1, cv2.LINE_AA)

    return canvas


def build_capture_source(config: AppConfig, override_source: str | None = None, override_path: str | None = None) -> FrameSource:
    source_type = override_source or config.source.type
    source_path = override_path or config.source.path

    if source_type == "folder":
        folder = Path(source_path)
        logger.info(f"Using Image Folder source: {folder}")
        return ImageFolderSource(folder, fps=config.source.fps, loop=config.source.loop, auto_generate_samples=True)
    elif source_type == "video":
        logger.info(f"Using Video source: {source_path}")
        return VideoSource(source_path, loop=config.source.loop)
    elif source_type == "camera":
        cam_idx = config.source.camera_index
        logger.info(f"Using Camera index: {cam_idx}")
        return CameraSource(cam_idx, width=config.source.width, height=config.source.height, fps=config.source.fps)
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def run_pipeline(
    config: AppConfig,
    source: FrameSource,
    detector: BaseDetector,
    severity_engine: SeverityEngine,
    db: DatabaseManager,
    advisor_worker: AsyncAdvisorWorker | None = None,
    alert_controller: GPIOAlertController | None = None,
    web_state: AppState | None = None,
    max_frames: int | None = None,
    headless: bool = True,
) -> int:
    """Execute main real-time visual inspection loop."""
    if not source.open():
        logger.error(f"Failed to open source: {source.source_type}")
        return 1

    detector.load()
    logger.info(f"Vision detector initialized: {detector.backend_name}")

    if advisor_worker:
        # Hook callback to update web state and trigger physical alert
        def on_advice(inspection_id: int, advice: dict, fallback_used: bool):
            if web_state:
                web_state.update_advice(advice)

        advisor_worker.register_callback(on_advice)
        advisor_worker.start()

    frame_count = 0
    t_start = time.perf_counter()
    last_fps_time = t_start
    fps_counter = 0
    current_fps = 0.0

    logger.info("Entering visual inspection loop. Press Ctrl+C or reach max_frames to finish.")
    try:
        while True:
            t_frame_start = time.perf_counter()
            ret, frame, meta = source.read()
            if not ret or frame is None:
                logger.info("End of source frame stream.")
                break

            frame_count += 1
            fps_counter += 1

            # Update FPS rolling counter
            now = time.perf_counter()
            if now - last_fps_time >= 1.0:
                current_fps = fps_counter / (now - last_fps_time)
                fps_counter = 0
                last_fps_time = now

            # 1. Vision Detection (separable pipeline)
            detections, inf_ms = detector.predict(frame)

            # 2. Severity and Verdict Evaluation (transparent deterministic engine)
            h, w = frame.shape[:2]
            verdict, severity, score, breakdown = severity_engine.evaluate(detections, image_width=w, image_height=h)

            part_id = f"PART-{frame_count:05d}"
            result = InspectionResult(
                part_id=part_id,
                timestamp=time.time(),
                frame_index=frame_count,
                detections=detections,
                verdict=verdict,
                severity=severity,
                severity_score=score,
                score_breakdown=breakdown,
                inference_ms=inf_ms,
                backend=detector.backend_name,
                source_metadata=meta,
            )

            # 3. Create annotated overlay image
            annotated = annotate_frame(frame, result, fps=current_fps, hardware_info=detector.backend_name)
            result.annotated_image = annotated

            # Update shared web portal state
            if web_state:
                web_state.update_frame(annotated, result, current_fps)

            # Trigger physical / audible alert on defects
            if alert_controller and verdict in ("REVIEW", "REJECT"):
                alert_controller.trigger_verdict_alert(verdict, severity)

            # 4. Save to local SQLite database & local storage
            rec_id = db.save_inspection(result, frame=frame)

            # 5. Dispatch to Async Advisor Worker (non-blocking)
            if advisor_worker:
                advisor_worker.submit_inspection(rec_id, result)

            # Print concise factory log
            det_summary = f"{len(detections)} defect(s): " + ", ".join([d.class_name for d in detections]) if detections else "Clean"
            logger.info(
                f"[{part_id}] Verdict: {verdict:<6} | Severity: {severity:<8} (score={score:.2f}) "
                f"| {det_summary} | Latency: {inf_ms:.1f}ms | FPS: {current_fps:.1f} | DB ID: {rec_id}"
            )

            if not headless:
                cv2.imshow("DefectSense Edge - Real-time Inspection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit request from GUI window.")
                    break

            if max_frames and frame_count >= max_frames:
                logger.info(f"Reached specified limit of {max_frames} frames.")
                break

    except KeyboardInterrupt:
        logger.info("Inspection loop interrupted by user.")
    finally:
        if advisor_worker:
            advisor_worker.stop()
        source.release()
        if not headless:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    logger.info(f"Inspection loop finished: {frame_count} frames processed in {total_time:.2f}s ({avg_fps:.1f} avg FPS).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DefectSense Edge: On-device visual inspection")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to default.yaml")
    parser.add_argument("--source", type=str, choices=["folder", "video", "camera"], help="Override input source type")
    parser.add_argument("--source-path", type=str, help="Override path to image folder or video file")
    parser.add_argument("--backend", type=str, choices=["auto", "cpu_onnx", "hailo", "dummy"], help="Override backend")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for tests)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without X11/GUI window")
    parser.add_argument("--gui", dest="headless", action="store_false", help="Open OpenCV display window")
    parser.add_argument("--offline-check", action="store_true", help="Run offline validation test")
    parser.add_argument("--no-web", action="store_true", help="Disable FastAPI web operator portal")
    parser.add_argument("--port", type=int, default=8000, help="Web portal port (default: 8000)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = BASE_DIR / cfg_path

    if args.offline_check:
        logger.info("=" * 60)
        logger.info("RUNNING AIR-GAPPED OFFLINE PIPELINE VALIDATION TEST")
        logger.info("=" * 60)
        import pytest
        test_path = BASE_DIR / "tests" / "test_offline_pipeline.py"
        ret_code = pytest.main(["-v", str(test_path)])
        if ret_code == 0:
            logger.info("ALL AIR-GAPPED OFFLINE TESTS PASSED: Zero cloud dependencies verified.")
        else:
            logger.error("OFFLINE VERIFICATION FAILED!")
        return ret_code

    config = load_config(cfg_path)

    # Hardware detection
    hw_pref = args.backend or config.hardware.preferred_backend
    hw_profile = detect_hardware(preferred_backend=hw_pref)

    # Initialize components
    source = build_capture_source(config, override_source=args.source, override_path=args.source_path)

    # Model / detector initialization
    # If explicit dummy requested, or onnx model does not yet exist, use DummyDetector
    onnx_path = BASE_DIR / config.vision.model_path
    if args.backend == "dummy" or not onnx_path.exists():
        logger.info(f"Using DummyDetector (simulation/test mode). Model path check: {onnx_path.name}")
        detector = DummyDetector(
            confidence_threshold=config.vision.confidence_threshold,
            iou_threshold=config.vision.iou_threshold,
        )
    else:
        # Load ONNX backend
        from app.vision.onnx_cpu import ONNXCPUDetector
        detector = ONNXCPUDetector(
            model_path=str(onnx_path),
            confidence_threshold=config.vision.confidence_threshold,
            iou_threshold=config.vision.iou_threshold,
            num_threads=config.hardware.num_threads,
        )

    rules_path = BASE_DIR / config.severity.rules_file
    severity_engine = SeverityEngine(rules_file=rules_path)

    db_path = BASE_DIR / config.storage.db_path
    image_dir = BASE_DIR / config.storage.image_dir
    db = DatabaseManager(
        db_path=db_path,
        image_dir=image_dir,
        save_all_images=config.storage.save_all_images,
        thumbnail_size=tuple(config.storage.thumbnail_size),
    )

    # Advisor worker initialization
    pte_path = BASE_DIR / config.advisor.pte_model_path
    tok_path = BASE_DIR / config.advisor.tokenizer_path
    runner = ExecuTorchLlamaRunner(
        pte_model_path=str(pte_path),
        tokenizer_path=str(tok_path),
        num_threads=config.hardware.num_threads,
        max_tokens=config.advisor.max_tokens,
        temperature=config.advisor.temperature,
    )
    advisor_worker = AsyncAdvisorWorker(
        runner=runner,
        db=db,
        debounce_seconds=config.advisor.debounce_seconds,
        timeout_seconds=config.advisor.timeout_seconds,
        line_name=config.app.line_name,
        product_name=config.app.product_name,
    )

    alert_controller = GPIOAlertController(
        enable_gpio=config.alerts.enable_gpio,
        enable_buzzer=config.alerts.enable_buzzer,
        reject_pin=config.alerts.gpio_reject_pin,
        pass_pin=config.alerts.gpio_pass_pin,
    )

    web_state = None
    if not args.no_web:
        web_state = AppState(config=config, hw_profile=hw_profile, db=db, advisor_runner=runner)
        web_app = create_app(web_state)

        def start_server():
            server_port = args.port or config.web.port
            logger.info(f"Starting Operator Web Portal at http://localhost:{server_port}")
            uvicorn.run(web_app, host=config.web.host, port=server_port, log_level="warning")

        web_thread = threading.Thread(target=start_server, daemon=True, name="WebPortalServer")
        web_thread.start()
        time.sleep(0.5)

    return run_pipeline(
        config=config,
        source=source,
        detector=detector,
        severity_engine=severity_engine,
        db=db,
        advisor_worker=advisor_worker,
        alert_controller=alert_controller,
        web_state=web_state,
        max_frames=args.max_frames,
        headless=args.headless,
    )


if __name__ == "__main__":
    sys.exit(main())
