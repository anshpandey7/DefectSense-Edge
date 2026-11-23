"""
Base interfaces and data structures for vision defect detection and inspection results.
"""

from __future__ import annotations

import abc
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2] in normalized [0, 1] coords or pixel coords
    class_name: str
    confidence: float
    area_pct: float = 0.0  # Percentage of total inspected surface area
    is_edge: bool = False  # True if defect touches critical edge margin
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InspectionResult:
    part_id: str
    timestamp: float
    frame_index: int
    detections: List[Detection]
    verdict: str  # "PASS", "REVIEW", "REJECT"
    severity: str  # "NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    severity_score: float  # 0.0 to 1.0
    score_breakdown: Dict[str, Any]
    inference_ms: float
    backend: str
    annotated_image: Optional[np.ndarray] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    advice: Optional[Dict[str, Any]] = None  # Filled asynchronously by Advisor

    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        d = {
            "part_id": self.part_id,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "detections": [det.to_dict() for det in self.detections],
            "verdict": self.verdict,
            "severity": self.severity,
            "severity_score": round(self.severity_score, 4),
            "score_breakdown": self.score_breakdown,
            "inference_ms": round(self.inference_ms, 2),
            "backend": self.backend,
            "source_metadata": self.source_metadata,
            "advice": self.advice,
        }
        return d


class BaseDetector(abc.ABC):
    """Abstract vision defect detector interface."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.35, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.is_loaded = False

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights / runtime sessions."""
        pass

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run inference on an image array (BGR or RGB).
        Returns:
            (detections, inference_time_ms)
        """
        pass

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Return backend name e.g. cpu_onnx, hailo, dummy."""
        pass


class DummyDetector(BaseDetector):
    """
    Deterministic dummy detector for testing, simulation, and benchmark verification
    when no trained neural network weights file is yet present.
    """

    def __init__(self, model_path: str = "dummy", confidence_threshold: float = 0.35, iou_threshold: float = 0.45):
        super().__init__(model_path, confidence_threshold, iou_threshold)
        self.step = 0

    def load(self) -> None:
        self.is_loaded = True

    @property
    def backend_name(self) -> str:
        return "dummy_sim"

    def predict(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        self.step += 1

        # Periodic defect generation for realistic pipeline testing
        detections: List[Detection] = []
        cycle = self.step % 6

        if cycle == 1:
            # Crack defect near top-left edge
            x1, y1, x2, y2 = 20, 25, 140, 50
            area_pct = ((x2 - x1) * (y2 - y1)) / (w * h) * 100.0
            detections.append(Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                class_name="crack",
                confidence=0.91,
                area_pct=area_pct,
                is_edge=True,
            ))
        elif cycle == 2:
            # Inclusion defect in center
            x1, y1, x2, y2 = int(w * 0.45), int(h * 0.45), int(w * 0.55), int(h * 0.55)
            area_pct = ((x2 - x1) * (y2 - y1)) / (w * h) * 100.0
            detections.append(Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                class_name="inclusion",
                confidence=0.78,
                area_pct=area_pct,
                is_edge=False,
            ))
        elif cycle == 3:
            # Multiple scratches
            for i, offset in enumerate([50, 180, 310]):
                x1, y1, x2, y2 = offset, 200, offset + 60, 215
                detections.append(Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    class_name="scratches",
                    confidence=0.68 + (i * 0.05),
                    area_pct=0.35,
                    is_edge=False,
                ))
        elif cycle == 4:
            # Pitted surface
            x1, y1, x2, y2 = 80, 150, 220, 290
            area_pct = ((x2 - x1) * (y2 - y1)) / (w * h) * 100.0
            detections.append(Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                class_name="pitted_surface",
                confidence=0.84,
                area_pct=area_pct,
                is_edge=False,
            ))
        # cycle 0 and 5: clean part (no defects -> PASS)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return detections, elapsed_ms
