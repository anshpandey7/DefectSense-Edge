"""
Severity engine and verdict post-processor.
Computes deterministic, transparent severity scores and verdicts (PASS / REVIEW / REJECT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import yaml

from app.vision.backend_base import Detection


class SeverityEngine:
    """
    Transparent severity calculation engine adhering to config/severity_rules.yaml.
    Formula:
        Score = (class_weight * w_class + area_factor * w_area + count_factor * w_count + edge_factor * w_edge) * max_conf
    """

    def __init__(self, rules_file: str | Path = "config/severity_rules.yaml"):
        self.rules_file = Path(rules_file)
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if not self.rules_file.exists():
            # Return resilient built-in defaults
            return {
                "formula_weights": {"class_weight": 0.40, "area_weight": 0.30, "count_weight": 0.15, "edge_weight": 0.15},
                "defect_class_weights": {
                    "crack": 1.0, "crazing": 0.8, "pitted_surface": 0.7,
                    "rolled_in_scale": 0.65, "inclusion": 0.60, "patches": 0.50,
                    "scratches": 0.40, "unknown_anomaly": 0.85
                },
                "area_scaling": {
                    "critical_threshold_pct": 10.0,
                    "high_threshold_pct": 4.0,
                    "medium_threshold_pct": 1.0,
                    "low_threshold_pct": 0.2
                },
                "edge_proximity_margin_pct": 0.05,
                "edge_penalty_multiplier": 1.25,
                "severity_thresholds": {"critical": 0.75, "high": 0.50, "medium": 0.25, "low": 0.0},
                "verdict_mapping": {
                    "CRITICAL": "REJECT", "HIGH": "REJECT",
                    "MEDIUM": "REVIEW", "LOW": "PASS", "NONE": "PASS"
                },
                "overrides": {
                    "reject_if_any_class": ["crack"],
                    "reject_if_total_defects_exceed": 5,
                    "review_if_defect_count_exceeds": 2
                }
            }
        with open(self.rules_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def calculate_area_factor(self, total_area_pct: float) -> float:
        scaling = self.rules.get("area_scaling", {})
        crit = scaling.get("critical_threshold_pct", 10.0)
        high = scaling.get("high_threshold_pct", 4.0)
        med = scaling.get("medium_threshold_pct", 1.0)
        low = scaling.get("low_threshold_pct", 0.2)

        if total_area_pct >= crit:
            return 1.0
        elif total_area_pct >= high:
            return 0.75 + 0.25 * ((total_area_pct - high) / (crit - high))
        elif total_area_pct >= med:
            return 0.50 + 0.25 * ((total_area_pct - med) / (high - med))
        elif total_area_pct >= low:
            return 0.25 + 0.25 * ((total_area_pct - low) / (med - low))
        else:
            return min(0.25, total_area_pct / low * 0.25) if low > 0 else 0.0

    def evaluate(self, detections: List[Detection], image_width: int = 640, image_height: int = 640) -> Tuple[str, str, float, Dict[str, Any]]:
        """
        Evaluate detections on a part.
        Returns:
            (verdict, severity_label, severity_score, breakdown_dict)
        """
        if not detections:
            return "PASS", "NONE", 0.0, {
                "defect_count": 0,
                "total_area_pct": 0.0,
                "max_class_hazard": 0.0,
                "edge_defects_count": 0,
                "rule_triggered": "clean_surface",
            }

        weights = self.rules.get("formula_weights", {})
        w_class = weights.get("class_weight", 0.40)
        w_area = weights.get("area_weight", 0.30)
        w_count = weights.get("count_weight", 0.15)
        w_edge = weights.get("edge_weight", 0.15)

        class_weights = self.rules.get("defect_class_weights", {})
        overrides = self.rules.get("overrides", {})
        reject_classes = overrides.get("reject_if_any_class", ["crack"])

        total_area_pct = 0.0
        max_class_hazard = 0.0
        edge_defects_count = 0
        confidences = []
        has_immediate_reject_defect = False
        immediate_reject_reason = ""

        # Check edge proximity
        margin_pct = self.rules.get("edge_proximity_margin_pct", 0.05)
        x_min_edge = image_width * margin_pct
        x_max_edge = image_width * (1.0 - margin_pct)
        y_min_edge = image_height * margin_pct
        y_max_edge = image_height * (1.0 - margin_pct)

        for det in detections:
            confidences.append(det.confidence)
            total_area_pct += det.area_pct

            # Lookup hazard weight
            hazard = class_weights.get(det.class_name, 0.50)
            if hazard > max_class_hazard:
                max_class_hazard = hazard

            # Check bbox edge proximity if coordinates are pixel coords
            x1, y1, x2, y2 = det.bbox
            is_edge = (x1 <= x_min_edge or x2 >= x_max_edge or y1 <= y_min_edge or y2 >= y_max_edge)
            det.is_edge = det.is_edge or is_edge
            if det.is_edge:
                edge_defects_count += 1

            if det.class_name in reject_classes and det.confidence >= 0.40:
                has_immediate_reject_defect = True
                immediate_reject_reason = f"Critical defect class detected: {det.class_name} (conf={det.confidence:.2f})"

        area_factor = self.calculate_area_factor(total_area_pct)
        count_factor = min(1.0, len(detections) / 5.0)
        edge_factor = min(1.0, edge_defects_count / max(1, len(detections)))
        max_confidence = max(confidences) if confidences else 1.0

        raw_score = (
            (max_class_hazard * w_class) +
            (area_factor * w_area) +
            (count_factor * w_count) +
            (edge_factor * w_edge)
        ) * max_confidence

        score = float(np.clip(raw_score, 0.0, 1.0))

        # Severity classification
        thresholds = self.rules.get("severity_thresholds", {})
        crit_th = thresholds.get("critical", 0.75)
        high_th = thresholds.get("high", 0.50)
        med_th = thresholds.get("medium", 0.25)

        if score >= crit_th:
            severity = "CRITICAL"
        elif score >= high_th:
            severity = "HIGH"
        elif score >= med_th:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Verdict calculation
        verdict_map = self.rules.get("verdict_mapping", {})
        verdict = verdict_map.get(severity, "REVIEW")

        # Check override rules
        triggered_override = None
        if has_immediate_reject_defect:
            verdict = "REJECT"
            severity = "CRITICAL" if score < crit_th else severity
            score = max(score, 0.85)
            triggered_override = immediate_reject_reason
        elif len(detections) > overrides.get("reject_if_total_defects_exceed", 5):
            verdict = "REJECT"
            triggered_override = f"Total defects ({len(detections)}) exceeded max limit"
        elif len(detections) > overrides.get("review_if_defect_count_exceeds", 2) and verdict == "PASS":
            verdict = "REVIEW"
            triggered_override = f"Defect count ({len(detections)}) elevated from PASS to REVIEW"

        breakdown = {
            "defect_count": len(detections),
            "total_area_pct": round(total_area_pct, 3),
            "max_class_hazard": round(max_class_hazard, 3),
            "area_factor": round(area_factor, 3),
            "count_factor": round(count_factor, 3),
            "edge_factor": round(edge_factor, 3),
            "edge_defects_count": edge_defects_count,
            "max_confidence": round(max_confidence, 3),
            "calculated_score": round(score, 4),
            "triggered_override": triggered_override,
        }

        return verdict, severity, score, breakdown
