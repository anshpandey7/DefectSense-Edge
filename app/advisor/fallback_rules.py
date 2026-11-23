"""
Deterministic rule-based fallback recommendation table for DefectSense Edge.
Guarantees reliable, instant advice even if the ExecuTorch model is missing, times out, or errors.
"""

from __future__ import annotations

from typing import Any, Dict


FALLBACK_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "crack": {
        "summary": "Structural fissure detected compromising mechanical integrity.",
        "likely_cause": "Excessive rolling tensile stress, thermal shock in cooling bed, or defective billet.",
        "recommended_action": "Halt line feed immediately. Inspect roll gap, tension leveling, and crop shear blades.",
        "escalate": True,
    },
    "crazing": {
        "summary": "Network of surface microcracks detected across the component.",
        "likely_cause": "Thermal fatigue from unequal roll cooling or water spray nozzle blockage.",
        "recommended_action": "Check cooling water spray headers and pressure gauges for nozzle blockage.",
        "escalate": False,
    },
    "inclusion": {
        "summary": "Non-metallic inclusion or slag particle embedded in sheet surface.",
        "likely_cause": "Refractory erosion or slag carryover during continuous casting.",
        "recommended_action": "Segregate affected batch for ultrasonic testing; verify upstream tundish shroud.",
        "escalate": True,
    },
    "pitted_surface": {
        "summary": "Localized surface pitting cavities detected.",
        "likely_cause": "Over-pickling acid attack or roll surface fatigue spalling.",
        "recommended_action": "Verify pickling bath acid concentration and check work roll surface for pitting.",
        "escalate": False,
    },
    "rolled_in_scale": {
        "summary": "Oxide scale pressed into the sheet surface during rolling passes.",
        "likely_cause": "Insufficient descaling spray pressure or furnace atmosphere over-oxidation.",
        "recommended_action": "Verify high-pressure descaler pressure (>15 MPa) and clean descaling nozzles.",
        "escalate": False,
    },
    "scratches": {
        "summary": "Longitudinal mechanical abrasive marks along rolling direction.",
        "likely_cause": "Friction against misaligned entry guide shoes, aprons, or worn table rollers.",
        "recommended_action": "Inspect guide rollers, pass line height, and side trimmer guides for debris.",
        "escalate": False,
    },
    "patches": {
        "summary": "Irregular localized discoloration or thickness thinning patch.",
        "likely_cause": "Non-uniform work roll wear or localized lubricating oil film breakdown.",
        "recommended_action": "Inspect roll profile and lubricant spray alignment on the active stand.",
        "escalate": False,
    },
    "unknown_anomaly": {
        "summary": "Unsupervised visual anomaly outside standard defect patterns.",
        "likely_cause": "Unusual metallurgical variation, foreign contamination, or lighting glare.",
        "recommended_action": "Perform manual visual inspection and isolate part for lab quality audit.",
        "escalate": False,
    },
}

CLEAN_RECOMMENDATION = {
    "summary": "Part inspected and within acceptable metallurgical tolerances.",
    "likely_cause": "Standard controlled rolling conditions.",
    "recommended_action": "Continue automated production flow.",
    "escalate": False,
}


def get_fallback_advice(
    defect_type: str | None,
    severity: str = "LOW",
    recent_rejects: int = 0,
) -> Dict[str, Any]:
    """Retrieve rule-based recommendation marked with fallback flag."""
    if not defect_type or defect_type.lower() in ("none", "clean"):
        advice = dict(CLEAN_RECOMMENDATION)
    else:
        key = defect_type.lower().replace("-", "_").replace(" ", "_")
        advice = dict(FALLBACK_RECOMMENDATIONS.get(key, FALLBACK_RECOMMENDATIONS["unknown_anomaly"]))

    # Escalate if multiple consecutive rejects on the line
    if recent_rejects >= 3:
        advice["escalate"] = True
        advice["summary"] += f" WARNING: Line has seen {recent_rejects} recent rejects."

    advice["fallback"] = True
    return advice
