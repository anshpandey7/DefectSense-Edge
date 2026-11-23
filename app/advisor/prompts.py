"""
Prompt templates, JSON schemas, context builders, and JSON repair utilities for Llama 3.2 1B Instruct.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are DefectSense Advisor, an expert on-device metallurgical and quality-control AI on a factory floor.
Your job is to provide concise, actionable recommendations to the machine operator.

CRITICAL INSTRUCTIONS:
1. Output STRICT JSON ONLY. Do not write introductory words, explanations, or markdown code fences outside JSON.
2. The JSON MUST follow this exact schema:
{
  "summary": "<One sentence overview of the defect situation, max 15 words>",
  "likely_cause": "<Root metallurgical or mechanical cause on the rolling/forming line, max 20 words>",
  "recommended_action": "<Immediate, concrete physical check or action for the operator, max 25 words>",
  "escalate": <true if severity is CRITICAL or defect requires supervisor intervention, else false>
}
3. GUARDRAILS:
   - YOU MUST NOT OVERRIDE the PASS/REVIEW/REJECT verdict.
   - DO NOT invent measurements, speeds, or numbers not given in the context.
   - Keep total response under 60 words.
"""


def build_defect_context(
    product_name: str,
    line_name: str,
    part_id: str,
    verdict: str,
    severity: str,
    severity_score: float,
    detections: list[dict[str, Any]],
    breakdown: dict[str, Any],
    recent_trend: dict[str, Any],
) -> str:
    """Format structured inspection context for the Llama prompt."""
    primary_defect = detections[0]["class_name"] if detections else "None"
    defect_count = len(detections)
    total_area_pct = breakdown.get("total_area_pct", 0.0)
    edge_defects = breakdown.get("edge_defects_count", 0)

    context_dict = {
        "production_line": line_name,
        "product": product_name,
        "part_id": part_id,
        "verdict": verdict,
        "severity": severity,
        "severity_score": round(severity_score, 2),
        "primary_defect": primary_defect,
        "defect_count": defect_count,
        "surface_area_affected_pct": total_area_pct,
        "defects_near_critical_edge": edge_defects > 0,
        "recent_trend_last_10_parts": {
            "reject_rate_pct": recent_trend.get("reject_rate_pct", 0.0),
            "recent_rejects": recent_trend.get("recent_rejects", 0),
            "common_classes": recent_trend.get("top_recent_defect_classes", {}),
        },
    }

    return json.dumps(context_dict, indent=2)


def format_llama_instruct_prompt(defect_context_json: str) -> str:
    """Format input string with Llama 3.2 Instruct special tokens."""
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Analyze this manufacturing inspection result and advise the line operator:\n"
        f"{defect_context_json}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def parse_and_repair_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON parser and repair engine for LLM outputs.
    Extracts JSON from text, removes stray markdown, and validates required schema keys.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # Remove markdown code fences if model wrapped response in ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Search for first { to last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    # Common JSON repair heuristics
    # Fix trailing commas before }
    cleaned = re.sub(r",\s*}", "}", cleaned)
    # Fix python True/False to true/false
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)

    try:
        data = json.loads(cleaned)
        # Validate required fields
        required_fields = ["summary", "likely_cause", "recommended_action", "escalate"]
        for field in required_fields:
            if field not in data:
                return None
        # Ensure escalate is boolean
        if not isinstance(data["escalate"], bool):
            data["escalate"] = bool(data["escalate"])
        return data
    except Exception as e:
        logger.debug(f"JSON repair failed on LLM output: {e} | Raw text: {raw_text[:120]}")
        return None
