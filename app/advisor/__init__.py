"""
Advisor package for DefectSense Edge.
"""

from app.advisor.executorch_llama import ExecuTorchLlamaRunner
from app.advisor.fallback_rules import get_fallback_advice, FALLBACK_RECOMMENDATIONS
from app.advisor.prompts import (
    SYSTEM_PROMPT,
    build_defect_context,
    format_llama_instruct_prompt,
    parse_and_repair_json,
)
from app.advisor.worker import AsyncAdvisorWorker

__all__ = [
    "ExecuTorchLlamaRunner",
    "get_fallback_advice",
    "FALLBACK_RECOMMENDATIONS",
    "SYSTEM_PROMPT",
    "build_defect_context",
    "format_llama_instruct_prompt",
    "parse_and_repair_json",
    "AsyncAdvisorWorker",
]
