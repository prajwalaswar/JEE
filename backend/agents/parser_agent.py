"""
parser_agent.py — Agent 1: Parser Agent

Responsibilities:
  - Cleans OCR / ASR output
  - Extracts structured problem JSON
  - Detects ambiguity → triggers HITL if needed
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from backend.models import (
    AgentTraceEntry,
    HITLTriggerReason,
    MathTopic,
    ParsedProblem,
)
from backend.agents.llm_client import chat
from backend.multimodal.text_processor import clean_text

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a math problem parser specialized in JEE (Joint Entrance Exam) questions.

Your job is to analyze raw text (possibly noisy from OCR or speech recognition) and extract a structured JSON representation of the math problem.

Output ONLY valid JSON with this exact schema:
{
  "problem_text": "<cleaned, well-formatted problem statement>",
  "topic": "<one of: algebra, calculus, probability, linear_algebra, trigonometry, coordinate_geometry, unknown>",
  "variables": ["<list of variable names found in the problem>"],
  "constraints": ["<list of constraints or conditions mentioned>"],
  "needs_clarification": <true if problem is ambiguous, else false>,
  "clarification_reason": "<explain ambiguity if needs_clarification is true, else null>"
}

Rules:
- Fix obvious OCR errors (e.g. 'O' vs '0', 'l' vs '1')
- Preserve mathematical notation accurately
- If the problem is clear, set needs_clarification to false
- Output ONLY the JSON object, no extra text
"""


def _build_user_message(raw_text: str) -> str:
    return f"Parse this math problem:\n\n{raw_text}"


# ── Agent function ────────────────────────────────────────────────────────────

def run_parser_agent(
    raw_text: str,
    session_id: str = "",
) -> tuple[ParsedProblem, AgentTraceEntry]:
    """
    Parse raw input text into a structured ParsedProblem.

    Args:
        raw_text:   Raw text from OCR, ASR, or direct input.
        session_id: Session ID for tracing.

    Returns:
        Tuple of (ParsedProblem, AgentTraceEntry).
    """
    t_start = time.time()
    logger.info("[ParserAgent] Starting. Input length: %d", len(raw_text))

    # Step 1: Basic text cleaning
    cleaned = clean_text(raw_text)

    # Step 2: LLM call
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(cleaned)},
    ]

    try:
        llm_response = chat(messages, temperature=0.0, max_tokens=512)
        # Extract JSON from response (remove markdown code fences if present)
        json_str = re.sub(r"```(?:json)?|```", "", llm_response).strip()
        data = json.loads(json_str)

        parsed = ParsedProblem(
            problem_text=data.get("problem_text", cleaned),
            topic=MathTopic(data.get("topic", "unknown")),
            variables=data.get("variables", []),
            constraints=data.get("constraints", []),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_reason=data.get("clarification_reason"),
        )

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("[ParserAgent] LLM parse failed (%s). Using fallback.", exc)
        # Fallback: return minimally structured result
        parsed = ParsedProblem(
            problem_text=cleaned,
            topic=MathTopic.UNKNOWN,
            needs_clarification=True,
            clarification_reason=f"Parser LLM failed: {exc}",
        )

    duration = (time.time() - t_start) * 1000
    trace = AgentTraceEntry(
        agent="ParserAgent",
        status="hitl_triggered" if parsed.needs_clarification else "done",
        input_summary=raw_text[:100],
        output_summary=f"Topic: {parsed.topic}, Ambiguous: {parsed.needs_clarification}",
        duration_ms=round(duration, 1),
    )

    logger.info(
        "[ParserAgent] Done in %.0fms. Topic: %s, Needs clarification: %s",
        duration,
        parsed.topic,
        parsed.needs_clarification,
    )
    return parsed, trace
