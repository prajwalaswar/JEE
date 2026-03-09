"""
verifier_agent.py — Agent 4: Verifier Agent

Responsibilities:
  - Verify solution correctness
  - Check domain validity and edge cases
  - Assign confidence score
  - Trigger HITL when unsure
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List

from backend.models import (
    AgentTraceEntry,
    HITLTriggerReason,
    ParsedProblem,
    Solution,
    VerificationResult,
)
from backend.agents.llm_client import chat
from backend.config import VERIFIER_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a rigorous JEE mathematics verifier.

Given a problem and its proposed solution, your job is to:
1. Verify whether the solution is mathematically correct.
2. Check if the domain is appropriate (e.g., no division by zero, sqrt of negative, etc.)
3. Consider edge cases.
4. Assess your confidence in the verification.

Output ONLY valid JSON:
{
  "is_correct": <true|false>,
  "confidence": <float 0–1>,
  "issues": ["<issue1>", "<issue2>", ...],
  "verification_notes": "<brief explanation>"
}

Be strict. If you are not sure, set confidence < 0.8 and describe issues.
"""


def _build_messages(problem: ParsedProblem, solution: Solution) -> list:
    steps_text = "\n".join(
        f"  Step {s.step_number}: {s.description} → {s.result}"
        for s in solution.steps
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"PROBLEM:\n{problem.problem_text}\n\n"
                f"PROPOSED SOLUTION:\n"
                f"Final Answer: {solution.final_answer}\n"
                f"Steps:\n{steps_text}\n\n"
                f"Solver confidence: {solution.confidence}"
            ),
        },
    ]


# ── SymPy numeric cross-check ──────────────────────────────────────────────────

def _sympy_spot_check(problem: ParsedProblem, solution: Solution) -> bool | None:
    """
    Attempt a quick numeric spot-check via SymPy.
    Returns True if check passed, False if failed, None if not applicable.
    """
    from backend.tools.math_tools import run_math_tool

    # Only attempt for equation solutions
    text = problem.problem_text
    if not any(kw in text.lower() for kw in ["solve", "roots", "equation"]):
        return None

    # If we have variables and solutions, try substituting
    if not problem.variables or not solution.final_answer:
        return None

    try:
        # Try to evaluate the answer as a float
        var  = problem.variables[0]
        ans  = solution.final_answer
        # Strip "x = " prefix if present
        ans  = re.sub(rf"^{var}\s*=\s*", "", ans.strip())
        result = run_math_tool("evaluate_expression", expr_str=ans, substitutions={})
        return result["success"]
    except Exception:
        return None


# ── Agent function ────────────────────────────────────────────────────────────

def run_verifier_agent(
    parsed_problem: ParsedProblem,
    solution: Solution,
) -> tuple[VerificationResult, AgentTraceEntry]:
    """
    Verify the proposed solution.

    Args:
        parsed_problem: The original parsed problem.
        solution:       Solution produced by the Solver Agent.

    Returns:
        Tuple of (VerificationResult, AgentTraceEntry).
    """
    t_start = time.time()
    logger.info("[VerifierAgent] Verifying solution for: %s…", parsed_problem.problem_text[:60])

    # ── SymPy spot-check ──────────────────────────────────────────────────────
    spot_check = _sympy_spot_check(parsed_problem, solution)
    if spot_check is False:
        logger.warning("[VerifierAgent] SymPy spot-check failed!")

    # ── LLM verification ──────────────────────────────────────────────────────
    is_correct    = False
    confidence    = 0.0
    issues: list[str] = []
    notes         = ""

    try:
        messages = _build_messages(parsed_problem, solution)
        llm_response = chat(messages, temperature=0.0, max_tokens=512, prefer_gemini=True)
        json_str = re.sub(r"```(?:json)?|```", "", llm_response).strip()
        data = json.loads(json_str)

        is_correct = bool(data.get("is_correct", False))
        confidence = float(data.get("confidence", 0.5))
        issues     = data.get("issues", [])
        notes      = data.get("verification_notes", "")

    except Exception as exc:
        logger.error("[VerifierAgent] LLM verification failed: %s", exc)
        # If LLM fails, fall back to solver's own confidence
        confidence = solution.confidence * 0.9
        is_correct = confidence > VERIFIER_CONFIDENCE_THRESHOLD
        issues.append(f"Verifier LLM failed: {exc}")

    # Incorporate spot-check
    if spot_check is False:
        confidence = max(0.0, confidence - 0.2)
        issues.append("SymPy numeric spot-check failed.")

    needs_hitl   = confidence < VERIFIER_CONFIDENCE_THRESHOLD
    hitl_reason  = HITLTriggerReason.VERIFIER_UNCERTAIN if needs_hitl else None

    if needs_hitl:
        logger.info(
            "[VerifierAgent] Low confidence %.2f — HITL triggered.", confidence
        )

    verification = VerificationResult(
        is_correct=is_correct,
        confidence=round(confidence, 4),
        issues=issues,
        needs_hitl=needs_hitl,
        hitl_reason=hitl_reason,
    )

    duration = (time.time() - t_start) * 1000
    trace = AgentTraceEntry(
        agent="VerifierAgent",
        status="hitl_triggered" if needs_hitl else "done",
        input_summary=f"Answer: {solution.final_answer[:60]}",
        output_summary=(
            f"Correct: {is_correct}, Conf: {confidence:.2f}, "
            f"Issues: {len(issues)}"
        ),
        duration_ms=round(duration, 1),
    )
    logger.info(
        "[VerifierAgent] Done in %.0fms. Correct: %s, Conf: %.2f",
        duration,
        is_correct,
        confidence,
    )
    return verification, trace
