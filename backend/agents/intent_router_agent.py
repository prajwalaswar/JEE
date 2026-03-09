"""
intent_router_agent.py — Agent 2: Intent Router Agent

Responsibilities:
  - Confirms / refines the problem topic classification
  - Selects the optimal solving strategy
  - Routes to the appropriate solver mode
"""

from __future__ import annotations

import json
import logging
import re
import time
from enum import Enum
from typing import Optional

from backend.models import AgentTraceEntry, MathTopic, ParsedProblem
from backend.agents.llm_client import chat

logger = logging.getLogger(__name__)


# ── Enum: Solving strategies ──────────────────────────────────────────────────

class SolvingStrategy(str, Enum):
    SYMBOLIC     = "symbolic"       # Use SymPy for exact computation
    NUMERIC      = "numeric"        # Numerical approximation
    STEP_BY_STEP = "step_by_step"   # Guided LLM reasoning
    HYBRID       = "hybrid"         # LLM reasoning + SymPy verification
    VISUAL       = "visual"         # Geometry / graphing (explain visually)


# Topic → default strategy mapping
_STRATEGY_MAP = {
    MathTopic.ALGEBRA:        SolvingStrategy.HYBRID,
    MathTopic.CALCULUS:       SolvingStrategy.HYBRID,
    MathTopic.PROBABILITY:    SolvingStrategy.STEP_BY_STEP,
    MathTopic.LINEAR_ALGEBRA: SolvingStrategy.SYMBOLIC,
    MathTopic.TRIGONOMETRY:   SolvingStrategy.HYBRID,
    MathTopic.COORDINATE:     SolvingStrategy.STEP_BY_STEP,
    MathTopic.UNKNOWN:        SolvingStrategy.STEP_BY_STEP,
}

# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a JEE math router. Given a parsed math problem, your task is:
1. Confirm or refine the topic classification.
2. Identify the best solving strategy from: symbolic, numeric, step_by_step, hybrid, visual.
3. List any required mathematical tools or techniques.

Output ONLY valid JSON:
{
  "confirmed_topic": "<topic>",
  "strategy": "<strategy>",
  "required_techniques": ["<technique1>", ...],
  "routing_notes": "<brief explanation>"
}
"""


def _build_messages(problem: ParsedProblem) -> list:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Problem: {problem.problem_text}\n"
                f"Preliminary topic: {problem.topic}\n"
                f"Variables: {problem.variables}\n"
                f"Constraints: {problem.constraints}"
            ),
        },
    ]


# ── Agent function ────────────────────────────────────────────────────────────

def run_intent_router_agent(
    parsed_problem: ParsedProblem,
) -> tuple[ParsedProblem, SolvingStrategy, list[str], AgentTraceEntry]:
    """
    Route the problem and select solving strategy.

    Args:
        parsed_problem: Output from Parser Agent.

    Returns:
        Tuple of (updated ParsedProblem, SolvingStrategy, required_techniques, AgentTraceEntry).
    """
    t_start = time.time()
    logger.info(
        "[IntentRouter] Topic: %s, Strategy heuristic: %s",
        parsed_problem.topic,
        _STRATEGY_MAP.get(parsed_problem.topic, SolvingStrategy.STEP_BY_STEP),
    )

    # Default from heuristic map
    default_strategy  = _STRATEGY_MAP.get(parsed_problem.topic, SolvingStrategy.STEP_BY_STEP)
    techniques: list[str] = []
    routing_notes = ""

    try:
        messages = _build_messages(parsed_problem)
        llm_response = chat(messages, temperature=0.0, max_tokens=512)
        json_str = re.sub(r"```(?:json)?|```", "", llm_response).strip()
        # Strip ALL ASCII control characters including \n \r \t before first parse
        json_str_clean = re.sub(r"[\x00-\x1f\x7f]", " ", json_str)
        try:
            data = json.loads(json_str_clean)
        except json.JSONDecodeError:
            # LLM may embed LaTeX (\frac, \sqrt) as bare backslashes — double-escape them
            json_str_fixed = re.sub(r'\\(?!["\\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str_clean)
            try:
                data = json.loads(json_str_fixed)
            except json.JSONDecodeError:
                # Last resort: allow control characters via strict=False
                data = json.loads(json_str, strict=False)

        # Possibly refine the topic
        refined_topic_str = data.get("confirmed_topic", parsed_problem.topic.value)
        try:
            refined_topic = MathTopic(refined_topic_str.lower())
        except ValueError:
            refined_topic = parsed_problem.topic

        parsed_problem = parsed_problem.copy(update={"topic": refined_topic})

        strategy_str = data.get("strategy", default_strategy.value)
        try:
            strategy = SolvingStrategy(strategy_str.lower())
        except ValueError:
            strategy = default_strategy

        techniques    = data.get("required_techniques", [])
        routing_notes = data.get("routing_notes", "")

    except Exception as exc:
        logger.warning("[IntentRouter] LLM routing failed (%s). Using defaults.", exc)
        strategy = default_strategy

    duration = (time.time() - t_start) * 1000
    trace = AgentTraceEntry(
        agent="IntentRouterAgent",
        status="done",
        input_summary=f"Topic: {parsed_problem.topic}",
        output_summary=f"Strategy: {strategy}, Techniques: {techniques}",
        duration_ms=round(duration, 1),
    )

    logger.info(
        "[IntentRouter] Done in %.0fms. Strategy: %s",
        duration,
        strategy,
    )
    return parsed_problem, strategy, techniques, trace
