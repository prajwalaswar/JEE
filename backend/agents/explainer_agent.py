"""
explainer_agent.py — Agent 5: Explainer Agent

Responsibilities:
  - Generate student-friendly explanations
  - Highlight key concepts
  - Point out common mistakes to avoid
  - Write for JEE aspirants (clear, rigorous, motivating)
"""

from __future__ import annotations

import json
import logging
import re
import time

from backend.models import (
    AgentTraceEntry,
    Explanation,
    ParsedProblem,
    RAGContext,
    Solution,
    VerificationResult,
)
from backend.agents.llm_client import chat

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a friendly and encouraging JEE math teacher.

Your job is to explain a math solution in a way that a class 11/12 Indian student can easily understand.

Given:
- The problem
- The step-by-step solution
- Retrieved knowledge context

Generate an explanation that:
1. Explains the intuition behind the approach
2. Connects to key concepts / formulas
3. Points out common mistakes students make on this type of problem
4. Uses simple language with occasional LaTeX for formulas

Output ONLY valid JSON:
{
  "student_friendly_text": "<full explanation in markdown, use \\n for newlines>",
  "key_concepts": ["<concept1>", "<concept2>", ...],
  "common_mistakes": ["<mistake1>", "<mistake2>", ...]
}
"""


def _format_steps(solution: Solution) -> str:
    if not solution.steps:
        return f"Answer: {solution.final_answer}"
    lines = []
    for s in solution.steps:
        line = f"Step {s.step_number}: {s.description}"
        if s.symbolic_expr:
            line += f"  [{s.symbolic_expr}]"
        if s.result:
            line += f"  → {s.result}"
        lines.append(line)
    return "\n".join(lines)


def _build_messages(
    problem: ParsedProblem,
    solution: Solution,
    rag_context: RAGContext,
    verification: VerificationResult,
) -> list:
    # Select top 2 context chunks for brevity
    ctx_snippets = "\n".join(c.content[:200] for c in rag_context.chunks[:2])

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"PROBLEM:\n{problem.problem_text}\n\n"
                f"TOPIC: {problem.topic}\n\n"
                f"SOLUTION STEPS:\n{_format_steps(solution)}\n\n"
                f"FINAL ANSWER: {solution.final_answer}\n\n"
                f"VERIFICATION: {'Correct ✓' if verification.is_correct else 'Uncertain ⚠'}\n\n"
                f"RELEVANT KNOWLEDGE:\n{ctx_snippets}"
            ),
        },
    ]


# ── Agent function ────────────────────────────────────────────────────────────

def run_explainer_agent(
    parsed_problem: ParsedProblem,
    solution: Solution,
    rag_context: RAGContext,
    verification: VerificationResult,
) -> tuple[Explanation, AgentTraceEntry]:
    """
    Generate a student-friendly explanation of the solution.

    Args:
        parsed_problem: Parsed problem.
        solution:       Computed solution.
        rag_context:    Retrieved knowledge context.
        verification:   Verifier result.

    Returns:
        Tuple of (Explanation, AgentTraceEntry).
    """
    t_start = time.time()
    logger.info("[ExplainerAgent] Generating explanation…")

    try:
        messages = _build_messages(parsed_problem, solution, rag_context, verification)
        llm_response = chat(messages, temperature=0.4, max_tokens=1024)
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

        explanation = Explanation(
            student_friendly_text=data.get("student_friendly_text", ""),
            key_concepts=data.get("key_concepts", []),
            common_mistakes=data.get("common_mistakes", []),
        )

    except Exception as exc:
        logger.error("[ExplainerAgent] Failed: %s", exc)
        explanation = Explanation(
            student_friendly_text=(
                f"**{problem_text_fallback(parsed_problem)}**\n\n"
                f"Final answer: {solution.final_answer}\n\n"
                f"*(Detailed explanation temporarily unavailable.)*"
            ),
            key_concepts=[parsed_problem.topic.value],
            common_mistakes=[],
        )

    duration = (time.time() - t_start) * 1000
    trace = AgentTraceEntry(
        agent="ExplainerAgent",
        status="done",
        input_summary=f"Answer: {solution.final_answer[:40]}",
        output_summary=(
            f"Concepts: {len(explanation.key_concepts)}, "
            f"Mistakes: {len(explanation.common_mistakes)}"
        ),
        duration_ms=round(duration, 1),
    )
    logger.info("[ExplainerAgent] Done in %.0fms.", duration)
    return explanation, trace


def problem_text_fallback(parsed_problem: ParsedProblem) -> str:
    return parsed_problem.problem_text[:80]
