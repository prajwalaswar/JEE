"""
solver_agent.py — Agent 3: Solver Agent

Responsibilities:
  - Uses RAG context for relevant formulas / patterns
  - Invokes SymPy tools for exact calculation
  - Generates a step-by-step solution
  - Combines symbolic and LLM reasoning (hybrid mode)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional

from backend.models import (
    AgentTraceEntry,
    ParsedProblem,
    RAGContext,
    Solution,
    SolutionStep,
)
from backend.agents.llm_client import chat
from backend.agents.intent_router_agent import SolvingStrategy
from backend.rag.retriever import retrieve, format_context_for_llm
from backend.tools.math_tools import run_math_tool

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert JEE mathematics tutor and solver.

You will receive:
1. A structured math problem
2. Relevant knowledge base context (formulas, identities, patterns)
3. (Optionally) pre-computed symbolic results from SymPy tools

Your task: Solve the problem STEP BY STEP as a strict JSON response.

Output ONLY valid JSON:
{
  "final_answer": "<concise final answer>",
  "steps": [
    {
      "step_number": 1,
      "description": "<what is being done>",
      "symbolic_expr": "<LaTeX expression if applicable, else null>",
      "result": "<result of this step>"
    }
  ],
  "confidence": <float 0–1>,
  "tool_used": "<sympy|llm|hybrid>"
}

Rules:
- Be mathematically rigorous
- Use the knowledge base context where applicable
- If a SymPy result is provided, verify and use it
- Confidence < 0.8 means you're uncertain
- ONLY output valid JSON. No extra text.
"""


def _attempt_sympy_solve(problem: ParsedProblem) -> Optional[dict]:
    """
    Heuristically attempt to run SymPy on the problem.
    Returns SymPy result dict, or None if not applicable.
    """
    text = problem.problem_text.lower()

    # Simple heuristics based on keywords
    if any(kw in text for kw in ["solve", "roots", "equation", "find x", "find y"]):
        # Try solve_equation — extract the equation part
        eq_match = re.search(r"[a-z0-9\s\+\-\*\/\^\(\)=]+=[=0-9\s\+\-\*\/\^\(\)]+", problem.problem_text)
        if eq_match:
            eq_str = eq_match.group(0).strip()
            var = problem.variables[0] if problem.variables else "x"
            result = run_math_tool("solve_equation", equation_str=eq_str, variable=var)
            if result["success"]:
                return result

    elif any(kw in text for kw in ["differentiate", "derivative", "d/dx", "dy/dx"]):
        expr_match = re.search(r"of\s+(.+?)(?:\s+with|\s+at|\s+w\.r\.t|$)", text)
        if expr_match:
            expr_str = expr_match.group(1).strip()
            var = problem.variables[0] if problem.variables else "x"
            result = run_math_tool("differentiate", expr_str=expr_str, variable=var)
            if result["success"]:
                return result

    elif any(kw in text for kw in ["integrate", "integral", "∫"]):
        expr_match = re.search(r"of\s+(.+?)(?:\s+from|\s+with|\s+dx|$)", text)
        if expr_match:
            expr_str = expr_match.group(1).strip()
            result = run_math_tool("integrate_expression", expr_str=expr_str)
            if result["success"]:
                return result

    elif any(kw in text for kw in ["simplify", "factorise", "factor"]):
        # Try the whole problem text as the expression
        expr = re.sub(r"[Ss]implify|[Ff]actoris[e]?|[Ff]actor", "", problem.problem_text).strip()
        result = run_math_tool("simplify_expression", expr_str=expr)
        if result["success"]:
            return result

    return None


def run_solver_agent(
    parsed_problem: ParsedProblem,
    strategy: SolvingStrategy,
    rag_context: Optional[RAGContext] = None,
) -> tuple[Solution, RAGContext, AgentTraceEntry]:
    """
    Solve the parsed math problem using RAG + SymPy + LLM.

    Args:
        parsed_problem: Structured problem from Parser Agent.
        strategy:       Solving strategy from Intent Router.
        rag_context:    Pre-retrieved RAG context (optional; retrieved here if None).

    Returns:
        Tuple of (Solution, RAGContext, AgentTraceEntry).
    """
    t_start = time.time()
    logger.info("[SolverAgent] Strategy: %s", strategy)

    # ── Step 1: RAG retrieval ─────────────────────────────────────────────────
    if rag_context is None:
        rag_context = retrieve(parsed_problem.problem_text)
    formatted_context = format_context_for_llm(rag_context)

    # ── Step 2: SymPy attempt (for symbolic / hybrid modes) ───────────────────
    sympy_result: Optional[dict] = None
    if strategy in (SolvingStrategy.SYMBOLIC, SolvingStrategy.HYBRID):
        sympy_result = _attempt_sympy_solve(parsed_problem)
        if sympy_result:
            logger.info("[SolverAgent] SymPy succeeded: %s", sympy_result)

    # ── Step 3: Build LLM prompt ──────────────────────────────────────────────
    user_parts = [
        f"PROBLEM:\n{parsed_problem.problem_text}",
        f"\nTOPIC: {parsed_problem.topic}",
        f"\nVARIABLES: {parsed_problem.variables}",
        f"\nCONSTRAINTS: {parsed_problem.constraints}",
        f"\n{formatted_context}",
    ]
    if sympy_result:
        user_parts.append(
            f"\nSYMPY PRE-COMPUTED RESULT:\n{json.dumps(sympy_result, indent=2)}"
        )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": "\n".join(user_parts)},
    ]

    # ── Step 4: LLM solve ─────────────────────────────────────────────────────
    try:
        llm_response = chat(messages, temperature=0.1, max_tokens=2048)
        json_str = re.sub(r"```(?:json)?|```", "", llm_response).strip()
        data = json.loads(json_str)

        steps = [
            SolutionStep(
                step_number=s.get("step_number", i + 1),
                description=s.get("description", ""),
                symbolic_expr=s.get("symbolic_expr"),
                result=s.get("result"),
            )
            for i, s in enumerate(data.get("steps", []))
        ]

        solution = Solution(
            final_answer=data.get("final_answer", "No answer found."),
            steps=steps,
            confidence=float(data.get("confidence", 0.8)),
            tool_used=data.get("tool_used", strategy.value),
        )

    except Exception as exc:
        logger.error("[SolverAgent] LLM solve failed: %s", exc)
        # Fallback: use SymPy result if available
        if sympy_result and sympy_result.get("solutions"):
            sol_str = str(sympy_result["solutions"])
            solution = Solution(
                final_answer=sol_str,
                steps=[SolutionStep(step_number=1, description="Solved using SymPy", result=sol_str)],
                confidence=0.85,
                tool_used="sympy",
            )
        else:
            solution = Solution(
                final_answer="Unable to compute solution.",
                confidence=0.0,
                tool_used="error",
            )

    duration = (time.time() - t_start) * 1000
    trace = AgentTraceEntry(
        agent="SolverAgent",
        status="done",
        input_summary=parsed_problem.problem_text[:80],
        output_summary=(
            f"Answer: {solution.final_answer[:60]}, "
            f"Steps: {len(solution.steps)}, Conf: {solution.confidence}"
        ),
        duration_ms=round(duration, 1),
    )
    logger.info(
        "[SolverAgent] Done in %.0fms. Answer: %s",
        duration,
        solution.final_answer[:60],
    )
    return solution, rag_context, trace
