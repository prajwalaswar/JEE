"""
math_tools.py — SymPy-powered math computation tools.
Used by the Solver Agent to get exact symbolic and numeric answers.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

logger = logging.getLogger(__name__)

# Parser transformations — allow implicit multiplication like "2x"
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
)


def _prep(expr: str) -> str:
    """Normalise expression string before SymPy parsing.

    Converts caret ``^`` (bitwise XOR in Python) to ``**`` so that
    user input like ``x^2`` or ``x^2 - 4x + 4 = 0`` is parsed correctly.
    """
    return expr.replace("^", "**").strip()


# ── Public Tool Functions ─────────────────────────────────────────────────────

def solve_equation(equation_str: str, variable: str = "x") -> Dict[str, Any]:
    """
    Solve an algebraic equation symbolically.

    Args:
        equation_str: Equation as string, e.g. "x**2 - 5*x + 6 = 0"
        variable:     Variable name to solve for.

    Returns:
        dict with keys: success, solutions, latex, error
    """
    try:
        var = sp.Symbol(variable)

        # Handle equations with or without "="
        if "=" in equation_str:
            lhs_str, rhs_str = equation_str.split("=", 1)
            lhs = parse_expr(_prep(lhs_str), transformations=_TRANSFORMATIONS)
            rhs = parse_expr(_prep(rhs_str), transformations=_TRANSFORMATIONS)
            equation = sp.Eq(lhs, rhs)
        else:
            # Assume expression = 0
            expr = parse_expr(
                _prep(equation_str), transformations=_TRANSFORMATIONS
            )
            equation = sp.Eq(expr, 0)

        solutions = sp.solve(equation, var)
        solutions_str = [str(sol) for sol in solutions]
        solutions_latex = [sp.latex(sol) for sol in solutions]

        return {
            "success":         True,
            "solutions":       solutions_str,
            "solutions_latex": solutions_latex,
            "latex":           sp.latex(equation),
            "error":           None,
        }

    except Exception as exc:
        logger.error("solve_equation error: %s", exc)
        return {"success": False, "solutions": [], "latex": "", "error": str(exc)}


def differentiate(expr_str: str, variable: str = "x", order: int = 1) -> Dict[str, Any]:
    """
    Compute the derivative of an expression.

    Args:
        expr_str: Expression string, e.g. "x**3 + sin(x)"
        variable: Variable to differentiate with respect to.
        order:    Derivative order.

    Returns:
        dict with keys: success, derivative, latex, error
    """
    try:
        var  = sp.Symbol(variable)
        expr = parse_expr(_prep(expr_str), transformations=_TRANSFORMATIONS)
        deriv = sp.diff(expr, var, order)
        simplified = sp.simplify(deriv)
        return {
            "success":    True,
            "derivative": str(simplified),
            "latex":      sp.latex(simplified),
            "error":      None,
        }
    except Exception as exc:
        logger.error("differentiate error: %s", exc)
        return {"success": False, "derivative": "", "latex": "", "error": str(exc)}


def integrate_expression(
    expr_str: str,
    variable: str = "x",
    lower: Optional[str] = None,
    upper: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute definite or indefinite integral.

    Args:
        expr_str: Expression string.
        variable: Integration variable.
        lower:    Lower bound (if definite).
        upper:    Upper bound (if definite).

    Returns:
        dict with keys: success, result, latex, error
    """
    try:
        var  = sp.Symbol(variable)
        expr = parse_expr(_prep(expr_str), transformations=_TRANSFORMATIONS)

        if lower is not None and upper is not None:
            lo  = parse_expr(_prep(lower), transformations=_TRANSFORMATIONS)
            hi  = parse_expr(_prep(upper), transformations=_TRANSFORMATIONS)
            result = sp.integrate(expr, (var, lo, hi))
        else:
            result = sp.integrate(expr, var)

        simplified = sp.simplify(result)
        return {
            "success": True,
            "result":  str(simplified),
            "latex":   sp.latex(simplified),
            "error":   None,
        }
    except Exception as exc:
        logger.error("integrate_expression error: %s", exc)
        return {"success": False, "result": "", "latex": "", "error": str(exc)}


def compute_limit(
    expr_str: str,
    variable: str = "x",
    point: str = "0",
    direction: str = "+",
) -> Dict[str, Any]:
    """
    Evaluate a symbolic limit.

    Args:
        expr_str:  Expression string.
        variable:  Variable name.
        point:     Point to approach (default "0").
        direction: "+" (right), "-" (left), or "+-" (both).

    Returns:
        dict with keys: success, limit, latex, error
    """
    try:
        var  = sp.Symbol(variable)
        expr = parse_expr(_prep(expr_str), transformations=_TRANSFORMATIONS)
        pt   = parse_expr(_prep(point), transformations=_TRANSFORMATIONS)
        lim  = sp.limit(expr, var, pt, direction)
        return {
            "success": True,
            "limit":   str(lim),
            "latex":   sp.latex(lim),
            "error":   None,
        }
    except Exception as exc:
        logger.error("compute_limit error: %s", exc)
        return {"success": False, "limit": "", "latex": "", "error": str(exc)}


def evaluate_expression(expr_str: str, substitutions: Dict[str, float] = {}) -> Dict[str, Any]:
    """
    Numerically evaluate an expression after substituting variable values.

    Args:
        expr_str:      Expression string.
        substitutions: Dict mapping variable names to numeric values.

    Returns:
        dict with keys: success, value, error
    """
    try:
        symbols_map = {k: sp.Symbol(k) for k in substitutions}
        expr = parse_expr(
            _prep(expr_str),
            local_dict=symbols_map,
            transformations=_TRANSFORMATIONS,
        )
        result = expr.subs(
            {sp.Symbol(k): v for k, v in substitutions.items()}
        )
        numeric = float(sp.N(result))
        return {"success": True, "value": numeric, "error": None}
    except Exception as exc:
        logger.error("evaluate_expression error: %s", exc)
        return {"success": False, "value": None, "error": str(exc)}


def simplify_expression(expr_str: str) -> Dict[str, Any]:
    """
    Simplify a symbolic expression.

    Args:
        expr_str: Expression string.

    Returns:
        dict with keys: success, simplified, latex, error
    """
    try:
        expr = parse_expr(_prep(expr_str), transformations=_TRANSFORMATIONS)
        simplified = sp.simplify(expr)
        return {
            "success":    True,
            "simplified": str(simplified),
            "latex":      sp.latex(simplified),
            "error":      None,
        }
    except Exception as exc:
        logger.error("simplify_expression error: %s", exc)
        return {"success": False, "simplified": "", "latex": "", "error": str(exc)}


def factor_expression(expr_str: str) -> Dict[str, Any]:
    """
    Factorise an expression.

    Args:
        expr_str: Expression string.

    Returns:
        dict with keys: success, factored, latex, error
    """
    try:
        expr    = parse_expr(_prep(expr_str), transformations=_TRANSFORMATIONS)
        factored = sp.factor(expr)
        return {
            "success": True,
            "factored": str(factored),
            "latex":    sp.latex(factored),
            "error":    None,
        }
    except Exception as exc:
        logger.error("factor_expression error: %s", exc)
        return {"success": False, "factored": "", "latex": "", "error": str(exc)}


def expand_expression(expr_str: str) -> Dict[str, Any]:
    """
    Expand a symbolic expression.

    Args:
        expr_str: Expression string.

    Returns:
        dict with keys: success, expanded, latex, error
    """
    try:
        expr     = parse_expr(_prep(expr_str), transformations=_TRANSFORMATIONS)
        expanded = sp.expand(expr)
        return {
            "success":  True,
            "expanded": str(expanded),
            "latex":    sp.latex(expanded),
            "error":    None,
        }
    except Exception as exc:
        logger.error("expand_expression error: %s", exc)
        return {"success": False, "expanded": "", "latex": "", "error": str(exc)}


# ── Dispatcher ────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "solve_equation":       solve_equation,
    "differentiate":        differentiate,
    "integrate_expression": integrate_expression,
    "compute_limit":        compute_limit,
    "evaluate_expression":  evaluate_expression,
    "simplify_expression":  simplify_expression,
    "factor_expression":    factor_expression,
    "expand_expression":    expand_expression,
}


def run_math_tool(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Dispatcher: call a named math tool with keyword arguments.

    Args:
        tool_name: One of the registered tool names.
        **kwargs:  Arguments forwarded to the tool function.

    Returns:
        Tool result dict.
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {
            "success": False,
            "error":   f"Unknown tool '{tool_name}'. "
                       f"Available: {list(TOOL_REGISTRY.keys())}",
        }
    return fn(**kwargs)
