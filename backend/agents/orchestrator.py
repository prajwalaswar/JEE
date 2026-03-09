"""
orchestrator.py — LangGraph-based multi-agent pipeline orchestration.

Graph structure (HITL implemented as proper state-modifying nodes):
  START
    │
    ▼
  [parse]               ← Parser Agent
    │
    ▼
  [input_hitl_check]    ← sets hitl_required / hitl_request in state
    │
    ├─ hitl_required=True  ──→ [hitl_gate] ──→ END
    │
    ▼
  [route]               ← Intent Router Agent
    │
    ▼
  [solve]               ← Solver Agent (RAG + SymPy + LLM)
    │
    ▼
  [verify]              ← Verifier Agent
    │
    ▼
  [verify_hitl_check]   ← sets hitl_required / hitl_request in state
    │
    ├─ hitl_required=True  ──→ [hitl_gate] ──→ END
    │
    ▼
  [explain]             ← Explainer Agent
    │
    ▼
  [save_memory]         ← SQLite persistence
    │
    ▼
  END

IMPORTANT: Conditional edge functions are pure routing functions (read-only).
All state mutations happen inside proper node functions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from backend.models import (
    AgentTraceEntry,
    Explanation,
    HITLRequest,
    HITLTriggerReason,
    MathRequest,
    MathResponse,
    MemoryRecord,
    ParsedProblem,
    RAGContext,
    Solution,
    VerificationResult,
)
from backend.agents.parser_agent       import run_parser_agent
from backend.agents.intent_router_agent import run_intent_router_agent, SolvingStrategy
from backend.agents.solver_agent       import run_solver_agent
from backend.agents.verifier_agent     import run_verifier_agent
from backend.agents.explainer_agent    import run_explainer_agent
from backend.hitl.hitl_manager        import create_hitl_request
from backend.memory.memory_store      import get_memory_store
from backend.config import OCR_CONFIDENCE_THRESHOLD, ASR_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


# ── State definition ─────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """Shared state object passed between all graph nodes."""
    # Input
    session_id:       str
    raw_text:         str
    ocr_confidence:   float    # 1.0 if not from OCR
    asr_confidence:   float    # 1.0 if not from audio

    # In-flight
    parsed_problem:   Optional[ParsedProblem]
    strategy:         Optional[SolvingStrategy]
    techniques:       List[str]
    rag_context:      Optional[RAGContext]
    solution:         Optional[Solution]
    verification:     Optional[VerificationResult]
    explanation:      Optional[Explanation]

    # Control
    hitl_required:    bool
    hitl_reason:      Optional[str]   # NEW: human-readable reason for HITL
    hitl_request:     Optional[HITLRequest]
    agent_trace:      List[AgentTraceEntry]
    error:            Optional[str]


# ── Node functions ────────────────────────────────────────────────────────────

def node_parse(state: PipelineState) -> PipelineState:
    """Run the Parser Agent."""
    logger.info("[Graph] node_parse")
    parsed, trace = run_parser_agent(
        raw_text=state["raw_text"],
        session_id=state["session_id"],
    )
    state["parsed_problem"] = parsed
    state["agent_trace"].append(trace)
    return state


def node_input_hitl_check(state: PipelineState) -> PipelineState:
    """
    PROPER STATE-MODIFYING NODE — decides if input needs human review.
    Sets state["hitl_required"], state["hitl_reason"], state["hitl_request"].
    Must be a node (not an edge function) so mutations are actually saved.
    """
    parsed = state["parsed_problem"]
    ocr_c  = state.get("ocr_confidence", 1.0)
    asr_c  = state.get("asr_confidence", 1.0)

    # Start clean
    state["hitl_required"] = False
    state["hitl_reason"]   = None
    state["hitl_request"]  = None

    if ocr_c < OCR_CONFIDENCE_THRESHOLD:
        req = create_hitl_request(
            state["session_id"],
            HITLTriggerReason.LOW_OCR_CONFIDENCE,
            state["raw_text"],
        )
        state["hitl_required"] = True
        state["hitl_reason"]   = HITLTriggerReason.LOW_OCR_CONFIDENCE.value
        state["hitl_request"]  = req
        logger.info("[Graph] HITL triggered: low OCR confidence (%.2f)", ocr_c)
        return state

    if asr_c < ASR_CONFIDENCE_THRESHOLD:
        req = create_hitl_request(
            state["session_id"],
            HITLTriggerReason.LOW_ASR_CONFIDENCE,
            state["raw_text"],
        )
        state["hitl_required"] = True
        state["hitl_reason"]   = HITLTriggerReason.LOW_ASR_CONFIDENCE.value
        state["hitl_request"]  = req
        logger.info("[Graph] HITL triggered: low ASR confidence (%.2f)", asr_c)
        return state

    if parsed and parsed.needs_clarification:
        req = create_hitl_request(
            state["session_id"],
            HITLTriggerReason.PARSER_AMBIGUITY,
            parsed.problem_text,
        )
        state["hitl_required"] = True
        state["hitl_reason"]   = HITLTriggerReason.PARSER_AMBIGUITY.value
        state["hitl_request"]  = req
        logger.info("[Graph] HITL triggered: parser ambiguity")

    return state


def _route_after_input_check(state: PipelineState) -> str:
    """Pure read-only routing function — used by add_conditional_edges."""
    return "hitl_gate" if state.get("hitl_required") else "route"


def node_hitl_gate(state: PipelineState) -> PipelineState:
    """
    HITL gate node — pipeline pauses here and returns to caller.
    The UI will display the HITL request and resume via the API.
    """
    logger.info("[Graph] node_hitl_gate — pipeline paused for HITL.")
    return state


def node_route(state: PipelineState) -> PipelineState:
    """Run the Intent Router Agent."""
    logger.info("[Graph] node_route")
    if not state["parsed_problem"]:
        state["error"] = "No parsed problem available for routing."
        return state

    updated_problem, strategy, techniques, trace = run_intent_router_agent(
        state["parsed_problem"]
    )
    state["parsed_problem"] = updated_problem
    state["strategy"]       = strategy
    state["techniques"]     = techniques
    state["agent_trace"].append(trace)
    return state


def node_solve(state: PipelineState) -> PipelineState:
    """Run the Solver Agent."""
    logger.info("[Graph] node_solve")
    if not state["parsed_problem"] or not state["strategy"]:
        state["error"] = "Cannot solve: missing parsed problem or strategy."
        return state

    solution, rag_ctx, trace = run_solver_agent(
        parsed_problem=state["parsed_problem"],
        strategy=state["strategy"],
        rag_context=state.get("rag_context"),
    )
    state["solution"]    = solution
    state["rag_context"] = rag_ctx
    state["agent_trace"].append(trace)
    return state


def node_verify(state: PipelineState) -> PipelineState:
    """Run the Verifier Agent."""
    logger.info("[Graph] node_verify")
    if not state["parsed_problem"] or not state["solution"]:
        state["error"] = "Cannot verify: missing problem or solution."
        return state

    verification, trace = run_verifier_agent(
        parsed_problem=state["parsed_problem"],
        solution=state["solution"],
    )
    state["verification"] = verification
    state["agent_trace"].append(trace)
    return state


def node_verify_hitl_check(state: PipelineState) -> PipelineState:
    """
    PROPER STATE-MODIFYING NODE — trigger HITL if verifier is uncertain.
    Resets hitl_required before deciding (input HITL already cleared if here).
    """
    state["hitl_required"] = False
    state["hitl_reason"]   = None
    state["hitl_request"]  = None

    verification = state.get("verification")
    if verification and verification.needs_hitl:
        solution_text = state["solution"].final_answer if state.get("solution") else ""
        issues_text   = "; ".join(verification.issues) if verification.issues else "low confidence"
        req = create_hitl_request(
            state["session_id"],
            HITLTriggerReason.VERIFIER_UNCERTAIN,
            f"Proposed answer: {solution_text}\nVerifier issues: {issues_text}",
        )
        state["hitl_required"] = True
        state["hitl_reason"]   = HITLTriggerReason.VERIFIER_UNCERTAIN.value
        state["hitl_request"]  = req
        logger.info(
            "[Graph] HITL triggered: verifier uncertain (confidence=%.2f)",
            verification.confidence,
        )
    return state


def _route_after_verify(state: PipelineState) -> str:
    """Pure read-only routing function after verify_hitl_check node."""
    return "hitl_gate" if state.get("hitl_required") else "explain"


def node_explain(state: PipelineState) -> PipelineState:
    """Run the Explainer Agent."""
    logger.info("[Graph] node_explain")
    if not all([
        state.get("parsed_problem"),
        state.get("solution"),
        state.get("rag_context"),
        state.get("verification"),
    ]):
        logger.warning("[Graph] Skipping explainer — required state not available.")
        return state

    explanation, trace = run_explainer_agent(
        parsed_problem=state["parsed_problem"],
        solution=state["solution"],
        rag_context=state["rag_context"],
        verification=state["verification"],
    )
    state["explanation"] = explanation
    state["agent_trace"].append(trace)
    return state


def node_save_memory(state: PipelineState) -> PipelineState:
    """Persist the full pipeline run to SQLite."""
    logger.info("[Graph] node_save_memory  session=%s", state["session_id"])
    try:
        store = get_memory_store()
        record = MemoryRecord(
            session_id=state["session_id"],
            original_input=state["raw_text"],
            parsed_problem=(
                state["parsed_problem"].dict() if state.get("parsed_problem") else None
            ),
            retrieved_context=(
                state["rag_context"].dict() if state.get("rag_context") else None
            ),
            solution=(
                state["solution"].dict() if state.get("solution") else None
            ),
            verifier_result=(
                state["verification"].dict() if state.get("verification") else None
            ),
        )
        store.save(record)
        logger.info("[Graph] Memory record saved for session: %s", state["session_id"])
    except Exception as exc:
        logger.error("[Graph] Memory save failed: %s", exc)
    return state


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Construct and compile the LangGraph pipeline.

    Node layout:
      parse → input_hitl_check → (hitl_gate | route)
      route → solve → verify → verify_hitl_check → (hitl_gate | explain)
      explain → save_memory → END
      hitl_gate → END
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("parse",             node_parse)
    graph.add_node("input_hitl_check",  node_input_hitl_check)
    graph.add_node("hitl_gate",         node_hitl_gate)
    graph.add_node("route",             node_route)
    graph.add_node("solve",             node_solve)
    graph.add_node("verify",            node_verify)
    graph.add_node("verify_hitl_check", node_verify_hitl_check)
    graph.add_node("explain",           node_explain)
    graph.add_node("save_memory",       node_save_memory)

    # Entry point
    graph.set_entry_point("parse")

    # parse always goes to input_hitl_check (state-mutating node)
    graph.add_edge("parse", "input_hitl_check")

    # input_hitl_check → conditional routing (pure function, read-only)
    graph.add_conditional_edges(
        "input_hitl_check",
        _route_after_input_check,
        {"hitl_gate": "hitl_gate", "route": "route"},
    )

    # HITL gate is a terminal node
    graph.add_edge("hitl_gate", END)

    # Normal solve pipeline
    graph.add_edge("route",  "solve")
    graph.add_edge("solve",  "verify")
    graph.add_edge("verify", "verify_hitl_check")

    # verify_hitl_check → conditional routing (pure function, read-only)
    graph.add_conditional_edges(
        "verify_hitl_check",
        _route_after_verify,
        {"hitl_gate": "hitl_gate", "explain": "explain"},
    )

    graph.add_edge("explain",     "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


# ── Public pipeline runner ────────────────────────────────────────────────────

# Compiled graph singleton
_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(request: MathRequest) -> MathResponse:
    """
    Execute the full multi-agent pipeline for a math request.

    Args:
        request: MathRequest with input mode, raw text / OCR / ASR results.

    Returns:
        MathResponse with solution, explanation, trace, and HITL info.
    """
    # Determine effective input text
    raw_text = ""
    ocr_conf = 1.0
    asr_conf = 1.0

    if request.user_corrected_text:
        raw_text = request.user_corrected_text
    elif request.ocr_result:
        raw_text = request.ocr_result.extracted_text
        ocr_conf = request.ocr_result.confidence
    elif request.asr_result:
        raw_text = request.asr_result.transcript
        asr_conf = request.asr_result.confidence
    elif request.raw_text:
        raw_text = request.raw_text

    if not raw_text.strip():
        return MathResponse(
            session_id=request.session_id,
            hitl_required=False,
        )

    # Initial state
    initial_state: PipelineState = {
        "session_id":     request.session_id,
        "raw_text":       raw_text,
        "ocr_confidence": ocr_conf,
        "asr_confidence": asr_conf,
        "parsed_problem": None,
        "strategy":       None,
        "techniques":     [],
        "rag_context":    None,
        "solution":       None,
        "verification":   None,
        "explanation":    None,
        "hitl_required":  False,
        "hitl_reason":    None,
        "hitl_request":   None,
        "agent_trace":    [],
        "error":          None,
    }

    graph     = get_compiled_graph()
    final_state: PipelineState = graph.invoke(initial_state)

    return MathResponse(
        session_id=request.session_id,
        parsed_problem=final_state.get("parsed_problem"),
        rag_context=final_state.get("rag_context"),
        solution=final_state.get("solution"),
        verification=final_state.get("verification"),
        explanation=final_state.get("explanation"),
        agent_trace=final_state.get("agent_trace", []),
        hitl_required=final_state.get("hitl_required", False),
        hitl_request=final_state.get("hitl_request"),
    )
