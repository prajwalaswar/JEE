"""
models.py — Pydantic data models shared across the entire application.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timezone


# ── Enumerations ──────────────────────────────────────────────────────────────

class InputMode(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    AUDIO = "audio"


class MathTopic(str, Enum):
    ALGEBRA         = "algebra"
    CALCULUS        = "calculus"
    PROBABILITY     = "probability"
    LINEAR_ALGEBRA  = "linear_algebra"
    TRIGONOMETRY    = "trigonometry"
    COORDINATE      = "coordinate_geometry"
    UNKNOWN         = "unknown"


class HITLTriggerReason(str, Enum):
    LOW_OCR_CONFIDENCE  = "low_ocr_confidence"
    LOW_ASR_CONFIDENCE  = "low_asr_confidence"
    PARSER_AMBIGUITY    = "parser_ambiguity"
    VERIFIER_UNCERTAIN  = "verifier_uncertain"
    USER_REQUESTED      = "user_requested"


class FeedbackType(str, Enum):
    CORRECT   = "correct"
    INCORRECT = "incorrect"


# ── OCR / ASR intermediate results ────────────────────────────────────────────

class OCRResult(BaseModel):
    extracted_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_hitl: bool = False


class ASRResult(BaseModel):
    transcript: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_hitl: bool = False


# ── Problem representation ────────────────────────────────────────────────────

class ParsedProblem(BaseModel):
    problem_text: str
    topic: MathTopic = MathTopic.UNKNOWN
    variables: List[str] = []
    constraints: List[str] = []
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None


# ── RAG context ───────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    content:    str
    source:     str
    score:      float
    chunk_id:   str


class RAGContext(BaseModel):
    chunks:  List[RetrievedChunk] = []
    query:   str = ""


# ── Solution ──────────────────────────────────────────────────────────────────

class SolutionStep(BaseModel):
    step_number: int
    description: str
    symbolic_expr: Optional[str] = None
    result: Optional[str] = None


class Solution(BaseModel):
    final_answer:   str
    steps:          List[SolutionStep] = []
    confidence:     float = Field(ge=0.0, le=1.0, default=1.0)
    tool_used:      Optional[str] = None  # "sympy" | "llm" | "hybrid"


# ── Verification ──────────────────────────────────────────────────────────────

class VerificationResult(BaseModel):
    is_correct:    bool
    confidence:    float = Field(ge=0.0, le=1.0)
    issues:        List[str] = []
    needs_hitl:    bool = False
    hitl_reason:   Optional[HITLTriggerReason] = None


# ── Explanation ───────────────────────────────────────────────────────────────

class Explanation(BaseModel):
    student_friendly_text:  str
    key_concepts:           List[str] = []
    common_mistakes:        List[str] = []


# ── Agent trace (for UI display) ──────────────────────────────────────────────

class AgentTraceEntry(BaseModel):
    agent:      str
    status:     str           # "running" | "done" | "hitl_triggered"
    input_summary:  str = ""
    output_summary: str = ""
    duration_ms:    Optional[float] = None


# ── HITL ──────────────────────────────────────────────────────────────────────

class HITLRequest(BaseModel):
    session_id:     str
    reason:         HITLTriggerReason
    current_text:   str           # the text that needs human review
    prompt_message: str


class HITLResponse(BaseModel):
    approved:       bool
    edited_text:    Optional[str] = None
    comment:        Optional[str] = None


# ── Full pipeline request / response ─────────────────────────────────────────

class MathRequest(BaseModel):
    session_id:     str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_mode:     InputMode
    raw_text:       Optional[str] = None
    # For image / audio, the backend receives bytes via multipart upload
    # These fields store post-processing results
    ocr_result:     Optional[OCRResult]  = None
    asr_result:     Optional[ASRResult]  = None
    # Optional override from HITL edit
    user_corrected_text: Optional[str]  = None


class MathResponse(BaseModel):
    session_id:         str
    parsed_problem:     Optional[ParsedProblem]     = None
    rag_context:        Optional[RAGContext]         = None
    solution:           Optional[Solution]           = None
    verification:       Optional[VerificationResult] = None
    explanation:        Optional[Explanation]        = None
    agent_trace:        List[AgentTraceEntry]        = []
    hitl_required:      bool = False
    hitl_request:       Optional[HITLRequest]        = None
    created_at:         str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Feedback ──────────────────────────────────────────────────────────────────

class UserFeedback(BaseModel):
    session_id:     str
    feedback_type:  FeedbackType
    comment:        Optional[str] = None


# ── Memory record ─────────────────────────────────────────────────────────────

class MemoryRecord(BaseModel):
    record_id:          str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id:         str
    original_input:     str
    parsed_problem:     Optional[Dict[str, Any]] = None
    retrieved_context:  Optional[Dict[str, Any]] = None
    solution:           Optional[Dict[str, Any]] = None
    verifier_result:    Optional[Dict[str, Any]] = None
    user_feedback:      Optional[str]             = None
    feedback_comment:   Optional[str]             = None
    created_at:         str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
