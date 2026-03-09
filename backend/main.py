"""
main.py — FastAPI backend for Multimodal Math Mentor.

Endpoints:
  POST /solve/text        — solve from plain text
  POST /solve/image       — solve from uploaded image (OCR)
  POST /solve/audio       — solve from uploaded audio (Whisper ASR)
  POST /hitl/respond      — submit HITL human response
  POST /feedback          — submit user feedback
  GET  /memory/recent     — retrieve recent session records
  GET  /health            — health check
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import API_HOST, API_PORT, LOG_LEVEL
from backend.models import (
    HITLResponse,
    InputMode,
    MathRequest,
    MathResponse,
    OCRResult,
    ASRResult,
    UserFeedback,
)
from backend.agents.orchestrator import run_pipeline
from backend.multimodal.ocr_processor import process_image
from backend.multimodal.audio_processor import process_audio
from backend.memory.memory_store import get_memory_store
from backend.hitl.hitl_manager import process_hitl_response
from backend.rag.retriever import get_store
from backend.rag.knowledge_base import build_knowledge_base

# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_LOG_LEVEL  = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

# Console handler
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

# File handler — always write logs to backend.log in the project root
_log_file = Path(__file__).resolve().parent.parent / "backend.log"
_file_handler = logging.handlers.RotatingFileHandler(
    _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

# Apply to root logger so every module's logger inherits
logging.basicConfig(level=_LOG_LEVEL, handlers=[_console_handler, _file_handler])

# Also capture uvicorn access/error logs
for _uv_logger in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _ul = logging.getLogger(_uv_logger)
    _ul.setLevel(_LOG_LEVEL)
    _ul.addHandler(_file_handler)

logger = logging.getLogger(__name__)
logger.info("Logging initialised — writing to console AND %s", _log_file)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Math Mentor API",
    description="JEE Math Solver with RAG, Multi-Agent, HITL, and Memory",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Build knowledge base index on startup and pre-warm the embedding model."""
    logger.info("Starting Multimodal Math Mentor API…")
    store = get_store()
    build_knowledge_base(store)
    logger.info("Knowledge base ready. Documents: %d", store.document_count())

    # Pre-warm the sentence-transformer so the first request isn't slow
    from backend.rag.embeddings import embed_query
    embed_query("warmup")
    logger.info("Embedding model pre-warmed.")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    store = get_store()
    return {
        "status": "ok",
        "knowledge_base_chunks": store.document_count(),
    }


# ── Preview endpoints (OCR / ASR only, no solving) ───────────────────────────

@app.post("/preview/image", tags=["Preview"])
async def preview_image_ocr(
    file: UploadFile = File(..., description="Image file (PNG/JPEG)"),
):
    """
    Run OCR on an uploaded image and return the extracted text + confidence.
    Does NOT run the full solver pipeline — useful for previewing extraction.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")
    logger.info("POST /preview/image — file: %s", file.filename)
    ocr_result = process_image(image_bytes)
    return {
        "extracted_text": ocr_result.extracted_text,
        "confidence":     ocr_result.confidence,
        "needs_hitl":     ocr_result.needs_hitl,
        "backend":        "gemini_vision" if ocr_result.confidence >= 0.90 else "easyocr",
    }


@app.post("/preview/audio", tags=["Preview"])
async def preview_audio_asr(
    file: UploadFile = File(..., description="Audio file (WAV/MP3/WebM)"),
):
    """
    Transcribe audio and return transcript + confidence.
    Does NOT run the full solver pipeline — useful for previewing transcription.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    ext = Path(file.filename or "audio.wav").suffix or ".wav"
    logger.info("POST /preview/audio — file: %s (%d bytes)", file.filename, len(audio_bytes))
    asr_result = process_audio(audio_bytes, file_extension=ext)
    return {
        "transcript":  asr_result.transcript,
        "confidence":  asr_result.confidence,
        "needs_hitl":  asr_result.needs_hitl,
    }


# ── Text endpoint ─────────────────────────────────────────────────────────────

@app.post("/solve/text", response_model=MathResponse, tags=["Solve"])
async def solve_text(
    problem: str = Form(..., description="Plain text math problem"),
    session_id: Optional[str] = Form(None),
):
    """
    Solve a math problem from plain text input.

    Args:
        problem:    The math problem text.
        session_id: Optional session identifier (auto-generated if omitted).
    """
    sid = session_id or str(uuid.uuid4())
    logger.info("POST /solve/text — session: %s", sid)

    request = MathRequest(
        session_id=sid,
        input_mode=InputMode.TEXT,
        raw_text=problem,
    )
    response = run_pipeline(request)
    return response


# ── Image endpoint ────────────────────────────────────────────────────────────

@app.post("/solve/image", response_model=MathResponse, tags=["Solve"])
async def solve_image(
    file: UploadFile = File(..., description="Image file (PNG/JPEG)"),
    session_id: Optional[str] = Form(None),
):
    """
    Extract text from an uploaded image via OCR, then solve.

    Triggers HITL if OCR confidence is below threshold.
    """
    sid = session_id or str(uuid.uuid4())
    logger.info("POST /solve/image — session: %s, file: %s", sid, file.filename)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded.")

    ocr_result: OCRResult = process_image(image_bytes)
    logger.info(
        "OCR result — text: '%s…', confidence: %.2f, hitl: %s",
        ocr_result.extracted_text[:40],
        ocr_result.confidence,
        ocr_result.needs_hitl,
    )

    request = MathRequest(
        session_id=sid,
        input_mode=InputMode.IMAGE,
        ocr_result=ocr_result,
    )
    response = run_pipeline(request)
    return response


# ── Audio endpoint ────────────────────────────────────────────────────────────

@app.post("/solve/audio", response_model=MathResponse, tags=["Solve"])
async def solve_audio(
    file: UploadFile = File(..., description="Audio file (WAV/MP3/M4A)"),
    session_id: Optional[str] = Form(None),
):
    """
    Transcribe audio via local Whisper, then solve the math problem.

    Triggers HITL if ASR confidence is below threshold.
    """
    sid = session_id or str(uuid.uuid4())
    logger.info("POST /solve/audio — session: %s, file: %s", sid, file.filename)

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

    # Determine extension for temp file
    ext = Path(file.filename or "audio.wav").suffix or ".wav"

    asr_result: ASRResult = process_audio(audio_bytes, file_extension=ext)
    logger.info(
        "ASR result — transcript: '%s…', confidence: %.2f, hitl: %s",
        asr_result.transcript[:40],
        asr_result.confidence,
        asr_result.needs_hitl,
    )

    request = MathRequest(
        session_id=sid,
        input_mode=InputMode.AUDIO,
        asr_result=asr_result,
    )
    response = run_pipeline(request)
    return response


# ── HITL endpoint ──────────────────────────────────────────────────────────────

@app.post("/hitl/respond", response_model=MathResponse, tags=["HITL"])
async def hitl_respond(
    session_id: str = Form(...),
    approved:   bool = Form(...),
    edited_text: Optional[str] = Form(None),
    comment:    Optional[str]  = Form(None),
):
    """
    Submit human-in-the-loop response and resume the pipeline.

    After a HITL pause, the user approves or edits the text.
    The corrected text is fed back into the pipeline.
    """
    logger.info("POST /hitl/respond — session: %s, approved: %s", session_id, approved)

    hitl_resp = HITLResponse(
        approved=approved,
        edited_text=edited_text,
        comment=comment,
    )
    final_text = process_hitl_response(session_id, hitl_resp)

    if not approved or not final_text:
        return MathResponse(session_id=session_id, hitl_required=False)

    # Resume pipeline with corrected text
    request = MathRequest(
        session_id=session_id,
        input_mode=InputMode.TEXT,
        user_corrected_text=final_text,
    )
    response = run_pipeline(request)
    return response


# ── Feedback endpoint ─────────────────────────────────────────────────────────

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback: UserFeedback):
    """
    Record user feedback (correct / incorrect) for a session.
    Stored in SQLite memory for future improvements.
    """
    logger.info(
        "POST /feedback — session: %s, type: %s",
        feedback.session_id,
        feedback.feedback_type,
    )
    store = get_memory_store()
    store.update_feedback(feedback)
    return {"status": "ok", "session_id": feedback.session_id}


# ── Memory endpoint ───────────────────────────────────────────────────────────

@app.get("/memory/recent", tags=["Memory"])
async def get_recent_memory(limit: int = 10):
    """
    Retrieve the most recent memory records.

    Args:
        limit: Maximum number of records to return.
    """
    store   = get_memory_store()
    records = store.get_recent(limit=limit)
    return [r.dict() for r in records]


# ── Logs endpoint ─────────────────────────────────────────────────────────────

@app.get("/logs", tags=["System"])
async def get_logs(lines: int = 100):
    """
    Return the last N lines from the backend log file.

    Args:
        lines: Number of tail lines to return (default 100).
    """
    log_file = Path(__file__).resolve().parent.parent / "backend.log"
    if not log_file.exists():
        return {"lines": [], "file": str(log_file), "exists": False}

    try:
        # Read tail efficiently for large files
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        tail = [l.rstrip("\n") for l in all_lines[-lines:]]
        return {
            "lines":  tail,
            "total":  len(all_lines),
            "file":   str(log_file),
            "exists": True,
        }
    except Exception as exc:
        return {"lines": [f"Error reading log: {exc}"], "exists": True}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower(),
    )
