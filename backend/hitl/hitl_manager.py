"""
hitl_manager.py — Human-in-the-Loop (HITL) management layer.

Responsibilities:
  1. Build HITLRequest objects when a pipeline stage has low confidence.
  2. Process HITLResponse (approve / edit / reject) and store corrections.
  3. Retrieve past human corrections to auto-fix recurring OCR errors.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from backend.models import (
    HITLRequest,
    HITLResponse,
    HITLTriggerReason,
    MemoryRecord,
)
from backend.memory.memory_store import get_memory_store

logger = logging.getLogger(__name__)

# In-memory store of pending HITL requests keyed by session_id.
# In production this would be stored in Redis or a database.
_pending: Dict[str, HITLRequest] = {}


# ── Creating HITL requests ────────────────────────────────────────────────────

def create_hitl_request(
    session_id: str,
    reason: HITLTriggerReason,
    current_text: str,
) -> HITLRequest:
    """
    Build and register a HITL request for a session.

    Args:
        session_id:   Session identifier.
        reason:       Why HITL was triggered.
        current_text: The text that needs human review / correction.

    Returns:
        HITLRequest ready to be sent to the UI.
    """
    prompt_messages = {
        HITLTriggerReason.LOW_OCR_CONFIDENCE: (
            "The OCR confidence is low. Please review the extracted text "
            "and correct any errors before we solve the problem."
        ),
        HITLTriggerReason.LOW_ASR_CONFIDENCE: (
            "The speech-to-text confidence is low. Please review the "
            "transcript and edit if necessary."
        ),
        HITLTriggerReason.PARSER_AMBIGUITY: (
            "The problem parser detected ambiguity. "
            "Please clarify the problem statement."
        ),
        HITLTriggerReason.VERIFIER_UNCERTAIN: (
            "The verifier agent is not confident about the solution. "
            "Would you like to approve it, or request a re-solve?"
        ),
        HITLTriggerReason.USER_REQUESTED: (
            "You requested a human check. "
            "Please review and approve or edit the content below."
        ),
    }

    req = HITLRequest(
        session_id=session_id,
        reason=reason,
        current_text=current_text,
        prompt_message=prompt_messages.get(
            reason, "Please review the following and confirm or edit."
        ),
    )
    _pending[session_id] = req
    logger.info("HITL request created for session %s — reason: %s", session_id, reason.value)
    return req


def get_pending_request(session_id: str) -> Optional[HITLRequest]:
    """Return any pending HITL request for a session."""
    return _pending.get(session_id)


# ── Processing HITL responses ─────────────────────────────────────────────────

def process_hitl_response(
    session_id: str,
    response: HITLResponse,
) -> str:
    """
    Process the human's HITL response and persist the correction.

    Args:
        session_id: Session identifier.
        response:   The human's decision (approve / edit / reject).

    Returns:
        The final text to use downstream (original if approved, edited if changed).
    """
    req = _pending.pop(session_id, None)
    if req is None:
        logger.warning("No pending HITL for session %s", session_id)
        return ""

    if not response.approved:
        logger.info("HITL rejected for session %s.", session_id)
        return ""  # Caller should handle rejection (e.g., prompt user to re-upload)

    final_text = response.edited_text if response.edited_text else req.current_text

    # Persist correction to memory
    store = get_memory_store()
    existing = store.get_by_session(session_id)
    if existing:
        existing.feedback_comment = (
            f"[HITL correction: {req.reason.value}] {response.comment or ''}"
        )
        store.save(existing)

    logger.info(
        "HITL accepted for session %s. Text length: %d → %d",
        session_id,
        len(req.current_text),
        len(final_text),
    )
    return final_text


# ── Auto-correction from memory ───────────────────────────────────────────────

def suggest_correction(raw_text: str) -> Optional[str]:
    """
    Check memory for prior human corrections to similar text.
    Returns a suggested corrected version if found.

    Args:
        raw_text: The raw (potentially mis-recognised) text.

    Returns:
        Corrected text suggestion, or None if no match found.
    """
    store = get_memory_store()
    similar = store.get_similar(raw_text, limit=3)

    for record in similar:
        if record.feedback_comment and "HITL correction" in (record.feedback_comment or ""):
            # In a full implementation: use edit-distance to align and apply correction
            logger.info(
                "Found prior HITL correction for session %s — suggesting.",
                record.session_id,
            )
            return record.original_input  # naive: return the corrected original

    return None
