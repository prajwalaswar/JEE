"""
frontend/app.py — Streamlit frontend for Multimodal Math Mentor.

Panels:
  1. Input Selector       (Text / Image / Audio)
  2. OCR / Transcript Preview & Edit
  3. HITL confirmation dialog
  4. Agent Trace Panel
  5. Retrieved Context Viewer
  6. Solution Section (steps + final answer)
  7. Confidence Score
  8. Explanation Section
  9. Feedback Buttons

Run from project root:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env from project root (one level above this file)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal Math Mentor 🧮",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
  .main-title { font-size: 2.4rem; font-weight: 700; color: #1E40AF; }
  .section-header { font-size: 1.2rem; font-weight: 600;
                    border-bottom: 2px solid #3B82F6; padding-bottom: 4px;
                    margin-bottom: 12px; color: #1E3A8A; }
  .agent-card { background: #F0F9FF; border-left: 4px solid #3B82F6;
                padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
  .hitl-box { background: #FEF3C7; border: 2px solid #F59E0B;
              border-radius: 8px; padding: 16px; }
  .confidence-high  { color: #16A34A; font-weight: 700; }
  .confidence-mid   { color: #D97706; font-weight: 700; }
  .confidence-low   { color: #DC2626; font-weight: 700; }
  .source-chip { display: inline-block; background: #DBEAFE;
                 color: #1D4ED8; padding: 2px 8px; border-radius: 12px;
                 font-size: 0.8rem; margin: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Session state helpers ──────────────────────────────────────────────────────

def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ── API helpers ────────────────────────────────────────────────────────────────

def api_solve_text(text: str, session_id: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/solve/text",
        data={"problem": text, "session_id": session_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def api_solve_image(image_bytes: bytes, filename: str, session_id: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/solve/image",
        files={"file": (filename, image_bytes, "image/png")},
        data={"session_id": session_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def api_solve_audio(audio_bytes: bytes, filename: str, session_id: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/solve/audio",
        files={"file": (filename, audio_bytes, "audio/wav")},
        data={"session_id": session_id},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def api_hitl_respond(
    session_id: str,
    approved: bool,
    edited_text: Optional[str] = None,
    comment: Optional[str] = None,
) -> dict:
    resp = requests.post(
        f"{API_BASE}/hitl/respond",
        data={
            "session_id":  session_id,
            "approved":    str(approved).lower(),
            "edited_text": edited_text or "",
            "comment":     comment or "",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def api_feedback(session_id: str, feedback_type: str, comment: str = "") -> dict:
    resp = requests.post(
        f"{API_BASE}/feedback",
        json={
            "session_id":    session_id,
            "feedback_type": feedback_type,
            "comment":       comment or None,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Rendering helpers ──────────────────────────────────────────────────────────

def render_confidence(confidence: float) -> str:
    pct = int(confidence * 100)
    if confidence >= 0.80:
        cls = "confidence-high"
        icon = "✅"
    elif confidence >= 0.60:
        cls = "confidence-mid"
        icon = "⚠️"
    else:
        cls = "confidence-low"
        icon = "❌"
    return f'<span class="{cls}">{icon} {pct}%</span>'


def render_agent_trace(trace: list):
    st.markdown('<div class="section-header">🤖 Agent Trace</div>', unsafe_allow_html=True)
    icons = {
        "ParserAgent":       "🔍",
        "IntentRouterAgent": "🧭",
        "SolverAgent":       "🧮",
        "VerifierAgent":     "✅",
        "ExplainerAgent":    "📖",
    }
    for entry in trace:
        icon  = icons.get(entry.get("agent", ""), "🔹")
        agent = entry.get("agent", "Unknown")
        status = entry.get("status", "")
        ms     = entry.get("duration_ms", 0)
        out    = entry.get("output_summary", "")
        status_badge = {
            "done":           "🟢",
            "hitl_triggered": "🟡",
            "running":        "🔵",
        }.get(status, "⚪")

        st.markdown(
            f'<div class="agent-card">'
            f'{icon} <b>{agent}</b> {status_badge} '
            f'<small style="color:#6B7280">({ms:.0f}ms)</small><br/>'
            f'<small>{out}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_rag_context(context: dict):
    chunks = context.get("chunks", [])
    if not chunks:
        st.info("No knowledge base context retrieved.")
        return

    st.markdown(
        '<div class="section-header">📚 Retrieved Knowledge Base Context</div>',
        unsafe_allow_html=True,
    )
    for i, chunk in enumerate(chunks, start=1):
        src   = chunk.get("source", "unknown")
        score = chunk.get("score", 0.0)
        text  = chunk.get("content", "")
        with st.expander(f"[{i}] {src}  —  similarity: {score:.3f}"):
            st.text(text[:600] + ("…" if len(text) > 600 else ""))


def render_solution(solution: dict):
    st.markdown('<div class="section-header">🎯 Solution</div>', unsafe_allow_html=True)

    steps = solution.get("steps", [])
    if steps:
        for step in steps:
            n   = step.get("step_number", "?")
            desc = step.get("description", "")
            expr = step.get("symbolic_expr")
            res  = step.get("result")

            with st.container():
                st.markdown(f"**Step {n}:** {desc}")
                if expr:
                    st.latex(expr)
                if res:
                    st.markdown(f"> **Result:** `{res}`")

    # Final answer — prominent
    final = solution.get("final_answer", "N/A")
    st.success(f"### 🏆 Final Answer\n\n{final}")

    # Tool info
    tool = solution.get("tool_used", "")
    if tool:
        st.caption(f"Tool used: `{tool}`")


def render_verification(verification: dict):
    is_correct = verification.get("is_correct", False)
    confidence = verification.get("confidence", 0.0)
    issues     = verification.get("issues", [])

    col1, col2 = st.columns(2)
    with col1:
        if is_correct:
            st.success("✅ Verification: Correct")
        else:
            st.warning("⚠️ Verification: Uncertain / Incorrect")
    with col2:
        st.markdown(
            f"**Confidence:** {render_confidence(confidence)}",
            unsafe_allow_html=True,
        )

    if issues:
        with st.expander("⚠️ Issues noted by verifier"):
            for issue in issues:
                st.markdown(f"- {issue}")


def render_explanation(explanation: dict):
    st.markdown(
        '<div class="section-header">📖 Student-Friendly Explanation</div>',
        unsafe_allow_html=True,
    )
    text = explanation.get("student_friendly_text", "")
    st.markdown(text)

    concepts  = explanation.get("key_concepts", [])
    mistakes  = explanation.get("common_mistakes", [])

    col1, col2 = st.columns(2)
    with col1:
        if concepts:
            st.markdown("**🔑 Key Concepts:**")
            for c in concepts:
                st.markdown(f"- {c}")
    with col2:
        if mistakes:
            st.markdown("**⚠️ Common Mistakes:**")
            for m in mistakes:
                st.markdown(f"- {m}")


def render_feedback_buttons(session_id: str):
    st.markdown("---")
    st.markdown("**Was this solution helpful?**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Correct", key=f"fb_correct_{session_id}", use_container_width=True):
            try:
                api_feedback(session_id, "correct")
                st.success("Thanks for the feedback!")
            except Exception as e:
                st.error(f"Feedback error: {e}")

    with col2:
        if st.button("❌ Incorrect", key=f"fb_incorrect_{session_id}", use_container_width=True):
            comment = st.text_input("What was wrong?", key=f"fb_comment_{session_id}")
            if st.button("Submit", key=f"fb_submit_{session_id}"):
                try:
                    api_feedback(session_id, "incorrect", comment)
                    st.success("Feedback recorded. We'll improve!")
                except Exception as e:
                    st.error(f"Feedback error: {e}")


def render_hitl_dialog(hitl_request: dict, session_id: str):
    reason  = hitl_request.get("reason", "")
    message = hitl_request.get("prompt_message", "Please review the following:")
    text    = hitl_request.get("current_text", "")

    st.markdown('<div class="hitl-box">', unsafe_allow_html=True)
    st.markdown("### 🧑‍💻 Human Review Required")
    st.markdown(f"**Reason:** `{reason}`")
    st.markdown(message)

    edited = st.text_area(
        "Review / edit the text below:",
        value=text,
        height=120,
        key="hitl_text_area",
    )
    comment = st.text_input("Optional comment:", key="hitl_comment")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Continue", use_container_width=True, key="hitl_approve"):
            st.session_state.hitl_edited_text  = edited
            st.session_state.hitl_comment_saved = comment   # use separate key — widget key cannot be overwritten
            st.session_state.hitl_approved     = True
            st.rerun()
    with col2:
        if st.button("❌ Reject", use_container_width=True, key="hitl_reject"):
            st.session_state.hitl_approved = False
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ── Main UI ────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        '<div class="main-title">🧮 Multimodal Math Mentor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Solve JEE-level math problems from **text, image, or audio** "
        "using AI agents, RAG, and symbolic computation."
    )
    st.markdown("---")

    session_id = get_session_id()

    # ── HITL pending? ─────────────────────────────────────────────────────────
    if st.session_state.get("hitl_pending"):
        hitl_req = st.session_state.get("hitl_request", {})
        render_hitl_dialog(hitl_req, session_id)

        # Check if user responded
        if "hitl_approved" in st.session_state:
            approved     = st.session_state.hitl_approved
            edited_text  = st.session_state.get("hitl_edited_text")
            comment      = st.session_state.get("hitl_comment_saved")  # read from saved key

            with st.spinner("Resuming pipeline…"):
                try:
                    data = api_hitl_respond(session_id, approved, edited_text, comment)
                    st.session_state.last_response = data
                    st.session_state.hitl_pending  = False
                    del st.session_state["hitl_approved"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline error after HITL: {e}")

        return  # Don't show input section while HITL is active

    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📝 Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Select input mode:",
        options=["Text", "Image", "Audio"],
        horizontal=True,
    )

    response_data = None

    # ── Text input ────────────────────────────────────────────────────────────
    if input_mode == "Text":
        problem_text = st.text_area(
            "Enter your JEE math problem:",
            placeholder="e.g. Solve x² - 5x + 6 = 0 using the quadratic formula.",
            height=120,
        )
        if st.button("🧮 Solve", use_container_width=True, key="solve_text"):
            if not problem_text.strip():
                st.warning("Please enter a problem first.")
            else:
                with st.spinner("Agents working…"):
                    try:
                        response_data = api_solve_text(problem_text, session_id)
                        st.session_state.last_response = response_data
                    except Exception as e:
                        st.error(f"API error: {e}")

    # ── Image input ───────────────────────────────────────────────────────────
    elif input_mode == "Image":
        image_sub_mode = st.radio(
            "Choose image input method:",
            options=["📁 Upload Image", "📷 Take Photo"],
            horizontal=True,
            key="image_sub_mode",
        )

        img_bytes = None
        img_filename = "photo.png"

        if image_sub_mode == "📁 Upload Image":
            uploaded_img = st.file_uploader(
                "Upload a photo of the math problem (JPG/PNG):",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
            )
            if uploaded_img:
                img_bytes = uploaded_img.read()
                img_filename = uploaded_img.name
                st.image(img_bytes, caption="Uploaded image", width=400)

        else:  # Take Photo
            camera_img = st.camera_input("Take a photo of the math problem:")
            if camera_img:
                img_bytes = camera_img.read()
                img_filename = "camera_capture.png"
                st.image(img_bytes, caption="Captured photo", width=400)

        if img_bytes:
            col_prev_img, col_solve_img = st.columns(2)
            with col_prev_img:
                if st.button("👁️ Preview OCR", use_container_width=True, key="preview_img"):
                    with st.spinner("Extracting text from image…"):
                        try:
                            resp = requests.post(
                                f"{API_BASE}/preview/image",
                                files={"file": (img_filename, img_bytes, "image/png")},
                                timeout=60,
                            )
                            resp.raise_for_status()
                            st.session_state.img_preview = resp.json()
                        except Exception as e:
                            st.error(f"Preview error: {e}")
            with col_solve_img:
                if st.button("🔍 Extract & Solve", use_container_width=True, key="solve_img"):
                    with st.spinner("Running OCR then solving…"):
                        try:
                            response_data = api_solve_image(
                                img_bytes, img_filename, session_id
                            )
                            st.session_state.last_response = response_data
                        except Exception as e:
                            st.error(f"API error: {e}")

            if "img_preview" in st.session_state:
                prev = st.session_state.img_preview
                conf_pct = int(prev["confidence"] * 100)
                conf_color = "green" if conf_pct >= 80 else "orange" if conf_pct >= 60 else "red"
                st.markdown(f"**📝 Extracted Text** (OCR backend: `{prev.get('backend','?')}`, confidence: :{conf_color}[{conf_pct}%])")
                edited_ocr = st.text_area(
                    "Review / edit extracted text before solving:",
                    value=prev["extracted_text"],
                    height=120,
                    key="ocr_preview_text",
                )
                if prev.get("needs_hitl"):
                    st.warning("⚠️ Low confidence — human review will be triggered during solving.")

    # ── Audio input ───────────────────────────────────────────────────────────
    elif input_mode == "Audio":
        audio_sub_mode = st.radio(
            "Choose audio input method:",
            options=["🎤 Record Live Audio", "📁 Upload Audio File"],
            horizontal=True,
            key="audio_sub_mode",
        )

        audio_bytes = None
        audio_filename = "recording.wav"

        if audio_sub_mode == "🎤 Record Live Audio":
            st.info("🎙️ Click the microphone to start recording. Speak your math problem clearly, then stop.")
            recorded_audio = st.audio_input(
                "Record your math problem:",
                key="audio_recorder",
            )
            if recorded_audio:
                audio_bytes = recorded_audio.read()
                # Detect format from bytes header
                # WebM signature: 0x1A 0x45 0xDF 0xA3
                if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
                    audio_filename = "live_recording.webm"
                elif audio_bytes[:4] == b'RIFF':
                    audio_filename = "live_recording.wav"
                elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
                    audio_filename = "live_recording.mp3"
                elif audio_bytes[:4] == b'OggS':
                    audio_filename = "live_recording.ogg"
                else:
                    audio_filename = "live_recording.webm"  # default for browser
                st.audio(audio_bytes)
                st.caption(f"Recorded {len(audio_bytes):,} bytes — format detected: `{audio_filename.split('.')[-1].upper()}`")

        else:  # Upload Audio File
            uploaded_audio = st.file_uploader(
                "Upload an audio recording of the problem (WAV/MP3/M4A):",
                type=["wav", "mp3", "m4a", "ogg"],
            )
            if uploaded_audio:
                audio_bytes = uploaded_audio.read()
                audio_filename = uploaded_audio.name
                st.audio(audio_bytes)

        if audio_bytes:
            col_prev_aud, col_solve_aud = st.columns(2)
            with col_prev_aud:
                if st.button("👁️ Preview Transcript", use_container_width=True, key="preview_audio"):
                    with st.spinner("Transcribing audio (Whisper)…"):
                        try:
                            resp = requests.post(
                                f"{API_BASE}/preview/audio",
                                files={"file": (audio_filename, audio_bytes, "audio/webm")},
                                timeout=120,
                            )
                            resp.raise_for_status()
                            st.session_state.audio_preview = resp.json()
                        except Exception as e:
                            st.error(f"Preview error: {e}")
            with col_solve_aud:
                if st.button("🎙️ Transcribe & Solve", use_container_width=True, key="solve_audio"):
                    with st.spinner("Transcribing audio (Whisper) then solving…"):
                        try:
                            response_data = api_solve_audio(
                                audio_bytes, audio_filename, session_id
                            )
                            st.session_state.last_response = response_data
                        except Exception as e:
                            st.error(f"API error: {e}")

            if "audio_preview" in st.session_state:
                prev = st.session_state.audio_preview
                conf_pct = int(prev["confidence"] * 100)
                conf_color = "green" if conf_pct >= 80 else "orange" if conf_pct >= 60 else "red"
                st.markdown(f"**🗣️ Transcript** (confidence: :{conf_color}[{conf_pct}%])")
                edited_transcript = st.text_area(
                    "Review / edit transcript before solving:",
                    value=prev["transcript"],
                    height=100,
                    key="audio_preview_text",
                )
                if prev.get("needs_hitl"):
                    st.warning("⚠️ Low confidence — human review will be triggered during solving.")

    # ── Load previous response ────────────────────────────────────────────────
    if response_data is None:
        response_data = st.session_state.get("last_response")
    else:
        st.session_state.last_response = response_data

    # ── Render response ───────────────────────────────────────────────────────
    if response_data:
        # HITL check
        if response_data.get("hitl_required"):
            st.session_state.hitl_pending  = True
            st.session_state.hitl_request  = response_data.get("hitl_request", {})
            st.rerun()
            return

        st.markdown("---")

        # Layout: left (main) and right (trace)
        col_main, col_trace = st.columns([2, 1])

        with col_main:
            # Parsed problem info
            parsed = response_data.get("parsed_problem")
            if parsed:
                st.markdown(
                    '<div class="section-header">🔍 Parsed Problem</div>',
                    unsafe_allow_html=True,
                )
                st.info(parsed.get("problem_text", ""))
                meta_cols = st.columns(3)
                meta_cols[0].metric("Topic",     parsed.get("topic", "unknown").replace("_", " ").title())
                meta_cols[1].metric("Variables", ", ".join(parsed.get("variables", [])) or "—")
                meta_cols[2].metric(
                    "Constraints",
                    str(len(parsed.get("constraints", [])))
                )

            # Solution
            solution = response_data.get("solution")
            if solution:
                render_solution(solution)
            else:
                st.warning("No solution generated.")

            # Verification
            verification = response_data.get("verification")
            if verification:
                render_verification(verification)

            # Explanation
            explanation = response_data.get("explanation")
            if explanation:
                render_explanation(explanation)

            # Feedback
            render_feedback_buttons(session_id)

        with col_trace:
            # Agent trace
            trace = response_data.get("agent_trace", [])
            if trace:
                render_agent_trace(trace)

            st.markdown("")  # spacer

            # RAG context
            rag_ctx = response_data.get("rag_context")
            if rag_ctx:
                render_rag_context(rag_ctx)

    # ── Sidebar: session info ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Session Info")
        st.code(session_id[:8] + "…", language=None)
        if st.button("🔄 New Session"):
            reset_session()

        st.markdown("---")
        st.markdown("### API Status")
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            data = r.json()
            st.success(f"API OK — {data.get('knowledge_base_chunks', 0)} KB chunks")
        except Exception:
            st.error("API unreachable")

        st.markdown("---")
        st.markdown("### 📋 Backend Logs")
        log_lines_count = st.slider("Lines to show", 20, 200, 50, step=10, key="log_lines")
        st.button("🔄 Refresh Logs", key="refresh_logs")
        try:
            r = requests.get(f"{API_BASE}/logs?lines={log_lines_count}", timeout=5)
            log_data = r.json()
            log_lines = log_data.get("lines", [])
            if log_lines:
                log_text = "\n".join(log_lines)
                st.text_area(
                    f"Last {len(log_lines)} lines:",
                    value=log_text,
                    height=300,
                    key="log_viewer",
                )
                total = log_data.get("total", 0)
                if total > log_lines_count:
                    st.caption(f"Showing {log_lines_count}/{total} lines. Full history in `backend.log`.")
            else:
                st.info("No logs yet.")
        except Exception as log_err:
            st.warning(f"Cannot fetch logs: {log_err}")

        st.markdown("---")
        st.markdown("### Recent Sessions")
        try:
            r = requests.get(f"{API_BASE}/memory/recent?limit=5", timeout=10)
            records = r.json()
            for rec in records[:5]:
                with st.expander(f"Session {rec['session_id'][:8]}…"):
                    st.write(f"**Input:** {rec['original_input'][:80]}…")
                    fb = rec.get("user_feedback")
                    if fb:
                        st.write(f"**Feedback:** {fb}")
        except Exception:
            st.info("No memory records yet.")


if __name__ == "__main__":
    main()
