"""
audio_processor.py — Speech → text transcription using faster-whisper.

Supports: WAV, MP3, M4A, OGG, WebM (live browser recording).

faster-whisper is a reimplementation of OpenAI's Whisper using CTranslate2,
delivering 2–4× faster inference with lower memory usage and int8 quantization.

Key fix for Windows: faster-whisper still needs audio decoded to a 16 kHz
mono float32 numpy array (or a WAV path). We solve this by:
  1. Detecting the bundled ffmpeg binary from `imageio-ffmpeg`.
  2. Using that binary to pre-convert any audio format → 16 kHz mono WAV.
  3. Loading the WAV into numpy with soundfile and passing the array directly
     to faster-whisper — no internal ffmpeg subprocess needed.

faster-whisper segments are objects (not dicts); we read `.avg_logprob` to
compute proxy confidence. Triggers HITL when confidence < ASR_CONFIDENCE_THRESHOLD.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import (
    WHISPER_MODEL_SIZE,
    ASR_CONFIDENCE_THRESHOLD,
    FASTER_WHISPER_DEVICE,
    FASTER_WHISPER_COMPUTE_TYPE,
)
from backend.models import ASRResult

logger = logging.getLogger(__name__)

# Module-level singleton — Whisper model loaded once
_whisper_model = None


# ── ffmpeg setup ──────────────────────────────────────────────────────────────

def _get_ffmpeg_exe() -> Optional[str]:
    """
    Locate the ffmpeg executable.

    Priority:
      1. System PATH (user has ffmpeg installed globally).
      2. imageio-ffmpeg bundled binary (automatic Windows install).

    Returns:
        Absolute path to ffmpeg executable, or None if not found.
    """
    # Check system PATH first
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        logger.debug("ffmpeg found in PATH: %s", system_ffmpeg)
        return system_ffmpeg

    # Fall back to imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled):
            logger.info("Using imageio-ffmpeg bundled binary: %s", bundled)
            return bundled
    except ImportError:
        pass

    logger.warning(
        "ffmpeg not found. Non-WAV audio formats may fail. "
        "Install imageio-ffmpeg: pip install imageio-ffmpeg"
    )
    return None


def _convert_to_wav(input_path: str, ffmpeg_exe: str) -> Optional[str]:
    """
    Convert any audio file to 16 kHz mono WAV using ffmpeg.

    This bypasses Whisper's internal ffmpeg call (which relies on PATH)
    by doing the conversion ourselves with the full binary path.

    Args:
        input_path: Path to the source audio file.
        ffmpeg_exe: Full path to the ffmpeg binary.

    Returns:
        Path to the converted WAV file (caller must delete it), or None on error.
    """
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    cmd = [
        ffmpeg_exe,
        "-y",               # overwrite output
        "-i", input_path,   # input file
        "-ar", "16000",     # 16 kHz sample rate (Whisper requirement)
        "-ac", "1",         # mono channel
        "-f", "wav",        # output format
        out_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            logger.error("ffmpeg conversion failed (rc=%d): %s", proc.returncode, err[-500:])
            Path(out_path).unlink(missing_ok=True)
            return None
        logger.info("ffmpeg converted audio → WAV: %s", out_path)
        return out_path
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg conversion timed out.")
        Path(out_path).unlink(missing_ok=True)
        return None
    except Exception as exc:
        logger.error("ffmpeg conversion error: %s", exc)
        Path(out_path).unlink(missing_ok=True)
        return None


# ── Whisper model ─────────────────────────────────────────────────────────────

def _get_whisper_model():
    """Lazy-load the faster-whisper model (singleton)."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info(
                "Loading faster-whisper model — size: %s, device: %s, compute: %s",
                WHISPER_MODEL_SIZE,
                FASTER_WHISPER_DEVICE,
                FASTER_WHISPER_COMPUTE_TYPE,
            )
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=FASTER_WHISPER_DEVICE,
                compute_type=FASTER_WHISPER_COMPUTE_TYPE,
            )
            logger.info("faster-whisper model loaded.")
        except ImportError:
            logger.error(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
            raise
    return _whisper_model


# ── Confidence computation ────────────────────────────────────────────────────

def _compute_confidence_from_segments(segments: list) -> float:
    """
    Compute an overall confidence proxy from faster-whisper segment log-probabilities.

    faster-whisper Segment objects expose `.avg_logprob` (typically in [-1, 0]).
    We convert to [0, 1] using exp(avg_logprob).

    Returns:
        Confidence value in [0.0, 1.0].
    """
    if not segments:
        return 0.0

    log_probs = []
    for seg in segments:
        # faster-whisper → Segment object with .avg_logprob attribute
        if hasattr(seg, "avg_logprob"):
            log_probs.append(seg.avg_logprob)
        elif isinstance(seg, dict):
            log_probs.append(seg.get("avg_logprob", -1.0))
        else:
            log_probs.append(-1.0)

    log_probs = [max(-5.0, min(0.0, lp)) for lp in log_probs]
    avg_log_prob = float(np.mean(log_probs))
    confidence = math.exp(avg_log_prob)
    return round(confidence, 4)


# ── Public interface ──────────────────────────────────────────────────────────

def process_audio(audio_bytes: bytes, file_extension: str = ".wav") -> ASRResult:
    """
    Transcribe audio bytes to text using local Whisper.

    Handles WAV, MP3, M4A, OGG, WebM (browser live recording).
    On Windows, uses bundled imageio-ffmpeg to convert audio → WAV first,
    bypassing the missing system ffmpeg issue.

    Args:
        audio_bytes:    Raw audio file bytes.
        file_extension: File extension hint (".wav", ".mp3", ".webm", etc.).

    Returns:
        ASRResult with transcript, confidence, and needs_hitl flag.
    """
    logger.info(
        "Starting audio transcription — format: %s, size: %d bytes",
        file_extension,
        len(audio_bytes),
    )

    # Normalise extension
    ext = file_extension.lower()
    if not ext.startswith("."):
        ext = "." + ext

    tmp_input_path: Optional[str] = None
    tmp_wav_path:   Optional[str] = None

    try:
        model = _get_whisper_model()

        # ── Step 1: Write incoming bytes to a temp file ───────────────────────
        fd, tmp_input_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as fh:
            fh.write(audio_bytes)
        logger.info("Wrote audio to temp file: %s", tmp_input_path)

        # ── Step 2: Convert to 16 kHz mono WAV via our ffmpeg binary ─────────
        # Whisper's model.transcribe(path) ALWAYS calls load_audio(path)
        # internally, which spawns `ffmpeg` by name — failing on Windows when
        # ffmpeg is not in PATH.  The only reliable workaround is to:
        #   a) convert the file → WAV ourselves (using full ffmpeg path), then
        #   b) read the WAV into a numpy array with soundfile, then
        #   c) pass the array to model.transcribe() — this path skips
        #      load_audio() entirely.
        ffmpeg_exe = _get_ffmpeg_exe()
        audio_array: Optional[np.ndarray] = None

        if ffmpeg_exe:
            logger.info("Converting %s → WAV via ffmpeg…", ext)
            tmp_wav_path = _convert_to_wav(tmp_input_path, ffmpeg_exe)
            if tmp_wav_path:
                try:
                    import soundfile as sf
                    audio_array, _ = sf.read(tmp_wav_path, dtype="float32")
                    # Ensure mono
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)
                    logger.info(
                        "Loaded WAV as numpy array — shape: %s", audio_array.shape
                    )
                except Exception as sf_exc:
                    logger.warning("soundfile read failed: %s — trying scipy", sf_exc)
                    try:
                        from scipy.io import wavfile
                        rate, data = wavfile.read(tmp_wav_path)
                        if data.ndim > 1:
                            data = data.mean(axis=1)
                        audio_array = data.astype(np.float32) / 32768.0
                        logger.info("Loaded WAV via scipy — shape: %s", audio_array.shape)
                    except Exception as sp_exc:
                        logger.warning("scipy read also failed: %s", sp_exc)
            else:
                logger.warning("ffmpeg conversion returned None; will pass file path.")
        else:
            logger.warning(
                "ffmpeg not available; install imageio-ffmpeg or system ffmpeg. "
                "Attempting to load WAV via soundfile directly."
            )
            if ext == ".wav":
                try:
                    import soundfile as sf
                    audio_array, _ = sf.read(tmp_input_path, dtype="float32")
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)
                except Exception as sf_exc:
                    logger.warning("Direct soundfile read failed: %s", sf_exc)

        # ── Step 3: Transcribe with faster-whisper ───────────────────────────
        # faster-whisper.transcribe() accepts:
        #   • a 1-D float32 numpy array at 16 kHz (preferred — no internal ffmpeg)
        #   • a file path string (faster-whisper handles decoding internally)
        # transcribe() returns (segments_generator, TranscriptionInfo).
        if audio_array is not None:
            logger.info("Transcribing from numpy array via faster-whisper.")
            segments_gen, info = model.transcribe(
                audio_array,
                language="en",
                beam_size=5,
                vad_filter=True,           # silence suppression
                vad_parameters=dict(min_silence_duration_ms=500),
            )
        else:
            logger.warning("Falling back to file-path transcription via faster-whisper.")
            segments_gen, info = model.transcribe(
                tmp_input_path,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

        # Materialise the generator — required before computing confidence
        segments   = list(segments_gen)
        transcript = " ".join(seg.text for seg in segments).strip()
        confidence = _compute_confidence_from_segments(segments)
        logger.info(
            "faster-whisper detected language: %s (prob=%.2f)",
            info.language,
            info.language_probability,
        )

        logger.info(
            "Transcription done — confidence: %.3f, transcript: '%s…'",
            confidence,
            transcript[:80],
        )

        needs_hitl = confidence < ASR_CONFIDENCE_THRESHOLD
        if needs_hitl:
            logger.info(
                "ASR confidence %.2f below threshold %.2f — HITL triggered.",
                confidence,
                ASR_CONFIDENCE_THRESHOLD,
            )

        return ASRResult(
            transcript=transcript,
            confidence=confidence,
            needs_hitl=needs_hitl,
        )

    except Exception as exc:
        logger.error("Audio transcription failed: %s", exc, exc_info=True)
        return ASRResult(
            transcript=f"[Transcription Error: {exc}]",
            confidence=0.0,
            needs_hitl=True,
        )

    finally:
        # ── Cleanup temp files ────────────────────────────────────────────────
        for p in (tmp_input_path, tmp_wav_path):
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
