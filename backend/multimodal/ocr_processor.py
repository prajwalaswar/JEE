"""
ocr_processor.py — Image → text extraction using a multi-strategy approach:

  1. **Gemini Vision** (primary) — sends the image directly to Google Gemini
     which natively understands math notation, diagrams, and noisy photos.
  2. **EasyOCR with preprocessing** (fallback) — used when Gemini is
     unavailable or rate-limited; includes image preprocessing (grayscale,
     contrast enhancement, adaptive thresholding, denoising).
  3. **PaddleOCR** (optional alternative backend).

Returns OCRResult with extracted text and aggregate confidence score.
Triggers HITL when confidence < OCR_CONFIDENCE_THRESHOLD.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

from backend.config import (
    OCR_BACKEND,
    OCR_CONFIDENCE_THRESHOLD,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from backend.models import OCRResult

logger = logging.getLogger(__name__)


# ── Image Preprocessing ──────────────────────────────────────────────────────

def _preprocess_image(image_bytes: bytes) -> "np.ndarray":
    """
    Apply preprocessing to improve OCR accuracy on math images.

    Steps:
      1. Decode to BGR.
      2. Convert to grayscale.
      3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
      4. Apply bilateral filter (denoising while preserving edges).
      5. Apply adaptive threshold for clean text extraction.
      6. Morphological opening to remove tiny noise dots.

    Returns:
        Preprocessed grayscale numpy image.
    """
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image bytes.")

    # Resize if too small (OCR works better on larger images)
    h, w = img.shape[:2]
    if max(h, w) < 600:
        scale = 600 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Bilateral filter: removes noise while keeping sharp edges
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive thresholding — better for uneven lighting (photos)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=8,
    )

    # Morphological opening to remove small noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


# ── Gemini Vision backend (primary) ──────────────────────────────────────────

def _run_gemini_vision(image_bytes: bytes) -> Tuple[str, float]:
    """
    Use Google Gemini Vision to extract math problem text from an image.

    Gemini natively understands mathematical notation, printed/handwritten
    text, and complex layouts — far superior to traditional OCR for math.

    Returns:
        (text, confidence) tuple.  Confidence is 0.95 when Gemini succeeds
        (it is very reliable at math extraction).
    """
    try:
        from google import genai
        from google.genai import types as genai_types

        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Determine MIME type from image header bytes
        mime_type = "image/png"
        if image_bytes[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        elif image_bytes[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif image_bytes[:4] == b'RIFF':
            mime_type = "image/webp"

        prompt = (
            "You are an expert at reading math problems from images.\n"
            "Extract the COMPLETE math problem from this image.\n"
            "Rules:\n"
            "- Preserve ALL equations exactly as written (use standard notation: +, -, *, /, ^, =)\n"
            "- Include all multiple choice options if present (label them A, B, C, D etc.)\n"
            "- Preserve variable names exactly (x, y, z, etc.)\n"
            "- If there are multiple equations, put each on its own line\n"
            "- Output ONLY the math problem text, nothing else\n"
            "- Do NOT solve the problem, just extract it\n"
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )

        text = response.text.strip()
        if not text:
            return "", 0.0

        # Gemini is very reliable for math extraction — give it high confidence
        return text, 0.95

    except ImportError:
        logger.warning("google-genai not installed, falling back to OCR.")
        raise
    except Exception as exc:
        logger.warning("Gemini Vision failed (%s), falling back to OCR.", exc)
        raise


# ── EasyOCR backend (fallback) ────────────────────────────────────────────────

def _run_easyocr(image_bytes: bytes) -> Tuple[str, float]:
    """
    Run EasyOCR on raw image bytes with preprocessing.

    Returns:
        (text, confidence) tuple.
    """
    try:
        import easyocr
        import cv2

        # Preprocess the image for better OCR accuracy
        preprocessed = _preprocess_image(image_bytes)

        # Also decode the original (we'll run OCR on both and pick the better)
        nparr = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Initialise reader (English only; set gpu=True if CUDA available)
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        # Run on both original and preprocessed
        results_orig = reader.readtext(original)
        results_prep = reader.readtext(preprocessed)

        def _aggregate(results):
            if not results:
                return "", 0.0
            texts       = [r[1] for r in results]
            confidences = [r[2] for r in results]
            return " ".join(texts), float(np.mean(confidences))

        text_orig, conf_orig = _aggregate(results_orig)
        text_prep, conf_prep = _aggregate(results_prep)

        # Pick the result with higher confidence
        if conf_prep > conf_orig:
            logger.info("Preprocessed image OCR better: %.3f vs %.3f", conf_prep, conf_orig)
            return text_prep, conf_prep
        else:
            return text_orig, conf_orig

    except ImportError:
        logger.error("EasyOCR not installed. Run: pip install easyocr")
        raise


# ── PaddleOCR backend ─────────────────────────────────────────────────────────

def _run_paddleocr(image_bytes: bytes) -> Tuple[str, float]:
    """
    Run PaddleOCR 3.x on raw image bytes.

    PaddleOCR 3.x uses the predict() API and bundles its own inference runtime
    (no separate paddlepaddle install required). Sets
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK to skip the network availability
    check that happens at import time.

    Returns:
        (text, confidence) tuple.
    """
    import os as _os
    # Skip internet connectivity check at import — safe for offline/demo use
    _os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        from paddleocr import PaddleOCR

        # Save bytes to temp file — PaddleOCR accepts file paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        texts: list = []
        confidences: list = []

        try:
            ocr = PaddleOCR(lang="en", show_log=False)

            # ── PaddleOCR 3.x: predict() API ─────────────────────────────────
            # result is a list of result objects (one per input image).
            # Each object supports dict-like access: res['rec_texts'] / res['rec_scores']
            # or iteration over individual text-line dicts.
            try:
                raw = ocr.predict(tmp_path)
                for page in (raw or []):
                    # Approach A: dict-like access (most 3.x versions)
                    if hasattr(page, "__getitem__"):
                        try:
                            rec_texts  = page["rec_texts"]
                            rec_scores = page["rec_scores"]
                            texts.extend(rec_texts)
                            confidences.extend(rec_scores)
                            continue
                        except (KeyError, TypeError):
                            pass
                    # Approach B: iterate items (each item is a line dict)
                    try:
                        for item in page:
                            if isinstance(item, dict):
                                texts.append(item.get("rec_text", ""))
                                confidences.append(float(item.get("rec_score", 0.0)))
                            elif hasattr(item, "rec_text"):
                                texts.append(item.rec_text)
                                confidences.append(float(item.rec_score))
                    except TypeError:
                        pass

            except (AttributeError, TypeError):
                # ── PaddleOCR 2.x fallback: ocr() API ────────────────────────
                logger.info("Falling back to PaddleOCR 2.x ocr() API.")
                result = ocr.ocr(tmp_path, cls=True)
                if result and result[0]:
                    texts       = [line[1][0] for line in result[0]]
                    confidences = [line[1][1] for line in result[0]]

        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not texts:
            return "", 0.0

        combined_text  = " ".join(t for t in texts if t)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        return combined_text, avg_confidence

    except ImportError:
        logger.error("PaddleOCR not installed. Run: pip install paddleocr")
        raise


# ── Math-aware post-processing ────────────────────────────────────────────────

def _math_post_process(text: str) -> str:
    """
    Fix common OCR mistakes specific to math notation.

    - 'O' → '0' when inside equations
    - 'l' → '1' when standalone in math context
    - 'S' → '5' in numeric sequences
    - Fix spacing around operators
    """
    import re

    if not text:
        return text

    # Fix common character confusions
    # Standalone lowercase L → 1 (but not inside words)
    text = re.sub(r'(?<![a-zA-Z])l(?![a-zA-Z])', '1', text)

    # Capital O between digits → 0
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    text = re.sub(r'(?<=\d)O(?=\s*[+\-*/=])', '0', text)
    text = re.sub(r'(?<=[+\-*/=])O(?=\d)', '0', text)
    text = re.sub(r'(?<=[+\-*/=] )O(?=\d)', '0', text)

    # 'Jy' or 'Ty' with no meaning → likely OCR errors for '3y' etc
    # (common EasyOCR mistake: J/T for digits near variables)
    text = re.sub(r'\bJ(?=[a-z])', '3', text)
    text = re.sub(r'\bT(?=[a-z])', '1', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ── Public interface ──────────────────────────────────────────────────────────

def process_image(image_bytes: bytes) -> OCRResult:
    """
    Extract text from an image using a multi-strategy approach.

    Strategy:
      1. Try Gemini Vision first (best accuracy for math).
      2. Fall back to EasyOCR/PaddleOCR with preprocessing.
      3. Apply math-aware post-processing.

    Args:
        image_bytes: Raw image file bytes (PNG / JPEG / etc.).

    Returns:
        OCRResult with extracted_text, confidence, and needs_hitl flag.
    """
    text = ""
    confidence = 0.0

    # ── Strategy 1: Gemini Vision (primary) ───────────────────────────────────
    if GEMINI_API_KEY:
        try:
            text, confidence = _run_gemini_vision(image_bytes)
            logger.info(
                "Gemini Vision succeeded — confidence: %.2f, text: '%s…'",
                confidence,
                text[:60],
            )
        except Exception as exc:
            logger.warning("Gemini Vision unavailable: %s. Using OCR fallback.", exc)
            text = ""
            confidence = 0.0

    # ── Strategy 2: OCR fallback ──────────────────────────────────────────────
    if not text:
        logger.info("Running OCR fallback with backend: %s", OCR_BACKEND)
        try:
            if OCR_BACKEND == "paddleocr":
                text, confidence = _run_paddleocr(image_bytes)
            else:
                text, confidence = _run_easyocr(image_bytes)
        except Exception as exc:
            logger.error("OCR processing failed: %s", exc)
            return OCRResult(
                extracted_text=f"[OCR Error: {exc}]",
                confidence=0.0,
                needs_hitl=True,
            )

    # ── Post-processing ───────────────────────────────────────────────────────
    text = _math_post_process(text)

    needs_hitl = confidence < OCR_CONFIDENCE_THRESHOLD
    if needs_hitl:
        logger.info(
            "OCR confidence %.2f below threshold %.2f — HITL triggered.",
            confidence,
            OCR_CONFIDENCE_THRESHOLD,
        )

    return OCRResult(
        extracted_text=text.strip(),
        confidence=round(confidence, 4),
        needs_hitl=needs_hitl,
    )
