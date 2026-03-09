"""
text_processor.py — Math-aware text pre-processing for all input sources.

Normalises whitespace, fixes common OCR/ASR artefacts, and corrects
typical math-notation mistakes.
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """
    Clean raw text from any input source (OCR, ASR, or direct typing).

    Steps:
    1. Strip leading/trailing whitespace.
    2. Remove non-printable characters.
    3. Normalize whitespace (multiple spaces/newlines).
    4. Fix common OCR character substitutions in math context.
    5. Normalize math operators and notation.
    6. Fix mis-spaced equations.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Remove non-printable chars (keep newlines for structure)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── Math-specific OCR corrections ─────────────────────────────────────────

    # Standalone lowercase 'l' → '1' (common OCR error, but NOT inside words)
    text = re.sub(r"(?<![a-zA-Z])l(?![a-zA-Z])", "1", text)

    # Capital O between digits → 0
    text = re.sub(r"(?<=\d)O(?=\d)", "0", text)

    # 'x' (multiplication) in numeric context: e.g. "3 x 4" → "3 * 4"
    # Only when surrounded by digits (not when it's a variable)
    text = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", text)

    # ── Normalize math notation ───────────────────────────────────────────────

    # Convert fancy Unicode operators to standard ASCII
    replacements = {
        "−": "-",    # Unicode minus → ASCII hyphen-minus
        "×": "*",    # Multiplication sign
        "÷": "/",    # Division sign
        "²": "^2",   # Superscript 2
        "³": "^3",   # Superscript 3
        "√": "sqrt", # Square root
        "π": "pi",   # Pi
        "∞": "oo",   # Infinity (SymPy notation)
        "≤": "<=",   # Less than or equal
        "≥": ">=",   # Greater than or equal
        "≠": "!=",   # Not equal
        "\u2264": "<=",
        "\u2265": ">=",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # ── Fix equation spacing ──────────────────────────────────────────────────

    # Ensure spaces around '=' in equations
    text = re.sub(r"(?<=\S)=(?=\S)", " = ", text)

    return text.strip()
