"""
llm_client.py — Unified LLM interface supporting Groq and Google Gemini.

Uses the openai-compatible Groq SDK as primary.
Falls back to Google GenerativeAI (Gemini) for reasoning-heavy tasks.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from backend.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

logger = logging.getLogger(__name__)


# ── Groq ──────────────────────────────────────────────────────────────────────

def chat_groq(
    messages: List[dict],
    model: str = GROQ_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """
    Call Groq API (OpenAI-compatible).

    Args:
        messages:    List of {"role": ..., "content": ...} dicts.
        model:       Groq model identifier.
        temperature: Sampling temperature.
        max_tokens:  Max output tokens.

    Returns:
        Assistant message string.
    """
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq")
        raise
    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        raise


# ── Gemini ────────────────────────────────────────────────────────────────────

def chat_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """
    Call Google Gemini via google-generativeai.

    Args:
        prompt:      Full prompt string.
        model:       Gemini model name.
        temperature: Sampling temperature.
        max_tokens:  Max output tokens.

    Returns:
        Response text string.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()
    except ImportError:
        logger.error(
            "google-genai not installed. Run: pip install google-genai"
        )
        raise
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise


# ── Auto-select ────────────────────────────────────────────────────────────────

def chat(
    messages: List[dict],
    prefer_gemini: bool = False,
    **kwargs,
) -> str:
    """
    Auto-select LLM backend.

    Priority:
      1. Groq (if GROQ_API_KEY set and prefer_gemini=False)
      2. Gemini (if GEMINI_API_KEY set)
      3. Raise RuntimeError

    Args:
        messages:       OpenAI-format message list.
        prefer_gemini:  Force Gemini for reasoning-heavy tasks.

    Returns:
        Response text.
    """
    if GROQ_API_KEY and not prefer_gemini:
        return chat_groq(messages, **kwargs)
    elif GEMINI_API_KEY:
        # Gemini takes a single string — concatenate messages
        combined = "\n\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in messages
        )
        try:
            return chat_gemini(combined, **kwargs)
        except Exception as gemini_exc:
            logger.warning(
                "Gemini call failed (%s). Falling back to Groq.", gemini_exc
            )
            if GROQ_API_KEY:
                return chat_groq(messages, **kwargs)
            raise
    else:
        raise RuntimeError(
            "No LLM API key configured. "
            "Set GROQ_API_KEY or GEMINI_API_KEY in your .env file."
        )
