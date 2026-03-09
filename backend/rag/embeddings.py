"""
embeddings.py — Sentence-transformer embedding helper.
Uses HuggingFace sentence-transformers (free, local).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (avoids repeated disk I/O)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts.

    Args:
        texts: List of strings to embed.

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine = dot product after L2 norm
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Args:
        query: Query text.

    Returns:
        numpy array of shape (embedding_dim,)
    """
    return embed_texts([query])[0]
