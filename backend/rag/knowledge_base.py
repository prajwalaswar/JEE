"""
knowledge_base.py — Loads text files from data/knowledge_base/,
chunks them, and indexes them into FAISS.
"""

from __future__ import annotations

import hashlib
import logging
import os
import textwrap
from pathlib import Path
from typing import List, Tuple

from backend.config import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from backend.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping character-level chunks.

    Args:
        text:       Full document text.
        chunk_size: Approximate chunk length in characters.
        overlap:    Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _make_chunk_id(source: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_knowledge_base(store: FAISSVectorStore) -> None:
    """
    Read all .txt files in KNOWLEDGE_BASE_DIR, chunk them, and index into FAISS.
    Skips indexing if the store already has documents (idempotent).
    """
    if not store.is_empty:
        logger.info(
            "Knowledge base already indexed (%d chunks). Skipping build.",
            store.document_count(),
        )
        return

    kb_dir = Path(KNOWLEDGE_BASE_DIR)
    if not kb_dir.exists():
        logger.error("Knowledge base directory not found: %s", kb_dir)
        return

    all_texts:     List[str] = []
    all_sources:   List[str] = []
    all_chunk_ids: List[str] = []

    for txt_file in sorted(kb_dir.glob("*.txt")):
        logger.info("Indexing file: %s", txt_file.name)
        raw = txt_file.read_text(encoding="utf-8")

        chunks = _chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_sources.append(txt_file.name)
            all_chunk_ids.append(_make_chunk_id(txt_file.name, i))

    if all_texts:
        logger.info("Total chunks to index: %d", len(all_texts))
        store.add_documents(all_texts, all_sources, all_chunk_ids)
        logger.info("Knowledge base build complete.")
    else:
        logger.warning("No text files found in %s.", kb_dir)
