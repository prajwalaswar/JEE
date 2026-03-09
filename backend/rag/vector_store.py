"""
vector_store.py — FAISS vector store wrapper.
Handles index creation, persistence, and similarity search.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from backend.config import FAISS_INDEX_PATH
from backend.rag.embeddings import embed_texts, embed_query
from backend.models import RetrievedChunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    A lightweight FAISS-backed vector store with metadata persistence.

    Internal layout on disk:
        FAISS_INDEX_PATH/
            index.faiss      — the raw FAISS index
            metadata.pkl     — list of dict with {content, source, chunk_id}
    """

    def __init__(self, index_path: str = FAISS_INDEX_PATH) -> None:
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index_file    = self.index_path / "index.faiss"
        self._metadata_file = self.index_path / "metadata.pkl"

        self._index: faiss.Index | None      = None
        self._metadata: List[Dict[str, Any]] = []

        # Try loading existing index
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load existing FAISS index and metadata from disk."""
        if self._index_file.exists() and self._metadata_file.exists():
            try:
                self._index = faiss.read_index(str(self._index_file))
                with open(self._metadata_file, "rb") as f:
                    self._metadata = pickle.load(f)
                logger.info(
                    "FAISS index loaded: %d vectors.", self._index.ntotal
                )
            except Exception as exc:
                logger.warning("Failed to load FAISS index: %s", exc)
                self._index    = None
                self._metadata = []

    def _save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_file))
            with open(self._metadata_file, "wb") as f:
                pickle.dump(self._metadata, f)
            logger.info("FAISS index saved: %d vectors.", self._index.ntotal)

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_documents(
        self,
        texts:      List[str],
        sources:    List[str],
        chunk_ids:  List[str],
    ) -> None:
        """
        Embed and add documents to the FAISS index.

        Args:
            texts:     List of text chunks.
            sources:   List of source file names (same length as texts).
            chunk_ids: Unique IDs for each chunk.
        """
        if not texts:
            return

        logger.info("Embedding %d documents…", len(texts))
        embeddings = embed_texts(texts).astype("float32")
        dim = embeddings.shape[1]

        if self._index is None:
            # Inner-product index works as cosine after L2-normalisation
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(embeddings)

        for text, source, chunk_id in zip(texts, sources, chunk_ids):
            self._metadata.append(
                {"content": text, "source": source, "chunk_id": chunk_id}
            )

        self._save()
        logger.info("Added %d chunks. Total: %d", len(texts), self._index.ntotal)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-k most similar chunks.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by relevance (best first).
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no results returned.")
            return []

        q_emb = embed_query(query).astype("float32").reshape(1, -1)
        effective_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, effective_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:          # FAISS returns -1 for padded slots
                continue
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    content=meta["content"],
                    source=meta["source"],
                    score=float(score),
                    chunk_id=meta["chunk_id"],
                )
            )
        return results

    @property
    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    def document_count(self) -> int:
        return self._index.ntotal if self._index else 0
