"""
retriever.py — High-level RAG retriever that combines
the vector store with optional re-ranking.
"""

from __future__ import annotations

import logging
from typing import List

from backend.config import TOP_K_RETRIEVAL
from backend.models import RAGContext, RetrievedChunk
from backend.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

# Module-level shared store — initialised once by the FastAPI startup handler
_store: FAISSVectorStore | None = None


def get_store() -> FAISSVectorStore:
    """Return (or lazily create) the shared FAISS vector store."""
    global _store
    if _store is None:
        _store = FAISSVectorStore()
    return _store


def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> RAGContext:
    """
    Retrieve the most relevant knowledge-base chunks for *query*.

    Args:
        query:  The user math question (or processed version).
        top_k:  Number of chunks to retrieve.

    Returns:
        RAGContext containing retrieved chunks and the original query.
    """
    store = get_store()

    if store.is_empty:
        logger.warning("Vector store is empty — returning empty RAG context.")
        return RAGContext(query=query, chunks=[])

    chunks: List[RetrievedChunk] = store.search(query, top_k=top_k)
    logger.info("Retrieved %d chunks for query: '%s…'", len(chunks), query[:60])

    return RAGContext(query=query, chunks=chunks)


def format_context_for_llm(ctx: RAGContext) -> str:
    """
    Format retrieved chunks into a compact string for LLM prompts.
    Includes source citations so the model never needs to hallucinate them.
    """
    if not ctx.chunks:
        return "No relevant context found in knowledge base."

    parts = ["=== Retrieved Knowledge Base Context ==="]
    for i, chunk in enumerate(ctx.chunks, start=1):
        parts.append(
            f"\n[Source {i}: {chunk.source} | similarity={chunk.score:.3f}]\n"
            f"{chunk.content}"
        )
    parts.append("\n=== End of Context ===")
    return "\n".join(parts)
