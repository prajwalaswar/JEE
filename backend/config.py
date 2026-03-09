"""
config.py — Centralised configuration for Multimodal Math Mentor.
All secrets are loaded from environment variables (never hard-coded).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── LLM providers ─────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Primary model (Groq – free tier)
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Fallback reasoning model (Google Gemini Flash – free tier)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── RAG / Embeddings ──────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
FAISS_INDEX_PATH: str = str(BASE_DIR / "data" / "faiss_index")
KNOWLEDGE_BASE_DIR: str = str(BASE_DIR / "data" / "knowledge_base")
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── Memory ────────────────────────────────────────────────────────────────────
SQLITE_DB_PATH: str = str(BASE_DIR / "data" / "memory.db")

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75")
)
OCR_BACKEND: str = os.getenv("OCR_BACKEND", "paddleocr")  # "paddleocr" | "easyocr"

# ── Audio / faster-whisper ──────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
ASR_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("ASR_CONFIDENCE_THRESHOLD", "0.70")
)
# faster-whisper runs on CPU with int8 by default (GPU: set device="cuda", compute_type="float16")
FASTER_WHISPER_DEVICE: str = os.getenv("FASTER_WHISPER_DEVICE", "cpu")
FASTER_WHISPER_COMPUTE_TYPE: str = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")

# ── Agent thresholds ──────────────────────────────────────────────────────────
VERIFIER_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("VERIFIER_CONFIDENCE_THRESHOLD", "0.80")
)

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
