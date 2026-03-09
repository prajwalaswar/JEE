# ─── Dockerfile — FastAPI Backend ────────────────────────────────────────────
# Builds the Multimodal Math Mentor FastAPI + Uvicorn service.
# Exposes port 8000.
#
# Build:  docker build -t math-mentor-backend .
# Run:    docker run -p 8000:8000 --env-file .env math-mentor-backend

FROM python:3.11-slim

# System deps — ffmpeg for audio conversion, libgomp for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer-cache friendly)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY backend/   ./backend/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY .env.example .env.example

# Directory for runtime-generated data
RUN mkdir -p data/faiss_index

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# The FAISS index is built automatically on startup (see main.py startup_event)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
