# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ ./
RUN npm run build

# ── Stage 2: Python backend + final image ─────────────────────────────────────
FROM python:3.9-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with exact pinned versions
# torch/torchaudio pinned to 2.4 — works with both pyannote 3.x and speechbrain 1.0.2
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1

RUN pip install --no-cache-dir \
    hyperpyyaml==0.0.16 \
    speechbrain==1.0.2

RUN pip install --no-cache-dir \
    onnxruntime==1.17.3 \
    scikit-learn==1.5.2 \
    cryptography==42.0.8 \
    flask==3.0.3

RUN pip install --no-cache-dir \
    huggingface-hub==0.24.7 \
    pyannote.audio==3.3.2 \
    transformers==4.44.2 \
    datasets==2.19.0 \
    tqdm==4.66.5

# Copy Python source
COPY speaker_diarization/ ./speaker_diarization/

# Copy Flask app
COPY ui/app.py ./ui/app.py

# Copy pre-built React frontend from Stage 1
COPY --from=frontend /app/ui/dist ./ui/dist

# Copy pre-exported ONNX model (must exist before building image)
# Run: PYTHONPATH=speaker_diarization python speaker_diarization/models/export_onnx.py --fp32
COPY speaker_diarization/models/ecapa_tdnn_int8.onnx ./speaker_diarization/models/ecapa_tdnn_int8.onnx

# Copy environment template
COPY .env.example ./.env.example

# Set SpeechBrain to copy files instead of symlink (avoids permission issues)
ENV SB_LOCAL_FETCH_STRATEGY=copy
ENV PYTHONPATH=speaker_diarization

EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001')" || exit 1

CMD ["python", "ui/app.py"]
