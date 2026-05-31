#!/usr/bin/env bash
set -e

echo ""
echo "══════════════════════════════════════════════════"
echo "  Speaker Diarization — Setup"
echo "══════════════════════════════════════════════════"
echo ""

# ── 1. Check prerequisites ─────────────────────────────────────────────────────
command -v python3 >/dev/null || { echo "❌ python3 not found. Install from python.org"; exit 1; }
command -v node    >/dev/null || { echo "❌ node not found. Install from nodejs.org"; exit 1; }
command -v npm     >/dev/null || { echo "❌ npm not found. Install from nodejs.org"; exit 1; }
command -v ffmpeg  >/dev/null || { echo "❌ ffmpeg not found. Run: brew install ffmpeg"; exit 1; }

echo "✓ Prerequisites OK (python3, node, npm, ffmpeg)"
echo ""

# ── 2. Python virtual environment ─────────────────────────────────────────────
if [ ! -f "venv/bin/python3" ]; then
    echo "→ Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "→ Installing Python dependencies (~5 min first time)..."
venv/bin/python3 -m pip install --upgrade pip -q
venv/bin/python3 -m pip install -r requirements.txt -q
echo "✓ Python dependencies installed"
echo ""

# ── 3. Export ONNX model ───────────────────────────────────────────────────────
ONNX_PATH="speaker_diarization/models/ecapa_tdnn_int8.onnx"
if [ ! -f "$ONNX_PATH" ]; then
    echo "→ Exporting ECAPA-TDNN ONNX model (~3 min)..."
    PYTHONPATH=speaker_diarization venv/bin/python3 speaker_diarization/models/export_onnx.py --fp32
    echo "✓ ONNX model exported"
else
    echo "✓ ONNX model already exists"
fi
echo ""

# ── 4. Environment file ────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠  Created .env from template."
    echo "   Add your HuggingFace token:"
    echo "   → Go to https://huggingface.co/settings/tokens"
    echo "   → Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "   → Paste token into .env as: HF_TOKEN=hf_xxxx"
    echo ""
else
    echo "✓ .env already exists"
fi

# ── 5. React frontend ──────────────────────────────────────────────────────────
echo "→ Installing frontend dependencies..."
cd ui && npm install -q && npm run build -q && cd ..
echo "✓ Frontend built"
echo ""

echo "══════════════════════════════════════════════════"
echo "  ✓ Setup complete!"
echo ""
echo "  Start the app:"
echo "    PYTHONPATH=speaker_diarization venv/bin/python3 ui/app.py"
echo ""
echo "  Then open: http://localhost:5001"
echo "══════════════════════════════════════════════════"
echo ""
