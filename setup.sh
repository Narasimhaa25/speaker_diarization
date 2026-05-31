#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — One-shot setup script
# Run once: bash setup.sh
# After this, just run: PYTHONPATH=speaker_diarization venv/bin/python3 demo.py
# ─────────────────────────────────────────────────────────────────────────────

set -e  # stop on any error

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }
step() { echo -e "\n${YELLOW}── $1 ────────────────────────────────────────${NC}"; }

echo "═══════════════════════════════════════════════════════════"
echo "  Module 2 Setup — Speaker Diarization & Staff Voice ID"
echo "═══════════════════════════════════════════════════════════"

# ─── Load .env ────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        warn ".env not found — created from .env.example"
        warn "Edit .env and set HF_TOKEN, then re-run this script"
        exit 1
    else
        fail ".env file not found. Create it with HF_TOKEN=hf_your_token"
    fi
fi

export $(grep -v '^#' .env | xargs)

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "hf_your_token_here" ]; then
    fail "HF_TOKEN not set in .env. Get a token at https://huggingface.co/settings/tokens"
fi
ok "Environment loaded from .env"

# ─── Step 1: Python venv ──────────────────────────────────────────────────────
step "1/7  Creating Python virtual environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "venv created"
else
    ok "venv already exists — skipping"
fi

# ─── Step 2: Install all dependencies ────────────────────────────────────────
step "2/7  Installing Python dependencies"
venv/bin/python3 -m pip install --upgrade pip -q

venv/bin/python3 -m pip install -q \
    numpy scipy librosa soundfile \
    onnxruntime onnx \
    cryptography \
    speechbrain \
    "pyannote.audio>=3.1.0" \
    scikit-learn matplotlib \
    tqdm pyyaml \
    torch torchaudio \
    "datasets==2.19.0" \
    transformers \
    huggingface_hub \
    "pyannote.metrics>=3.2.0" \
    pytest pytest-cov \
    python-dotenv

ok "All packages installed"

# ─── Step 3: Apply compatibility patches ──────────────────────────────────────
step "3/7  Applying compatibility patches"
venv/bin/python3 - <<'PYEOF'
import pathlib, re

# Patch 1: lightning_fabric weights_only (PyTorch 2.6+ compat)
p = pathlib.Path("venv/lib")
matches = list(p.glob("python*/site-packages/lightning_fabric/utilities/cloud_io.py"))
for f in matches:
    t = f.read_text()
    if "weights_only=False" not in t:
        t = re.sub(
            r'(with fs\.open.*?\n\s+return torch\.load\(.*?\n\s+.*?)weights_only=weights_only,',
            r'\1weights_only=False,',
            t, flags=re.DOTALL, count=1
        )
        f.write_text(t)
        print(f"  Patched {f.name} (weights_only)")

# Patch 2: SpeechBrain k2 optional import
matches2 = list(p.glob("python*/site-packages/speechbrain/integrations/k2_fsa/__init__.py"))
for f in matches2:
    t = f.read_text()
    if "raise ImportError(MSG)" in t:
        t = t.replace("raise ImportError(MSG) from e", "pass  # k2 is optional")
        f.write_text(t)
        print(f"  Patched {f.name} (k2 optional)")

# Patch 3: SpeechBrain linecache / inspect clash
matches3 = list(p.glob("python*/site-packages/speechbrain/utils/importutils.py"))
for f in matches3:
    t = f.read_text()
    old = 'importer_frame.filename.endswith("/inspect.py")'
    new = '(importer_frame.filename.endswith("/inspect.py") or importer_frame.filename.endswith("/linecache.py"))'
    if old in t and new not in t:
        t = t.replace(old, new)
        f.write_text(t)
        print(f"  Patched {f.name} (linecache)")

print("  All patches applied")
PYEOF
ok "Patches applied"

# ─── Step 4: HuggingFace login ────────────────────────────────────────────────
step "4/7  Logging in to HuggingFace"
venv/bin/python3 - <<PYEOF
from huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("  Logged in to HuggingFace")
PYEOF
ok "HuggingFace login done"

# ─── Step 5: Export ONNX model ────────────────────────────────────────────────
step "5/7  Exporting ECAPA-TDNN to ONNX"
ONNX_PATH="speaker_diarization/models/ecapa_tdnn_int8.onnx"
if [ -f "$ONNX_PATH" ]; then
    ok "ONNX model already exists — skipping export"
else
    PYTHONPATH=speaker_diarization venv/bin/python3 \
        speaker_diarization/models/export_onnx.py --fp32 \
        --output "$ONNX_PATH"
    ok "ONNX model exported to $ONNX_PATH"
fi

# ─── Step 6: Download AMI audio ───────────────────────────────────────────────
step "6/7  Downloading AMI audio data"
MIX="amicorpus/ES2013a/audio/ES2013a.Mix-Headset.wav"
if [ -f "$MIX" ]; then
    ok "AMI audio already downloaded — skipping"
else
    if ! command -v wget &> /dev/null; then
        warn "wget not found — installing via Homebrew"
        brew install wget
    fi
    bash download_ami.sh
    ok "AMI audio downloaded"
fi

# ─── Step 7: Enroll staff voices ──────────────────────────────────────────────
step "7/7  Enrolling staff voices into DB"
if [ -f "ami_staff.staffdb" ]; then
    ok "Staff DB already exists — skipping enrollment"
else
    PYTHONPATH=speaker_diarization venv/bin/python3 - <<'PYEOF'
import sys, numpy as np
sys.path.insert(0, "speaker_diarization")
from utils.audio_utils import load_audio, normalize_audio
from utils.crypto_utils import generate_key
from staff_db.db_manager import StaffDBManager
from models.ecapa_tdnn import ECAPATDNNEmbedder
from pathlib import Path

SR = 16000
embedder = ECAPATDNNEmbedder(Path("speaker_diarization/models/ecapa_tdnn_int8.onnx"))
staff = [
    ("Headset-0", "Alice",  "manager"),
    ("Headset-1", "Bob",    "associate"),
    ("Headset-2", "Carol",  "associate"),
    ("Headset-3", "David",  "associate"),
]
key = generate_key()
Path("ami_staff.key").write_bytes(key)
db = StaffDBManager(Path("ami_staff.staffdb"), key)
db.create("AMI_STORE")
for headset, name, role in staff:
    audio, _ = load_audio(f"amicorpus/ES2013a/audio/ES2013a.{headset}.wav", target_sr=SR)
    clips = [normalize_audio(audio[t*SR:(t+8)*SR]) for t in [30, 100, 200, 400, 500]]
    db.add_staff(name, role, embedder.mean_embedding(clips), n_samples=5)
    print(f"  Enrolled {name}")
print(f"  Staff DB ready with {db.staff_count()[0]} staff")
PYEOF
    ok "Staff voices enrolled"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}  Setup complete! Everything is ready.${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Run the demo:"
echo "    PYTHONPATH=speaker_diarization venv/bin/python3 demo.py"
echo ""
echo "  Run the tests:"
echo "    PYTHONPATH=speaker_diarization venv/bin/python3 -m pytest speaker_diarization/tests/ -v"
echo ""
echo "  Evaluate DER:"
echo "    PYTHONPATH=speaker_diarization venv/bin/python3 speaker_diarization/evaluation/evaluate_der.py --split dev --n_meetings 10"
echo ""
echo "  Test on your own audio:"
echo "    PYTHONPATH=speaker_diarization venv/bin/python3 test_real_voice.py --file audio.wav --db ami_staff.staffdb"
echo ""
