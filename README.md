# Speaker Diarization & Staff Voice ID

> **Who spoke when? Is the speaker staff or a customer?**  
> Real-time speaker diarization with encrypted staff voice identification — powered by pyannote + ECAPA-TDNN.

---

## What it does

| Feature | Detail |
|---|---|
| **Speaker Diarization** | Detects who spoke when using pyannote 3.1 (neural) with ECAPA-TDNN fallback |
| **Staff Identification** | Matches voice embeddings against an AES-256-GCM encrypted staff database |
| **Real-time Mic** | Record from browser mic → instant analysis |
| **File Upload** | WAV, MP3, MP4, MPEG, M4A, FLAC, OGG, AAC, OPUS, AIFF, WMA |
| **Staff Enrollment** | Enroll staff voices in 3 samples via the browser UI |
| **Web UI** | React + Vite SPA with light/dark mode |

---

## Project Structure

```
speaker_diarization/          ← Python backend module
├── core/
│   ├── diarizer.py           ← Full pipeline orchestrator
│   ├── speaker_count.py      ← Speaker count estimator
│   └── staff_identifier.py   ← Cosine similarity classifier
├── models/
│   ├── diarization_model.py  ← pyannote wrapper
│   ├── ecapa_tdnn.py         ← ECAPA-TDNN ONNX embedder
│   └── export_onnx.py        ← SpeechBrain → ONNX export
├── staff_db/
│   ├── db_manager.py         ← CRUD + AES-256-GCM encryption
│   ├── schema.py             ← DB schema
│   └── similarity_search.py  ← Cosine top-1 search
├── enrollment/
│   └── enrollment_validator.py ← Audio quality checks
├── utils/
│   ├── audio_utils.py        ← Load, VAD, normalize, noise reduce
│   └── crypto_utils.py       ← AES-256-GCM key generation
└── tests/                    ← pytest suite

ui/                           ← React + Flask web app
├── app.py                    ← Flask API server (port 5001)
└── src/
    ├── pages/                ← Dashboard, Analyse, Enroll, StaffDB
    ├── components/           ← Timeline, IdPanel, PieChart, etc.
    └── hooks/                ← useMicRecorder

requirements.txt              ← All Python dependencies
setup.sh                      ← One-shot setup script
.env.example                  ← Environment template
```

---

## Quick Start

### Prerequisites

| Tool | Install |
|---|---|
| Python 3.9+ | [python.org](https://python.org) |
| Node.js 18+ | [nodejs.org](https://nodejs.org) |
| ffmpeg | `brew install ffmpeg` (macOS) |
| HuggingFace account | [huggingface.co](https://huggingface.co) — free |

### 1. Clone

```bash
git clone https://github.com/Narasimhaa25/speaker_diarization.git
cd speaker_diarization
```

### 2. Get a HuggingFace token

1. Sign up at [huggingface.co](https://huggingface.co)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** → Read scope
3. Accept model terms at:
   - [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Copy token into `.env`:

```bash
cp .env.example .env
# Edit .env → paste your token:
# HF_TOKEN=hf_your_token_here
```

### 3. Run setup (one command)

```bash
bash setup.sh
```

This installs all Python deps, exports the ONNX model, and builds the React frontend. Takes ~5 minutes on first run.

### 4. Start the app

```bash
PYTHONPATH=speaker_diarization venv/bin/python3 ui/app.py
```

Open **http://localhost:5001** in your browser.

---

## Usage

### Analyse audio
1. Go to **Analyse** tab
2. Click the 🎤 mic button → speak → click ⏹ to stop → **Analyse Recording**
3. Or drag & drop any audio file (WAV/MP3/MP4/MPEG/FLAC/OGG/AAC...)

### Enroll staff
1. Go to **Enroll** tab
2. Enter name and role
3. Record 3 voice samples (10–30s each, quiet environment)
4. Click **Save Enrollment** — embedding stored encrypted, raw audio discarded

### View results
- **Speaker Timeline** — colour-coded who spoke when (hover for details)
- **Identification Panel** — per-speaker match, similarity score, ⚠ borderline warning
- **Staff vs Customer** — donut chart breakdown
- **Segment Table** — full per-segment detail

---

## How it works

```
Audio input (mic/file)
        ↓
   ffmpeg decode → 16kHz mono WAV
        ↓
   pyannote 3.1 diarization (neural)
   └── if 0 segments → ECAPA-TDNN + AgglomerativeClustering fallback
        ↓
   Per segment: ECAPA-TDNN embedding (192-dim)
        ↓
   Cosine similarity vs AES-256-GCM staff DB
   └── score ≥ threshold (default 0.90) → Staff
   └── score < threshold → Customer
        ↓
   JSON response → React UI
```

---

## Configuration

`.env` file:

```env
HF_TOKEN=hf_your_token_here     # Required for pyannote neural model
STAFF_DB_PATH=ami_staff.staffdb  # Path to staff voice database
STAFF_THRESHOLD=0.90             # Similarity threshold (0.85–0.95 recommended)
DEMO_SECONDS=120                 # Max audio duration to process
```

**Threshold guide:**

| Value | Behaviour |
|---|---|
| `0.95` | Very strict — only enroll voices match |
| `0.90` | Recommended — low false matches |
| `0.85` | Moderate |
| `0.75` | Permissive — risk of false staff matches |

---

## Run tests

```bash
PYTHONPATH=speaker_diarization venv/bin/python3 -m pytest speaker_diarization/tests/ -v
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Diarization | [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) |
| Speaker embeddings | ECAPA-TDNN via [SpeechBrain](https://speechbrain.github.io) + ONNX Runtime |
| Fallback diarizer | ECAPA-TDNN + scikit-learn AgglomerativeClustering |
| Audio decoding | librosa + ffmpeg |
| Encryption | AES-256-GCM (cryptography library) |
| Web server | Flask 3 |
| Frontend | React 19 + Vite 8 |
| Audio format support | WAV, MP3, MP4, MPEG, M4A, FLAC, OGG, AAC, OPUS, AIFF, WMA |

---

## License

Audio data: [CC BY 4.0](CCBY4.0.txt) (AMI Corpus)  
Code: MIT
