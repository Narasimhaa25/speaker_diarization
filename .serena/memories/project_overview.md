# Speaker Diarization Module — Project Overview

## Purpose
Module 2 of a retail analytics pipeline. Answers: "Who spoke when?" and "Is the speaker staff or customer?"
Uses pyannote neural diarization + ECAPA-TDNN voice embeddings matched against an encrypted staff voice DB.

## Tech Stack
- **Backend**: Python (Flask), pyannote.audio, SpeechBrain ECAPA-TDNN, ONNX Runtime, AES-256-GCM
- **Frontend**: React 19 + Vite (in `ui/`), built to `ui/dist/`, served by Flask
- **Python version**: 3.9+ (venv at `venv/` — must be recreated on macOS, was originally Windows)

## Key Commands
```bash
# Setup (first time on macOS)
rm -rf venv && python3 -m venv venv
venv/bin/pip install -r speaker_diarization/requirements.txt
venv/bin/pip install datasets==2.19.0 transformers soundfile flask

# Export ONNX model (one-time ~5min)
PYTHONPATH=speaker_diarization venv/bin/python3 speaker_diarization/models/export_onnx.py --fp32

# Run Flask backend
PYTHONPATH=speaker_diarization venv/bin/python3 ui/app.py

# React dev (with hot reload, proxies API to :5000)
cd ui && npm run dev

# React production build
cd ui && npm run build

# Run tests
PYTHONPATH=speaker_diarization venv/bin/python3 -m pytest speaker_diarization/tests/ -v
```

## File Structure
```
speaker_diarization_module_clean/
├── ui/                        ← React app (Vite) + Flask backend
│   ├── app.py                 ← Flask server (serves dist/ + API routes)
│   ├── dist/                  ← React production build
│   ├── src/
│   │   ├── App.jsx
│   │   ├── utils.js
│   │   ├── components/        ← Header, Timeline, IdPanel, PieChart, SpeakerBars, SegmentTable, AnalysisResults, WaveformCanvas
│   │   ├── pages/             ← Dashboard, Analyse, EnrollStaff, StaffDB
│   │   └── hooks/useMicRecorder.js
│   └── vite.config.js         ← proxies /analyse* /enroll /staff → :5000
├── speaker_diarization/       ← Python source (models, core, staff_db, utils, tests)
├── ami_staff.staffdb          ← encrypted staff voice DB
├── ami_staff.key              ← DB encryption key
├── amicorpus/                 ← AMI audio test files
└── .env                       ← HF_TOKEN, STAFF_DB_PATH, etc.
```

## API Endpoints (Flask)
- GET  `/ami_files` — list AMI corpus audio files
- POST `/analyse` — upload file for diarization
- POST `/analyse_sample` — analyse a server-side AMI file
- POST `/analyse_realtime` — analyse mic recording blob
- POST `/enroll` — enroll new staff (3 voice samples required)
- GET  `/staff` — list all staff
- DELETE `/staff/<id>` — delete staff
- POST `/staff/<id>/deactivate` — soft-delete
- POST `/staff/<id>/reenroll` — update embeddings
