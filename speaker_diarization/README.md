# Module 1 — Speaker Diarization & Staff Voice Identification

Answers two questions for every audio chunk:
1. **Who spoke when?** — Speaker diarization via pyannote-onnx
2. **Is the speaker staff or customer?** — ECAPA-TDNN embeddings matched against the staff voice DB

This is the substrate every downstream module relies on.

---

## Directory Layout

```
speaker_diarization/
├── data/
│   ├── download_voxceleb.py        # VoxCeleb 1+2 download helper (email-signup gated)
│   ├── download_ami.py             # AMI Diarization split downloader (CC-BY 4.0, free)
│   └── dataset_loader.py          # Unified loader for both datasets
├── models/
│   ├── diarization_model.py        # pyannote-onnx wrapper → (start, end, speaker_id) tuples
│   ├── ecapa_tdnn.py               # ECAPA-TDNN 192-dim embedding extractor (ONNX INT8)
│   └── export_onnx.py              # SpeechBrain → ONNX INT8 export script
├── core/
│   ├── diarizer.py                 # Orchestrates diarization pipeline
│   ├── speaker_count.py            # Lightweight speaker-count estimator
│   └── staff_identifier.py        # Cosine-similarity staff/customer classifier
├── staff_db/
│   ├── schema.py                   # Encrypted on-device staff DB schema
│   ├── db_manager.py               # Add / remove / update staff voice signatures
│   └── similarity_search.py        # Cosine similarity search over staff embeddings
├── enrollment/
│   ├── enrollment_spec.py          # Spec for Module 8: sample length, prompts, QA
│   └── enrollment_validator.py     # Validates enrollment audio quality (SNR, length)
├── evaluation/
│   ├── evaluate_der.py             # DER evaluation against AMI ground-truth RTTM
│   ├── evaluate_identification.py  # Top-1 accuracy on staff ID
│   └── threshold_tuner.py          # Sweeps similarity threshold, plots ROC
├── utils/
│   ├── audio_utils.py              # Chunking, resampling, VAD helpers
│   └── crypto_utils.py             # AES-256-GCM encrypt/decrypt for staff DB
└── tests/
    ├── test_diarizer.py
    ├── test_staff_db.py
    └── test_enrollment_validator.py
```

---

## Datasets Used (exactly as specified)

| Dataset | Hours | License | Purpose |
|---|---|---|---|
| VoxCeleb 1 + 2 | ~2,800 | Free (email signup) | Train/evaluate ECAPA-TDNN speaker embeddings |
| AMI Diarization split | ~40 | CC-BY 4.0 | Evaluate diarization DER in conversational audio |

> **DIHARD III** is listed in the spec as "not required" — it is referenced only in comments.
> No other datasets are used.

---

## Key Targets

| Metric | Target |
|---|---|
| Diarization Error Rate (DER) | < 15% on in-domain audio |
| Staff ID top-1 accuracy | > 95% |
| Embedding dim | 192 (ECAPA-TDNN) |
| Quantisation | INT8 ONNX (mobile) |
| Staff DB size | 5–25 staff per store |

---

## Module Contracts

- **→ Module 3**: shares the 192-dim ECAPA-TDNN ONNX model and embedding contract (see `models/ecapa_tdnn.py`)
- **→ Module 7**: exposes `speaker_count` signal for group vs one-on-one detection
- **← Module 8**: enrollment recording flow — this module specifies the requirements (see `enrollment/enrollment_spec.py`)

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download datasets
python data/download_ami.py --output_dir ./datasets/ami
# VoxCeleb requires email registration — see data/download_voxceleb.py for instructions

# 2. Export ECAPA-TDNN to ONNX INT8
python models/export_onnx.py --output models/ecapa_tdnn_int8.onnx

# 3. Run diarization on an audio file
python core/diarizer.py --audio path/to/audio.wav --staff_db path/to/store_db.enc

# 4. Evaluate DER on AMI
python evaluation/evaluate_der.py --ami_dir ./datasets/ami --model models/diarization.onnx

# 5. Tune staff/customer threshold
python evaluation/threshold_tuner.py --embeddings_dir ./datasets/voxceleb --plot
```
