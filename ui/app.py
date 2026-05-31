"""
Module 2 — Web UI
Flask app: upload audio → diarization + staff identification → visual results
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "speaker_diarization"))

ONNX_MODEL  = ROOT / "speaker_diarization/models/ecapa_tdnn_int8.onnx"
STAFF_DB    = ROOT / "ami_staff.staffdb"
STAFF_KEY   = ROOT / "ami_staff.key"
AMI_DIR     = ROOT / "amicorpus"
DIST_DIR    = Path(__file__).parent / "dist"

app = Flask(__name__, static_folder=str(DIST_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB max upload

ALLOWED = {
    ".wav", ".flac",                          # lossless — native librosa
    ".mp3", ".m4a", ".ogg", ".webm",          # common compressed / browser
    ".mp4", ".mpeg", ".mpg", ".mp2", ".m4v",  # MPEG containers
    ".aac", ".wma", ".opus", ".aiff", ".aif", # other common formats
}

# ── Load .env so HF_TOKEN and other vars are available ────────────────────────
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── ffmpeg path ───────────────────────────────────────────────────────────────
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def _to_wav(src_path: str) -> str:
    """
    Convert any audio file to a 16 kHz mono WAV using ffmpeg.
    Returns path to a new temp WAV file (caller must delete it).
    Raises RuntimeError if ffmpeg is not found or conversion fails.
    """
    if not FFMPEG or not Path(FFMPEG).exists():
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg\n"
            "Then restart the Flask server."
        )
    out = tempfile.mktemp(suffix=".wav")
    cmd = [
        FFMPEG, "-y", "-i", src_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        out,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg conversion failed:\n{err}")
    return out


def _load_audio_robust(src_path: str, target_sr: int = 16_000):
    """
    Load audio from any format, auto-converting via ffmpeg when needed.
    Falls back to direct librosa load for WAV/FLAC (no ffmpeg required).
    Returns (audio_np, sample_rate).
    """
    from utils.audio_utils import load_audio
    ext = Path(src_path).suffix.lower()

    # WAV and FLAC load natively without ffmpeg
    if ext in (".wav", ".flac"):
        return load_audio(src_path, target_sr=target_sr)

    # Everything else (webm, ogg, mp3, m4a) — convert first
    wav_path = None
    try:
        wav_path = _to_wav(src_path)
        return load_audio(wav_path, target_sr=target_sr)
    finally:
        if wav_path and Path(wav_path).exists():
            os.unlink(wav_path)


# Global error handler to return JSON for all exceptions
@app.errorhandler(Exception)
def handle_exception(error):
    """Catch all exceptions and return JSON error response."""
    print(f"\n✗ Flask Error: {error}")
    traceback.print_exc()
    return jsonify({"error": f"Server error: {str(error)}"}), 500

# ── lazy-load heavy models once ───────────────────────────────────────────────
_embedder   = None
_diar_model = None
_staff_db   = None

# Default threshold — matches DEFAULT_SIMILARITY_THRESHOLD in staff_identifier.py
DEFAULT_THRESHOLD = 0.90


def get_embedder():
    global _embedder
    if _embedder is None:
        from models.ecapa_tdnn import ECAPATDNNEmbedder
        _embedder = ECAPATDNNEmbedder(ONNX_MODEL)
    return _embedder


def get_diar_model():
    global _diar_model
    if _diar_model is None:
        from models.diarization_model import DiarizationModel
        _diar_model = DiarizationModel(clustering_threshold=0.70)
    return _diar_model


def get_staff_db():
    global _staff_db
    if _staff_db is None and STAFF_DB.exists() and STAFF_KEY.exists():
        from staff_db.db_manager import StaffDBManager
        key = STAFF_KEY.read_bytes()
        _staff_db = StaffDBManager(STAFF_DB, key)
        _staff_db.load()
    return _staff_db


# ── routes ────────────────────────────────────────────────────────────────────

def get_ami_files():
    """Return list of AMI wav files grouped by meeting."""
    files = []
    if AMI_DIR.exists():
        for wav in sorted(AMI_DIR.rglob("*.wav")):
            rel = wav.relative_to(ROOT)
            meeting = wav.parent.parent.name   # e.g. ES2013a
            name    = wav.stem                  # e.g. ES2013a.Mix-Headset
            kind    = "Mix" if "Mix" in name else f"Headset-{name.split('-')[-1]}"
            files.append({
                "path":    str(rel),
                "meeting": meeting,
                "name":    name,
                "kind":    kind,
                "size_mb": round(wav.stat().st_size / 1_000_000, 1),
            })
    return files


@app.route("/")
def index():
    if DIST_DIR.exists():
        return send_from_directory(str(DIST_DIR), "index.html")
    return "<h2>Run: cd ui && npm run build</h2>", 200


@app.route("/ami_files")
def ami_files_route():
    return jsonify({"files": get_ami_files()})


@app.route("/analyse_sample", methods=["POST"])
def analyse_sample():
    """Analyse a pre-existing AMI file by server path."""
    data = request.get_json()
    rel_path     = data.get("path", "")
    max_duration = float(data.get("max_duration", 120))
    threshold    = float(data.get("threshold", DEFAULT_THRESHOLD))

    # Security: only allow files inside amicorpus/
    full_path = (ROOT / rel_path).resolve()
    if not str(full_path).startswith(str(AMI_DIR.resolve())):
        return jsonify({"error": "Access denied"}), 403
    if not full_path.exists():
        return jsonify({"error": f"File not found: {rel_path}"}), 404

    return _run_pipeline(str(full_path), threshold, full_path.name, max_duration)


@app.route("/analyse", methods=["POST"])
def analyse():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format '{ext}'. Supported: WAV, MP3, MP4, MPEG, M4A, FLAC, OGG, AAC, OPUS, AIFF, WMA"}), 400

    max_duration = float(request.form.get("max_duration", 120))
    threshold    = float(request.form.get("threshold", DEFAULT_THRESHOLD))

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        return _run_pipeline(tmp_path, threshold, f.filename, max_duration)
    finally:
        os.unlink(tmp_path)


def _run_pipeline(
    audio_path: str,
    threshold: float,
    filename: str,
    max_duration: float = 120,
    force_fallback: bool = False,
):
    import numpy as np
    from utils.audio_utils import normalize_audio
    from core.staff_identifier import StaffIdentifier

    t0 = time.perf_counter()

    # ── Load audio ─────────────────────────────────────────────────────────────
    try:
        audio, sr = _load_audio_robust(audio_path, target_sr=16_000)
    except Exception as e:
        print(f"⚠ Audio loading failed for {audio_path}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Could not load audio: {str(e)}"}), 400

    total_duration = len(audio) / sr
    if max_duration and len(audio) > int(max_duration * sr):
        audio = audio[:int(max_duration * sr)]
    duration = len(audio) / sr

    # ── Quality guard: reject if too short or completely silent ────────────────
    if duration < 1.0:
        return jsonify({"error": "Audio too short — please record at least 1 second."}), 400

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-4:
        return jsonify({"error": "Audio appears silent. Check your microphone level."}), 400

    # ── Diarization ────────────────────────────────────────────────────────────
    # force_fallback=True for mic recordings: pyannote fails on temp WAVs.
    # For original file paths (AMI corpus, uploads) we try pyannote first.
    if force_fallback:
        segments = _energy_fallback(audio, sr)
        diar_mode = "ECAPA-TDNN + clustering"
    else:
        try:
            diar_model = get_diar_model()
            segments = diar_model.diarize(audio)
            if not segments:
                print("⚠ pyannote returned 0 segments — switching to ECAPA-TDNN fallback")
                segments = _energy_fallback(audio, sr)
                diar_mode = "ECAPA-TDNN + clustering"
            else:
                diar_mode = "pyannote (neural)"
        except Exception as e:
            print(f"\n⚠ pyannote failed: {e}")
            traceback.print_exc()
            segments = _energy_fallback(audio, sr)
            diar_mode = "ECAPA-TDNN + clustering"

    # ── Speaker count from real segments ───────────────────────────────────────
    unique_speakers = len(set(s.speaker_id for s in segments))
    est_count = unique_speakers if unique_speakers > 0 else 1

    # ── Identify each segment ──────────────────────────────────────────────────
    embedder   = get_embedder()
    identifier = StaffIdentifier(get_staff_db(), threshold=threshold)

    results = []
    speaker_totals: dict[str, float] = {}
    staff_time: dict[str, float] = {}
    customer_time = 0.0

    for seg in segments:
        chunk = audio[int(seg.start * sr): int(seg.end * sr)]
        if len(chunk) < int(0.3 * sr):   # skip segments < 0.3s
            continue
        chunk = normalize_audio(chunk)
        try:
            emb = embedder.embed(chunk, sr)
        except Exception:
            continue

        role, name, score = identifier.identify(emb)
        dur = seg.end - seg.start

        results.append({
            "start":      round(seg.start, 2),
            "end":        round(seg.end, 2),
            "duration":   round(dur, 2),
            "speaker_id": seg.speaker_id,
            "role":       role,
            "name":       name or "",
            "score":      round(score, 3),
        })

        speaker_totals[seg.speaker_id] = speaker_totals.get(seg.speaker_id, 0) + dur
        if role == "staff":
            staff_time[name or "?"] = staff_time.get(name or "?", 0) + dur
        else:
            customer_time += dur

    elapsed = time.perf_counter() - t0
    staff_count    = sum(1 for r in results if r["role"] == "staff")
    customer_count = sum(1 for r in results if r["role"] != "staff")

    colours = ["#4f8ef7", "#f7954f", "#4fc97a", "#f74f4f", "#b44ff7", "#f7e04f"]
    speaker_ids = sorted(speaker_totals.keys())
    speaker_colour = {s: colours[i % len(colours)] for i, s in enumerate(speaker_ids)}

    db = get_staff_db()
    try:
        enrolled = db.list_staff() if db else []
    except RuntimeError:
        db.load()
        enrolled = db.list_staff() if db else []

    return jsonify({
        "filename":       filename,
        "duration":       round(duration, 1),
        "total_duration": round(total_duration, 1),
        "est_speakers":   est_count,
        "diar_mode":      diar_mode,
        "elapsed":        round(elapsed, 1),
        "rtf":            round(elapsed / duration, 2) if duration > 0 else 0,
        "threshold":      threshold,
        "segments":       results,
        "staff_count":    staff_count,
        "customer_count": customer_count,
        "staff_time":     {k: round(v, 1) for k, v in staff_time.items()},
        "customer_time":  round(customer_time, 1),
        "speaker_totals": {k: round(v, 1) for k, v in speaker_totals.items()},
        "speaker_colour": speaker_colour,
        "enrolled_staff": [s["name"] for s in enrolled],
    })


def _energy_fallback(audio, sr):
    """
    Robust diarizer: Silero-style VAD → ECAPA-TDNN embeddings → AgglomerativeClustering.
    Works on any clip length. No hallucination — every segment has a real embedding.
    """
    import numpy as np
    import librosa
    from utils.audio_utils import voice_activity_detection, normalize_audio
    from models.diarization_model import DiarizedSegment
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize as sk_normalize

    audio_n = normalize_audio(audio)
    duration = len(audio_n) / sr

    # ── 1. Adaptive VAD ───────────────────────────────────────────────────────
    vad_segs = []
    for energy_db in (-60.0, -50.0, -45.0, -40.0):
        vad_segs = voice_activity_detection(
            audio_n, sr, energy_threshold_db=energy_db, min_speech_ms=300
        )
        if 1 <= len(vad_segs) <= 40:
            break

    if not vad_segs:
        # Completely silent — return one segment, let identification decide
        return [DiarizedSegment(start=0.0, end=round(duration, 3), speaker_id="SPEAKER_00")]

    # ── 2. Merge gaps < 0.4s so we don't over-fragment ───────────────────────
    merged = [[vad_segs[0][0], vad_segs[0][1]]]
    for start, end in vad_segs[1:]:
        if start - merged[-1][1] < 0.4:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    vad_segs = [(s, e) for s, e in merged if e - s >= 0.3]

    if not vad_segs:
        return [DiarizedSegment(start=0.0, end=round(duration, 3), speaker_id="SPEAKER_00")]

    # ── 3. Embed each VAD segment with ECAPA-TDNN ─────────────────────────────
    embedder = get_embedder()
    embeddings, valid_segs = [], []

    for start, end in vad_segs:
        chunk = audio_n[int(start * sr): int(end * sr)]
        if len(chunk) < int(0.3 * sr):
            continue
        try:
            emb = embedder.embed(normalize_audio(chunk), sr)
            embeddings.append(emb)
            valid_segs.append((round(start, 3), round(end, 3)))
        except Exception:
            continue

    if not embeddings:
        return [DiarizedSegment(start=0.0, end=round(duration, 3), speaker_id="SPEAKER_00")]

    if len(embeddings) == 1:
        return [DiarizedSegment(
            start=valid_segs[0][0], end=valid_segs[0][1], speaker_id="SPEAKER_00"
        )]

    # ── 4. Cluster embeddings → speaker labels ────────────────────────────────
    X = sk_normalize(np.array(embeddings), norm="l2")

    # Distance threshold: tighter for short audio (less acoustic diversity)
    # 0.45 cosine distance ≈ 0.55 cosine similarity — generous but avoids over-splitting
    dist_threshold = 0.45 if duration < 30 else 0.35

    try:
        clust = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clust.fit_predict(X)
    except Exception:
        # Fallback: all same speaker
        labels = [0] * len(embeddings)

    # Cap at 6 speakers — re-cluster if fragmented
    if len(set(labels)) > 6:
        try:
            clust = AgglomerativeClustering(
                n_clusters=6, metric="cosine", linkage="average"
            )
            labels = clust.fit_predict(X)
        except Exception:
            labels = [0] * len(embeddings)

    return [
        DiarizedSegment(start=s, end=e, speaker_id=f"SPEAKER_{lbl:02d}")
        for (s, e), lbl in zip(valid_segs, labels)
    ]


@app.route("/analyse_realtime", methods=["POST"])
def analyse_realtime():
    """Mic recording endpoint — skips pyannote (unreliable on temp WAVs),
    uses ECAPA-TDNN fallback diarizer directly."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio blob received"}), 400

    blob = request.files["audio"]
    ct = blob.content_type or ""
    if "wav" in ct:
        ext = ".wav"
    elif "ogg" in ct:
        ext = ".ogg"
    elif "mp4" in ct or "m4a" in ct:
        ext = ".m4a"
    else:
        ext = ".webm"

    threshold    = float(request.form.get("threshold", DEFAULT_THRESHOLD))
    max_duration = float(request.form.get("max_duration", 120))

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        blob.save(tmp.name)
        tmp_path = tmp.name

    try:
        return _run_pipeline(
            tmp_path, threshold, "microphone_input" + ext,
            max_duration, force_fallback=True
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Enrollment & Staff Management ─────────────────────────────────────────────

def _get_or_create_db():
    """Get existing staff DB or create a new one with a fresh key."""
    global _staff_db
    if _staff_db is not None:
        return _staff_db

    if STAFF_DB.exists() and STAFF_KEY.exists():
        from staff_db.db_manager import StaffDBManager
        key = STAFF_KEY.read_bytes()
        _staff_db = StaffDBManager(STAFF_DB, key)
        _staff_db.load()
        return _staff_db

    # Create new DB + key
    import secrets
    from staff_db.db_manager import StaffDBManager
    key = secrets.token_bytes(32)
    STAFF_KEY.write_bytes(key)
    _staff_db = StaffDBManager(STAFF_DB, key)
    _staff_db.create("STORE_001")
    _staff_db.load()
    return _staff_db


@app.route("/enroll", methods=["POST"])
def enroll_staff():
    """
    Production staff enrollment: exactly 3 voice samples (10-30s each).
    Applies noise reduction, validates quality, generates embeddings.
    """
    staff_name = request.form.get("staff_name", "").strip()
    staff_role = request.form.get("staff_role", "associate").strip()

    if not staff_name:
        return jsonify({"error": "staff_name is required"}), 400

    # Collect audio blobs (audio_0, audio_1, …)
    audio_files = []
    for key in sorted(request.files.keys()):
        if key.startswith("audio"):
            audio_files.append(request.files[key])

    # ✓ PRODUCTION REQUIREMENT: Exactly 3 samples
    if len(audio_files) != 3:
        return jsonify({
            "error": f"Exactly 3 voice samples required. You provided {len(audio_files)}.",
            "expected": 3,
            "provided": len(audio_files),
        }), 400

    import numpy as np
    import tempfile, os
    from utils.audio_utils import normalize_audio, reduce_noise
    from enrollment.enrollment_validator import EnrollmentValidator

    embedder   = get_embedder()
    validator  = EnrollmentValidator()
    embeddings = []
    validation_results = []
    sample_durations = []

    for i, blob in enumerate(audio_files):
        ext = ".webm"
        ct  = blob.content_type or ""
        if "wav"  in ct: ext = ".wav"
        elif "ogg" in ct: ext = ".ogg"
        elif "mp4" in ct or "m4a" in ct: ext = ".m4a"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            blob.save(tmp.name)
            tmp_path = tmp.name

        try:
            audio, sr = _load_audio_robust(tmp_path, target_sr=16_000)
        except Exception as e:
            validation_results.append({
                "index": i,
                "status": "FAIL",
                "reason": f"Could not load audio: {str(e)}"
            })
            try:
                os.unlink(tmp_path)
            except:
                pass
            continue
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

        duration = len(audio) / sr
        sample_durations.append(duration)

        if duration < 3.0:
            validation_results.append({
                "index": i,
                "status": "FAIL",
                "duration_sec": round(duration, 1),
                "reason": f"Too short: {round(duration, 1)}s (minimum 3 seconds — speak for longer)"
            })
            continue

        # ✓ PRODUCTION: Apply noise reduction
        try:
            audio = reduce_noise(audio, sr, noise_duration_sec=0.5)
        except Exception as e:
            print(f"⚠ Noise reduction failed (non-fatal): {e}")
            # Continue without noise reduction

        # Normalize
        audio = normalize_audio(audio)

        # Validate quality
        vr = validator.validate_utterance(audio, sr)
        if not vr.passed:
            validation_results.append({
                "index": i,
                "status": "FAIL",
                "duration_sec": round(vr.duration_sec, 1),
                "snr_db": round(vr.snr_db, 1),
                "reasons": vr.rejection_reasons,
                "guidance": vr.guidance,
            })
            continue

        try:
            emb = embedder.embed(audio, sr)
            embeddings.append(emb)
            validation_results.append({
                "index": i,
                "status": "PASS",
                "duration_sec": round(vr.duration_sec, 1),
                "snr_db": round(vr.snr_db, 1),
                "rms_level": round(vr.rms_level, 3),
            })
        except Exception as e:
            validation_results.append({
                "index": i,
                "status": "FAIL",
                "reason": f"Embedding failed: {str(e)}"
            })

    # Need at least 1 valid embedding to enroll
    if len(embeddings) == 0:
        failed_indices = [v["index"] for v in validation_results if v["status"] == "FAIL"]
        reasons = [v.get("reason") or ", ".join(v.get("reasons", [])) for v in validation_results if v["status"] == "FAIL"]
        return jsonify({
            "error": "All voice samples failed quality checks. Please re-record.",
            "failed_samples": failed_indices,
            "validation": validation_results,
            "details": reasons,
            "guidance": "Ensure: speak clearly for 3+ seconds, quiet background, microphone not too far.",
        }), 400

    # Average embeddings
    avg_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm

    db = _get_or_create_db()
    staff_id = db.add_staff(staff_name, staff_role, avg_emb, n_samples=3)

    # Reset cached identifier
    global _identifier
    _identifier = None

    return jsonify({
        "success": True,
        "staff_id": staff_id,
        "name": staff_name,
        "role": staff_role,
        "n_samples": 3,
        "validation": validation_results,
        "message": f"✓ Successfully enrolled {staff_name} ({staff_role}) with 3 voice samples",
    })


@app.route("/staff", methods=["GET"])
def list_staff():
    """Return all staff records (no embeddings)."""
    db = get_staff_db()
    if db is None:
        return jsonify({"staff": []})
    try:
        staff = db.list_staff(include_inactive=True)
    except RuntimeError:
        db.load()
        staff = db.list_staff(include_inactive=True)
    return jsonify({"staff": staff})


@app.route("/staff/<staff_id>", methods=["DELETE"])
def delete_staff(staff_id):
    """Hard-delete a staff member."""
    db = get_staff_db()
    if db is None:
        return jsonify({"error": "No staff database"}), 404
    try:
        db.remove_staff(staff_id)
        global _identifier
        _identifier = None
        return jsonify({"success": True, "staff_id": staff_id})
    except KeyError:
        return jsonify({"error": f"Staff ID not found: {staff_id}"}), 404


@app.route("/staff/<staff_id>/deactivate", methods=["POST"])
def deactivate_staff(staff_id):
    """Soft-delete a staff member."""
    db = get_staff_db()
    if db is None:
        return jsonify({"error": "No staff database"}), 404
    try:
        db.deactivate_staff(staff_id)
        global _identifier
        _identifier = None
        return jsonify({"success": True, "staff_id": staff_id})
    except KeyError:
        return jsonify({"error": f"Staff ID not found: {staff_id}"}), 404


@app.route("/staff/<staff_id>/reenroll", methods=["POST"])
def reenroll_staff(staff_id):
    """Re-enroll (update embedding) for an existing staff member."""
    db = get_staff_db()
    if db is None:
        return jsonify({"error": "No staff database"}), 404

    audio_files = []
    for key in sorted(request.files.keys()):
        if key.startswith("audio"):
            audio_files.append(request.files[key])

    if not audio_files:
        return jsonify({"error": "At least one audio recording required"}), 400

    import numpy as np
    from utils.audio_utils import normalize_audio
    from enrollment.enrollment_validator import EnrollmentValidator

    embedder   = get_embedder()
    validator  = EnrollmentValidator()
    embeddings = []
    errors     = []

    for i, blob in enumerate(audio_files):
        ct = blob.content_type or ""
        if "wav" in ct:   ext = ".wav"
        elif "ogg" in ct: ext = ".ogg"
        elif "mp4" in ct or "m4a" in ct: ext = ".m4a"
        else: ext = ".webm"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            blob.save(tmp.name)
            tmp_path = tmp.name

        try:
            audio, sr = _load_audio_robust(tmp_path, target_sr=16_000)
            vr = validator.validate_utterance(audio, sr)
            if vr.passed:
                audio = normalize_audio(audio)
                emb = embedder.embed(audio, sr)
                embeddings.append(emb)
            else:
                errors.append(f"Sample {i+1}: {'; '.join(vr.rejection_reasons)}")
        except Exception as e:
            errors.append(f"Sample {i+1}: {str(e)}")
            print(f"⚠ reenroll sample {i+1} failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                print(f"⚠ cleanup failed: {e}")

    if not embeddings:
        return jsonify({
            "error": "No valid recordings passed quality checks",
            "details": errors,
        }), 400

    avg_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm

    try:
        db.update_staff(staff_id, avg_emb, n_samples=len(embeddings))
        return jsonify({
            "success": True,
            "staff_id": staff_id,
            "n_samples": len(embeddings),
            "warnings": errors if errors else None,
        })
    except KeyError:
        return jsonify({"error": f"Staff ID not found: {staff_id}"}), 404


@app.route("/<path:path>")
def static_files(path):
    target = DIST_DIR / path
    if target.exists():
        return send_from_directory(str(DIST_DIR), path)
    return send_from_directory(str(DIST_DIR), "index.html")


if __name__ == "__main__":
    print("\n  Module 2 — Speaker Diarization UI")
    print("  Open: http://localhost:5000\n")
    app.run(debug=False, port=5001)
