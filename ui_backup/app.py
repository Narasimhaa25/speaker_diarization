"""
Module 2 — Web UI
Flask app: upload audio → diarization + staff identification → visual results
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "speaker_diarization"))

ONNX_MODEL  = ROOT / "speaker_diarization/models/ecapa_tdnn_int8.onnx"
STAFF_DB    = ROOT / "ami_staff.staffdb"
STAFF_KEY   = ROOT / "ami_staff.key"
AMI_DIR     = ROOT / "amicorpus"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB max upload

ALLOWED = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


# Global error handler to return JSON for all exceptions
@app.errorhandler(Exception)
def handle_exception(error):
    """Catch all exceptions and return JSON error response."""
    print(f"\n✗ Flask Error: {error}")
    traceback.print_exc()
    return jsonify({"error": f"Server error: {str(error)}"}), 500

# ── lazy-load heavy models once ───────────────────────────────────────────────
_embedder      = None
_diar_model    = None
_staff_db      = None
_identifier    = None


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


def get_identifier():
    global _identifier
    if _identifier is None:
        from core.staff_identifier import StaffIdentifier
        _identifier = StaffIdentifier(get_staff_db(), threshold=0.75)  # Production threshold
    return _identifier


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
    staff_list = []
    db = get_staff_db()
    if db:
        staff_list = db.list_staff()
    return render_template("index.html", staff_list=staff_list, ami_files=get_ami_files())


@app.route("/analyse_sample", methods=["POST"])
def analyse_sample():
    """Analyse a pre-existing AMI file by server path."""
    data = request.get_json()
    rel_path     = data.get("path", "")
    max_duration = float(data.get("max_duration", 120))

    # Security: only allow files inside amicorpus/
    full_path = (ROOT / rel_path).resolve()
    if not str(full_path).startswith(str(AMI_DIR.resolve())):
        return jsonify({"error": "Access denied"}), 403
    if not full_path.exists():
        return jsonify({"error": f"File not found: {rel_path}"}), 404

    return _run_pipeline(str(full_path), 0.75, full_path.name, max_duration)


@app.route("/analyse", methods=["POST"])
def analyse():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format {ext}. Use WAV/MP3/M4A/FLAC"}), 400

    max_duration = float(request.form.get("max_duration", 120))

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        return _run_pipeline(tmp_path, 0.75, f.filename, max_duration)
    finally:
        os.unlink(tmp_path)


def _run_pipeline(audio_path: str, threshold: float, filename: str, max_duration: float = 120):
    import numpy as np
    from utils.audio_utils import load_audio, normalize_audio
    from core.staff_identifier import StaffIdentifier

    t0 = time.perf_counter()

    # Load audio and trim to max_duration
    try:
        audio, sr = load_audio(audio_path, target_sr=16_000)
    except Exception as e:
        print(f"⚠ Audio loading failed for {audio_path}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Could not load audio: {str(e)}. Ensure the file is a valid WAV/MP3/M4A/FLAC/OGG and properly encoded."}), 400
    total_duration = len(audio) / sr
    if max_duration and len(audio) > int(max_duration * sr):
        audio = audio[:int(max_duration * sr)]
    duration = len(audio) / sr

    # Speaker count
    from core.speaker_count import SpeakerCountEstimator
    est_count = SpeakerCountEstimator().estimate_heuristic(audio, sr)

    # Diarization
    try:
        diar_model = get_diar_model()
        segments = diar_model.diarize(audio)
        diar_mode = "pyannote (neural)"
    except Exception as e:
        import traceback
        print(f"\n⚠ pyannote diarize() failed: {e}")
        traceback.print_exc()
        segments, diar_mode = _energy_fallback(audio, sr), "energy fallback"

    # Embed + identify
    embedder   = get_embedder()
    identifier = StaffIdentifier(get_staff_db(), threshold=0.75)

    results = []
    speaker_totals: dict[str, float] = {}
    staff_time: dict[str, float] = {}
    customer_time = 0.0

    for seg in segments:
        chunk = audio[int(seg.start * sr): int(seg.end * sr)]
        if len(chunk) < int(0.5 * sr):
            continue
        chunk = normalize_audio(chunk)
        try:
            emb = embedder.embed(chunk, sr)
        except ValueError:
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

    # Assign a colour per speaker for the timeline
    colours = ["#4f8ef7", "#f7954f", "#4fc97a", "#f74f4f", "#b44ff7", "#f7e04f"]
    speaker_ids = sorted(speaker_totals.keys())
    speaker_colour = {s: colours[i % len(colours)] for i, s in enumerate(speaker_ids)}

    db = get_staff_db()
    enrolled = db.list_staff() if db else []

    return jsonify({
        "filename":       filename,
        "duration":       round(duration, 1),
        "total_duration": round(total_duration, 1),
        "est_speakers":  est_count,
        "diar_mode":     diar_mode,
        "elapsed":       round(elapsed, 1),
        "rtf":           round(elapsed / duration, 2) if duration > 0 else 0,
        "threshold":     threshold,
        "segments":      results,
        "staff_count":   staff_count,
        "customer_count": customer_count,
        "staff_time":    {k: round(v, 1) for k, v in staff_time.items()},
        "customer_time": round(customer_time, 1),
        "speaker_totals": {k: round(v, 1) for k, v in speaker_totals.items()},
        "speaker_colour": speaker_colour,
        "enrolled_staff": [s["name"] for s in enrolled],
    })


def _energy_fallback(audio, sr):
    """Simple VAD-based fallback when pyannote unavailable."""
    import numpy as np
    import librosa
    from utils.audio_utils import voice_activity_detection, normalize_audio
    from models.diarization_model import DiarizedSegment

    audio_n = normalize_audio(audio)
    vad = voice_activity_detection(audio_n, sr, energy_threshold_db=-40.0, min_speech_ms=200)
    segments, prev_mfcc, speaker_mfccs, label = [], None, {}, 0
    for start, end in vad:
        if end - start < 0.3:
            continue
        chunk = audio_n[int(start * sr): int(end * sr)]
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13).mean(axis=1)
        if prev_mfcc is not None:
            best_l = min(speaker_mfccs, key=lambda l: np.linalg.norm(mfcc - speaker_mfccs[l]))
            if np.linalg.norm(mfcc - speaker_mfccs[best_l]) > 10:
                label = max(speaker_mfccs) + 1
            else:
                label = best_l
        speaker_mfccs[label] = mfcc
        segments.append(DiarizedSegment(start=start, end=end, speaker_id=f"SPEAKER_{label:02d}"))
        prev_mfcc = mfcc
    return segments


@app.route("/analyse_realtime", methods=["POST"])
def analyse_realtime():
    """
    Accept a raw audio blob captured from the browser microphone (WebM/Opus or WAV),
    run the full diarization + staff-ID pipeline, and return JSON results.
    This endpoint is identical in behaviour to /analyse but is purpose-built for the
    live-microphone flow: smaller chunks, no filename display requirement.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio blob received"}), 400

    blob = request.files["audio"]
    # Browser MediaRecorder emits audio/webm;codecs=opus by default on most browsers.
    # Fall back to .wav so ffmpeg/librosa can decode either format.
    ext = ".webm"
    ct  = blob.content_type or ""
    if "wav" in ct:
        ext = ".wav"
    elif "ogg" in ct:
        ext = ".ogg"
    elif "mp4" in ct or "m4a" in ct:
        ext = ".m4a"

    threshold    = float(request.form.get("threshold", 0.75))  # Production threshold
    max_duration = float(request.form.get("max_duration", 120))

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        blob.save(tmp.name)
        tmp_path = tmp.name

    try:
        return _run_pipeline(tmp_path, threshold, "microphone_input" + ext, max_duration)
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
    from utils.audio_utils import load_audio, normalize_audio, reduce_noise
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
            audio, sr = load_audio(tmp_path, target_sr=16_000)
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

        # ✓ PRODUCTION: Check duration FIRST (10-30 seconds)
        duration = len(audio) / sr
        sample_durations.append(duration)
        
        if duration < 10.0:
            validation_results.append({
                "index": i,
                "status": "FAIL",
                "duration_sec": round(duration, 1),
                "reason": f"Too short: {round(duration, 1)}s (minimum 10 seconds)"
            })
            continue
        
        if duration > 30.0:
            validation_results.append({
                "index": i,
                "status": "FAIL",
                "duration_sec": round(duration, 1),
                "reason": f"Too long: {round(duration, 1)}s (maximum 30 seconds)"
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

    # ✓ PRODUCTION: All 3 must pass
    if len(embeddings) != 3:
        failed_count = 3 - len(embeddings)
        failed_indices = [v["index"] for v in validation_results if v["status"] == "FAIL"]
        return jsonify({
            "error": f"{failed_count} of 3 samples failed quality checks. All 3 are required for enrollment.",
            "failed_samples": failed_indices,
            "validation": validation_results,
            "guidance": "Please re-record the failed samples. Ensure: quiet background, clear speech, 10-30 seconds each.",
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
    import tempfile, os
    from utils.audio_utils import load_audio, normalize_audio
    from enrollment.enrollment_validator import EnrollmentValidator

    embedder   = get_embedder()
    validator  = EnrollmentValidator()
    embeddings = []

    for blob in audio_files:
        ext = ".webm"
        ct  = blob.content_type or ""
        if "wav"  in ct: ext = ".wav"
        elif "ogg" in ct: ext = ".ogg"
        elif "mp4" in ct or "m4a" in ct: ext = ".m4a"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            blob.save(tmp.name)
            tmp_path = tmp.name

        try:
            audio, sr = load_audio(tmp_path, target_sr=16_000)
            vr = validator.validate_utterance(audio, sr)
            if vr.passed:
                audio = normalize_audio(audio)
                emb = embedder.embed(audio, sr)
                embeddings.append(emb)
        except Exception:
            pass
        finally:
            try: os.unlink(tmp_path)
            except: pass

    if not embeddings:
        return jsonify({"error": "No valid recordings passed quality checks"}), 400

    avg_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm

    try:
        db.update_staff(staff_id, avg_emb, n_samples=len(embeddings))
        global _identifier
        _identifier = None
        return jsonify({"success": True, "staff_id": staff_id, "n_samples": len(embeddings)})
    except KeyError:
        return jsonify({"error": f"Staff ID not found: {staff_id}"}), 404


if __name__ == "__main__":
    print("\n  Module 2 — Speaker Diarization UI")
    print("  Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)
