"""
test_real_voice.py
───────────────────
Real-voice testing for the speaker diarization & staff identification pipeline.

Three modes:
  1. FILE mode   — analyse an existing WAV/MP3/M4A file
  2. RECORD mode — record from microphone, then analyse
  3. ENROLL mode — enroll a real voice as "staff", then test identification

Usage
-----
  # Analyse an existing audio file
  python test_real_voice.py --file path/to/conversation.wav

  # Record 30 seconds from mic, then analyse
  python test_real_voice.py --record --duration 30

  # Enroll yourself as staff, then record a test clip and identify
  python test_real_voice.py --enroll --name "Alice"
"""

from __future__ import annotations

import argparse
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

# ── make sure the package is on the path ─────────────────────────────────────
ROOT = Path(__file__).parent / "speaker_diarization"
sys.path.insert(0, str(ROOT))

from utils.audio_utils import load_audio, normalize_audio, estimate_snr_db
from utils.crypto_utils import generate_key
from staff_db.db_manager import StaffDBManager
from staff_db.similarity_search import SimilaritySearch
from core.staff_identifier import StaffIdentifier
from core.speaker_count import SpeakerCountEstimator
from enrollment.enrollment_validator import EnrollmentValidator

SR = 16_000   # everything resampled to 16 kHz


# ─── Microphone helpers ───────────────────────────────────────────────────────

def record_mic(duration_sec: float, label: str = "recording") -> np.ndarray:
    """Record from the default microphone. Requires: pip install sounddevice"""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run:  venv/bin/pip install sounddevice")
        sys.exit(1)

    print(f"\n🎙  Recording {label} for {duration_sec:.0f} seconds ...")
    print("    Speak now  ▶", end="", flush=True)
    for i in range(int(duration_sec)):
        time.sleep(1)
        print("█", end="", flush=True)
    print()

    audio = sd.rec(int(duration_sec * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def play_audio(audio: np.ndarray, label: str = "") -> None:
    """Play back audio through speakers."""
    try:
        import sounddevice as sd
        print(f"\n🔊  Playing back {label} ...")
        sd.play(audio, samplerate=SR)
        sd.wait()
    except ImportError:
        pass  # silently skip playback if sounddevice not available


# ─── ECAPA-TDNN embedding (with graceful fallback) ────────────────────────────

def get_embedder():
    """
    Load the ECAPA-TDNN ONNX embedder if the model file exists.
    Falls back to a random-projection stub so the rest of the pipeline
    still runs — identification results will be meaningless in stub mode.
    """
    onnx_path = ROOT / "models" / "ecapa_tdnn_int8.onnx"
    if onnx_path.exists():
        from models.ecapa_tdnn import ECAPATDNNEmbedder
        print(f"✓ Loaded ECAPA-TDNN ONNX model from {onnx_path}")
        return ECAPATDNNEmbedder(onnx_path), "real"
    else:
        print(
            "\n⚠  ONNX model not found at models/ecapa_tdnn_int8.onnx\n"
            "   Using random-projection stub — identification scores are random.\n"
            "   To get real embeddings run:\n"
            "     venv/bin/python speaker_diarization/models/export_onnx.py\n"
        )
        return _StubEmbedder(), "stub"


class _StubEmbedder:
    """Deterministic random-projection stub — same audio → same embedding."""
    DIM = 192

    def embed(self, audio: np.ndarray, sample_rate: int = SR) -> np.ndarray:
        # Use mean + std of audio as a tiny fingerprint, then project to 192-dim
        seed = int(abs(audio.mean()) * 1e6 + abs(audio.std()) * 1e4) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def mean_embedding(self, audios: list) -> np.ndarray:
        embs = np.stack([self.embed(a) for a in audios])
        m = embs.mean(axis=0)
        return (m / np.linalg.norm(m)).astype(np.float32)


# ─── Diarization (with graceful fallback) ────────────────────────────────────

def run_diarization(audio: np.ndarray):
    """
    Run pyannote diarization. Requires a HuggingFace token accepted at:
      https://huggingface.co/pyannote/speaker-diarization-3.1

    If the model download fails, falls back to a simple energy-based
    speaker segmentation so the rest of the demo still runs.
    """
    try:
        from models.diarization_model import DiarizationModel
        model = DiarizationModel(clustering_threshold=0.70)
        return model.diarize(audio), "pyannote"
    except Exception as e:
        print(f"\n⚠  pyannote diarization unavailable ({e.__class__.__name__}).")
        print("   Falling back to energy-based segmentation (less accurate).\n"
              "   For real diarization: accept the pyannote model at\n"
              "   https://huggingface.co/pyannote/speaker-diarization-3.1\n"
              "   and set HF_TOKEN in your environment.\n")
        return _energy_diarize(audio), "energy_fallback"


def _energy_diarize(audio: np.ndarray, min_dur: float = 0.3):
    """
    Energy + MFCC-based speaker diarization fallback.

    Steps:
      1. Normalize audio so quiet laptop-mic recordings are detected
      2. VAD: split into voiced segments
      3. For each segment pair, compute MFCC distance
         - Large distance (> threshold) → different speaker → flip label
         - Small distance                → same speaker    → keep label

    MFCC distance is much more reliable than spectral centroid for real
    voices — two people talking normally differ by ~30–80 MFCC units,
    while same-speaker variation is typically < 20.
    """
    import librosa
    from utils.audio_utils import voice_activity_detection, normalize_audio
    from models.diarization_model import DiarizedSegment

    # Normalize so quiet mic recordings are picked up by VAD
    audio = normalize_audio(audio, target_db=-20.0)

    # -40 dB catches phone/call recordings with compressed dynamic range
    # min_speech_ms=200 allows short conversational bursts typical in calls
    vad = voice_activity_detection(audio, SR, energy_threshold_db=-40.0, min_speech_ms=200)
    if not vad:
        return []

    # Merge gaps shorter than 1 second (breath pauses within one speaker turn)
    merged = [list(vad[0])]
    for start, end in vad[1:]:
        if start - merged[-1][1] < 1.0:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    def segment_mfcc(chunk: np.ndarray) -> np.ndarray:
        """Mean MFCC vector for a chunk — 13-dim voice fingerprint."""
        return librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=13).mean(axis=1)

    # MFCC distance threshold tuned on real laptop-mic recordings.
    # Same speaker across turns: ~3–8.  Different speakers: ~12–25+.
    MFCC_CHANGE_THRESHOLD = 10.0

    segments = []
    prev_mfcc = None
    current_label = 0
    speaker_mfccs: dict[int, np.ndarray] = {}  # label → running mean MFCC

    for start, end in merged:
        if end - start < min_dur:
            continue
        s = int(start * SR)
        e = int(end * SR)
        chunk = audio[s:e]
        mfcc = segment_mfcc(chunk)

        if prev_mfcc is None:
            # First segment — always SPEAKER_00
            current_label = 0
        else:
            # Compare against all seen speakers, assign to closest
            best_label = current_label
            best_dist = float("inf")
            for label, ref_mfcc in speaker_mfccs.items():
                dist = float(np.linalg.norm(mfcc - ref_mfcc))
                if dist < best_dist:
                    best_dist = dist
                    best_label = label

            print(f"     MFCC dist to SPEAKER_{best_label:02d}: {best_dist:.2f}  "
                  f"({'NEW SPEAKER' if best_dist > MFCC_CHANGE_THRESHOLD else 'SAME SPEAKER'})")
            if best_dist > MFCC_CHANGE_THRESHOLD:
                current_label = max(speaker_mfccs.keys()) + 1
            else:
                current_label = best_label

        # Update running mean MFCC for this speaker
        if current_label in speaker_mfccs:
            # Exponential moving average — adapts to voice drift within turn
            speaker_mfccs[current_label] = 0.7 * speaker_mfccs[current_label] + 0.3 * mfcc
        else:
            speaker_mfccs[current_label] = mfcc

        segments.append(DiarizedSegment(
            start=start,
            end=end,
            speaker_id=f"SPEAKER_{current_label:02d}",
        ))
        prev_mfcc = mfcc

    return segments


# ─── Result display ───────────────────────────────────────────────────────────

def print_segment_table(results: list, audio_duration: float) -> None:
    """Pretty-print the identified segments."""
    print(f"\n{'─'*65}")
    print(f"  {'START':>7}  {'END':>7}  {'DUR':>6}  {'SPEAKER':<12}  {'ROLE':<10}  {'SCORE'}")
    print(f"{'─'*65}")
    for r in results:
        dur = r["end"] - r["start"]
        name = r.get("staff_name") or ""
        role_display = f"{r['role'].upper():<10}"
        score_display = f"{r['score']:.3f}" if r["score"] > 0 else "  —  "
        print(
            f"  {r['start']:>6.2f}s  {r['end']:>6.2f}s  "
            f"{dur:>5.2f}s  {r['speaker_id']:<12}  "
            f"{role_display}  {score_display}  {name}"
        )
    print(f"{'─'*65}")
    total_staff    = sum(1 for r in results if r["role"] == "staff")
    total_customer = sum(1 for r in results if r["role"] == "customer")
    total_unknown  = sum(1 for r in results if r["role"] == "unknown")
    print(f"  Audio: {audio_duration:.1f}s | "
          f"Segments: {len(results)} | "
          f"Staff: {total_staff} | Customer: {total_customer} | Unknown: {total_unknown}")


# ─── Core analysis function ───────────────────────────────────────────────────

def analyse(
    audio: np.ndarray,
    embedder,
    staff_db: StaffDBManager | None,
    threshold: float = 0.82,
) -> list:
    """
    Full pipeline: audio → labelled segments.
    Returns list of dicts with start, end, speaker_id, role, staff_name, score.
    """
    print("\n── Step 1: Estimating speaker count ─────────────────────────────")
    estimator = SpeakerCountEstimator()
    count = estimator.estimate_heuristic(audio, SR)
    print(f"   Estimated speakers: {count}")

    print("\n── Step 2: Diarization (who spoke when) ─────────────────────────")
    segments, diar_mode = run_diarization(audio)
    print(f"   Mode: {diar_mode}")
    print(f"   Found {len(segments)} segments:")
    for s in segments:
        print(f"     {s.start:.2f}s – {s.end:.2f}s  →  {s.speaker_id}")

    if not segments:
        print("   No speech segments found.")
        return []

    print("\n── Step 3: Voice embedding + identification ──────────────────────")
    identifier = StaffIdentifier(staff_db, threshold=threshold) if staff_db else None

    results = []
    for seg in segments:
        chunk = audio[int(seg.start * SR): int(seg.end * SR)]
        if len(chunk) < int(0.5 * SR):
            continue

        chunk = normalize_audio(chunk)

        try:
            emb = embedder.embed(chunk, SR)
        except ValueError as e:
            print(f"   [skip] {seg.speaker_id}: {e}")
            continue

        if identifier:
            role, staff_name, score = identifier.identify(emb)
        else:
            role, staff_name, score = "unknown", None, 0.0

        results.append({
            "start":      seg.start,
            "end":        seg.end,
            "speaker_id": seg.speaker_id,
            "role":       role,
            "staff_name": staff_name,
            "score":      score,
            "embedding":  emb,
        })

    return results


# ─── Mode: FILE ───────────────────────────────────────────────────────────────

def mode_file(args):
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  FILE MODE: {path.name}")
    print(f"{'='*65}")

    print(f"\nLoading audio from {path} ...")
    audio, sr = load_audio(path, target_sr=SR)
    print(f"  Duration   : {len(audio)/SR:.2f}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  SNR        : {estimate_snr_db(audio, SR):.1f} dB")

    embedder, mode = get_embedder()

    # Optional: load a staff DB if --db was passed
    staff_db = _load_db(args) if hasattr(args, "db") and args.db else None
    if staff_db is None:
        print("\n  (No staff DB provided — all speakers will be labelled 'unknown')")
        print("  Run with --enroll first to build a staff DB, then pass --db path")

    results = analyse(audio, embedder, staff_db, threshold=args.threshold)
    print_segment_table(results, len(audio) / SR)

    if mode == "stub":
        print("\n⚠  Identification scores are random (stub embedder). "
              "Export the ONNX model for real results.")


# ─── Mode: RECORD ────────────────────────────────────────────────────────────

def mode_record(args):
    print(f"\n{'='*65}")
    print(f"  RECORD MODE: {args.duration}s from microphone")
    print(f"{'='*65}")

    audio = record_mic(args.duration, label="conversation")

    if args.playback:
        play_audio(audio, "your recording")

    print(f"\n  Duration   : {len(audio)/SR:.2f}s")
    print(f"  SNR        : {estimate_snr_db(audio, SR):.1f} dB")

    embedder, mode = get_embedder()
    staff_db = _load_db(args) if hasattr(args, "db") and args.db else None

    results = analyse(audio, embedder, staff_db, threshold=args.threshold)
    print_segment_table(results, len(audio) / SR)

    # Optionally save the recording
    if args.save:
        import soundfile as sf
        out = Path(args.save)
        sf.write(str(out), audio, SR)
        print(f"\nSaved recording → {out}")


# ─── Mode: ENROLL ────────────────────────────────────────────────────────────

def mode_enroll(args):
    """
    Interactive enrollment flow:
      1. Record 3 utterances for a staff member
      2. Validate each (SNR, duration, clipping)
      3. Compute embedding and save to staff DB
      4. Immediately test identification with a new recording
    """
    print(f"\n{'='*65}")
    print(f"  ENROLL MODE: enrolling '{args.name}' as staff")
    print(f"{'='*65}")

    embedder, emb_mode = get_embedder()

    # Thresholds relaxed for laptop-mic / dev testing.
    # SNR is disabled (-999) because MacBook mics give 1–2 dB even with clear
    # speech — the estimator uses p10/p50 frame energy which collapses when the
    # mic input is very quiet or another app is holding the device.
    # Production enrollment uses a phone mic in a quiet room (SNR ≥ 20 dB).
    laptop_thresholds = {
        "min_duration_sec":   4.0,
        "min_snr_db":         -999,  # disabled for laptop testing
        "min_rms_level":      0.0,   # disabled — mic volume varies widely
        "max_clipping_ratio": 0.001,
        "sample_rate_hz":     SR,
    }
    validator = EnrollmentValidator(thresholds=laptop_thresholds)
    print("  (SNR check disabled for laptop mic — production requires SNR ≥ 20 dB)")

    db_path = Path(args.db) if args.db else Path("real_voice_test.staffdb")
    key_path = db_path.with_suffix(".key")

    # Load or create staff DB
    if db_path.exists() and key_path.exists():
        key = key_path.read_bytes()
        staff_db = StaffDBManager(db_path, key)
        staff_db.load()
        print(f"\nLoaded existing staff DB: {db_path}")
    else:
        key = generate_key()
        key_path.write_bytes(key)
        staff_db = StaffDBManager(db_path, key)
        staff_db.create(store_id="REAL_VOICE_TEST")
        print(f"\nCreated new staff DB: {db_path}")
        print(f"Encryption key saved to: {key_path}  (keep this safe!)")

    # Enrollment prompts from the spec
    prompts = [
        "Hello, welcome to the store. How can I help you today?",
        "Sure, let me check that for you. What size are you looking for?",
        "Your total comes to fourteen ninety-five. Would you like a receipt?",
    ]

    valid_audios = []
    print(f"\nYou will be asked to read {len(prompts)} phrases.")
    print("Speak naturally at normal volume, 10–30 cm from the microphone.\n")

    for i, prompt in enumerate(prompts):
        print(f"\nPhrase {i+1}/{len(prompts)}:")
        print(f'  "{prompt}"')

        while True:
            audio = record_mic(8.0, label=f"phrase {i+1}")
            result = validator.validate_utterance(audio, SR)

            print(f"\n  Quality check:")
            print(f"    Duration : {result.duration_sec:.1f}s  {'✓' if result.duration_sec >= 5 else '✗'}")
            print(f"    SNR      : {result.snr_db:.1f} dB  ✓  (check disabled for laptop mic)")
            print(f"    Clipping : {result.clipping_ratio:.3%}  {'✓' if result.clipping_ratio < 0.001 else '✗'}")
            print(f"    Status   : {'PASS ✓' if result.passed else 'FAIL ✗'}")

            if result.passed:
                valid_audios.append(audio)
                break
            else:
                print(f"\n  ✗ Rejected: {result.guidance}")
                retry = input("  Try again? [Y/n]: ").strip().lower()
                if retry == "n":
                    break

    if len(valid_audios) < 1:
        print("\nNo valid recordings captured. Enrollment cancelled.")
        sys.exit(1)

    print(f"\n── Computing voice embedding from {len(valid_audios)} recordings ──")
    mean_emb = embedder.mean_embedding(valid_audios)
    staff_id = staff_db.add_staff(
        name=args.name,
        role=args.role,
        embedding=mean_emb,
        n_samples=len(valid_audios),
    )
    print(f"✓ Enrolled: {args.name} ({args.role})  →  ID: {staff_id}")

    if emb_mode == "stub":
        print("\n⚠  Using stub embedder — enrollment is stored but identification")
        print("   will not work reliably without the real ONNX model.")

    # Immediately test identification
    print(f"\n{'─'*65}")
    print("  IDENTIFICATION TEST")
    print(f"  Now speak again as {args.name} so we can verify the enrollment.")
    print(f"{'─'*65}")
    test_audio = record_mic(6.0, label="identification test")
    test_emb = embedder.embed(normalize_audio(test_audio), SR)

    search = SimilaritySearch(staff_db)
    matched_name, score = search.top1(test_emb)

    print(f"\n  Result:")
    print(f"    Best match  : {matched_name}")
    print(f"    Similarity  : {score:.4f}")
    print(f"    Threshold   : {args.threshold}")
    role_result = "STAFF ✓" if score >= args.threshold else "CUSTOMER ✗ (below threshold)"
    print(f"    Decision    : {role_result}")

    if score < args.threshold:
        print(f"\n  Tip: score {score:.3f} is below threshold {args.threshold}.")
        print("  This can happen with stub embedder or poor audio quality.")
        print("  Try lowering --threshold or re-enrolling in a quieter space.")

    # List all enrolled staff
    print(f"\n  All enrolled staff in DB:")
    for s in staff_db.list_staff():
        print(f"    • {s['name']} ({s['role']})  [{s['n_samples']} samples]")


# ─── DB loader helper ─────────────────────────────────────────────────────────

def _load_db(args) -> StaffDBManager | None:
    db_path = Path(args.db) if args.db else None
    if db_path is None or not db_path.exists():
        return None
    key_path = db_path.with_suffix(".key")
    if not key_path.exists():
        print(f"⚠  Key file not found: {key_path}")
        return None
    key = key_path.read_bytes()
    db = StaffDBManager(db_path, key)
    db.load()
    print(f"✓ Loaded staff DB: {db_path}  ({db.staff_count()[0]} active staff)")
    return db


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-voice testing for the speaker diarization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_real_voice.py --file conversation.wav
  python test_real_voice.py --record --duration 30
  python test_real_voice.py --enroll --name "Alice" --role associate
  python test_real_voice.py --file conversation.wav --db real_voice_test.staffdb
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file",   metavar="PATH", help="Analyse an existing audio file")
    mode.add_argument("--record", action="store_true", help="Record from microphone then analyse")
    mode.add_argument("--enroll", action="store_true", help="Enroll a staff member by voice")

    parser.add_argument("--duration",  type=float, default=30.0, help="Recording duration in seconds (--record mode)")
    parser.add_argument("--name",      type=str,   default="Staff Member", help="Name for enrollment")
    parser.add_argument("--role",      type=str,   default="associate",    help="Role for enrollment")
    parser.add_argument("--db",        type=str,   default=None,           help="Path to .staffdb file")
    parser.add_argument("--threshold", type=float, default=0.82,           help="Staff similarity threshold (default 0.82)")
    parser.add_argument("--playback",  action="store_true",                help="Play back recording after capture")
    parser.add_argument("--save",      type=str,   default=None,           help="Save recording to this WAV file (--record mode)")

    args = parser.parse_args()

    if args.file:
        mode_file(args)
    elif args.record:
        mode_record(args)
    elif args.enroll:
        mode_enroll(args)


if __name__ == "__main__":
    main()
