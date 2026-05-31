"""
Module 2 — Speaker Diarization & Staff Voice Identification
============================================================
Demo: answers two questions for every audio chunk:
  1. WHO spoke WHEN?
  2. Is each speaker STAFF or CUSTOMER?
"""

from __future__ import annotations
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# Support both Docker (/app) and local runs
BASE = Path("/app") if Path("/app/speaker_diarization").exists() else Path(__file__).parent
sys.path.insert(0, str(BASE / "speaker_diarization"))

ONNX_MODEL   = BASE / "speaker_diarization/models/ecapa_tdnn_int8.onnx"
STAFF_DB     = BASE / "ami_staff.staffdb"
STAFF_KEY    = BASE / "ami_staff.key"
AUDIO_MIX    = BASE / "amicorpus/ES2013a/audio/ES2013a.Mix-Headset.wav"
HEADSETS     = [BASE / f"amicorpus/ES2013a/audio/ES2013a.Headset-{i}.wav" for i in range(4)]
NAMES        = ["Alice", "Bob", "Carol", "David"]
THRESHOLD    = 0.82
DEMO_SECONDS = 120   # process first 2 minutes for a fast demo


def banner(title: str) -> None:
    w = 65
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def step(n: int, title: str) -> None:
    print(f"\n── Step {n}: {title} {'─'*(50-len(title))}")


def ok(msg: str)  -> None: print(f"  ✓ {msg}")
def info(msg: str)-> None: print(f"  │ {msg}")


# ─── Step 0: Load audio ───────────────────────────────────────────────────────

banner("MODULE 2 · SPEAKER DIARIZATION & STAFF VOICE ID")
print("""
  Scenario: a retail store meeting recording (AMI corpus).
  4 speakers are enrolled as staff.
  The system must answer: WHO spoke WHEN, and are they STAFF?
""")

step(0, "Loading audio")
from utils.audio_utils import load_audio, normalize_audio
audio_full, sr = load_audio(str(AUDIO_MIX), target_sr=16_000)
audio = audio_full[:DEMO_SECONDS * sr]
ok(f"Audio loaded  — {len(audio_full)/sr:.0f}s total, using first {DEMO_SECONDS}s for demo")
info(f"Sample rate   : {sr} Hz")
info(f"File          : {AUDIO_MIX.name}")


# ─── Step 1: Speaker count estimate ──────────────────────────────────────────

step(1, "Estimating speaker count")
from core.speaker_count import SpeakerCountEstimator
count = SpeakerCountEstimator().estimate_heuristic(audio, sr)
ok(f"Estimated speakers in clip: {count}  (actual: 4)")


# ─── Step 2: Staff enrollment from individual headset files ──────────────────

step(2, "Building staff voice database")
from models.ecapa_tdnn import ECAPATDNNEmbedder
from utils.crypto_utils import generate_key
from staff_db.db_manager import StaffDBManager

embedder = ECAPATDNNEmbedder(ONNX_MODEL)
ok(f"ECAPA-TDNN ONNX model loaded  ({ONNX_MODEL.stat().st_size // 1_000_000}MB, 192-dim embeddings)")

if STAFF_DB.exists() and STAFF_KEY.exists():
    key = STAFF_KEY.read_bytes()
    db = StaffDBManager(STAFF_DB, key)
    db.load()
    ok(f"Staff DB loaded  — {db.staff_count()[0]} enrolled staff")
    for s in db.list_staff():
        info(f"  {s['name']:<10} ({s['role']})  [{s['n_samples']} enrollment samples]")
else:
    # Build DB from individual headset recordings
    key = generate_key()
    STAFF_KEY.write_bytes(key)
    db = StaffDBManager(STAFF_DB, key)
    db.create("DEMO_STORE")

    for headset_path, name in zip(HEADSETS, NAMES):
        if not headset_path.exists():
            continue
        h_audio, _ = load_audio(str(headset_path), target_sr=16_000)
        clips = [
            normalize_audio(h_audio[30*sr:38*sr]),
            normalize_audio(h_audio[100*sr:108*sr]),
            normalize_audio(h_audio[200*sr:208*sr]),
        ]
        emb = embedder.mean_embedding(clips)
        db.add_staff(name, "staff", emb, n_samples=len(clips))
        ok(f"Enrolled {name} from {headset_path.name}")


# ─── Step 3: Diarization ──────────────────────────────────────────────────────

step(3, "Speaker diarization  (who spoke when?)")
from models.diarization_model import DiarizationModel

t0 = time.perf_counter()
try:
    model = DiarizationModel(clustering_threshold=0.70)
    segments = model.diarize(audio)
    mode = "pyannote (neural)"
except Exception:
    # Energy fallback
    import librosa
    from models.diarization_model import DiarizedSegment
    from utils.audio_utils import voice_activity_detection
    audio_n = normalize_audio(audio)
    vad = voice_activity_detection(audio_n, sr, energy_threshold_db=-40.0, min_speech_ms=200)
    segments, prev_mfcc, speaker_mfccs, label = [], None, {}, 0
    for start, end in vad:
        if end - start < 0.3:
            continue
        chunk = audio_n[int(start*sr):int(end*sr)]
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13).mean(axis=1)
        if prev_mfcc is not None:
            best_l = min(speaker_mfccs, key=lambda l: np.linalg.norm(mfcc - speaker_mfccs[l]))
            best_d = np.linalg.norm(mfcc - speaker_mfccs[best_l])
            label = (max(speaker_mfccs)+1) if best_d > 10 else best_l
        speaker_mfccs[label] = mfcc
        segments.append(DiarizedSegment(start=start, end=end, speaker_id=f"SPEAKER_{label:02d}"))
        prev_mfcc = mfcc
    mode = "energy fallback"

elapsed = time.perf_counter() - t0
rtf = elapsed / DEMO_SECONDS
ok(f"Diarization done  — mode: {mode}")
ok(f"Found {len(segments)} segments in {elapsed:.1f}s  (RTF={rtf:.2f}x)")

# Count speaking time per speaker
speaker_time: dict[str, float] = defaultdict(float)
for s in segments:
    speaker_time[s.speaker_id] += s.end - s.start
info("Speaking time per diarized speaker:")
for spk, t in sorted(speaker_time.items()):
    bar = "█" * int(t / 2)
    info(f"  {spk}: {t:5.1f}s  {bar}")


# ─── Step 4: Staff identification ─────────────────────────────────────────────

step(4, "Staff identification  (staff or customer?)")
from core.staff_identifier import StaffIdentifier

identifier = StaffIdentifier(db, threshold=THRESHOLD)
ok(f"Threshold: {THRESHOLD}  (score ≥ {THRESHOLD} → STAFF, below → CUSTOMER)")

results = []
for seg in segments:
    chunk = audio[int(seg.start*sr):int(seg.end*sr)]
    if len(chunk) < int(0.5*sr):
        continue
    chunk = normalize_audio(chunk)
    try:
        emb = embedder.embed(chunk, sr)
    except ValueError:
        continue
    role, name, score = identifier.identify(emb)
    results.append((seg, role, name, score))


# ─── Step 5: Results table ────────────────────────────────────────────────────

step(5, "Results")
print(f"\n  {'START':>7}  {'END':>7}  {'DUR':>5}  {'PYANNOTE ID':<12}  {'ROLE':<10}  {'SCORE':>6}  NAME")
print(f"  {'─'*62}")

staff_time: dict[str, float] = defaultdict(float)
customer_time = 0.0
staff_count = customer_count = 0

for seg, role, name, score in results:
    dur = seg.end - seg.start
    role_str  = f"{'STAFF':<10}" if role == "staff" else f"{'CUSTOMER':<10}"
    score_str = f"{score:.3f}" if score > 0 else "  —  "
    name_str  = name or ""
    print(f"  {seg.start:>6.1f}s  {seg.end:>6.1f}s  {dur:>4.1f}s"
          f"  {seg.speaker_id:<12}  {role_str}  {score_str}  {name_str}")
    if role == "staff":
        staff_time[name] += dur
        staff_count += 1
    else:
        customer_time += dur
        customer_count += 1


# ─── Step 6: Summary ──────────────────────────────────────────────────────────

banner("SUMMARY")
total = staff_count + customer_count
print(f"""
  Audio processed  : {DEMO_SECONDS}s ({DEMO_SECONDS//60}m {DEMO_SECONDS%60}s)
  Total segments   : {total}
  Staff segments   : {staff_count}  ({staff_count/total*100:.0f}%)
  Customer segments: {customer_count}  ({customer_count/total*100:.0f}%)

  Staff speaking time breakdown:""")

for name, t in sorted(staff_time.items(), key=lambda x: -x[1]):
    bar = "█" * int(t / 3)
    print(f"    {name:<10}: {t:5.1f}s  {bar}")

print(f"""
  Embedding model  : ECAPA-TDNN ONNX INT8 (192-dim, VoxCeleb-trained)
  Diarization      : {mode}
  Identification   : cosine similarity, threshold={THRESHOLD}
  Staff DB         : AES-256-GCM encrypted, {db.staff_count()[0]} enrolled staff

  → This output feeds Module 3 (sentiment), Module 7 (group detection)
""")
print("═"*65)
