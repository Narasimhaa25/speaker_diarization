"""
Speaker Diarization Project - Comprehensive Demo
=================================================
Demonstrates all major features of the speaker diarization system.
"""

import sys
import numpy as np
from pathlib import Path

# Add speaker_diarization to path
sys.path.insert(0, str(Path(__file__).parent / "speaker_diarization"))

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_subsection(title):
    print(f"\n[{title}]")
    print("-" * 70)

print_section("SPEAKER DIARIZATION PROJECT - COMPREHENSIVE DEMO")

# ============================================================================
# 1. AUDIO UTILITIES
# ============================================================================
print_subsection("1. Audio Processing Utilities")

from utils.audio_utils import (
    chunk_audio, normalize_audio, estimate_snr_db, 
    voice_activity_detection, is_clipping, duration_sec
)

# Generate synthetic audio
sr = 16000
audio = np.random.randn(10 * sr).astype(np.float32) * 0.1

print(f"✓ Generated synthetic audio:")
print(f"  • Duration: {duration_sec(audio, sr):.2f} seconds")
print(f"  • Sample rate: {sr} Hz")
print(f"  • Shape: {audio.shape}")
print(f"  • SNR: {estimate_snr_db(audio, sr):.2f} dB")
print(f"  • Clipping detected: {is_clipping(audio)}")

# Chunking
chunks = list(chunk_audio(audio, sr, chunk_duration_sec=3.0, overlap_sec=0.5))
print(f"\n✓ Audio chunking (3s chunks, 0.5s overlap):")
print(f"  • Created {len(chunks)} chunks")

# VAD
vad_segments = voice_activity_detection(audio, sr)
print(f"\n✓ Voice Activity Detection:")
print(f"  • Found {len(vad_segments)} speech segments")

# ============================================================================
# 2. ENCRYPTION (Staff DB Security)
# ============================================================================
print_subsection("2. Encryption System (AES-256-GCM)")

from utils.crypto_utils import generate_key, encrypt_bytes, decrypt_bytes

key = generate_key()
sensitive_data = b"Staff voice embeddings - confidential data"

print(f"✓ Generated 256-bit encryption key")
print(f"  • Key length: {len(key)} bytes")

encrypted = encrypt_bytes(sensitive_data, key)
print(f"\n✓ Encrypted sensitive data:")
print(f"  • Original size: {len(sensitive_data)} bytes")
print(f"  • Encrypted size: {len(encrypted)} bytes")
print(f"  • Overhead: {len(encrypted) - len(sensitive_data)} bytes (magic + nonce + tag)")

decrypted = decrypt_bytes(encrypted, key)
print(f"\n✓ Decrypted data:")
print(f"  • Match: {decrypted == sensitive_data} ✓")

# ============================================================================
# 3. SPEAKER COUNT ESTIMATION
# ============================================================================
print_subsection("3. Speaker Count Estimation")

from core.speaker_count import SpeakerCountEstimator

estimator = SpeakerCountEstimator()
estimated_speakers = estimator.estimate(audio, sr)

print(f"✓ Estimated number of speakers: {estimated_speakers}")
print(f"  • Method: Energy distribution heuristic")
print(f"  • Range: 1-10 speakers")

# ============================================================================
# 4. STAFF DATABASE MANAGEMENT
# ============================================================================
print_subsection("4. Staff Database Management")

from staff_db.db_manager import StaffDBManager
from staff_db.schema import EMBEDDING_DIM

db_path = Path("demo_staff.db.enc")
db_key = generate_key()  # Generate encryption key for the database

print(f"✓ Creating encrypted staff database: {db_path}")
staff_db = StaffDBManager(db_path, db_key)
staff_db.create(store_id="DEMO_STORE_001")

# Add staff members
staff_members = [
    ("Alice Johnson", "associate"),
    ("Bob Smith", "manager"),
    ("Carol Davis", "associate"),
]

print(f"\n✓ Adding {len(staff_members)} staff members:")
for name, role in staff_members:
    # Generate random normalized embedding (192-dim)
    embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
    
    staff_db.add_staff(name, role, embedding, n_samples=5)
    print(f"  • {name} ({role})")

# List all staff
all_staff_list = staff_db.list_staff()
print(f"\n✓ Database contents:")
print(f"  • Total staff: {len(all_staff_list)}")
print(f"  • Embedding dimension: {EMBEDDING_DIM}")
print(f"  • Encryption: AES-256-GCM")

# Get embeddings for similarity search
all_staff = staff_db.get_active_embeddings()

# ============================================================================
# 5. SIMILARITY SEARCH (Staff Identification)
# ============================================================================
print_subsection("5. Similarity Search & Staff Identification")

from staff_db.similarity_search import SimilaritySearch

search = SimilaritySearch(staff_db)

# Test 1: Query similar to Alice
alice_id, alice_name, alice_embedding = all_staff[0]
query_embedding = alice_embedding + np.random.randn(EMBEDDING_DIM).astype(np.float32) * 0.05
query_embedding = query_embedding / np.linalg.norm(query_embedding)

print(f"✓ Test 1 - Voice similar to Alice:")
matched_name, similarity = search.top1(query_embedding)
if matched_name:
    print(f"  • Matched: {matched_name}")
    print(f"  • Similarity: {similarity:.4f}")
    print(f"  • Classification: {'STAFF ✓' if similarity > 0.70 else 'CUSTOMER'}")

# Test 2: Random unknown voice
random_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
random_embedding = random_embedding / np.linalg.norm(random_embedding)

print(f"\n✓ Test 2 - Unknown voice (customer):")
matched_name, similarity = search.top1(random_embedding)
if matched_name:
    print(f"  • Best match: {matched_name}")
    print(f"  • Similarity: {similarity:.4f}")
    print(f"  • Classification: {'STAFF' if similarity > 0.70 else 'CUSTOMER ✓'}")

# ============================================================================
# 6. STAFF IDENTIFIER (Role Classification)
# ============================================================================
print_subsection("6. Staff Identifier (Automated Role Classification)")

from core.staff_identifier import StaffIdentifier

identifier = StaffIdentifier(staff_db, threshold=0.70)

print(f"✓ Staff identifier configuration:")
print(f"  • Threshold: 0.70 (cosine similarity)")
print(f"  • Database: {len(all_staff)} staff members")

# Test with Alice's voice
alice_id, alice_name, alice_embedding = all_staff[0]
role, name, score = identifier.identify(alice_embedding)
print(f"\n✓ Identification test - Alice's voice:")
print(f"  • Role: {role}")
print(f"  • Name: {name}")
print(f"  • Confidence: {score:.4f}")

# Test with unknown voice
role, name, score = identifier.identify(random_embedding)
print(f"\n✓ Identification test - Unknown voice:")
print(f"  • Role: {role}")
print(f"  • Name: {name if name else 'N/A'}")
print(f"  • Confidence: {score:.4f}")

# ============================================================================
# 7. ENROLLMENT VALIDATION
# ============================================================================
print_subsection("7. Enrollment Validation (Voice Quality Checks)")

from enrollment.enrollment_validator import EnrollmentValidator

validator = EnrollmentValidator()  # Uses default thresholds

print(f"✓ Enrollment validator configuration (default thresholds):")
print(f"  • Duration: 5.0 - 30.0 seconds")
print(f"  • Minimum SNR: 20.0 dB")
print(f"  • Minimum RMS: 0.01")
print(f"  • Max clipping: 1%")

# Test with synthetic audio
test_audio = np.random.randn(int(7 * sr)) * 0.2
result = validator.validate_utterance(test_audio, sr)

print(f"\n✓ Validation result:")
print(f"  • Duration: {result.duration_sec:.2f}s")
print(f"  • SNR: {result.snr_db:.2f} dB")
print(f"  • RMS Level: {result.rms_level:.4f}")
print(f"  • Clipping: {result.clipping_ratio:.2%}")
print(f"  • Status: {'PASSED ✓' if result.passed else 'FAILED ✗'}")
if not result.passed:
    print(f"  • Reasons: {', '.join(result.rejection_reasons)}")

# ============================================================================
# 8. DIARIZATION MODEL
# ============================================================================
print_subsection("8. Diarization Model (Who Spoke When)")

from models.diarization_model import DiarizationModel

print(f"✓ Initializing diarization model...")
print(f"  • Backend: pyannote.audio")
print(f"  • Clustering threshold: 0.70")

try:
    diar_model = DiarizationModel(clustering_threshold=0.70)
    print(f"\n✓ Running diarization on synthetic audio...")
    segments = diar_model.diarize(audio, num_speakers=2)
    print(f"  • Found {len(segments)} speaker segments")
    
    if segments:
        print(f"\n  Sample segments:")
        for i, seg in enumerate(segments[:3]):
            print(f"    {i+1}. {seg.start:.2f}s - {seg.end:.2f}s | "
                  f"Speaker: {seg.speaker_id} | Confidence: {seg.confidence:.2f}")
        if len(segments) > 3:
            print(f"    ... and {len(segments) - 3} more segments")
except Exception as e:
    print(f"  • Note: Full diarization requires pre-trained models or authenticated HuggingFace access")
    print(f"  • Error: {str(e)[:200]}")
    segments = []

# ============================================================================
# 9. ECAPA-TDNN EMBEDDINGS
# ============================================================================
print_subsection("9. ECAPA-TDNN Speaker Embeddings")

print(f"✓ ECAPA-TDNN configuration:")
print(f"  • Embedding dimension: 192")
print(f"  • Quantization: INT8 ONNX")
print(f"  • Model: speechbrain/spkrec-ecapa-voxceleb")
print(f"  • Training data: VoxCeleb 1+2 (~2,800 hours)")
print(f"\n  Note: To export ONNX model, run:")
print(f"    python speaker_diarization/models/export_onnx.py")

# ============================================================================
# SUMMARY
# ============================================================================
print_section("PROJECT SUMMARY")

print("""
This speaker diarization system provides:

✓ Core Features:
  • Speaker Diarization - Identify "who spoke when" in conversations
  • Staff Identification - Classify speakers as staff or customer
  • Speaker Embeddings - 192-dim ECAPA-TDNN voice representations
  • Encrypted Staff DB - Secure storage with AES-256-GCM
  • Enrollment Validation - Quality checks for voice enrollment
  • Audio Processing - Load, chunk, normalize, and analyze audio

✓ Key Metrics:
  • Target DER: < 15% on in-domain audio
  • Staff ID Accuracy: > 95%
  • Embedding Dimension: 192 (ECAPA-TDNN)
  • Quantization: INT8 ONNX for mobile deployment
  • Staff DB Capacity: 5-25 staff per store

✓ Datasets:
  • VoxCeleb 1+2: ~2,800 hours for embedding training
  • AMI Corpus: ~40 hours for diarization evaluation

✓ Security:
  • AES-256-GCM authenticated encryption
  • On-device encrypted staff voice database
  • No cloud dependencies for voice matching

✓ Integration Points:
  • Module 3: Shares ECAPA-TDNN model for emotion detection
  • Module 7: Provides speaker count for group detection
  • Module 8: Defines enrollment recording requirements
""")

print("\n" + "=" * 70)
print("  AVAILABLE COMMANDS")
print("=" * 70)
print("""
• Run tests:
  pytest speaker_diarization/tests/ -v

• Export ONNX model:
  python speaker_diarization/models/export_onnx.py

• Run diarization on audio file:
  python speaker_diarization/core/diarizer.py --audio <file.wav>

• Evaluate DER on AMI dataset:
  python speaker_diarization/evaluation/evaluate_der.py --ami_dir <path>

• Tune similarity threshold:
  python speaker_diarization/evaluation/threshold_tuner.py
""")

# Cleanup
print("\n✓ Cleaning up demo files...")
if db_path.exists():
    db_path.unlink()
    print(f"  • Removed: {db_path}")

print("\n" + "=" * 70)
print("  DEMO COMPLETE!")
print("=" * 70)
