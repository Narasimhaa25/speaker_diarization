"""
tests/test_diarizer.py
───────────────────────
Unit tests for the diarization pipeline.
Run with: pytest tests/test_diarizer.py -v
"""

import numpy as np
import pytest

from models.diarization_model import DiarizationModel, DiarizedSegment
from core.speaker_count import SpeakerCountEstimator


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_sine_wave(freq: float, duration_sec: float, sr: int = 16_000) -> np.ndarray:
    """Generate a simple sine wave as synthetic audio."""
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_multi_speaker_audio(sr: int = 16_000) -> np.ndarray:
    """
    Synthetic 2-speaker audio: alternating 3-second segments at different frequencies.
    Speaker A: 220 Hz, Speaker B: 440 Hz.
    """
    seg_a = make_sine_wave(220, 3.0, sr)
    seg_b = make_sine_wave(440, 3.0, sr)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    return np.concatenate([seg_a, silence, seg_b, silence, seg_a])


# ─── DiarizedSegment tests ────────────────────────────────────────────────────

class TestDiarizedSegment:
    def test_basic_fields(self):
        seg = DiarizedSegment(start=0.0, end=3.5, speaker_id="SPEAKER_00")
        assert seg.start == 0.0
        assert seg.end == 3.5
        assert seg.speaker_id == "SPEAKER_00"
        assert seg.confidence == 1.0

    def test_duration(self):
        seg = DiarizedSegment(start=1.0, end=4.0, speaker_id="SPEAKER_01")
        duration = seg.end - seg.start
        assert pytest.approx(duration, 0.01) == 3.0


# ─── SpeakerCountEstimator tests ─────────────────────────────────────────────

class TestSpeakerCountEstimator:
    def test_heuristic_returns_in_range(self):
        """Heuristic should always return a count within [min, max]."""
        estimator = SpeakerCountEstimator(min_speakers=2, max_speakers=6)
        audio = make_multi_speaker_audio()
        count = estimator.estimate_heuristic(audio, 16_000)
        assert 2 <= count <= 6

    def test_short_audio_returns_minimum(self):
        """Short audio (< 2 windows) should return min_speakers."""
        estimator = SpeakerCountEstimator(min_speakers=2, max_speakers=6)
        short_audio = make_sine_wave(220, 1.0)  # 1 second — too short for windows
        count = estimator.estimate(short_audio, 16_000)
        assert count == 2  # falls back to minimum

    def test_bounds_respected(self):
        """Count must always be within [min_speakers, max_speakers]."""
        estimator = SpeakerCountEstimator(min_speakers=2, max_speakers=4)
        audio = make_multi_speaker_audio()
        count = estimator.estimate_heuristic(audio, 16_000)
        assert 2 <= count <= 4


# ─── DiarizationModel segment activity conversion tests ──────────────────────

class TestActivityToSegments:
    def test_basic_activity_conversion(self):
        """Simple on/off activity should produce one segment."""
        model = DiarizationModel.__new__(DiarizationModel)
        model.clustering_threshold = 0.5
        model.min_duration_on = 0.05
        model.min_duration_off = 0.05

        # 10 frames active, then 5 frames silent
        activity = np.zeros((15, 1), dtype=np.float32)
        activity[:10, 0] = 0.9  # above threshold

        segments = model._activity_to_segments(activity, frame_duration=0.01, num_speakers=1)
        assert len(segments) == 1
        assert segments[0].speaker_id == "SPEAKER_00"
        assert pytest.approx(segments[0].start, abs=0.01) == 0.0
        assert pytest.approx(segments[0].end, abs=0.01) == 0.10

    def test_below_threshold_no_segments(self):
        """Activity below threshold should produce no segments."""
        model = DiarizationModel.__new__(DiarizationModel)
        model.clustering_threshold = 0.8
        model.min_duration_on = 0.05
        model.min_duration_off = 0.05

        activity = np.full((20, 1), 0.3, dtype=np.float32)  # all below threshold
        segments = model._activity_to_segments(activity, frame_duration=0.01, num_speakers=1)
        assert len(segments) == 0

    def test_segments_sorted_by_start(self):
        """Output segments must always be sorted by start time."""
        model = DiarizationModel.__new__(DiarizationModel)
        model.clustering_threshold = 0.5
        model.min_duration_on = 0.01
        model.min_duration_off = 0.01

        # Two speakers, interleaved
        activity = np.zeros((30, 2), dtype=np.float32)
        activity[:10, 0] = 0.9   # Speaker 0 first
        activity[15:25, 1] = 0.9  # Speaker 1 second

        segments = model._activity_to_segments(activity, frame_duration=0.01, num_speakers=2)
        starts = [s.start for s in segments]
        assert starts == sorted(starts)


# ─── Crypto round-trip (smoke test) ──────────────────────────────────────────

class TestCrypto:
    def test_encrypt_decrypt_roundtrip(self):
        from utils.crypto_utils import encrypt_bytes, decrypt_bytes, generate_key
        key = generate_key()
        plaintext = b'{"store_id": "TEST", "staff": {}}'
        encrypted = encrypt_bytes(plaintext, key)
        decrypted = decrypt_bytes(encrypted, key)
        assert decrypted == plaintext

    def test_wrong_key_raises(self):
        from utils.crypto_utils import encrypt_bytes, decrypt_bytes, generate_key
        from cryptography.exceptions import InvalidTag
        key1 = generate_key()
        key2 = generate_key()
        encrypted = encrypt_bytes(b"secret data", key1)
        with pytest.raises(InvalidTag):
            decrypt_bytes(encrypted, key2)

    def test_tampered_data_raises(self):
        from utils.crypto_utils import encrypt_bytes, decrypt_bytes, generate_key
        from cryptography.exceptions import InvalidTag
        key = generate_key()
        encrypted = bytearray(encrypt_bytes(b"secret", key))
        encrypted[-1] ^= 0xFF  # flip last bit
        with pytest.raises(InvalidTag):
            decrypt_bytes(bytes(encrypted), key)
