"""
tests/test_enrollment_validator.py
────────────────────────────────────
Unit tests for the enrollment audio quality validator.
Run with: pytest tests/test_enrollment_validator.py -v
"""

import numpy as np
import pytest

from enrollment.enrollment_validator import EnrollmentValidator


SR = 16_000


def make_audio(duration_sec: float, freq: float = 220.0, amplitude: float = 0.3) -> np.ndarray:
    """
    Simulate speech-like audio: bursts of tone separated by near-silence.

    The energy-based SNR estimator works by comparing the 10th-percentile frame
    energy (noise floor) against the median frame energy (signal). A pure
    continuous sine wave has uniform energy across all frames, so percentile
    spread is zero and SNR ≈ 0 dB. This helper alternates loud and near-silent
    frames to produce a realistic SNR spread well above the 20 dB threshold.
    """
    n_samples = int(duration_sec * SR)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t)

    # 20 ms frames — match the validator's frame size
    frame_samples = int(0.020 * SR)
    n_frames = n_samples // frame_samples

    rng = np.random.default_rng(0)
    # Alternate: ~70% of frames are "speech" (full amplitude), ~30% are "silence" (1% amplitude)
    # This gives 10th-pct energy ≈ silence level → SNR well above 20 dB
    mask = np.ones(n_samples, dtype=np.float32)
    for i in range(n_frames):
        if rng.random() < 0.30:  # silence frame
            start = i * frame_samples
            mask[start: start + frame_samples] = 0.01

    return (signal * mask).astype(np.float32)


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to achieve the target SNR."""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


class TestEnrollmentValidator:
    def setup_method(self):
        self.validator = EnrollmentValidator()

    # ── Passing cases ─────────────────────────────────────────────────────────

    def test_clean_audio_passes(self):
        audio = make_audio(7.0, amplitude=0.3)
        result = self.validator.validate_utterance(audio, SR)
        assert result.passed, f"Expected pass, got: {result.rejection_reasons}"

    def test_minimum_duration_passes(self):
        audio = make_audio(5.0, amplitude=0.3)
        result = self.validator.validate_utterance(audio, SR)
        assert result.passed, f"Expected pass: {result.rejection_reasons}"

    # ── Failing cases ─────────────────────────────────────────────────────────

    def test_too_short_fails(self):
        audio = make_audio(3.0)  # below 5s minimum
        result = self.validator.validate_utterance(audio, SR)
        assert not result.passed
        assert any("short" in r.lower() for r in result.rejection_reasons)

    def test_too_quiet_fails(self):
        audio = make_audio(7.0, amplitude=0.001)  # RMS far below threshold
        result = self.validator.validate_utterance(audio, SR)
        assert not result.passed
        assert any("quiet" in r.lower() for r in result.rejection_reasons)

    def test_clipping_fails(self):
        audio = make_audio(7.0, amplitude=2.0)  # clips above 1.0
        audio = np.clip(audio, -1.0, 1.0)
        result = self.validator.validate_utterance(audio, SR)
        assert not result.passed
        assert any("clipping" in r.lower() for r in result.rejection_reasons)

    def test_noisy_audio_fails(self):
        audio = make_audio(7.0, amplitude=0.3)
        noisy = add_noise(audio, snr_db=5.0)  # 5 dB SNR — well below 20 dB threshold
        result = self.validator.validate_utterance(noisy, SR)
        assert not result.passed
        assert any("noise" in r.lower() or "snr" in r.lower() for r in result.rejection_reasons)

    def test_wrong_sample_rate_fails(self):
        audio = make_audio(7.0)
        result = self.validator.validate_utterance(audio, sample_rate=8_000)
        assert not result.passed
        assert any("sample rate" in r.lower() for r in result.rejection_reasons)

    # ── ValidationResult fields ───────────────────────────────────────────────

    def test_result_fields_populated(self):
        audio = make_audio(7.0)
        result = self.validator.validate_utterance(audio, SR)
        assert result.duration_sec > 0
        assert result.rms_level > 0
        assert 0.0 <= result.clipping_ratio <= 1.0
        assert isinstance(result.guidance, str)
        assert len(result.guidance) > 0

    def test_guidance_non_empty_on_failure(self):
        audio = make_audio(2.0)  # too short
        result = self.validator.validate_utterance(audio, SR)
        assert not result.passed
        assert len(result.guidance) > 10

    # ── Batch validation ──────────────────────────────────────────────────────

    def test_batch_filters_bad_utterances(self):
        good = make_audio(7.0)
        bad = make_audio(2.0)  # too short
        valid, results = self.validator.validate_enrollment_set([good, bad], SR)
        assert len(valid) == 1
        assert len(results) == 2
        assert results[0].passed
        assert not results[1].passed

    def test_batch_all_pass(self):
        utterances = [make_audio(6.0 + i * 0.5) for i in range(5)]
        valid, results = self.validator.validate_enrollment_set(utterances, SR)
        assert len(valid) == 5
        assert all(r.passed for r in results)
