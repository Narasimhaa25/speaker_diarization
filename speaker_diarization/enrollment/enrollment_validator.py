"""
enrollment/enrollment_validator.py
────────────────────────────────────
Validates enrollment audio quality before embedding.

Called by Module 8's recording flow to reject bad utterances
and guide the staff member to re-record.

Checks:
  - Duration ≥ 5 seconds
  - SNR ≥ 20 dB (estimated via noise floor)
  - No clipping (< 0.1% samples at ±1.0)
  - Sufficient RMS level (not too quiet)
  - Correct sample rate (16 kHz)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from enrollment.enrollment_spec import VALIDATION_THRESHOLDS


@dataclass
class ValidationResult:
    """Result of validating a single utterance."""
    passed: bool
    duration_sec: float
    snr_db: float
    rms_level: float
    clipping_ratio: float
    rejection_reasons: List[str]
    guidance: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


class EnrollmentValidator:
    """
    Validates a single enrollment utterance against the spec thresholds.

    Usage
    -----
        validator = EnrollmentValidator()
        result = validator.validate_utterance(audio_float32, sample_rate=16000)
        if not result.passed:
            show_user(result.guidance)
            prompt_re_record()
    """

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = thresholds or VALIDATION_THRESHOLDS

    def validate_utterance(
        self, audio: np.ndarray, sample_rate: int
    ) -> ValidationResult:
        """
        Validate a single enrollment utterance.

        Parameters
        ----------
        audio : np.ndarray (N,) float32 mono
        sample_rate : int

        Returns
        -------
        ValidationResult
        """
        reasons: List[str] = []

        # 1. Sample rate check
        if sample_rate != self.thresholds["sample_rate_hz"]:
            reasons.append(
                f"Wrong sample rate: {sample_rate} Hz (required: "
                f"{self.thresholds['sample_rate_hz']} Hz)"
            )

        # 2. Duration
        duration_sec = len(audio) / sample_rate
        if duration_sec < self.thresholds["min_duration_sec"]:
            reasons.append(
                f"Too short: {duration_sec:.1f}s "
                f"(minimum: {self.thresholds['min_duration_sec']}s)"
            )

        # 3. RMS level
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.thresholds["min_rms_level"]:
            reasons.append(
                f"Signal too quiet: RMS={rms:.4f} "
                f"(minimum: {self.thresholds['min_rms_level']})"
            )

        # 4. Clipping
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
        if clipping_ratio > self.thresholds["max_clipping_ratio"]:
            reasons.append(
                f"Audio clipping detected: {clipping_ratio:.4%} of samples "
                f"(maximum: {self.thresholds['max_clipping_ratio']:.4%})"
            )

        # 5. SNR estimation (energy-based)
        snr_db = self._estimate_snr(audio, sample_rate)
        if snr_db < self.thresholds["min_snr_db"]:
            reasons.append(
                f"Background noise too high: SNR={snr_db:.1f} dB "
                f"(minimum: {self.thresholds['min_snr_db']} dB)"
            )

        passed = len(reasons) == 0
        guidance = self._build_guidance(reasons) if not passed else "Recording accepted."

        return ValidationResult(
            passed=passed,
            duration_sec=duration_sec,
            snr_db=snr_db,
            rms_level=rms,
            clipping_ratio=clipping_ratio,
            rejection_reasons=reasons,
            guidance=guidance,
        )

    def validate_enrollment_set(
        self, utterances: List[np.ndarray], sample_rate: int
    ) -> Tuple[List[np.ndarray], List[ValidationResult]]:
        """
        Validate a full enrollment set.

        Returns
        -------
        (valid_utterances, all_results)
        valid_utterances : only the utterances that passed — used for embedding.
        """
        valid: List[np.ndarray] = []
        results: List[ValidationResult] = []
        for utt in utterances:
            result = self.validate_utterance(utt, sample_rate)
            results.append(result)
            if result.passed:
                valid.append(utt)
        return valid, results

    # ── SNR estimation ────────────────────────────────────────────────────────

    def _estimate_snr(
        self, audio: np.ndarray, sample_rate: int, frame_ms: int = 20
    ) -> float:
        """
        Energy-based SNR estimate.

        Split audio into short frames. Frames in the bottom 10th percentile
        of energy are treated as noise; the median frame energy is treated as
        signal. Returns SNR in dB.

        This is an approximation — for deployment on noisy shop floors,
        consider a proper VAD-based noise estimation.
        """
        frame_samples = int(frame_ms / 1000 * sample_rate)
        if len(audio) < frame_samples:
            return 0.0

        n_frames = len(audio) // frame_samples
        frames = audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)
        energies = np.mean(frames ** 2, axis=1)

        noise_energy = float(np.percentile(energies, 10))
        signal_energy = float(np.median(energies))

        if noise_energy <= 1e-12:
            return 60.0  # Essentially silent background

        snr = 10 * np.log10(signal_energy / noise_energy)
        return float(snr)

    # ── Guidance builder ──────────────────────────────────────────────────────

    @staticmethod
    def _build_guidance(reasons: List[str]) -> str:
        """Map rejection reasons to actionable user-facing guidance."""
        messages = []
        for reason in reasons:
            if "short" in reason.lower():
                messages.append(
                    "Please speak for at least 5 seconds. "
                    "Read the full prompt out loud."
                )
            elif "quiet" in reason.lower():
                messages.append(
                    "Hold the phone 10–30 cm from your mouth and speak at "
                    "a normal volume."
                )
            elif "clipping" in reason.lower():
                messages.append(
                    "You're speaking too loudly. "
                    "Hold the phone a little further away and speak normally."
                )
            elif "noise" in reason.lower() or "snr" in reason.lower():
                messages.append(
                    "There's too much background noise. "
                    "Please move to a quieter area (staff room, stock area) "
                    "and try again."
                )
            elif "sample rate" in reason.lower():
                messages.append(
                    "Audio format error. Please restart the enrollment process."
                )
            else:
                messages.append(reason)
        return " ".join(messages)
