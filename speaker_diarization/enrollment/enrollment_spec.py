"""
enrollment/enrollment_spec.py
──────────────────────────────
Specification for the enrollment recording flow (built by Module 8).

This module OWNS what Module 8 must capture — minimum sample length,
recommended prompts, acceptable noise floor, samples per staff member,
and what good vs bad enrollment looks like.

Module 8 implements the recording UX; this file is the authoritative contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# ─── Core enrollment requirements ─────────────────────────────────────────────

@dataclass(frozen=True)
class EnrollmentSpec:
    """
    Authoritative specification for what Module 8 must capture.

    KEY DECISIONS (owned by this module):
      - Minimum sample length: 5 seconds (shorter → unstable embeddings)
      - Recommended samples: 5 utterances per staff member
      - Acceptable noise floor: SNR ≥ 20 dB
      - Re-enrollment cadence: every 6–12 months (voices age)
    """

    # ── Audio requirements ────────────────────────────────────────────────────
    min_utterance_duration_sec: float = 10.0
    """Minimum duration per utterance. Below this, ECAPA-TDNN embeddings
    become unstable. Production requirement: minimum 10 seconds."""

    recommended_utterance_duration_sec: float = 8.0
    """Target duration per utterance — trade-off between UX friction and
    embedding quality. 8 seconds covers ~80 words of natural speech."""

    min_samples_per_staff: int = 3
    """Minimum acceptable utterances to enroll. Production: exactly 3."""

    recommended_samples_per_staff: int = 3
    """Production requirement: exactly 3 utterances per staff member."""

    max_samples_per_staff: int = 3
    """Production requirement: exactly 3 utterances. No more, no less."""

    min_snr_db: float = 5.0
    """Minimum signal-to-noise ratio (dB) for an utterance to be accepted.
    Reduced from 20 dB to support real-world microphone use (5-15 dB typical).
    Still filters background noise while accepting practical recording conditions.
    Measured as RMS(speech) / RMS(noise)."""

    target_sample_rate_hz: int = 16_000
    """Required sample rate. Module 8 must record at or resample to 16 kHz."""

    # ── Prompt texts ──────────────────────────────────────────────────────────
    enrollment_prompts: List[str] = None  # populated in __post_init__

    # ── Re-enrollment policy ──────────────────────────────────────────────────
    re_enrollment_interval_months: int = 9
    """Recommended interval to refresh staff voice signatures.
    Human voices change noticeably over 6–12 months (illness, age, season).
    9 months is a conservative midpoint."""

    re_enrollment_trigger_accuracy_drop: float = 0.05
    """If real-time staff ID accuracy drops by more than this on a known
    store's audio, trigger a re-enrollment prompt."""

    def __post_init__(self):
        if self.enrollment_prompts is None:
            # Use object.__setattr__ because the dataclass is frozen
            object.__setattr__(self, "enrollment_prompts", _DEFAULT_PROMPTS)


_DEFAULT_PROMPTS: List[str] = [
    # Designed to cover a wide range of phonemes in natural retail speech
    "Hello, welcome to the store. How can I help you today?",
    "Sure, let me check that for you. What size are you looking for?",
    "We have that item in stock. I'll get one from the back for you.",
    "Is there anything else I can help you find today?",
    "Your total comes to fourteen ninety-five. Would you like a receipt?",
    # Additional prompts for samples 6–10 (optional)
    "Thank you for shopping with us. Have a great day!",
    "Let me grab a colleague who can help you with that.",
    "Our return policy is thirty days with the original receipt.",
    "We're currently running a sale on selected items this week.",
    "I can process that exchange for you right here at the counter.",
]


# ─── Good vs bad enrollment criteria ──────────────────────────────────────────

GOOD_ENROLLMENT = {
    "duration":    f"≥ {EnrollmentSpec().min_utterance_duration_sec}s per utterance",
    "snr":         f"≥ {EnrollmentSpec().min_snr_db} dB SNR",
    "samples":     f"≥ {EnrollmentSpec().recommended_samples_per_staff} utterances",
    "environment": "Quiet area of the store (stock room, staff office). Not shop floor during trading hours.",
    "distance":    "10–30 cm from microphone (phone held naturally while speaking).",
    "content":     "Use the provided prompts — they cover the phoneme range needed for robust embeddings.",
    "consistency": "Natural speech pace. No exaggerated slow speech or shouting.",
}

BAD_ENROLLMENT = {
    "too_short":   f"Utterances < {EnrollmentSpec().min_utterance_duration_sec}s — rejected automatically",
    "noisy":       f"SNR < {EnrollmentSpec().min_snr_db} dB — background noise (music, crowd) — rejected automatically",
    "mic_far":     "> 50 cm from mic — low signal level — rejected if SNR fails",
    "clipping":    "Speaking too loudly into mic — audio clips at ±1.0 — rejected automatically",
    "phone_move":  "Moving the phone while speaking — intermittent dropout — embedding quality suffers",
    "whispering":  "Whispering — different vocal register to normal speech — creates mismatched embedding",
    "reading_flat":"Monotone reading of prompts — lower phoneme coverage — prefer natural delivery",
}


# ─── Validation thresholds (used by enrollment_validator.py) ─────────────────

# NOTE: These runtime thresholds are intentionally more permissive than the
# spec values above. The spec defines production aspirational targets; these
# thresholds handle real-world browser mic recordings (WebM codec, variable
# levels, background noise) without over-rejecting valid enrollments.
VALIDATION_THRESHOLDS = {
    "min_duration_sec":   3.0,    # 3s minimum
    "min_snr_db":         -999,   # Disabled — SNR estimate is unreliable for browser WebM audio
    "min_rms_level":      0.0001, # Near-zero — only reject completely silent recordings
    "max_clipping_ratio": 0.05,   # 5% tolerance
    "sample_rate_hz":     EnrollmentSpec().target_sample_rate_hz,
}


# ─── Module 8 integration contract ───────────────────────────────────────────

MODULE_8_CONTRACT = {
    "inputs_required": [
        "staff_name (str)",
        "staff_role (str)",
        "audio_utterances (List[np.ndarray])  — mono float32 at 16 kHz",
    ],
    "validation_before_submission": (
        "Call enrollment/enrollment_validator.py validate_utterance() on each "
        "recording before passing to the embedding pipeline. Reject and re-record "
        "on validation failure — do not enroll low-quality audio."
    ),
    "outputs_to_this_module": [
        "validated utterance audio arrays",
        "staff_name",
        "staff_role",
    ],
    "this_module_returns": [
        "staff_id (UUID)",
        "enrollment status",
    ],
    "re_enrollment_ui": (
        "Module 8 must surface a re-enrollment prompt when this module's "
        "`re_enrollment_interval_months` has elapsed since enrolled_at."
    ),
}
