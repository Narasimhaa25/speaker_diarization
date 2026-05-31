"""
utils/audio_utils.py
─────────────────────
Audio loading, chunking, resampling, and basic VAD helpers.

All audio in this module is expected to be:
  - Mono (single channel)
  - float32
  - 16,000 Hz (resampled on load if needed)
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

TARGET_SR = 16_000


def load_audio(
    path: Path, target_sr: int = TARGET_SR, mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target_sr if needed.

    Supports WAV, FLAC, MP3, M4A, OGG (via librosa/soundfile).

    Parameters
    ----------
    path : Path
    target_sr : int   Target sample rate (default 16 kHz)
    mono : bool       Downmix to mono if True

    Returns
    -------
    (audio, sample_rate) — float32 mono ndarray
    """
    if not AUDIO_LIBS_AVAILABLE:
        raise ImportError("librosa and soundfile required: pip install librosa soundfile")

    audio, sr = librosa.load(str(path), sr=target_sr, mono=mono)
    return audio.astype(np.float32), sr


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_sec: float = 30.0,
    overlap_sec: float = 2.0,
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """
    Yield (chunk, start_sec) pairs of fixed-length overlapping windows.

    Parameters
    ----------
    audio : np.ndarray (N,) float32
    sample_rate : int
    chunk_duration_sec : float  Window size in seconds
    overlap_sec : float         Overlap between consecutive chunks

    Yields
    ------
    (chunk_audio, start_sec)
    """
    chunk_samples = int(chunk_duration_sec * sample_rate)
    hop_samples = int((chunk_duration_sec - overlap_sec) * sample_rate)
    if hop_samples <= 0:
        hop_samples = chunk_samples

    start = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        if len(chunk) > 0:
            yield chunk, start / sample_rate
        if end >= len(audio):
            break
        start += hop_samples


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def normalize_audio(audio: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    """
    Normalise audio to a target RMS loudness in dBFS.
    Used to standardise input levels before embedding.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    return (audio * (target_rms / rms)).astype(np.float32)


def estimate_snr_db(audio: np.ndarray, sample_rate: int, frame_ms: int = 20) -> float:
    """
    Simple energy-based SNR estimate (see enrollment_validator.py for details).
    Exposed here for shared use.
    """
    frame_samples = int(frame_ms / 1000 * sample_rate)
    if len(audio) < frame_samples:
        return 0.0
    n_frames = len(audio) // frame_samples
    frames = audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    energies = np.mean(frames ** 2, axis=1)
    noise_e = float(np.percentile(energies, 10))
    signal_e = float(np.median(energies))
    if noise_e <= 1e-12:
        return 60.0
    return float(10 * np.log10(signal_e / noise_e))


def voice_activity_detection(
    audio: np.ndarray,
    sample_rate: int,
    energy_threshold_db: float = -40.0,
    frame_ms: int = 20,
    min_speech_ms: float = 200.0,
) -> List[Tuple[float, float]]:
    """
    Simple energy-based VAD.

    Returns list of (start_sec, end_sec) speech segments.
    Not as accurate as pyannote VAD — use for pre-filtering only.
    """
    frame_samples = int(frame_ms / 1000 * sample_rate)
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return []

    frames = audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)
    is_speech = rms_db > energy_threshold_db

    min_speech_frames = int(min_speech_ms / frame_ms)
    segments: List[Tuple[float, float]] = []
    in_speech = False
    start_frame = 0

    for i, active in enumerate(is_speech):
        if active and not in_speech:
            start_frame = i
            in_speech = True
        elif not active and in_speech:
            if (i - start_frame) >= min_speech_frames:
                segments.append((
                    start_frame * frame_ms / 1000,
                    i * frame_ms / 1000,
                ))
            in_speech = False

    if in_speech and (n_frames - start_frame) >= min_speech_frames:
        segments.append((
            start_frame * frame_ms / 1000,
            n_frames * frame_ms / 1000,
        ))

    return segments


def is_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    """Return True if the audio has significant clipping."""
    return float(np.mean(np.abs(audio) >= threshold)) > 0.001


def duration_sec(audio: np.ndarray, sample_rate: int) -> float:
    """Return audio duration in seconds."""
    return len(audio) / sample_rate


def reduce_noise(audio: np.ndarray, sample_rate: int, noise_duration_sec: float = 1.0) -> np.ndarray:
    """
    Simple spectral subtraction noise reduction.
    
    Assumes the first `noise_duration_sec` contains only noise.
    Subtracts noise spectrum from entire signal.
    
    Parameters
    ----------
    audio : np.ndarray  Input audio (mono, float32)
    sample_rate : int   Sample rate (Hz)
    noise_duration_sec : float  Duration of noise-only segment at start
    
    Returns
    -------
    np.ndarray  Noise-reduced audio
    """
    try:
        import scipy.signal
    except ImportError:
        print("⚠ scipy.signal not available; returning audio unchanged")
        return audio
    
    # Extract noise profile from beginning
    noise_samples = int(noise_duration_sec * sample_rate)
    noise_samples = min(noise_samples, len(audio) // 4)  # Max 25% of signal
    
    if noise_samples < 512:
        return audio
    
    noise = audio[:noise_samples]
    
    # STFT parameters
    nperseg = 512
    noverlap = nperseg // 2
    
    # Compute STFT
    f, t, Zxx = scipy.signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_noise = scipy.signal.stft(noise, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    # Noise power spectrum (average over time)
    noise_power = np.mean(np.abs(Zxx_noise) ** 2, axis=1, keepdims=True)
    
    # Spectral subtraction with flooring
    signal_power = np.abs(Zxx) ** 2
    cleaned_power = np.maximum(signal_power - 2.0 * noise_power, 0.1 * noise_power)
    
    # Reconstruct phase from original signal
    phase = np.angle(Zxx)
    Zxx_cleaned = np.sqrt(cleaned_power) * np.exp(1j * phase)
    
    # Inverse STFT
    _, reduced = scipy.signal.istft(Zxx_cleaned, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    # Match length and normalize
    reduced = reduced[:len(audio)]
    reduced = reduced.astype(np.float32)
    
    # Normalize to original RMS
    orig_rms = np.sqrt(np.mean(audio ** 2))
    reduced_rms = np.sqrt(np.mean(reduced ** 2))
    if reduced_rms > 1e-10:
        reduced = reduced * (orig_rms / reduced_rms)
    
    return reduced
