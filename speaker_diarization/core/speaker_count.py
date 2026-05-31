"""
core/speaker_count.py
──────────────────────
Lightweight speaker-count estimator.

Serves two purposes:
  1. Sanity check when diarization clustering gets confused.
  2. Signal to Module 7 (group vs one-on-one customer detection).

Strategy: spectral clustering on short-window embeddings.
Works without the full ECAPA-TDNN model — uses MFCC-based pseudo-embeddings
for speed. Accuracy is sufficient as a *count* signal (not speaker identity).

Target: 2-6 speakers typical of retail conversations.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from sklearn.cluster import SpectralClustering
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SpeakerCountEstimator:
    """
    Estimates the number of distinct speakers in an audio clip.

    Uses a lightweight MFCC + spectral clustering approach.
    Results are constrained to [min_speakers, max_speakers].

    Parameters
    ----------
    min_speakers : int   Lower bound (retail always has ≥ 1 staff + 1 customer)
    max_speakers : int   Upper bound (retail rarely exceeds 6 in scope)
    window_sec   : float MFCC window size per "embedding"
    hop_sec      : float Hop between windows
    n_mfcc       : int   Number of MFCC coefficients
    """

    def __init__(
        self,
        min_speakers: int = 2,
        max_speakers: int = 6,
        window_sec: float = 2.0,
        hop_sec: float = 1.0,
        n_mfcc: int = 40,
    ):
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required: pip install librosa")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.n_mfcc = n_mfcc

    def estimate(self, audio: np.ndarray, sample_rate: int) -> int:
        """
        Estimate number of distinct speakers.

        Parameters
        ----------
        audio : np.ndarray (N,) float32 mono at sample_rate
        sample_rate : int

        Returns
        -------
        int — estimated speaker count in [min_speakers, max_speakers]
        """
        windows = self._extract_windows(audio, sample_rate)

        if len(windows) < self.min_speakers:
            # Not enough audio to cluster — return minimum
            return self.min_speakers

        features = self._window_features(windows, sample_rate)

        best_k = self._spectral_cluster_count(features)
        return int(np.clip(best_k, self.min_speakers, self.max_speakers))

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_windows(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[np.ndarray]:
        """Slice audio into overlapping windows."""
        window_samples = int(self.window_sec * sample_rate)
        hop_samples = int(self.hop_sec * sample_rate)
        windows = []
        start = 0
        while start + window_samples <= len(audio):
            windows.append(audio[start: start + window_samples])
            start += hop_samples
        return windows

    def _window_features(
        self, windows: list[np.ndarray], sample_rate: int
    ) -> np.ndarray:
        """
        Compute mean + std MFCC per window → (n_windows, 2*n_mfcc) feature matrix.
        Standardised for clustering stability.
        """
        feats = []
        for w in windows:
            mfcc = librosa.feature.mfcc(y=w, sr=sample_rate, n_mfcc=self.n_mfcc)
            feats.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
        X = np.array(feats, dtype=np.float32)
        X = StandardScaler().fit_transform(X)
        return X

    # ── Clustering ────────────────────────────────────────────────────────────

    def _spectral_cluster_count(self, X: np.ndarray) -> int:
        """
        Find optimal k in [min_speakers, max_speakers] via eigengap heuristic.

        The eigengap heuristic looks at gaps in the eigenvalues of the
        normalised Laplacian. The largest gap indicates the natural cluster count.
        """
        k_max = min(self.max_speakers, len(X) - 1)
        k_min = self.min_speakers

        if k_max < k_min:
            return k_min

        # Compute affinity matrix (cosine similarity)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-12)
        affinity = X_norm @ X_norm.T

        # Clip to [0, 1] — negative similarities → 0
        affinity = np.clip(affinity, 0, 1)

        # Degree matrix and normalised Laplacian
        degree = affinity.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-12))
        L_sym = np.eye(len(X)) - D_inv_sqrt @ affinity @ D_inv_sqrt

        eigenvalues = np.sort(np.linalg.eigvalsh(L_sym))

        # Eigengap: largest gap among eigenvalues [1..k_max]
        gaps = np.diff(eigenvalues[k_min - 1: k_max + 1])
        if len(gaps) == 0:
            return k_min
        best_k = k_min + int(np.argmax(gaps))
        return best_k

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def estimate_heuristic(self, audio: np.ndarray, sample_rate: int) -> int:
        """
        Faster energy-variance heuristic — no clustering.

        Counts "voice change" events based on sudden energy shifts.
        Less accurate but runs in O(N) — useful on very short clips.
        """
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # Count significant energy transitions
        delta = np.abs(np.diff(rms))
        threshold = delta.mean() + 2 * delta.std()
        transitions = int((delta > threshold).sum())

        # Rough heuristic: one transition every 3-5 seconds → new speaker
        audio_duration = len(audio) / sample_rate
        estimated = max(self.min_speakers, min(self.max_speakers, 1 + transitions // 4))
        return estimated
