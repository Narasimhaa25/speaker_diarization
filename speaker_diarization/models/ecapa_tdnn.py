"""
models/ecapa_tdnn.py
─────────────────────
ECAPA-TDNN 192-dim voice embedding extractor.

Loads SpeechBrain's pre-trained ECAPA-TDNN and runs ONNX INT8 inference.
Shared with Module 3 — see embedding contract at the bottom of this file.

Embedding contract (agreed with Module 3)
─────────────────────────────────────────
  shape      : (192,)          float32
  normalised : L2-normalised   yes (cosine similarity = dot product)
  model      : ECAPA-TDNN trained on VoxCeleb 1+2
  export     : ONNX INT8 quantised (see export_onnx.py)
  input      : mono float32 audio at 16 kHz, any duration ≥ 0.5 s
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


EMBEDDING_DIM = 192          # agreed contract with Module 3
TARGET_SR = 16_000


class ECAPATDNNEmbedder:
    """
    192-dim L2-normalised speaker embedding extractor via ONNX INT8 model.

    Parameters
    ----------
    onnx_path : Path
        Path to the INT8-quantised ONNX model (produced by export_onnx.py).
    num_threads : int
        ORT intra-op threads. 2 is a good mobile default.

    Usage
    -----
        embedder = ECAPATDNNEmbedder(Path("models/ecapa_tdnn_int8.onnx"))
        embedding = embedder.embed(audio_float32_16khz)  # → (192,) float32
    """

    def __init__(self, onnx_path: Path, num_threads: int = 2):
        if not ORT_AVAILABLE:
            raise ImportError("onnxruntime required: pip install onnxruntime")
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Run: python models/export_onnx.py --output models/ecapa_tdnn_int8.onnx"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Cache input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Verify output shape — dim may be symbolic (string) when dynamic axes used
        output_shape = self._session.get_outputs()[0].shape
        expected_dim = output_shape[-1] if output_shape else None
        if isinstance(expected_dim, int) and expected_dim != EMBEDDING_DIM:
            raise ValueError(
                f"ONNX model output dim {expected_dim} ≠ contract dim {EMBEDDING_DIM}. "
                "Re-export with export_onnx.py."
            )
        print(f"ECAPATDNNEmbedder: loaded {onnx_path} (INT8, {EMBEDDING_DIM}-dim)")

    # ── Core embedding ────────────────────────────────────────────────────────

    @staticmethod
    def _audio_to_fbank(audio: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
        """
        Convert raw audio to 80-dim log-mel filterbank features.
        Output shape: (1, T, 80) float32 — matches the ONNX input contract.
        Uses librosa so there is no torch/torchaudio dependency at inference.
        """
        import librosa
        # 25ms window, 10ms hop — standard for SpeechBrain ECAPA-TDNN
        hop_length  = int(0.010 * sample_rate)
        n_fft       = int(0.025 * sample_rate)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=80, fmin=20, fmax=sample_rate // 2,
        )
        log_mel = np.log(mel + 1e-6).T.astype(np.float32)   # (T, 80)
        return log_mel[np.newaxis, :, :]                      # (1, T, 80)

    def embed(self, audio: np.ndarray, sample_rate: int = TARGET_SR) -> np.ndarray:
        """
        Compute a 192-dim L2-normalised embedding from raw audio.

        Parameters
        ----------
        audio : np.ndarray shape (N,)
            Mono float32 audio. Must be at 16 kHz — no internal resampling.
        sample_rate : int
            Must equal 16_000. Provided for caller validation only.

        Returns
        -------
        np.ndarray shape (192,)
            L2-normalised float32 embedding.
        """
        if sample_rate != TARGET_SR:
            raise ValueError(
                f"ECAPATDNNEmbedder expects 16 kHz audio, got {sample_rate} Hz."
            )
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio (1D), got shape {audio.shape}")

        min_samples = int(0.5 * TARGET_SR)
        if len(audio) < min_samples:
            raise ValueError(
                f"Audio too short ({len(audio)/TARGET_SR:.2f}s < 0.5s minimum)."
            )

        # Convert raw audio → log-mel filterbanks → ONNX encoder → embedding
        feats = self._audio_to_fbank(audio, sample_rate)  # (1, T, 80)

        outputs = self._session.run(
            [self._output_name], {self._input_name: feats}
        )

        embedding = outputs[0].squeeze()           # (192,)
        embedding = self._l2_normalise(embedding)
        return embedding

    def embed_batch(self, audio_list: list[np.ndarray]) -> np.ndarray:
        """
        Compute embeddings for a list of audio arrays.
        Processes one at a time (filterbank lengths vary per utterance).
        Returns shape (B, 192).
        """
        return np.stack([self.embed(a) for a in audio_list], axis=0)

    # ── Mean embedding for an enrollment set ─────────────────────────────────

    def mean_embedding(self, audio_segments: list[np.ndarray]) -> np.ndarray:
        """
        Average embedding over multiple samples — used for staff enrollment.

        More samples → more robust centroid. Minimum 3 samples recommended
        (see enrollment/enrollment_spec.py).

        Returns shape (192,) L2-normalised.
        """
        embeddings = self.embed_batch(audio_segments)         # (N, 192)
        mean = embeddings.mean(axis=0)
        return self._l2_normalise(mean)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    @staticmethod
    def _l2_normalise_batch(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / (norms + 1e-12)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Dot product of two L2-normalised vectors = cosine similarity."""
        return float(np.dot(a, b))

    def __repr__(self) -> str:
        return f"ECAPATDNNEmbedder(dim={EMBEDDING_DIM}, quantisation=INT8)"


# ─── Embedding contract (shared with Module 3) ────────────────────────────────
EMBEDDING_CONTRACT = {
    "model":       "ECAPA-TDNN",
    "trained_on":  ["VoxCeleb1", "VoxCeleb2"],
    "dim":         EMBEDDING_DIM,
    "dtype":       "float32",
    "normalised":  True,
    "similarity":  "cosine (dot product after L2-norm)",
    "onnx_quant":  "INT8",
    "input_sr":    TARGET_SR,
    "min_dur_sec": 0.5,
}
