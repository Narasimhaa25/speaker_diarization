"""
models/diarization_model.py
────────────────────────────
pyannote-onnx wrapper — produces (start, end, speaker_id) tuples per audio chunk.

Clustering threshold is tunable; default is set for 2-6 speaker conversations
typical of retail environments. Target DER < 15% on in-domain audio.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime required: pip install onnxruntime")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


# ─── Data contract ────────────────────────────────────────────────────────────

@dataclass
class DiarizedSegment:
    """One speaker turn — the primary output contract of this module."""
    start: float        # seconds
    end: float          # seconds
    speaker_id: str     # e.g. "SPEAKER_00", "SPEAKER_01"
    confidence: float = 1.0


# ─── Diarization model ────────────────────────────────────────────────────────

class DiarizationModel:
    """
    Wraps pyannote-onnx to produce (start, end, speaker_id) segments.

    Parameters
    ----------
    model_path : Path | None
        Path to a pyannote ONNX segmentation model. If None, falls back to
        pyannote.audio Pipeline (requires HuggingFace token).
    clustering_threshold : float
        Agglomerative clustering threshold for speaker merging.
        Lower = more speakers split. Tuned for retail 2-6 speaker conversations.
        Default 0.7 is a good starting point; run threshold_tuner.py to optimise.
    min_duration_on : float
        Minimum duration (seconds) of an active speaker segment.
    min_duration_off : float
        Minimum silence gap before a new speaker turn starts.
    sample_rate : int
        Expected input sample rate. Audio is NOT resampled here — caller must
        ensure 16 kHz mono float32 input.
    """

    # Retail-tuned defaults — see KEY DECISIONS in README
    DEFAULT_CLUSTERING_THRESHOLD = 0.70
    DEFAULT_MIN_DURATION_ON = 0.10   # 100 ms
    DEFAULT_MIN_DURATION_OFF = 0.05  # 50 ms

    def __init__(
        self,
        model_path: Optional[Path] = None,
        clustering_threshold: float = DEFAULT_CLUSTERING_THRESHOLD,
        min_duration_on: float = DEFAULT_MIN_DURATION_ON,
        min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
        sample_rate: int = 16_000,
    ):
        self.clustering_threshold = clustering_threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.sample_rate = sample_rate
        self._model_path = model_path
        self._session: Optional[ort.InferenceSession] = None
        self._pipeline = None

        if model_path is not None:
            self._load_onnx(model_path)
        elif PYANNOTE_AVAILABLE:
            self._load_pyannote_pipeline()
        else:
            raise RuntimeError(
                "Provide model_path (ONNX) or install pyannote.audio for pipeline mode."
            )

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_onnx(self, model_path: Path) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print(f"DiarizationModel: loaded ONNX from {model_path}")

    def _load_pyannote_pipeline(self) -> None:
        """Load pyannote.audio Pipeline with fixes for short-clip diarization."""
        import os
        from pyannote.audio import Pipeline

        # Load .env if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not token:
            hf_cache = os.path.expanduser("~/.cache/huggingface/token")
            if os.path.exists(hf_cache):
                token = open(hf_cache).read().strip()
        if not token:
            raise RuntimeError(
                "HF_TOKEN not set. Add it to .env:\n  HF_TOKEN=hf_your_token_here"
            )

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )

        # Fix: default min_cluster_size=12 causes 0 segments on clips < ~30s.
        # Setting to 1 allows diarization on clips of any length.
        try:
            self._pipeline._pipelines["clustering"]._instantiated["min_cluster_size"] = 1
        except Exception:
            pass

        print("DiarizationModel: loaded pyannote.audio Pipeline (neural mode) ✓")

    # ── Inference ─────────────────────────────────────────────────────────────

    def diarize(
        self, audio: np.ndarray, num_speakers: Optional[int] = None
    ) -> List[DiarizedSegment]:
        """
        Run diarization on a mono float32 audio array at self.sample_rate.

        Parameters
        ----------
        audio : np.ndarray shape (N,)
            Mono float32 audio, already at self.sample_rate.
        num_speakers : int | None
            If known, constrains the clustering. None = auto-detect.

        Returns
        -------
        List[DiarizedSegment]
            Sorted by start time.
        """
        if self._session is not None:
            return self._diarize_onnx(audio, num_speakers)
        elif self._pipeline is not None:
            return self._diarize_pyannote(audio, num_speakers)
        else:
            raise RuntimeError("No model loaded.")

    def _diarize_onnx(
        self, audio: np.ndarray, num_speakers: Optional[int]
    ) -> List[DiarizedSegment]:
        """
        ONNX inference path.

        The pyannote-onnx model expects:
          input  : float32 [1, 1, N]  (batch, channel, samples)
          output : dict with "diarization" containing speaker turn boundaries.

        Post-processing mirrors pyannote.audio's internal clustering but runs
        offline without a HuggingFace token.
        """
        audio_input = audio[np.newaxis, np.newaxis, :]  # [1, 1, N]

        outputs = self._session.run(
            None,
            {"input": audio_input.astype(np.float32)},
        )

        # Output[0]: float32 [1, num_frames, num_speakers] — speaker activity scores
        speaker_activity = outputs[0][0]  # [num_frames, num_speakers]
        num_frames, num_local_speakers = speaker_activity.shape

        # Convert frame-level activity to segments
        frame_duration = 10.0 / 1000.0  # 10 ms frames
        segments = self._activity_to_segments(
            speaker_activity, frame_duration, num_speakers
        )
        return segments

    def _activity_to_segments(
        self,
        activity: np.ndarray,
        frame_duration: float,
        num_speakers: Optional[int],
    ) -> List[DiarizedSegment]:
        """Convert frame-level activity matrix to DiarizedSegment list."""
        n_frames, n_speakers = activity.shape
        if num_speakers is not None:
            n_speakers = min(n_speakers, num_speakers)

        segments: List[DiarizedSegment] = []
        for spk_idx in range(n_speakers):
            active = activity[:, spk_idx] > self.clustering_threshold
            in_segment = False
            start_frame = 0
            for f, is_active in enumerate(active):
                if is_active and not in_segment:
                    start_frame = f
                    in_segment = True
                elif not is_active and in_segment:
                    start_sec = start_frame * frame_duration
                    end_sec = f * frame_duration
                    if (end_sec - start_sec) >= self.min_duration_on:
                        segments.append(DiarizedSegment(
                            start=start_sec,
                            end=end_sec,
                            speaker_id=f"SPEAKER_{spk_idx:02d}",
                        ))
                    in_segment = False
            if in_segment:
                start_sec = start_frame * frame_duration
                end_sec = n_frames * frame_duration
                if (end_sec - start_sec) >= self.min_duration_on:
                    segments.append(DiarizedSegment(
                        start=start_sec,
                        end=end_sec,
                        speaker_id=f"SPEAKER_{spk_idx:02d}",
                    ))

        segments.sort(key=lambda s: s.start)
        return segments

    def _diarize_pyannote(
        self, audio: np.ndarray, num_speakers: Optional[int]
    ) -> List[DiarizedSegment]:
        """Pyannote.audio Pipeline — passes audio via a temp WAV file.

        Passing a raw numpy tensor via the dict API produces empty results
        in pyannote 3.x on certain torchaudio backends. Writing to a temp
        WAV and passing the file path is the reliable path.
        """
        import tempfile, os, soundfile as sf

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, self.sample_rate, subtype="PCM_16")

            kwargs = {}
            if num_speakers is not None:
                kwargs["num_speakers"] = num_speakers

            diarization = self._pipeline(tmp_path, **kwargs)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        segments: List[DiarizedSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizedSegment(
                start=turn.start,
                end=turn.end,
                speaker_id=speaker,
            ))
        return sorted(segments, key=lambda s: s.start)

    # ── Configuration ─────────────────────────────────────────────────────────

    def set_clustering_threshold(self, threshold: float) -> None:
        """Adjust clustering threshold at runtime (used by threshold_tuner.py)."""
        self.clustering_threshold = threshold

    def __repr__(self) -> str:
        mode = "ONNX" if self._session else "pyannote Pipeline"
        return (
            f"DiarizationModel(mode={mode}, "
            f"clustering_threshold={self.clustering_threshold:.3f})"
        )
