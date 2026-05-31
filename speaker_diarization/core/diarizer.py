"""
core/diarizer.py
─────────────────
Orchestrates the full diarization pipeline:
  audio chunk → (start, end, speaker_id, role) tuples

Role is "staff" | "customer" | "unknown" depending on staff DB availability.

This is the primary entry point for downstream modules.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from models.diarization_model import DiarizationModel, DiarizedSegment
from models.ecapa_tdnn import ECAPATDNNEmbedder
from core.speaker_count import SpeakerCountEstimator
from core.staff_identifier import StaffIdentifier
from staff_db.db_manager import StaffDBManager
from utils.audio_utils import load_audio, chunk_audio


@dataclass
class AnnotatedSegment:
    """Full output per speaker turn — contract for downstream modules."""
    start: float              # seconds
    end: float                # seconds
    speaker_id: str           # local within-conversation ID, e.g. "SPEAKER_00"
    role: str                 # "staff" | "customer" | "unknown"
    staff_name: Optional[str] = None    # populated if role == "staff"
    embedding: Optional[list] = None   # 192-dim float list; included if requested
    confidence: float = 1.0


class Diarizer:
    """
    Full pipeline: audio → AnnotatedSegment list.

    Parameters
    ----------
    diarization_model : DiarizationModel
    embedder : ECAPATDNNEmbedder
    staff_db : StaffDBManager | None
        If None, all speakers are labelled "unknown".
    include_embeddings : bool
        Whether to include raw embeddings in output (e.g. for Module 3).
    min_segment_duration : float
        Segments shorter than this (seconds) are discarded.
    """

    def __init__(
        self,
        diarization_model: DiarizationModel,
        embedder: ECAPATDNNEmbedder,
        staff_db: Optional[StaffDBManager] = None,
        include_embeddings: bool = False,
        min_segment_duration: float = 0.5,
    ):
        self.diarization_model = diarization_model
        self.embedder = embedder
        self.staff_identifier = StaffIdentifier(staff_db) if staff_db else None
        self.speaker_count_estimator = SpeakerCountEstimator()
        self.include_embeddings = include_embeddings
        self.min_segment_duration = min_segment_duration

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None,
    ) -> List[AnnotatedSegment]:
        """
        Process a full audio array.

        Parameters
        ----------
        audio : np.ndarray (N,) float32 mono
        sample_rate : int — must be 16_000
        num_speakers : int | None — constrain clustering if known

        Returns
        -------
        List[AnnotatedSegment] sorted by start time.
        """
        t0 = time.perf_counter()

        # 1. Estimate speaker count (sanity check / signal to Module 7)
        estimated_count = self.speaker_count_estimator.estimate(audio, sample_rate)

        # 2. Diarize
        diarized: List[DiarizedSegment] = self.diarization_model.diarize(
            audio, num_speakers=num_speakers or estimated_count
        )

        # 3. Filter short segments
        diarized = [
            s for s in diarized
            if (s.end - s.start) >= self.min_segment_duration
        ]

        # 4. Embed each segment & identify staff/customer
        results: List[AnnotatedSegment] = []
        for seg in diarized:
            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            chunk = audio[start_sample:end_sample]

            try:
                embedding = self.embedder.embed(chunk, sample_rate)
            except ValueError:
                # Segment too short to embed — skip
                continue

            role = "unknown"
            staff_name = None
            if self.staff_identifier:
                role, staff_name, _score = self.staff_identifier.identify(embedding)

            annotated = AnnotatedSegment(
                start=seg.start,
                end=seg.end,
                speaker_id=seg.speaker_id,
                role=role,
                staff_name=staff_name,
                embedding=embedding.tolist() if self.include_embeddings else None,
                confidence=seg.confidence,
            )
            results.append(annotated)

        elapsed = time.perf_counter() - t0
        audio_duration = len(audio) / sample_rate
        print(
            f"Diarizer: {len(results)} segments, "
            f"{estimated_count} estimated speakers, "
            f"RTF={elapsed/audio_duration:.3f}"
        )
        return results

    def process_file(
        self, audio_path: Path, staff_db_path: Optional[Path] = None
    ) -> List[AnnotatedSegment]:
        """Convenience wrapper — loads audio from file."""
        audio, sr = load_audio(audio_path, target_sr=16_000)
        return self.process(audio, sr)

    def to_json(self, segments: List[AnnotatedSegment], indent: int = 2) -> str:
        """Serialise results as JSON for downstream consumption."""
        data = [asdict(s) for s in segments]
        return json.dumps(data, indent=indent)

    def to_rttm(self, segments: List[AnnotatedSegment], meeting_id: str = "AUDIO") -> str:
        """
        Serialise results in RTTM format for DER evaluation.
        SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
        """
        lines = []
        for s in segments:
            dur = s.end - s.start
            lines.append(
                f"SPEAKER {meeting_id} 1 {s.start:.3f} {dur:.3f} "
                f"<NA> <NA> {s.speaker_id} <NA> <NA>"
            )
        return "\n".join(lines)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run diarization on an audio file")
    parser.add_argument("--audio",    required=True,  type=Path)
    parser.add_argument("--onnx",     required=False, type=Path,
                        default=Path("models/ecapa_tdnn_int8.onnx"))
    parser.add_argument("--staff_db", required=False, type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--output",    type=Path, default=None,
                        help="Write JSON results to this path")
    parser.add_argument("--include_embeddings", action="store_true")
    args = parser.parse_args()

    diar_model = DiarizationModel(clustering_threshold=args.threshold)
    embedder = ECAPATDNNEmbedder(args.onnx)

    staff_db = None
    if args.staff_db:
        staff_db = StaffDBManager(args.staff_db)

    diarizer = Diarizer(
        diarization_model=diar_model,
        embedder=embedder,
        staff_db=staff_db,
        include_embeddings=args.include_embeddings,
    )

    segments = diarizer.process_file(args.audio)
    output_json = diarizer.to_json(segments)
    print(output_json)

    if args.output:
        args.output.write_text(output_json)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
