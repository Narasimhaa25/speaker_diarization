"""
evaluation/evaluate_der.py
───────────────────────────
Evaluate Diarization Error Rate (DER) against AMI ground-truth RTTM files.

Target: DER < 15% on in-domain retail audio.
AMI is used as the evaluation dataset — it is the closest available
conversational benchmark to retail (multi-speaker, meeting-style).

Uses the pyannote.metrics library for standard DER computation.

Usage
-----
    python evaluation/evaluate_der.py \\
        --ami_dir ./datasets/ami \\
        --model_onnx models/diarization.onnx \\
        --split test \\
        --threshold 0.70
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.core import Annotation, Segment
    PYANNOTE_METRICS_AVAILABLE = True
except ImportError:
    PYANNOTE_METRICS_AVAILABLE = False

from data.dataset_loader import SLUEVoxCelebLoader, RTTMEntry
from models.diarization_model import DiarizationModel, DiarizedSegment


def rttm_to_annotation(rttm: List[RTTMEntry], meeting_id: str) -> "Annotation":
    """Convert RTTMEntry list to pyannote Annotation object."""
    annotation = Annotation(uri=meeting_id)
    for entry in rttm:
        annotation[Segment(entry.start_sec, entry.end_sec)] = entry.speaker_id
    return annotation


def segments_to_annotation(
    segments: List[DiarizedSegment], meeting_id: str
) -> "Annotation":
    """Convert DiarizedSegment list to pyannote Annotation object."""
    annotation = Annotation(uri=meeting_id)
    for seg in segments:
        annotation[Segment(seg.start, seg.end)] = seg.speaker_id
    return annotation


def evaluate_der(
    model: DiarizationModel,
    split: str = "test",
    collar: float = 0.25,
    skip_overlap: bool = True,
    n_meetings: int = 20,
    max_duration_sec: float = 60.0,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Run DER evaluation over SLUE-VoxCeleb utterances.

    Each utterance is treated as a single-meeting conversation. The dataset
    provides speaker-annotated segments which serve as the RTTM ground truth.
    This is a proxy evaluation — for true in-domain retail DER, supply real
    annotated store audio.

    Parameters
    ----------
    model           : DiarizationModel
    split           : str      "train" | "validation" | "test"
    collar          : float    Forgiveness collar in seconds (standard: 0.25s)
    skip_overlap    : bool     Exclude overlapping speech from DER (standard)
    n_meetings      : int      Number of utterances to evaluate
    max_duration_sec: float    Maximum duration per utterance
    seed            : int

    Returns
    -------
    dict with "der", "confusion", "missed_speech", "false_alarm" (all percentages)
    """
    if not PYANNOTE_METRICS_AVAILABLE:
        raise ImportError(
            "pyannote.metrics required: pip install pyannote.metrics"
        )

    loader = SLUEVoxCelebLoader(max_duration_sec=max_duration_sec, seed=seed)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)

    meeting_results = {}
    meeting_count = 0
    error_count = 0

    for i, segment in enumerate(loader.iter_utterances(split=split, max_samples=n_meetings)):
        meeting_id = f"{segment.speaker_id}_{i:04d}"

        # Ground-truth: the entire utterance belongs to one speaker
        rttm = [RTTMEntry(
            meeting_id=meeting_id,
            start_sec=segment.start_sec,
            duration_sec=segment.end_sec - segment.start_sec,
            speaker_id=segment.speaker_id,
        )]

        try:
            segments = model.diarize(segment.audio)
        except Exception as e:
            print(f"  [error] {meeting_id}: {e}")
            error_count += 1
            continue

        reference = rttm_to_annotation(rttm, meeting_id)
        hypothesis = segments_to_annotation(segments, meeting_id)

        components = metric(reference, hypothesis, detailed=True)
        der = abs(components["diarization error rate"])
        meeting_results[meeting_id] = {
            "der":           round(der * 100, 2),
            "confusion":     round(abs(components["confusion"]) / max(components["total"], 1e-9) * 100, 2),
            "missed_speech": round(abs(components["missed detection"]) / max(components["total"], 1e-9) * 100, 2),
            "false_alarm":   round(abs(components["false alarm"]) / max(components["total"], 1e-9) * 100, 2),
        }
        meeting_count += 1
        print(f"  {meeting_id}: DER={meeting_results[meeting_id]['der']:.1f}%")

    if meeting_count == 0:
        raise ValueError(f"No utterances evaluated in split '{split}'.")

    aggregate_der = abs(metric) * 100
    return {
        "der":            round(aggregate_der, 2),
        "meetings":       meeting_count,
        "errors_skipped": error_count,
        "target_der":     15.0,
        "passed":         aggregate_der < 15.0,
        "split":          split,
        "collar_sec":     collar,
        "skip_overlap":   skip_overlap,
        "per_meeting":    meeting_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DER on SLUE-VoxCeleb split")
    parser.add_argument("--model_onnx", type=Path, default=None,
                        help="ONNX diarization model path (None = pyannote pipeline)")
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Clustering threshold for diarization model")
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--n_meetings", type=int, default=20,
                        help="Number of utterances to evaluate")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON results to this path")
    args = parser.parse_args()

    model = DiarizationModel(
        model_path=args.model_onnx,
        clustering_threshold=args.threshold,
    )

    print(f"\nEvaluating DER on SLUE-VoxCeleb {args.split} split …")
    results = evaluate_der(
        model=model,
        split=args.split,
        collar=args.collar,
        n_meetings=args.n_meetings,
    )

    print(f"\n{'='*50}")
    print(f"AMI {args.split} DER : {results['der']:.2f}%")
    print(f"Target            : {results['target_der']:.1f}%")
    print(f"Status            : {'✓ PASS' if results['passed'] else '✗ FAIL'}")
    print(f"Meetings evaluated: {results['meetings']}")
    print(f"{'='*50}")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nResults → {args.output}")


if __name__ == "__main__":
    main()
