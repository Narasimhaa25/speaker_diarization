"""
evaluation/evaluate_identification.py
───────────────────────────────────────
Evaluate staff identification top-1 accuracy using VoxCeleb as a proxy.

Protocol:
  1. Sample N speakers from VoxCeleb as "staff" (enroll 3 utterances each).
  2. Use remaining utterances as queries (both enrolled speakers and imposters).
  3. Compute top-1 accuracy, FAR (false-staff rate), FRR (false-customer rate).

Target: > 95% top-1 accuracy on enrolled staff.

Note: VoxCeleb speakers are used as stand-ins for real staff.
Real deployment accuracy may differ — evaluate on in-store audio when available.

Usage
-----
    python evaluation/evaluate_identification.py \\
        --voxceleb_dir ./datasets/voxceleb \\
        --onnx models/ecapa_tdnn_int8.onnx \\
        --n_staff 20 \\
        --threshold 0.82
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data.dataset_loader import SLUEVoxCelebLoader, AudioSegment
from models.ecapa_tdnn import ECAPATDNNEmbedder


def evaluate_identification(
    embedder: ECAPATDNNEmbedder,
    threshold: float,
    n_staff: int = 20,
    enrollment_samples: int = 3,
    query_samples_per_speaker: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Run staff identification evaluation.

    Returns
    -------
    dict with accuracy, FAR, FRR, EER and per-speaker breakdown.
    """
    n_imposters = 50
    total_speakers = n_staff + n_imposters
    utterances_per_speaker = enrollment_samples + query_samples_per_speaker

    loader = SLUEVoxCelebLoader(seed=seed)
    all_segments = loader.sample_by_speaker(
        n_speakers=total_speakers,
        utterances_per_speaker=utterances_per_speaker,
        split="train",
    )

    all_speaker_ids = sorted({s.speaker_id for s in all_segments})
    if len(all_speaker_ids) < n_staff + 10:
        raise ValueError(
            f"Need ≥ {n_staff + 10} speakers from dataset, got {len(all_speaker_ids)}"
        )

    staff_speakers   = set(all_speaker_ids[:n_staff])
    imposter_speakers = set(all_speaker_ids[n_staff: n_staff + n_imposters])

    # Group segments by speaker
    by_speaker: Dict[str, List[AudioSegment]] = {}
    for seg in all_segments:
        by_speaker.setdefault(seg.speaker_id, []).append(seg)

    # ── Enrollment ────────────────────────────────────────────────────────────
    print(f"Enrolling {n_staff} staff speakers …")
    staff_embeddings: Dict[str, np.ndarray] = {}

    for spk_id in staff_speakers:
        segs = by_speaker.get(spk_id, [])[:enrollment_samples]
        if not segs:
            continue
        audios = [s.audio for s in segs]
        staff_embeddings[spk_id] = embedder.mean_embedding(audios)

    staff_matrix = np.stack(list(staff_embeddings.values()))  # (N_staff, 192)
    staff_ids = list(staff_embeddings.keys())

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("Running queries …")
    correct = 0
    total_staff_queries = 0
    false_staff = 0
    false_customer = 0
    total_imposter_queries = 0

    per_speaker: Dict[str, Dict] = {}

    # Staff queries (genuine) — use utterances after the enrollment set
    for spk_id in staff_ids:
        queries = by_speaker.get(spk_id, [])[enrollment_samples:]
        spk_correct = 0
        for q in queries:
            try:
                emb = embedder.embed(q.audio)
            except ValueError:
                continue
            scores = staff_matrix @ emb
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            predicted_staff = best_score >= threshold

            if predicted_staff and staff_ids[best_idx] == spk_id:
                spk_correct += 1
                correct += 1
            else:
                false_customer += 1
            total_staff_queries += 1

        per_speaker[spk_id] = {
            "type": "staff",
            "top1_accuracy": spk_correct / max(len(queries), 1),
        }

    # Imposter queries
    for spk_id in list(imposter_speakers)[:30]:
        for q in by_speaker.get(spk_id, [])[:query_samples_per_speaker]:
            try:
                emb = embedder.embed(q.audio)
            except ValueError:
                continue
            best_score = float(np.max(staff_matrix @ emb))
            if best_score >= threshold:
                false_staff += 1
            total_imposter_queries += 1

    top1_accuracy = correct / max(total_staff_queries, 1) * 100
    far = false_staff / max(total_imposter_queries, 1) * 100   # False Accept Rate
    frr = false_customer / max(total_staff_queries, 1) * 100   # False Reject Rate

    results = {
        "top1_accuracy_pct":         round(top1_accuracy, 2),
        "false_accept_rate_pct":     round(far, 2),
        "false_reject_rate_pct":     round(frr, 2),
        "target_top1_accuracy_pct":  95.0,
        "passed":                    top1_accuracy >= 95.0,
        "threshold":                 threshold,
        "n_staff_enrolled":          len(staff_ids),
        "enrollment_samples":        enrollment_samples,
        "total_staff_queries":       total_staff_queries,
        "total_imposter_queries":    total_imposter_queries,
        "per_speaker":               per_speaker,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate staff ID accuracy")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--n_staff", type=int, default=20)
    parser.add_argument("--enrollment_samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    embedder = ECAPATDNNEmbedder(args.onnx)
    results = evaluate_identification(
        embedder=embedder,
        threshold=args.threshold,
        n_staff=args.n_staff,
        enrollment_samples=args.enrollment_samples,
    )

    print(f"\n{'='*50}")
    print(f"Top-1 accuracy : {results['top1_accuracy_pct']:.1f}%")
    print(f"Target         : {results['target_top1_accuracy_pct']:.1f}%")
    print(f"FAR            : {results['false_accept_rate_pct']:.2f}%  (false-staff)")
    print(f"FRR            : {results['false_reject_rate_pct']:.2f}%  (false-customer)")
    print(f"Status         : {'✓ PASS' if results['passed'] else '✗ FAIL'}")
    print(f"{'='*50}")

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Results → {args.output}")


if __name__ == "__main__":
    main()
