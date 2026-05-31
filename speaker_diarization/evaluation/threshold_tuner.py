"""
evaluation/threshold_tuner.py
──────────────────────────────
Sweep the staff/customer similarity threshold and find the optimal value.

Plots ROC curve (FAR vs FRR) and identifies:
  - EER (Equal Error Rate) threshold
  - The threshold that achieves > 95% top-1 at minimum FAR
  - Recommended threshold for deployment

Uses VoxCeleb as the evaluation corpus (same as evaluate_identification.py).

Usage
-----
    python evaluation/threshold_tuner.py \\
        --voxceleb_dir ./datasets/voxceleb \\
        --onnx models/ecapa_tdnn_int8.onnx \\
        --plot \\
        --output results/threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from data.dataset_loader import SLUEVoxCelebLoader
from models.ecapa_tdnn import ECAPATDNNEmbedder


def compute_scores(
    embedder: ECAPATDNNEmbedder,
    n_staff: int = 25,
    enrollment_samples: int = 5,
    query_samples: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute genuine and imposter similarity score distributions.

    Returns
    -------
    genuine_scores  : (N,) float32 — staff vs their own queries
    imposter_scores : (M,) float32 — imposter vs enrolled staff
    """
    n_imposters = 100
    total_speakers = n_staff + n_imposters
    utterances_per_speaker = enrollment_samples + query_samples

    loader = SLUEVoxCelebLoader(seed=seed)
    all_segments = loader.sample_by_speaker(
        n_speakers=total_speakers,
        utterances_per_speaker=utterances_per_speaker,
        split="train",
    )

    # Partition into staff and imposter groups
    all_speaker_ids = sorted({s.speaker_id for s in all_segments})
    staff_speakers   = set(all_speaker_ids[:n_staff])
    imposter_speakers = set(all_speaker_ids[n_staff: n_staff + n_imposters])

    # Build per-speaker segment lists
    by_speaker: Dict[str, List] = {}
    for seg in all_segments:
        by_speaker.setdefault(seg.speaker_id, []).append(seg)

    # Enroll staff (first enrollment_samples utterances per staff speaker)
    staff_embs: Dict[str, np.ndarray] = {}
    for spk in staff_speakers:
        segs = by_speaker.get(spk, [])[:enrollment_samples]
        if not segs:
            continue
        staff_embs[spk] = embedder.mean_embedding([s.audio for s in segs])

    if not staff_embs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    staff_matrix = np.stack(list(staff_embs.values()))  # (N_staff, 192)
    staff_ids    = list(staff_embs.keys())

    genuine_scores:  List[float] = []
    imposter_scores: List[float] = []

    # Genuine scores: remaining utterances of each staff speaker
    for spk in staff_ids:
        idx = staff_ids.index(spk)
        for seg in by_speaker.get(spk, [])[enrollment_samples:]:
            try:
                emb = embedder.embed(seg.audio)
                score = float(np.dot(staff_matrix[idx], emb))
                genuine_scores.append(score)
            except ValueError:
                pass

    # Imposter scores: max similarity of each imposter against staff DB
    for spk in imposter_speakers:
        for seg in by_speaker.get(spk, [])[:query_samples]:
            try:
                emb = embedder.embed(seg.audio)
                max_score = float((staff_matrix @ emb).max())
                imposter_scores.append(max_score)
            except ValueError:
                pass

    return np.array(genuine_scores, dtype=np.float32), np.array(imposter_scores, dtype=np.float32)


def sweep_threshold(
    genuine_scores: np.ndarray,
    imposter_scores: np.ndarray,
    n_steps: int = 200,
) -> List[Dict]:
    """
    Sweep threshold from 0 to 1 and compute FAR, FRR, top-1 accuracy.

    Returns list of dicts with threshold, far, frr, top1_accuracy.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    results = []
    for t in thresholds:
        far = float((imposter_scores >= t).mean() * 100)  # % imposters accepted
        frr = float((genuine_scores < t).mean() * 100)    # % genuine rejected
        top1 = 100.0 - frr
        results.append({
            "threshold":       round(float(t), 4),
            "far_pct":         round(far, 3),
            "frr_pct":         round(frr, 3),
            "top1_accuracy":   round(top1, 3),
        })
    return results


def find_optimal_threshold(sweep: List[Dict]) -> Dict:
    """
    Find recommended thresholds:
      - EER: where FAR ≈ FRR
      - target_95: lowest threshold where top-1 ≥ 95%
    """
    # EER
    eer_thresh = None
    eer_val = None
    min_diff = float("inf")
    for r in sweep:
        diff = abs(r["far_pct"] - r["frr_pct"])
        if diff < min_diff:
            min_diff = diff
            eer_thresh = r["threshold"]
            eer_val = (r["far_pct"] + r["frr_pct"]) / 2

    # Threshold where top-1 ≥ 95% and FAR is minimised
    candidates = [r for r in sweep if r["top1_accuracy"] >= 95.0]
    target_95_thresh = None
    if candidates:
        # Among those meeting 95% accuracy, pick the one with lowest FAR
        best = min(candidates, key=lambda r: r["far_pct"])
        target_95_thresh = best["threshold"]

    return {
        "eer_threshold":      eer_thresh,
        "eer_pct":            round(eer_val, 3) if eer_val else None,
        "target_95_threshold": target_95_thresh,
        "recommendation":     target_95_thresh or eer_thresh,
        "note": (
            "Deploy target_95_threshold for > 95% top-1 accuracy. "
            "If no threshold achieves 95%, increase enrollment samples or "
            "improve audio quality."
        ),
    }


def plot_roc(sweep: List[Dict], optimal: Dict, output_path: Path) -> None:
    if not MPL_AVAILABLE:
        print("matplotlib not available — skipping plot")
        return

    far_vals = [r["far_pct"] for r in sweep]
    frr_vals = [r["frr_pct"] for r in sweep]
    thresholds = [r["threshold"] for r in sweep]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # DET curve (FAR vs FRR)
    axes[0].plot(far_vals, frr_vals, "b-", linewidth=2, label="DET curve")
    if optimal["eer_threshold"] is not None:
        axes[0].scatter(
            [optimal["eer_pct"]], [optimal["eer_pct"]],
            color="red", zorder=5, label=f"EER={optimal['eer_pct']:.1f}%", s=80
        )
    axes[0].set_xlabel("False Accept Rate (%) — false-staff")
    axes[0].set_ylabel("False Reject Rate (%) — false-customer")
    axes[0].set_title("DET Curve — Staff/Customer Threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # FAR/FRR vs threshold
    axes[1].plot(thresholds, far_vals, "r-", label="FAR (false-staff %)")
    axes[1].plot(thresholds, frr_vals, "b-", label="FRR (false-customer %)")
    if optimal["recommendation"]:
        axes[1].axvline(
            optimal["recommendation"], color="green", linestyle="--",
            label=f"Recommended={optimal['recommendation']:.3f}"
        )
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("%")
    axes[1].set_title("FAR / FRR vs Similarity Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved → {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune staff/customer similarity threshold")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--n_staff", type=int, default=25)
    parser.add_argument("--enrollment_samples", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/threshold_sweep.json"))
    args = parser.parse_args()

    embedder = ECAPATDNNEmbedder(args.onnx)

    print("Computing genuine / imposter score distributions …")
    genuine, imposter = compute_scores(
        embedder=embedder,
        n_staff=args.n_staff,
        enrollment_samples=args.enrollment_samples,
    )
    print(f"  Genuine scores:  {len(genuine)} samples, mean={genuine.mean():.3f}")
    print(f"  Imposter scores: {len(imposter)} samples, mean={imposter.mean():.3f}")

    sweep = sweep_threshold(genuine, imposter)
    optimal = find_optimal_threshold(sweep)

    print(f"\n{'='*50}")
    print(f"EER threshold      : {optimal['eer_threshold']:.4f} (EER={optimal['eer_pct']:.2f}%)")
    print(f"95% top-1 threshold: {optimal['target_95_threshold']}")
    print(f"Recommended        : {optimal['recommendation']}")
    print(f"{'='*50}")

    results = {"sweep": sweep, "optimal": optimal}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Results → {args.output}")

    if args.plot:
        plot_path = args.output.with_suffix(".png")
        plot_roc(sweep, optimal, plot_path)


if __name__ == "__main__":
    main()
