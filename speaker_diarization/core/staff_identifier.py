"""
core/staff_identifier.py
─────────────────────────
Staff / customer classifier using cosine similarity against the staff DB.

KEY DECISION (owned by this module):
  The similarity threshold balances:
    false-staff (privacy leak) — a customer mistakenly labelled as known staff
    false-customer (analytics noise) — staff mistakenly labelled as customer

  Target: > 95% top-1 accuracy on staff identification.
  Default threshold: 0.82 (tune with evaluation/threshold_tuner.py).

  Guidance:
    threshold > 0.85 → very conservative, low false-staff, higher false-customer
    threshold ~ 0.82 → balanced (recommended starting point)
    threshold < 0.75 → aggressive, more false-staff (privacy risk)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from staff_db.db_manager import StaffDBManager
from staff_db.similarity_search import SimilaritySearch


# Default threshold — tune via evaluation/threshold_tuner.py
DEFAULT_SIMILARITY_THRESHOLD = 0.82


class StaffIdentifier:
    """
    Identifies whether a speaker is staff or customer.

    For every embedding, computes cosine similarity against all enrolled
    staff embeddings. Above threshold → staff (with name). Below → customer.

    Parameters
    ----------
    staff_db : StaffDBManager
        The encrypted on-device staff database for the current store.
    threshold : float
        Cosine similarity threshold for staff classification.
        Must be tuned per deployment — see KEY DECISIONS in README.
    """

    def __init__(
        self,
        staff_db: Optional[StaffDBManager],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.staff_db = staff_db
        self.threshold = threshold
        self._search: Optional[SimilaritySearch] = None

        if staff_db is not None:
            self._search = SimilaritySearch(staff_db)

    def identify(
        self, embedding: np.ndarray
    ) -> Tuple[str, Optional[str], float]:
        """
        Classify a single embedding.

        Parameters
        ----------
        embedding : np.ndarray (192,) L2-normalised float32

        Returns
        -------
        role : str          "staff" | "customer" | "unknown"
        staff_name : str | None
        score : float       Best cosine similarity score (0.0 if no DB)
        """
        if self._search is None or self.staff_db is None:
            return "unknown", None, 0.0

        best_name, best_score = self._search.top1(embedding)

        if best_score >= self.threshold:
            return "staff", best_name, best_score
        else:
            return "customer", None, best_score

    def set_threshold(self, threshold: float) -> None:
        """Update threshold at runtime — used by threshold_tuner.py."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def batch_identify(
        self, embeddings: np.ndarray
    ) -> list[Tuple[str, Optional[str], float]]:
        """
        Classify a batch of embeddings.

        Parameters
        ----------
        embeddings : np.ndarray (B, 192)

        Returns
        -------
        List of (role, staff_name, score) tuples.
        """
        return [self.identify(e) for e in embeddings]

    # ── Threshold guidance ────────────────────────────────────────────────────

    @staticmethod
    def threshold_guidance() -> dict:
        """
        Human-readable threshold guidance for the store admin.

        Run evaluation/threshold_tuner.py to find the optimal value
        for a specific store's enrollment data.
        """
        return {
            "recommended": DEFAULT_SIMILARITY_THRESHOLD,
            "conservative": 0.87,
            "aggressive": 0.75,
            "tradeoffs": {
                "high_threshold": "Fewer false-staff (better privacy), more false-customer (noisier analytics)",
                "low_threshold": "More false-staff (privacy risk), fewer false-customer",
            },
            "tuning_script": "evaluation/threshold_tuner.py",
            "target_accuracy": ">95% top-1 on enrolled staff",
        }
