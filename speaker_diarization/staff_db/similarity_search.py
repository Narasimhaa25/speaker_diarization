"""
staff_db/similarity_search.py
──────────────────────────────
Cosine similarity search over the staff embedding database.

Store sizes of 5–25 staff make exhaustive cosine search trivially fast.
No approximate nearest-neighbour index needed.

Returns top-1 match and score for the staff/customer threshold decision.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from staff_db.db_manager import StaffDBManager


class SimilaritySearch:
    """
    Fast exhaustive cosine similarity search over enrolled staff embeddings.

    For 5–25 staff the matrix multiply is microseconds — no ANN needed.

    Parameters
    ----------
    db : StaffDBManager
        The loaded staff DB for the current store.
    """

    def __init__(self, db: StaffDBManager):
        self._db = db
        self._cache: Optional[tuple] = None  # (ids, names, matrix)

    def _get_matrix(self) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Build [N, 192] embedding matrix from active staff.
        Cached until invalidated (call invalidate_cache() after DB updates).
        """
        if self._cache is not None:
            return self._cache

        records = self._db.get_active_embeddings()  # [(id, name, embedding)]
        if not records:
            self._cache = ([], [], np.empty((0, 192), dtype=np.float32))
            return self._cache

        ids, names, embeddings = zip(*records)
        matrix = np.stack(embeddings, axis=0)  # (N, 192)
        self._cache = (list(ids), list(names), matrix)
        return self._cache

    def invalidate_cache(self) -> None:
        """Call after any add/update/remove to reload embeddings."""
        self._cache = None

    def top1(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the best-matching staff member for a query embedding.

        Parameters
        ----------
        query : np.ndarray (192,) L2-normalised float32

        Returns
        -------
        (name, score) — name is None and score is 0.0 if no staff enrolled.
        """
        ids, names, matrix = self._get_matrix()
        if len(ids) == 0:
            return None, 0.0

        # All embeddings are L2-normalised → dot product = cosine similarity
        scores = matrix @ query                  # (N,)
        best_idx = int(np.argmax(scores))
        return names[best_idx], float(scores[best_idx])

    def top_k(
        self, query: np.ndarray, k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Return top-k matches.

        Returns
        -------
        List of (staff_id, name, score) sorted by descending score.
        """
        ids, names, matrix = self._get_matrix()
        if len(ids) == 0:
            return []

        scores = matrix @ query
        k = min(k, len(ids))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(ids[i], names[i], float(scores[i])) for i in top_idx]

    def all_scores(self, query: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Return similarity scores for ALL active staff — used by threshold_tuner.py.

        Returns
        -------
        List of (staff_id, name, score) sorted by descending score.
        """
        return self.top_k(query, k=len(self._get_matrix()[0]))
