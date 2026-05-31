"""
tests/test_staff_db.py
───────────────────────
Unit tests for the staff voice signature database.
Run with: pytest tests/test_staff_db.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from staff_db.schema import StaffRecord, StaffDatabase, EMBEDDING_DIM
from staff_db.db_manager import StaffDBManager
from staff_db.similarity_search import SimilaritySearch
from utils.crypto_utils import generate_key


# ─── Helpers ─────────────────────────────────────────────────────────────────

def random_embedding() -> np.ndarray:
    """Random L2-normalised 192-dim embedding."""
    v = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_db(tmp_path: Path, n_staff: int = 3) -> tuple:
    """Create a temp DB with n_staff enrolled members."""
    key = generate_key()
    db_path = tmp_path / "test.staffdb"
    manager = StaffDBManager(db_path, key)
    manager.create("TEST_STORE")

    staff_ids = []
    embeddings = []
    for i in range(n_staff):
        emb = random_embedding()
        embeddings.append(emb)
        sid = manager.add_staff(f"Staff {i}", "associate", emb, n_samples=5)
        staff_ids.append(sid)

    return manager, staff_ids, embeddings


# ─── Schema tests ─────────────────────────────────────────────────────────────

class TestStaffRecord:
    def test_create_valid(self):
        emb = random_embedding()
        record = StaffRecord.create("Alice", "manager", emb.tolist(), n_samples=5)
        assert record.name == "Alice"
        assert record.role == "manager"
        assert len(record.embedding) == EMBEDDING_DIM
        assert record.active is True

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError, match="Embedding must be"):
            StaffRecord.create("Bob", "associate", [0.1, 0.2], n_samples=1)

    def test_roundtrip(self):
        emb = random_embedding()
        record = StaffRecord.create("Charlie", "supervisor", emb.tolist(), n_samples=3)
        d = record.to_dict()
        record2 = StaffRecord.from_dict(d)
        assert record2.name == record.name
        assert record2.staff_id == record.staff_id
        np.testing.assert_allclose(
            np.array(record2.embedding), np.array(record.embedding), atol=1e-6
        )


# ─── DB manager tests ─────────────────────────────────────────────────────────

class TestStaffDBManager:
    def test_create_and_load(self, tmp_path):
        key = generate_key()
        db_path = tmp_path / "store.staffdb"
        manager = StaffDBManager(db_path, key)
        manager.create("STORE_001")

        # Reload and verify
        manager2 = StaffDBManager(db_path, key)
        manager2.load()
        assert manager2._db.store_id == "STORE_001"

    def test_add_and_list(self, tmp_path):
        manager, staff_ids, _ = make_db(tmp_path, n_staff=3)
        listing = manager.list_staff()
        assert len(listing) == 3
        names = {s["name"] for s in listing}
        assert names == {"Staff 0", "Staff 1", "Staff 2"}

    def test_deactivate(self, tmp_path):
        manager, staff_ids, _ = make_db(tmp_path, n_staff=2)
        manager.deactivate_staff(staff_ids[0])
        listing = manager.list_staff(include_inactive=False)
        assert len(listing) == 1
        assert listing[0]["staff_id"] == staff_ids[1]

    def test_remove(self, tmp_path):
        manager, staff_ids, _ = make_db(tmp_path, n_staff=2)
        manager.remove_staff(staff_ids[0])
        listing = manager.list_staff(include_inactive=True)
        assert len(listing) == 1

    def test_update(self, tmp_path):
        manager, staff_ids, _ = make_db(tmp_path, n_staff=1)
        new_emb = random_embedding()
        manager.update_staff(staff_ids[0], new_emb, n_samples=7)
        embs = manager.get_active_embeddings()
        stored_emb = embs[0][2]
        np.testing.assert_allclose(stored_emb, new_emb, atol=1e-6)

    def test_max_25_staff(self, tmp_path):
        key = generate_key()
        db_path = tmp_path / "full.staffdb"
        manager = StaffDBManager(db_path, key)
        manager.create("BIG_STORE")

        for i in range(25):
            manager.add_staff(f"Staff {i}", "associate", random_embedding(), n_samples=3)

        with pytest.raises(ValueError, match="Maximum 25"):
            manager.add_staff("Staff 26", "associate", random_embedding(), n_samples=3)

    def test_persistence_through_reload(self, tmp_path):
        """Data must survive encrypt → save → load → decrypt."""
        key = generate_key()
        db_path = tmp_path / "persist.staffdb"
        manager = StaffDBManager(db_path, key)
        manager.create("PERSIST_TEST")
        emb = random_embedding()
        sid = manager.add_staff("Persist Alice", "manager", emb, n_samples=5)

        # Reload from disk
        manager2 = StaffDBManager(db_path, key)
        manager2.load()
        listing = manager2.list_staff()
        assert len(listing) == 1
        assert listing[0]["name"] == "Persist Alice"


# ─── Similarity search tests ──────────────────────────────────────────────────

class TestSimilaritySearch:
    def test_top1_returns_correct_speaker(self, tmp_path):
        """A query identical to an enrollment should score highest."""
        manager, staff_ids, embeddings = make_db(tmp_path, n_staff=3)
        search = SimilaritySearch(manager)

        # Query with the exact embedding of staff index 1
        name, score = search.top1(embeddings[1])
        assert score > 0.99   # near-perfect cosine similarity with itself

    def test_top_k(self, tmp_path):
        manager, _, embeddings = make_db(tmp_path, n_staff=5)
        search = SimilaritySearch(manager)
        results = search.top_k(embeddings[0], k=3)
        assert len(results) == 3
        # Results should be sorted by descending score
        scores = [r[2] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_db_returns_none(self, tmp_path):
        key = generate_key()
        db_path = tmp_path / "empty.staffdb"
        manager = StaffDBManager(db_path, key)
        manager.create("EMPTY_STORE")
        search = SimilaritySearch(manager)
        name, score = search.top1(random_embedding())
        assert name is None
        assert score == 0.0

    def test_cache_invalidation(self, tmp_path):
        """Adding new staff must be reflected after cache invalidation."""
        manager, _, _ = make_db(tmp_path, n_staff=1)
        search = SimilaritySearch(manager)

        # Prime cache
        _ = search.top1(random_embedding())

        # Add new staff
        new_emb = random_embedding()
        manager.add_staff("New Person", "associate", new_emb, n_samples=3)

        # Without invalidation, cache returns old results
        search.invalidate_cache()
        name, score = search.top1(new_emb)
        assert score > 0.99
