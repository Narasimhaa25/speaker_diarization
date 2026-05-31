"""
staff_db/db_manager.py
───────────────────────
Add / remove / update staff voice signatures.

Handles the full lifecycle:
  - Create a new store DB
  - Enroll a new staff member (add)
  - Update an existing staff member's embedding (re-enroll)
  - Soft-delete / hard-delete when staff leave
  - Load / save with AES-256-GCM encryption

Staff turnover flow
───────────────────
  New hire   → db.add_staff(name, role, embedding, n_samples)
  Re-enroll  → db.update_staff(staff_id, new_embedding, n_samples)
  Staff left → db.deactivate_staff(staff_id)   # soft-delete (recommended)
               db.remove_staff(staff_id)        # hard-delete (irreversible)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from staff_db.schema import StaffDatabase, StaffRecord, EMBEDDING_DIM
from utils.crypto_utils import encrypt_bytes, decrypt_bytes


class StaffDBManager:
    """
    Manages the encrypted on-device staff voice DB for one store.

    Parameters
    ----------
    db_path : Path
        Path to the encrypted .staffdb file (or where to create it).
    key : bytes
        32-byte AES-256 key. Never hard-code — load from a secure key store.
    """

    FILE_EXTENSION = ".staffdb"

    def __init__(self, db_path: Path, key: bytes):
        if len(key) != 32:
            raise ValueError("AES-256-GCM key must be 32 bytes.")
        self._db_path = Path(db_path)
        self._key = key
        self._db: Optional[StaffDatabase] = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def create(self, store_id: str) -> None:
        """Create a new empty DB for a store. Fails if file already exists."""
        if self._db_path.exists():
            raise FileExistsError(
                f"Staff DB already exists at {self._db_path}. "
                "Use load() to open an existing DB."
            )
        self._db = StaffDatabase(store_id=store_id)
        self.save()
        print(f"Created staff DB for store '{store_id}' at {self._db_path}")

    def load(self) -> None:
        """Load and decrypt the staff DB from disk."""
        if not self._db_path.exists():
            raise FileNotFoundError(f"Staff DB not found: {self._db_path}")
        encrypted = self._db_path.read_bytes()
        plaintext = decrypt_bytes(encrypted, self._key)
        data = json.loads(plaintext.decode())
        self._db = StaffDatabase.from_dict(data)
        print(f"Loaded staff DB: {self._db.store_id} "
              f"({len(self._db.active_staff())} active staff)")

    def save(self) -> None:
        """Encrypt and write DB to disk."""
        if self._db is None:
            raise RuntimeError("No DB loaded. Call create() or load() first.")
        self._db.touch()
        plaintext = json.dumps(self._db.to_dict()).encode()
        encrypted = encrypt_bytes(plaintext, self._key)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path.write_bytes(encrypted)

    def _require_db(self) -> StaffDatabase:
        if self._db is None:
            raise RuntimeError("No DB loaded. Call create() or load() first.")
        return self._db

    # ── Staff CRUD ────────────────────────────────────────────────────────────

    def add_staff(
        self,
        name: str,
        role: str,
        embedding: np.ndarray,
        n_samples: int,
    ) -> str:
        """
        Enroll a new staff member.

        Parameters
        ----------
        name : str          Full name (displayed in analytics).
        role : str          e.g. "associate", "manager".
        embedding : ndarray (192,) L2-normalised float32.
        n_samples : int     Number of utterances averaged to produce embedding.

        Returns
        -------
        str — the new staff_id (UUID).
        """
        db = self._require_db()

        if len(db.active_staff()) >= 25:
            raise ValueError(
                "Maximum 25 active staff per store. "
                "Deactivate departed staff before adding new ones."
            )

        record = StaffRecord.create(
            name=name,
            role=role,
            embedding=embedding.tolist(),
            n_samples=n_samples,
        )
        db.staff[record.staff_id] = record
        self.save()
        print(f"Enrolled: {name} ({role}) → {record.staff_id}")
        return record.staff_id

    def update_staff(
        self,
        staff_id: str,
        new_embedding: np.ndarray,
        n_samples: int,
    ) -> None:
        """
        Re-enroll an existing staff member (voice changes over time).

        Replaces the stored embedding — previous embedding is not retained.
        Re-enrollment is recommended every 6–12 months (see KEY DECISIONS).
        """
        db = self._require_db()
        if staff_id not in db.staff:
            raise KeyError(f"Staff ID not found: {staff_id}")

        from datetime import datetime, timezone
        record = db.staff[staff_id]
        record.embedding = new_embedding.tolist()
        record.n_samples = n_samples
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self.save()
        print(f"Re-enrolled: {record.name} ({staff_id})")

    def deactivate_staff(self, staff_id: str) -> None:
        """
        Soft-delete a staff member (mark inactive, keep record).

        Inactive staff are excluded from similarity search.
        Recommended over hard-delete — preserves audit trail.
        """
        db = self._require_db()
        if staff_id not in db.staff:
            raise KeyError(f"Staff ID not found: {staff_id}")
        db.staff[staff_id].active = False
        self.save()
        print(f"Deactivated: {db.staff[staff_id].name} ({staff_id})")

    def remove_staff(self, staff_id: str) -> None:
        """
        Hard-delete a staff member. IRREVERSIBLE.

        Prefer deactivate_staff() unless privacy deletion is legally required.
        """
        db = self._require_db()
        if staff_id not in db.staff:
            raise KeyError(f"Staff ID not found: {staff_id}")
        name = db.staff[staff_id].name
        del db.staff[staff_id]
        self.save()
        print(f"Permanently removed: {name} ({staff_id})")

    # ── Query helpers ─────────────────────────────────────────────────────────

    def list_staff(self, include_inactive: bool = False) -> List[dict]:
        """Return a summary list (no embeddings) for admin UIs."""
        db = self._require_db()
        records = db.staff.values() if include_inactive else db.active_staff()
        return [
            {
                "staff_id": r.staff_id,
                "name": r.name,
                "role": r.role,
                "enrolled_at": r.enrolled_at,
                "updated_at": r.updated_at,
                "n_samples": r.n_samples,
                "active": r.active,
            }
            for r in records
        ]

    def get_active_embeddings(self) -> List[Tuple[str, str, np.ndarray]]:
        """
        Return [(staff_id, name, embedding)] for all active staff.
        Used by SimilaritySearch.
        """
        db = self._require_db()
        results = []
        for record in db.active_staff():
            emb = np.array(record.embedding, dtype=np.float32)
            results.append((record.staff_id, record.name, emb))
        return results

    def staff_count(self) -> Tuple[int, int]:
        """Return (active_count, total_count)."""
        db = self._require_db()
        total = len(db.staff)
        active = len(db.active_staff())
        return active, total
