"""
staff_db/schema.py
───────────────────
Encrypted on-device staff voice signature database schema.

Store size: 5–25 staff members per store (closed-set, on-device).
Encryption: AES-256-GCM via utils/crypto_utils.py.

Schema is stored as a JSON structure, encrypted at rest.
In-memory it is a plain Python dict (decrypted on load, re-encrypted on save).

Database layout
───────────────
{
    "store_id": "STORE_001",
    "version": 1,
    "created_at": "2024-01-15T09:00:00Z",
    "updated_at": "2024-01-15T09:00:00Z",
    "staff": {
        "<staff_id>": {
            "name": "Alice Smith",
            "role": "associate",
            "embedding": [0.12, -0.34, ...],   # 192-dim float list, L2-normalised
            "enrolled_at": "2024-01-15T09:00:00Z",
            "updated_at": "2024-01-15T09:00:00Z",
            "n_samples": 5,                     # number of enrollment utterances averaged
            "active": true
        },
        ...
    }
}
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional
import uuid

EMBEDDING_DIM = 192   # must match ECAPA-TDNN contract


@dataclass
class StaffRecord:
    """A single staff member's voice record."""
    staff_id: str
    name: str
    role: str                       # e.g. "associate", "manager", "supervisor"
    embedding: List[float]          # 192-dim L2-normalised
    enrolled_at: str                # ISO 8601
    updated_at: str                 # ISO 8601
    n_samples: int                  # number of utterances averaged into embedding
    active: bool = True             # False = soft-deleted (not searched)

    @classmethod
    def create(
        cls,
        name: str,
        role: str,
        embedding: List[float],
        n_samples: int,
    ) -> "StaffRecord":
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must be {EMBEDDING_DIM}-dim, got {len(embedding)}"
            )
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            staff_id=str(uuid.uuid4()),
            name=name,
            role=role,
            embedding=embedding,
            enrolled_at=now,
            updated_at=now,
            n_samples=n_samples,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StaffRecord":
        return cls(**d)


@dataclass
class StaffDatabase:
    """In-memory representation of the staff DB for one store."""
    store_id: str
    version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    staff: dict[str, StaffRecord] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def active_staff(self) -> list[StaffRecord]:
        return [s for s in self.staff.values() if s.active]

    def to_dict(self) -> dict:
        return {
            "store_id": self.store_id,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "staff": {sid: rec.to_dict() for sid, rec in self.staff.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StaffDatabase":
        db = cls(
            store_id=d["store_id"],
            version=d.get("version", 1),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
        for sid, rec_dict in d.get("staff", {}).items():
            db.staff[sid] = StaffRecord.from_dict(rec_dict)
        return db
