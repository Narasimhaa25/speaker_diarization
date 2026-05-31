"""
utils/crypto_utils.py
──────────────────────
AES-256-GCM encrypt/decrypt for the staff voice signature database.

Security properties:
  - AES-256-GCM: authenticated encryption (integrity + confidentiality)
  - Random 96-bit nonce per encryption operation
  - 128-bit authentication tag (GCM default)
  - No key derivation — caller must supply a 32-byte key from a secure store

Wire format (all bytes):
  [4 bytes: magic] [12 bytes: nonce] [16 bytes: tag] [N bytes: ciphertext]
"""

from __future__ import annotations

import os
import struct

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# File magic for integrity check — "STAFFDB\x01"
MAGIC = b"STAFFDB\x01"
NONCE_SIZE = 12     # bytes (96 bits — GCM standard)
TAG_SIZE = 16       # bytes (128 bits — GCM default)
HEADER_SIZE = len(MAGIC) + NONCE_SIZE  # magic + nonce (tag is appended by AESGCM)


def _check_deps() -> None:
    if not CRYPTO_AVAILABLE:
        raise ImportError(
            "cryptography package required: pip install cryptography"
        )


def generate_key() -> bytes:
    """Generate a random 32-byte AES-256 key. Store securely — never hard-code."""
    return os.urandom(32)


def encrypt_bytes(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt plaintext with AES-256-GCM.

    Parameters
    ----------
    plaintext : bytes  Data to encrypt (e.g. JSON-serialised staff DB)
    key : bytes        32-byte AES-256 key

    Returns
    -------
    bytes — [magic (8)] + [nonce (12)] + [ciphertext+tag (N+16)]
    """
    _check_deps()
    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes for AES-256, got {len(key)}")

    nonce = os.urandom(NONCE_SIZE)
    aesgcm = AESGCM(key)
    # AESGCM.encrypt appends the 16-byte GCM tag to the ciphertext
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data=MAGIC)

    return MAGIC + nonce + ciphertext_with_tag


def decrypt_bytes(encrypted: bytes, key: bytes) -> bytes:
    """
    Decrypt AES-256-GCM ciphertext.

    Parameters
    ----------
    encrypted : bytes  Output of encrypt_bytes()
    key : bytes        Same 32-byte key used for encryption

    Returns
    -------
    bytes — original plaintext

    Raises
    ------
    ValueError  : if magic header is wrong (wrong file or corruption)
    cryptography.exceptions.InvalidTag : if authentication fails (tampered data)
    """
    _check_deps()
    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes for AES-256, got {len(key)}")

    # Validate magic
    if len(encrypted) < HEADER_SIZE + TAG_SIZE:
        raise ValueError("Encrypted data too short — corrupted or wrong format")

    magic = encrypted[:len(MAGIC)]
    if magic != MAGIC:
        raise ValueError(
            f"Invalid file magic: expected {MAGIC!r}, got {magic!r}. "
            "Wrong file or key?"
        )

    nonce = encrypted[len(MAGIC): len(MAGIC) + NONCE_SIZE]
    ciphertext_with_tag = encrypted[HEADER_SIZE:]

    aesgcm = AESGCM(key)
    # InvalidTag is raised by cryptography if the data was tampered with
    plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data=MAGIC)
    return plaintext


def rotate_key(encrypted: bytes, old_key: bytes, new_key: bytes) -> bytes:
    """
    Re-encrypt data with a new key (key rotation for staff DB).

    Decrypt with old_key, re-encrypt with new_key atomically.
    Useful when a store admin changes the encryption key.
    """
    plaintext = decrypt_bytes(encrypted, old_key)
    return encrypt_bytes(plaintext, new_key)
