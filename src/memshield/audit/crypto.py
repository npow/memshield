"""AES-256-GCM encryption and per-user key store for MemShield audit PII."""
from __future__ import annotations

import base64
import os
import sqlite3
from datetime import datetime, timezone

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class KeyStore:
    """Per-user AES-256 key store backed by a SQLite database.

    Each user gets a unique 32-byte random key which is created on first access
    and persisted.  Deleting a key (crypto-shredding) renders all ciphertext
    encrypted under that key permanently unreadable.

    Args:
        key_store_path: Filesystem path for the SQLite key store database.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS user_keys (
        user_id  TEXT PRIMARY KEY,
        key_bytes BLOB NOT NULL,
        created_at TEXT NOT NULL
    );
    """

    def __init__(self, key_store_path: str) -> None:
        """Open (or create) the key store at *key_store_path*."""
        self._path = key_store_path
        conn = self._connect()
        try:
            conn.executescript(self._SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_or_create_key(self, user_id: str) -> bytes:
        """Return the 32-byte AES-256 key for *user_id*, creating it if needed.

        Args:
            user_id: Opaque user identifier.

        Returns:
            32-byte AES key.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT key_bytes FROM user_keys WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row is not None:
                return bytes(row["key_bytes"])

            key = os.urandom(32)
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO user_keys (user_id, key_bytes, created_at) VALUES (?, ?, ?)",
                (user_id, key, now),
            )
            conn.commit()
            return key
        finally:
            conn.close()

    def delete_key(self, user_id: str) -> None:
        """Delete the AES key for *user_id* (crypto-shredding).

        After this call, any ciphertext encrypted under that user's key is
        permanently unreadable.  Audit records themselves are **not** modified.

        Args:
            user_id: Opaque user identifier whose key should be destroyed.
        """
        conn = self._connect()
        try:
            conn.execute("DELETE FROM user_keys WHERE user_id = ?", (user_id,))
            conn.commit()
        finally:
            conn.close()


class AESCipher:
    """Stateless AES-256-GCM encrypt/decrypt helpers.

    The wire format is ``base64(nonce + ciphertext + tag)`` where the nonce is
    12 bytes and the GCM authentication tag is 16 bytes.  A fresh random nonce
    is generated for every :meth:`encrypt` call.
    """

    _NONCE_SIZE = 12

    @staticmethod
    def encrypt(plaintext: str, key: bytes) -> str:
        """Encrypt *plaintext* with AES-256-GCM under *key*.

        Args:
            plaintext: UTF-8 string to encrypt.
            key: 32-byte AES key.

        Returns:
            Base64-encoded string of ``nonce + ciphertext + tag``.
        """
        aesgcm = AESGCM(key)
        nonce = os.urandom(AESCipher._NONCE_SIZE)
        # AESGCM.encrypt returns ciphertext || tag
        ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        blob = nonce + ciphertext_with_tag
        return base64.b64encode(blob).decode("ascii")

    @staticmethod
    def decrypt(ciphertext: str, key: bytes) -> str:
        """Decrypt a ciphertext produced by :meth:`encrypt`.

        Args:
            ciphertext: Base64-encoded ``nonce + ciphertext + tag`` string.
            key: 32-byte AES key.

        Returns:
            Decrypted UTF-8 plaintext.

        Raises:
            ValueError: If the ciphertext is malformed or authentication fails.
        """
        try:
            blob = base64.b64decode(ciphertext)
            nonce = blob[: AESCipher._NONCE_SIZE]
            payload = blob[AESCipher._NONCE_SIZE :]
            aesgcm = AESGCM(key)
            plaintext_bytes = aesgcm.decrypt(nonce, payload, None)
            return plaintext_bytes.decode("utf-8")
        except Exception as exc:
            raise ValueError(f"AES-GCM decryption failed: {exc}") from exc
