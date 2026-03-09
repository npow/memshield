"""Ed25519 signing key management for the MemShield audit log."""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
)

logger = logging.getLogger(__name__)


class SigningKey:
    """Ed25519 signing key that persists to a PEM file.

    If ``key_file`` does not exist, a new key pair is generated and saved.
    If it does exist, the existing key is loaded from disk.

    Args:
        key_file: Path to the PEM file containing the Ed25519 private key.
    """

    def __init__(self, key_file: str) -> None:
        """Load an existing PEM key or generate and persist a new one.

        Args:
            key_file: Filesystem path for the private key PEM file.
        """
        self._key_file = key_file
        path = Path(key_file)
        if path.exists():
            logger.debug("Loading Ed25519 signing key from %s", key_file)
            pem_bytes = path.read_bytes()
            self._private_key: Ed25519PrivateKey = load_pem_private_key(  # type: ignore[assignment]
                pem_bytes, password=None
            )
        else:
            logger.info("Generating new Ed25519 signing key at %s", key_file)
            self._private_key = Ed25519PrivateKey.generate()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                self._private_key.private_bytes(
                    encoding=Encoding.PEM,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption(),
                )
            )

        self._public_key: Ed25519PublicKey = self._private_key.public_key()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def key_id(self) -> str:
        """SHA-256 fingerprint of the public key bytes, hex-encoded."""
        pub_bytes = self._public_key.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        )
        return hashlib.sha256(pub_bytes).hexdigest()

    def sign(self, data: bytes) -> str:
        """Sign *data* with the Ed25519 private key.

        Args:
            data: Raw bytes to sign.

        Returns:
            Base64-encoded signature string.
        """
        raw_sig = self._private_key.sign(data)
        return base64.b64encode(raw_sig).decode("ascii")

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify a base64-encoded Ed25519 signature against *data*.

        Args:
            data: The original bytes that were signed.
            signature: Base64-encoded signature to verify.

        Returns:
            ``True`` if the signature is valid, ``False`` otherwise.
        """
        try:
            raw_sig = base64.b64decode(signature)
            self._public_key.verify(raw_sig, data)
            return True
        except Exception:
            return False

    @classmethod
    def rotate(cls, key_file: str) -> SigningKey:
        """Archive the current key file and generate a fresh key.

        The old key file is renamed to ``<key_file>.<unix_timestamp>.bak``
        before the new key is written.

        Args:
            key_file: Path to the existing (or target) PEM key file.

        Returns:
            A new :class:`SigningKey` instance backed by the fresh key.
        """
        path = Path(key_file)
        if path.exists():
            backup = f"{key_file}.{int(time.time())}.bak"
            logger.info("Archiving old signing key to %s", backup)
            path.rename(backup)
        return cls(key_file)
