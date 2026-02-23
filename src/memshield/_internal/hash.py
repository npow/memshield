"""SHA-256 hash chain for provenance tracking."""
from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS_HASH = "0" * 64


def compute_entry_hash(content: str, metadata: dict[str, Any] | None = None) -> str:
    """Compute a SHA-256 hash of a memory entry's content and metadata."""
    payload = {"content": content}
    if metadata:
        payload["metadata"] = metadata
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_chain_hash(entry_hash: str, previous_hash: str) -> str:
    """Compute the chain hash linking an entry to the previous hash in the chain."""
    combined = (previous_hash + entry_hash).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def verify_chain_link(
    entry_hash: str,
    previous_hash: str,
    expected_chain_hash: str,
) -> bool:
    """Verify that a chain link is valid by recomputing the chain hash."""
    actual = compute_chain_hash(entry_hash, previous_hash)
    return actual == expected_chain_hash
