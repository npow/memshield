"""Provenance tracker — tag writes, verify chains, assign trust levels."""
from __future__ import annotations

import logging
import time
from typing import Any

from memshield._internal.hash import (
    GENESIS_HASH,
    compute_chain_hash,
    compute_entry_hash,
    verify_chain_link,
)
from memshield._types import ProvenanceRecord, TrustLevel

logger = logging.getLogger(__name__)

# Sources considered trusted by default
TRUSTED_SOURCES = frozenset({"user_input", "system", "internal"})


class ProvenanceTracker:
    """Tracks provenance of memory writes with a cryptographic hash chain."""

    def __init__(self) -> None:
        self._chain: list[ProvenanceRecord] = []
        self._last_chain_hash: str = GENESIS_HASH

    @property
    def chain_length(self) -> int:
        """Number of entries in the provenance chain."""
        return len(self._chain)

    @property
    def last_hash(self) -> str:
        """The most recent chain hash."""
        return self._last_chain_hash

    def record_write(
        self,
        content: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceRecord:
        """Record a memory write operation in the provenance chain."""
        trust_level = self._assess_trust(source)
        entry_hash = compute_entry_hash(content, metadata)
        chain_hash = compute_chain_hash(entry_hash, self._last_chain_hash)

        record = ProvenanceRecord(
            source=source,
            timestamp=time.time(),
            trust_level=trust_level,
            previous_hash=self._last_chain_hash,
            entry_hash=entry_hash,
            metadata=dict(metadata) if metadata else {},
        )

        self._chain.append(record)
        self._last_chain_hash = chain_hash
        return record

    def verify_chain(self) -> bool:
        """Verify the entire provenance chain's integrity.

        Returns True if all links are valid, False if any link is broken.
        """
        if not self._chain:
            return True

        previous_hash = GENESIS_HASH
        for record in self._chain:
            expected_chain_hash = compute_chain_hash(record.entry_hash, previous_hash)
            if record.previous_hash != previous_hash:
                logger.warning(
                    "Provenance chain broken: record for source=%s has "
                    "previous_hash=%s but expected %s",
                    record.source,
                    record.previous_hash,
                    previous_hash,
                )
                return False
            previous_hash = expected_chain_hash

        return True

    def get_record(self, index: int) -> ProvenanceRecord:
        """Get a provenance record by index."""
        return self._chain[index]

    def get_trust_level(self, content: str, metadata: dict[str, Any] | None = None) -> TrustLevel:
        """Look up the trust level for a memory entry by its content hash.

        Returns UNVERIFIED if the entry is not found in the chain.
        """
        target_hash = compute_entry_hash(content, metadata)
        for record in reversed(self._chain):
            if record.entry_hash == target_hash:
                return record.trust_level
        return TrustLevel.UNVERIFIED

    @staticmethod
    def _assess_trust(source: str) -> TrustLevel:
        """Determine trust level based on the data source."""
        if source in TRUSTED_SOURCES:
            return TrustLevel.VERIFIED
        if source.startswith("tool:"):
            return TrustLevel.UNVERIFIED
        return TrustLevel.UNTRUSTED
