"""Frozen dataclasses for the MemShield audit log schema."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievedChunk:
    """An individual chunk that passed validation and was returned to the caller.

    Attributes:
        doc_id: Identifier of the source document.
        chunk_index: Position of this chunk within its document.
        content_hash: SHA-256 hex digest of the raw chunk content.
        content_encrypted: AES-256-GCM ciphertext (base64) or None.
        score: Retrieval similarity score.
        verdict: Always ``"clean"`` for retrieved chunks.
        trust_level: Provenance trust level — ``"verified"``, ``"unverified"``, or
            ``"untrusted"``.
    """

    doc_id: str
    chunk_index: int
    content_hash: str
    content_encrypted: str | None
    score: float
    verdict: str
    trust_level: str


@dataclass(frozen=True)
class BlockedChunk:
    """A chunk that was blocked by the shield before reaching the caller.

    Attributes:
        content_hash: SHA-256 hex digest of the raw chunk content.
        content_encrypted: AES-256-GCM ciphertext (base64), plaintext if no
            PII encryption is configured, or None.
        verdict: ``"poisoned"`` or ``"ambiguous"``.
        confidence: Model confidence in the verdict (0.0–1.0).
        attack_type: Optional classifier label for the detected attack pattern.
    """

    content_hash: str
    content_encrypted: str | None
    verdict: str
    confidence: float
    attack_type: str | None


@dataclass(frozen=True)
class AuditRecord:
    """Immutable, signed record of a single inference retrieval event.

    Attributes:
        inference_id: UUID identifying this inference call.
        timestamp_iso: UTC timestamp in ISO 8601 format.
        timestamp_rfc3161: RFC 3161 timestamp token (base64) or None.
        key_id: SHA-256 fingerprint (hex) of the signing public key.
        user_id: Optional identifier of the requesting user.
        query_hash: SHA-256 hex digest of the raw query string.
        query_encrypted: AES-256-GCM ciphertext (base64) of the query, or None.
        knowledge_base_id: Identifier of the vector store / knowledge base.
        retrieved: Ordered list of chunks returned to the caller.
        blocked: Ordered list of chunks blocked by the shield.
        chain_hash: SHA-256 linking this record to the previous one.
        previous_chain_hash: ``chain_hash`` of the immediately preceding record.
        signature: Base64-encoded Ed25519 signature over the record hash.
        iso24970_event_type: ISO/IEC 24970 event category (default ``"retrieval"``).
        iso24970_schema_version: Schema version string (default ``"DIS-2025"``).
    """

    inference_id: str
    timestamp_iso: str
    timestamp_rfc3161: str | None
    key_id: str
    user_id: str | None
    query_hash: str
    query_encrypted: str | None
    knowledge_base_id: str
    retrieved: list[RetrievedChunk]
    blocked: list[BlockedChunk]
    chain_hash: str
    previous_chain_hash: str
    signature: str
    iso24970_event_type: str = "retrieval"
    iso24970_schema_version: str = "DIS-2025"

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of this record."""
        return {
            "inference_id": self.inference_id,
            "timestamp_iso": self.timestamp_iso,
            "timestamp_rfc3161": self.timestamp_rfc3161,
            "key_id": self.key_id,
            "user_id": self.user_id,
            "query_hash": self.query_hash,
            "query_encrypted": self.query_encrypted,
            "knowledge_base_id": self.knowledge_base_id,
            "retrieved": [
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "content_hash": c.content_hash,
                    "content_encrypted": c.content_encrypted,
                    "score": c.score,
                    "verdict": c.verdict,
                    "trust_level": c.trust_level,
                }
                for c in self.retrieved
            ],
            "blocked": [
                {
                    "content_hash": c.content_hash,
                    "content_encrypted": c.content_encrypted,
                    "verdict": c.verdict,
                    "confidence": c.confidence,
                    "attack_type": c.attack_type,
                }
                for c in self.blocked
            ],
            "chain_hash": self.chain_hash,
            "previous_chain_hash": self.previous_chain_hash,
            "signature": self.signature,
            "iso24970_event_type": self.iso24970_event_type,
            "iso24970_schema_version": self.iso24970_schema_version,
        }


@dataclass(frozen=True)
class TombstoneRecord:
    """Replaces an :class:`AuditRecord` after its retention period expires.

    PII-bearing fields are scrubbed, but chain fields are preserved so that
    downstream records remain verifiable.

    Attributes:
        inference_id: UUID of the original record that was purged.
        timestamp_iso: UTC timestamp of the *original* record (preserved).
        chain_hash: Chain hash of the original record (preserved).
        previous_chain_hash: Previous chain hash of the original record (preserved).
        key_id: Key fingerprint of the original record (preserved).
        signature: Signature of the original record (preserved).
        is_tombstone: Always ``True``.
    """

    inference_id: str
    timestamp_iso: str
    chain_hash: str
    previous_chain_hash: str
    key_id: str
    signature: str
    is_tombstone: bool = True
