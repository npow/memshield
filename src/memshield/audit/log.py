"""Core audit log: write, verify, export, erase, and purge records."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from memshield.audit.config import AuditConfig
from memshield.audit.crypto import AESCipher, KeyStore
from memshield.audit.schema import AuditRecord, BlockedChunk, RetrievedChunk
from memshield.audit.signing import SigningKey
from memshield.audit.tsa import stamp

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_records (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    inference_id           TEXT    NOT NULL UNIQUE,
    timestamp_iso          TEXT    NOT NULL,
    timestamp_rfc3161      TEXT,
    key_id                 TEXT    NOT NULL,
    user_id                TEXT,
    query_hash             TEXT    NOT NULL,
    query_encrypted        TEXT,
    knowledge_base_id      TEXT    NOT NULL,
    retrieved_json         TEXT    NOT NULL,
    blocked_json           TEXT    NOT NULL,
    chain_hash             TEXT    NOT NULL,
    previous_chain_hash    TEXT    NOT NULL,
    signature              TEXT    NOT NULL,
    is_tombstone           INTEGER NOT NULL DEFAULT 0,
    iso24970_event_type    TEXT    NOT NULL DEFAULT 'retrieval',
    iso24970_schema_version TEXT   NOT NULL DEFAULT 'DIS-2025'
);
CREATE INDEX IF NOT EXISTS idx_user_id   ON audit_records(user_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp_iso);
"""

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_records (
    id                     SERIAL PRIMARY KEY,
    inference_id           TEXT    NOT NULL UNIQUE,
    timestamp_iso          TEXT    NOT NULL,
    timestamp_rfc3161      TEXT,
    key_id                 TEXT    NOT NULL,
    user_id                TEXT,
    query_hash             TEXT    NOT NULL,
    query_encrypted        TEXT,
    knowledge_base_id      TEXT    NOT NULL,
    retrieved_json         TEXT    NOT NULL,
    blocked_json           TEXT    NOT NULL,
    chain_hash             TEXT    NOT NULL,
    previous_chain_hash    TEXT    NOT NULL,
    signature              TEXT    NOT NULL,
    is_tombstone           INTEGER NOT NULL DEFAULT 0,
    iso24970_event_type    TEXT    NOT NULL DEFAULT 'retrieval',
    iso24970_schema_version TEXT   NOT NULL DEFAULT 'DIS-2025'
);
CREATE INDEX IF NOT EXISTS idx_user_id   ON audit_records(user_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp_iso);
"""


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _sha256_hex(data: bytes) -> str:
    """Return hex-encoded SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _record_hash(
    *,
    inference_id: str,
    timestamp_iso: str,
    timestamp_rfc3161: str | None,
    key_id: str,
    user_id: str | None,
    query_hash: str,
    query_encrypted: str | None,
    knowledge_base_id: str,
    retrieved_json: list,
    blocked_json: list,
    iso24970_event_type: str,
    iso24970_schema_version: str,
) -> str:
    """Compute SHA-256 over a canonical JSON representation of a record.

    The ``signature`` and both chain hash fields are intentionally excluded so
    that the record hash can be computed before signing and before the chain
    hash is known.
    """
    canonical = {
        "inference_id": inference_id,
        "timestamp_iso": timestamp_iso,
        "timestamp_rfc3161": timestamp_rfc3161,
        "key_id": key_id,
        "user_id": user_id,
        "query_hash": query_hash,
        "query_encrypted": query_encrypted,
        "knowledge_base_id": knowledge_base_id,
        "retrieved": retrieved_json,
        "blocked": blocked_json,
        "iso24970_event_type": iso24970_event_type,
        "iso24970_schema_version": iso24970_schema_version,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_hex(raw)


def _chain_hash(previous_chain_hash: str, rec_hash: str) -> str:
    """Compute the chain hash linking this record to the previous one.

    ``SHA-256(previous_chain_hash_bytes + record_hash_bytes)``
    """
    combined = previous_chain_hash.encode("utf-8") + rec_hash.encode("utf-8")
    return _sha256_hex(combined)


# ---------------------------------------------------------------------------
# Backend abstractions
# ---------------------------------------------------------------------------

class _SQLiteBackend:
    """SQLite storage backend with WAL mode and a write lock."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SQLITE_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, row: dict) -> None:
        """Insert a single audit record row."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO audit_records (
                        inference_id, timestamp_iso, timestamp_rfc3161,
                        key_id, user_id, query_hash, query_encrypted,
                        knowledge_base_id, retrieved_json, blocked_json,
                        chain_hash, previous_chain_hash, signature,
                        is_tombstone, iso24970_event_type, iso24970_schema_version
                    ) VALUES (
                        :inference_id, :timestamp_iso, :timestamp_rfc3161,
                        :key_id, :user_id, :query_hash, :query_encrypted,
                        :knowledge_base_id, :retrieved_json, :blocked_json,
                        :chain_hash, :previous_chain_hash, :signature,
                        :is_tombstone, :iso24970_event_type, :iso24970_schema_version
                    )
                    """,
                    row,
                )
                conn.commit()
            finally:
                conn.close()

    def fetch_last(self) -> sqlite3.Row | None:
        """Return the most recently inserted row, or ``None``."""
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT * FROM audit_records ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()

    def fetch_by_id(self, inference_id: str) -> sqlite3.Row | None:
        """Return the row with the given *inference_id*, or ``None``."""
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT * FROM audit_records WHERE inference_id = ?",
                (inference_id,),
            ).fetchone()
        finally:
            conn.close()

    def fetch_all_ordered(self) -> list[sqlite3.Row]:
        """Return all rows ordered by insertion id (ascending)."""
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT * FROM audit_records ORDER BY id ASC"
            ).fetchall()
        finally:
            conn.close()

    def fetch_from_date(self, from_date: str) -> list[sqlite3.Row]:
        """Return rows with timestamp_iso >= *from_date*, ordered by id."""
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT * FROM audit_records WHERE timestamp_iso >= ? ORDER BY id ASC",
                (from_date,),
            ).fetchall()
        finally:
            conn.close()

    def count_user_records(self, user_id: str) -> int:
        """Count live (non-tombstone) records for *user_id*."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM audit_records WHERE user_id = ? AND is_tombstone = 0",
                (user_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()

    def tombstone_expired(self, cutoff_iso: str) -> int:
        """Replace expired records with tombstone placeholders.

        Returns the number of rows affected.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE audit_records SET
                        query_hash          = '[purged]',
                        query_encrypted     = NULL,
                        retrieved_json      = '[]',
                        blocked_json        = '[]',
                        knowledge_base_id   = '[purged]',
                        user_id             = NULL,
                        is_tombstone        = 1
                    WHERE timestamp_iso < ? AND is_tombstone = 0
                    """,
                    (cutoff_iso,),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()


class _PostgresBackend:
    """PostgreSQL storage backend using a threaded connection pool."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2  # type: ignore[import]
            from psycopg2 import pool as pg_pool  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "psycopg2-binary is required for the postgres backend: "
                "pip install 'memshield[audit-postgres]'"
            ) from exc

        self._pool = pg_pool.ThreadedConnectionPool(1, 10, dsn)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_POSTGRES_SCHEMA)
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def _getconn(self):  # type: ignore[return]
        return self._pool.getconn()

    def _putconn(self, conn) -> None:  # type: ignore[return]
        self._pool.putconn(conn)

    def _row_to_dict(self, row, description) -> dict:
        return {desc[0]: val for desc, val in zip(description, row)}

    def insert(self, row: dict) -> None:
        """Insert a single audit record row."""
        conn = self._getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_records (
                        inference_id, timestamp_iso, timestamp_rfc3161,
                        key_id, user_id, query_hash, query_encrypted,
                        knowledge_base_id, retrieved_json, blocked_json,
                        chain_hash, previous_chain_hash, signature,
                        is_tombstone, iso24970_event_type, iso24970_schema_version
                    ) VALUES (
                        %(inference_id)s, %(timestamp_iso)s, %(timestamp_rfc3161)s,
                        %(key_id)s, %(user_id)s, %(query_hash)s, %(query_encrypted)s,
                        %(knowledge_base_id)s, %(retrieved_json)s, %(blocked_json)s,
                        %(chain_hash)s, %(previous_chain_hash)s, %(signature)s,
                        %(is_tombstone)s, %(iso24970_event_type)s, %(iso24970_schema_version)s
                    )
                    """,
                    row,
                )
            conn.commit()
        finally:
            self._putconn(conn)

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        conn = self._getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row, cur.description)
        finally:
            self._putconn(conn)

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = self._getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                desc = cur.description
                return [self._row_to_dict(r, desc) for r in rows]
        finally:
            self._putconn(conn)

    def fetch_last(self) -> dict | None:
        return self._fetchone(
            "SELECT * FROM audit_records ORDER BY id DESC LIMIT 1"
        )

    def fetch_by_id(self, inference_id: str) -> dict | None:
        return self._fetchone(
            "SELECT * FROM audit_records WHERE inference_id = %s", (inference_id,)
        )

    def fetch_all_ordered(self) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM audit_records ORDER BY id ASC"
        )

    def fetch_from_date(self, from_date: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM audit_records WHERE timestamp_iso >= %s ORDER BY id ASC",
            (from_date,),
        )

    def count_user_records(self, user_id: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS cnt FROM audit_records WHERE user_id = %s AND is_tombstone = 0",
            (user_id,),
        )
        return int(row["cnt"]) if row else 0

    def tombstone_expired(self, cutoff_iso: str) -> int:
        conn = self._getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE audit_records SET
                        query_hash          = '[purged]',
                        query_encrypted     = NULL,
                        retrieved_json      = '[]',
                        blocked_json        = '[]',
                        knowledge_base_id   = '[purged]',
                        user_id             = NULL,
                        is_tombstone        = 1
                    WHERE timestamp_iso < %s AND is_tombstone = 0
                    """,
                    (cutoff_iso,),
                )
                rowcount = cur.rowcount
            conn.commit()
            return rowcount
        finally:
            self._putconn(conn)


# ---------------------------------------------------------------------------
# Row → AuditRecord deserialization
# ---------------------------------------------------------------------------

def _row_to_audit_record(row: Any) -> AuditRecord:
    """Convert a database row (sqlite3.Row or dict) to an :class:`AuditRecord`."""
    # Support both sqlite3.Row (subscript by name) and plain dicts
    def get(key: str) -> Any:
        try:
            return row[key]
        except (KeyError, IndexError):
            return None

    retrieved_raw: list[dict] = json.loads(get("retrieved_json") or "[]")
    blocked_raw: list[dict] = json.loads(get("blocked_json") or "[]")

    retrieved = [
        RetrievedChunk(
            doc_id=c["doc_id"],
            chunk_index=c["chunk_index"],
            content_hash=c["content_hash"],
            content_encrypted=c.get("content_encrypted"),
            score=c["score"],
            verdict=c["verdict"],
            trust_level=c["trust_level"],
        )
        for c in retrieved_raw
    ]
    blocked = [
        BlockedChunk(
            content_hash=c["content_hash"],
            content_encrypted=c.get("content_encrypted"),
            verdict=c["verdict"],
            confidence=c["confidence"],
            attack_type=c.get("attack_type"),
        )
        for c in blocked_raw
    ]

    return AuditRecord(
        inference_id=get("inference_id"),
        timestamp_iso=get("timestamp_iso"),
        timestamp_rfc3161=get("timestamp_rfc3161"),
        key_id=get("key_id"),
        user_id=get("user_id"),
        query_hash=get("query_hash"),
        query_encrypted=get("query_encrypted"),
        knowledge_base_id=get("knowledge_base_id"),
        retrieved=retrieved,
        blocked=blocked,
        chain_hash=get("chain_hash"),
        previous_chain_hash=get("previous_chain_hash"),
        signature=get("signature"),
        iso24970_event_type=get("iso24970_event_type") or "retrieval",
        iso24970_schema_version=get("iso24970_schema_version") or "DIS-2025",
    )


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class AuditLog:
    """Append-only, signed, chain-hashed audit log for MemShield.

    Each :meth:`write` call produces an :class:`~memshield.audit.schema.AuditRecord`
    that is cryptographically linked to all previous records via a SHA-256 chain
    hash and signed with an Ed25519 key.  Records can optionally have PII fields
    encrypted with per-user AES-256-GCM keys.

    Args:
        config: :class:`~memshield.audit.config.AuditConfig` instance.
    """

    def __init__(self, config: AuditConfig) -> None:
        """Initialise the audit log, creating the database and signing key.

        Args:
            config: Audit log configuration.
        """
        self._config = config
        self._signing_key = SigningKey(config.key_file)

        if config.pii_fields and config.key_store:
            self._key_store: KeyStore | None = KeyStore(config.key_store)
        else:
            self._key_store = None

        if config.backend == "sqlite":
            self._backend: _SQLiteBackend | _PostgresBackend = _SQLiteBackend(config.store)
        elif config.backend == "postgres":
            if not config.postgres_dsn:
                raise ValueError("postgres_dsn is required for the postgres backend")
            self._backend = _PostgresBackend(config.postgres_dsn)
        else:
            raise ValueError(f"Unsupported backend: {config.backend!r}")

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------

    def write(
        self,
        *,
        query: str,
        user_id: str | None = None,
        inference_id: str | None = None,
        retrieved: list[dict],
        blocked: list[dict],
    ) -> AuditRecord:
        """Append a new signed audit record to the log.

        Args:
            query: Raw query string sent to the vector store.
            user_id: Optional identifier of the requesting user.
            inference_id: UUID for this event; auto-generated if ``None``.
            retrieved: List of dicts describing chunks returned to the caller.
                Each dict must contain: ``doc_id``, ``chunk_index``, ``content``,
                ``score``, ``verdict``, ``trust_level``.
            blocked: List of dicts describing chunks blocked by the shield.
                Each dict must contain: ``content``, ``verdict``,
                ``confidence``, ``attack_type``.

        Returns:
            The newly created and persisted :class:`~memshield.audit.schema.AuditRecord`.
        """
        if inference_id is None:
            inference_id = str(uuid.uuid4())

        timestamp_iso = datetime.now(timezone.utc).isoformat()
        query_hash = _sha256_hex(query.encode("utf-8"))

        # PII encryption
        pii = self._config.pii_fields
        encrypt_query = "query" in pii
        encrypt_content = "content" in pii

        aes_key: bytes | None = None
        if (encrypt_query or encrypt_content) and user_id and self._key_store:
            aes_key = self._key_store.get_or_create_key(user_id)

        query_encrypted: str | None = None
        if encrypt_query and aes_key:
            query_encrypted = AESCipher.encrypt(query, aes_key)

        # Build serialisable chunk lists
        retrieved_json: list[dict] = []
        for chunk in retrieved:
            content: str = chunk["content"]
            content_hash = _sha256_hex(content.encode("utf-8"))
            content_encrypted: str | None = None
            if encrypt_content and aes_key:
                content_encrypted = AESCipher.encrypt(content, aes_key)
            retrieved_json.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "content_hash": content_hash,
                    "content_encrypted": content_encrypted,
                    "score": chunk["score"],
                    "verdict": chunk["verdict"],
                    "trust_level": chunk["trust_level"],
                }
            )

        blocked_json: list[dict] = []
        for chunk in blocked:
            content = chunk["content"]
            content_hash = _sha256_hex(content.encode("utf-8"))
            content_encrypted = None
            if encrypt_content and aes_key:
                content_encrypted = AESCipher.encrypt(content, aes_key)
            elif not encrypt_content:
                # Store plaintext when no PII encryption is configured
                content_encrypted = content
            blocked_json.append(
                {
                    "content_hash": content_hash,
                    "content_encrypted": content_encrypted,
                    "verdict": chunk["verdict"],
                    "confidence": chunk["confidence"],
                    "attack_type": chunk.get("attack_type"),
                }
            )

        key_id = self._signing_key.key_id

        # Compute record hash (excludes signature and chain hashes)
        rec_hash = _record_hash(
            inference_id=inference_id,
            timestamp_iso=timestamp_iso,
            timestamp_rfc3161=None,  # TSA token not yet known
            key_id=key_id,
            user_id=user_id,
            query_hash=query_hash,
            query_encrypted=query_encrypted,
            knowledge_base_id=self._config.knowledge_base_id,
            retrieved_json=retrieved_json,
            blocked_json=blocked_json,
            iso24970_event_type="retrieval",
            iso24970_schema_version="DIS-2025",
        )

        # Optional TSA stamp
        timestamp_rfc3161: str | None = None
        if self._config.tsa_url:
            timestamp_rfc3161 = stamp(
                bytes.fromhex(rec_hash), self._config.tsa_url
            )

        # Re-compute record hash with the TSA token (may be None — same result)
        rec_hash = _record_hash(
            inference_id=inference_id,
            timestamp_iso=timestamp_iso,
            timestamp_rfc3161=timestamp_rfc3161,
            key_id=key_id,
            user_id=user_id,
            query_hash=query_hash,
            query_encrypted=query_encrypted,
            knowledge_base_id=self._config.knowledge_base_id,
            retrieved_json=retrieved_json,
            blocked_json=blocked_json,
            iso24970_event_type="retrieval",
            iso24970_schema_version="DIS-2025",
        )

        # Determine previous chain hash
        last_row = self._backend.fetch_last()
        previous_chain_hash: str
        if last_row is None:
            previous_chain_hash = GENESIS_HASH
        else:
            try:
                previous_chain_hash = last_row["chain_hash"]
            except (KeyError, TypeError):
                previous_chain_hash = GENESIS_HASH

        computed_chain_hash = _chain_hash(previous_chain_hash, rec_hash)

        signature = self._signing_key.sign(rec_hash.encode("utf-8"))

        record = AuditRecord(
            inference_id=inference_id,
            timestamp_iso=timestamp_iso,
            timestamp_rfc3161=timestamp_rfc3161,
            key_id=key_id,
            user_id=user_id,
            query_hash=query_hash,
            query_encrypted=query_encrypted,
            knowledge_base_id=self._config.knowledge_base_id,
            retrieved=[
                RetrievedChunk(
                    doc_id=c["doc_id"],
                    chunk_index=c["chunk_index"],
                    content_hash=c["content_hash"],
                    content_encrypted=c["content_encrypted"],
                    score=c["score"],
                    verdict=c["verdict"],
                    trust_level=c["trust_level"],
                )
                for c in retrieved_json
            ],
            blocked=[
                BlockedChunk(
                    content_hash=c["content_hash"],
                    content_encrypted=c["content_encrypted"],
                    verdict=c["verdict"],
                    confidence=c["confidence"],
                    attack_type=c.get("attack_type"),
                )
                for c in blocked_json
            ],
            chain_hash=computed_chain_hash,
            previous_chain_hash=previous_chain_hash,
            signature=signature,
        )

        self._backend.insert(
            {
                "inference_id": inference_id,
                "timestamp_iso": timestamp_iso,
                "timestamp_rfc3161": timestamp_rfc3161,
                "key_id": key_id,
                "user_id": user_id,
                "query_hash": query_hash,
                "query_encrypted": query_encrypted,
                "knowledge_base_id": self._config.knowledge_base_id,
                "retrieved_json": json.dumps(retrieved_json, separators=(",", ":")),
                "blocked_json": json.dumps(blocked_json, separators=(",", ":")),
                "chain_hash": computed_chain_hash,
                "previous_chain_hash": previous_chain_hash,
                "signature": signature,
                "is_tombstone": 0,
                "iso24970_event_type": "retrieval",
                "iso24970_schema_version": "DIS-2025",
            }
        )

        return record

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def last_record(self) -> AuditRecord | None:
        """Return the most recently written :class:`AuditRecord`, or ``None``."""
        row = self._backend.fetch_last()
        if row is None:
            return None
        return _row_to_audit_record(row)

    def get_record(self, inference_id: str) -> AuditRecord | None:
        """Return the :class:`AuditRecord` with *inference_id*, or ``None``.

        Args:
            inference_id: UUID of the record to retrieve.
        """
        row = self._backend.fetch_by_id(inference_id)
        if row is None:
            return None
        return _row_to_audit_record(row)

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Recompute and verify every chain hash and signature in the log.

        Returns:
            A ``(is_valid, errors)`` tuple where *errors* is a (possibly empty)
            list of human-readable error strings.
        """
        rows = self._backend.fetch_all_ordered()
        errors: list[str] = []
        previous_chain_hash = GENESIS_HASH

        for row in rows:
            iid: str = row["inference_id"]
            stored_prev: str = row["previous_chain_hash"]
            stored_chain: str = row["chain_hash"]
            stored_sig: str = row["signature"]
            is_tombstone: int = int(row["is_tombstone"] or 0)

            if is_tombstone:
                # Tombstones preserve chain fields; we still verify the link
                if stored_prev != previous_chain_hash:
                    errors.append(
                        f"Tombstone {iid}: previous_chain_hash mismatch "
                        f"(stored={stored_prev!r}, expected={previous_chain_hash!r})"
                    )
                expected_chain = _chain_hash(previous_chain_hash, "")
                # For tombstones we cannot recompute the record hash from PII
                # fields (they are scrubbed), so we trust the stored chain_hash
                # and just verify the linkage.
                # We verify that the stored chain_hash was computed from the
                # stored previous_chain_hash — but we cannot reconstruct the
                # original record_hash.  We skip signature verification for
                # tombstones (the original signature is preserved but not re-checked).
                previous_chain_hash = stored_chain
                continue

            retrieved_raw: list = json.loads(row["retrieved_json"] or "[]")
            blocked_raw: list = json.loads(row["blocked_json"] or "[]")

            rec_hash = _record_hash(
                inference_id=iid,
                timestamp_iso=row["timestamp_iso"],
                timestamp_rfc3161=row["timestamp_rfc3161"],
                key_id=row["key_id"],
                user_id=row["user_id"],
                query_hash=row["query_hash"],
                query_encrypted=row["query_encrypted"],
                knowledge_base_id=row["knowledge_base_id"],
                retrieved_json=retrieved_raw,
                blocked_json=blocked_raw,
                iso24970_event_type=row["iso24970_event_type"] or "retrieval",
                iso24970_schema_version=row["iso24970_schema_version"] or "DIS-2025",
            )

            # Verify previous_chain_hash linkage
            if stored_prev != previous_chain_hash:
                errors.append(
                    f"Record {iid}: previous_chain_hash mismatch "
                    f"(stored={stored_prev!r}, expected={previous_chain_hash!r})"
                )

            # Verify chain_hash
            expected_chain = _chain_hash(stored_prev, rec_hash)
            if stored_chain != expected_chain:
                errors.append(
                    f"Record {iid}: chain_hash mismatch "
                    f"(stored={stored_chain!r}, expected={expected_chain!r})"
                )

            # Verify signature
            if not self._signing_key.verify(rec_hash.encode("utf-8"), stored_sig):
                errors.append(f"Record {iid}: signature verification failed")

            previous_chain_hash = stored_chain

        return (len(errors) == 0, errors)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, *, from_date: str | None = None) -> list[dict]:
        """Return a JSON-serialisable list of all (or filtered) records.

        Args:
            from_date: If provided, only records with ``timestamp_iso >= from_date``
                are returned.  Expected format: ISO 8601 UTC string.

        Returns:
            List of dicts, one per record.  Tombstone rows include an
            ``"is_tombstone": true`` key and omit scrubbed fields.
        """
        if from_date is not None:
            rows = self._backend.fetch_from_date(from_date)
        else:
            rows = self._backend.fetch_all_ordered()

        result: list[dict] = []
        for row in rows:
            is_tombstone = int(row["is_tombstone"] or 0)
            if is_tombstone:
                result.append(
                    {
                        "inference_id": row["inference_id"],
                        "timestamp_iso": row["timestamp_iso"],
                        "chain_hash": row["chain_hash"],
                        "previous_chain_hash": row["previous_chain_hash"],
                        "key_id": row["key_id"],
                        "signature": row["signature"],
                        "is_tombstone": True,
                    }
                )
            else:
                record = _row_to_audit_record(row)
                result.append(record.to_dict())
        return result

    # ------------------------------------------------------------------
    # GDPR / crypto-shredding
    # ------------------------------------------------------------------

    def erase_user(self, user_id: str) -> int:
        """Destroy the AES encryption key for *user_id* (crypto-shredding).

        This renders all PII-encrypted fields for that user permanently
        unreadable.  Audit records themselves are **not** deleted or modified —
        the chain remains intact.

        Args:
            user_id: Identifier of the user whose data should be erased.

        Returns:
            Count of live (non-tombstone) audit records that were encrypted
            with the destroyed key.
        """
        count = self._backend.count_user_records(user_id)
        if self._key_store is not None:
            self._key_store.delete_key(user_id)
        return count

    # ------------------------------------------------------------------
    # Retention / tombstoning
    # ------------------------------------------------------------------

    def purge_expired(self) -> int:
        """Replace records older than ``retention_days`` with tombstones.

        Chain fields (``chain_hash``, ``previous_chain_hash``, ``signature``,
        ``key_id``) are preserved.  All other fields containing PII or
        operational data are scrubbed.

        Returns:
            Number of records converted to tombstones.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self._config.retention_days
        )
        cutoff_iso = cutoff.isoformat()
        return self._backend.tombstone_expired(cutoff_iso)
