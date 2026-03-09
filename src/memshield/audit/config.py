"""Configuration dataclass for the MemShield audit log."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AuditConfig:
    """Configuration for an :class:`~memshield.audit.AuditLog` instance.

    Attributes:
        store: Path to the SQLite database file (e.g. ``"./audit.db"``).
        knowledge_base_id: Identifier of the vector store being audited.
            **Required** — no default value.
        key_file: Path to the Ed25519 private key PEM file.  Generated
            automatically if the file does not exist.
        key_store: Path to the SQLite key store used for per-user AES-256 keys
            (required when ``pii_fields`` is non-empty).
        pii_fields: Fields to encrypt at rest.  Supported values are
            ``"query"`` and ``"content"``.
        tsa_url: URL of an RFC 3161 Time-Stamp Authority.  Set to ``None`` to
            disable TSA stamping.
        retention_days: Number of days after which records are replaced with
            :class:`~memshield.audit.schema.TombstoneRecord` entries.
        backend: Storage backend — ``"sqlite"`` (default) or ``"postgres"``.
        postgres_dsn: libpq connection string; required when
            ``backend="postgres"``.
    """

    store: str
    knowledge_base_id: str
    key_file: str = "./memshield.key"
    key_store: str | None = None
    pii_fields: list[str] = field(default_factory=list)
    tsa_url: str | None = "http://timestamp.sectigo.com"
    retention_days: int = 180
    backend: str = "sqlite"
    postgres_dsn: str | None = None

    def __post_init__(self) -> None:
        """Validate field combinations."""
        if self.pii_fields and not self.key_store:
            raise ValueError(
                "key_store is required when pii_fields is set"
            )
        if self.backend == "postgres" and not self.postgres_dsn:
            raise ValueError(
                "postgres_dsn is required when backend='postgres'"
            )
        if self.backend not in ("sqlite", "postgres"):
            raise ValueError(
                f"backend must be 'sqlite' or 'postgres', got {self.backend!r}"
            )
