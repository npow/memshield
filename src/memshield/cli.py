"""MemShield command-line interface."""
from __future__ import annotations

import json
import sys

try:
    import click
except ImportError:  # pragma: no cover
    raise ImportError(
        "click is required for the CLI. Install with: pip install memshield[cli]"
    ) from None


@click.group()
def cli() -> None:
    """MemShield — production-grade memory integrity defense for AI agents."""


# ---------------------------------------------------------------------------
# audit sub-group
# ---------------------------------------------------------------------------

@cli.group()
def audit() -> None:
    """Audit log management commands."""


def _make_log(db: str, key_file: str = "./memshield.key") -> "AuditLog":  # type: ignore[name-defined]
    from memshield.audit.config import AuditConfig
    from memshield.audit.log import AuditLog

    config = AuditConfig(
        store=db, knowledge_base_id="cli", key_file=key_file, tsa_url=None
    )
    return AuditLog(config)


@audit.command()
@click.option("--db", required=True, help="Path to audit.db")
@click.option(
    "--key-file",
    default="./memshield.key",
    show_default=True,
    help="Path to the Ed25519 signing key PEM file",
)
def verify(db: str, key_file: str) -> None:
    """Verify audit chain integrity."""
    log = _make_log(db, key_file=key_file)
    is_valid, errors = log.verify_chain()
    if is_valid:
        click.echo("Chain integrity: VALID")
    else:
        click.echo("Chain integrity: INVALID", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)


@audit.command()
@click.option("--db", required=True, help="Path to audit.db")
@click.option(
    "--key-file",
    default="./memshield.key",
    show_default=True,
    help="Path to the Ed25519 signing key PEM file",
)
@click.option("--from", "from_date", default=None, help="ISO 8601 date filter")
@click.option(
    "--format",
    "fmt",
    default="jsonl",
    type=click.Choice(["jsonl", "json"]),
    show_default=True,
)
def export(db: str, key_file: str, from_date: str | None, fmt: str) -> None:
    """Export audit records to stdout."""
    log = _make_log(db, key_file=key_file)
    records = log.export(from_date=from_date)
    if fmt == "json":
        click.echo(json.dumps(records, indent=2))
    else:
        for record in records:
            click.echo(json.dumps(record))


@audit.command()
@click.option("--db", required=True, help="Path to audit.db")
@click.option(
    "--key-file",
    default="./memshield.key",
    show_default=True,
    help="Path to the Ed25519 signing key PEM file",
)
@click.option("--inference-id", required=True, help="Inference ID to inspect")
def inspect(db: str, key_file: str, inference_id: str) -> None:
    """Inspect a single audit record by inference ID."""
    log = _make_log(db, key_file=key_file)
    record = log.get_record(inference_id)
    if record is None:
        click.echo(f"No record found for inference_id={inference_id!r}", err=True)
        sys.exit(1)
    click.echo(json.dumps(record.to_dict(), indent=2))


@audit.command("erase-user")
@click.option("--db", required=True, help="Path to audit.db")
@click.option("--key-store", required=True, help="Path to keys.db")
@click.option("--user-id", required=True, help="User ID to crypto-shred")
@click.option("--knowledge-base-id", required=True, help="Knowledge base ID")
def erase_user(db: str, key_store: str, user_id: str, knowledge_base_id: str) -> None:
    """Crypto-shred a user's data from the audit log."""
    from memshield.audit.config import AuditConfig
    from memshield.audit.log import AuditLog

    config = AuditConfig(
        store=db,
        knowledge_base_id=knowledge_base_id,
        key_store=key_store,
        pii_fields=["query", "content"],
        tsa_url=None,
    )
    log = AuditLog(config)
    count = log.erase_user(user_id)
    click.echo(f"Erased {count} record(s) for user {user_id!r}")


# ---------------------------------------------------------------------------
# keys sub-group
# ---------------------------------------------------------------------------

@cli.group()
def keys() -> None:
    """Signing key management commands."""


@keys.command()
@click.option("--key-file", default="./memshield.key", show_default=True, help="Path to PEM key file")
def rotate(key_file: str) -> None:
    """Rotate the Ed25519 signing key."""
    from memshield.audit.signing import SigningKey

    new_key = SigningKey.rotate(key_file)
    click.echo(f"New key_id: {new_key.key_id}")
