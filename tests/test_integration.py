"""Integration tests: MemShield + audit + proxy + CLI wired together."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memshield.audit.config import AuditConfig
from memshield.audit.log import AuditLog
from memshield.proxy import VectorStoreProxy
from memshield.shield import MemShield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audit_config(tmp_path: Path, *, knowledge_base_id: str = "kb-test") -> AuditConfig:
    """Minimal AuditConfig with TSA disabled."""
    return AuditConfig(
        store=str(tmp_path / "audit.db"),
        knowledge_base_id=knowledge_base_id,
        key_file=str(tmp_path / "test.key"),
        tsa_url=None,
    )


def _make_doc(content: str, source: str = "doc-1", chunk_index: int = 0) -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "chunk_index": chunk_index, "id": source}
    return doc


def _clean_strategy() -> MagicMock:
    """Strategy that always returns clean with high confidence."""
    from memshield._types import ValidationResult, Verdict

    strategy = MagicMock()
    strategy.name = "clean_strategy"
    strategy.validate.return_value = ValidationResult(
        verdict=Verdict.CLEAN, confidence=0.99, explanation="clean"
    )
    return strategy


def _poison_strategy() -> MagicMock:
    """Strategy that always returns poisoned with high confidence."""
    from memshield._types import ValidationResult, Verdict

    strategy = MagicMock()
    strategy.name = "poison_strategy"
    strategy.validate.return_value = ValidationResult(
        verdict=Verdict.POISONED, confidence=0.99, explanation="poisoned"
    )
    return strategy


# ---------------------------------------------------------------------------
# 1. MemShield with audit=AuditConfig creates audit_log
# ---------------------------------------------------------------------------


class TestMemShieldAuditInit:
    def test_with_audit_creates_audit_log(self, tmp_path: Path) -> None:
        config = _make_audit_config(tmp_path)
        shield = MemShield(audit=config)
        assert shield.audit_log is not None
        assert isinstance(shield.audit_log, AuditLog)

    def test_without_audit_has_none_audit_log(self) -> None:
        shield = MemShield()
        assert shield.audit_log is None

    def test_audit_config_exported_from_top_level(self) -> None:
        import memshield
        assert hasattr(memshield, "AuditConfig")
        assert hasattr(memshield, "AuditLog")
        assert hasattr(memshield, "AuditRecord")
        from memshield import AuditConfig, AuditLog, AuditRecord  # noqa: F401


# ---------------------------------------------------------------------------
# 3. VectorStoreProxy.similarity_search writes audit record when configured
# ---------------------------------------------------------------------------


class TestProxyAuditIntegration:
    def test_similarity_search_writes_audit_record(self, tmp_path: Path) -> None:
        """similarity_search writes an audit record when audit is configured."""
        audit_cfg = _make_audit_config(tmp_path)
        shield = MemShield(strategy=_clean_strategy(), audit=audit_cfg)

        doc = _make_doc("clean content")
        store = MagicMock()
        store.similarity_search.return_value = [doc]
        store.similarity_search_with_score.return_value = [(doc, 0.95)]

        proxy = VectorStoreProxy(store, shield)
        results = proxy.similarity_search("what is X?")

        assert len(results) == 1
        assert results[0] is doc

        # An audit record should have been written
        record = shield.audit_log.last_record()  # type: ignore[union-attr]
        assert record is not None
        assert len(record.retrieved) == 1
        assert len(record.blocked) == 0

    def test_similarity_search_filters_blocked_docs(self, tmp_path: Path) -> None:
        """similarity_search returns only clean docs — blocked are filtered out."""
        audit_cfg = _make_audit_config(tmp_path)
        shield = MemShield(strategy=_poison_strategy(), audit=audit_cfg)

        doc = _make_doc("poisoned content")
        store = MagicMock()
        store.similarity_search.return_value = [doc]
        store.similarity_search_with_score.return_value = [(doc, 0.95)]

        proxy = VectorStoreProxy(store, shield)
        results = proxy.similarity_search("query")

        assert results == []

        record = shield.audit_log.last_record()  # type: ignore[union-attr]
        assert record is not None
        assert len(record.retrieved) == 0
        assert len(record.blocked) == 1

    def test_similarity_search_persists_inference_id(self, tmp_path: Path) -> None:
        """inference_id kwarg is persisted in the audit record."""
        audit_cfg = _make_audit_config(tmp_path)
        shield = MemShield(strategy=_clean_strategy(), audit=audit_cfg)

        doc = _make_doc("content")
        store = MagicMock()
        store.similarity_search.return_value = [doc]
        store.similarity_search_with_score.return_value = [(doc, 0.8)]

        proxy = VectorStoreProxy(store, shield)
        proxy.similarity_search("my query", inference_id="fixed-id-xyz")

        record = shield.audit_log.get_record("fixed-id-xyz")  # type: ignore[union-attr]
        assert record is not None
        assert record.inference_id == "fixed-id-xyz"

    def test_similarity_search_without_audit_still_works(self) -> None:
        """Without audit, similarity_search still returns validated docs."""
        shield = MemShield(strategy=_clean_strategy())

        doc = _make_doc("safe content")
        store = MagicMock()
        store.similarity_search.return_value = [doc]

        proxy = VectorStoreProxy(store, shield)
        results = proxy.similarity_search("query", inference_id="ignored", user_id="u1")

        assert len(results) == 1
        assert results[0] is doc
        assert shield.audit_log is None

    def test_similarity_search_with_score_writes_audit_with_scores(
        self, tmp_path: Path
    ) -> None:
        """similarity_search_with_score writes audit record with correct scores."""
        audit_cfg = _make_audit_config(tmp_path)
        shield = MemShield(strategy=_clean_strategy(), audit=audit_cfg)

        doc = _make_doc("some content", source="src-42", chunk_index=3)
        store = MagicMock()
        store.similarity_search_with_score.return_value = [(doc, 0.87)]

        proxy = VectorStoreProxy(store, shield)
        scored_results = proxy.similarity_search_with_score("query")

        assert len(scored_results) == 1
        assert scored_results[0][0] is doc
        assert scored_results[0][1] == pytest.approx(0.87)

        record = shield.audit_log.last_record()  # type: ignore[union-attr]
        assert record is not None
        assert len(record.retrieved) == 1
        assert record.retrieved[0].score == pytest.approx(0.87)
        assert record.retrieved[0].chunk_index == 3

    def test_similarity_search_with_score_filters_blocked(self, tmp_path: Path) -> None:
        """similarity_search_with_score filters blocked docs from scored results."""
        audit_cfg = _make_audit_config(tmp_path)
        shield = MemShield(strategy=_poison_strategy(), audit=audit_cfg)

        doc = _make_doc("bad content")
        store = MagicMock()
        store.similarity_search_with_score.return_value = [(doc, 0.9)]

        proxy = VectorStoreProxy(store, shield)
        results = proxy.similarity_search_with_score("q")

        assert results == []


# ---------------------------------------------------------------------------
# 7. AuditConfig exported from top-level memshield package
# ---------------------------------------------------------------------------


def test_audit_exports_at_top_level() -> None:
    import memshield
    assert "AuditConfig" in memshield.__all__
    assert "AuditLog" in memshield.__all__
    assert "AuditRecord" in memshield.__all__


# ---------------------------------------------------------------------------
# CLI tests (8–12)
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the memshield CLI."""

    def _make_log(self, tmp_path: Path) -> tuple[AuditLog, AuditConfig]:
        cfg = _make_audit_config(tmp_path)
        log = AuditLog(cfg)
        return log, cfg

    def test_verify_valid_chain(self, tmp_path: Path) -> None:
        """CLI verify reports VALID on an intact chain."""
        from click.testing import CliRunner
        from memshield.cli import cli

        log, cfg = self._make_log(tmp_path)
        log.write(
            query="test",
            retrieved=[
                {
                    "doc_id": "d1",
                    "chunk_index": 0,
                    "content": "some content",
                    "score": 0.9,
                    "verdict": "clean",
                    "trust_level": "unverified",
                }
            ],
            blocked=[],
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["audit", "verify", "--db", cfg.store, "--key-file", cfg.key_file],
        )
        assert result.exit_code == 0
        assert "VALID" in result.output

    def test_verify_invalid_chain(self, tmp_path: Path) -> None:
        """CLI verify exits non-zero and reports INVALID on a tampered chain."""
        import sqlite3
        from click.testing import CliRunner
        from memshield.cli import cli

        log, cfg = self._make_log(tmp_path)
        r = log.write(query="q", retrieved=[], blocked=[])

        # Tamper with the chain
        conn = sqlite3.connect(cfg.store)
        conn.execute(
            "UPDATE audit_records SET query_hash = 'deadbeef' WHERE inference_id = ?",
            (r.inference_id,),
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["audit", "verify", "--db", cfg.store, "--key-file", cfg.key_file],
        )
        assert result.exit_code != 0
        assert "INVALID" in result.output

    def test_inspect_outputs_json(self, tmp_path: Path) -> None:
        """CLI inspect outputs valid JSON for a known record."""
        from click.testing import CliRunner
        from memshield.cli import cli

        log, cfg = self._make_log(tmp_path)
        record = log.write(query="inspect me", retrieved=[], blocked=[])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "audit",
                "inspect",
                "--db",
                cfg.store,
                "--key-file",
                cfg.key_file,
                "--inference-id",
                record.inference_id,
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["inference_id"] == record.inference_id

    def test_inspect_missing_record_exits_nonzero(self, tmp_path: Path) -> None:
        """CLI inspect exits non-zero for an unknown inference ID."""
        from click.testing import CliRunner
        from memshield.cli import cli

        _, cfg = self._make_log(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "audit",
                "inspect",
                "--db",
                cfg.store,
                "--key-file",
                cfg.key_file,
                "--inference-id",
                "no-such-id",
            ],
        )
        assert result.exit_code != 0

    def test_export_jsonl(self, tmp_path: Path) -> None:
        """CLI export outputs JSONL (one JSON object per line) by default."""
        from click.testing import CliRunner
        from memshield.cli import cli

        log, cfg = self._make_log(tmp_path)
        log.write(query="q1", retrieved=[], blocked=[])
        log.write(query="q2", retrieved=[], blocked=[])

        runner = CliRunner()
        result = runner.invoke(
            cli, ["audit", "export", "--db", cfg.store, "--key-file", cfg.key_file]
        )
        assert result.exit_code == 0
        lines = [line for line in result.output.strip().splitlines() if line]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "inference_id" in obj

    def test_export_json_format(self, tmp_path: Path) -> None:
        """CLI export with --format json outputs a JSON array."""
        from click.testing import CliRunner
        from memshield.cli import cli

        log, cfg = self._make_log(tmp_path)
        log.write(query="q", retrieved=[], blocked=[])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "audit",
                "export",
                "--db",
                cfg.store,
                "--key-file",
                cfg.key_file,
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        arr = json.loads(result.output)
        assert isinstance(arr, list)
        assert len(arr) == 1

    def test_erase_user_prints_count(self, tmp_path: Path) -> None:
        """CLI erase-user prints the count of affected records."""
        from click.testing import CliRunner
        from memshield.cli import cli

        # Create a log with PII encryption so key_store is meaningful
        cfg = AuditConfig(
            store=str(tmp_path / "audit.db"),
            knowledge_base_id="kb-erase",
            key_file=str(tmp_path / "test.key"),
            key_store=str(tmp_path / "keys.db"),
            pii_fields=["query"],
            tsa_url=None,
        )
        log = AuditLog(cfg)
        log.write(query="private", user_id="alice", retrieved=[], blocked=[])
        log.write(query="also private", user_id="alice", retrieved=[], blocked=[])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "audit",
                "erase-user",
                "--db",
                cfg.store,
                "--key-store",
                cfg.key_store,
                "--user-id",
                "alice",
                "--knowledge-base-id",
                "kb-erase",
            ],
        )
        assert result.exit_code == 0
        assert "2" in result.output

    def test_keys_rotate_creates_backup(self, tmp_path: Path) -> None:
        """CLI keys rotate archives old key and prints new key_id."""
        from click.testing import CliRunner
        from memshield.audit.signing import SigningKey
        from memshield.cli import cli

        key_file = str(tmp_path / "memshield.key")
        # Create an initial key
        sk1 = SigningKey(key_file)
        original_key_id = sk1.key_id

        runner = CliRunner()
        result = runner.invoke(cli, ["keys", "rotate", "--key-file", key_file])
        assert result.exit_code == 0
        assert "key_id" in result.output.lower()

        # Backup should exist
        backups = list(tmp_path.glob("memshield.key.*.bak"))
        assert len(backups) == 1

        # New key should have a different key_id
        new_key_id = result.output.strip().split(":")[-1].strip()
        assert new_key_id != original_key_id
