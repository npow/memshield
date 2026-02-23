"""Tests for memshield._internal.hash."""
from __future__ import annotations

from memshield._internal.hash import (
    GENESIS_HASH,
    compute_chain_hash,
    compute_entry_hash,
    verify_chain_link,
)


class TestComputeEntryHash:
    """Tests for compute_entry_hash."""

    def test_deterministic(self) -> None:
        """Same content produces same hash."""
        h1 = compute_entry_hash("hello world")
        h2 = compute_entry_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        """Different content produces different hash."""
        h1 = compute_entry_hash("hello world")
        h2 = compute_entry_hash("goodbye world")
        assert h1 != h2

    def test_with_metadata(self) -> None:
        """Metadata is included in the hash."""
        h1 = compute_entry_hash("hello", metadata={"source": "user"})
        h2 = compute_entry_hash("hello", metadata={"source": "web"})
        assert h1 != h2

    def test_without_metadata(self) -> None:
        """Works without metadata."""
        h = compute_entry_hash("hello")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_empty_content(self) -> None:
        """Works with empty string."""
        h = compute_entry_hash("")
        assert isinstance(h, str)
        assert len(h) == 64


class TestComputeChainHash:
    """Tests for compute_chain_hash."""

    def test_deterministic(self) -> None:
        """Same inputs produce same chain hash."""
        h1 = compute_chain_hash("entry_hash", "previous_hash")
        h2 = compute_chain_hash("entry_hash", "previous_hash")
        assert h1 == h2

    def test_order_matters(self) -> None:
        """Swapping entry and previous hash produces different result."""
        h1 = compute_chain_hash("aaa", "bbb")
        h2 = compute_chain_hash("bbb", "aaa")
        assert h1 != h2

    def test_with_genesis(self) -> None:
        """Works with genesis hash as previous."""
        h = compute_chain_hash("first_entry", GENESIS_HASH)
        assert isinstance(h, str)
        assert len(h) == 64


class TestVerifyChainLink:
    """Tests for verify_chain_link."""

    def test_valid_link(self) -> None:
        """Valid chain link verifies correctly."""
        entry_hash = compute_entry_hash("test content")
        chain_hash = compute_chain_hash(entry_hash, GENESIS_HASH)
        assert verify_chain_link(entry_hash, GENESIS_HASH, chain_hash) is True

    def test_invalid_link(self) -> None:
        """Tampered chain link fails verification."""
        entry_hash = compute_entry_hash("test content")
        assert verify_chain_link(entry_hash, GENESIS_HASH, "tampered_hash") is False

    def test_wrong_previous_hash(self) -> None:
        """Wrong previous hash fails verification."""
        entry_hash = compute_entry_hash("test content")
        chain_hash = compute_chain_hash(entry_hash, GENESIS_HASH)
        assert verify_chain_link(entry_hash, "wrong_previous", chain_hash) is False

    def test_multi_link_chain(self) -> None:
        """A chain of multiple links verifies correctly."""
        # Entry 1
        e1_hash = compute_entry_hash("entry 1")
        c1_hash = compute_chain_hash(e1_hash, GENESIS_HASH)
        # Entry 2
        e2_hash = compute_entry_hash("entry 2")
        c2_hash = compute_chain_hash(e2_hash, c1_hash)
        # Entry 3
        e3_hash = compute_entry_hash("entry 3")
        c3_hash = compute_chain_hash(e3_hash, c2_hash)
        # Verify all links
        assert verify_chain_link(e1_hash, GENESIS_HASH, c1_hash)
        assert verify_chain_link(e2_hash, c1_hash, c2_hash)
        assert verify_chain_link(e3_hash, c2_hash, c3_hash)
