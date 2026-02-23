"""Tests for memshield.provenance."""
from __future__ import annotations

from memshield._types import TrustLevel
from memshield.provenance import ProvenanceTracker


class TestProvenanceTracker:
    """Tests for ProvenanceTracker."""

    def test_empty_chain(self) -> None:
        """New tracker has empty chain."""
        tracker = ProvenanceTracker()
        assert tracker.chain_length == 0

    def test_record_write(self) -> None:
        """Recording a write adds to the chain."""
        tracker = ProvenanceTracker()
        record = tracker.record_write("hello world", source="user_input")
        assert tracker.chain_length == 1
        assert record.source == "user_input"
        assert record.trust_level == TrustLevel.VERIFIED

    def test_chain_grows(self) -> None:
        """Multiple writes grow the chain."""
        tracker = ProvenanceTracker()
        tracker.record_write("entry 1", source="user_input")
        tracker.record_write("entry 2", source="web_search")
        assert tracker.chain_length == 2

    def test_hash_chain_links(self) -> None:
        """Each record's previous_hash links to the prior chain state."""
        tracker = ProvenanceTracker()
        r1 = tracker.record_write("entry 1", source="user_input")
        r2 = tracker.record_write("entry 2", source="user_input")
        # r2's previous_hash should not equal r1's previous_hash
        # (they link to different chain states)
        assert r2.previous_hash != r1.previous_hash

    def test_verify_chain_empty(self) -> None:
        """Empty chain is valid."""
        tracker = ProvenanceTracker()
        assert tracker.verify_chain() is True

    def test_verify_chain_valid(self) -> None:
        """Normal chain verifies correctly."""
        tracker = ProvenanceTracker()
        tracker.record_write("entry 1", source="user_input")
        tracker.record_write("entry 2", source="web_search")
        tracker.record_write("entry 3", source="system")
        assert tracker.verify_chain() is True

    def test_trust_level_user_input(self) -> None:
        """User input is VERIFIED."""
        tracker = ProvenanceTracker()
        record = tracker.record_write("content", source="user_input")
        assert record.trust_level == TrustLevel.VERIFIED

    def test_trust_level_system(self) -> None:
        """System source is VERIFIED."""
        tracker = ProvenanceTracker()
        record = tracker.record_write("content", source="system")
        assert record.trust_level == TrustLevel.VERIFIED

    def test_trust_level_tool(self) -> None:
        """Tool sources are UNVERIFIED."""
        tracker = ProvenanceTracker()
        record = tracker.record_write("content", source="tool:web_search")
        assert record.trust_level == TrustLevel.UNVERIFIED

    def test_trust_level_external(self) -> None:
        """External/unknown sources are UNTRUSTED."""
        tracker = ProvenanceTracker()
        record = tracker.record_write("content", source="web_scrape")
        assert record.trust_level == TrustLevel.UNTRUSTED

    def test_get_trust_level_found(self) -> None:
        """Looks up trust level by content hash."""
        tracker = ProvenanceTracker()
        tracker.record_write("specific content", source="user_input")
        trust = tracker.get_trust_level("specific content")
        assert trust == TrustLevel.VERIFIED

    def test_get_trust_level_not_found(self) -> None:
        """Returns UNVERIFIED for unknown content."""
        tracker = ProvenanceTracker()
        trust = tracker.get_trust_level("never written")
        assert trust == TrustLevel.UNVERIFIED

    def test_get_record_by_index(self) -> None:
        """Records are retrievable by index."""
        tracker = ProvenanceTracker()
        tracker.record_write("first", source="user_input")
        tracker.record_write("second", source="web_search")
        r0 = tracker.get_record(0)
        r1 = tracker.get_record(1)
        assert r0.source == "user_input"
        assert r1.source == "web_search"

    def test_metadata_preserved(self) -> None:
        """Metadata passed to record_write is preserved."""
        tracker = ProvenanceTracker()
        record = tracker.record_write(
            "content",
            source="tool:search",
            metadata={"url": "https://example.com"},
        )
        assert record.metadata["url"] == "https://example.com"
