"""Tests for memshield._types."""
from __future__ import annotations

import pytest

from memshield._types import (
    DriftAlert,
    FailurePolicy,
    LLMProvider,
    ProvenanceRecord,
    ShieldConfig,
    TrustLevel,
    ValidationResult,
    Verdict,
)


class TestVerdict:
    """Tests for Verdict enum."""

    def test_values(self) -> None:
        """All expected verdict values exist."""
        assert Verdict.CLEAN.value == "clean"
        assert Verdict.POISONED.value == "poisoned"
        assert Verdict.AMBIGUOUS.value == "ambiguous"


class TestTrustLevel:
    """Tests for TrustLevel enum."""

    def test_values(self) -> None:
        """All expected trust level values exist."""
        assert TrustLevel.VERIFIED.value == "verified"
        assert TrustLevel.UNVERIFIED.value == "unverified"
        assert TrustLevel.UNTRUSTED.value == "untrusted"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_with_defaults(self) -> None:
        """Constructor works with required fields only."""
        result = ValidationResult(verdict=Verdict.CLEAN, confidence=0.95)
        assert result.verdict == Verdict.CLEAN
        assert result.confidence == 0.95
        assert result.explanation == ""
        assert result.escalated is False

    def test_create_with_all_fields(self) -> None:
        """Constructor accepts all fields."""
        result = ValidationResult(
            verdict=Verdict.POISONED,
            confidence=0.94,
            explanation="Contains embedded instructions",
            escalated=True,
        )
        assert result.explanation == "Contains embedded instructions"
        assert result.escalated is True

    def test_frozen(self) -> None:
        """Result is immutable."""
        result = ValidationResult(verdict=Verdict.CLEAN, confidence=0.9)
        with pytest.raises(AttributeError):
            result.confidence = 0.5  # type: ignore[misc]

    def test_confidence_below_zero(self) -> None:
        """Confidence below 0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between"):
            ValidationResult(verdict=Verdict.CLEAN, confidence=-0.1)

    def test_confidence_above_one(self) -> None:
        """Confidence above 1 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between"):
            ValidationResult(verdict=Verdict.CLEAN, confidence=1.1)

    def test_confidence_boundary_zero(self) -> None:
        """Confidence of exactly 0 is valid."""
        result = ValidationResult(verdict=Verdict.AMBIGUOUS, confidence=0.0)
        assert result.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        """Confidence of exactly 1 is valid."""
        result = ValidationResult(verdict=Verdict.CLEAN, confidence=1.0)
        assert result.confidence == 1.0


class TestProvenanceRecord:
    """Tests for ProvenanceRecord dataclass."""

    def test_create(self) -> None:
        """Constructor works with all required fields."""
        record = ProvenanceRecord(
            source="user_input",
            timestamp=1700000000.0,
            trust_level=TrustLevel.VERIFIED,
            previous_hash="abc123",
            entry_hash="def456",
        )
        assert record.source == "user_input"
        assert record.metadata == {}

    def test_create_with_metadata(self) -> None:
        """Constructor accepts optional metadata."""
        record = ProvenanceRecord(
            source="web_search",
            timestamp=1700000000.0,
            trust_level=TrustLevel.UNTRUSTED,
            previous_hash="abc",
            entry_hash="def",
            metadata={"tool": "search", "url": "https://example.com"},
        )
        assert record.metadata["tool"] == "search"

    def test_frozen(self) -> None:
        """Record is immutable."""
        record = ProvenanceRecord(
            source="test",
            timestamp=0.0,
            trust_level=TrustLevel.UNVERIFIED,
            previous_hash="",
            entry_hash="",
        )
        with pytest.raises(AttributeError):
            record.source = "changed"  # type: ignore[misc]


class TestShieldConfig:
    """Tests for ShieldConfig dataclass."""

    def test_defaults(self) -> None:
        """Default configuration values are set correctly."""
        config = ShieldConfig()
        assert config.confidence_threshold == 0.7
        assert config.untrusted_confidence_boost == 0.15
        assert config.failure_policy == FailurePolicy.ALLOW_WITH_WARNING
        assert config.enable_provenance is True
        assert config.enable_drift_detection is True

    def test_custom_values(self) -> None:
        """Custom configuration values are accepted."""
        config = ShieldConfig(
            confidence_threshold=0.8,
            untrusted_confidence_boost=0.2,
            failure_policy=FailurePolicy.BLOCK,
        )
        assert config.confidence_threshold == 0.8
        assert config.failure_policy == FailurePolicy.BLOCK

    def test_invalid_confidence_threshold(self) -> None:
        """Invalid confidence threshold raises ValueError."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            ShieldConfig(confidence_threshold=1.5)

    def test_invalid_untrusted_boost(self) -> None:
        """Invalid untrusted confidence boost raises ValueError."""
        with pytest.raises(ValueError, match="untrusted_confidence_boost"):
            ShieldConfig(untrusted_confidence_boost=-0.1)


class TestLLMProviderProtocol:
    """Tests for LLMProvider protocol."""

    def test_class_with_generate_satisfies_protocol(self) -> None:
        """A class with generate() method satisfies the protocol."""

        class MyProvider:
            def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
                return "response"

        assert isinstance(MyProvider(), LLMProvider)

    def test_class_without_generate_fails_protocol(self) -> None:
        """A class without generate() does not satisfy the protocol."""

        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), LLMProvider)


class TestDriftAlert:
    """Tests for DriftAlert dataclass."""

    def test_create(self) -> None:
        """Constructor works with all fields."""
        alert = DriftAlert(
            metric="category_distribution",
            baseline_value=0.3,
            current_value=0.8,
            deviation=1.67,
            message="Unusual access pattern detected",
        )
        assert alert.metric == "category_distribution"
        assert alert.deviation == 1.67
