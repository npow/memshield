"""Shared types, protocols, and dataclasses for MemShield."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Verdict(Enum):
    """Result of validating a memory entry."""
    CLEAN = "clean"
    POISONED = "poisoned"
    AMBIGUOUS = "ambiguous"


class TrustLevel(Enum):
    """Trust level assigned based on a memory entry's provenance."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    UNTRUSTED = "untrusted"


class FailurePolicy(Enum):
    """What to do when the cloud LLM is unavailable for ambiguous entries."""
    BLOCK = "block"
    ALLOW_WITH_WARNING = "allow_with_warning"
    ALLOW_WITH_REVIEW = "allow_with_review"


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a single memory entry."""
    verdict: Verdict
    confidence: float
    explanation: str = ""
    escalated: bool = False

    def __post_init__(self) -> None:
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable record of a memory write operation."""
    source: str
    timestamp: float
    trust_level: TrustLevel
    previous_hash: str
    entry_hash: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ShieldConfig:
    """Configuration for a MemShield instance.

    confidence_threshold: Minimum confidence from the local LLM to resolve
        an entry locally. Below this, the entry is marked ambiguous and
        escalated to the cloud LLM (if configured). This value is a
        PLACEHOLDER (0.7) — it must be calibrated empirically by running
        the benchmark against a labeled dataset of poisoned/clean entries.
    untrusted_confidence_boost: Extra confidence required for entries from
        untrusted sources. Applied on top of confidence_threshold.
    failure_policy: What to do with ambiguous entries when no cloud LLM
        is available or the cloud also returns ambiguous.
    """
    confidence_threshold: float = 0.7
    untrusted_confidence_boost: float = 0.15
    failure_policy: FailurePolicy = FailurePolicy.ALLOW_WITH_WARNING
    enable_provenance: bool = True
    enable_drift_detection: bool = True

    def __post_init__(self) -> None:
        """Validate thresholds."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )
        if not 0.0 <= self.untrusted_confidence_boost <= 1.0:
            raise ValueError(
                f"untrusted_confidence_boost must be between 0.0 and 1.0, "
                f"got {self.untrusted_confidence_boost}"
            )


@runtime_checkable
class LLMProvider(Protocol):
    """Any LLM that can generate text from a prompt."""

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Generate a completion. Returns the text content."""
        ...


@runtime_checkable
class ValidationStrategy(Protocol):
    """A pluggable strategy for validating memory entries."""

    def validate(self, content: str) -> ValidationResult:
        """Validate a single memory entry. Returns a ValidationResult."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name of this strategy."""
        ...


@dataclass(frozen=True)
class DriftAlert:
    """Alert generated when memory access patterns deviate from baseline."""
    metric: str
    baseline_value: float
    current_value: float
    deviation: float
    message: str
