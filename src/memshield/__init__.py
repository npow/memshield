"""MemShield — production-grade memory integrity defense for AI agents."""

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

__all__ = [
    "DriftAlert",
    "FailurePolicy",
    "LLMProvider",
    "MemShield",
    "ProvenanceRecord",
    "ShieldConfig",
    "TrustLevel",
    "ValidationResult",
    "Verdict",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    """Lazy import for MemShield to avoid circular imports during module setup."""
    if name == "MemShield":
        from memshield.shield import MemShield
        return MemShield
    raise AttributeError(f"module 'memshield' has no attribute {name!r}")
