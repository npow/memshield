"""MemShield — production-grade memory integrity defense for AI agents."""

from memshield._types import (
    DriftAlert,
    FailurePolicy,
    LLMProvider,
    ProvenanceRecord,
    ShieldConfig,
    TrustLevel,
    ValidationResult,
    ValidationStrategy,
    Verdict,
)
from memshield.audit import AuditConfig, AuditLog, AuditRecord
from memshield.strategies import (
    ConsensusStrategy,
    EnsembleStrategy,
    KeywordHeuristicStrategy,
)

__all__ = [
    "AuditConfig",
    "AuditLog",
    "AuditRecord",
    "ConsensusStrategy",
    "DriftAlert",
    "EnsembleStrategy",
    "FailurePolicy",
    "KeywordHeuristicStrategy",
    "LLMProvider",
    "MemShield",
    "ProvenanceRecord",
    "ShieldConfig",
    "TrustLevel",
    "ValidationResult",
    "ValidationStrategy",
    "Verdict",
]

__version__ = "0.2.0"


def __getattr__(name: str) -> object:
    """Lazy import for MemShield to avoid circular imports during module setup."""
    if name == "MemShield":
        from memshield.shield import MemShield
        return MemShield
    raise AttributeError(f"module 'memshield' has no attribute {name!r}")
