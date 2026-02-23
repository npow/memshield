"""Main MemShield class — orchestrates all defense layers."""
from __future__ import annotations

import logging
from typing import Any, TypeVar

from memshield._internal.drift import DriftDetector
from memshield._types import (
    FailurePolicy,
    LLMProvider,
    ShieldConfig,
    TrustLevel,
    ValidationResult,
    Verdict,
)
from memshield.provenance import ProvenanceTracker
from memshield.proxy import VectorStoreProxy
from memshield.validation import validate_entry

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MemShield:
    """Production-grade memory integrity defense for AI agents.

    Wraps a vector store to validate reads and track provenance on writes.
    """

    def __init__(
        self,
        local_provider: LLMProvider | None = None,
        cloud_provider: LLMProvider | None = None,
        config: ShieldConfig | None = None,
    ) -> None:
        self._config = config or ShieldConfig()
        self._local_provider = local_provider
        self._cloud_provider = cloud_provider
        self._provenance = ProvenanceTracker()
        self._drift = DriftDetector()
        self._stats = _ShieldStats()

    @property
    def config(self) -> ShieldConfig:
        """Current shield configuration."""
        return self._config

    @property
    def provenance(self) -> ProvenanceTracker:
        """The provenance tracker."""
        return self._provenance

    @property
    def drift_detector(self) -> DriftDetector:
        """The drift detector."""
        return self._drift

    @property
    def stats(self) -> _ShieldStats:
        """Operational statistics."""
        return self._stats

    def wrap(self, store: T) -> T:
        """Wrap a vector store with memory integrity defense.

        Returns a proxy with the same interface as the original store.
        """
        return VectorStoreProxy(store, self)  # type: ignore[return-value]

    def validate_reads(self, documents: list[Any]) -> list[Any]:
        """Validate a list of retrieved documents, filtering out poisoned entries."""
        validated: list[Any] = []

        for doc in documents:
            content = self._extract_content(doc)
            category = self._extract_category(doc)

            # Record access for drift detection
            if self._config.enable_drift_detection:
                self._drift.record_access(content, category)

            # Check drift
            if self._config.enable_drift_detection:
                drift_alerts = self._drift.check_drift(content, category)
                for alert in drift_alerts:
                    logger.warning("Drift alert: %s", alert.message)

            # Check provenance trust level
            trust = TrustLevel.UNVERIFIED
            if self._config.enable_provenance:
                trust = self._provenance.get_trust_level(content)

            # Validate with local LLM
            result = self._validate_with_local(content, trust)

            # Escalate if ambiguous and cloud provider available
            if result.verdict == Verdict.AMBIGUOUS and self._cloud_provider is not None:
                result = self._validate_with_cloud(content)

            # Apply policy
            if result.verdict == Verdict.CLEAN:
                self._stats.clean += 1
                validated.append(doc)
            elif result.verdict == Verdict.POISONED:
                self._stats.blocked += 1
                logger.warning(
                    "Blocked poisoned memory entry (confidence=%.2f): %s",
                    result.confidence,
                    result.explanation,
                )
            else:
                # Ambiguous — apply failure policy
                self._stats.ambiguous += 1
                if self._config.failure_policy == FailurePolicy.BLOCK:
                    logger.warning("Blocking ambiguous entry per policy: %s", result.explanation)
                elif self._config.failure_policy == FailurePolicy.ALLOW_WITH_WARNING:
                    logger.warning("Allowing ambiguous entry with warning: %s", result.explanation)
                    validated.append(doc)
                elif self._config.failure_policy == FailurePolicy.ALLOW_WITH_REVIEW:
                    logger.warning("Allowing ambiguous entry for review: %s", result.explanation)
                    validated.append(doc)

        return validated

    def tag_provenance(self, documents: list[Any]) -> None:
        """Tag documents with provenance metadata."""
        if not self._config.enable_provenance:
            return
        for doc in documents:
            content = self._extract_content(doc)
            source = self._extract_source(doc)
            metadata = self._extract_metadata(doc)
            self._provenance.record_write(content, source, metadata)

    def tag_provenance_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Tag text entries with provenance metadata."""
        if not self._config.enable_provenance:
            return
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            source = meta.get("source", "unknown")
            self._provenance.record_write(text, source, meta)

    def _validate_with_local(self, content: str, trust: TrustLevel) -> ValidationResult:
        """Validate using the local LLM provider."""
        if self._local_provider is None:
            # No local provider — escalate to cloud or return ambiguous
            if self._cloud_provider is not None:
                return self._validate_with_cloud(content)
            return ValidationResult(
                verdict=Verdict.AMBIGUOUS,
                confidence=0.0,
                explanation="No LLM provider configured",
            )

        result = validate_entry(content, self._local_provider)
        self._stats.local_calls += 1

        # Lower confidence threshold for untrusted entries
        threshold = self._config.confidence_threshold
        if trust == TrustLevel.UNTRUSTED:
            threshold = min(threshold + self._config.untrusted_confidence_boost, 0.95)

        # Determine if we can resolve locally
        if result.verdict == Verdict.CLEAN and result.confidence >= threshold:
            return result
        if result.verdict == Verdict.POISONED and result.confidence >= threshold:
            return result

        # Below threshold — mark as ambiguous for potential escalation
        return ValidationResult(
            verdict=Verdict.AMBIGUOUS,
            confidence=result.confidence,
            explanation=result.explanation,
        )

    def _validate_with_cloud(self, content: str) -> ValidationResult:
        """Validate using the cloud LLM provider."""
        if self._cloud_provider is None:
            return ValidationResult(
                verdict=Verdict.AMBIGUOUS,
                confidence=0.0,
                explanation="No cloud provider configured",
            )
        result = validate_entry(content, self._cloud_provider)
        self._stats.cloud_calls += 1
        return ValidationResult(
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=result.explanation,
            escalated=True,
        )

    @staticmethod
    def _extract_content(doc: Any) -> str:
        """Extract text content from a document object."""
        if hasattr(doc, "page_content"):
            return doc.page_content
        if isinstance(doc, dict):
            return doc.get("content", doc.get("text", str(doc)))
        return str(doc)

    @staticmethod
    def _extract_category(doc: Any) -> str:
        """Extract category from a document's metadata."""
        metadata = getattr(doc, "metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("category", "default")
        return "default"

    @staticmethod
    def _extract_source(doc: Any) -> str:
        """Extract source from a document's metadata."""
        metadata = getattr(doc, "metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("source", "unknown")
        return "unknown"

    @staticmethod
    def _extract_metadata(doc: Any) -> dict[str, Any]:
        """Extract metadata dict from a document."""
        metadata = getattr(doc, "metadata", {})
        if isinstance(metadata, dict):
            return metadata
        return {}


class _ShieldStats:
    """Operational statistics for a MemShield instance."""

    def __init__(self) -> None:
        self.clean: int = 0
        self.blocked: int = 0
        self.ambiguous: int = 0
        self.local_calls: int = 0
        self.cloud_calls: int = 0

    @property
    def total_validated(self) -> int:
        """Total number of documents validated."""
        return self.clean + self.blocked + self.ambiguous
