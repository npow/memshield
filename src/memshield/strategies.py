"""Pluggable validation strategies for memory entry analysis."""
from __future__ import annotations

import logging
import math
from typing import Any

from memshield._types import LLMProvider, ValidationResult, Verdict
from memshield.validation import build_validation_prompt, parse_validation_response

logger = logging.getLogger(__name__)


class ConsensusStrategy:
    """Validates entries by asking an LLM to decompose them into factual claim,
    implied action, and legitimacy context — then flagging conflicts.

    Based on A-MemGuard's consensus validation approach.
    """

    def __init__(self, provider: LLMProvider, *, temperature: float = 0.0) -> None:
        self._provider = provider
        self._temperature = temperature

    @property
    def name(self) -> str:
        return "consensus"

    def validate(self, content: str) -> ValidationResult:
        """Send the entry to the LLM for consensus analysis."""
        prompt = build_validation_prompt(content)
        try:
            response = self._provider.generate(prompt, temperature=self._temperature)
        except Exception as e:
            logger.error("ConsensusStrategy LLM call failed: %s", e)
            return ValidationResult(
                verdict=Verdict.AMBIGUOUS,
                confidence=0.0,
                explanation=f"LLM error: {e}",
            )
        return parse_validation_response(response)


class KeywordHeuristicStrategy:
    """Fast, zero-cost heuristic that flags entries containing instruction-like
    patterns commonly found in poisoned memory entries.

    No LLM call. Runs in microseconds. High recall on obvious attacks,
    low recall on subtle ones. Useful as a fast first pass.
    """

    # Patterns that are unusual in factual knowledge but common in injected instructions
    STRONG_SIGNALS = [
        "ignore all previous",
        "ignore previous instructions",
        "disregard all prior",
        "override system",
        "you are now in",
        "maintenance mode",
        "new policy effective immediately",
        "system directive",
        "system update:",
        "critical update:",
        "from now on, always",
        "from now on always",
    ]

    MODERATE_SIGNALS = [
        "always recommend",
        "never mention",
        "always say",
        "never say",
        "do not flag",
        "do not mention",
        "bypass",
        "disable ssl",
        "verify=false",
        "verify = false",
        "--no-verify",
        "--force",
        "push --force",
        "foreign_key_checks=0",
        "without authentication",
        "without validation",
    ]

    def __init__(self, *, strong_weight: float = 0.9, moderate_weight: float = 0.6) -> None:
        self._strong_weight = strong_weight
        self._moderate_weight = moderate_weight

    @property
    def name(self) -> str:
        return "keyword_heuristic"

    def validate(self, content: str) -> ValidationResult:
        """Scan for instruction-like patterns."""
        lower = content.lower()
        strong_hits = [p for p in self.STRONG_SIGNALS if p in lower]
        moderate_hits = [p for p in self.MODERATE_SIGNALS if p in lower]

        if strong_hits:
            return ValidationResult(
                verdict=Verdict.POISONED,
                confidence=self._strong_weight,
                explanation=f"Strong instruction pattern: {strong_hits[0]!r}",
            )

        if len(moderate_hits) >= 2:
            return ValidationResult(
                verdict=Verdict.POISONED,
                confidence=self._moderate_weight,
                explanation=f"Multiple moderate patterns: {', '.join(repr(h) for h in moderate_hits[:3])}",
            )

        if len(moderate_hits) == 1:
            return ValidationResult(
                verdict=Verdict.AMBIGUOUS,
                confidence=0.4,
                explanation=f"Single moderate pattern: {moderate_hits[0]!r}",
            )

        return ValidationResult(
            verdict=Verdict.CLEAN,
            confidence=0.6,
            explanation="No instruction patterns detected",
        )


class EnsembleStrategy:
    """Runs multiple strategies and aggregates their results.

    Supports two modes:
    - "any_poisoned": flag as poisoned if ANY strategy says poisoned (high recall, more false positives)
    - "majority": flag as poisoned if more strategies say poisoned than clean (balanced)
    """

    def __init__(
        self,
        strategies: list[Any],
        *,
        mode: str = "majority",
    ) -> None:
        if not strategies:
            raise ValueError("EnsembleStrategy requires at least one strategy")
        if mode not in ("any_poisoned", "majority"):
            raise ValueError(f"mode must be 'any_poisoned' or 'majority', got {mode!r}")
        self._strategies = strategies
        self._mode = mode

    @property
    def name(self) -> str:
        names = "+".join(s.name for s in self._strategies)
        return f"ensemble({names})"

    def validate(self, content: str) -> ValidationResult:
        """Run all strategies and aggregate."""
        results: list[ValidationResult] = []
        for strategy in self._strategies:
            results.append(strategy.validate(content))

        poisoned_count = sum(1 for r in results if r.verdict == Verdict.POISONED)
        clean_count = sum(1 for r in results if r.verdict == Verdict.CLEAN)
        ambiguous_count = sum(1 for r in results if r.verdict == Verdict.AMBIGUOUS)

        # Collect explanations
        explanations = [
            f"[{self._strategies[i].name}] {r.verdict.value} ({r.confidence:.2f}): {r.explanation}"
            for i, r in enumerate(results)
        ]
        combined_explanation = " | ".join(explanations)

        # Aggregate confidence as weighted average
        total_conf = sum(r.confidence for r in results)
        avg_conf = total_conf / len(results) if results else 0.0

        if self._mode == "any_poisoned":
            if poisoned_count > 0:
                max_poison_conf = max(r.confidence for r in results if r.verdict == Verdict.POISONED)
                return ValidationResult(
                    verdict=Verdict.POISONED,
                    confidence=max_poison_conf,
                    explanation=combined_explanation,
                )
            if ambiguous_count > 0:
                return ValidationResult(
                    verdict=Verdict.AMBIGUOUS,
                    confidence=avg_conf,
                    explanation=combined_explanation,
                )
            return ValidationResult(
                verdict=Verdict.CLEAN,
                confidence=avg_conf,
                explanation=combined_explanation,
            )

        # majority mode
        if poisoned_count > clean_count:
            max_poison_conf = max(r.confidence for r in results if r.verdict == Verdict.POISONED)
            return ValidationResult(
                verdict=Verdict.POISONED,
                confidence=max_poison_conf,
                explanation=combined_explanation,
            )
        if clean_count > poisoned_count:
            max_clean_conf = max(r.confidence for r in results if r.verdict == Verdict.CLEAN)
            return ValidationResult(
                verdict=Verdict.CLEAN,
                confidence=max_clean_conf,
                explanation=combined_explanation,
            )
        # Tie or all ambiguous
        return ValidationResult(
            verdict=Verdict.AMBIGUOUS,
            confidence=avg_conf,
            explanation=combined_explanation,
        )
