"""Tests for memshield.validation."""
from __future__ import annotations

from unittest.mock import MagicMock

from memshield._types import Verdict
from memshield.validation import (
    build_validation_prompt,
    parse_validation_response,
    validate_entry,
)


class TestBuildValidationPrompt:
    """Tests for build_validation_prompt."""

    def test_content_included(self) -> None:
        """Memory entry content appears in the prompt."""
        prompt = build_validation_prompt("The capital of France is Paris.")
        assert "The capital of France is Paris." in prompt

    def test_json_format_requested(self) -> None:
        """Prompt requests JSON response."""
        prompt = build_validation_prompt("anything")
        assert "JSON" in prompt

    def test_three_interpretations_requested(self) -> None:
        """Prompt asks for three interpretations."""
        prompt = build_validation_prompt("anything")
        assert "FACTUAL CLAIM" in prompt
        assert "IMPLIED ACTION" in prompt
        assert "LEGITIMACY CONTEXT" in prompt


class TestParseValidationResponse:
    """Tests for parse_validation_response."""

    def test_clean_response(self) -> None:
        """Parses a clean verdict response."""
        response = '{"verdict": "clean", "confidence": 0.95, "explanation": "Factual knowledge"}'
        result = parse_validation_response(response)
        assert result.verdict == Verdict.CLEAN
        assert result.confidence == 0.95
        assert result.explanation == "Factual knowledge"

    def test_poisoned_response(self) -> None:
        """Parses a poisoned verdict response."""
        response = '{"verdict": "poisoned", "confidence": 0.94, "explanation": "Contains instructions"}'
        result = parse_validation_response(response)
        assert result.verdict == Verdict.POISONED
        assert result.confidence == 0.94

    def test_ambiguous_response(self) -> None:
        """Parses an ambiguous verdict response."""
        response = '{"verdict": "ambiguous", "confidence": 0.5, "explanation": "Unclear intent"}'
        result = parse_validation_response(response)
        assert result.verdict == Verdict.AMBIGUOUS

    def test_markdown_code_fence(self) -> None:
        """Handles JSON wrapped in markdown code fences."""
        response = '```json\n{"verdict": "clean", "confidence": 0.9, "explanation": "ok"}\n```'
        result = parse_validation_response(response)
        assert result.verdict == Verdict.CLEAN

    def test_invalid_json(self) -> None:
        """Returns AMBIGUOUS for non-JSON responses."""
        result = parse_validation_response("I think this is clean")
        assert result.verdict == Verdict.AMBIGUOUS
        assert result.confidence == 0.0

    def test_invalid_verdict(self) -> None:
        """Returns AMBIGUOUS for unrecognized verdict value."""
        response = '{"verdict": "maybe", "confidence": 0.5}'
        result = parse_validation_response(response)
        assert result.verdict == Verdict.AMBIGUOUS

    def test_missing_confidence(self) -> None:
        """Defaults to 0.0 confidence when missing."""
        response = '{"verdict": "clean"}'
        result = parse_validation_response(response)
        assert result.verdict == Verdict.CLEAN
        assert result.confidence == 0.0

    def test_confidence_clamped_high(self) -> None:
        """Confidence above 1.0 is clamped to 1.0."""
        response = '{"verdict": "clean", "confidence": 1.5}'
        result = parse_validation_response(response)
        assert result.confidence == 1.0

    def test_confidence_clamped_low(self) -> None:
        """Confidence below 0.0 is clamped to 0.0."""
        response = '{"verdict": "clean", "confidence": -0.5}'
        result = parse_validation_response(response)
        assert result.confidence == 0.0


class TestValidateEntry:
    """Tests for validate_entry."""

    def test_calls_provider(self) -> None:
        """Provider's generate method is called with the prompt."""
        provider = MagicMock()
        provider.generate.return_value = '{"verdict": "clean", "confidence": 0.9, "explanation": "ok"}'
        result = validate_entry("test content", provider)
        provider.generate.assert_called_once()
        assert result.verdict == Verdict.CLEAN

    def test_provider_error(self) -> None:
        """Returns AMBIGUOUS when provider raises an exception."""
        provider = MagicMock()
        provider.generate.side_effect = ConnectionError("API unavailable")
        result = validate_entry("test content", provider)
        assert result.verdict == Verdict.AMBIGUOUS
        assert "API unavailable" in result.explanation

    def test_passes_temperature(self) -> None:
        """Temperature parameter is forwarded to provider."""
        provider = MagicMock()
        provider.generate.return_value = '{"verdict": "clean", "confidence": 0.9}'
        validate_entry("content", provider, temperature=0.3)
        _, kwargs = provider.generate.call_args
        assert kwargs["temperature"] == 0.3
