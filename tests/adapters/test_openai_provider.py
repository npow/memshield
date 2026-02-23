"""Tests for memshield.adapters.openai_provider."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memshield.adapters.openai_provider import OpenAIProvider


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_init_defaults(self) -> None:
        """Constructor sets default model."""
        provider = OpenAIProvider()
        assert provider._model == "gpt-4o"

    def test_init_custom(self) -> None:
        """Constructor accepts custom model and base_url."""
        provider = OpenAIProvider(
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
        )
        assert provider._model == "llama3.1:8b"
        assert provider._base_url == "http://localhost:11434/v1"

    def test_lazy_client_creation(self) -> None:
        """Client is not created until first generate call."""
        provider = OpenAIProvider()
        assert provider._client is None

    def test_generate_calls_api(self) -> None:
        """generate() calls the OpenAI chat completions API."""
        provider = OpenAIProvider()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        result = provider.generate("test prompt", temperature=0.1)

        assert result == "test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.1,
        )

    def test_generate_handles_none_content(self) -> None:
        """generate() returns empty string when content is None."""
        provider = OpenAIProvider()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        provider._client = mock_client

        result = provider.generate("prompt")
        assert result == ""

    def test_satisfies_llm_provider_protocol(self) -> None:
        """OpenAIProvider satisfies the LLMProvider protocol."""
        from memshield._types import LLMProvider
        provider = OpenAIProvider()
        assert isinstance(provider, LLMProvider)

    def test_import_error_message(self) -> None:
        """Helpful error when openai is not installed."""
        provider = OpenAIProvider()
        provider._client = None  # force lazy init
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="pip install memshield\\[openai\\]"):
                provider._get_client()
