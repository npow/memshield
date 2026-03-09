"""Tests for memshield.adapters.anthropic_provider."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memshield.adapters.anthropic_provider import AnthropicProvider


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_init_defaults(self) -> None:
        """Constructor sets default model, max_tokens, and lazy client."""
        provider = AnthropicProvider()
        assert provider._model == "claude-haiku-4-5-20251001"
        assert provider._max_tokens == 256
        assert provider._api_key is None
        assert provider._client is None

    def test_init_custom(self) -> None:
        """Constructor accepts custom model, api_key, and max_tokens."""
        provider = AnthropicProvider(
            model="claude-opus-4-5",
            api_key="sk-test",
            max_tokens=512,
        )
        assert provider._model == "claude-opus-4-5"
        assert provider._api_key == "sk-test"
        assert provider._max_tokens == 512

    def test_lazy_client_creation(self) -> None:
        """Client is not created until first generate call."""
        provider = AnthropicProvider()
        assert provider._client is None

    def test_generate_calls_messages_create(self) -> None:
        """generate() calls client.messages.create with correct params."""
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001", max_tokens=128)
        mock_client = MagicMock()
        text_block = MagicMock()
        text_block.text = "generated response"
        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        result = provider.generate("test prompt", temperature=0.5)

        assert result == "generated response"
        mock_client.messages.create.assert_called_once_with(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            temperature=0.5,
            messages=[{"role": "user", "content": "test prompt"}],
        )

    def test_generate_returns_first_text_block(self) -> None:
        """generate() returns the text from the first content block."""
        provider = AnthropicProvider()
        mock_client = MagicMock()

        block1 = MagicMock()
        block1.text = "first block text"
        block2 = MagicMock()
        block2.text = "second block text"
        mock_response = MagicMock()
        mock_response.content = [block1, block2]
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        result = provider.generate("prompt")
        assert result == "first block text"

    def test_generate_returns_empty_string_when_no_text_blocks(self) -> None:
        """generate() returns '' when no content blocks have a text attribute."""
        provider = AnthropicProvider()
        mock_client = MagicMock()

        block = MagicMock(spec=[])  # no attributes at all
        mock_response = MagicMock()
        mock_response.content = [block]
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        result = provider.generate("prompt")
        assert result == ""

    def test_generate_default_temperature(self) -> None:
        """generate() uses temperature=0.0 by default."""
        provider = AnthropicProvider()
        mock_client = MagicMock()
        text_block = MagicMock()
        text_block.text = "response"
        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        provider.generate("prompt")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_satisfies_llm_provider_protocol(self) -> None:
        """AnthropicProvider satisfies the LLMProvider protocol."""
        from memshield._types import LLMProvider

        provider = AnthropicProvider()
        assert isinstance(provider, LLMProvider)

    def test_get_client_passes_api_key(self) -> None:
        """_get_client passes api_key to the Anthropic constructor when set."""
        provider = AnthropicProvider(api_key="my-key")
        mock_anthropic_cls = MagicMock()
        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic = mock_anthropic_cls

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            provider._client = None  # ensure lazy init runs
            provider._get_client()

        mock_anthropic_cls.assert_called_once_with(api_key="my-key")

    def test_get_client_no_api_key_omits_kwarg(self) -> None:
        """_get_client omits api_key kwarg when api_key is None."""
        provider = AnthropicProvider()
        mock_anthropic_cls = MagicMock()
        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic = mock_anthropic_cls

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            provider._client = None
            provider._get_client()

        mock_anthropic_cls.assert_called_once_with()

    def test_import_error_message(self) -> None:
        """Helpful ImportError when anthropic is not installed."""
        provider = AnthropicProvider()
        provider._client = None

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install memshield\\[anthropic\\]"):
                provider._get_client()
