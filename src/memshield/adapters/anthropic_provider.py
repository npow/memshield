"""Anthropic Messages API LLM provider for MemShield."""
from __future__ import annotations

from typing import Any


class AnthropicProvider:
    """LLM provider using the Anthropic Messages API.

    Implements the :class:`~memshield._types.LLMProvider` protocol so that it
    can be passed directly to ``MemShield(llm=...)``.

    Usage::

        from memshield.adapters.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
        text = provider.generate("Summarise the following: ...")
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 256,
    ) -> None:
        """Initialise the provider.

        Args:
            model: Anthropic model identifier.
            api_key: Anthropic API key.  If ``None``, the client reads the
                ``ANTHROPIC_API_KEY`` environment variable.
            max_tokens: Maximum number of tokens to generate per request.
        """
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily create the Anthropic client.

        Returns:
            An initialised ``anthropic.Anthropic`` client instance.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic is required. "
                    "Install with: pip install memshield[anthropic]"
                ) from None
            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Generate a completion using the Anthropic Messages API.

        Args:
            prompt: The user prompt text.
            temperature: Sampling temperature.  ``0.0`` means deterministic.

        Returns:
            The generated text content as a plain string.
        """
        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content is a list of content blocks; get the first text block.
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
