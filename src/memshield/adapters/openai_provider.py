"""OpenAI-compatible LLM provider (works with OpenAI, vLLM, Ollama in OpenAI mode)."""
from __future__ import annotations

from typing import Any


class OpenAIProvider:
    """LLM provider using the OpenAI Chat Completions API.

    Compatible with any OpenAI-compatible endpoint: OpenAI, Azure OpenAI,
    vLLM, Ollama (with OpenAI compatibility mode), LiteLLM, etc.
    Set base_url to point to a local inference server.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai is required for the OpenAI provider. "
                    "Install it with: pip install memshield[openai]"
                ) from None
            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Generate a completion using the OpenAI Chat Completions API."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
