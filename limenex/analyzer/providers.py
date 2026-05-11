"""LLM provider abstraction for the Limenex Skill Risk Analyzer.

The provider layer is a thin shim over "take messages, get a response
string back." Providers return the raw content string from the model;
parsing and validation are harness concerns.

Providers raise ``LLMProviderError`` only on transport- or envelope-level
failures (network error, non-2xx HTTP, unreadable provider response). A
provider does NOT raise on semantically wrong model output.
"""
from __future__ import annotations

import json
import httpx
from typing import Any, Protocol, runtime_checkable

_ERROR_BODY_MAX_CHARS = 500

class LLMProviderError(Exception):
    """Raised by an ``LLMProvider`` on transport or envelope failures.

    Examples:
        - Network error (connection refused, DNS failure, timeout)
        - Non-2xx HTTP status from the provider
        - Response body is not valid JSON or lacks ``choices[0].message.content``
        - Authentication failure reported by the provider

    NOT raised for:
        - The model returning syntactically valid output that fails
          downstream schema validation (harness concern).
        - The model refusing to answer or returning an empty string.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used by the analyzer.

    Provider instances own their own transport state (e.g. an
    ``httpx.AsyncClient``) and must be safe for concurrent use by the
    harness's bounded-concurrency analyze loop.

    Messages use the OpenAI chat format: a list of dicts with ``role``
    and ``content`` keys, where ``role`` is ``"system"``, ``"user"``, or
    ``"assistant"``.
    """

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send messages to the model and return its raw response string.

        Parameters
        ----------
        messages
            Chat-format message list.
        response_format
            Optional response-format directive passed through to the
            provider. Use ``{"type": "json_object"}`` to request
            structured output. ``None`` applies the provider's default.

        Returns
        -------
        str
            Raw content string. Not parsed, not validated.

        Raises
        ------
        LLMProviderError
            On transport or envelope failure. See class docstring.
        """
        ...


def _truncate_body(body: str) -> str:
    """Truncate a response body for inclusion in exception messages."""
    if len(body) <= _ERROR_BODY_MAX_CHARS:
        return body
    return body[:_ERROR_BODY_MAX_CHARS] + f"... [truncated, {len(body)} chars total]"


class OpenAICompatProvider:
    """LLM provider for OpenAI-compatible chat completion endpoints.

    Works with any endpoint implementing the OpenAI ``/chat/completions``
    API shape: OpenAI itself, OpenRouter, Together, Groq, local llama.cpp
    servers, ollama, LiteLLM proxies, etc.

    The provider owns an ``httpx.AsyncClient`` for its lifetime. Close
    it via ``aclose()`` or use the instance as an async context manager.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def model(self) -> str:
        """The model identifier passed to the provider."""
        return self._model

    @property
    def temperature(self) -> float:
        """The temperature applied to every request."""
        return self._temperature

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "OpenAICompatProvider":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if response_format is not None:
            body["response_format"] = response_format

        try:
            response = await self._client.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise LLMProviderError(f"HTTP transport error calling {url}: {e}") from e

        if response.status_code >= 400:
            raise LLMProviderError(
                f"Provider returned HTTP {response.status_code} from {url}: "
                f"{_truncate_body(response.text)}"
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as e:
            raise LLMProviderError(
                f"Provider response is not valid JSON: {e}. "
                f"Body: {_truncate_body(response.text)}"
            ) from e

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMProviderError(
                f"Provider response missing choices[0].message.content: {e}. "
                f"Payload: {_truncate_body(json.dumps(payload, default=repr))}"
            ) from e

        if not isinstance(content, str):
            raise LLMProviderError(
                f"Provider response content is not a string "
                f"(got {type(content).__name__}). "
                f"Payload: {_truncate_body(json.dumps(payload, default=repr))}"
            )

        return content
