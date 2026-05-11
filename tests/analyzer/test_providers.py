"""Tests for the LLM provider layer (A.5)."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from limenex.analyzer.providers import (
    LLMProvider,
    LLMProviderError,
    OpenAICompatProvider,
    _ERROR_BODY_MAX_CHARS,
)


BASE_URL = "https://api.example.com/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
MODEL = "test-model"


def _chat_response(content: str | Any) -> dict[str, Any]:
    """Minimal well-formed OpenAI chat completion response."""
    return {
        "choices": [{"message": {"content": content}}],
    }


@pytest.fixture
async def provider():
    """Yield a provider with a real API key, closed on teardown."""
    p = OpenAICompatProvider(
        base_url=BASE_URL, model=MODEL, api_key="sk-test"
    )
    try:
        yield p
    finally:
        await p.aclose()


@pytest.fixture
async def provider_no_key():
    """Yield a provider with no API key."""
    p = OpenAICompatProvider(base_url=BASE_URL, model=MODEL, api_key=None)
    try:
        yield p
    finally:
        await p.aclose()


# ---------- success paths ----------

@respx.mock
async def test_complete_happy_path(provider):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_response("hello world"))
    )
    result = await provider.complete(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert result == "hello world"


@respx.mock
async def test_complete_passes_response_format(provider):
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_response("{}"))
    )
    await provider.complete(
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"] == {"type": "json_object"}


@respx.mock
async def test_complete_no_auth_header_when_api_key_none(provider_no_key):
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_response("ok"))
    )
    await provider_no_key.complete(messages=[{"role": "user", "content": "hi"}])
    headers = route.calls.last.request.headers
    assert "authorization" not in {k.lower() for k in headers}


@respx.mock
async def test_complete_auth_header_when_api_key_set(provider):
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_response("ok"))
    )
    await provider.complete(messages=[{"role": "user", "content": "hi"}])
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"


@respx.mock
async def test_complete_empty_content_allowed(provider):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_response(""))
    )
    result = await provider.complete(messages=[{"role": "user", "content": "hi"}])
    assert result == ""


# ---------- transport / envelope failures ----------

@respx.mock
async def test_complete_raises_on_transport_error(provider):
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("conn refused"))
    with pytest.raises(LLMProviderError, match="HTTP transport error"):
        await provider.complete(messages=[{"role": "user", "content": "hi"}])


@respx.mock
async def test_complete_raises_on_http_4xx(provider):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    with pytest.raises(LLMProviderError) as exc_info:
        await provider.complete(messages=[{"role": "user", "content": "hi"}])
    msg = str(exc_info.value)
    assert "401" in msg
    assert "unauthorized" in msg


@respx.mock
async def test_complete_raises_on_http_5xx(provider):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(503, text="service down")
    )
    with pytest.raises(LLMProviderError) as exc_info:
        await provider.complete(messages=[{"role": "user", "content": "hi"}])
    assert "503" in str(exc_info.value)


@respx.mock
async def test_complete_raises_on_invalid_json_body(provider):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, text="<html>not json</html>")
    )
    with pytest.raises(LLMProviderError, match="not valid JSON"):
        await provider.complete(messages=[{"role": "user", "content": "hi"}])


@respx.mock
async def test_complete_raises_on_missing_choices_key(provider):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json={"no_choices_here": True})
    )
    with pytest.raises(LLMProviderError, match="missing choices"):
        await provider.complete(messages=[{"role": "user", "content": "hi"}])


@respx.mock
async def test_complete_raises_on_empty_choices_array(provider):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    with pytest.raises(LLMProviderError, match="missing choices"):
        await provider.complete(messages=[{"role": "user", "content": "hi"}])


@respx.mock
async def test_complete_raises_on_content_not_string(provider):
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_response(12345))
    )
    with pytest.raises(LLMProviderError, match="content is not a string"):
        await provider.complete(messages=[{"role": "user", "content": "hi"}])


@respx.mock
async def test_complete_truncates_large_error_bodies(provider):
    long_body = "x" * (_ERROR_BODY_MAX_CHARS * 3)
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(500, text=long_body)
    )
    with pytest.raises(LLMProviderError) as exc_info:
        await provider.complete(messages=[{"role": "user", "content": "hi"}])
    msg = str(exc_info.value)
    assert "truncated" in msg
    assert "x" * (_ERROR_BODY_MAX_CHARS * 3) not in msg


# ---------- request-shape contract ----------

@respx.mock
async def test_complete_sends_correct_model_and_temperature():
    p = OpenAICompatProvider(
        base_url=BASE_URL, model=MODEL, api_key="sk-x", temperature=0.7
    )
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_response("ok"))
    )
    try:
        await p.complete(messages=[{"role": "user", "content": "hi"}])
    finally:
        await p.aclose()
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == MODEL
    assert sent["temperature"] == 0.7
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


@respx.mock
async def test_complete_url_join_handles_trailing_slash():
    p = OpenAICompatProvider(
        base_url=BASE_URL + "/", model=MODEL, api_key="sk-x"
    )
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_chat_response("ok"))
    )
    try:
        await p.complete(messages=[{"role": "user", "content": "hi"}])
    finally:
        await p.aclose()
    assert route.called


# ---------- lifecycle + protocol ----------

async def test_async_context_manager_closes_client():
    async with OpenAICompatProvider(
        base_url=BASE_URL, model=MODEL, api_key="sk-x"
    ) as p:
        assert not p._client.is_closed
    assert p._client.is_closed


def test_protocol_runtime_check():
    p = OpenAICompatProvider(base_url=BASE_URL, model=MODEL, api_key="sk-x")
    try:
        assert isinstance(p, LLMProvider)
    finally:
        import asyncio
        asyncio.run(p.aclose())