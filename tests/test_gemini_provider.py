from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.contracts.llm import LlmMessage, TextStreamRequest
from app.core.config import Settings
from app.providers.base import ProviderInvalidOutputError, ProviderNotConfiguredError
from app.providers.gemini import GeminiProvider


def settings(*, gemini_api_key: str | None = "test-key") -> Settings:
    return Settings(
        internal_jwt_secret=SecretStr("test-internal-secret-that-is-at-least-32-characters"),
        gemini_api_key=SecretStr(gemini_api_key) if gemini_api_key else None,
        environment="test",
    )


def client_with(response: object) -> SimpleNamespace:
    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=AsyncMock(return_value=response),
                generate_content_stream=AsyncMock(),
            )
        )
    )


@pytest.mark.asyncio
async def test_structured_generation_parses_json_and_reports_provider_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        text='{"intent":"job_search"}',
        parsed=None,
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=4),
    )
    client = client_with(response)
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    value, input_tokens, output_tokens = await provider.generate_structured(
        system_instruction="Return JSON only.",
        messages=[("user", "Find React jobs")],
        response_schema={"type": "object"},
        temperature=0.2,
    )

    assert value == {"intent": "job_search"}
    assert (input_tokens, output_tokens) == (12, 4)
    assert client.aio.models.generate_content.await_count == 1


@pytest.mark.asyncio
async def test_structured_generation_rejects_invalid_provider_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(text="not-json", parsed=None, usage_metadata=None)
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client_with(response))
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderInvalidOutputError):
        await provider.generate_structured(
            system_instruction="Return JSON only.",
            messages=[("user", "hello")],
            response_schema={"type": "object"},
            temperature=None,
        )


@pytest.mark.asyncio
async def test_stream_yields_text_and_final_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def chunks() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            text="Hello",
            usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=1),
        )
        yield SimpleNamespace(
            text=" world",
            usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=2),
        )

    client = client_with(SimpleNamespace())
    client.aio.models.generate_content_stream = AsyncMock(return_value=chunks())
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    events = [
        event
        async for event in provider.stream_text(
            TextStreamRequest(
                systemInstruction="Be concise.",
                messages=[LlmMessage(role="user", text="hello")],
            )
        )
    ]

    assert events == [("text", "Hello"), ("text", " world"), ("usage", 5), ("output_tokens", 2)]


@pytest.mark.asyncio
async def test_provider_fails_closed_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    provider = GeminiProvider(settings(gemini_api_key=None))

    with pytest.raises(ProviderNotConfiguredError):
        await provider.generate_structured(
            system_instruction="x",
            messages=[("user", "x")],
            response_schema={},
            temperature=None,
        )
