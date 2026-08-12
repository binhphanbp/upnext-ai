from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.contracts.llm import LlmMessage, TextStreamRequest
from app.core.config import Settings
from app.providers.base import (
    ProviderInvalidOutputError,
    ProviderNotConfiguredError,
    ProviderTimeoutError,
)
from app.providers.gemini import GeminiProvider
from app.providers.json_schema import normalize_response_json_schema


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
async def test_structured_generation_normalizes_backend_schema_before_gemini_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        text='{"intent":"GENERAL_GUIDANCE","toolCalls":null}',
        parsed=None,
        usage_metadata=None,
    )
    client = client_with(response)
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())
    backend_schema = {
        "type": "OBJECT",
        "properties": {
            "intent": {"type": "STRING", "enum": ["GENERAL_GUIDANCE"]},
            "toolCalls": {
                "type": "ARRAY",
                "nullable": True,
                "items": {
                    "type": "OBJECT",
                    "properties": {"name": {"type": "STRING"}},
                    "required": ["name"],
                },
            },
        },
        "required": ["intent"],
    }

    await provider.generate_structured(
        system_instruction="Return JSON only.",
        messages=[("user", "hello")],
        response_schema=backend_schema,
        temperature=0,
    )

    config = client.aio.models.generate_content.await_args.kwargs["config"]
    assert config.response_json_schema == {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["GENERAL_GUIDANCE"]},
            "toolCalls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        "required": ["intent"],
    }
    assert backend_schema["type"] == "OBJECT"


def test_schema_normalization_drops_legacy_nullable_for_gemini_compatibility() -> None:
    assert normalize_response_json_schema(
        {
            "type": "OBJECT",
            "nullable": True,
            "properties": {
                "note": {"type": "STRING", "nullable": True},
            },
        }
    ) == {
        "type": "object",
        "properties": {"note": {"type": "string"}},
    }


def test_schema_normalization_preserves_standard_schema_and_rejects_unknown_types() -> None:
    assert normalize_response_json_schema({"type": "object", "properties": {}}) == {
        "type": "object",
        "properties": {},
    }
    with pytest.raises(ValueError, match="Unsupported JSON Schema type"):
        normalize_response_json_schema({"type": "DATE"})


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
async def test_structured_generation_accepts_sdk_parsed_value_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        text=None,
        parsed={"intent": "GENERAL_GUIDANCE"},
        usage_metadata=None,
    )
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client_with(response))
    provider = GeminiProvider(settings())

    value, _, _ = await provider.generate_structured(
        system_instruction="Return JSON only.",
        messages=[("user", "hello")],
        response_schema={"type": "object"},
        temperature=0,
    )

    assert value == {"intent": "GENERAL_GUIDANCE"}


@pytest.mark.asyncio
async def test_structured_generation_maps_timeout_to_stable_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_with(SimpleNamespace())
    client.aio.models.generate_content = AsyncMock(side_effect=TimeoutError)
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderTimeoutError):
        await provider.generate_structured(
            system_instruction="Return JSON only.",
            messages=[("user", "hello")],
            response_schema={"type": "object"},
            temperature=0,
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
