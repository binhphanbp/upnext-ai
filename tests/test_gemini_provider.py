from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors
from pydantic import SecretStr

from app.contracts.llm import LlmMessage, TextStreamRequest
from app.core.config import Settings
from app.providers.base import (
    ProviderError,
    ProviderInvalidOutputError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderRegionBlockedError,
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
                embed_content=AsyncMock(),
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
async def test_structured_generation_uses_quality_model_only_for_quality_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(text='{"ok":true}', parsed=None, usage_metadata=None)
    client = client_with(response)
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    await provider.generate_structured(
        system_instruction="Return JSON only.",
        messages=[("user", "Write a JD")],
        response_schema={"type": "object"},
        temperature=0.2,
        model_tier="quality",
    )

    assert client.aio.models.generate_content.await_args.kwargs["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_structured_generation_uses_the_controlled_batch_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(text='{"ok":true}', parsed=None, usage_metadata=None)
    client = client_with(response)
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(
        settings().model_copy(
            update={"structured_timeout_seconds": 1, "batch_structured_timeout_seconds": 30}
        )
    )

    observed_timeouts: list[int] = []

    class TimeoutProbe:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *args: object) -> None:
            return None

    def timeout_probe(seconds: int) -> TimeoutProbe:
        observed_timeouts.append(seconds)
        return TimeoutProbe()

    monkeypatch.setattr("app.providers.gemini.asyncio.timeout", timeout_probe)

    await provider.generate_structured(
        system_instruction="Return JSON only.",
        messages=[("user", "Score these CVs")],
        response_schema={"type": "object"},
        temperature=0,
        execution_profile="batch",
    )

    assert observed_timeouts == [30]


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


@pytest.mark.asyncio
async def test_embedding_generation_preserves_model_space_and_l2_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = [3.0, 4.0] + [0.0] * 766
    response = SimpleNamespace(embeddings=[SimpleNamespace(values=values)])
    client = client_with(SimpleNamespace())
    client.aio.models.embed_content = AsyncMock(return_value=response)
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    vector = await provider.embed_text(text="TypeScript", dimensions=768)

    assert len(vector) == 768
    assert vector[:2] == pytest.approx([0.6, 0.8])
    await_args = client.aio.models.embed_content.await_args
    assert await_args is not None
    call = await_args.kwargs
    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == "TypeScript"
    assert call["config"].output_dimensionality == 768
    assert getattr(call["config"], "task_type", None) is None


@pytest.mark.asyncio
async def test_embedding_generation_rejects_invalid_provider_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0])])
    client = client_with(SimpleNamespace())
    client.aio.models.embed_content = AsyncMock(return_value=response)
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderInvalidOutputError):
        await provider.embed_text(text="TypeScript", dimensions=768)


def client_error(status: str, message: str, code: int = 400) -> errors.ClientError:
    """Build a ClientError shaped like a real Gemini REST error body."""

    return errors.ClientError(
        code,
        {"error": {"code": code, "status": status, "message": message}},
    )


@pytest.mark.asyncio
async def test_structured_generation_maps_unsupported_region_to_region_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exact status/message Gemini returns for a deployment it refuses to serve.
    client = client_with(SimpleNamespace())
    client.aio.models.generate_content = AsyncMock(
        side_effect=client_error(
            "FAILED_PRECONDITION", "User location is not supported for the API use."
        )
    )
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderRegionBlockedError) as raised:
        await provider.generate_structured(
            system_instruction="Return JSON only.",
            messages=[("user", "hello")],
            response_schema={"type": "object"},
            temperature=0,
        )

    assert raised.value.code == "AI_PROVIDER_REGION_BLOCKED"


@pytest.mark.asyncio
async def test_embedding_maps_unsupported_region_to_region_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_with(SimpleNamespace())
    client.aio.models.embed_content = AsyncMock(
        side_effect=client_error(
            "FAILED_PRECONDITION", "User location is not supported for the API use."
        )
    )
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderRegionBlockedError):
        await provider.embed_text(text="TypeScript", dimensions=768)


@pytest.mark.asyncio
async def test_other_failed_precondition_stays_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FAILED_PRECONDITION is reused for unrelated causes (e.g. billing), so the
    # status alone must not be treated as a geography block.
    client = client_with(SimpleNamespace())
    client.aio.models.generate_content = AsyncMock(
        side_effect=client_error("FAILED_PRECONDITION", "Billing account is not configured.")
    )
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderError) as raised:
        await provider.generate_structured(
            system_instruction="Return JSON only.",
            messages=[("user", "hello")],
            response_schema={"type": "object"},
            temperature=0,
        )

    assert not isinstance(raised.value, ProviderRegionBlockedError)
    assert raised.value.code == "AI_SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_rate_limit_still_maps_before_region_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_with(SimpleNamespace())
    client.aio.models.generate_content = AsyncMock(
        side_effect=client_error("RESOURCE_EXHAUSTED", "Quota exceeded.", code=429)
    )
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderRateLimitError):
        await provider.generate_structured(
            system_instruction="Return JSON only.",
            messages=[("user", "hello")],
            response_schema={"type": "object"},
            temperature=0,
        )


def test_client_uses_the_api_version_that_serves_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Structured output is rejected on v1 with "JSON mode is not enabled for
    # api version v1", which would break every structured capability.
    captured: dict[str, object] = {}

    def capture_client(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return client_with(SimpleNamespace())

    monkeypatch.setattr("app.providers.gemini.genai.Client", capture_client)
    GeminiProvider(settings())

    http_options = captured["http_options"]
    assert getattr(http_options, "api_version", None) == "v1beta"


def test_google_genai_async_streaming_dependency_is_installed() -> None:
    """google-genai's async stream path needs aiohttp at runtime.

    The SDK marks aiohttp optional and sets `has_aiohttp` at import time, but
    its `_aiter_response_stream` isinstance check dereferences
    `aiohttp.ClientResponse` without consulting that flag. When the package is
    absent, streaming fails with `NameError` only after the provider has
    already returned 200, which reads as a generic AI outage rather than a
    missing dependency. Assert it is installed so the failure cannot come back
    silently.
    """

    import aiohttp  # noqa: PLC0415 - deliberately imported inside the guard test.

    assert aiohttp.ClientResponse is not None


def grounded_response(
    *,
    text: str = '{"median": 30}',
    chunks: list[object] | None = None,
    queries: list[str] | None = None,
) -> SimpleNamespace:
    metadata = SimpleNamespace(
        grounding_chunks=chunks if chunks is not None else [],
        web_search_queries=queries if queries is not None else [],
    )
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(grounding_metadata=metadata)],
        usage_metadata=SimpleNamespace(prompt_token_count=30, candidates_token_count=9),
    )


def web_chunk(title: str, uri: str) -> SimpleNamespace:
    return SimpleNamespace(web=SimpleNamespace(title=title, uri=uri))


@pytest.mark.asyncio
async def test_grounded_generation_requests_search_and_never_a_response_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pairing a response schema with the search tool makes Gemini return empty
    # grounding metadata, which would silently strip every citation.
    client = client_with(grounded_response())
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    await provider.generate_grounded(
        system_instruction="Cite your sources.",
        prompt="Backend salaries in Ha Noi?",
        temperature=0.2,
    )

    kwargs = client.aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "gemini-2.5-pro"
    config = kwargs["config"]
    assert config.tools and config.tools[0].google_search is not None
    assert getattr(config, "response_json_schema", None) is None
    assert getattr(config, "response_mime_type", None) is None


@pytest.mark.asyncio
async def test_grounded_generation_collects_distinct_sources_and_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_with(
        grounded_response(
            chunks=[
                web_chunk("VietnamWorks", "https://example.test/a"),
                # Repeated URL: the caller counts distinct sources to decide
                # confidence, so duplicates must not inflate that count.
                web_chunk("VietnamWorks mirror", "https://example.test/a"),
                web_chunk("TopDev", "https://example.test/b"),
                web_chunk("", "https://example.test/c"),
                web_chunk("No link", ""),
            ],
            queries=["backend salary hanoi", "  ", "luong backend ha noi"],
        )
    )
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    answer = await provider.generate_grounded(
        system_instruction="Cite your sources.",
        prompt="Backend salaries in Ha Noi?",
        temperature=None,
    )

    assert answer.sources == (
        ("VietnamWorks", "https://example.test/a"),
        ("TopDev", "https://example.test/b"),
    )
    assert answer.search_queries == ("backend salary hanoi", "luong backend ha noi")
    assert (answer.input_tokens, answer.output_tokens) == (30, 9)
    assert answer.text == '{"median": 30}'


@pytest.mark.asyncio
async def test_grounded_generation_reports_an_answer_without_grounding_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An ungrounded answer must arrive as verifiably empty evidence rather than
    # crash, so the caller can reject it on its own rules.
    client = client_with(SimpleNamespace(text="No data found.", candidates=[], usage_metadata=None))
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    answer = await provider.generate_grounded(
        system_instruction="Cite your sources.",
        prompt="Backend salaries in Ha Noi?",
        temperature=None,
    )

    assert answer.sources == ()
    assert answer.search_queries == ()


@pytest.mark.asyncio
async def test_grounded_generation_rejects_an_empty_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_with(grounded_response(text="   "))
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderInvalidOutputError):
        await provider.generate_grounded(
            system_instruction="Cite your sources.",
            prompt="Backend salaries in Ha Noi?",
            temperature=None,
        )


@pytest.mark.asyncio
async def test_grounded_generation_maps_unsupported_region_to_region_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_with(SimpleNamespace())
    client.aio.models.generate_content = AsyncMock(
        side_effect=client_error(
            "FAILED_PRECONDITION", "User location is not supported for the API use."
        )
    )
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderRegionBlockedError):
        await provider.generate_grounded(
            system_instruction="Cite your sources.",
            prompt="Backend salaries in Ha Noi?",
            temperature=None,
        )


@pytest.mark.asyncio
async def test_grounded_generation_fails_closed_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    provider = GeminiProvider(settings(gemini_api_key=None))

    with pytest.raises(ProviderNotConfiguredError):
        await provider.generate_grounded(
            system_instruction="Cite your sources.",
            prompt="Backend salaries in Ha Noi?",
            temperature=None,
        )


@pytest.mark.asyncio
async def test_grounded_generation_maps_timeout_to_stable_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = client_with(SimpleNamespace())
    client.aio.models.generate_content = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr("app.providers.gemini.genai.Client", lambda **_: client)
    provider = GeminiProvider(settings())

    with pytest.raises(ProviderTimeoutError):
        await provider.generate_grounded(
            system_instruction="Cite your sources.",
            prompt="Backend salaries in Ha Noi?",
            temperature=None,
        )
