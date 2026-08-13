from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_llm_provider
from app.main import create_app
from app.providers.base import ProviderInvalidOutputError, ProviderTimeoutError
from tests.helpers import StubProvider, auth_headers


def client_with_stub() -> tuple[TestClient, StubProvider]:
    provider = StubProvider()
    app: FastAPI = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return TestClient(app), provider


def test_structured_endpoint_preserves_provider_contract() -> None:
    client, provider = client_with_stub()
    response = client.post(
        "/internal/v1/llm/structured",
        headers=auth_headers(),
        json={
            "systemInstruction": "Return JSON only.",
            "messages": [{"role": "user", "text": "Analyse my CV"}],
            "responseSchema": {"type": "object"},
            "temperature": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "value": {"ok": True},
        "inputTokens": 12,
        "outputTokens": 8,
        "model": "test-structured",
    }
    assert provider.structured_calls[0]["messages"] == [("user", "Analyse my CV")]
    assert provider.structured_calls[0]["model_tier"] == "fast"
    assert provider.structured_calls[0]["execution_profile"] == "interactive"


def test_structured_endpoint_routes_quality_tier_without_accepting_model_names() -> None:
    client, provider = client_with_stub()
    response = client.post(
        "/internal/v1/llm/structured",
        headers=auth_headers(),
        json={
            "systemInstruction": "Return JSON only.",
            "messages": [{"role": "user", "text": "Write a production JD"}],
            "responseSchema": {"type": "object"},
            "modelTier": "quality",
            "executionProfile": "batch",
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "test-quality"
    assert provider.structured_calls[0]["model_tier"] == "quality"
    assert provider.structured_calls[0]["execution_profile"] == "batch"


def test_stream_endpoint_returns_event_stream_without_leaking_request_content() -> None:
    client, _ = client_with_stub()
    response = client.post(
        "/internal/v1/llm/stream",
        headers=auth_headers(),
        json={
            "systemInstruction": "You are helpful.",
            "messages": [{"role": "user", "text": "secret candidate context"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: text\ndata: {"text":"Hello"}' in response.text
    assert "event: usage" in response.text
    assert "secret candidate context" not in response.text


def test_structured_timeout_uses_gateway_timeout_and_stable_code() -> None:
    client, provider = client_with_stub()

    async def timeout(**_: object) -> tuple[object, int, int]:
        raise ProviderTimeoutError()

    provider.generate_structured = timeout  # type: ignore[method-assign]
    response = client.post(
        "/internal/v1/llm/structured",
        headers=auth_headers(),
        json={
            "systemInstruction": "Return JSON only.",
            "messages": [{"role": "user", "text": "hello"}],
            "responseSchema": {"type": "object"},
        },
    )

    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "AI_MODEL_TIMEOUT"


def test_structured_invalid_output_uses_bad_gateway_and_stable_code() -> None:
    client, provider = client_with_stub()

    async def invalid(**_: object) -> tuple[object, int, int]:
        raise ProviderInvalidOutputError()

    provider.generate_structured = invalid  # type: ignore[method-assign]
    response = client.post(
        "/internal/v1/llm/structured",
        headers=auth_headers(),
        json={
            "systemInstruction": "Return JSON only.",
            "messages": [{"role": "user", "text": "hello"}],
            "responseSchema": {"type": "object"},
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "AI_INVALID_OUTPUT"
