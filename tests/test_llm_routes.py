from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_llm_provider
from app.main import create_app
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
