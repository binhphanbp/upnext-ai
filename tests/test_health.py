from fastapi.testclient import TestClient


def test_live_health_does_not_require_provider_configuration(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_health_includes_non_sensitive_capabilities(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "upnext-ai"
    assert body["models"] == {"structured": "gemini-2.5-flash-lite", "text": "gemini-2.5-flash"}
