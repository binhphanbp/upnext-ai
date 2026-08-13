import base64

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


def test_job_post_extraction_accepts_only_its_dedicated_scope_and_forwards_file() -> None:
    client, provider = client_with_stub()
    response = client.post(
        "/internal/v1/job-posts/extract",
        headers={"Authorization": "Bearer ignored"},
    )
    assert response.status_code == 401

    response = client.post(
        "/internal/v1/job-posts/extract",
        headers=auth_headers(scope="llm:invoke"),
        json={
            "systemInstruction": "Return JSON only.",
            "prompt": "Extract this JD.",
            "responseSchema": {"type": "object"},
        },
    )
    assert response.status_code == 401

    response = client.post(
        "/internal/v1/job-posts/extract",
        headers=auth_headers(scope="job-post:extract"),
        json={
            "systemInstruction": "Return JSON only.",
            "prompt": "Extract this JD.",
            "responseSchema": {"type": "object"},
            "file": {
                "mimeType": "application/pdf",
                "base64Data": base64.b64encode(b"%PDF-1.7 sample").decode(),
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["model"] == "test-quality"
    assert provider.structured_calls[0]["file"] == ("application/pdf", b"%PDF-1.7 sample")
    assert provider.structured_calls[0]["model_tier"] == "quality"


def test_job_post_extraction_rejects_invalid_base64_files() -> None:
    client, _ = client_with_stub()
    common = {
        "systemInstruction": "Return JSON only.",
        "prompt": "Extract this JD.",
        "responseSchema": {"type": "object"},
        "file": {"mimeType": "application/pdf", "base64Data": "not-base64"},
    }
    response = client.post(
        "/internal/v1/job-posts/extract",
        headers=auth_headers(scope="job-post:extract"),
        json=common,
    )
    assert response.status_code == 422
