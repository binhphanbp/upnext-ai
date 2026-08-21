import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_llm_provider
from app.main import create_app
from tests.helpers import StubProvider, assert_contract_rejected, auth_headers


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
    assert_contract_rejected(response)


def test_job_post_generation_accepts_only_its_dedicated_scope() -> None:
    client, provider = client_with_stub()
    payload = {
        "systemInstruction": "Return JSON only.",
        "prompt": "Create a JD for a backend engineer.",
        "responseSchema": {"type": "object"},
    }

    response = client.post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="llm:invoke"),
        json=payload,
    )
    assert response.status_code == 401

    response = client.post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="job-post:extract"),
        json=payload,
    )
    assert response.status_code == 401

    response = client.post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="job-post:generate"),
        json=payload,
    )
    assert response.status_code == 200
    assert response.json()["model"] == "test-quality"
    assert provider.structured_calls[0]["file"] is None

    # JD generation is deliberately text-only. Documents must use the separate
    # extraction capability, which has a tighter file-validation boundary.
    response = client.post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="job-post:generate"),
        json={
            **payload,
            "file": {
                "mimeType": "application/pdf",
                "base64Data": base64.b64encode(b"%PDF-1.7 sample").decode(),
            },
        },
    )
    assert_contract_rejected(response)

    response = client.post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="job-post:generate"),
        json={**payload, "modelTier": "fast"},
    )
    assert_contract_rejected(response)

    response = client.post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="job-post:generate"),
        json={**payload, "executionProfile": "batch"},
    )
    assert_contract_rejected(response)
