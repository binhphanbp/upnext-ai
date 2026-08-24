import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_llm_provider
from app.main import create_app
from tests.helpers import StubProvider, assert_contract_rejected, auth_headers

LICENCE = base64.b64encode(b"%PDF-1.7 business licence").decode()


def client_with_stub() -> tuple[TestClient, StubProvider]:
    provider = StubProvider()
    app: FastAPI = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return TestClient(app), provider


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "systemInstruction": "Return JSON only.",
        "prompt": "Extract the licence fields.",
        "responseSchema": {"type": "object"},
        "file": {"mimeType": "application/pdf", "base64Data": LICENCE},
    }
    body.update(overrides)
    return body


def test_license_extraction_requires_its_own_scope_and_forwards_the_document() -> None:
    client, provider = client_with_stub()

    # No usable credential at all.
    response = client.post(
        "/internal/v1/companies/license-extract",
        headers={"Authorization": "Bearer ignored"},
    )
    assert response.status_code == 401

    # A token good for other capabilities must not read registration documents.
    for foreign_scope in ("llm:invoke", "job-post:extract", "embedding:invoke"):
        response = client.post(
            "/internal/v1/companies/license-extract",
            headers=auth_headers(scope=foreign_scope),
            json=payload(),
        )
        assert response.status_code == 401, foreign_scope

    response = client.post(
        "/internal/v1/companies/license-extract",
        headers=auth_headers(scope="company-license:extract"),
        json=payload(),
    )

    assert response.status_code == 200
    assert response.json()["model"] == "test-quality"
    call = provider.structured_calls[0]
    assert call["file"] == ("application/pdf", b"%PDF-1.7 business licence")
    assert call["model_tier"] == "quality"


def test_license_extraction_rejects_a_request_without_a_document() -> None:
    # Without this the licence scope would double as a general prompt endpoint.
    client, _ = client_with_stub()
    body = payload()
    del body["file"]

    response = client.post(
        "/internal/v1/companies/license-extract",
        headers=auth_headers(scope="company-license:extract"),
        json=body,
    )

    assert_contract_rejected(response)


def test_license_extraction_rejects_an_unsupported_document_type() -> None:
    client, _ = client_with_stub()

    response = client.post(
        "/internal/v1/companies/license-extract",
        headers=auth_headers(scope="company-license:extract"),
        json=payload(file={"mimeType": "text/html", "base64Data": LICENCE}),
    )

    assert_contract_rejected(response)


def test_license_extraction_rejects_malformed_base64() -> None:
    client, _ = client_with_stub()

    response = client.post(
        "/internal/v1/companies/license-extract",
        headers=auth_headers(scope="company-license:extract"),
        json=payload(file={"mimeType": "application/pdf", "base64Data": "not-base64!!"}),
    )

    assert_contract_rejected(response)
