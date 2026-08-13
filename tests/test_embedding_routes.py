from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_embedding_provider
from app.main import create_app
from app.providers.base import ProviderInvalidOutputError, ProviderTimeoutError
from tests.helpers import StubEmbeddingProvider, auth_headers


def client_with_stub() -> tuple[TestClient, StubEmbeddingProvider]:
    provider = StubEmbeddingProvider()
    app: FastAPI = create_app()
    app.dependency_overrides[get_embedding_provider] = lambda: provider
    return TestClient(app), provider


def test_embedding_endpoint_preserves_compatibility_contract() -> None:
    client, provider = client_with_stub()
    response = client.post(
        "/internal/v1/embeddings",
        headers=auth_headers(scope="embedding:invoke"),
        json={"text": "Senior TypeScript engineer", "dimensions": 768},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["vector"]) == 768
    assert payload["model"] == "gemini-embedding-001"
    assert payload["dimensions"] == 768
    assert payload["normalization"] == "l2-v1"
    assert payload["cacheKey"] == "gemini-embedding-001:768:l2-v1"
    assert provider.calls == [{"text": "Senior TypeScript engineer", "dimensions": 768}]


def test_embedding_endpoint_requires_its_narrow_scope() -> None:
    client, _ = client_with_stub()
    response = client.post(
        "/internal/v1/embeddings",
        headers=auth_headers(scope="llm:invoke"),
        json={"text": "hello", "dimensions": 768},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AI_INTERNAL_UNAUTHORIZED"


def test_embedding_endpoint_rejects_dimension_drift_and_unknown_fields() -> None:
    client, _ = client_with_stub()
    headers = auth_headers(scope="embedding:invoke")

    wrong_dimensions = client.post(
        "/internal/v1/embeddings",
        headers=headers,
        json={"text": "hello", "dimensions": 1536},
    )
    unknown_field = client.post(
        "/internal/v1/embeddings",
        headers=headers,
        json={"text": "hello", "dimensions": 768, "model": "another-model"},
    )

    assert wrong_dimensions.status_code == 422
    assert unknown_field.status_code == 422


def test_embedding_endpoint_maps_provider_failures_to_stable_codes() -> None:
    client, provider = client_with_stub()

    async def invalid(**_: object) -> list[float]:
        raise ProviderInvalidOutputError()

    provider.embed_text = invalid  # type: ignore[method-assign]
    response = client.post(
        "/internal/v1/embeddings",
        headers=auth_headers(scope="embedding:invoke"),
        json={"text": "hello"},
    )
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "AI_INVALID_OUTPUT"

    async def timeout(**_: object) -> list[float]:
        raise ProviderTimeoutError()

    provider.embed_text = timeout  # type: ignore[method-assign]
    response = client.post(
        "/internal/v1/embeddings",
        headers=auth_headers(scope="embedding:invoke"),
        json={"text": "hello"},
    )
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "AI_MODEL_TIMEOUT"
