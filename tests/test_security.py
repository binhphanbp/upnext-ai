from fastapi.testclient import TestClient

from tests.helpers import auth_headers, internal_token


def test_internal_routes_reject_missing_service_token(client: TestClient) -> None:
    response = client.post("/internal/v1/llm/structured", json={})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AI_INTERNAL_UNAUTHORIZED"


def test_internal_routes_reject_wrong_scope(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/llm/structured",
        headers={"Authorization": f"Bearer {internal_token(scope='candidate:read')}"},
        json={},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AI_INTERNAL_UNAUTHORIZED"


def test_internal_routes_reject_wrong_audience(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/llm/structured",
        headers=auth_headers(aud="another-service"),
        json={},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AI_INTERNAL_UNAUTHORIZED"


def test_internal_routes_reject_a_token_not_issued_for_the_backend_service(
    client: TestClient,
) -> None:
    response = client.post(
        "/internal/v1/llm/structured",
        headers=auth_headers(sub="another-internal-service"),
        json={},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AI_INTERNAL_UNAUTHORIZED"
