from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_grounded_provider
from app.main import create_app
from app.providers.base import GroundedAnswer
from tests.helpers import StubGroundedProvider, assert_contract_rejected, auth_headers


def client_with_stub(
    answer: GroundedAnswer | None = None,
) -> tuple[TestClient, StubGroundedProvider]:
    provider = StubGroundedProvider(answer)
    app: FastAPI = create_app()
    app.dependency_overrides[get_grounded_provider] = lambda: provider
    return TestClient(app), provider


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "systemInstruction": "Answer with JSON and cite your sources.",
        "prompt": "Backend salaries in Ha Noi, 3 years experience?",
        "temperature": 0.2,
    }
    body.update(overrides)
    return body


def test_grounded_requires_its_own_scope() -> None:
    client, _ = client_with_stub()

    response = client.post(
        "/internal/v1/research/grounded",
        headers={"Authorization": "Bearer ignored"},
    )
    assert response.status_code == 401

    # Grounded runs fan out into paid web searches on a premium model, so no
    # other capability's token may reach them.
    for foreign_scope in (
        "llm:invoke",
        "job-post:extract",
        "job-post:generate",
        "embedding:invoke",
        "company-license:extract",
    ):
        response = client.post(
            "/internal/v1/research/grounded",
            headers=auth_headers(scope=foreign_scope),
            json=payload(),
        )
        assert response.status_code == 401, foreign_scope


def test_grounded_returns_the_answer_with_the_evidence_it_consulted() -> None:
    client, provider = client_with_stub()

    response = client.post(
        "/internal/v1/research/grounded",
        headers=auth_headers(scope="research:grounded"),
        json=payload(),
    )

    assert response.status_code == 200
    body = response.json()
    # The text stays verbatim: the caller owns the shape it asked for, because a
    # response schema cannot be combined with the search tool.
    assert body["text"] == '{"median": 30}'
    assert body["sources"] == [{"title": "VietnamWorks", "url": "https://example.test/a"}]
    assert body["searchQueries"] == ["backend salary hanoi"]
    assert body["inputTokens"] == 41
    assert body["outputTokens"] == 17
    assert body["model"] == "test-grounded"
    assert provider.calls[0]["temperature"] == 0.2


def test_grounded_reports_an_ungrounded_answer_instead_of_hiding_it() -> None:
    # An answer with no citations is the failure this capability exists to catch;
    # it must reach the caller as empty evidence, not as a plausible number.
    client, _ = client_with_stub(
        GroundedAnswer(
            text="I could not find data.",
            sources=(),
            search_queries=(),
            input_tokens=5,
            output_tokens=6,
        )
    )

    response = client.post(
        "/internal/v1/research/grounded",
        headers=auth_headers(scope="research:grounded"),
        json=payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["searchQueries"] == []


def test_grounded_rejects_an_empty_prompt() -> None:
    client, _ = client_with_stub()

    response = client.post(
        "/internal/v1/research/grounded",
        headers=auth_headers(scope="research:grounded"),
        json=payload(prompt=""),
    )

    assert_contract_rejected(response)
