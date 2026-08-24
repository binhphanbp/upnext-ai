"""Lock the error envelope this service is allowed to emit.

The backend adapter derives its internal error code from `detail.code` and, when
it cannot find one there, guesses from the HTTP status. Two of those guesses are
dangerous: 400 and 422 both become `AI_INVALID_OUTPUT`, the single code that is
deliberately excluded from provider failover, so an error body shaped wrongly
turns our own bug into "the AI could not read your content" with no second
provider attempted.

These tests therefore assert two properties for every error this service can
produce, rather than for one route:

1. `detail` is always an object carrying a code the backend recognises -- never
   FastAPI's default list.
2. No error body repeats any part of the caller's payload. The payload is a
   recruiter's prompt or the base64 of their private document.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_llm_provider
from app.main import create_app
from tests.helpers import StubProvider, auth_headers

_KNOWN_CODES = {
    "AI_MODEL_TIMEOUT",
    "AI_MODEL_RATE_LIMIT",
    "AI_INVALID_OUTPUT",
    "AI_SERVICE_UNAVAILABLE",
    # Emitted by the auth dependency. Unknown to the backend's allow-list, so it
    # falls through to the status map: 401 is not mapped, giving the correct
    # AI_SERVICE_UNAVAILABLE and a failover.
    "AI_INTERNAL_UNAUTHORIZED",
}

_VALID_BODY = {
    "systemInstruction": "Return JSON only.",
    "prompt": "Generate a JD.",
    "responseSchema": {"type": "object"},
}


def _client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: StubProvider()
    return TestClient(app)


# Every way a caller can get a request wrong. None may produce a list `detail`.
_MALFORMED_BODIES = [
    pytest.param({}, id="empty-body"),
    pytest.param({"systemInstruction": "x"}, id="missing-required-fields"),
    pytest.param({**_VALID_BODY, "temperature": 9}, id="out-of-range-number"),
    pytest.param({**_VALID_BODY, "unknownField": "x"}, id="extra-field-forbidden"),
    pytest.param({**_VALID_BODY, "modelTier": "fast"}, id="literal-not-allowed"),
    pytest.param({**_VALID_BODY, "prompt": ""}, id="empty-string"),
    pytest.param({**_VALID_BODY, "responseSchema": "not-an-object"}, id="wrong-type"),
    pytest.param({**_VALID_BODY, "prompt": "x" * 50_001}, id="over-max-length"),
]


@pytest.mark.parametrize("body", _MALFORMED_BODIES)
def test_no_request_shape_produces_a_list_detail(body: dict[str, object]) -> None:
    response = _client().post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="job-post:generate"),
        json=body,
    )

    detail = response.json()["detail"]
    assert isinstance(detail, dict), f"list detail would be read as AI_INVALID_OUTPUT: {detail}"
    assert detail["code"] in _KNOWN_CODES
    # 400 and 422 are the two statuses the backend reads as AI_INVALID_OUTPUT
    # when no code is found. Never emit them for a request-shape problem.
    assert response.status_code not in {400, 422}


def test_an_unparseable_body_still_carries_a_readable_code() -> None:
    """The third shape `detail` can take: a bare string.

    FastAPI raises `400 {"detail": "There was an error parsing the body"}` when it
    cannot decode the JSON at all -- a different path from schema validation, so
    the validation handler never sees it. Left alone, the backend reads no code,
    guesses from the 400, and lands on AI_INVALID_OUTPUT: no failover for what is
    a serialisation disagreement between the two services.
    """

    response = _client().post(
        "/internal/v1/job-posts/generate",
        headers={**auth_headers(scope="job-post:generate"), "Content-Type": "application/json"},
        content=b'{"systemInstruction": "x", "prompt": ',
    )

    detail = response.json()["detail"]
    assert isinstance(detail, dict), f"a string detail is as unreadable as a list: {detail}"
    assert detail["code"] in _KNOWN_CODES


@pytest.mark.parametrize(
    ("method", "path"),
    [
        pytest.param("post", "/internal/v1/job-posts/does-not-exist", id="unknown-route-404"),
        pytest.param("get", "/internal/v1/job-posts/generate", id="wrong-method-405"),
    ],
)
def test_routing_errors_carry_a_readable_code(method: str, path: str) -> None:
    # A mistyped AI_SERVICE_URL or a path renamed on one side only lands here.
    response = getattr(_client(), method)(path, headers=auth_headers(scope="job-post:generate"))

    detail = response.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["code"] in _KNOWN_CODES


def test_unauthorized_detail_is_an_object_too() -> None:
    response = _client().post("/internal/v1/job-posts/generate", json=_VALID_BODY)

    assert response.status_code == 401
    assert isinstance(response.json()["detail"], dict)
    assert response.json()["detail"]["code"] in _KNOWN_CODES


def test_a_rejected_document_is_never_echoed_back() -> None:
    # Invalid base64: the failure comes from the field validator, whose error
    # would carry the whole `file` object as `input` if it were model-scoped.
    payload = "QUJD" * 4_000
    response = _client().post(
        "/internal/v1/job-posts/extract",
        headers=auth_headers(scope="job-post:extract"),
        json={
            **_VALID_BODY,
            "file": {"mimeType": "application/pdf", "base64Data": payload + "!!!!"},
        },
    )

    body = response.text
    assert "QUJDQUJD" not in body
    # An 8 MiB upload must not be able to provoke a multi-megabyte error body.
    assert len(body) < 500, body


def test_a_valid_document_that_fails_a_sibling_field_is_not_echoed() -> None:
    # The offending field is `temperature`, but the body also carries a document.
    # Pydantic reports only the failing field, so the document must not appear.
    payload = base64.b64encode(b"%PDF-1.7 " + b"x" * 3_000).decode()
    response = _client().post(
        "/internal/v1/job-posts/extract",
        headers=auth_headers(scope="job-post:extract"),
        json={
            **_VALID_BODY,
            "temperature": 9,
            "file": {"mimeType": "application/pdf", "base64Data": payload},
        },
    )

    assert payload[:32] not in response.text
    assert len(response.text) < 500, response.text


def test_prompt_text_is_never_echoed_back() -> None:
    secret = "MUC-LUONG-BI-MAT-CUA-CONG-TY"
    response = _client().post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="job-post:generate"),
        json={**_VALID_BODY, "prompt": secret, "temperature": 9},
    )

    assert secret not in response.text


def test_an_unhandled_error_uses_the_same_envelope() -> None:
    app = create_app()

    def exploding_provider() -> StubProvider:
        raise RuntimeError("provider construction blew up with secrets in the message")

    app.dependency_overrides[get_llm_provider] = exploding_provider
    # raise_server_exceptions=False so the handler runs instead of the test client
    # re-raising, which is what a real deployment does.
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/internal/v1/job-posts/generate",
        headers=auth_headers(scope="job-post:generate"),
        json=_VALID_BODY,
    )

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "AI_SERVICE_UNAVAILABLE"
    assert "secrets in the message" not in response.text
