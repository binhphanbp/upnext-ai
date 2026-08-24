from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.contracts.llm import TextStreamRequest
from app.providers.base import EmbeddingProvider, GroundedAnswer, GroundedProvider, LlmProvider


def internal_token(*, secret: str | None = None, scope: str = "llm:invoke", **claims: Any) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": "upnext-be",
        "iss": "upnext-be",
        "aud": "upnext-ai",
        "scope": scope,
        "jti": "test-run-id",
        "environment": "test",
        "iat": now,
        "exp": now + timedelta(seconds=60),
        **claims,
    }
    return jwt.encode(
        payload,
        secret or "test-internal-secret-that-is-at-least-32-characters",
        algorithm="HS256",
    )


def auth_headers(**claims: Any) -> dict[str, str]:
    return {"Authorization": f"Bearer {internal_token(**claims)}"}


def assert_contract_rejected(response: Any) -> None:
    """Assert a body the narrow contract refuses is reported as *our* outage.

    `upnext-be` is the only caller and builds every body from its own code, so a
    validation failure means the two services disagree about the schema -- our
    bug, never the recruiter's. It must therefore be failover-eligible.

    These assertions used to read `422`, which quietly inverted that: the backend
    reads a code from `detail`, cannot find one in FastAPI's list-shaped default
    body, and falls back to guessing from the status -- where 422 means
    AI_INVALID_OUTPUT, the single code that must never fail over. The recruiter
    got "the AI could not read your content" and no second provider was tried.
    """

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "AI_SERVICE_UNAVAILABLE"


class StubProvider(LlmProvider):
    def __init__(self) -> None:
        self.structured_calls: list[dict[str, Any]] = []

    @property
    def configured(self) -> bool:
        return True

    @property
    def structured_model(self) -> str:
        return "test-structured"

    @property
    def text_model(self) -> str:
        return "test-text"

    def structured_model_for(self, model_tier: str) -> str:
        return "test-quality" if model_tier == "quality" else self.structured_model

    async def generate_structured(self, **kwargs: Any) -> tuple[Any, int, int]:
        self.structured_calls.append(kwargs)
        return {"ok": True}, 12, 8

    async def generate_structured_with_file(self, **kwargs: Any) -> tuple[Any, int, int]:
        self.structured_calls.append(kwargs)
        return {"ok": True}, 12, 8

    async def _stream(self) -> AsyncIterator[tuple[str, str | int]]:
        yield "text", "Hello"
        yield "text", " world"
        yield "usage", 12
        yield "output_tokens", 8

    def stream_text(self, request: TextStreamRequest) -> AsyncIterator[tuple[str, str | int]]:
        _ = request
        return self._stream()


class StubGroundedProvider(GroundedProvider):
    def __init__(self, answer: GroundedAnswer | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.answer = answer or GroundedAnswer(
            text='{"median": 30}',
            sources=(("VietnamWorks", "https://example.test/a"),),
            search_queries=("backend salary hanoi",),
            input_tokens=41,
            output_tokens=17,
        )

    @property
    def configured(self) -> bool:
        return True

    @property
    def grounded_model(self) -> str:
        return "test-grounded"

    async def generate_grounded(self, **kwargs: Any) -> GroundedAnswer:
        self.calls.append(kwargs)
        return self.answer


class StubEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def configured(self) -> bool:
        return True

    @property
    def embedding_model(self) -> str:
        return "gemini-embedding-001"

    async def embed_text(self, *, text: str, dimensions: int) -> list[float]:
        self.calls.append({"text": text, "dimensions": dimensions})
        return [1.0] + [0.0] * (dimensions - 1)
