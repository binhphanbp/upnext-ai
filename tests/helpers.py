from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.contracts.llm import TextStreamRequest
from app.providers.base import LlmProvider


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

    async def generate_structured(self, **kwargs: Any) -> tuple[Any, int, int]:
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
