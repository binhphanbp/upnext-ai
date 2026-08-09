from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from app.contracts.llm import TextStreamRequest


class ProviderError(Exception):
    """Known provider failure that can be mapped to a stable internal error."""

    code = "AI_SERVICE_UNAVAILABLE"


class ProviderNotConfiguredError(ProviderError):
    code = "AI_SERVICE_UNAVAILABLE"


class ProviderRateLimitError(ProviderError):
    code = "AI_MODEL_RATE_LIMIT"


class ProviderInvalidOutputError(ProviderError):
    code = "AI_INVALID_OUTPUT"


class ProviderTimeoutError(ProviderError):
    code = "AI_MODEL_TIMEOUT"


class LlmProvider(Protocol):
    @property
    def configured(self) -> bool: ...

    @property
    def structured_model(self) -> str: ...

    @property
    def text_model(self) -> str: ...

    async def generate_structured(
        self,
        *,
        system_instruction: str,
        messages: list[tuple[str, str]],
        response_schema: dict[str, Any],
        temperature: float | None,
    ) -> tuple[Any, int, int]: ...

    def stream_text(self, request: TextStreamRequest) -> AsyncIterator[tuple[str, str | int]]: ...
