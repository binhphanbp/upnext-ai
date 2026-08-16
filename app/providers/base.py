from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from app.contracts.llm import TextStreamRequest

StructuredModelTier = Literal["fast", "quality"]
StructuredExecutionProfile = Literal["interactive", "batch"]


class ProviderError(Exception):
    """Known provider failure that can be mapped to a stable internal error."""

    code = "AI_SERVICE_UNAVAILABLE"


class ProviderNotConfiguredError(ProviderError):
    code = "AI_SERVICE_UNAVAILABLE"


class ProviderRateLimitError(ProviderError):
    code = "AI_MODEL_RATE_LIMIT"


class ProviderRegionBlockedError(ProviderError):
    """The provider refuses to serve this deployment's geography.

    Distinct from a generic outage because retrying, failing over to another
    model tier, or fixing the request schema cannot resolve it -- the call is
    rejected before the payload is considered. Collapsing it into the generic
    error previously made a hard infrastructure block look like a transient
    or schema fault.
    """

    code = "AI_PROVIDER_REGION_BLOCKED"


class ProviderInvalidOutputError(ProviderError):
    code = "AI_INVALID_OUTPUT"


class ProviderTimeoutError(ProviderError):
    code = "AI_MODEL_TIMEOUT"


class LlmProvider(Protocol):
    @property
    def configured(self) -> bool: ...

    @property
    def structured_model(self) -> str: ...

    def structured_model_for(self, model_tier: StructuredModelTier) -> str: ...

    @property
    def text_model(self) -> str: ...

    async def generate_structured(
        self,
        *,
        system_instruction: str,
        messages: list[tuple[str, str]],
        response_schema: dict[str, Any],
        temperature: float | None,
        model_tier: StructuredModelTier,
        execution_profile: StructuredExecutionProfile,
    ) -> tuple[Any, int, int]: ...

    async def generate_structured_with_file(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_schema: dict[str, Any],
        file: tuple[str, bytes] | None,
        temperature: float | None,
        model_tier: StructuredModelTier,
        execution_profile: StructuredExecutionProfile,
    ) -> tuple[Any, int, int]: ...

    def stream_text(self, request: TextStreamRequest) -> AsyncIterator[tuple[str, str | int]]: ...


class EmbeddingProvider(Protocol):
    @property
    def configured(self) -> bool: ...

    @property
    def embedding_model(self) -> str: ...

    async def embed_text(self, *, text: str, dimensions: int) -> list[float]: ...
