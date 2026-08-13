from __future__ import annotations

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.providers.base import EmbeddingProvider, LlmProvider
from app.providers.gemini import GeminiProvider


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LlmProvider:
    """Create a provider for the request after FastAPI resolves configuration.

    Settings is a Pydantic model and deliberately not used as an ``lru_cache``
    key. Caching it here caused a runtime ``TypeError`` before the first AI
    request. The provider client is lightweight; process-level configuration is
    still cached by ``get_settings``.
    """

    return GeminiProvider(settings)


def get_embedding_provider(settings: Settings = Depends(get_settings)) -> EmbeddingProvider:
    return GeminiProvider(settings)


def clear_provider_cache() -> None:
    """Compatibility hook for tests; provider instances are request scoped."""

    return None
