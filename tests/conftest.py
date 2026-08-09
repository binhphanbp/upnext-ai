from __future__ import annotations

import os
from collections.abc import Generator

# Must exist before importing app.main because the app validates settings at boot.
os.environ.setdefault(
    "AI_INTERNAL_JWT_SECRET", "test-internal-secret-that-is-at-least-32-characters"
)
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")
os.environ.setdefault("AI_ENVIRONMENT", "test")

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import clear_provider_cache
from app.core.config import get_settings
from app.main import create_app


@pytest.fixture(autouse=True)
def reset_cached_settings() -> Generator[None, None, None]:
    get_settings.cache_clear()
    clear_provider_cache()
    yield
    get_settings.cache_clear()
    clear_provider_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
