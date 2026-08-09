from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_llm_provider
from app.providers.base import LlmProvider

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    response: Response,
    provider: LlmProvider = Depends(get_llm_provider),
) -> dict[str, Any]:
    if not provider.configured:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "service": "upnext-ai",
            "reason": "provider_not_configured",
        }
    return {
        "status": "ok",
        "service": "upnext-ai",
        "models": {"structured": provider.structured_model, "text": provider.text_model},
    }
