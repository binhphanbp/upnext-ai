from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_llm_provider
from app.contracts.llm import (
    StructuredRequest,
    StructuredResponse,
    TextChunk,
    TextStreamRequest,
    UsageChunk,
)
from app.core.security import InternalPrincipal, require_internal_principal
from app.providers.base import LlmProvider, ProviderError

router = APIRouter(prefix="/internal/v1/llm", tags=["internal-llm"])


def _provider_exception(error: ProviderError) -> HTTPException:
    code = error.code
    status_code = status.HTTP_429_TOO_MANY_REQUESTS if code == "AI_MODEL_RATE_LIMIT" else 503
    if code == "AI_INVALID_OUTPUT":
        status_code = 502
    elif code == "AI_MODEL_TIMEOUT":
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": "AI provider could not complete this request."},
    )


@router.post("/structured", response_model=StructuredResponse, response_model_by_alias=True)
async def generate_structured(
    request: StructuredRequest,
    _: InternalPrincipal = Depends(require_internal_principal),
    provider: LlmProvider = Depends(get_llm_provider),
) -> StructuredResponse:
    try:
        value, input_tokens, output_tokens = await provider.generate_structured(
            system_instruction=request.system_instruction,
            messages=[(message.role, message.text) for message in request.messages],
            response_schema=request.response_schema,
            temperature=request.temperature,
            model_tier=request.model_tier,
        )
    except ProviderError as error:
        raise _provider_exception(error) from error
    return StructuredResponse(
        value=value,
        inputTokens=input_tokens,
        outputTokens=output_tokens,
        model=provider.structured_model_for(request.model_tier),
    )


def _sse(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_events(provider: LlmProvider, request: TextStreamRequest) -> AsyncIterator[str]:
    input_tokens = 0
    try:
        async for kind, payload in provider.stream_text(request):
            if kind == "text":
                yield _sse("text", TextChunk(text=str(payload)).model_dump())
            elif kind == "usage":
                input_tokens = int(payload)
            elif kind == "output_tokens":
                yield _sse(
                    "usage",
                    UsageChunk(
                        inputTokens=input_tokens,
                        outputTokens=int(payload),
                        model=provider.text_model,
                    ).model_dump(by_alias=True),
                )
    except ProviderError as error:
        # Status cannot change after SSE headers are sent. Preserve an explicit error frame.
        yield _sse(
            "error",
            {"code": error.code, "message": "AI provider could not complete this request."},
        )
    finally:
        yield _sse("done", {})


@router.post("/stream")
async def stream_text(
    request: TextStreamRequest,
    _: InternalPrincipal = Depends(require_internal_principal),
    provider: LlmProvider = Depends(get_llm_provider),
) -> StreamingResponse:
    if not provider.configured:
        raise _provider_exception(ProviderError())
    return StreamingResponse(
        _stream_events(provider, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
