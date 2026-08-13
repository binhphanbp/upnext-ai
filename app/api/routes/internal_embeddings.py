from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_embedding_provider
from app.contracts.embedding import (
    EMBEDDING_CACHE_KEY,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZATION,
    EmbeddingRequest,
    EmbeddingResponse,
)
from app.core.security import InternalPrincipal, require_embedding_principal
from app.providers.base import EmbeddingProvider, ProviderError

router = APIRouter(prefix="/internal/v1/embeddings", tags=["internal-embeddings"])


def _provider_exception(error: ProviderError) -> HTTPException:
    code = error.code
    status_code = status.HTTP_429_TOO_MANY_REQUESTS if code == "AI_MODEL_RATE_LIMIT" else 503
    if code == "AI_INVALID_OUTPUT":
        status_code = status.HTTP_502_BAD_GATEWAY
    elif code == "AI_MODEL_TIMEOUT":
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": "AI provider could not create the embedding."},
    )


@router.post("", response_model=EmbeddingResponse, response_model_by_alias=True)
async def create_embedding(
    request: EmbeddingRequest,
    _: InternalPrincipal = Depends(require_embedding_principal),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> EmbeddingResponse:
    try:
        vector = await provider.embed_text(text=request.text, dimensions=request.dimensions)
    except ProviderError as error:
        raise _provider_exception(error) from error
    return EmbeddingResponse(
        vector=vector,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        normalization=EMBEDDING_NORMALIZATION,
        cacheKey=EMBEDDING_CACHE_KEY,
    )
