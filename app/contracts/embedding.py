from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_MODEL: Final[Literal["gemini-embedding-001"]] = "gemini-embedding-001"
EMBEDDING_DIMENSIONS: Final[Literal[768]] = 768
EMBEDDING_NORMALIZATION: Final[Literal["l2-v1"]] = "l2-v1"
EMBEDDING_CACHE_KEY: Final[Literal["gemini-embedding-001:768:l2-v1"]] = (
    "gemini-embedding-001:768:l2-v1"
)


class StrictEmbeddingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EmbeddingRequest(StrictEmbeddingModel):
    text: str = Field(min_length=1, max_length=12_000)
    dimensions: Literal[768] = EMBEDDING_DIMENSIONS


class EmbeddingResponse(StrictEmbeddingModel):
    vector: list[float] = Field(min_length=EMBEDDING_DIMENSIONS, max_length=EMBEDDING_DIMENSIONS)
    model: Literal["gemini-embedding-001"]
    dimensions: Literal[768]
    normalization: Literal["l2-v1"]
    cache_key: Literal["gemini-embedding-001:768:l2-v1"] = Field(alias="cacheKey")
