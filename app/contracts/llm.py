from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LlmMessage(StrictModel):
    role: Literal["user", "model"]
    text: str = Field(min_length=1, max_length=20_000)


class StructuredRequest(StrictModel):
    system_instruction: str = Field(min_length=1, max_length=30_000, alias="systemInstruction")
    messages: list[LlmMessage] = Field(min_length=1, max_length=50)
    response_schema: dict[str, Any] = Field(alias="responseSchema")
    temperature: float | None = Field(default=None, ge=0, le=2)
    model_tier: Literal["fast", "quality"] = Field(default="fast", alias="modelTier")


class StructuredResponse(StrictModel):
    value: Any
    input_tokens: int = Field(ge=0, alias="inputTokens")
    output_tokens: int = Field(ge=0, alias="outputTokens")
    model: str


class TextStreamRequest(StrictModel):
    system_instruction: str = Field(min_length=1, max_length=30_000, alias="systemInstruction")
    messages: list[LlmMessage] = Field(min_length=1, max_length=50)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1, le=4_096, alias="maxOutputTokens")


class TextChunk(StrictModel):
    text: str = Field(min_length=1)


class UsageChunk(StrictModel):
    input_tokens: int = Field(ge=0, alias="inputTokens")
    output_tokens: int = Field(ge=0, alias="outputTokens")
    model: str
