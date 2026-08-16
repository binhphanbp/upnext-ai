from __future__ import annotations

import base64
import binascii
from typing import Any, Literal

from pydantic import Field, model_validator

from app.contracts.llm import StrictModel

_MAX_SOURCE_FILE_BYTES = 8 * 1024 * 1024
_MAX_BASE64_CHARACTERS = 12_000_000


class JobPostSourceFile(StrictModel):
    """A recruiter-uploaded JD forwarded only by the authenticated backend."""

    mime_type: Literal[
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp",
    ] = Field(alias="mimeType")
    base64_data: str = Field(min_length=4, max_length=_MAX_BASE64_CHARACTERS, alias="base64Data")

    @model_validator(mode="after")
    def validate_encoded_size(self) -> JobPostSourceFile:
        try:
            decoded = base64.b64decode(self.base64_data, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("file must be valid base64") from error
        if not decoded or len(decoded) > _MAX_SOURCE_FILE_BYTES:
            raise ValueError(f"file must be between 1 and {_MAX_SOURCE_FILE_BYTES} bytes")
        return self

    def content(self) -> bytes:
        """Decode at the provider boundary; never persist or log source data."""

        return base64.b64decode(self.base64_data, validate=True)


class JobPostExtractionRequest(StrictModel):
    """Narrow internal contract for JD extraction, not a public upload API."""

    system_instruction: str = Field(min_length=1, max_length=30_000, alias="systemInstruction")
    prompt: str = Field(min_length=1, max_length=50_000)
    response_schema: dict[str, Any] = Field(alias="responseSchema")
    file: JobPostSourceFile | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    model_tier: Literal["fast", "quality"] = Field(default="quality", alias="modelTier")
    execution_profile: Literal["interactive", "batch"] = Field(
        default="interactive", alias="executionProfile"
    )


class JobPostGenerationRequest(StrictModel):
    """Narrow internal contract for generating a recruiter JD draft.

    A JD generation run is intentionally a single prompt, not a general chat.
    This keeps the capability's context budget, permission and observability
    independent from Candidate Copilot.
    """

    system_instruction: str = Field(min_length=1, max_length=30_000, alias="systemInstruction")
    prompt: str = Field(min_length=1, max_length=50_000)
    response_schema: dict[str, Any] = Field(alias="responseSchema")
    temperature: float | None = Field(default=None, ge=0, le=2)
    # Keep the rollout comparable with the direct recruiter flow: this
    # capability always uses the quality model in one interactive request.
    model_tier: Literal["quality"] = Field(default="quality", alias="modelTier")
    execution_profile: Literal["interactive"] = Field(
        default="interactive", alias="executionProfile"
    )
