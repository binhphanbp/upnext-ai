from __future__ import annotations

import base64
import binascii
from typing import Any, Literal

from pydantic import Field, model_validator

from app.contracts.llm import StrictModel

_MAX_SOURCE_FILE_BYTES = 8 * 1024 * 1024
_MAX_BASE64_CHARACTERS = 12_000_000


class CompanyLicenseSourceFile(StrictModel):
    """A recruiter-uploaded business licence forwarded only by the backend.

    Deliberately a separate model from the job-post source file even though the
    fields match today: a licence is a company registration document, so its
    accepted formats and size ceiling should be able to move without dragging
    the JD upload contract along with them.
    """

    mime_type: Literal[
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp",
    ] = Field(alias="mimeType")
    base64_data: str = Field(min_length=4, max_length=_MAX_BASE64_CHARACTERS, alias="base64Data")

    @model_validator(mode="after")
    def validate_encoded_size(self) -> CompanyLicenseSourceFile:
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


class CompanyLicenseExtractionRequest(StrictModel):
    """Narrow internal contract for reading a business licence.

    The file is required here, unlike JD extraction which also accepts pasted
    text: there is no meaningful licence extraction without the document, and
    accepting a text-only call would quietly turn this into a general prompt
    endpoint under a licence-scoped token.
    """

    system_instruction: str = Field(min_length=1, max_length=30_000, alias="systemInstruction")
    prompt: str = Field(min_length=1, max_length=50_000)
    response_schema: dict[str, Any] = Field(alias="responseSchema")
    file: CompanyLicenseSourceFile
    temperature: float | None = Field(default=None, ge=0, le=2)
    # A licence is read once during company onboarding and the fields drive
    # verification, so accuracy matters more than latency here.
    model_tier: Literal["quality"] = Field(default="quality", alias="modelTier")
    execution_profile: Literal["interactive"] = Field(
        default="interactive", alias="executionProfile"
    )
