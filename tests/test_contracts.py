import pytest
from pydantic import ValidationError

from app.contracts.llm import StructuredRequest


def test_structured_request_defaults_to_fast_model_tier() -> None:
    request = StructuredRequest.model_validate(
        {
            "systemInstruction": "x",
            "messages": [{"role": "user", "text": "x"}],
            "responseSchema": {},
        }
    )

    assert request.model_tier == "fast"


def test_structured_request_accepts_only_controlled_model_tiers() -> None:
    request = StructuredRequest.model_validate(
        {
            "systemInstruction": "x",
            "messages": [{"role": "user", "text": "x"}],
            "responseSchema": {},
            "modelTier": "quality",
        }
    )
    assert request.model_tier == "quality"

    with pytest.raises(ValidationError):
        StructuredRequest.model_validate(
            {
                "systemInstruction": "x",
                "messages": [{"role": "user", "text": "x"}],
                "responseSchema": {},
                "modelTier": "gemini-arbitrary-model",
            }
        )


def test_structured_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate(
            {
                "systemInstruction": "x",
                "messages": [{"role": "user", "text": "x"}],
                "responseSchema": {},
                "untrustedField": "must not silently reach a provider",
            },
        )
