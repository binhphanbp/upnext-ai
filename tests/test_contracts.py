import pytest
from pydantic import ValidationError

from app.contracts.llm import StructuredRequest, TextStreamRequest


def test_structured_request_defaults_to_fast_model_tier() -> None:
    request = StructuredRequest.model_validate(
        {
            "systemInstruction": "x",
            "messages": [{"role": "user", "text": "x"}],
            "responseSchema": {},
        }
    )

    assert request.model_tier == "fast"
    assert request.execution_profile == "interactive"


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


def test_structured_request_allows_larger_payload_only_for_batch_profile() -> None:
    payload = {
        "systemInstruction": "x",
        "messages": [{"role": "user", "text": "x" * 20_001}],
        "responseSchema": {},
    }

    with pytest.raises(ValidationError):
        StructuredRequest.model_validate(payload)

    request = StructuredRequest.model_validate({**payload, "executionProfile": "batch"})
    assert request.execution_profile == "batch"


def test_text_stream_request_keeps_the_interactive_payload_budget() -> None:
    with pytest.raises(ValidationError):
        TextStreamRequest.model_validate(
            {
                "systemInstruction": "x",
                "messages": [{"role": "user", "text": "x" * 20_001}],
            }
        )
