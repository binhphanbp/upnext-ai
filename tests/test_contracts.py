import pytest
from pydantic import ValidationError

from app.contracts.llm import StructuredRequest


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
