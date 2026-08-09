from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from google import genai
from google.genai import errors, types

from app.contracts.llm import TextStreamRequest
from app.core.config import Settings
from app.providers.base import (
    ProviderError,
    ProviderInvalidOutputError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = structlog.get_logger(__name__)


class GeminiProvider:
    """Gemini adapter isolated from UpNext business and HTTP concerns."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        api_key = (
            settings.gemini_api_key.get_secret_value().strip() if settings.gemini_api_key else ""
        )
        self._client = (
            genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version="v1"),
            )
            if api_key
            else None
        )

    @property
    def configured(self) -> bool:
        return self._client is not None

    @property
    def structured_model(self) -> str:
        return self._settings.structured_model

    @property
    def text_model(self) -> str:
        return self._settings.text_model

    def _contents(self, messages: list[tuple[str, str]]) -> list[types.Content]:
        return [
            types.Content(role=role, parts=[types.Part.from_text(text=text)])
            for role, text in messages
        ]

    async def generate_structured(
        self,
        *,
        system_instruction: str,
        messages: list[tuple[str, str]],
        response_schema: dict[str, Any],
        temperature: float | None,
    ) -> tuple[Any, int, int]:
        if not self._client:
            raise ProviderNotConfiguredError()

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0 if temperature is None else temperature,
            response_mime_type="application/json",
            response_json_schema=response_schema,
        )
        try:
            async with asyncio.timeout(self._settings.structured_timeout_seconds):
                response = await self._client.aio.models.generate_content(
                    model=self.structured_model,
                    contents=self._contents(messages),
                    config=config,
                )
        except TimeoutError as error:
            raise ProviderTimeoutError() from error
        except errors.ClientError as error:
            raise self._map_error(error) from error
        except Exception as error:  # noqa: BLE001 - provider boundary normalizes provider failures.
            logger.exception("gemini_structured_request_failed")
            raise ProviderError() from error

        text = (response.text or "").strip()
        if not text:
            raise ProviderInvalidOutputError()
        try:
            value = response.parsed if response.parsed is not None else json.loads(text)
        except (TypeError, ValueError) as error:
            raise ProviderInvalidOutputError() from error

        usage = response.usage_metadata
        return (
            value,
            int(usage.prompt_token_count or 0) if usage else 0,
            int(usage.candidates_token_count or 0) if usage else 0,
        )

    async def _stream(self, request: TextStreamRequest) -> AsyncIterator[tuple[str, str | int]]:
        if not self._client:
            raise ProviderNotConfiguredError()

        config = types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            temperature=0.4 if request.temperature is None else request.temperature,
            max_output_tokens=request.max_output_tokens or 2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        input_tokens = 0
        output_tokens = 0
        try:
            async with asyncio.timeout(self._settings.stream_timeout_seconds):
                stream = await self._client.aio.models.generate_content_stream(
                    model=self.text_model,
                    contents=self._contents([(item.role, item.text) for item in request.messages]),
                    config=config,
                )
                async for chunk in stream:
                    if chunk.text:
                        yield "text", chunk.text
                    usage = chunk.usage_metadata
                    if usage:
                        input_tokens = int(usage.prompt_token_count or input_tokens)
                        output_tokens = int(usage.candidates_token_count or output_tokens)
        except TimeoutError as error:
            raise ProviderTimeoutError() from error
        except errors.ClientError as error:
            raise self._map_error(error) from error
        except ProviderError:
            raise
        except Exception as error:  # noqa: BLE001 - provider boundary normalizes provider failures.
            logger.exception("gemini_stream_request_failed")
            raise ProviderError() from error

        yield "usage", input_tokens
        yield "output_tokens", output_tokens

    def stream_text(self, request: TextStreamRequest) -> AsyncIterator[tuple[str, str | int]]:
        return self._stream(request)

    @staticmethod
    def _map_error(error: errors.ClientError) -> ProviderError:
        status = getattr(error, "code", None)
        if status == 429:
            return ProviderRateLimitError()
        if status == 400:
            return ProviderInvalidOutputError()
        return ProviderError()
