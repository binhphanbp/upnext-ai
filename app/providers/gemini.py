from __future__ import annotations

import asyncio
import json
import math
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
    ProviderRegionBlockedError,
    ProviderTimeoutError,
    StructuredExecutionProfile,
    StructuredModelTier,
)
from app.providers.json_schema import normalize_response_json_schema

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

    @property
    def embedding_model(self) -> str:
        return self._settings.embedding_model

    def structured_model_for(self, model_tier: StructuredModelTier) -> str:
        return (
            self._settings.quality_structured_model
            if model_tier == "quality"
            else self.structured_model
        )

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
        model_tier: StructuredModelTier = "fast",
        execution_profile: StructuredExecutionProfile = "interactive",
    ) -> tuple[Any, int, int]:
        return await self._generate_structured(
            system_instruction=system_instruction,
            contents=self._contents(messages),
            response_schema=response_schema,
            temperature=temperature,
            model_tier=model_tier,
            execution_profile=execution_profile,
        )

    async def generate_structured_with_file(
        self,
        *,
        system_instruction: str,
        prompt: str,
        response_schema: dict[str, Any],
        file: tuple[str, bytes] | None,
        temperature: float | None,
        model_tier: StructuredModelTier = "quality",
        execution_profile: StructuredExecutionProfile = "interactive",
    ) -> tuple[Any, int, int]:
        parts: list[types.Part] = []
        if file is not None:
            mime_type, content = file
            parts.append(types.Part.from_bytes(data=content, mime_type=mime_type))
        parts.append(types.Part.from_text(text=prompt))
        return await self._generate_structured(
            system_instruction=system_instruction,
            contents=[types.Content(role="user", parts=parts)],
            response_schema=response_schema,
            temperature=temperature,
            model_tier=model_tier,
            execution_profile=execution_profile,
        )

    async def _generate_structured(
        self,
        *,
        system_instruction: str,
        contents: list[types.Content],
        response_schema: dict[str, Any],
        temperature: float | None,
        model_tier: StructuredModelTier,
        execution_profile: StructuredExecutionProfile,
    ) -> tuple[Any, int, int]:
        if not self._client:
            raise ProviderNotConfiguredError()

        try:
            normalized_schema = normalize_response_json_schema(response_schema)
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0 if temperature is None else temperature,
                response_mime_type="application/json",
                response_json_schema=normalized_schema,
            )
            timeout_seconds = (
                self._settings.batch_structured_timeout_seconds
                if execution_profile == "batch"
                else self._settings.structured_timeout_seconds
            )
            async with asyncio.timeout(timeout_seconds):
                response = await self._client.aio.models.generate_content(
                    model=self.structured_model_for(model_tier),
                    contents=contents,
                    config=config,
                )
        except TimeoutError as error:
            raise ProviderTimeoutError() from error
        except ValueError as error:
            logger.warning("gemini_structured_schema_invalid")
            raise ProviderError() from error
        except errors.ClientError as error:
            raise self._map_error(error) from error
        except Exception as error:  # noqa: BLE001 - provider boundary normalizes provider failures.
            logger.exception("gemini_structured_request_failed")
            raise ProviderError() from error

        value = response.parsed
        if value is None:
            text = (response.text or "").strip()
            if not text:
                raise ProviderInvalidOutputError()
            try:
                value = json.loads(text)
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

    async def embed_text(self, *, text: str, dimensions: int) -> list[float]:
        if not self._client:
            raise ProviderNotConfiguredError()
        if dimensions != self._settings.embedding_dimensions:
            raise ProviderInvalidOutputError()

        try:
            config = types.EmbedContentConfig(output_dimensionality=dimensions)
            async with asyncio.timeout(self._settings.embedding_timeout_seconds):
                response = await self._client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=config,
                )
        except TimeoutError as error:
            raise ProviderTimeoutError() from error
        except errors.ClientError as error:
            raise self._map_error(error) from error
        except ProviderError:
            raise
        except Exception as error:  # noqa: BLE001 - provider boundary normalizes provider failures.
            logger.exception("gemini_embedding_request_failed")
            raise ProviderError() from error

        embeddings = getattr(response, "embeddings", None)
        values = getattr(embeddings[0], "values", None) if embeddings else None
        if not isinstance(values, list) or len(values) != dimensions:
            raise ProviderInvalidOutputError()
        vector = [float(value) for value in values]
        if not all(math.isfinite(value) for value in vector):
            raise ProviderInvalidOutputError()
        magnitude = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(magnitude) or magnitude <= 0:
            raise ProviderInvalidOutputError()
        return [value / magnitude for value in vector]

    @staticmethod
    def _map_error(error: errors.ClientError) -> ProviderError:
        status = getattr(error, "code", None)
        if status == 429:
            return ProviderRateLimitError()
        if GeminiProvider._is_region_block(error):
            # Logged without the provider message: it is operator-facing
            # infrastructure detail, and the message can echo request context.
            logger.error("gemini_region_blocked", provider_status=getattr(error, "status", None))
            return ProviderRegionBlockedError()
        return ProviderError()

    @staticmethod
    def _is_region_block(error: errors.ClientError) -> bool:
        """Detect a geography refusal rather than a request-level fault.

        Gemini answers an unsupported deployment region with HTTP 400 and
        `FAILED_PRECONDITION`, which is otherwise indistinguishable from an
        ordinary bad request. The message is checked as well because
        FAILED_PRECONDITION is a general-purpose status the provider also
        uses for unrelated preconditions such as missing billing.
        """

        if getattr(error, "status", None) != "FAILED_PRECONDITION":
            return False
        message = str(getattr(error, "message", "") or "").lower()
        return "location is not supported" in message
