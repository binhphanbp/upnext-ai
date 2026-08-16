from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. No user-facing secret is accepted by this service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        validate_default=True,
    )

    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development", validation_alias="AI_ENVIRONMENT"
    )
    internal_jwt_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        validation_alias="AI_INTERNAL_JWT_SECRET",
    )
    internal_jwt_issuer: str = Field(default="upnext-be", validation_alias="AI_INTERNAL_JWT_ISSUER")
    internal_jwt_audience: str = Field(
        default="upnext-ai", validation_alias="AI_INTERNAL_JWT_AUDIENCE"
    )
    internal_jwt_max_ttl_seconds: int = Field(
        default=90, ge=30, le=300, validation_alias="AI_INTERNAL_JWT_MAX_TTL_SECONDS"
    )
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    structured_model: str = Field(
        default="gemini-2.5-flash-lite", validation_alias="AI_STRUCTURED_MODEL"
    )
    quality_structured_model: str = Field(
        default="gemini-2.5-flash", validation_alias="AI_QUALITY_STRUCTURED_MODEL"
    )
    text_model: str = Field(default="gemini-2.5-flash", validation_alias="AI_TEXT_MODEL")
    embedding_model: str = Field(
        default="gemini-embedding-001", validation_alias="AI_EMBEDDING_MODEL"
    )
    # Grounded answers are pinned to their own model rather than the fast/quality
    # tiers: measured on this workload, 3.x models answer from memory without
    # ever calling the search tool, which produces an ungrounded answer with no
    # citations at all. 2.5-pro searches every time and is faster here.
    grounded_model: str = Field(default="gemini-2.5-pro", validation_alias="AI_GROUNDED_MODEL")
    embedding_dimensions: int = Field(
        default=768, ge=768, le=768, validation_alias="AI_EMBEDDING_DIMENSIONS"
    )
    structured_timeout_seconds: int = Field(
        default=15, ge=1, le=60, validation_alias="AI_STRUCTURED_TIMEOUT_SECONDS"
    )
    batch_structured_timeout_seconds: int = Field(
        default=60, ge=10, le=120, validation_alias="AI_BATCH_STRUCTURED_TIMEOUT_SECONDS"
    )
    stream_timeout_seconds: int = Field(
        default=20, ge=1, le=90, validation_alias="AI_STREAM_TIMEOUT_SECONDS"
    )
    embedding_timeout_seconds: int = Field(
        default=20, ge=1, le=60, validation_alias="AI_EMBEDDING_TIMEOUT_SECONDS"
    )
    # A grounded run issues several web searches before answering; measured
    # round trips sit at 42-50s, so the usual 15-20s budgets abort every call.
    grounded_timeout_seconds: int = Field(
        default=75, ge=20, le=150, validation_alias="AI_GROUNDED_TIMEOUT_SECONDS"
    )
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @field_validator("internal_jwt_secret")
    @classmethod
    def validate_internal_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("AI_INTERNAL_JWT_SECRET must be at least 32 characters")
        return value

    @model_validator(mode="after")
    def validate_production_secret(self) -> Settings:
        secret = self.internal_jwt_secret.get_secret_value()
        if self.environment == "production" and secret.startswith("change-me"):
            raise ValueError("AI_INTERNAL_JWT_SECRET must not use a placeholder in production")
        return self

    @property
    def provider_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
