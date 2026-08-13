from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)
_llm_scope = "llm:invoke"
_embedding_scope = "embedding:invoke"
_job_post_extraction_scope = "job-post:extract"
_job_post_generation_scope = "job-post:generate"


@dataclass(frozen=True)
class InternalPrincipal:
    subject: str
    run_id: str
    scopes: frozenset[str]


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": "AI_INTERNAL_UNAUTHORIZED",
            "message": "Internal service authentication failed.",
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def _scopes(value: object) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset(item for item in value.split(" ") if item)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return frozenset(value)
    return frozenset()


def _principal_for_scope(
    *,
    credentials: HTTPAuthorizationCredentials | None,
    settings: Settings,
    required_scope: str,
) -> InternalPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    try:
        claims: dict[str, Any] = jwt.decode(
            credentials.credentials,
            settings.internal_jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            audience=settings.internal_jwt_audience,
            issuer=settings.internal_jwt_issuer,
            options={"require": ["sub", "jti", "iat", "exp", "iss", "aud"]},
        )
    except jwt.PyJWTError as error:
        raise _unauthorized() from error

    try:
        issued_at = datetime.fromtimestamp(int(claims["iat"]), tz=UTC)
        expires_at = datetime.fromtimestamp(int(claims["exp"]), tz=UTC)
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise _unauthorized() from error

    if (expires_at - issued_at).total_seconds() > settings.internal_jwt_max_ttl_seconds:
        raise _unauthorized()
    if claims.get("environment") != settings.environment:
        raise _unauthorized()

    scopes = _scopes(claims.get("scope"))
    subject = claims.get("sub")
    run_id = claims.get("jti")
    if subject != "upnext-be" or not isinstance(run_id, str) or not run_id:
        raise _unauthorized()
    if required_scope not in scopes:
        raise _unauthorized()
    return InternalPrincipal(subject=subject, run_id=run_id, scopes=scopes)


async def require_internal_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> InternalPrincipal:
    return _principal_for_scope(
        credentials=credentials,
        settings=settings,
        required_scope=_llm_scope,
    )


async def require_embedding_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> InternalPrincipal:
    return _principal_for_scope(
        credentials=credentials,
        settings=settings,
        required_scope=_embedding_scope,
    )


async def require_job_post_extraction_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> InternalPrincipal:
    return _principal_for_scope(
        credentials=credentials,
        settings=settings,
        required_scope=_job_post_extraction_scope,
    )


async def require_job_post_generation_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> InternalPrincipal:
    return _principal_for_scope(
        credentials=credentials,
        settings=settings,
        required_scope=_job_post_generation_scope,
    )
