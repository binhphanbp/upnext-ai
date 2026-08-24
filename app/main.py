from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import (
    health,
    internal_companies,
    internal_embeddings,
    internal_job_posts,
    internal_llm,
    internal_research,
)
from app.core.config import get_settings
from app.core.observability import configure_logging, configure_tracing, request_id_middleware

logger = structlog.get_logger(__name__)

# Every error body this service emits nests its code under `detail`, because that
# is the only shape the backend adapter can read: it does `parsed.detail?.code`
# and matches the result against a fixed allow-list. A code placed anywhere else
# is invisible to it, and the adapter then falls back to guessing from the HTTP
# status -- which is exactly how a 422 used to become AI_INVALID_OUTPUT.
_INTERNAL_ERROR_BODY: dict[str, Any] = {
    "detail": {"code": "AI_SERVICE_UNAVAILABLE", "message": "Internal AI service error."}
}


def _redacted_errors(error: RequestValidationError) -> list[dict[str, Any]]:
    """Describe a validation failure without repeating any caller payload.

    Pydantic attaches the offending value to every error as `input`, and `ctx`
    can carry it too. For this service that value is a recruiter's prompt or the
    base64 of their private document, so both keys are dropped before the failure
    reaches a log line. `loc`/`type`/`msg` are enough to identify the mismatch.
    """

    return [
        {"loc": list(item.get("loc", ())), "type": item.get("type"), "msg": item.get("msg")}
        for item in error.errors()
    ]


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="UpNext AI Internal API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.middleware("http")(request_id_middleware)
    app.include_router(health.router)
    app.include_router(internal_llm.router)
    app.include_router(internal_job_posts.router)
    app.include_router(internal_embeddings.router)
    app.include_router(internal_companies.router)
    app.include_router(internal_research.router)
    configure_tracing(app, settings.environment, settings.otel_exporter_otlp_endpoint)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, error: RequestValidationError) -> JSONResponse:
        """Report a rejected request as our fault, never as the recruiter's.

        FastAPI's default handler answers 422 with `detail` as a *list*, which
        breaks the contract in two ways at once. The backend cannot find a code
        in a list, so it guesses from the status: 422 maps to AI_INVALID_OUTPUT,
        which is the one code that must never fail over. A schema mismatch --
        our bug -- therefore reached the recruiter as "the AI could not read your
        content" with no second provider tried.

        `upnext-be` is the only caller and it builds every body from its own code,
        so a validation failure means the two services disagree about the schema.
        That is an outage of this service, hence 500 and a failover-eligible code.
        The default handler also echoed the rejected value back; see
        `_redacted_errors` for why nothing derived from the payload is returned
        or logged.
        """

        logger.warning("internal request failed validation", errors=_redacted_errors(error))
        return JSONResponse(status_code=500, content=_INTERNAL_ERROR_BODY)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_, error: StarletteHTTPException) -> JSONResponse:
        """Normalise any error body that is not already a coded object.

        Routes raise `HTTPException` with a `{"code": ..., "message": ...}` detail
        and those pass through untouched. But FastAPI and Starlette raise their
        own with a plain *string* detail -- an unparseable request body gives
        `400 {"detail": "There was an error parsing the body"}`, and routing
        errors give 404/405 the same way. A string is no more readable to the
        backend than a list: it finds no code, guesses from the status, and 400
        means AI_INVALID_OUTPUT -- no failover, and the recruiter is told their
        content is unreadable because two services disagreed about a byte.

        The status is preserved so HTTP semantics stay honest; only the body is
        rewritten, which is enough for the backend to read a real code.
        """

        detail = error.detail
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(
                status_code=error.status_code,
                content={"detail": detail},
                headers=getattr(error, "headers", None),
            )
        logger.warning(
            "non-coded http error normalised",
            status_code=error.status_code,
            detail_type=type(detail).__name__,
        )
        return JSONResponse(status_code=error.status_code, content=_INTERNAL_ERROR_BODY)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_, error: Exception) -> JSONResponse:
        # Do not leak provider/configuration details to a caller, even on the private network.
        logger.error("unhandled internal error", error_type=type(error).__name__)
        return JSONResponse(status_code=500, content=_INTERNAL_ERROR_BODY)

    return app


app = create_app()
