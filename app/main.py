from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes import health, internal_embeddings, internal_llm
from app.core.config import get_settings
from app.core.observability import configure_logging, configure_tracing, request_id_middleware


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
    app.include_router(internal_embeddings.router)
    configure_tracing(app, settings.environment, settings.otel_exporter_otlp_endpoint)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_, __: Exception) -> JSONResponse:
        # Do not leak provider/configuration details to a caller, even on the private network.
        return JSONResponse(
            status_code=500,
            content={"code": "AI_INTERNAL_ERROR", "message": "Internal AI service error."},
        )

    return app


app = create_app()
