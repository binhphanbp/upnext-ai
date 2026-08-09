from __future__ import annotations

import logging
import sys
from collections.abc import Awaitable, Callable
from uuid import uuid4

import structlog
from fastapi import Request, Response
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

_tracing_configured = False


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric_level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )


def configure_tracing(app: object, environment: str, otlp_endpoint: str | None = None) -> None:
    global _tracing_configured
    if not _tracing_configured:
        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": "upnext-ai", "deployment.environment": environment}
            )
        )
        # Never emit traces to stdout: a prompt or model payload must not end up in
        # container logs. Export telemetry only when a trusted collector is configured.
        if otlp_endpoint:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        set_tracer_provider(provider)
        _tracing_configured = True
    FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]


async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    # Do not trust caller-provided identifiers as the primary trace identifier.
    request_id = str(uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        structlog.contextvars.clear_contextvars()
