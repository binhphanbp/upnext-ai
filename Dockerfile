FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system upnext && adduser --system --ingroup upnext --home /nonexistent upnext

COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m pip install --upgrade "pip==25.2" && python -m pip install .

USER upnext
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "from urllib.request import urlopen; response = urlopen('http://127.0.0.1:8000/health/live', timeout=3); raise SystemExit(0 if response.status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
