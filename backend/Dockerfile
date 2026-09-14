# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.9.26 AS uv

FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.12-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 neuronscope \
    && useradd --system --uid 10001 --gid neuronscope --home-dir /nonexistent neuronscope

WORKDIR /app
COPY --from=builder --chown=neuronscope:neuronscope /app/.venv /app/.venv
COPY --from=builder --chown=neuronscope:neuronscope /app/src /app/src

USER neuronscope
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=4 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)"]

CMD ["uvicorn", "neuronscope.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
