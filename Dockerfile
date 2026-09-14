# syntax=docker/dockerfile:1.7

FROM node:24-bookworm-slim AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY frontend ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.9.26 AS uv

FROM python:3.12-slim-bookworm AS backend-builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock backend/README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY backend/src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.12-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEURONSCOPE_FRONTEND_DIST=/app/static \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PORT=8000

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 neuronscope \
    && useradd --system --uid 10001 --gid neuronscope --home-dir /nonexistent neuronscope

WORKDIR /app
COPY --from=backend-builder --chown=neuronscope:neuronscope /app/.venv /app/.venv
COPY --from=backend-builder --chown=neuronscope:neuronscope /app/src /app/src
COPY --from=frontend-builder --chown=neuronscope:neuronscope /build/frontend/dist /app/static

USER neuronscope
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=4 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/api/v1/health', timeout=2)"

CMD ["sh", "-c", "exec uvicorn neuronscope.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
