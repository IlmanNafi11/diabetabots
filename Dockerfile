# syntax=docker/dockerfile:1

FROM python:3.12-slim

# uv is the single source of truth for dependency resolution (uv.lock).
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first so the layer caches across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The runtime app, the delivered TKPI+GI dataset, and the indexing script all
# ship in the image so a deployer can populate ChromaDB without cloning the repo.
COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint

# Run as an unprivileged user; the app never needs to write outside its volumes.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/chroma_db \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

ENTRYPOINT ["docker-entrypoint"]
CMD ["serve"]
