# Spec: deployment

**Status:** Implemented (Docker + Compose Mode A/Mode B)  
**Plan ref:** task-14 · **SPEC.md §8:** deployment

## Objective

Run the FastAPI health endpoint, Telegram long-polling engine, midnight-WIB
energy reset scheduler, PostgreSQL persistence, and ChromaDB persistence as one
reproducible container deployment. Credentials and database locations remain
environment-controlled; no secret is copied into the image.

## Public API

- `app/runtime.py`
  - `AppRuntime.start()` — initialise the database schema, start the reset
    scheduler, and start Telegram polling.
  - `AppRuntime.stop()` — stop polling, shut down the scheduler, and dispose the
    async database engine.
  - `build_runtime(settings)` — construct production services from typed
    `Settings`.
- `app/main.py`
  - `create_app(runtime=None)` — build the FastAPI app; an injected runtime is
    bracketed by the ASGI lifespan.
  - `create_production_app()` — uvicorn factory for the real runtime.
  - `GET /health` — process liveness response `{"status": "ok"}`.
- `Dockerfile`
  - Runs as unprivileged `appuser`.
  - Starts `uvicorn --factory app.main:create_production_app`.
  - Exposes port `8000` and checks `/health`.
- `docker-compose.yml`
  - Mode A: app + bundled PostgreSQL and Ollama services.
  - Named `pg_data`, `chroma_data`, and `ollama_data` persistent volumes.
  - `ollama` (pinned `ollama/ollama:0.40.6`) serves embeddings; the one-shot
    `ollama-init` service pulls `snowflake-arctic-embed2` and exits.
  - The app starts only after `postgres` is healthy and `ollama-init` completed
    successfully.
- `docker-compose.external-postgres.yml`
  - Mode B: removes the bundled PostgreSQL service and app dependency using
    Compose `!reset` tags, then attaches the app to `${EXTERNAL_NETWORK}`.
- `docker-compose.external-ollama.yml`
  - Mode B for embeddings: resets `ollama` and `ollama-init`, clears the app's
    `depends_on`, and requires `${OLLAMA_BASE_URL:?}`.
  - Combinable with the external-PostgreSQL override.

## Dependencies

- Python 3.12 slim image (`python:3.12-slim`).
- Pinned dependencies resolved by `uv` from `pyproject.toml` and `uv.lock`;
  the image installs them with `uv sync --frozen --no-dev`.
- PostgreSQL 16 for Mode A, or a reachable PostgreSQL instance for Mode B.
- Ollama serving `snowflake-arctic-embed2` for embeddings — bundled in Mode A,
  external in Mode B (set `OLLAMA_BASE_URL`).
- Docker Compose v2.24+ for the `!reset` override tags.
- Environment variables documented in `.env.example`.

## Continuous integration and release gates

Two GitHub Actions workflows enforce the same quality bar before anything ships.

`.github/workflows/ci.yml` runs on pull requests into `staging` and `main`, and
on pushes to `staging`: `ruff check`, `ruff format --check`, unit tests
(`-m "not integration"`), and integration tests
(`-m "integration and not requires_ollama"`) against a PostgreSQL 16 service
container. `FAIL_ON_SKIP=1` makes the session fail on any unexpected skip, so a
green run means every selected test actually executed.

`.github/workflows/release.yml` runs on `v*` tags: the same quality job, a
gitleaks scan scoped to the released commit's history, and only then a Docker
Hub push tagged from the semver tag (`DOCKERHUB_USERNAME` repository variable,
`DOCKERHUB_TOKEN` secret).

Branch promotion is `dev` → `staging` → `main`, and a release tag is cut from
`main`. A repository ruleset protects `staging` and `main`: no direct pushes,
no force pushes, no deletion, a pull request is required, and the
`Lint, format, unit & integration` check must pass on an up-to-date branch.
`dev` and any experimental branches stay unprotected.

`main` has no history shared with `dev` yet, so the first promotion needs a
one-time bootstrap: push `staging`'s commit to `main` (temporarily scoping the
ruleset, or with an authorised bypass), after which every later change reaches
`main` through a pull request like any other branch. Until that bootstrap
happens the default branch is empty.

Ollama is not available in CI. Tests that need a live embedding endpoint carry
`@pytest.mark.requires_ollama` and are deselected explicitly by the marker
expression above — never skipped implicitly at runtime.

Local test database: any PostgreSQL reachable through `TEST_DATABASE_URL`
defaults to `postgresql+asyncpg://diabetabot:diabetabot@localhost:5432/diabetabot_test`.

## Data Flow

```text
uvicorn factory
  → create_production_app()
  → build_runtime(Settings)
  → FastAPI lifespan startup
      → init_db(engine)
      → scheduler.start()  [00:00 Asia/Jakarta]
      → Telegram polling task
          → getUpdates
          → per-update AsyncSession
          → TelegramUpdateHandler
  → /health
  → FastAPI lifespan shutdown
      → stop polling
      → scheduler.shutdown()
      → engine.dispose()
```

### Mode A — bundled PostgreSQL and Ollama

```bash
cp .env.example .env
# Set Telegram and LLM credentials in .env. Embedding defaults point at the
# bundled Ollama service: OLLAMA_BASE_URL=http://ollama:11434,
# EMBEDDING_MODEL_NAME=snowflake-arctic-embed2, EMBEDDING_DIM=1024.
# Keep DATABASE_URL pointed at the compose service:
# postgresql+asyncpg://diabetabot:diabetabot@postgres:5432/diabetabot
docker compose up -d --build
```

The app waits for the PostgreSQL healthcheck and the completed `ollama-init`
model pull before starting. ChromaDB is persisted at `/app/chroma_db` through
the `chroma_data` volume; PostgreSQL data is persisted through `pg_data`;
Ollama models are persisted through `ollama_data`.

### Mode B — external PostgreSQL

```bash
export EXTERNAL_NETWORK=the_existing_docker_network
# In .env, set DATABASE_URL to the externally reachable host.
docker compose \
  -f docker-compose.yml \
  -f docker-compose.external-postgres.yml \
  up -d --build
```

The override removes the local `postgres` service and the app's `depends_on`
entry. The external database host must be reachable from the external Docker
network; application code is unchanged.

### Mode B — external Ollama

```bash
export OLLAMA_BASE_URL=http://the_external_host:11434
docker compose -f docker-compose.yml -f docker-compose.external-ollama.yml up -d --build
```

The override removes the bundled `ollama` and `ollama-init` services and
requires `OLLAMA_BASE_URL` to be set. Both overrides can be combined to run
the app against external PostgreSQL and external Ollama at the same time.

## Edge Cases

- Missing or sentinel LLM credentials fail production service construction;
  the image contains no fallback production credentials.
- A missing external network fails Compose validation rather than silently
  starting the app without database connectivity.
- A missing `OLLAMA_BASE_URL` in the external-Ollama override fails Compose
  validation rather than silently pointing at the removed bundled service.
- A PostgreSQL outage after startup is handled by SQLAlchemy's
  `pool_pre_ping`; the runtime still owns and disposes the engine at shutdown.
- A missed midnight scheduler tick is safe because `EnergyService.ensure_today`
  lazily seeds the current WIB date and reset rows are idempotent.
- FastAPI tests can pass a fake runtime to `create_app`; no network or Telegram
  calls are needed to verify lifespan behavior.

## Acceptance Criteria

- [x] Docker image installs the pinned runtime and runs as non-root.
- [x] Container healthcheck probes `GET /health`.
- [x] Production entrypoint wires database, scheduler, Telegram polling, RAG,
  LLM, registration, and energy services through `AppRuntime`.
- [x] Mode A starts app + PostgreSQL with a healthy database dependency.
- [x] ChromaDB and PostgreSQL use persistent named volumes in Mode A.
- [x] Mode B uses an external PostgreSQL network and requires no code change.
- [x] Mode B uses an external Ollama endpoint and requires no code change.
- [x] Shutdown stops polling and scheduler and disposes the database engine.
- [x] Image builds from `python:3.12-slim` and installs the locked dependency
  set with `uv sync --frozen --no-dev`.
- [x] CI blocks `staging` merges on lint, format, unit, and integration failures.
- [x] A skipped test fails the CI run instead of producing a green build.
- [x] A `v*` tag publishes the image to Docker Hub only after quality and
  gitleaks gates pass.

## Boundaries

**Always:** keep all credentials in environment variables; run the production
factory in the container; preserve persistent data volumes; use the same
`Settings` and service wiring in both database modes.

**Ask first:** changing the database image/version, adding a migration system,
adding a reverse proxy/TLS surface, or introducing another deployment service.

**Never:** bake `.env`, Telegram tokens, LLM keys, database passwords, or
ChromaDB data into the image; require a public webhook for Telegram; or change
application code merely to switch between Mode A and Mode B.
