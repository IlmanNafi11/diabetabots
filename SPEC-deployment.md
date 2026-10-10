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
  - Mode A: app + bundled PostgreSQL service.
  - Named `pg_data` and `chroma_data` persistent volumes.
- `docker-compose.external-postgres.yml`
  - Mode B: removes the bundled PostgreSQL service and app dependency using
    Compose `!reset` tags, then attaches the app to `${EXTERNAL_NETWORK}`.

## Dependencies

- Python 3.11 slim image.
- Pinned dependencies from `requirements.txt`.
- PostgreSQL 16 for Mode A, or a reachable PostgreSQL instance for Mode B.
- Docker Compose v2.24+ for the `!reset` override tags.
- Environment variables documented in `.env.example`.

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

### Mode A — bundled PostgreSQL

```bash
cp .env.example .env
# Set Telegram, LLM, and embedding credentials in .env.
# Keep DATABASE_URL pointed at the compose service:
# postgresql+asyncpg://diabetabot:diabetabot@postgres:5432/diabetabot
docker compose up -d --build
```

The app waits for the PostgreSQL healthcheck before starting. ChromaDB is
persisted at `/app/chroma_db` through the `chroma_data` volume; PostgreSQL data
is persisted through `pg_data`.

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

## Edge Cases

- Missing or sentinel LLM credentials fail production service construction;
  the image contains no fallback production credentials.
- A missing external network fails Compose validation rather than silently
  starting the app without database connectivity.
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
- [x] Shutdown stops polling and scheduler and disposes the database engine.

## Boundaries

**Always:** keep all credentials in environment variables; run the production
factory in the container; preserve persistent data volumes; use the same
`Settings` and service wiring in both database modes.

**Ask first:** changing the database image/version, adding a migration system,
adding a reverse proxy/TLS surface, or introducing another deployment service.

**Never:** bake `.env`, Telegram tokens, LLM keys, database passwords, or
ChromaDB data into the image; require a public webhook for Telegram; or change
application code merely to switch between Mode A and Mode B.
