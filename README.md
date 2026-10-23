# DiabetaBot

Telegram chatbot for personal nutrition and glycemic feasibility analysis. FastAPI plus Telegram long polling, PostgreSQL persistence, ChromaDB RAG over the TKPI plus GI dataset, self hosted Ollama embeddings, TDEE and energy tracking.

## Requirements

Docker Engine with Compose v2.24 or newer, 16 GB RAM, a Telegram bot token from @BotFather, and an OpenAI compatible LLM endpoint.

## Deploy

```bash
cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN, LLM_*, POSTGRES_PASSWORD
docker compose up -d --build
docker compose run --rm app index
```

The stack starts Postgres, pulls the embedding model once, then boots the app. Database tables are created automatically. The `index` command embeds the bundled TKPI plus GI dataset into ChromaDB (1,146 entries, idempotent, takes minutes on CPU). Skip it and every food query answers "not found".

To use the published image instead of building locally:

```bash
DIABETABOT_VERSION=0.1.1 docker compose -f docker-compose.yml -f docker-compose.release.yml up -d
```

Check it works: `docker compose ps` (healthy), then `/start` in Telegram.

## Configuration

All settings are environment variables, see `.env.example`. The ones you must set: `TELEGRAM_BOT_TOKEN`, `LLM_BASE_URL`, `LLM_API_KEY`, `CHAT_MODEL_NAME`, `POSTGRES_PASSWORD`, `DATABASE_URL`. Defaults cover everything else.

Exposing the HTTP port publicly: add your hostname to `ALLOWED_HOSTS` (comma separated). An external Ollama host goes in `OLLAMA_BASE_URL` plus `OLLAMA_ALLOWED_HOSTS` (bare hostnames or IPs, no scheme or port). Details in `SECURITY.md`.

## Develop

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not integration" -q
```

## Release flow

`dev` to `staging` to `main` through pull requests. A `v*` tag cut from `main` runs quality plus secret scan, then pushes `ilmannafi/diabetabot` to Docker Hub.
