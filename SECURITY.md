# DiabetaBot — Security notes

This document records the security boundaries the service enforces and the
assumptions operators must satisfy before running it.

## Credential hygiene (fail-fast at start-up)

The runtime refuses to start with placeholder credentials. `Settings`
validates at construction time:

| Setting | Required | Guard |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | rejects `your-*` and empty values |
| `LLM_API_KEY` | yes | rejects `your-*` and empty values |
| `DATABASE_URL` | yes | no default; must use `postgresql+asyncpg://` |

Composer mode A (bundled postgres) binds `POSTGRES_PASSWORD` through
`docker-compose.yml`. There is **no fallback default** for
`POSTGRES_PASSWORD` — it must be set in `.env` or the environment, or compose
aborts with `POSTGRES_PASSWORD must be set in .env or environment`.

## Log redaction

Production log lines never print credentials, tokens, full URLs, or stack
traces. Exception details are captured locally but only the context (chat id,
operation name) is emitted to the log. Stack traces stay in the local
debug log and are never returned to the Telegram user; the user receives a
generic message instead (`Maaf, analisis gagal sementara`, etc.).

## Ollama base URL — SSRF allowlist

`OLLAMA_BASE_URL` is validated so the app cannot be tricked into talking to
an arbitrary internal host:

```
Allowed hosts:
  - ollama        (compose service name, mode A)
  - localhost     (loopback)
  - 127.0.0.1     (loopback)
```

Raw IP addresses other than `127.0.0.1` (including private ranges
`10.x.x.x`, `172.16-31.x.x`, `192.168.x.x`) and arbitrary public FQDNs are
rejected. Only `http://` and `https://` schemes are accepted.

## Prompt length cap

User questions are capped at `MAX_USER_QUESTION_CHARS` (default 512
characters) by the Telegram handler before reaching the LLM, protecting
against prompt-stuffing / token-exhaustion. Over-long questions receive a
short Bahasa Indonesia refusal instead of being forwarded to the model.

## Embedding client ownership

`OllamaEmbedder.close()` only shuts down a connection pool it created itself
(`client is None` at construction). A caller-supplied `httpx.Client` is left
intact — callers own their own clients. `RAGService.close()` releases the
process-lifetime embedder pool and clears the ChromaDB system cache; it is
invoked from `AppRuntime.stop()`.

## Migration safety

`scripts/migrate_embedding.py` performs an atomic staging swap:

- Aborts if a `_pre_migration` backup collection already exists (don't delete
  the last good backup on a re-run).
- Restore failures are raised, never swallowed.
- Dry-run builds and verifies staging without touching the serving
  collection.
- The preflight refuses to run while the app answers on `/health`
  (ChromaDB does not support concurrent multi-process access).

## Out of scope

- Telegram message contents are not encrypted at rest (out-of-scope for this
  iteration).
- The bundled Ollama service has no TLS termination by default; run behind a
  reverse proxy with TLS for external exposure.
