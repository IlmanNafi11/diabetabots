# Spec: embedding-arctic-migration

**Status:** Implemented (2026-10-22) ·
**Intent ref:** `docs/intent/arctic-embedding-migration.md` (confirmed 2026-10-22)
**Branch:** `dev` · **Plan ref:** tasks/plan.md (arctic migration)

## Objective

Migrate the production RAG embedding path from `text-embedding-3-large`
(OpenAI, 3072 dim, paid, key blocked) to self-hosted `snowflake-arctic-embed2`
(Ollama, 1024 dim, cost 0) on branch `dev`, without changing retrieval
behaviour or collection naming.

**Users:** DiabetaBot operators/developers deploying via Docker Compose.

**Why now:** the experiment on `worktree/eksperimental` proved arctic-embed2
matches the binding baseline (mean P@5 = 0.800 on the five PRD §7.3 queries,
10/10 E2E scenarios PASS against `tkpi_gi_arctic`); the OpenAI key is blocked,
so the current production path is not queryable anyway.

**Success criteria**
- [x] `RAGService` and `scripts/index_data.py` embed via a self-hosted Ollama
      client (`/api/embed`, `snowflake-arctic-embed2:latest`, query prefix
      `"query: "`), no `OpenAIEmbeddings` in the production embedding path.
- [x] `tkpi_gi` collection re-indexed in place: 1146 documents, 1024 dim,
      cosine space; raw log `reports/logs/indexing-raw.log` records the
      migration run (document_count 1146).
- [x] Integration tests (retrieval P@5, nutrition accuracy) pass against the
      migrated `tkpi_gi` and skip when Ollama is unreachable.
- [x] 10 PRD §7.3 E2E scenarios PASS against the migrated `tkpi_gi`
      (`scripts/run_e2e.py`, unmodified).
- [x] `docker compose up -d` starts app + postgres + ollama + one-shot init
      (pulls `snowflake-arctic-embed2` into a named volume) and the app waits
      for the init service to complete; external instances selectable via
      compose overrides (postgres, ollama, or both).
- [x] Migration report written to `reports/` (not committed; `reports/` is
      gitignored) documenting snapshot -> drop -> reindex -> verification.

## Capability Map

| Module id | Responsibility | Depends on |
|---|---|---|
| `embedder-client` | Production `OllamaEmbedder` (httpx → `/api/embed`, prefix `query: `, configurable base URL), embedder wiring in `RAGService` + `index_data`, credential-gate adaptation, unit tests | — |
| `index-migration` | `tkpi_gi` cutover runbook (snapshot `chroma_data` → drop → reindex 1146 @1024), integration tests P@5 ≥ 0.800, spec updates (`SPEC-data-indexing`, `SPEC-rag-retrieval`) | `embedder-client` |
| `deploy-optional-services` | Compose bundled default (app + postgres + ollama + init pull), overrides `external-postgres` + `external-ollama` (combinable), `.env.example`, `SPEC-deployment` update | `embedder-client` |
| `retrieval-verification` | E2E 10 scenarios vs migrated `tkpi_gi` + migration report (not committed) | `index-migration`, `deploy-optional-services` |

Build order: `embedder-client` → `index-migration` + `deploy-optional-services`
(parallel) → `retrieval-verification`.

## Tech Stack

- Python 3.12, FastAPI, ChromaDB 1.5.9, httpx 0.28.1 (already in
  `pyproject.toml` — no new runtime dependencies).
- Ollama 0.40.6 (host) / `ollama/ollama` container, model
  `snowflake-arctic-embed2:latest` (1.2 GB).
- `langchain-openai` stays only for `ChatOpenAI` (chat LLM path unchanged).
- Docker Compose v2.24+ (`!reset` overrides), PostgreSQL 16.

## Commands

```bash
# Unit + integration suite (integration skips when Ollama is unreachable)
uv run pytest -q

# Lint + format
ruff check app scripts tests
ruff format --check app scripts tests

# Re-index the production collection (idempotent re-run; drop happens via the
# migration runbook below, not by this command)
uv run python -m scripts.index_data --dataset data/tkpi-gi-final.jsonl

# Migration runbook (snapshot -> drop -> reindex) — executed once, see
# SPEC index-migration section
scripts/migrate_embedding.py --confirm-drop-collection

# E2E: 10 PRD §7.3 scenarios against the migrated tkpi_gi
uv run python -m scripts.run_e2e --date YYYY-MM-DD

# Deployment — everything bundled (default)
docker compose up -d --build

# Deployment — external PostgreSQL, bundled Ollama
docker compose -f docker-compose.yml -f docker-compose.external-postgres.yml up -d --build

# Deployment — bundled PostgreSQL, external Ollama
docker compose -f docker-compose.yml -f docker-compose.external-ollama.yml up -d --build

# Deployment — both external
docker compose -f docker-compose.yml \
  -f docker-compose.external-postgres.yml \
  -f docker-compose.external-ollama.yml up -d --build
```

## Project Structure

```
app/
  data/embedder.py        → unchanged (Protocol + index_documents reused)
  services/ollama_embedder.py → NEW: OllamaEmbedder (httpx → /api/embed,
                            prefix "query: ", no network at construction)
  services/rag_service.py → wires OllamaEmbedder; OpenAIEmbeddings removed;
                            credential gate moved to LLM-only check
  config.py               → + ollama_base_url (default http://ollama:11434);
                            embedding_dim default 1024
  index_data.py           → _create_embedder returns OllamaEmbedder
  run_e2e.py              → unchanged (reads settings; tkpi_gi)
  migrate_embedding.py    → NEW: snapshot → drop → reindex runbook
tests/
  test_config.py          → updated defaults + ollama_base_url
  test_rag_service.py     → unchanged (mocks _make_embedder); assertion for
                            OllamaEmbedder wiring
  test_index_data.py      → unchanged (fake client)
  test_integration/
    test_retrieval_5queries.py → OllamaEmbedder + skip-if-unreachable
    test_nutrition_accuracy.py → same
  test_deployment.py      → compose contract updated (ollama + init services,
                            overrides)
docker-compose.yml               → app + postgres + ollama + ollama-init (default)
docker-compose.external-postgres.yml  → exists; unchanged contract
docker-compose.external-ollama.yml   → NEW override
.env.example                → OLLAMA_BASE_URL + updated comments/defaults
SPEC-data-indexing.md       → update embedding config (arctic, 1024)
SPEC-rag-retrieval.md       → update embedding config + dependency list
SPEC-deployment.md          → update compose topology (optional services)
SPEC-embedding-arctic-migration.md → this spec (committed)
reports/                    → migration + verification reports (gitignored)
docs/intent/arctic-embedding-migration.md → intent (gitignored)
```

## Code Style

Repository conventions (ruff, line-length 100, py312 target) apply unchanged.
Embedder follows the existing `EmbeddingClient` Protocol so
`app/data/embedder.py` stays untouched:

```python
class OllamaEmbedder:
    """LangChain-compatible embedder calling Ollama /api/embed."""

    def __init__(self, base_url: str, model: str = "snowflake-arctic-embed2:latest") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": list(texts)},
            timeout=...,
        )
        response.raise_for_status()
        return [list(map(float, vector)) for vector in response.json()["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"query: {text}"])[0]
```

- Query prefix `"query: "` is hardcoded (official arctic config; documents
  embed as-is, no passage prefix).
- Constructing the embedder performs **no network call**; errors surface at
  retrieve/index time.
- No new protocol, no abstraction beyond what exists.

## Testing Strategy

- **Unit (no network):** `tests/test_rag_service.py` already mocks
  `_make_embedder`; add a wiring test asserting `_make_embedder` returns an
  `OllamaEmbedder` configured from settings (with `_make_client` mocked).
  `tests/test_index_data.py` fake client unchanged. `test_config.py` updated
  for `ollama_base_url` + defaults.
- **Integration (network, marked `integration`, skip when Ollama unreachable):**
  `test_retrieval_5queries.py` and `test_nutrition_accuracy.py` embed via
  `OllamaEmbedder`; skip fixture probes `OLLAMA_BASE_URL/api/version`. P@5
  threshold stays 0.800.
- **Deployment contract:** `test_deployment.py` updated to assert the bundled
  compose contains `postgres`, `ollama`, `ollama-init` (with
  `condition: service_completed_successfully` on the app dependency),
  persistent volumes, and both external overrides.
- **E2E:** `scripts/run_e2e.py` unchanged; 10 scenarios must PASS against the
  migrated `tkpi_gi`. Evidence logs to `reports/logs/` (gitignored).
- Coverage expectations unchanged (`pytest-cov`, `app` source).

## Boundaries

**Always**

- Keep `tkpi_gi` as the production collection name; keep `recursive` chunking,
  cosine space, `rag_top_k`/`rag_candidate_k`, and `QUERY_CASES` untouched.
- Take a snapshot of `chroma_data` (Docker volume, or local `chroma_db/`) before
  dropping the old collection; document the exact restore steps in the runbook.
- Prefix `"query: "` on queries only; documents embed as-is.
- Re-index 1146 documents; verify `collection.count() == 1146` and log the run
  to `reports/logs/indexing-raw.log`.
- App must boot without contacting Ollama; embedding errors surface at
  retrieve/index time.
- Chat LLM path stays OpenAI-compatible (`LLM_*` env, `ChatOpenAI`).
- Update specs (`data-indexing`, `rag-retrieval`, `deployment`) in the same
  change as the code they describe.
- Migration and verification reports go to `reports/` and are **not committed**
  (`reports/` gitignored).

**Ask first**

- Changing the embedding model, query prefix, `QUERY_CASES`, top-k values,
  chunking strategy, or the P@5 threshold (0.800).
- Keeping a side-by-side A/B collection after cutover.
- Adding a new compose service beyond `ollama`/`ollama-init` (e.g. reverse
  proxy, TLS).
- Touching the pinned dependency set in `pyproject.toml`/`uv.lock` or
  `langchain-openai`.
- Merging any experimental artifact from `worktree/eksperimental`.
- Changing the DB image/version or adding a migration system.

**Never**

- Commit `reports/`, `docs/`, `.env`, or `chroma_db/`.
- Fabricate or hand-adjust P@5/E2E numbers or raw logs.
- Re-run or modify `app/scripts/run_spike.py`, `tests/test_spike.py`, or
  `SPEC-spike-embedding.md` (Iterasi 0 history stays intact).
- Keep `OpenAIEmbeddings` in the production embedding path.
- Delete the old collection before the snapshot is verified restorable.

## Open Questions

1. Rollback drill (restore snapshot) documented in the runbook but not
   executed as a live test unless the user asks.

## Resolved during planning

- `OllamaEmbedder` lives in a new `app/services/ollama_embedder.py` so
  `app/data/embedder.py` stays untouched and both `rag_service` and
  `index_data` share one implementation.
- `ollama-init` runs the server image with a one-shot entrypoint that waits
  for the server health then `ollama pull snowflake-arctic-embed2`.
- `OLLAMA_BASE_URL` defaults to `http://ollama:11434` (compose service name);
  external override files force `${OLLAMA_BASE_URL:?}` into the app.