# Spec: data-indexing

**Status:** Implemented · **Plan ref:** task-07 · **SPEC.md §8:** data-indexing

## Objective

Parse the provided TKPI+GI dataset (`data/tkpi-gi-final.jsonl`, 1.146 entries —
provided, never modified) into one embedding document per food entry and index
them into the persistent ChromaDB collection `tkpi_gi` using the binding
Iterasi 0 spike configuration (recursive chunking × text-embedding-3-large).

## Public API

- `app/data/parser.py`:
  - `FoodRecord` — frozen dataclass: `record_id, name, category, nutrients,
    gi_value, gl_value` (one per JSONL line)
  - `load_records(path: Path) -> list[FoodRecord]` — record IDs derive from the
    raw line number (`food-NNNN`), stable across blank lines
  - `NUTRIENT_FIELDS` — the 21 TKPI nutrient columns carried into each record
- `app/data/embedder.py`:
  - `IndexDocument` — frozen dataclass: `record_id, text, metadata` (one per
    food entry; a document never crosses an entry boundary)
  - `EmbeddingClient` / `IndexCollection` — Protocols over the LangChain
    embedding interface and the ChromaDB collection interface used here
  - `build_documents(records, strategy) -> list[IndexDocument]` — strategies
    `fixed | recursive | late`; unknown strategy raises `ValueError`
  - `index_documents(documents, collection, embedder, batch_size=100,
    max_retries=5) -> int` — embeds per batch with exponential-backoff retry
    (2 s base), deletes existing IDs before re-adding (idempotent re-runs)
- `app/scripts/index_data.py`:
  - `index_dataset(*, dataset_path, settings, client, embedder) -> int` —
    parse → build → upsert; returns final `collection.count()`
  - CLI: `uv run python -m scripts.index_data [--dataset PATH] [--log PATH]`,
    writes `reports/logs/indexing-raw.log`, exits 1 when count ≠ 1.146

## Dependencies

- Dataset `data/tkpi-gi-final.jsonl` (provided; read-only)
- `chromadb` (persistent client at `Settings.chroma_persist_dir`, collection
  created with `metadata={"hnsw:space": "cosine"}`), `langchain_openai.OpenAIEmbeddings`
- Real provider credentials via env — indexing never fabricates embeddings

## Data Flow

```
data/tkpi-gi-final.jsonl
    → load_records → FoodRecord (1 per line)
    → build_documents(strategy=CHUNKING_STRATEGY from .env) → IndexDocument
    → OpenAIEmbeddings(embed_documents, batches of 100, retry w/ backoff)
    → ChromaDB PersistentClient → collection `tkpi_gi` (cosine space)
    → reports/logs/indexing-raw.log (document_count evidence)
```

## Edge Cases

- Missing GI/GL on a record → rendered as explicit "tidak tersedia" text,
  never dropped or fabricated
- Provider transient failure during a batch → retried up to 5 times with
  exponential backoff, then raises with context
- Re-running the indexer → existing IDs deleted before re-adding; collection
  count stays exactly 1.146 (idempotent)
- Blank JSONL lines skipped without shifting later record IDs

## Acceptance Criteria

- [x] 1.146 entries indexed into the `tkpi_gi` collection
- [x] Raw unedited log at `reports/logs/indexing-raw.log`
- [x] One document per food entry (never split across documents)
- [x] Binding spike config (recursive × text-embedding-3-large) read from `.env`
- [x] Unit tests for parser, embedder, and index script pass
      (`tests/test_parser.py`, `tests/test_embedder.py`, `tests/test_index_data.py`)
- [x] Integration retrieval evidence (task-08):
      `tests/test_integration/test_retrieval_5queries.py` reaches mean P@5 ≥ 0.800
      on the indexed dataset; raw log at `reports/logs/retrieval-raw.log`

## Boundaries (module-specific)

**Always:** keep raw logs unedited; re-derive reports from logs only; verify
count = 1.146 after any re-index.
**Ask first:** changing the chunking strategy or embedding model after Iterasi 0
(SPEC.md §7.2); adding new data sources.
**Never:** index food data outside TKPI+GI (SPEC.md §7.3); modify files under
`data/`; commit the ChromaDB persistence directory.
