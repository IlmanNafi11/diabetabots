# Spec: spike-embedding

**Status:** Completed (Iterasi 0 run 2026-10-10) · **Plan ref:** task-03 · **SPEC.md §8:** spike-embedding

## Objective

Select the binding document-representation × embedding-model configuration for all
subsequent RAG development by measuring Precision@5 on the five PRD §7.3 queries across
a 3×2 grid. One food entry = one document, never split across documents.

## Public API

- `app/scripts/run_spike.py` (standalone, `python -m app.scripts.run_spike`):
  - `FoodRecord` — frozen dataclass: `record_id, name, category, nutrients, gi_value,
    gl_value, serving_size_g, available_carbohydrate_raw`
  - `SpikeDocument` — frozen dataclass: `record_id, text, metadata` (one per food entry)
  - `QueryCase` — frozen dataclass: `text, relevant_terms`
  - `EmbeddingClient` — Protocol exposing `embed_documents`, `embed_query`
  - CLI entry runs the grid, writes `reports/logs/spike-raw.log`,
    `reports/laporan-spike-YYYY-MM-DD.md`, and the winning config to `.env`

## Dependencies

- Dataset `data/tkpi-gi-final.jsonl` (1146 entries — provided, never modified)
- `chromadb` (ephemeral client per experiment), `langchain_openai.OpenAIEmbeddings`
- Real provider credentials via env — the script never fabricates spike results

## Data Flow

```
tkpi-gi-final.jsonl → FoodRecord → SpikeDocument (strategy: fixed | recursive | late)
    → embed (3-large 3072d | 3-small 1536d) → ChromaDB collection → 5 queries → P@5
    → raw log + report + winning config → .env
```

## Edge Cases

- Winner margin < 0.02 between best large and best small → choose `text-embedding-3-small`
  (cost/latency per PRD §7.2)
- Entries with `gi_value`/`gl_value`/`serving_size_g` null → represented as explicit
  "tidak tersedia" text, never dropped
- Provider failure mid-grid → script aborts; no partial result is recorded as binding

## Acceptance Criteria

- [x] 3×2 grid (fixed/recursive/late × 3-large/3-small) evaluated on the 5 §7.3 queries
- [x] Raw unedited log at `reports/logs/spike-raw.log`; markdown report at
      `reports/laporan-spike-2026-10-10.md`
- [x] Winner: **recursive × text-embedding-3-large, mean P@5 = 0.800** (latency ~582 ms)
- [x] Winning config written to `.env` and treated as binding for data-indexing and
      rag-retrieval modules

## Boundaries (module-specific)

**Always:** keep raw logs unedited; re-derive reports from logs only.
**Ask first:** changing the grid configuration after Iterasi 0 (SPEC.md §7.2).
**Never:** fabricate or hand-adjust P@5 numbers; index food data outside TKPI+GI.
