# Spec: rag-retrieval

**Status:** Implemented (evaluation harness; LLM-facing service lands in task-09) ·
**Plan ref:** task-08 / task-09 · **SPEC.md §8:** rag-retrieval

## Objective

Retrieve the most nutritionally relevant TKPI+GI food entries for a user query
using the binding spike configuration (recursive chunking ×
text-embedding-3-large, cosine space, top-k = 5), and prove retrieval quality
with Precision@5 on the five PRD §7.3 queries ≥ the Iterasi 0 spike threshold
(0.800).

## Public API

- `app/evaluation/retrieval.py` (evaluation harness, shared by tests and audit runs):
  - `QueryCase` — frozen dataclass: `text, relevant_terms`
  - `QUERY_CASES` — the five binding PRD §7.3 queries (verbatim; must stay in
    sync with `app/scripts/run_spike.py` and `reports/logs/spike-raw.log`)
  - `precision_at_k(query, names, k=5) -> float` — fraction of top-k food names
    whose name contains any relevant term (case-insensitive)
  - `mean_precision_at_5(results) -> float` — mean P@5 across result dicts
- `app/services/rag_service.py` (task-09, not yet implemented): embedding +
  top-k query service on the persistent collection, LangChain document loader.

## Dependencies

- `data-indexing` module (collection `tkpi_gi`, cosine space, spike config)
- `chromadb`, `langchain_openai.OpenAIEmbeddings` (same client configuration as
  `app/scripts/index_data.py`)
- Threshold source: binding Iterasi 0 spike result
  (`reports/laporan-spike-2026-10-10.md`, mean P@5 = 0.800); also stored as
  `SPIKE_PRECISION_AT_5` in `.env` for auditability

## Data Flow

```
user query (or §7.3 query case)
    → embed_query (text-embedding-3-large, 3072 d)
    → ChromaDB collection.query(n_results = RAG_TOP_K, cosine)
    → top-k IndexDocument metadata (name, category, strategy) + distances
    → precision_at_k per query → mean P@5
    → reports/logs/retrieval-raw.log (unedited evidence, §7.4)
```

## Edge Cases

- Query hits fewer than `rag_top_k` results (collection smaller than k) →
  precision is computed over the returned names only; test asserts full k for
  the 1.146-entry collection
- Food names with casing/spacing variance (`"Pisang ambon, segar"`) →
  relevance matching is case-insensitive substring on the name
- Spike threshold regression (mean P@5 < 0.800) → integration test fails and
  blocks; investigate before proceeding (per plan risk: consider query
  expansion or hybrid BM25 fallback — Ask First before implementing)
- No real credentials in env → integration test skips (unit tests still cover
  the harness math)

## Acceptance Criteria

- [x] Five PRD §7.3 queries evaluated verbatim against the indexed dataset
- [x] Mean Precision@5 ≥ 0.800 (binding spike threshold) on
      `tests/test_integration/test_retrieval_5queries.py`
- [x] Raw unedited log at `reports/logs/retrieval-raw.log`
- [x] Retrieval quality documented against the spike result (this spec +
      `SPEC-data-indexing.md`)
- [x] Harness math unit-testable without network (pure `precision_at_k` /
      `mean_precision_at_5` functions)
- [ ] task-09: `app/services/rag_service.py` wraps this retrieval path behind a
      service API for the LLM analysis chain (pending)

## Boundaries (module-specific)

**Always:** evaluate on the same five §7.3 queries as the spike; keep raw logs
unedited; record threshold alongside every retrieval evaluation.
**Ask first:** changing `QUERY_CASES`, the relevance-term definitions, or the
top-k default (affects comparability with the spike and PRD §7.3 evidence);
adding reranking, query expansion, or hybrid retrieval.
**Never:** fabricate P@5 numbers or hand-adjust raw logs; evaluate against a
collection indexed with a non-binding configuration.
