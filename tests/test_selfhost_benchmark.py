"""Pure unit tests for the self-hosted embedding benchmark.

Never downloads models or starts Ollama. The live grid is exercised by the
benchmark command during the experimental run.
"""

from __future__ import annotations

from io import StringIO

from app.scripts.run_selfhost_benchmark import (
    BASELINE_PRECISION_AT_5,
    CANDIDATES,
    _candidate_failed,
    recommend,
)


def test_candidates_grid_matches_intent() -> None:
    keys = [candidate.key for candidate in CANDIDATES]
    assert keys == ["qwen3-0.6b", "qwen3-4b", "bge-m3", "e5-large"]
    by_key = {candidate.key: candidate for candidate in CANDIDATES}
    assert by_key["qwen3-0.6b"].runtime == "fastembed"
    assert by_key["qwen3-0.6b"].dim == 1024
    assert by_key["qwen3-4b"].runtime == "ollama"
    assert by_key["qwen3-4b"].dim == 2560
    assert by_key["bge-m3"].runtime == "ollama"
    assert by_key["e5-large"].runtime == "fastembed"


def test_qwen_prefixes_use_instruct_template() -> None:
    qwen = next(candidate for candidate in CANDIDATES if candidate.key == "qwen3-0.6b")
    assert qwen.query_prefix is not None
    assert qwen.query_prefix.startswith("Instruct:")
    assert "Query: " in qwen.query_prefix
    assert qwen.passage_prefix is None


def test_e5_uses_query_passage_prefixes_but_bge_does_not() -> None:
    e5 = next(candidate for candidate in CANDIDATES if candidate.key == "e5-large")
    assert e5.query_prefix == "query: "
    assert e5.passage_prefix == "passage: "

    bge = next(candidate for candidate in CANDIDATES if candidate.key == "bge-m3")
    assert bge.query_prefix is None
    assert bge.passage_prefix is None


def test_recommend_picks_best_passing_candidate() -> None:
    results = [
        {
            "label": "Qwen3-Embedding-0.6B",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "mean_precision_at_5": 0.720,
        },
        {
            "label": "Qwen3-Embedding-4B (GGUF Q4)",
            "model": "qwen3-embedding:4b",
            "mean_precision_at_5": 0.840,
        },
        {
            "label": "BGE-M3",
            "model": "bge-m3:latest",
            "mean_precision_at_5": None,
        },
    ]
    text = recommend(results, BASELINE_PRECISION_AT_5)
    assert "Qwen3-Embedding-4B" in text
    assert "0.840" in text


def test_recommend_keeps_openai_when_none_pass() -> None:
    results = [
        {
            "label": "Qwen3-Embedding-0.6B",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "mean_precision_at_5": 0.600,
        }
    ]
    text = recommend(results, BASELINE_PRECISION_AT_5)
    assert "tetap text-embedding-3-large" in text


def test_recommend_ignores_failed_candidates() -> None:
    results = [
        {
            "label": "Qwen3-Embedding-4B (GGUF Q4)",
            "model": "qwen3-embedding:4b",
            "mean_precision_at_5": None,
            "failed": "OOM",
        },
        {
            "label": "BGE-M3",
            "model": "bge-m3:latest",
            "mean_precision_at_5": 0.900,
        },
    ]
    text = recommend(results, BASELINE_PRECISION_AT_5)
    assert "BGE-M3" in text


def test_failed_candidate_record_marks_failure_and_logs_it() -> None:
    candidate = next(candidate for candidate in CANDIDATES if candidate.key == "bge-m3")
    raw_log = StringIO()
    record = _candidate_failed(raw_log, candidate, "out of memory")
    assert record["mean_precision_at_5"] is None
    assert record["failed"] == "out of memory"
    assert record["label"] == "BGE-M3"
    assert '"event": "failed"' in raw_log.getvalue()
