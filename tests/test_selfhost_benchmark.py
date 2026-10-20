"""Pure unit tests for the self-hosted embedding benchmark.

Never downloads models or starts Ollama. The live grid is exercised by the
benchmark command during the experimental run.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from app.scripts.run_selfhost_benchmark import (
    BASELINE_PRECISION_AT_5,
    CANDIDATES,
    _candidate_failed,
    recommend,
    render_report_from_log,
)


def test_render_report_from_log_aggregates_candidates_across_runs() -> None:
    """The aggregate report must include candidates finished in earlier runs."""
    raw_log = StringIO()
    for candidate in (CANDIDATES[0], CANDIDATES[2]):
        for _ in range(5):
            raw_log.write(
                json.dumps(
                    {
                        "timestamp": "2026-10-19T00:00:00+00:00",
                        "candidate": candidate.key,
                        "model": candidate.model,
                        "runtime": candidate.runtime,
                        "dimension": candidate.dim,
                        "query": "q",
                        "relevant_terms": ["x"],
                        "retrieved_names": ["a", "b", "c", "d", "e"],
                        "distances": [0.1] * 5,
                        "precision_at_5": 0.8,
                        "latency_ms": 100.0,
                    }
                )
                + "\n"
            )
        raw_log.write(
            json.dumps(
                {
                    "timestamp": "2026-10-19T00:00:00+00:00",
                    "candidate": candidate.key,
                    "event": "embedding_complete",
                    "embedding_latency_ms": 12345.0,
                }
            )
            + "\n"
        )
    raw_log_path = Path("/tmp/test-selfhost-aggregate.log")
    raw_log_path.write_text(raw_log.getvalue(), encoding="utf-8")
    report_path = render_report_from_log(
        raw_log_path=raw_log_path,
        reports_dir=Path("/tmp"),
        dataset_path=Path("data/tkpi-gi-final.jsonl"),
    )
    body = report_path.read_text(encoding="utf-8")
    assert "Qwen3-Embedding-0.6B" in body
    assert "BGE-M3" in body
    assert "0.800" in body
    raw_log_path.unlink()
    report_path.unlink()


def test_render_report_from_log_marks_incomplete_candidate_failed() -> None:
    raw_log = StringIO()
    raw_log.write(
        json.dumps(
            {
                "timestamp": "2026-10-19T00:00:00+00:00",
                "candidate": "qwen3-0.6b",
                "query": "q",
                "precision_at_5": 0.8,
                "latency_ms": 100.0,
            }
        )
        + "\n"
    )
    raw_log_path = Path("/tmp/test-selfhost-incomplete.log")
    raw_log_path.write_text(raw_log.getvalue(), encoding="utf-8")
    report_path = render_report_from_log(
        raw_log_path=raw_log_path,
        reports_dir=Path("/tmp"),
        dataset_path=Path("data/tkpi-gi-final.jsonl"),
    )
    body = report_path.read_text(encoding="utf-8")
    assert "GAGAL" in body
    raw_log_path.unlink()
    report_path.unlink()

def test_candidates_grid_matches_intent() -> None:
    keys = [candidate.key for candidate in CANDIDATES]
    assert keys == [
        "qwen3-0.6b",
        "qwen3-4b",
        "bge-m3",
        "e5-large",
        "nomic-v2-moe",
        "arctic-embed2",
        "e5-large-instruct",
    ]
    by_key = {candidate.key: candidate for candidate in CANDIDATES}
    assert by_key["qwen3-0.6b"].runtime == "fastembed"
    assert by_key["qwen3-0.6b"].dim == 1024
    assert by_key["qwen3-4b"].runtime == "ollama"
    assert by_key["qwen3-4b"].dim == 2560
    assert by_key["bge-m3"].runtime == "ollama"
    assert by_key["nomic-v2-moe"].runtime == "ollama"
    assert by_key["nomic-v2-moe"].dim == 768
    assert by_key["arctic-embed2"].runtime == "ollama"
    assert by_key["arctic-embed2"].dim == 1024
    assert by_key["e5-large-instruct"].runtime == "fastembed"
    assert by_key["e5-large-instruct"].dim == 1024


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

def test_new_candidates_use_official_prefixes() -> None:
    nomic = next(candidate for candidate in CANDIDATES if candidate.key == "nomic-v2-moe")
    assert nomic.query_prefix == "search_query: "
    assert nomic.passage_prefix == "search_document: "

    arctic = next(candidate for candidate in CANDIDATES if candidate.key == "arctic-embed2")
    assert arctic.query_prefix == "query: "
    assert arctic.passage_prefix is None

    e5i = next(candidate for candidate in CANDIDATES if candidate.key == "e5-large-instruct")
    assert e5i.query_prefix.startswith("Instruct: ")
    assert "Query: " in e5i.query_prefix
    assert e5i.passage_prefix is None


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
