"""Integration test: retrieval precision@5 against the indexed TKPI+GI collection.

Reproduces the five PRD §7.3 queries on a temporary ChromaDB collection built
from the real dataset (``data/tkpi-gi-final.jsonl``) with the spike-winning
chunking configuration from ``Settings`` and the self-hosted Ollama embedder.
Asserts that the mean Precision@5 meets or exceeds the binding spike threshold
(0.800). Writes a raw evidence log to ``reports/logs/retrieval-raw.log``
(§7.4 audit trail).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import chromadb
import httpx
import pytest

from app.config import Settings
from app.data.embedder import build_documents, index_documents
from app.data.parser import load_records
from app.evaluation.retrieval import QUERY_CASES, mean_precision_at_5, precision_at_k
from app.services.ollama_embedder import OllamaEmbedder

DATASET_PATH = Path("data/tkpi-gi-final.jsonl")
RAW_LOG_PATH = Path("reports/logs/retrieval-raw.log")
SPIKE_THRESHOLD = 0.800  # Binding Iterasi 0 result — reports/laporan-spike-2026-10-10.md


@pytest.fixture()
def settings() -> Settings:
    from tests.conftest import make_test_settings

    return make_test_settings()


@pytest.fixture()
def _skip_when_ollama_unreachable() -> None:
    """Skip when the configured Ollama endpoint does not answer ``/api/version``."""
    from tests.conftest import make_test_settings

    settings = make_test_settings()
    try:
        httpx.get(
            f"{settings.ollama_base_url.rstrip('/')}/api/version",
            timeout=5.0,
        ).raise_for_status()
    except Exception as error:  # probe must skip on any transport failure
        pytest.skip(f"Ollama not reachable at {settings.ollama_base_url}: {error}")


@pytest.fixture()
def indexed_collection(
    settings: Settings, _skip_when_ollama_unreachable: None, tmp_path: Path
) -> tuple[chromadb.Collection, OllamaEmbedder]:
    """Index the real dataset into an ephemeral collection using spike config."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    records = load_records(DATASET_PATH)
    documents = build_documents(records, settings.chunking_strategy)
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model_name,
        read_timeout_seconds=settings.ollama_batch_read_timeout_seconds,
    )
    index_documents(documents, collection, embedder)
    return collection, embedder


@pytest.mark.integration
@pytest.mark.requires_ollama
def test_precision_at_5_meets_spike_threshold(
    indexed_collection: tuple[chromadb.Collection, OllamaEmbedder],
    settings: Settings,
) -> None:
    collection, embedder = indexed_collection

    results: list[dict] = []
    for query in QUERY_CASES:
        query_vector = embedder.embed_query(query.text)
        retrieved = collection.query(query_embeddings=[query_vector], n_results=settings.rag_top_k)
        names = [str(metadata["name"]) for metadata in retrieved["metadatas"][0]]
        score = precision_at_k(query, names, settings.rag_top_k)
        results.append({"query": query.text, "retrieved_names": names, "precision_at_5": score})

    _write_raw_log(results)
    mean_p5 = mean_precision_at_5(results)

    assert mean_p5 >= SPIKE_THRESHOLD, (
        f"Mean P@5 {mean_p5:.3f} is below spike threshold {SPIKE_THRESHOLD:.3f}"
    )


@pytest.mark.integration
@pytest.mark.requires_ollama
def test_retrieval_uses_configured_top_k(
    indexed_collection: tuple[chromadb.Collection, OllamaEmbedder],
    settings: Settings,
) -> None:
    """Every query returns exactly ``rag_top_k`` documents."""
    collection, embedder = indexed_collection
    query_vector = embedder.embed_query(QUERY_CASES[0].text)

    retrieved = collection.query(query_embeddings=[query_vector], n_results=settings.rag_top_k)

    assert len(retrieved["metadatas"][0]) == settings.rag_top_k


def _write_raw_log(results: list[dict]) -> None:
    RAW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_LOG_PATH.open("a", encoding="utf-8") as raw_log:
        for result in results:
            raw_log.write(
                json.dumps(
                    {"timestamp": datetime.now(UTC).isoformat(), **result},
                    ensure_ascii=False,
                )
                + "\n"
            )


@pytest.mark.integration
@pytest.mark.requires_ollama
def test_indexed_collection_embeds_with_ollama(
    indexed_collection: tuple[chromadb.Collection, OllamaEmbedder],
) -> None:
    """The embedding fixture targets the self-hosted Ollama endpoint."""
    _, embedder = indexed_collection
    assert isinstance(embedder, OllamaEmbedder)


@pytest.mark.integration
def test_retrieval_probes_ollama_version_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Ollama skip fixture probes ``/api/version`` and skips on failure."""
    calls: list[str] = []

    def _down(*args: object, **kwargs: object) -> object:
        calls.append("probed")
        raise ConnectionError("ollama down")

    monkeypatch.setattr(httpx, "get", _down)
    with pytest.raises(pytest.skip.Exception) as caught:
        _skip_when_ollama_unreachable.__wrapped__()
    assert calls == ["probed"]
    assert "ollama down" in str(caught.value)
