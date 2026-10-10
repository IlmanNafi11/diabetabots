"""Index the TKPI+GI dataset into persistent ChromaDB (task-07).

Parses ``data/tkpi-gi-final.jsonl``, builds one embedding document per food
entry using the chunking strategy decided by the Iterasi 0 spike, embeds them
with the configured OpenAI-compatible embedding endpoint, and writes the
result into the ``tkpi_gi`` collection. Expects 1.146 indexed entries.

Usage:
    uv run python -m scripts.index_data [--dataset PATH] [--log PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.data.embedder import EmbeddingClient, build_documents, index_documents
from app.data.parser import load_records


def index_dataset(
    *,
    dataset_path: Path,
    settings: Settings,
    client: Any,
    embedder: EmbeddingClient,
) -> int:
    """Parse, embed, and index the dataset into the configured collection.

    Returns the number of documents present in the collection afterwards.
    Re-runs replace existing entries with the same IDs, so the operation is
    idempotent.
    """
    records = load_records(dataset_path)
    documents = build_documents(records, settings.chunking_strategy)

    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    index_documents(
        documents,
        collection,
        embedder,
    )
    return collection.count()


def _create_embedder(settings: Settings) -> EmbeddingClient:
    return OpenAIEmbeddings(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.embedding_model_name,
        dimensions=settings.embedding_dim,
        check_embedding_ctx_length=False,
    )


def _write_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as raw_log:
        raw_log.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/tkpi-gi-final.jsonl"),
        help="Path to the TKPI+GI JSONL dataset",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("reports/logs/indexing-raw.log"),
        help="Where to append the raw indexing log",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embedder = _create_embedder(settings)

    started = datetime.now(UTC).isoformat()
    count = index_dataset(
        dataset_path=args.dataset,
        settings=settings,
        client=client,
        embedder=embedder,
    )
    _write_log(
        args.log,
        {
            "timestamp": started,
            "dataset": str(args.dataset),
            "collection": settings.collection_name,
            "chunking_strategy": settings.chunking_strategy,
            "embedding_model": settings.embedding_model_name,
            "embedding_dim": settings.embedding_dim,
            "document_count": count,
        },
    )
    print(f"Indexed {count} documents into {settings.collection_name!r}")

    expected = 1146
    if count != expected:
        print(
            f"WARNING: expected {expected} entries, collection holds {count}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
