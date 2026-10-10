"""Document building and indexing into ChromaDB (task-07).

Builds one embedding document per food record using the chunking strategy
decided by the Iterasi 0 spike (recursive x text-embedding-3-large), embeds
them through the configured embedding client, and writes them to the
persistent ChromaDB collection.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.data.parser import FoodRecord

CHUNKING_STRATEGIES = ("fixed", "recursive", "late")
DEFAULT_BATCH_SIZE = 100
_RETRY_BASE_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class IndexDocument:
    """A single embedding document that never crosses a food-entry boundary."""

    record_id: str
    text: str
    metadata: dict[str, str | float]


class EmbeddingClient(Protocol):
    """Subset of the LangChain embedding interface used for indexing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class IndexCollection(Protocol):
    """Subset of the ChromaDB collection interface used for indexing."""

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def delete(self, ids: list[str]) -> None: ...


def _nutrition_text(record: FoodRecord) -> str:
    return ", ".join(f"{name}: {value:g}" for name, value in record.nutrients.items())


def _fixed_text(record: FoodRecord) -> str:
    values = [record.name, record.category, _nutrition_text(record)]
    if record.gi_value is not None:
        values.append(f"IG: {record.gi_value:g}")
    if record.gl_value is not None:
        values.append(f"BG: {record.gl_value:g}")
    return " | ".join(values)


def _recursive_text(record: FoodRecord) -> str:
    glycemic = "Data IG/BG tidak tersedia"
    if record.gi_value is not None:
        glycemic = f"Indeks glikemik: {record.gi_value:g}"
        if record.gl_value is not None:
            glycemic += f"\nBeban glikemik: {record.gl_value:g}"
    return (
        f"Nama bahan pangan: {record.name}\n"
        f"Kategori: {record.category}\n"
        f"Komposisi per 100 gram:\n{_nutrition_text(record)}\n"
        f"{glycemic}"
    )


def _late_text(record: FoodRecord) -> str:
    nutrition = _nutrition_text(record)
    glycemic = "tidak memiliki data indeks glikemik"
    if record.gi_value is not None:
        glycemic = f"memiliki indeks glikemik {record.gi_value:g}"
        if record.gl_value is not None:
            glycemic += f" dan beban glikemik {record.gl_value:g}"
    return (
        f"{record.name} adalah bahan pangan kategori {record.category}. "
        f"Komposisi gizinya per 100 gram adalah {nutrition}. "
        f"Pangan ini {glycemic}."
    )


_RENDERERS = {
    "fixed": _fixed_text,
    "recursive": _recursive_text,
    "late": _late_text,
}


def build_documents(records: Sequence[FoodRecord], strategy: str) -> list[IndexDocument]:
    """Create one independently embeddable document for every food record."""
    try:
        renderer = _RENDERERS[strategy]
    except KeyError as error:
        supported = ", ".join(CHUNKING_STRATEGIES)
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of: {supported}") from error

    return [
        IndexDocument(
            record_id=record.record_id,
            text=renderer(record),
            metadata={
                "name": record.name,
                "category": record.category,
                "strategy": strategy,
            },
        )
        for record in records
    ]


def _embed_with_retry(
    embedder: EmbeddingClient, texts: list[str], max_retries: int
) -> list[list[float]]:
    """Embed one batch, retrying transient endpoint failures with backoff."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return embedder.embed_documents(texts)
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))
    raise RuntimeError(
        f"Embedding failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def index_documents(
    documents: Sequence[IndexDocument],
    collection: IndexCollection,
    embedder: EmbeddingClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = 5,
) -> int:
    """Embed and write documents into the collection in batches.

    Returns the number of indexed documents.
    """
    indexed = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        embeddings = _embed_with_retry(embedder, [document.text for document in batch], max_retries)
        collection.delete(ids=[document.record_id for document in batch])
        collection.add(
            ids=[document.record_id for document in batch],
            documents=[document.text for document in batch],
            embeddings=embeddings,
            metadatas=[dict(document.metadata) for document in batch],
        )
        indexed += len(batch)
    return indexed
