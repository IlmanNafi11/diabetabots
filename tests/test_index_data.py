"""Unit tests for the data indexing script (task-07)."""

import json
from pathlib import Path

import pytest

from app.config import Settings
from scripts.index_data import index_dataset


class FakeEmbeddingClient:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class FakeClient:
    """Fake chromadb.PersistentClient: records created collections."""

    def __init__(self) -> None:
        self.collections: dict[str, object] = {}

    def get_or_create_collection(self, name: str, metadata: dict | None = None):
        collection = self.collections.setdefault(
            name, FakeCollection(name=name, metadata=metadata or {})
        )
        return collection


class FakeCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name = name
        self.metadata = metadata
        self.added: list[dict] = []

    def add(self, ids, documents, embeddings, metadatas) -> None:
        self.added.extend(
            {"id": i, "document": d, "metadata": m}
            for i, d, m in zip(ids, documents, metadatas, strict=False)
        )

    def delete(self, ids) -> None:
        drop = set(ids)
        self.added = [item for item in self.added if item["id"] not in drop]

    def count(self) -> int:
        return len(self.added)


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        collection_name="tkpi_gi",
        chunking_strategy="recursive",
        chroma_persist_dir="./tmp-chroma-test",
        embedding_model_name="text-embedding-3-large",
        embedding_dim=3072,
    )


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    rows = [
        {
            "Nama Bahan Pangan": "Beras giling, mentah",
            "Kategori": "serealia",
            "Energi": 357.0,
            "gi_value": 64.0,
            "gl_value": 49.3,
        },
        {
            "Nama Bahan Pangan": "Bayam rebus",
            "Kategori": "sayuran",
            "Energi": 37.0,
            "gi_value": None,
            "gl_value": None,
        },
    ]
    path = tmp_path / "tkpi-gi-final.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_index_dataset_returns_document_count(dataset: Path, settings: Settings) -> None:
    count = index_dataset(
        dataset_path=dataset,
        settings=settings,
        client=FakeClient(),
        embedder=FakeEmbeddingClient(),
    )

    assert count == 2


def test_index_dataset_uses_configured_collection_and_cosine_space(
    dataset: Path, settings: Settings
) -> None:
    client = FakeClient()

    index_dataset(
        dataset_path=dataset,
        settings=settings,
        client=client,
        embedder=FakeEmbeddingClient(),
    )

    collection = client.collections["tkpi_gi"]
    assert collection.metadata == {"hnsw:space": "cosine"}


def test_index_dataset_indexes_documents_with_spike_metadata(
    dataset: Path, settings: Settings
) -> None:
    client = FakeClient()

    index_dataset(
        dataset_path=dataset,
        settings=settings,
        client=client,
        embedder=FakeEmbeddingClient(),
    )

    collection = client.collections["tkpi_gi"]
    assert collection.count() == 2
    first = collection.added[0]
    assert first["id"] == "food-0000"
    assert "Beras giling, mentah" in first["document"]
    assert first["metadata"]["name"] == "Beras giling, mentah"
    assert first["metadata"]["strategy"] == "recursive"


def test_index_dataset_is_idempotent_across_reruns(dataset: Path, settings: Settings) -> None:
    """A second full run must not double the collection count."""
    client = FakeClient()

    index_dataset(
        dataset_path=dataset,
        settings=settings,
        client=client,
        embedder=FakeEmbeddingClient(),
    )
    index_dataset(
        dataset_path=dataset,
        settings=settings,
        client=client,
        embedder=FakeEmbeddingClient(),
    )

    collection = client.collections["tkpi_gi"]
    assert collection.count() == 2


def test_index_dataset_rejects_unknown_strategy(dataset: Path, settings: Settings) -> None:
    settings = settings.model_copy(update={"chunking_strategy": "invented"})

    with pytest.raises(ValueError, match="invented"):
        index_dataset(
            dataset_path=dataset,
            settings=settings,
            client=FakeClient(),
            embedder=FakeEmbeddingClient(),
        )
