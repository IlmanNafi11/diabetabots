"""Tests for the document builder and embedder (task-07 data indexing).

The embedder is the thin boundary to the OpenAI-compatible embedding endpoint;
these tests use a fake in-process client and a fake ChromaDB collection so no
network or persisted state is involved.
"""

from typing import Any

import pytest

from app.data.embedder import build_documents, index_documents
from app.data.parser import FoodRecord

RECORD_WITH_GI = FoodRecord(
    record_id="food-0000",
    name="Beras giling, mentah",
    category="serealia",
    nutrients={"Energi": 357.0, "Protein": 8.4},
    gi_value=64.0,
    gl_value=49.3,
)
RECORD_WITHOUT_GI = FoodRecord(
    record_id="food-0001",
    name="Bayam rebus",
    category="sayuran",
    nutrients={"Energi": 37.0},
)


class FakeEmbeddingClient:
    """Fake embedder: deterministic small vectors, records what it saw."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.embedded_texts: list[str] = []
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("embedding endpoint unavailable")
        self.embedded_texts.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]


class FakeCollection:
    """Fake ChromaDB collection capturing index calls."""

    def __init__(self) -> None:
        self.batches: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        return sum(len(batch["ids"]) for batch in self.batches)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self.batches.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

    def delete(self, ids: list[str]) -> None:
        drop = set(ids)
        self.batches = [batch for batch in self.batches if not drop & set(batch["ids"])]


class TestBuildDocuments:
    def test_one_document_per_food_record(self) -> None:
        documents = build_documents([RECORD_WITH_GI, RECORD_WITHOUT_GI], "recursive")

        assert [document.record_id for document in documents] == [
            "food-0000",
            "food-0001",
        ]

    def test_document_text_contains_name_and_category(self) -> None:
        documents = build_documents([RECORD_WITH_GI], "recursive")

        text = documents[0].text
        assert "Beras giling, mentah" in text
        assert "serealia" in text

    def test_document_text_includes_glycemic_values_when_present(self) -> None:
        documents = build_documents([RECORD_WITH_GI], "recursive")

        text = documents[0].text
        assert "Indeks glikemik: 64" in text
        assert "Beban glikemik: 49.3" in text

    def test_document_text_states_missing_glycemic_data(self) -> None:
        documents = build_documents([RECORD_WITHOUT_GI], "recursive")

        assert "Data IG/BG tidak tersedia" in documents[0].text

    def test_unknown_strategy_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="recursive"):
            build_documents([RECORD_WITH_GI], "invented")


class TestIndexDocuments:
    def test_indexes_every_document_into_collection(self) -> None:
        collection = FakeCollection()
        documents = build_documents([RECORD_WITH_GI, RECORD_WITHOUT_GI], "recursive")

        indexed = index_documents(documents, collection, FakeEmbeddingClient())

        assert indexed == 2
        assert collection.count == 2
        assert collection.batches[0]["ids"] == ["food-0000", "food-0001"]

    def test_embeds_the_document_text_not_the_id(self) -> None:
        embedder = FakeEmbeddingClient()
        collection = FakeCollection()
        documents = build_documents([RECORD_WITH_GI], "recursive")

        index_documents(documents, collection, embedder)

        assert len(embedder.embedded_texts) == 1
        assert "Beras giling, mentah" in embedder.embedded_texts[0]

    def test_passes_retrieval_metadata_with_each_document(self) -> None:
        collection = FakeCollection()
        documents = build_documents([RECORD_WITH_GI], "recursive")

        index_documents(documents, collection, FakeEmbeddingClient())

        metadata = collection.batches[0]["metadatas"][0]
        assert metadata["name"] == "Beras giling, mentah"
        assert metadata["category"] == "serealia"
        assert metadata["strategy"] == "recursive"

    def test_batches_large_document_sets(self) -> None:
        collection = FakeCollection()
        records = [
            FoodRecord(
                record_id=f"food-{index:04d}",
                name=f"Pangan {index}",
                category="serealia",
                nutrients={"Energi": 100.0},
            )
            for index in range(250)
        ]
        documents = build_documents(records, "recursive")

        indexed = index_documents(documents, collection, FakeEmbeddingClient(), batch_size=100)

        assert indexed == 250
        assert len(collection.batches) == 3
        assert [len(batch["ids"]) for batch in collection.batches] == [100, 100, 50]

    def test_retries_transient_embedding_failure(self) -> None:
        collection = FakeCollection()
        embedder = FakeEmbeddingClient(fail_on_call=1)
        documents = build_documents([RECORD_WITH_GI], "recursive")

        indexed = index_documents(documents, collection, embedder, max_retries=3)

        assert indexed == 1
        assert collection.count == 1
        assert embedder.calls == 2

    def test_raises_after_exhausting_retries(self) -> None:
        collection = FakeCollection()
        FakeEmbeddingClient(fail_on_call=1)

        # Fail every attempt: fail_on_call=1 would only trip once, so use a
        # persistent failure instead.
        class AlwaysFails(FakeEmbeddingClient):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                self.calls += 1
                raise RuntimeError("embedding endpoint unavailable")

        documents = build_documents([RECORD_WITH_GI], "recursive")

        with pytest.raises(RuntimeError, match="unavailable"):
            index_documents(documents, collection, AlwaysFails(), max_retries=2)

        assert collection.count == 0
