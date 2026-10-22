"""Unit tests for app/services/rag_service.py (task-09).

The RAG retrieval service wraps ChromaDB retrieval behind a clean API.
Tests here cover:
- Construction: embedder + ChromaDB client are created eagerly; the collection
  is loaded lazily on first retrieve (the service may be constructed before
  indexing has run).
- ``_make_embedder`` builds an ``OllamaEmbedder`` from settings; construction
  succeeds even with a sentinel LLM API key (credential fail-fast is
  ``LLMService``'s job, not ``RAGService``'s).
- ``retrieve`` embeds the query, queries ChromaDB with ``n_results=top_k``,
  and maps the raw result to ``RetrievedFood`` objects.
- ``build_retrieved_foods`` maps the raw ChromaDB query-result dict
  (``ids``/``distances``/``metadatas``/``documents``, each one result set per
  query embedding) and rejects mismatched inner lengths.
- ``build_query_context`` formats results for LLM prompt injection.

Real ChromaDB / embedding endpoint interaction is covered by the integration
suite (``tests/test_integration/test_retrieval_5queries.py``); these unit tests
never touch the network or the production ``./chroma_db`` directory.
"""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.services.ollama_embedder import OllamaEmbedder
from app.services.rag_service import (
    RAGService,
    RetrievedFood,
    build_query_context,
    build_retrieved_foods,
)

# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings() -> MagicMock:
    """Minimal settings matching the real Settings fields used by RAGService."""
    s = MagicMock()
    s.ollama_base_url = "http://ollama:11434"
    s.llm_base_url = "https://api.example.com/v1"
    s.llm_api_key = "test-key"
    s.embedding_model_name = "snowflake-arctic-embed2"
    s.rag_top_k = 5
    s.rag_candidate_k = 20
    s.chunking_strategy = "recursive"
    s.collection_name = "tkpi_gi"
    s.chroma_persist_dir = "./chroma_db"
    return s


@pytest.fixture()
def mock_embedder() -> MagicMock:
    """Embedding client that returns a fixed query vector."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 3072
    return embedder


@pytest.fixture()
def mock_collection() -> MagicMock:
    """ChromaDB collection returning canned results in the real response shape.

    ``collection.query`` returns one result set per query embedding; each
    result set (``ids``, ``distances``, ``metadatas``, ``documents``) is a
    flat inner list of ``n_results`` entries.
    """
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["food-0001", "food-0042", "food-0117"]],
        "distances": [[0.12, 0.34, 0.56]],
        "metadatas": [
            [
                {
                    "name": "Nasi Putih, Segar",
                    "category": "Nasi dan produk beras",
                    "gi_value": 73.0,
                },
                {
                    "name": "Nasi Merah, Segar",
                    "category": "Nasi dan produk beras",
                    "gi_value": 55.0,
                },
                {"name": "Ayam Goreng, Segar", "category": "Unggas", "gi_value": None},
            ]
        ],
        "documents": [["Nasi Putih text", "Nasi Merah text", "Ayam text"]],
    }
    return collection


def make_service(
    settings: MagicMock,
    embedder: MagicMock | None = None,
    collection: MagicMock | None = None,
) -> RAGService:
    """Build a RAGService with embedder and client creation patched out."""
    with (
        patch.object(RAGService, "_make_embedder", return_value=embedder or MagicMock()),
        patch.object(RAGService, "_make_client", return_value=MagicMock()),
    ):
        service = RAGService(settings=settings)  # type: ignore[arg-type]
    if collection is not None:
        service._collection = collection
    return service


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRAGServiceConstruction:
    """RAGService builds its embedder + client eagerly; collection lazily."""

    def test_exposes_settings(self, mock_settings: MagicMock) -> None:
        service = make_service(mock_settings)
        assert service.settings is mock_settings

    def test_creates_embedder_at_init(self, mock_settings: MagicMock) -> None:
        embedder = MagicMock()
        with patch.object(RAGService, "_make_embedder", return_value=embedder) as make:
            service = RAGService(settings=mock_settings)  # type: ignore[arg-type]
        make.assert_called_once_with()
        assert service._embedder is embedder

    def test_creates_chroma_client_at_init(self, mock_settings: MagicMock) -> None:
        client = MagicMock()
        with patch.object(RAGService, "_make_client", return_value=client) as make:
            service = RAGService(settings=mock_settings)  # type: ignore[arg-type]
        make.assert_called_once_with()
        assert service._client is client

    def test_collection_is_lazy_at_init(self, mock_settings: MagicMock) -> None:
        service = make_service(mock_settings)
        assert service._collection is None

    def test_retrieve_loads_collection_once(
        self,
        mock_settings: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        client = MagicMock()
        client.get_collection.return_value = mock_collection
        with (
            patch.object(RAGService, "_make_embedder", return_value=MagicMock()),
            patch.object(RAGService, "_make_client", return_value=client),
        ):
            service = RAGService(settings=mock_settings)  # type: ignore[arg-type]
            service.retrieve("nasi putih")
            service.retrieve("ayam goreng")
        client.get_collection.assert_called_once_with(name=mock_settings.collection_name)
        assert service._collection is mock_collection

    def test_constructs_with_sentinel_llm_api_key(self) -> None:
        """RAGService no longer gates on the LLM sentinel key (LLMService does)."""
        with (
            patch.object(RAGService, "_make_embedder", return_value=MagicMock()),
            patch.object(RAGService, "_make_client", return_value=MagicMock()),
        ):
            sentinel_settings = MagicMock()
            sentinel_settings.llm_api_key = "your-api-key"
            service = RAGService(settings=sentinel_settings)  # type: ignore[arg-type]
        assert service.settings is sentinel_settings

    def test_make_embedder_returns_ollama_embedder_from_settings(
        self, mock_settings: MagicMock
    ) -> None:
        """``_make_embedder`` builds an ``OllamaEmbedder`` from settings."""
        settings = Settings(
            ollama_base_url="http://ollama-test:11434",
            llm_api_key="test-key",
            embedding_model_name="custom-model:7b",
            ollama_query_read_timeout_seconds=12.5,
        )
        with patch.object(RAGService, "_make_client", return_value=MagicMock()):
            service = RAGService(settings=settings)
        embedder = service._embedder
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder._base_url == "http://ollama-test:11434"
        assert embedder._model == "custom-model:7b"
        assert embedder._client.timeout.read == 12.5

    @pytest.mark.asyncio()
    async def test_retrieve_async_offloads_blocking_retrieve(
        self, mock_settings: MagicMock
    ) -> None:
        expected = [make_food(1)]
        with (
            patch.object(RAGService, "_make_embedder", return_value=MagicMock()),
            patch.object(RAGService, "_make_client", return_value=MagicMock()),
        ):
            service = RAGService(settings=mock_settings)  # type: ignore[arg-type]
        with patch.object(service, "retrieve", return_value=expected) as retrieve:
            result = await service.retrieve_async("nasi", 5)

        assert result == expected
        retrieve.assert_called_once_with("nasi", 5, None)


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    """retrieve() embeds the query, queries ChromaDB, and returns RetrievedFood."""

    def test_embeds_query_text(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)
        service.retrieve("nasi putih")
        mock_embedder.embed_query.assert_called_once_with("nasi putih")

    def test_queries_collection_with_candidate_pool(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)
        service.retrieve("nasi putih")
        kwargs = mock_collection.query.call_args.kwargs
        assert kwargs["n_results"] == mock_settings.rag_candidate_k
        assert kwargs["where"] is None

    def test_rejects_semantically_similar_but_unnamed_food(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)

        assert service.retrieve("sayur aster") == []

    def test_reranks_name_grounded_candidate_ahead_of_nearest_unrelated_result(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [["food-1", "food-2"]],
            "distances": [[0.1, 0.5]],
            "metadatas": [
                [
                    {"name": "Kool kembang", "category": "sayuran"},
                    {"name": "Ikan oci, kembung, segar", "category": "ikan"},
                ]
            ],
            "documents": [["wrong", "right"]],
        }
        service = make_service(mock_settings, embedder=mock_embedder, collection=collection)

        results = service.retrieve("ikan kembung")

        assert results[0].name == "Ikan oci, kembung, segar"

    def test_candidate_pool_is_at_least_requested_top_k(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)
        service.retrieve("nasi", top_k=25)

        assert mock_collection.query.call_args.kwargs["n_results"] == 25

    def test_returns_list_of_retrieved_food(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)
        results = service.retrieve("nasi putih")
        assert len(results) == 2
        assert all(isinstance(r, RetrievedFood) for r in results)

    def test_maps_metadata_and_distance(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)
        first = service.retrieve("nasi putih")[0]
        assert first.record_id == "food-0001"
        assert first.name == "Nasi Putih, Segar"
        assert first.category == "Nasi dan produk beras"
        assert first.gi_value == 73.0
        assert first.distance == pytest.approx(0.12)
        assert first.text_snippet == "Nasi Putih text"

    def test_gi_none_survives_mapping(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)
        results = service.retrieve("ayam goreng")
        assert results[0].gi_value is None

    def test_returns_empty_when_collection_has_no_hits(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        empty_collection = MagicMock()
        empty_collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }
        service = make_service(mock_settings, embedder=mock_embedder, collection=empty_collection)
        assert service.retrieve("xyzzy nonsense query") == []

    def test_custom_top_k_remains_the_return_limit(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)
        results = service.retrieve("nasi", top_k=2)

        assert len(results) <= 2
        assert mock_collection.query.call_args.kwargs["n_results"] >= 2

    def test_passes_optional_where_filter(
        self,
        mock_settings: MagicMock,
        mock_embedder: MagicMock,
        mock_collection: MagicMock,
    ) -> None:
        service = make_service(mock_settings, embedder=mock_embedder, collection=mock_collection)
        where_filter = {"category": {"$eq": "Nasi dan produk beras"}}
        service.retrieve("nasi", where=where_filter)
        assert mock_collection.query.call_args.kwargs["where"] == where_filter


# ---------------------------------------------------------------------------
# build_retrieved_foods
# ---------------------------------------------------------------------------


class TestBuildRetrievedFoods:
    """build_retrieved_foods maps the raw ChromaDB result dict to RetrievedFood."""

    def test_maps_all_fields(self) -> None:
        raw = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [
                [
                    {"name": "Foo", "category": "Cat", "gi_value": 65.0},
                    {"name": "Bar", "category": "Dog", "gi_value": None},
                ]
            ],
            "documents": [["text1", "text2"]],
        }
        foods = build_retrieved_foods(raw)
        assert len(foods) == 2
        assert foods[0].record_id == "id1"
        assert foods[0].name == "Foo"
        assert foods[0].category == "Cat"
        assert foods[0].gi_value == 65.0
        assert foods[0].distance == 0.1
        assert foods[0].text_snippet == "text1"
        assert foods[1].gi_value is None

    def test_handles_non_numeric_gi_value(self) -> None:
        raw = {
            "ids": [["id1"]],
            "distances": [[0.1]],
            "metadatas": [[{"name": "Foo", "category": "Cat", "gi_value": "n/a"}]],
            "documents": [["text1"]],
        }
        assert build_retrieved_foods(raw)[0].gi_value is None

    def test_raises_on_mismatched_inner_lengths(self) -> None:
        raw = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1]],  # one fewer than ids
            "metadatas": [
                [
                    {"name": "Foo", "category": "Cat", "gi_value": None},
                    {"name": "Bar", "category": "Dog", "gi_value": None},
                ]
            ],
            "documents": [["text1", "text2"]],
        }
        with pytest.raises(ValueError, match="mismatch"):
            build_retrieved_foods(raw)


# ---------------------------------------------------------------------------
# build_query_context
# ---------------------------------------------------------------------------


def make_food(i: int, gi_value: float | None = 50.0) -> RetrievedFood:
    return RetrievedFood(
        record_id=f"f{i}",
        name=f"Food {i}",
        category="C",
        gi_value=gi_value,
        distance=0.01 * i,
        text_snippet=f"text {i}",
    )


class TestBuildQueryContext:
    """build_query_context formats top-k results for LLM prompt injection."""

    def test_empty_list_returns_empty_string(self) -> None:
        assert build_query_context([]) == ""

    def test_includes_name_category_and_gi(self) -> None:
        context = build_query_context([make_food(1, gi_value=73.0)])
        assert "Food 1" in context
        assert "C" in context  # category
        assert "73" in context

    def test_missing_gi_renders_explicitly(self) -> None:
        context = build_query_context([make_food(1, gi_value=None)])
        assert "tidak tersedia" in context

    def test_includes_text_snippet(self) -> None:
        context = build_query_context([make_food(1)])
        assert "text 1" in context

    def test_max_foods_limits_entries(self) -> None:
        foods = [make_food(i) for i in range(20)]
        context = build_query_context(foods, max_foods=5)
        headers = [line for line in context.splitlines() if line.startswith("[")]
        assert len(headers) == 5
        assert "Food 19" not in context


# ---------------------------------------------------------------------------
# LangChain document compatibility
# ---------------------------------------------------------------------------


class TestRetrievedFood:
    """RetrievedFood is a frozen dataclass exposing a page_content alias."""

    def test_is_frozen(self) -> None:
        food = make_food(1)
        with pytest.raises(FrozenInstanceError):
            food.name = "changed"  # type: ignore[misc]

    def test_page_content_aliases_text_snippet(self) -> None:
        food = make_food(1)
        assert food.page_content == food.text_snippet
