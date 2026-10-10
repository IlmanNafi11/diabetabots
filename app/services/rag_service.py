"""RAG retrieval service — ChromaDB embedding + top-k query (task-09).

Wraps the persistent ChromaDB collection (`app/data/embedder.py`) behind a
clean service API for the LLM analysis chain.

Public API
----------
``RAGService`` — holds embedder + persistent ChromaDB client; exposes:
  ``retrieve(query, top_k?, where?) -> list[RetrievedFood]``

``RetrievedFood`` — frozen dataclass matching ChromaDB document metadata + cosine
distance; also exposes ``page_content`` as an alias of ``text_snippet`` for
LangChain Document compatibility.

``build_retrieved_foods(query_result: dict) -> list[RetrievedFood]`` — maps raw
ChromaDB result dict to typed ``RetrievedFood`` list.

``build_query_context(foods: Sequence[RetrievedFood], max_foods?) -> str`` —
formats top-k results as a readable string for LLM prompt injection.

Dependencies
------------
- ``chromadb.PersistentClient`` at ``Settings.chroma_persist_dir``.
- ``langchain_openai.OpenAIEmbeddings`` (same configuration as
  ``app/scripts/index_data.py``).
- ``Settings`` from ``app.config``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings

__all__ = [
    "RAGService",
    "RetrievedFood",
    "build_query_context",
    "build_retrieved_foods",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedFood:
    """A food entry returned from ChromaDB retrieval.

    Attributes
    ----------
    record_id:
        Stable ID from the original dataset line (``food-NNNN``).
    name:
        Human-readable food name from TKPI.
    category:
        TKPI food category.
    gi_value:
        Glycemic Index (0-100), or ``None`` when not available.
    distance:
        Cosine distance from the query embedding (lower = more relevant).
    text_snippet:
        The full text that was embedded and stored in ChromaDB.
        Alias: ``page_content`` — LangChain Document compatibility.
    """

    record_id: str
    name: str
    category: str
    gi_value: float | None
    distance: float
    text_snippet: str

    @property
    def page_content(self) -> str:
        """LangChain ``Document.page_content`` alias for ``text_snippet``."""
        return self.text_snippet


class RAGService:
    """RAG retrieval backed by the persistent ChromaDB collection.

    Initialises once on construction: creates an ``OpenAIEmbeddings`` client
    and a ``chromadb.PersistentClient`` pointing at the configured directory.
    The ChromaDB collection is loaded lazily on first ``retrieve`` call.

    Raises
    ------
    RuntimeError
        When the API key is still a sentinel ``"your-*"`` value, indicating
        the environment has not been configured.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._check_credentials()
        # Embedder + ChromaDB client are created eagerly so that construction
        # fails fast on bad configuration; the collection is loaded lazily on
        # first retrieve (it may not exist yet right after a fresh checkout).
        self._embedder: OpenAIEmbeddings = self._make_embedder()
        self._client: chromadb.PersistentClient = self._make_client()
        self._collection: chromadb.Collection | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedFood]:
        """Retrieve the ``top_k`` most relevant food entries for ``query``.

        Parameters
        ----------
        query:
            Free-text food or nutrition question (embedded as-is).
        top_k:
            Override the default ``rag_top_k`` from settings.
        where:
            Optional ChromaDB metadata filter predicate
            (e.g. ``{"category": {"$eq": "Nasi dan produk beras"}}``).

        Returns
        -------
        list[RetrievedFood]
            Top-k results ordered by ascending cosine distance.
        """
        n_results = top_k if top_k is not None else self._settings.rag_top_k
        embedder = self._get_embedder()
        collection = self._get_collection()
        query_vector = embedder.embed_query(query)
        raw = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where,
        )
        return build_retrieved_foods(raw)

    # ------------------------------------------------------------------
    # Internal helpers (overridable for testing)
    # ------------------------------------------------------------------

    def _get_embedder(self) -> OpenAIEmbeddings:
        if self._embedder is None:
            self._embedder = self._make_embedder()
        return self._embedder

    def _make_embedder(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key,
            model=self._settings.embedding_model_name,
            dimensions=self._settings.embedding_dim,
            check_embedding_ctx_length=False,
        )

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_collection(name=self._settings.collection_name)
        return self._collection

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _make_client(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(
            path=self._settings.chroma_persist_dir,
            settings=ChromaSettings(allow_reset=False),
        )

    def _check_credentials(self) -> None:
        sentinel = self._settings.llm_api_key.startswith("your-")
        if sentinel:
            raise RuntimeError(
                "RAGService: LLM API key is a sentinel value. "
                "Set OPENAI_API_KEY (or the equivalent in your .env) before use."
            )


# ---------------------------------------------------------------------------
# Pure helpers (no I/O — unit-testable without mocks)
# ---------------------------------------------------------------------------


def build_retrieved_foods(query_result: dict[str, Any]) -> list[RetrievedFood]:
    """Map a raw ChromaDB ``query`` result dict to a list of ``RetrievedFood``.

    The input dict must contain the four keys returned by
    ``collection.query(include=["metadatas", "distances", "documents"])``:
    ``ids``, ``distances``, ``metadatas``, and ``documents``.
    All inner lists must have the same length for each result set (row index 0).

    Raises
    ------
    ValueError
        When the inner lists have mismatched lengths.
    """
    # ChromaDB returns one result set per query embedding (we always send one).
    ids: list[str] = query_result.get("ids", [[]])[0]
    distances: list[float] = query_result.get("distances", [[]])[0]
    metadatas: list[dict[str, Any]] = query_result.get("metadatas", [[]])[0]
    documents: list[str] = query_result.get("documents", [[]])[0]

    n = len(ids)
    if not (len(distances) == len(metadatas) == len(documents) == n):
        raise ValueError(
            f"ChromaDB result lists have mismatched lengths: "
            f"ids={n}, distances={len(distances)}, "
            f"metadatas={len(metadatas)}, documents={len(documents)}"
        )

    foods: list[RetrievedFood] = []
    for i in range(n):
        meta = metadatas[i]
        foods.append(
            RetrievedFood(
                record_id=ids[i],
                name=meta.get("name", ""),
                category=meta.get("category", ""),
                gi_value=_maybe_float(meta.get("gi_value")),
                distance=distances[i],
                text_snippet=documents[i],
            )
        )
    return foods


def build_query_context(
    foods: Sequence[RetrievedFood],
    max_foods: int | None = None,
) -> str:
    """Format a list of retrieved foods as a human-readable context string.

    Each food is rendered as::

        [name] (kategori: category, IG: gi_or_na)
        text_snippet

    Parameters
    ----------
    foods:
        Retrieved foods, typically the result of ``RAGService.retrieve``.
    max_foods:
        If set, truncate to the first ``max_foods`` entries to limit prompt size.
    """
    if not foods:
        return ""

    limit = max_foods if max_foods is not None else len(foods)
    lines: list[str] = []
    for food in foods[:limit]:
        gi_str = f"{food.gi_value:.0f}" if food.gi_value is not None else "tidak tersedia"
        lines.append(f"[{food.name}] (kategori: {food.category}, IG: {gi_str})")
        lines.append(f"  {food.text_snippet.strip()}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _maybe_float(value: Any) -> float | None:
    """Cast a value to float, returning None for None or non-numeric values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
