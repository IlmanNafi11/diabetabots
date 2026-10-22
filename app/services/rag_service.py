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
- ``OllamaEmbedder`` (httpx → Ollama ``/api/embed``) at
  ``Settings.ollama_base_url`` — self-hosted arctic replacement for
  ``OpenAIEmbeddings`` (``app/services/ollama_embedder.py``).
- ``Settings`` from ``app.config``.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.services.ollama_embedder import OllamaEmbedder

__all__ = [
    "RAGService",
    "RetrievedFood",
    "build_query_context",
    "build_retrieved_foods",
    "query_terms",
    "rank_retrieved_foods",
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
    energi_per_100g:
        Energy in kcal per 100 g edible portion, or ``None``.
    protein_per_100g:
        Protein in g per 100 g edible portion, or ``None``.
    lemak_per_100g:
        Fat in g per 100 g edible portion, or ``None``.
    karbohidrat_per_100g:
        Carbohydrate in g per 100 g edible portion, or ``None``.
    serat_per_100g:
        Dietary fibre in g per 100 g edible portion, or ``None``.
    gl_value:
        Per-serving glycemic load source value, or ``None``.
    serving_size_g:
        Serving size in grams (for GL computation), or ``None``.
    bdd:
        Bahan yang Dapat Dimakan (0-100 %); ``None`` means 100 %.
    """

    record_id: str
    name: str
    category: str
    gi_value: float | None
    distance: float
    text_snippet: str
    energi_per_100g: float | None = None
    protein_per_100g: float | None = None
    lemak_per_100g: float | None = None
    karbohidrat_per_100g: float | None = None
    serat_per_100g: float | None = None
    gl_value: float | None = None
    serving_size_g: float | None = None
    bdd: float | None = None

    @property
    def page_content(self) -> str:
        """LangChain ``Document.page_content`` alias for ``text_snippet``."""
        return self.text_snippet


class RAGService:
    """RAG retrieval backed by the persistent ChromaDB collection.

    Initialises once on construction: creates an ``OllamaEmbedder`` client
    and a ``chromadb.PersistentClient`` pointing at the configured directory.
    The ChromaDB collection is loaded lazily on first ``retrieve`` call.

    Credential fail-fast for the LLM API key happens in ``LLMService``, not
    here — ``RAGService`` only needs the self-hosted Ollama endpoint.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        # Embedder + ChromaDB client are created eagerly so that construction
        # fails fast on bad configuration; the collection is loaded lazily on
        # first retrieve (it may not exist yet right after a fresh checkout).
        self._embedder: OllamaEmbedder = self._make_embedder()
        self._client: chromadb.PersistentClient = self._make_client()
        self._collection: chromadb.Collection | None = None
        # ``retrieve_async`` fans out to worker threads; the lazy collection
        # load must happen exactly once (double get_collection on a fresh
        # directory races and one caller can hit a missing-collection error).
        self._collection_lock = threading.Lock()

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

        Synchronous and blocking: embeds ``query``, queries ChromaDB for the
        candidate pool, then grounds and ranks the results.

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
            Top-k name-grounded results, ordered by query/name token overlap and
            then cosine distance; empty when no distinctive query token matches a
            TKPI food name.
        """
        requested = top_k if top_k is not None else self._settings.rag_top_k
        candidate_k = max(getattr(self._settings, "rag_candidate_k", requested), requested)
        embedder = self._get_embedder()
        collection = self._get_collection()
        query_vector = embedder.embed_query(query)
        raw = collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
            where=where,
        )
        return rank_retrieved_foods(build_retrieved_foods(raw), query, limit=requested)

    async def retrieve_async(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedFood]:
        """Run the blocking :meth:`retrieve` in a worker thread.

        The embed call and ChromaDB query are synchronous I/O; running them on
        the event loop would block the whole bot for the duration of one Ollama
        request (up to the read timeout).
        """
        return await asyncio.to_thread(self.retrieve, query, top_k, where)

    # ------------------------------------------------------------------
    # Internal helpers (overridable for testing)
    # ------------------------------------------------------------------

    def _get_embedder(self) -> OllamaEmbedder:
        return self._embedder

    def _make_embedder(self) -> OllamaEmbedder:
        """Interactive embedder: fast-fail read timeout, never the batch one."""
        return OllamaEmbedder(
            base_url=self._settings.ollama_base_url,
            model=self._settings.embedding_model_name,
            read_timeout_seconds=self._settings.ollama_query_read_timeout_seconds,
        )

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            with self._collection_lock:
                if self._collection is None:
                    client = self._get_client()
                    self._collection = client.get_collection(name=self._settings.collection_name)
        return self._collection

    def _get_client(self) -> chromadb.PersistentClient:
        return self._client

    def _make_client(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(
            path=self._settings.chroma_persist_dir,
            settings=ChromaSettings(allow_reset=False),
        )

    def close(self) -> None:
        """Release the process-lifetime HTTP pool and ChromaDB client."""
        self._embedder.close()
        self._client.clear_system_cache()


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
                energi_per_100g=_maybe_float(meta.get("energi")),
                protein_per_100g=_maybe_float(meta.get("protein")),
                lemak_per_100g=_maybe_float(meta.get("lemak")),
                karbohidrat_per_100g=_maybe_float(meta.get("karbohidrat")),
                serat_per_100g=_maybe_float(meta.get("serat")),
                gl_value=_maybe_float(meta.get("gl_value")),
                serving_size_g=_maybe_float(meta.get("serving_size_g")),
                bdd=_maybe_float(meta.get("bdd")),
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


#: Words that carry no food-identifying signal. They are stripped from a query
#: before name grounding so that a portion word ("mangkuk") or a generic term
#: ("sayur") cannot make an unrelated TKPI entry look like a match.
GENERIC_QUERY_TERMS: frozenset[str] = frozenset(
    {
        "apa",
        "berapa",
        "banyak",
        "bg",
        "buah",
        "bungkus",
        "butir",
        "cek",
        "dan",
        "dengan",
        "ekor",
        "gelas",
        "gizi",
        "glikemik",
        "gram",
        "indeks",
        "ini",
        "itu",
        "kalori",
        "kandungan",
        "karbohidrat",
        "lauk",
        "lemak",
        "makan",
        "makanan",
        "mangkuk",
        "mengandung",
        "minum",
        "nutrisi",
        "ons",
        "piring",
        "pokok",
        "porsi",
        "potong",
        "protein",
        "saya",
        "sayur",
        "sdm",
        "sdt",
        "sedang",
        "segini",
        "sendok",
        "serat",
        "tinggi",
        "tolong",
        "tidak",
        "untuk",
        "yang",
    }
)


def _name_tokens(text: str) -> list[str]:
    """Split text into lower-case alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def query_terms(query: str) -> tuple[str, ...]:
    """Return the distinctive (non-generic) tokens that identify a query's food.

    A query of only generic/portion words yields an empty tuple; callers must
    then fall back to the raw embedding order rather than filtering everything.
    """
    return tuple(
        token
        for token in _name_tokens(query)
        if token not in GENERIC_QUERY_TERMS and len(token) >= 3
    )


def rank_retrieved_foods(
    foods: Sequence[RetrievedFood],
    query: str,
    limit: int | None = None,
) -> list[RetrievedFood]:
    """Ground retrieved foods against the query's distinctive tokens.

    Embedding similarity alone cannot tell that a query names a food absent
    from the corpus: the nearest neighbour of ``"sayur aster"`` is ``"Ares,
    sayur"``, an unrelated entry. This gate keeps only foods whose name shares
    at least one distinctive query token, ranked by token overlap (then by
    embedding distance), so unrelated queries return an empty list and the
    handler can answer "not found" instead of narrating the wrong food.

    When the query has no distinctive tokens the foods are returned unchanged
    in embedding order.
    """
    terms = query_terms(query)
    if not terms or not foods:
        return list(foods[:limit] if limit is not None else foods)

    scored: list[tuple[int, float, int, RetrievedFood]] = []
    for order, food in enumerate(foods):
        name_tokens = set(_name_tokens(food.name))
        overlap = sum(1 for term in terms if term in name_tokens)
        if overlap == 0:
            continue
        scored.append((-overlap, food.distance, order, food))

    scored.sort()
    ranked = [food for *_, food in scored]
    return ranked[:limit] if limit is not None else ranked


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
