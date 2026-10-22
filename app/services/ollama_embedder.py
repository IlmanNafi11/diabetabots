"""Ollama embedding client — httpx → ``POST {base}/api/embed`` (arctic migration).

Self-hosted replacement for ``OpenAIEmbeddings`` in the production embedding
path. Implements the ``EmbeddingClient`` subset used by
``app.data.embedder.index_documents`` (``embed_documents``) plus
``embed_query`` for retrieval.

Queries are prefixed with ``"query: "`` per the official
snowflake-arctic-embed2 configuration; documents are embedded as-is (no
passage prefix). Construction performs no network I/O — errors surface at
retrieve/index time.

Public API
----------
``OllamaEmbedder`` — holds base URL + model; exposes:
  ``embed_query(text) -> list[float]``
  ``embed_documents(texts) -> list[list[float]]``

Dependencies
------------
- ``httpx`` (sync client, reused per instance — one TCP connection pool).
- Model served by Ollama at ``Settings.ollama_base_url``.
"""

from __future__ import annotations

import httpx

__all__ = ["OllamaEmbedder"]


class OllamaEmbedder:
    """LangChain-compatible embedder calling Ollama's ``/api/embed`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str = "snowflake-arctic-embed2:latest",
        read_timeout_seconds: float = 600.0,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._read_timeout_seconds = read_timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=read_timeout_seconds,
                write=30.0,
                pool=10.0,
            ),
        )

    def close(self) -> None:
        """Release the internally owned HTTP connection pool."""
        if self._owns_client:
            self._client.close()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            "/api/embed", json={"model": self._model, "input": list(texts)}
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise ValueError(f"Ollama response missing 'embeddings' key: {payload!r:.200}")
        vectors = payload["embeddings"]
        if len(vectors) != len(texts):
            raise ValueError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} inputs")
        return [list(map(float, vector)) for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents as-is — no passage prefix."""
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the arctic ``"query: "`` prefix."""
        return self._embed([f"query: {text}"])[0]
