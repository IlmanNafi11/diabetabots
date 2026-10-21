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
- ``httpx`` (sync client, one-shot ``post`` per call).
- Model served by Ollama at ``Settings.ollama_base_url``.
"""

from __future__ import annotations

import httpx

__all__ = ["OllamaEmbedder"]


class OllamaEmbedder:
    """LangChain-compatible embedder calling Ollama's ``/api/embed`` endpoint."""

    def __init__(
        self, base_url: str, model: str = "snowflake-arctic-embed2:latest"
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": list(texts)},
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        )
        response.raise_for_status()
        return [list(map(float, vector)) for vector in response.json()["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents as-is — no passage prefix."""
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the arctic ``"query: "`` prefix."""
        return self._embed([f"query: {text}"])[0]