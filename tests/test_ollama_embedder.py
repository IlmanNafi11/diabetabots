"""Tests for app/services/ollama_embedder.py — OllamaEmbedder client."""

import httpx
import pytest

from app.services.ollama_embedder import OllamaEmbedder

_BASE_URL = "http://localhost:11434"
_MODEL = "snowflake-arctic-embed2"


def _fake_post_response(embeddings: list[list[float]]):
    """Return a fake ``httpx.post`` that captures the request and answers."""

    def fake_post(url: str, *, json: dict, timeout: httpx.Timeout) -> httpx.Response:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"embeddings": embeddings}, request=request)

    captured: dict[str, object] = {}
    return fake_post, captured


def test_constructor_performs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the embedder must not touch the network (errors surface later)."""

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("network call during construction")

    monkeypatch.setattr(httpx, "post", fail)

    OllamaEmbedder(base_url=_BASE_URL, model=_MODEL)


def test_embed_query_sends_prefixed_input_to_embed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries are prefixed with ``"query: "`` per the arctic embed2 config."""
    fake_post, captured = _fake_post_response([[0.1, 0.2, 0.3]])
    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = OllamaEmbedder(base_url=_BASE_URL, model=_MODEL)
    vector = embedder.embed_query("nasi putih")

    assert vector == [0.1, 0.2, 0.3]
    assert captured["url"] == f"{_BASE_URL}/api/embed"
    assert captured["json"] == {"model": _MODEL, "input": ["query: nasi putih"]}


def test_embed_documents_sends_texts_without_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents are embedded as-is — no passage prefix."""
    fake_post, captured = _fake_post_response([[0.1], [0.2]])
    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = OllamaEmbedder(base_url=_BASE_URL, model=_MODEL)
    vectors = embedder.embed_documents(["nasi putih", "ayam goreng"])

    assert vectors == [[0.1], [0.2]]
    assert captured["json"] == {"model": _MODEL, "input": ["nasi putih", "ayam goreng"]}


def test_base_url_trailing_slash_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post, captured = _fake_post_response([[0.1]])
    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = OllamaEmbedder(base_url=f"{_BASE_URL}/", model=_MODEL)
    embedder.embed_query("x")

    assert captured["url"] == f"{_BASE_URL}/api/embed"


def test_embed_query_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, json: dict, timeout: httpx.Timeout) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(500, json={"error": "model not found"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)

    embedder = OllamaEmbedder(base_url=_BASE_URL, model=_MODEL)
    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed_query("x")