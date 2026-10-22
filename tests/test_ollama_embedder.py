"""Tests for app/services/ollama_embedder.py — OllamaEmbedder client."""

import json
from typing import Any

import httpx
import pytest

from app.services.ollama_embedder import OllamaEmbedder

_BASE_URL = "http://localhost:11434"
_MODEL = "snowflake-arctic-embed2"


def _capturing_client(
    status_code: int = 200,
    payload: dict[str, Any] | None = None,
    captured: dict[str, Any] | None = None,
    transport_exception: Exception | None = None,
) -> httpx.Client:
    """Build an ``httpx.Client`` whose transport records requests and replies."""

    def handler(request: httpx.Request) -> httpx.Response:
        if transport_exception is not None:
            raise transport_exception
        if captured is not None:
            captured["url"] = str(request.url)
            captured["json"] = json.loads(request.content)
        body = {} if payload is None else payload
        return httpx.Response(status_code, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url=_BASE_URL)


def _embedder(client: httpx.Client, **kwargs: Any) -> OllamaEmbedder:
    kwargs.setdefault("model", _MODEL)
    return OllamaEmbedder(base_url=_BASE_URL, client=client, **kwargs)


def test_constructor_performs_no_network() -> None:
    """Constructing the embedder must not touch the network."""

    class _ExplodingTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise AssertionError("network call during construction")

    OllamaEmbedder(
        base_url=_BASE_URL,
        client=httpx.Client(transport=_ExplodingTransport()),
    )


def test_embed_query_sends_prefixed_input_to_embed_endpoint() -> None:
    """Queries are prefixed with ``"query: "`` per the arctic embed2 config."""
    captured: dict[str, Any] = {}
    embedder = _embedder(
        _capturing_client(payload={"embeddings": [[0.1, 0.2, 0.3]]}, captured=captured)
    )

    vector = embedder.embed_query("nasi putih")

    assert vector == [0.1, 0.2, 0.3]
    assert captured["url"] == f"{_BASE_URL}/api/embed"
    assert captured["json"] == {"model": _MODEL, "input": ["query: nasi putih"]}


def test_embed_documents_sends_texts_without_prefix() -> None:
    """Documents are embedded as-is — no passage prefix."""
    captured: dict[str, Any] = {}
    embedder = _embedder(
        _capturing_client(payload={"embeddings": [[0.1], [0.2]]}, captured=captured)
    )

    vectors = embedder.embed_documents(["nasi putih", "ayam goreng"])

    assert vectors == [[0.1], [0.2]]
    assert captured["json"] == {"model": _MODEL, "input": ["nasi putih", "ayam goreng"]}


def test_custom_read_timeout_is_applied() -> None:
    """A non-default read timeout must reach the HTTP layer."""
    embedder = OllamaEmbedder(
        base_url=_BASE_URL,
        model=_MODEL,
        read_timeout_seconds=999.5,
    )

    assert embedder._client.timeout.read == 999.5
    embedder.close()


def test_default_read_timeout_is_used_when_not_specified() -> None:
    embedder = OllamaEmbedder(base_url=_BASE_URL, model=_MODEL)

    assert embedder._client.timeout.read == 600.0
    embedder.close()


def test_base_url_trailing_slash_is_normalized() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    embedder = OllamaEmbedder(
        base_url=f"{_BASE_URL}/",
        model=_MODEL,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url=f"{_BASE_URL}/"),
    )
    embedder.embed_query("x")

    assert captured["url"] == f"{_BASE_URL}/api/embed"


def test_embed_query_raises_on_http_error() -> None:
    embedder = _embedder(_capturing_client(status_code=500, payload={"error": "model not found"}))

    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed_query("x")


def test_embed_documents_raises_on_http_error() -> None:
    embedder = _embedder(_capturing_client(status_code=404, payload={"error": "not found"}))

    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed_documents(["x"])


def test_connect_error_propagates() -> None:
    embedder = _embedder(
        _capturing_client(transport_exception=httpx.ConnectError("connection refused"))
    )

    with pytest.raises(httpx.ConnectError):
        embedder.embed_query("x")


def test_read_timeout_propagates() -> None:
    embedder = _embedder(_capturing_client(transport_exception=httpx.ReadTimeout("timed out")))

    with pytest.raises(httpx.ReadTimeout):
        embedder.embed_query("x")


def test_missing_embeddings_key_raises_value_error() -> None:
    embedder = _embedder(_capturing_client(payload={"unexpected": "shape"}))

    with pytest.raises(ValueError, match="missing 'embeddings'"):
        embedder.embed_query("x")


def test_non_list_embeddings_raises_value_error() -> None:
    embedder = _embedder(_capturing_client(payload={"embeddings": "nope"}))

    with pytest.raises(ValueError):
        embedder.embed_query("x")


def test_embedding_count_mismatch_raises_value_error() -> None:
    embedder = _embedder(_capturing_client(payload={"embeddings": [[0.1], [0.2]]}))

    with pytest.raises(ValueError, match="for 1 input"):
        embedder.embed_query("x")


def test_non_numeric_embeddings_raise_value_error() -> None:
    embedder = _embedder(_capturing_client(payload={"embeddings": [["not", "a", "number"]]}))

    with pytest.raises(ValueError, match="float"):
        embedder.embed_query("x")


def test_malformed_json_raises_value_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not json")),
        base_url=_BASE_URL,
    )
    embedder = _embedder(client)

    with pytest.raises(ValueError):
        embedder.embed_query("x")


def test_embed_documents_empty_list_returns_empty() -> None:
    """Embedding an empty document list must not touch the network."""
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        raise AssertionError("network must not be called for empty input")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=_BASE_URL)
    embedder = OllamaEmbedder(base_url=_BASE_URL, model=_MODEL, client=client)
    try:
        assert embedder.embed_documents([]) == []
        assert not called
    finally:
        embedder.close()
        client.close()


def test_close_owned_client_releases_pool() -> None:
    """close() must shut down the internally owned client and be idempotent."""
    embedder = OllamaEmbedder(base_url=_BASE_URL, model=_MODEL)
    embedder.close()
    embedder.close()


def test_embed_documents_empty_list_short_circuits() -> None:
    """Empty list should return [] without a server round-trip."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"embeddings": []})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=_BASE_URL)
    embedder = OllamaEmbedder(base_url=_BASE_URL, model=_MODEL, client=client)
    try:
        assert embedder.embed_documents([]) == []
        assert captured == []
    finally:
        embedder.close()
        client.close()


def test_external_client_left_open_after_close() -> None:
    """A client passed in stays usable after embedder.close() (not owned)."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"embeddings": [[0.1] * 3072]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=_BASE_URL)
    embedder = OllamaEmbedder(base_url=_BASE_URL, model=_MODEL, client=client)
    try:
        embedder.embed_query("nasi putih")
        assert len(captured) == 1
    finally:
        embedder.close()  # external client retained — close() is a no-op for it
    # The external client remains usable after embedder.close().
    assert client.get("/api/embed").status_code == 200
    client.close()
