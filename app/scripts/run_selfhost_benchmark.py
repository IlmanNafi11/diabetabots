"""Self-hosted embedding benchmark (worktree eksperimental).

Evaluates four open-source embedding candidates against the five PRD §7.3
queries with Precision@5 + latency, strategy ``recursive``, dataset
``data/tkpi-gi-final.jsonl`` (1146 entries). One model is evaluated at a time
and deleted from disk before the next candidate loads (``--keep-models``
disables cleanup). Baseline OpenAI P@5 = 0.800 is a stored reference — the
provider key is blocked, so it is never re-run here.

Run with the benchmark venv (has fastembed + onnxruntime):
    HF_ENDPOINT=https://hf-mirror.com \
      .venv-bench/bin/python -m app.scripts.run_selfhost_benchmark \
      --dataset data/tkpi-gi-final.jsonl --reports-dir reports
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import shutil
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Protocol

import chromadb
import httpx

from app.data.embedder import build_documents
from app.data.parser import load_records
from app.evaluation.retrieval import QUERY_CASES, mean_precision_at_5, precision_at_k

BASELINE_PRECISION_AT_5 = 0.800  # text-embedding-3-large, Iterasi 0 spike (stored)
BASELINE_MODEL = "text-embedding-3-large (OpenAI, key blocked — stored result)"
BASELINE_LATENCY_MS = 581.96  # mean query latency from reports/laporan-spike-2026-10-10.md

QWEN3_INSTRUCT = (
    "Instruct: Given a food nutrition query, retrieve the most relevant food entries\n"
    "Query: "
)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
E5_INSTRUCT_TASK = "Given a food nutrition query, retrieve the most relevant food entries"
E5_INSTRUCT_QUERY_PREFIX = f"Instruct: {E5_INSTRUCT_TASK}\nQuery: "

NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_PASSAGE_PREFIX = "search_document: "
ARCTIC_QUERY_PREFIX = "query: "

FASTEMBED_CACHE_DIR = Path("/tmp/selfhost-bench-fastembed")
OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_BATCH_SIZE = 100
OLLAMA_BATCH_SIZE = 20


@dataclass(frozen=True)
class Candidate:
    """One self-hosted embedding candidate in the benchmark grid."""

    key: str
    label: str
    runtime: str  # "fastembed" | "ollama"
    model: str  # fastembed model name or ollama tag
    dim: int
    query_prefix: str | None
    passage_prefix: str | None


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        key="qwen3-0.6b",
        label="Qwen3-Embedding-0.6B",
        runtime="fastembed",
        model="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        query_prefix=QWEN3_INSTRUCT,
        passage_prefix=None,
    ),
    Candidate(
        key="qwen3-4b",
        label="Qwen3-Embedding-4B (GGUF Q4)",
        runtime="ollama",
        model="qwen3-embedding:4b",
        dim=2560,
        query_prefix=QWEN3_INSTRUCT,
        passage_prefix=None,
    ),
    Candidate(
        key="bge-m3",
        label="BGE-M3",
        runtime="ollama",
        model="bge-m3:latest",
        query_prefix=None,
        passage_prefix=None,
        dim=1024,
    ),
    Candidate(
        key="e5-large",
        label="multilingual-e5-large",
        runtime="fastembed",
        model="intfloat/multilingual-e5-large",
        dim=1024,
        query_prefix=E5_QUERY_PREFIX,
        passage_prefix=E5_PASSAGE_PREFIX,
    ),
    Candidate(
        key="nomic-v2-moe",
        label="nomic-embed-text-v2-moe",
        runtime="ollama",
        model="nomic-embed-text-v2-moe:latest",
        dim=768,
        query_prefix=NOMIC_QUERY_PREFIX,
        passage_prefix=NOMIC_PASSAGE_PREFIX,
    ),
    Candidate(
        key="arctic-embed2",
        label="snowflake-arctic-embed2",
        runtime="ollama",
        model="snowflake-arctic-embed2:latest",
        dim=1024,
        query_prefix=ARCTIC_QUERY_PREFIX,
        passage_prefix=None,
    ),
    Candidate(
        key="e5-large-instruct",
        label="multilingual-e5-large-instruct",
        runtime="fastembed",
        model="intfloat/multilingual-e5-large-instruct",
        dim=1024,
        query_prefix=E5_INSTRUCT_QUERY_PREFIX,
        passage_prefix=None,
    ),
)


class EmbeddingClient(Protocol):
    """Dense embedding interface used by the benchmark."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    @property
    def prefix_note(self) -> str: ...


def _register_custom_model(model: str) -> None:
    """Register ONNX models that ship in the HF repo itself (no Qdrant ONNX
    mirror) so ``TextEmbedding`` can load them through its custom registry."""
    if model != "intfloat/multilingual-e5-large-instruct":
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    with contextlib.suppress(ValueError):
        # Already registered in this process (rerun/aggregation path).
        TextEmbedding.add_custom_model(
            model=model,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=model),
            dim=1024,
            model_file="onnx/model.onnx",
            description=(
                "Multilingual (~100 languages), 512 input tokens truncation, "
                "instruction-tuned retrieval queries (2025)."
            ),
            license="mit",
            size_in_gb=2.24,
            additional_files=["onnx/model.onnx_data"],
        )


class FastembedClient:
    """fastembed ONNX client; prefixes applied here per model best practice."""

    def __init__(
        self,
        model: str,
        query_prefix: str | None,
        passage_prefix: str | None,
        cache_dir: Path,
    ) -> None:
        from fastembed import TextEmbedding

        _register_custom_model(model)
        self._model = TextEmbedding(
            model_name=model,
            cache_dir=str(cache_dir),
            threads=max(1, (__import__("os").cpu_count() or 2) - 1),
        )
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = (
            [f"{self._passage_prefix}{text}" for text in texts]
            if self._passage_prefix
            else texts
        )
        vectors: list[list[float]] = []
        for start in range(0, len(prefixed), EMBEDDING_BATCH_SIZE):
            batch = prefixed[start : start + EMBEDDING_BATCH_SIZE]
            vectors.extend(vector.tolist() for vector in self._model.embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"{self._query_prefix}{text}" if self._query_prefix else text
        return next(self._model.embed([prefixed])).tolist()

    @property
    def prefix_note(self) -> str:
        return (
            f"query={self._query_prefix!r} passage={self._passage_prefix!r} (applied by client)"
        )


class OllamaClient:
    """Ollama /api/embed client; applies prefixes only when the server template
    does not already wrap prompts."""

    def __init__(
        self,
        model: str,
        query_prefix: str | None,
        passage_prefix: str | None,
        base_url: str = OLLAMA_URL,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._server_applies_prefix = self._template_wraps_prompt()

    def _template_wraps_prompt(self) -> bool:
        try:
            show = httpx.get(f"{self._base_url}/api/show", params={"model": self._model})
            show.raise_for_status()
            template = show.json().get("template", "")
            return "{{ .Prompt }}" not in template
        except httpx.HTTPError:
            # Model not pulled yet or server down: fall back to client prefixes.
            return False

    def _embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self._base_url}/api/embed", json=payload, timeout=3600
        )
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        input_ = (
            [f"{self._passage_prefix}{text}" for text in texts]
            if self._passage_prefix and not self._server_applies_prefix
            else texts
        )
        vectors: list[list[float]] = []
        for start in range(0, len(input_), OLLAMA_BATCH_SIZE):
            batch = input_[start : start + OLLAMA_BATCH_SIZE]
            data = self._embed({"model": self._model, "input": batch})
            vectors.extend(list(map(float, vector)) for vector in data["embeddings"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        input_ = (
            f"{self._query_prefix}{text}"
            if self._query_prefix and not self._server_applies_prefix
            else text
        )
        data = self._embed({"model": self._model, "input": input_})
        return list(map(float, data["embeddings"][0]))

    @property
    def prefix_note(self) -> str:
        return (
            f"query={self._query_prefix!r} passage={self._passage_prefix!r}; "
            f"server_template_applies_prefix={self._server_applies_prefix}"
        )


def _make_client(candidate: Candidate) -> EmbeddingClient:
    if candidate.runtime == "fastembed":
        return FastembedClient(
            model=candidate.model,
            query_prefix=candidate.query_prefix,
            passage_prefix=candidate.passage_prefix,
            cache_dir=FASTEMBED_CACHE_DIR,
        )
    if candidate.runtime == "ollama":
        return OllamaClient(
            model=candidate.model,
            query_prefix=candidate.query_prefix,
            passage_prefix=candidate.passage_prefix,
        )
    raise ValueError(f"Unknown runtime {candidate.runtime!r}")


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _ollama_model_size_bytes(model: str) -> int:
    """Return exact model blob bytes from Ollama's HTTP API."""
    response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=30)
    response.raise_for_status()
    for entry in response.json().get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            return int(entry.get("size", 0))
    return 0


def _delete_model(candidate: Candidate, keep: bool) -> None:
    if keep:
        return
    if candidate.runtime == "fastembed":
        shutil.rmtree(FASTEMBED_CACHE_DIR, ignore_errors=True)
    else:
        subprocess.run(["ollama", "rm", candidate.model], capture_output=True, check=False)


def _ensure_ollama_serving() -> subprocess.Popen[bytes] | None:
    """Start ``ollama serve`` if no server answers; return the process when we
    started it (so the caller can stop it afterwards)."""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/version", timeout=5)
        response.raise_for_status()
        return None
    except httpx.HTTPError:
        process = subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{OLLAMA_URL}/api/version", timeout=2)
                response.raise_for_status()
                return process
            except httpx.HTTPError:
                time.sleep(1)
        process.kill()
        raise RuntimeError("ollama serve did not become ready within 60s") from None


def _pull_ollama_model(model: str) -> None:
    pull = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
    if pull.returncode != 0:
        raise RuntimeError(f"ollama pull {model} failed: {pull.stderr[-500:]}")


def _evaluate_candidate(
    *,
    candidate: Candidate,
    documents: Sequence[Any],
    queries: Sequence[Any],
    raw_log: Any,
    top_k: int,
) -> dict[str, Any]:
    client = _make_client(candidate)
    collection = chromadb.Client().get_or_create_collection(
        name=f"selfhost_{candidate.key}",
        metadata={"hnsw:space": "cosine"},
    )
    texts = [document.text for document in documents]
    ids = [document.record_id for document in documents]
    metadatas = [document.metadata for document in documents]
    _log_event(
        raw_log,
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "candidate": candidate.key,
            "event": "embedding_start",
            "document_count": len(documents),
            "batch_size": (
                OLLAMA_BATCH_SIZE
                if candidate.runtime == "ollama"
                else EMBEDDING_BATCH_SIZE
            ),
        },
    )

    started = time.perf_counter()
    vectors = client.embed_documents(texts)
    embedding_ms = (time.perf_counter() - started) * 1000
    collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
    _log_event(
        raw_log,
        {
            "event": "embedding_complete",
            "candidate": candidate.key,
            "model": candidate.model,
            "dimension": len(vectors[0]),
            "document_count": len(vectors),
            "embedding_latency_ms": round(embedding_ms, 2),
            "prefix_note": client.prefix_note,
        },
    )

    query_results: list[dict[str, Any]] = []
    for query in queries:
        query_started = time.perf_counter()
        query_vector = client.embed_query(query.text)
        retrieved = collection.query(query_embeddings=[query_vector], n_results=top_k)
        latency_ms = (time.perf_counter() - query_started) * 1000
        names = [str(metadata["name"]) for metadata in retrieved["metadatas"][0]]
        score = precision_at_k(query, names, top_k)
        item = {
            "timestamp": datetime.now(UTC).isoformat(),
            "candidate": candidate.key,
            "model": candidate.model,
            "runtime": candidate.runtime,
            "dimension": candidate.dim,
            "query": query.text,
            "relevant_terms": list(query.relevant_terms),
            "retrieved_names": names,
            "distances": retrieved["distances"][0],
            "precision_at_5": score,
            "latency_ms": round(latency_ms, 2),
        }
        raw_log.write(json.dumps(item, ensure_ascii=False) + "\n")
        raw_log.flush()
        query_results.append(item)

    return {
        "candidate": candidate.key,
        "label": candidate.label,
        "model": candidate.model,
        "runtime": candidate.runtime,
        "dimension": candidate.dim,
        "mean_precision_at_5": mean_precision_at_5(query_results),
        "mean_latency_ms": statistics.fmean(
            result["latency_ms"] for result in query_results
        ),
        "embedding_latency_ms": embedding_ms,
        "queries": query_results,
    }


def _log_event(raw_log: Any, event: dict[str, Any]) -> None:
    raw_log.write(json.dumps(event, ensure_ascii=False) + "\n")
    raw_log.flush()


def _candidate_failed(raw_log: Any, candidate: Candidate, error: str) -> dict[str, Any]:
    _log_event(
        raw_log,
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "candidate": candidate.key,
            "event": "failed",
            "error": error,
        },
    )
    return {
        "candidate": candidate.key,
        "label": candidate.label,
        "model": candidate.model,
        "runtime": candidate.runtime,
        "dimension": candidate.dim,
        "mean_precision_at_5": None,
        "mean_latency_ms": None,
        "embedding_latency_ms": None,
        "failed": error,
    }


def recommend(results: Sequence[dict[str, Any]], baseline: float) -> str:
    """Pick the best passing candidate or state that OpenAI stays."""
    passed = [
        result
        for result in results
        if result.get("mean_precision_at_5") is not None
        and result["mean_precision_at_5"] >= baseline
    ]
    if not passed:
        return (
            f"Tidak ada kandidat mencapai P@5 >= {baseline:.3f}. "
            "Rekomendasi: tetap text-embedding-3-large (dengan catatan kondisi key)."
        )
    best = max(passed, key=lambda r: r["mean_precision_at_5"])
    return (
        f"{best['label']} ({best['model']}) layak menggantikan OpenAI: "
        f"mean P@5 = {best['mean_precision_at_5']:.3f} >= {baseline:.3f} dengan biaya 0."
    )


def _write_report(
    path: Path,
    *,
    dataset_path: Path,
    results: Sequence[dict[str, Any]],
    recommendation: str,
    raw_log_path: Path,
) -> None:
    rows = "\n".join(
        f"| {result['label']} | {result['runtime']} | {result['dimension']} | "
        f"{result['mean_precision_at_5']:.3f} | {result['mean_latency_ms']:.2f} | "
        f"{result['embedding_latency_ms']:.2f} |"
        if result.get("mean_precision_at_5") is not None
        else f"| {result['label']} | {result['runtime']} | {result['dimension']} | "
        f"GAGAL ({result.get('failed', '')[:60]}) | — | — |"
        for result in results
    )
    body = f"""# Laporan Benchmark Embedding Self-Host — {datetime.now(UTC).date().isoformat()}

## Tujuan

Membandingkan kandidat embedding open-source yang di-self-host di host ini
(2 vCPU, tanpa GPU) dengan baseline OpenAI pada lima kueri PRD §7.3, strategi
`recursive`, dataset `data/tkpi-gi-final.jsonl`. Ambang lolos: mean P@5 >=
{BASELINE_PRECISION_AT_5:.3f}.

## Bahan dan Konfigurasi

- Dataset: `{dataset_path}` ({len(load_records(dataset_path))} entri)
- Strategi chunking: recursive (satu dokumen per entri pangan)
- Kueri: 5 kueri PRD §7.3 (verbatim, `app/evaluation/retrieval.py`)
- Top-k: 5, ruang: cosine (ChromaDB ephemeral per kandidat)
- Baseline: {BASELINE_MODEL} — mean P@5 = {BASELINE_PRECISION_AT_5:.3f},
  latensi kueri rerata ~{BASELINE_LATENCY_MS:.0f} ms (angka tersimpan
  `reports/laporan-spike-2026-10-10.md`; key provider diblokir, tidak di-re-run)
- Bukti mentah: `{raw_log_path}`

## Hasil

| Kandidat | Runtime | Dimensi | P@5 | Latensi kueri (ms) | Latensi embed (ms) |
|---|---|---:|---:|---:|---:|
{rows}

## Keputusan

**Rekomendasi:** {recommendation}

Catatan per kandidat (detail di raw log): status load model, prefix yang
dipakai, dimensi aktual, dan status hapus model pasca-evaluasi.
"""
    path.write_text(body, encoding="utf-8")


def run_benchmark(
    *,
    dataset_path: Path,
    reports_dir: Path,
    candidate_keys: Sequence[str],
    keep_models: bool,
) -> dict[str, Any]:
    """Execute the grid serially, one model on disk at a time."""
    records = load_records(dataset_path)
    documents = build_documents(records, "recursive")

    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = reports_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = logs_dir / "selfhost-raw.log"
    report_path = reports_dir / f"laporan-selfhost-{datetime.now(UTC).date().isoformat()}.md"

    selected = [c for c in CANDIDATES if c.key in candidate_keys]
    missing = [k for k in candidate_keys if k not in {c.key for c in CANDIDATES}]
    if missing:
        raise ValueError(f"Unknown candidate keys: {', '.join(missing)}")
    if not selected:
        raise ValueError("No candidates selected")

    ollama_server = None
    if any(c.runtime == "ollama" for c in selected):
        ollama_server = _ensure_ollama_serving()

    results: list[dict[str, Any]] = []
    try:
        with raw_log_path.open(
            "a" if raw_log_path.exists() else "w", encoding="utf-8"
        ) as raw_log:
            _log_event(
                raw_log,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "event": "benchmark_start",
                    "dataset": str(dataset_path),
                    "strategy": "recursive",
                    "candidates": [c.key for c in selected],
                    "baseline_precision_at_5": BASELINE_PRECISION_AT_5,
                },
            )
            for candidate in selected:
                started = time.perf_counter()
                try:
                    if candidate.runtime == "ollama":
                        _pull_ollama_model(candidate.model)
                        _log_event(
                            raw_log,
                            {
                                "timestamp": datetime.now(UTC).isoformat(),
                                "candidate": candidate.key,
                                "event": "model_ready",
                                "model_size_bytes": _ollama_model_size_bytes(
                                    candidate.model
                                ),
                            },
                        )
                    result = _evaluate_candidate(
                        candidate=candidate,
                        documents=documents,
                        queries=QUERY_CASES,
                        raw_log=raw_log,
                        top_k=5,
                    )
                except Exception as error:  # one candidate must not abort the grid
                    result = _candidate_failed(raw_log, candidate, str(error)[:500])
                finally:
                    if candidate.runtime == "ollama" and not keep_models:
                        subprocess.run(
                            ["ollama", "rm", candidate.model],
                            capture_output=True,
                            check=False,
                        )
                    elif candidate.runtime == "fastembed":
                        _delete_model(candidate, keep=keep_models)
                _log_event(
                    raw_log,
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "candidate": candidate.key,
                        "event": "candidate_done",
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
                results.append(result)
                gc.collect()
    finally:
        if ollama_server is not None:
            ollama_server.terminate()

    recommendation = recommend(results, BASELINE_PRECISION_AT_5)
    _write_report(
        report_path,
        dataset_path=dataset_path,
        results=results,
        recommendation=recommendation,
        raw_log_path=raw_log_path,
    )
    return {"results": results, "recommendation": recommendation, "report_path": report_path}


def render_report_from_log(
    *,
    raw_log_path: Path,
    reports_dir: Path,
    dataset_path: Path,
) -> Path:
    """Render the aggregate report from a completed raw log (no models)."""
    lines = [
        json.loads(line)
        for line in raw_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        queries = [
            line
            for line in lines
            if line.get("candidate") == candidate.key and "precision_at_5" in line
        ]
        if not queries:
            continue
        if len(queries) != len(QUERY_CASES):
            results.append(
                _candidate_failed(
                    StringIO(), candidate, "incomplete query results in raw log"
                )
            )
            continue
        embedding_ms = next(
            (
                float(line["embedding_latency_ms"])
                for line in lines
                if line.get("candidate") == candidate.key
                and "embedding_latency_ms" in line
            ),
            0.0,
        )
        results.append(
            {
                "candidate": candidate.key,
                "label": candidate.label,
                "model": candidate.model,
                "runtime": candidate.runtime,
                "dimension": candidate.dim,
                "mean_precision_at_5": mean_precision_at_5(queries),
                "mean_latency_ms": statistics.fmean(
                    float(query["latency_ms"]) for query in queries
                ),
                "embedding_latency_ms": embedding_ms,
                "queries": queries,
            }
        )
    recommendation = recommend(results, BASELINE_PRECISION_AT_5)
    report_path = (
        reports_dir / f"laporan-selfhost-{datetime.now(UTC).date().isoformat()}.md"
    )
    _write_report(
        report_path,
        dataset_path=dataset_path,
        results=results,
        recommendation=recommendation,
        raw_log_path=raw_log_path,
    )
    return report_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/tkpi-gi-final.jsonl"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--candidates",
        default=",".join(c.key for c in CANDIDATES),
        help="comma-separated candidate keys (default: all)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="render the aggregate report from the existing raw log (no downloads)",
    )
    parser.add_argument(
        "--keep-models", action="store_true", help="keep downloaded models on disk"
    )
    args = parser.parse_args()
    if args.report_only:
        report_path = render_report_from_log(
            raw_log_path=args.reports_dir / "logs" / "selfhost-raw.log",
            reports_dir=args.reports_dir,
            dataset_path=args.dataset,
        )
        print(f"Report: {report_path}")
        return
    outcome = run_benchmark(
        dataset_path=args.dataset,
        reports_dir=args.reports_dir,
        candidate_keys=[key.strip() for key in args.candidates.split(",") if key.strip()],
        keep_models=args.keep_models,
    )
    print(outcome["recommendation"])
    print(f"Report: {outcome['report_path']}")


if __name__ == "__main__":
    main()