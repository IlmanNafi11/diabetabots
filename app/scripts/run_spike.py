"""Run the binding Iterasi 0 embedding x chunking spike.

The script evaluates three one-food-per-document representations against the
``text-embedding-3-large`` and ``text-embedding-3-small`` models. It writes the
raw retrieval log, a human-readable report, and the winning settings to
``.env``. Real provider credentials are required; this script never fabricates
spike results.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import chromadb
from langchain_openai import OpenAIEmbeddings

from app.config import Settings


@dataclass(frozen=True)
class FoodRecord:
    """One source row from the TKPI+GI dataset."""

    record_id: str
    name: str
    category: str
    nutrients: dict[str, float]
    gi_value: float | None
    gl_value: float | None
    serving_size_g: float | None = None
    available_carbohydrate_raw: float | None = None


@dataclass(frozen=True)
class SpikeDocument:
    """A single embedding document that never crosses a food-entry boundary."""

    record_id: str
    text: str
    metadata: dict[str, str | float]


@dataclass(frozen=True)
class QueryCase:
    """A retrieval query and the food-name terms considered relevant."""

    text: str
    relevant_terms: tuple[str, ...]


class EmbeddingClient(Protocol):
    """Subset of the LangChain embedding interface used by the spike."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


NUTRIENT_FIELDS = (
    "Energi",
    "Protein",
    "Lemak",
    "Karbohidrat",
    "Serat",
    "Air",
    "Abu",
    "Kalsium",
    "Besi",
    "Kalium",
    "Seng",
    "Fosfor",
    "Natrium",
    "Tembaga",
    "Retinol",
    "Karoten Total",
    "Beta Karoten",
    "Thiamin",
    "Riboflavin",
    "Niasin",
    "Vitamin C",
)
CHUNKING_STRATEGIES = ("fixed", "recursive", "late")
EMBEDDING_MODELS = (
    ("text-embedding-3-large", 3072),
    ("text-embedding-3-small", 1536),
)
QUERY_CASES = (
    QueryCase("Berapa kandungan gizi nasi putih?", ("nasi",)),
    QueryCase("Gizi ayam goreng segini banyak apa?", ("ayam", "goreng")),
    QueryCase("Tahu dan tempe mengandung protein berapa?", ("tahu", "tempe")),
    QueryCase("Kalori bayam rebus berapa?", ("bayam",)),
    QueryCase("Indeks glikemik pisang ambon tinggi tidak?", ("pisang ambon",)),
)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)  # type: ignore[arg-type]


def load_records(path: Path) -> list[FoodRecord]:
    """Load JSONL data while preserving exactly one record per non-empty line."""
    records: list[FoodRecord] = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            raw = json.loads(line)
            nutrients = {
                field: float(raw[field])
                for field in NUTRIENT_FIELDS
                if raw.get(field) not in (None, "")
            }
            records.append(
                FoodRecord(
                    record_id=f"food-{index:04d}",
                    name=str(raw["Nama Bahan Pangan"]).replace("\n", " ").strip(),
                    category=str(raw["Kategori"]),
                    nutrients=nutrients,
                    gi_value=_optional_float(raw.get("gi_value")),
                    gl_value=_optional_float(raw.get("gl_value")),
                    serving_size_g=_optional_float(raw.get("serving_size_g")),
                    available_carbohydrate_raw=_optional_float(
                        raw.get("available_carbohydrate_raw")
                    ),
                )
            )
    return records


def _nutrition_text(record: FoodRecord) -> str:
    return ", ".join(f"{name}: {value:g}" for name, value in record.nutrients.items())


def _fixed_text(record: FoodRecord) -> str:
    values = [record.name, record.category, _nutrition_text(record)]
    if record.gi_value is not None:
        values.append(f"IG: {record.gi_value:g}")
    if record.gl_value is not None:
        values.append(f"BG: {record.gl_value:g}")
    return " | ".join(values)


def _recursive_text(record: FoodRecord) -> str:
    glycemic = "Data IG/BG tidak tersedia"
    if record.gi_value is not None:
        glycemic = f"Indeks glikemik: {record.gi_value:g}"
        if record.gl_value is not None:
            glycemic += f"\nBeban glikemik: {record.gl_value:g}"
    return (
        f"Nama bahan pangan: {record.name}\n"
        f"Kategori: {record.category}\n"
        f"Komposisi per 100 gram:\n{_nutrition_text(record)}\n"
        f"{glycemic}"
    )


def _late_text(record: FoodRecord) -> str:
    nutrition = _nutrition_text(record)
    glycemic = "tidak memiliki data indeks glikemik"
    if record.gi_value is not None:
        glycemic = f"memiliki indeks glikemik {record.gi_value:g}"
        if record.gl_value is not None:
            glycemic += f" dan beban glikemik {record.gl_value:g}"
    return (
        f"{record.name} adalah bahan pangan kategori {record.category}. "
        f"Komposisi gizinya per 100 gram adalah {nutrition}. "
        f"Pangan ini {glycemic}."
    )


def build_documents(records: Sequence[FoodRecord], strategy: str) -> list[SpikeDocument]:
    """Create one independently embeddable document for every food record."""
    renderers = {"fixed": _fixed_text, "recursive": _recursive_text, "late": _late_text}
    try:
        renderer = renderers[strategy]
    except KeyError as error:
        supported = ", ".join(CHUNKING_STRATEGIES)
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of: {supported}") from error

    return [
        SpikeDocument(
            record_id=record.record_id,
            text=renderer(record),
            metadata={
                "name": record.name,
                "category": record.category,
                "strategy": strategy,
            },
        )
        for record in records
    ]


def precision_at_k(query: QueryCase, names: Sequence[str], k: int = 5) -> float:
    """Return the fraction of top-k food names relevant to ``query``."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = sum(
        any(term.casefold() in name.casefold() for term in query.relevant_terms)
        for name in names[:k]
    )
    return relevant / k


def choose_winner(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose highest P@5; prefer small for score differences below 0.02."""
    if not results:
        raise ValueError("Cannot choose a winner without results")
    best_score = max(float(result["mean_precision_at_5"]) for result in results)
    tied = [
        result for result in results if best_score - float(result["mean_precision_at_5"]) < 0.02
    ]
    small = [result for result in tied if result["model"] == "text-embedding-3-small"]
    candidates = small or tied
    return min(
        candidates,
        key=lambda result: (
            -float(result["mean_precision_at_5"]),
            float(result["mean_latency_ms"]),
            str(result["strategy"]),
        ),
    )


def _create_embeddings(settings: Settings, model: str, dimension: int) -> EmbeddingClient:
    return OpenAIEmbeddings(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        dimensions=dimension,
        check_embedding_ctx_length=False,
    )


def _evaluate_combination(
    *,
    client: chromadb.ClientAPI,
    embedding: EmbeddingClient,
    documents: Sequence[SpikeDocument],
    queries: Sequence[QueryCase],
    model: str,
    dimension: int,
    strategy: str,
    top_k: int,
    raw_log: Any,
) -> dict[str, Any]:
    collection = client.get_or_create_collection(
        name=f"spike_{strategy}_{dimension}",
        metadata={"hnsw:space": "cosine"},
    )
    collection.delete(where={"strategy": strategy})

    started = time.perf_counter()
    vectors = embedding.embed_documents([document.text for document in documents])
    embedding_ms = (time.perf_counter() - started) * 1000
    collection.add(
        ids=[document.record_id for document in documents],
        documents=[document.text for document in documents],
        embeddings=vectors,
        metadatas=[document.metadata for document in documents],
    )

    query_results: list[dict[str, Any]] = []
    for query in queries:
        query_started = time.perf_counter()
        query_vector = embedding.embed_query(query.text)
        retrieved = collection.query(query_embeddings=[query_vector], n_results=top_k)
        latency_ms = (time.perf_counter() - query_started) * 1000
        names = [str(metadata["name"]) for metadata in retrieved["metadatas"][0]]
        score = precision_at_k(query, names, top_k)
        item = {
            "timestamp": datetime.now(UTC).isoformat(),
            "strategy": strategy,
            "model": model,
            "dimension": dimension,
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
        "strategy": strategy,
        "model": model,
        "dimension": dimension,
        "document_count": len(documents),
        "embedding_latency_ms": round(embedding_ms, 2),
        "mean_precision_at_5": statistics.fmean(
            result["precision_at_5"] for result in query_results
        ),
        "mean_latency_ms": statistics.fmean(result["latency_ms"] for result in query_results),
        "queries": query_results,
    }


def _write_report(
    path: Path,
    *,
    dataset_path: Path,
    results: Sequence[dict[str, Any]],
    winner: dict[str, Any],
    raw_log_path: Path,
) -> None:
    rows = "\n".join(
        f"| {result['strategy']} | `{result['model']}` | {result['dimension']} | "
        f"{result['mean_precision_at_5']:.3f} | {result['mean_latency_ms']:.2f} |"
        for result in results
    )
    body = f"""# Laporan Spike Embedding x Chunking — {datetime.now(UTC).date().isoformat()}

## Tujuan

Membandingkan grid 3x2 strategi representasi dokumen dan model embedding dengan metrik
Precision@5 pada lima kueri PRD §7.3. Setiap entri pangan tetap satu dokumen mandiri.

## Bahan dan Konfigurasi

- Dataset: `{dataset_path}` ({results[0]["document_count"]} entri)
- Strategi: fixed, recursive, late
- Model: text-embedding-3-large (3072), text-embedding-3-small (1536)
- Kueri: 5 kueri PRD §7.3
- Bukti mentah: `{raw_log_path}`

## Hasil

| Strategi | Model | Dimensi | Rata-rata P@5 | Latensi kueri rerata (ms) |
|---|---|---:|---:|---:|
{rows}

## Keputusan

Pemenang: **{winner["strategy"]} x {winner["model"]}** dengan rata-rata P@5
**{winner["mean_precision_at_5"]:.3f}**. Jika selisih skor tertinggi kurang dari 0,02,
model small dipilih sesuai PRD §7.2 untuk biaya dan latensi lebih rendah.

Konfigurasi ini mengikat pengembangan berikutnya dan telah ditulis ke `.env`.
"""
    path.write_text(body, encoding="utf-8")


def _update_env(path: Path, updates: dict[str, str]) -> None:
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    output: list[str] = []
    for line in existing:
        key = line.partition("=")[0].strip()
        if key in remaining and not line.lstrip().startswith("#"):
            output.append(f"{key}={remaining.pop(key)}")
        else:
            output.append(line)
    if output and output[-1] != "":
        output.append("")
    output.extend(f"{key}={value}" for key, value in remaining.items())
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _validate_credentials(settings: Settings) -> None:
    placeholders = ("your-", "example", "changeme")
    if not settings.llm_api_key or any(
        marker in settings.llm_api_key.casefold() for marker in placeholders
    ):
        raise RuntimeError("Set LLM_API_KEY in .env before running the Iterasi 0 spike")
    if any(marker in settings.llm_base_url.casefold() for marker in placeholders):
        raise RuntimeError("Set LLM_BASE_URL in .env before running the Iterasi 0 spike")


def run_spike(
    *,
    dataset_path: Path,
    reports_dir: Path,
    env_path: Path,
    settings: Settings,
) -> dict[str, Any]:
    """Execute the full 3x2 grid and persist the binding winner."""
    _validate_credentials(settings)
    records = load_records(dataset_path)
    if len(records) != 1146:
        raise RuntimeError(f"Expected 1,146 source records, found {len(records):,}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = reports_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = logs_dir / "spike-raw.log"
    report_path = reports_dir / f"laporan-spike-{datetime.now(UTC).date().isoformat()}.md"
    client = chromadb.Client()
    results: list[dict[str, Any]] = []

    with raw_log_path.open("w", encoding="utf-8") as raw_log:
        for strategy in CHUNKING_STRATEGIES:
            documents = build_documents(records, strategy)
            for model, dimension in EMBEDDING_MODELS:
                embedding = _create_embeddings(settings, model, dimension)
                result = _evaluate_combination(
                    client=client,
                    embedding=embedding,
                    documents=documents,
                    queries=QUERY_CASES,
                    model=model,
                    dimension=dimension,
                    strategy=strategy,
                    top_k=5,
                    raw_log=raw_log,
                )
                results.append(result)

    winner = choose_winner(results)
    _update_env(
        env_path,
        {
            "CHUNKING_STRATEGY": str(winner["strategy"]),
            "EMBEDDING_MODEL_NAME": str(winner["model"]),
            "EMBEDDING_DIM": str(winner["dimension"]),
            "SPIKE_PRECISION_AT_5": f"{winner['mean_precision_at_5']:.3f}",
        },
    )
    _write_report(
        report_path,
        dataset_path=dataset_path,
        results=results,
        winner=winner,
        raw_log_path=raw_log_path,
    )
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/tkpi-gi-final.jsonl"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args()
    winner = run_spike(
        dataset_path=args.dataset,
        reports_dir=args.reports_dir,
        env_path=args.env,
        settings=Settings(),
    )
    print(
        f"Winner: {winner['strategy']} x {winner['model']} "
        f"(P@5={winner['mean_precision_at_5']:.3f})"
    )


if __name__ == "__main__":
    main()
