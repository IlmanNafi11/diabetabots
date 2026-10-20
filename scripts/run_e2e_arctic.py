"""Run the ten PRD §7.3 Telegram E2E scenarios against a self-hosted
snowflake-arctic-embed2 index (worktree eksperimental).

Experimental variant of ``scripts/run_e2e.py``: embeds the TKPI+GI dataset
into a separate ChromaDB collection (``tkpi_gi_arctic``) with
``snowflake-arctic-embed2`` served by the local Ollama instance, then drives
the ten PRD §7.3 scenarios through the real ``TelegramUpdateHandler``. The
chat LLM provider comes from the environment (``LLM_*`` / ``.env``); the
production ``tkpi_gi`` collection and baseline E2E logs are never touched.

Artifacts:
    reports/logs/arctic-e2e/e2e-scenario-01.log … e2e-scenario-10.log
    reports/laporan-e2e-arctic-10-scenarios-YYYY-MM-DD.md  (environment
    shows collection ``tkpi_gi_arctic`` + embedding ``snowflake-arctic-embed2``)

Usage:
    uv run python -m scripts.run_e2e_arctic [--index-only | --run-only]
        [--dataset PATH] [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import chromadb
import httpx

from app.config import Settings
from app.evaluation.e2e import E2E_SCENARIOS, Scenario, ScenarioResult, render_e2e_report
from app.services.rag_service import RAGService
from scripts import index_data, run_e2e

OLLAMA_URL = "http://127.0.0.1:11434"
ARCTIC_MODEL = "snowflake-arctic-embed2:latest"
QUERY_PREFIX = "query: "  # official arctic query prefix (docs: no passage prefix)
EXPERIMENT_COLLECTION = "tkpi_gi_arctic"
EXPERIMENT_DIM = 1024
EXPERIMENT_EMBEDDING_MODEL = "snowflake-arctic-embed2"
CHAT_MODEL_ALIAS = "diabetabot"
DEFAULT_DATASET = Path("data/tkpi-gi-final.jsonl")
EXPECTED_DOCUMENTS = 1146


class OllamaArcticEmbedder:
    """LangChain-compatible embedder calling Ollama ``/api/embed`` with the
    official arctic query prefix. Documents embed as-is (no passage prefix)."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = ARCTIC_MODEL) -> None:
        self._base_url = base_url
        self._model = model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": list(texts)},
            timeout=3600,
        )
        response.raise_for_status()
        return [list(map(float, vector)) for vector in response.json()["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"{QUERY_PREFIX}{text}"])[0]


class ArcticRAGService(RAGService):
    """RAGService variant embedding with the local Ollama arctic-embed2."""

    def _make_embedder(self) -> OllamaArcticEmbedder:  # type: ignore[override]
        return OllamaArcticEmbedder()


def _ensure_ollama_serving() -> None:
    """Start ``ollama serve`` if nothing answers, then pull the model."""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/version", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{OLLAMA_URL}/api/version", timeout=2)
                response.raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            raise RuntimeError("ollama serve did not become ready within 60s")
    pull = subprocess.run(["ollama", "pull", ARCTIC_MODEL], capture_output=True, text=True)
    if pull.returncode != 0:
        raise RuntimeError(f"ollama pull {ARCTIC_MODEL} failed: {pull.stderr[-500:]}")


def index_experiment_collection(settings: Settings) -> int:
    """Embed the dataset into ``tkpi_gi_arctic`` with arctic-embed2."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return index_data.index_dataset(
        dataset_path=DEFAULT_DATASET,
        settings=settings,
        client=client,
        embedder=OllamaArcticEmbedder(),
    )


def _experiment_settings() -> Settings:
    """Production settings with only the embedding experiment fields swapped."""
    settings = Settings()
    settings = settings.model_copy(
        update={
            "collection_name": EXPERIMENT_COLLECTION,
            "embedding_dim": EXPERIMENT_DIM,
            "embedding_model_name": EXPERIMENT_EMBEDDING_MODEL,
        }
    )
    if not settings.database_url.endswith("/diabetabot_test"):
        settings = settings.model_copy(update={"database_url": run_e2e.DEFAULT_E2E_DATABASE_URL})
    return settings.model_copy(update={"chat_model_name": CHAT_MODEL_ALIAS})


async def _run_with_retries(
    settings: Settings,
    scenarios: list[Scenario],
    *,
    report_dir: Path,
    log_dir: Path,
    date: str,
    max_retries: int,
) -> list[ScenarioResult]:
    results, _ = await run_e2e.run_all(
        settings,
        scenarios,
        report_dir=report_dir,
        log_dir=log_dir,
        generated_at=date,
    )
    merged = {result.number: result for result in results}
    failed = [scenario for scenario in scenarios if not merged[scenario.number].passed]

    for attempt in range(1, max_retries + 1):
        if not failed:
            break
        print(f"retry {attempt}/{max_retries} for: {[s.number for s in failed]}")
        retried, _ = await run_e2e.run_all(
            settings,
            failed,
            report_dir=report_dir,
            log_dir=log_dir,
            generated_at=date,
        )
        for result in retried:
            merged[result.number] = result
        failed = [scenario for scenario in failed if not merged[scenario.number].passed]

    return [merged[scenario.number] for scenario in scenarios]


def run_experiment(
    settings: Settings,
    *,
    report_dir: Path,
    log_dir: Path,
    date: str,
    scenarios: list[Scenario] | None = None,
    max_retries: int = 3,
) -> int:
    """Run scenarios against arctic, with bounded retries for flaky LLM calls."""
    run_e2e.RAGService = ArcticRAGService  # patch module binding used by run_all
    selected = scenarios if scenarios is not None else list(E2E_SCENARIOS)
    final = asyncio.run(
        _run_with_retries(
            settings,
            selected,
            report_dir=report_dir,
            log_dir=log_dir,
            date=date,
            max_retries=max_retries,
        )
    )

    report_path = report_dir / f"laporan-e2e-arctic-10-scenarios-{date[:10]}.md"
    report = render_e2e_report(
        final,
        generated_at=date,
        environment=run_e2e._safe_environment(settings, run_e2e._collection_count(settings)),
    ).replace("`reports/logs/e2e-", "`reports/logs/arctic-e2e/e2e-")
    report_path.write_text(
        report
        + (
            "\n> **Catatan eksperimen:** raw log per skenario disimpan di "
            f"`reports/logs/arctic-e2e/e2e-scenario-*.log` (koleksi "
            f"{EXPERIMENT_COLLECTION}, embedding {EXPERIMENT_EMBEDDING_MODEL}).\n"
        ),
        encoding="utf-8",
    )
    print(f"report: {report_path}")
    passed = sum(1 for result in final if result.passed)
    return 0 if passed == len(final) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-only", action="store_true", help="embed dataset into tkpi_gi_arctic and stop"
    )
    parser.add_argument("--run-only", action="store_true", help="run E2E scenarios and stop")
    parser.add_argument(
        "--scenario",
        type=int,
        action="append",
        dest="scenarios",
        help="Run only these scenario numbers (repeatable).",
    )
    parser.add_argument(
        "--date",
        default=datetime.now(UTC).astimezone().strftime("%Y-%m-%d"),
        help="Report date stamp (YYYY-MM-DD).",
    )
    args = parser.parse_args(argv)

    settings = _experiment_settings()
    _ensure_ollama_serving()

    if not args.run_only:
        print(f"Indexing {DEFAULT_DATASET} into {EXPERIMENT_COLLECTION!r} ...")
        count = index_experiment_collection(settings)
        print(f"Indexed {count} documents into {EXPERIMENT_COLLECTION!r}")
        if count != EXPECTED_DOCUMENTS:
            print(f"WARNING: expected {EXPECTED_DOCUMENTS}, got {count}", file=sys.stderr)
            return 2

    if args.index_only:
        return 0

    report_dir = Path("reports")
    log_dir = Path("reports/logs/arctic-e2e")
    selected = [
        scenario
        for scenario in E2E_SCENARIOS
        if not args.scenarios or scenario.number in args.scenarios
    ]
    print(f"Running {len(selected)} PRD §7.3 scenarios against tkpi_gi_arctic ...")
    return run_experiment(
        settings, report_dir=report_dir, log_dir=log_dir, date=args.date, scenarios=selected
    )


if __name__ == "__main__":
    raise SystemExit(main())
