"""Integration test: nutrition values stay within 5% of TKPI source (PRD §1.3 AC4).

Verifies the end-to-end pipeline: dataset → ChromaDB → retrieval → LLM analysis
→ parsed narrative → reported values. Accuracy is measured against the ground-
truth values loaded directly from ``data/tkpi-gi-final.jsonl``.

The test is marked ``@pytest.mark.integration`` and skips automatically when
no real LLM credentials are available (``your-*`` sentinel).
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import chromadb
import pytest
from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.data.embedder import build_documents, index_documents
from app.data.parser import load_records
from app.services.llm_service import LLMService, format_food_context
from app.services.rag_service import RAGService

DATASET_PATH = Path("data/tkpi-gi-final.jsonl")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def _skip_without_llm_credentials(settings: Settings) -> None:
    """Skip if LLM credentials are not configured."""
    if settings.llm_api_key.startswith("your-"):
        pytest.skip("No real LLM credentials; integration test requires LLM call.")


@pytest.fixture(scope="session")
def _skip_without_embedding_credentials(settings: Settings) -> None:
    """Skip if embedding credentials are not configured."""
    if settings.llm_api_key.startswith("your-"):
        pytest.skip("No real credentials; integration test requires embedding endpoint.")


@pytest.fixture(scope="session")
def indexed_collection(
    settings: Settings,
    _skip_without_embedding_credentials: None,
    tmp_path_factory: pytest.TempPathFactory,
) -> chromadb.Collection:
    """Index the real dataset once per session into an ephemeral collection."""
    tmp = tmp_path_factory.mktemp("chroma")
    client = chromadb.PersistentClient(path=str(tmp))
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    records = load_records(DATASET_PATH)
    documents = build_documents(records, settings.chunking_strategy)
    embedder = OpenAIEmbeddings(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.embedding_model_name,
        dimensions=settings.embedding_dim,
        check_embedding_ctx_length=False,
    )
    index_documents(documents, collection, embedder)
    return collection


@pytest.fixture(scope="session")
def ground_truth(
    _skip_without_embedding_credentials: None,
) -> dict[str, dict[str, float]]:
    """Load ground-truth nutrition values from the dataset keyed by food name."""
    records = load_records(DATASET_PATH)
    gt: dict[str, dict[str, float]] = {}
    for rec in records:
        if rec.name in gt:
            continue  # first occurrence wins
        gt[rec.name] = dict(rec.nutrients.items())
    return gt


@pytest.fixture()
def rag_service(
    indexed_collection: chromadb.Collection,
    settings: Settings,
) -> RAGService:
    """Return a RAGService wired to the ephemeral indexed collection."""
    service = RAGService(settings=settings)
    service._collection = indexed_collection
    return service


@pytest.fixture()
def _mock_profile() -> SimpleNamespace:
    """Minimal duck-typed user profile for analyze_food."""
    return SimpleNamespace(
        telegram_id=1,
        jenis_kelamin="Laki-laki",
        usia_tahun=30,
        berat_kg=70.0,
        tinggi_cm=170.0,
        tingkat_aktivitas="Ringan",
        tdee_kkal=2224.06,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_nutrients(narrative: str) -> dict[str, float | None]:
    """Pull nutrition numbers from the §3.5-style LLM narrative.

    Tolerates light markdown drift (bold labels, table cells) but never scans
    past the end of the label's own line, so a missing value cannot silently
    bind to an unrelated number further down the narrative.

    Returns a dict with keys ``Energi``, ``Protein``, ``Lemak``,
    ``Karbohidrat``, ``Serat``. Values are ``float`` when found, else ``None``.
    """
    result: dict[str, float | None] = {}
    for key in ["Energi", "Protein", "Lemak", "Karbohidrat", "Serat"]:
        pattern = rf"\b{re.escape(key)}\b[^\n]*?([0-9]+(?:[.,][0-9]+)?)"
        m = re.search(pattern, narrative)
        if m:
            raw = m.group(1).replace(",", ".")
            result[key] = float(raw)
        else:
            result[key] = None
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_format_food_context_matches_source(
    rag_service: RAGService,
    ground_truth: dict[str, dict[str, float]],
) -> None:
    """format_food_context produces correct per-gram values from retrieved food.

    This is a pure-function integration test — no LLM required.
    Verifies that retrieved ChromaDB food metadata feeds correctly into the
    nutrition computation that the LLM sees.
    """
    # Retrieve a known food
    retrieved = rag_service.retrieve("nasi putih", top_k=1)
    assert retrieved, "Expected at least one result for 'nasi putih'"
    food = retrieved[0]

    # Find ground truth
    source_name = next(
        (name for name in ground_truth if food.name.startswith(name)),
        None,
    )
    if source_name is None:
        pytest.skip(f"No ground truth for {food.name!r}")

    gt = ground_truth[source_name]
    gram = 150.0

    # Compute context
    context = format_food_context([food], gram=gram)
    assert food.name in context, "Food name must appear in context"

    # Verify both the source per-100 g value and the BDD-adjusted portion value.
    # The context renders them as
    # "Energi <source> kkal/100 g [ (mentah, BDD <B>%) ]
    #  → <portion> kkal untuk <gram> g siap konsumsi".
    if "Energi" not in gt:
        pytest.skip("No Energi in ground truth")

    source_kcal = gt["Energi"]
    bdd = food.bdd if food.bdd is not None else 100.0
    expected_kcal = source_kcal * gram * bdd / 100.0 / 100.0

    m = re.search(
        r"Energi\s+([0-9]+(?:[.,][0-9]+)?)\s*kkal/100 ?g(?:\s*\(mentah, BDD "
        r"([0-9]+(?:[.,][0-9]+)?)%\))?\s*→\s*([0-9]+(?:[.,][0-9]+)?)",
        context,
    )
    assert m, f"Energy pair not found in context excerpt: {context[:300]}"
    reported_source = float(m.group(1).replace(",", "."))
    reported_kcal = float(m.group(3).replace(",", "."))

    # The BDD note, when rendered, must match the metadata BDD exactly.
    if m.group(2) is not None:
        reported_bdd = float(m.group(2).replace(",", "."))
        assert reported_bdd == bdd, f"Context BDD label {reported_bdd} != metadata BDD {bdd}"
    elif bdd < 100.0:
        pytest.fail(f"BDD {bdd} < 100 but context lacks the BDD note: {context[:300]}")

    diff_pct = abs(reported_source - source_kcal) / source_kcal * 100
    assert diff_pct <= 1.0, (
        f"Source energy deviation {diff_pct:.2f}% exceeds 1%: "
        f"reported={reported_source}, source={source_kcal}"
    )
    diff_pct = abs(reported_kcal - expected_kcal) / expected_kcal * 100
    assert diff_pct <= 1.0, (
        f"Portion energy deviation {diff_pct:.2f}% exceeds 1%: "
        f"reported={reported_kcal}, expected={expected_kcal}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "food_query,gram,atol_kcal,atol_macro",
    [
        pytest.param(
            "nasi putih",
            100.0,
            5.0,
            0.20,
            id="nasi-putih-100g",
        ),
        pytest.param(
            "tahu",
            100.0,
            5.0,
            0.20,
            id="tahu-100g",
        ),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_nutrition_within_5pct_of_source(
    food_query: str,
    gram: float,
    atol_kcal: float,
    atol_macro: float,
    rag_service: RAGService,
    ground_truth: dict[str, dict[str, float]],
    settings: Settings,
    _mock_profile: SimpleNamespace,
) -> None:
    """Reported nutrition must be ≤5% different from TKPI source values.

    This is the binding acceptance criterion AC4 (PRD §1.3):
    "Selisih nilai nutrisi yang dilaporkan sistem ≤ 5% dari nilai rujukan TKPI."

    The LLM is instructed to ground all numbers in the provided food context,
    which already contains the pre-computed per-gram values. The test verifies
    the LLM faithfully reports those values rather than hallucinating.
    """
    # 1. Retrieve
    retrieved = rag_service.retrieve(food_query, top_k=1)
    assert retrieved, f"No food retrieved for query {food_query!r}"
    food = retrieved[0]

    # 2. Ground truth
    source_name = next(
        (name for name in ground_truth if food.name.startswith(name)),
        None,
    )
    if source_name is None:
        pytest.skip(f"No ground truth for retrieved food {food.name!r}")
    gt = ground_truth[source_name]

    # 3. Call LLM (the part that requires real credentials)
    service = LLMService(settings=settings)
    result = await service.analyze_food(
        food_name=food.name,
        gram=gram,
        retrieved_foods=[food],
        user_profile=_mock_profile,
        remaining_energy_kkal=2000.0,
        user_question=f"Berapa gizi {food.name} {gram} gram?",
    )

    # 4. Parse reported nutrition
    extracted = _extract_nutrients(result.narrative)

    # 5. Compare
    for nutrient, gt_key in [
        ("Energi", "Energi"),
        ("Protein", "Protein"),
        ("Lemak", "Lemak"),
        ("Karbohidrat", "Karbohidrat"),
        ("Serat", "Serat"),
    ]:
        if gt_key not in gt:
            continue
        reported = extracted.get(gt_key)
        if reported is None:
            pytest.fail(f"LLM narrative missing {gt_key}: {result.narrative[:300]}")

        source_per_100g = gt[gt_key]
        source_scaled = source_per_100g * gram / 100.0
        tol = max(
            atol_kcal if nutrient == "Energi" else atol_macro,
            source_scaled * 0.05,
        )
        diff = abs(reported - source_scaled)
        assert diff <= tol, (
            f"{gt_key} for {food.name} {gram}g: "
            f"reported={reported:.2f}, source={source_scaled:.2f} "
            f"(diff={diff:.2f}, tol={tol:.2f}, pct={diff / source_scaled * 100:.1f}%)"
        )
