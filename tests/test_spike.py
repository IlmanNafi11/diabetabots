"""Tests for the Iterasi 0 embedding x chunking spike."""

from pathlib import Path

import pytest

from app.scripts.run_spike import (
    FoodRecord,
    QueryCase,
    build_documents,
    choose_winner,
    load_records,
    precision_at_k,
)


@pytest.fixture()
def sample_record() -> FoodRecord:
    return FoodRecord(
        record_id="food-0001",
        name="Nasi putih",
        category="serealia olahan",
        nutrients={"Energi": 175.0, "Protein": 3.2},
        gi_value=64.0,
        gl_value=29.0,
    )


def test_load_records_preserves_one_food_per_record(tmp_path: Path) -> None:
    source = tmp_path / "foods.jsonl"
    source.write_text(
        '{"Nama Bahan Pangan":"Nasi putih","Kategori":"serealia",'
        '"Energi":175,"Protein":3.2,"gi_value":64,"gl_value":29}\n'
        '{"Nama Bahan Pangan":"Bayam rebus","Kategori":"sayuran",'
        '"Energi":16,"Protein":0.9,"gi_value":null,"gl_value":null}\n',
        encoding="utf-8",
    )

    records = load_records(source)

    assert [record.record_id for record in records] == ["food-0000", "food-0001"]
    assert [record.name for record in records] == ["Nasi putih", "Bayam rebus"]
    assert records[0].gi_value == 64.0
    assert records[1].gi_value is None


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "late"])
def test_build_documents_keeps_each_food_as_one_document(
    sample_record: FoodRecord, strategy: str
) -> None:
    documents = build_documents([sample_record], strategy)

    assert len(documents) == 1
    assert documents[0].record_id == sample_record.record_id
    assert "Nasi putih" in documents[0].text
    assert documents[0].metadata["name"] == "Nasi putih"
    assert documents[0].metadata["strategy"] == strategy


def test_precision_at_k_counts_only_relevant_food_names() -> None:
    query = QueryCase(
        text="Berapa kandungan gizi nasi putih?",
        relevant_terms=("nasi",),
    )

    score = precision_at_k(
        query,
        ["Nasi putih", "Beras giling", "Nasi tim", "Mi basah", "Jagung rebus"],
        k=5,
    )

    assert score == pytest.approx(0.4)


def test_choose_winner_prefers_small_when_scores_are_within_tie_threshold() -> None:
    results = [
        {
            "strategy": "recursive",
            "model": "text-embedding-3-large",
            "dimension": 3072,
            "mean_precision_at_5": 0.804,
            "mean_latency_ms": 120.0,
        },
        {
            "strategy": "recursive",
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "mean_precision_at_5": 0.800,
            "mean_latency_ms": 70.0,
        },
        {
            "strategy": "fixed",
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "mean_precision_at_5": 0.700,
            "mean_latency_ms": 60.0,
        },
    ]

    winner = choose_winner(results)

    assert winner["model"] == "text-embedding-3-small"
    assert winner["strategy"] == "recursive"
