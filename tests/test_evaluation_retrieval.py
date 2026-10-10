"""Unit tests for the retrieval evaluation harness (task-08)."""

import pytest

from app.evaluation.retrieval import (
    QUERY_CASES,
    QueryCase,
    mean_precision_at_5,
    precision_at_k,
)


def test_query_cases_are_the_five_prd_73_queries() -> None:
    """The binding query set must stay in lockstep with the spike."""
    assert len(QUERY_CASES) == 5
    assert QUERY_CASES[0].text == "Berapa kandungan gizi nasi putih?"
    assert QUERY_CASES[-1].relevant_terms == ("pisang ambon",)


def test_precision_at_k_counts_only_relevant_names() -> None:
    query = QueryCase(text="Kalori bayam rebus berapa?", relevant_terms=("bayam",))

    score = precision_at_k(query, ["Bayam, rebus", "Bayam, kukus", "Nasi", "Tempe", "Tahu"], k=5)

    assert score == pytest.approx(0.4)


def test_precision_at_k_matches_terms_case_insensitively() -> None:
    query = QueryCase(text="nasi", relevant_terms=("nasi",))

    score = precision_at_k(query, ["NASI putih", "Tempe", "Tahu", "Ayam", "Ikan"], k=5)

    assert score == pytest.approx(0.2)


def test_precision_at_k_scores_zero_when_nothing_is_relevant() -> None:
    query = QueryCase(text="nasi", relevant_terms=("nasi",))

    score = precision_at_k(query, ["Bayam", "Tempe", "Tahu", "Ayam", "Ikan"], k=5)

    assert score == pytest.approx(0.0)


def test_precision_at_k_treats_a_multi_term_query_as_relevant_for_either_term() -> None:
    query = QueryCase(text="tahu tempe", relevant_terms=("tahu", "tempe"))

    score = precision_at_k(query, ["Tahu, mentah", "Tempe goreng", "Nasi", "Ayam", "Ikan"], k=5)

    assert score == pytest.approx(0.4)


def test_precision_at_k_rejects_non_positive_k() -> None:
    query = QueryCase(text="nasi", relevant_terms=("nasi",))

    with pytest.raises(ValueError):
        precision_at_k(query, ["Nasi"], k=0)


def test_mean_precision_at_5_averages_query_scores() -> None:
    results = [
        {"precision_at_5": 0.8},
        {"precision_at_5": 1.0},
        {"precision_at_5": 1.0},
        {"precision_at_5": 1.0},
        {"precision_at_5": 0.2},
    ]

    assert mean_precision_at_5(results) == pytest.approx(0.8)
