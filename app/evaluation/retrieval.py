"""Retrieval evaluation harness shared by the test suite and audit scripts.

Encapsulates the exact same five PRD §7.3 queries and the Precision@5 metric
used by the Iterasi 0 spike (``app/scripts/run_spike.py``) so that the
integration test and the raw evidence logs stay in sync with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryCase:
    """A retrieval query and the food-name terms counted as relevant."""

    text: str
    relevant_terms: tuple[str, ...]


QUERY_CASES: tuple[QueryCase, ...] = (
    QueryCase("Berapa kandungan gizi nasi putih?", ("nasi",)),
    QueryCase("Gizi ayam goreng segini banyak apa?", ("ayam", "goreng")),
    QueryCase("Tahu dan tempe mengandung protein berapa?", ("tahu", "tempe")),
    QueryCase("Kalori bayam rebus berapa?", ("bayam",)),
    QueryCase("Indeks glikemik pisang ambon tinggi tidak?", ("pisang ambon",)),
)


def precision_at_k(query: QueryCase, names: Sequence[str], k: int = 5) -> float:
    """Return the fraction of top-k food names relevant to ``query``."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = sum(
        any(term.casefold() in name.casefold() for term in query.relevant_terms)
        for name in names[:k]
    )
    return relevant / k


def mean_precision_at_5(results: Sequence[dict]) -> float:
    """Return the mean ``precision_at_5`` across evaluation results."""
    return sum(float(result["precision_at_5"]) for result in results) / len(results)
