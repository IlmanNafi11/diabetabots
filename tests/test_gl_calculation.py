"""Unit tests for nutrition portion + GL formulas (SPEC section 5.3, D4/D5).

Covers ``nutrition_for_gram`` (per-100g scaling with BDD), ``glycemic_load``
(per-serving GL scaled to the requested portion), and GI/GL category
thresholds. Pure deterministic math — no LLM involved.
"""

import pytest

from app.services.nutrition_service import (
    GI_LOW,
    GI_MED,
    GL_LOW,
    GL_MED,
    gi_category,
    gl_category,
    glycemic_load,
    nutrition_for_gram,
)

# --- nutrition_for_gram -------------------------------------------------------


class TestNutritionForGram:
    """nutrition_for_gram = round(per_100g * gram * bdd / 100 / 100, 2)."""

    def test_simple_scaling(self) -> None:
        # 250 kkal/100g eaten as 150 g -> 375 kkal
        assert nutrition_for_gram(250.0, 150.0) == pytest.approx(375.0)

    def test_bdd_reduces_edible_portion(self) -> None:
        # BDD 80% (e.g. fruit with peel): 250 * 150 * 80 / 100 / 100 = 300
        assert nutrition_for_gram(250.0, 150.0, bdd=80.0) == pytest.approx(300.0)

    def test_rounds_to_two_decimals(self) -> None:
        # 33.3 kkal * 37 g / 100 = 12.321 -> 12.32
        assert nutrition_for_gram(33.3, 37.0) == pytest.approx(12.32)

    def test_zero_gram_is_zero(self) -> None:
        assert nutrition_for_gram(250.0, 0.0) == 0.0

    def test_full_bdd_default_matches_simple_scaling(self) -> None:
        assert nutrition_for_gram(120.0, 80.0) == nutrition_for_gram(120.0, 80.0, bdd=100.0)


# --- glycemic_load ------------------------------------------------------------


class TestGlycemicLoad:
    """glycemic_load = round(gl_value * gram / serving_size_g, 2)."""

    def test_double_portion_doubles_gl(self) -> None:
        # GL 15 per 100 g serving, eaten 200 g -> 30
        assert glycemic_load(15.0, 200.0, 100.0) == pytest.approx(30.0)

    def test_half_portion_halves_gl(self) -> None:
        assert glycemic_load(15.0, 50.0, 100.0) == pytest.approx(7.5)

    def test_rounds_to_two_decimals(self) -> None:
        # 15.37 * 123 / 100 = 18.9051 -> 18.91
        assert glycemic_load(15.37, 123.0, 100.0) == pytest.approx(18.91)

    def test_zero_serving_size_raises(self) -> None:
        with pytest.raises(ValueError, match="serving"):
            glycemic_load(15.0, 100.0, 0.0)

    def test_negative_serving_size_raises(self) -> None:
        with pytest.raises(ValueError, match="serving"):
            glycemic_load(15.0, 100.0, -50.0)


# --- GI category thresholds -----------------------------------------------------


class TestGiCategory:
    """GI: <= 55 rendah, 56-69 sedang, >= 70 tinggi, None tidak tersedia."""

    @pytest.mark.parametrize(
        ("gi", "expected"),
        [
            (0, "rendah"),
            (GI_LOW, "rendah"),
            (GI_LOW + 1, "sedang"),
            (GI_MED, "sedang"),
            (GI_MED + 1, "tinggi"),
            (100, "tinggi"),
        ],
    )
    def test_gi_thresholds(self, gi: int, expected: str) -> None:
        assert gi_category(gi) == expected

    def test_gi_none(self) -> None:
        assert gi_category(None) == "tidak tersedia"


# --- GL category thresholds -----------------------------------------------------


class TestGlCategory:
    """GL: <= 10 rendah, 11-19 sedang, >= 20 tinggi, None tidak tersedia."""

    @pytest.mark.parametrize(
        ("gl", "expected"),
        [
            (0.0, "rendah"),
            (GL_LOW, "rendah"),
            (10.5, "sedang"),
            (GL_MED, "sedang"),
            (GL_MED + 0.5, "tinggi"),
            (45.0, "tinggi"),
        ],
    )
    def test_gl_thresholds(self, gl: float, expected: str) -> None:
        assert gl_category(gl) == expected

    def test_gl_none(self) -> None:
        assert gl_category(None) == "tidak tersedia"
