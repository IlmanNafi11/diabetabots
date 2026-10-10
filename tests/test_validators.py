"""Unit tests for input validation ranges (SPEC section 3.4, §6.2).

Covers :mod:`app.bot.validators`: gender free-text, age 18-100,
weight 25-300 kg, height 100-250 cm, activity 1-5 + keywords, and
gram parsing with unit suffixes / clarification triggers.
"""

import pytest

from app.bot.validators import (
    ValidationError,
    parse_activity,
    parse_age,
    parse_gender,
    parse_gram,
    parse_height,
    parse_weight,
)
from app.models.models import ActivityLevel, Gender

# --- parse_gender ---------------------------------------------------------------


class TestParseGender:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Laki-laki", Gender.LakiLaki),
            ("laki-laki", Gender.LakiLaki),
            ("laki", Gender.LakiLaki),
            ("pria", Gender.LakiLaki),
            ("L", Gender.LakiLaki),
            ("Perempuan", Gender.Perempuan),
            ("perempuan", Gender.Perempuan),
            ("wanita", Gender.Perempuan),
            ("P", Gender.Perempuan),
            ("  pria  ", Gender.LakiLaki),
        ],
    )
    def test_valid_gender_inputs(self, raw: str, expected: Gender) -> None:
        assert parse_gender(raw) == expected

    @pytest.mark.parametrize("raw", ["", "laki2", "L-L", "perempuan sekali", "abc"])
    def test_invalid_gender_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_gender(raw)


# --- parse_age --------------------------------------------------------------------


class TestParseAge:
    @pytest.mark.parametrize("raw", ["18", "25", "100"])
    def test_valid_age_inclusive_bounds(self, raw: str) -> None:
        assert parse_age(raw) == int(raw)

    @pytest.mark.parametrize("raw", [" 30 "])
    def test_age_accepts_whitespace(self, raw: str) -> None:
        assert parse_age(raw) == 30

    @pytest.mark.parametrize("raw", ["17", "101", "0", "-5"])
    def test_age_out_of_range_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_age(raw)

    @pytest.mark.parametrize("raw", ["tiga", "30.5", ""])
    def test_age_non_integer_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_age(raw)


# --- parse_weight -------------------------------------------------------------------


class TestParseWeight:
    @pytest.mark.parametrize("raw", ["25", "70", "300"])
    def test_valid_weight_inclusive_bounds(self, raw: str) -> None:
        assert parse_weight(raw) == float(raw)

    def test_weight_accepts_comma_decimal(self) -> None:
        assert parse_weight("70,5") == pytest.approx(70.5)

    @pytest.mark.parametrize("raw", ["24.9", "300.1", "0", "-70"])
    def test_weight_out_of_range_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_weight(raw)

    @pytest.mark.parametrize("raw", ["berat", "", "70 kg"])
    def test_weight_non_numeric_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_weight(raw)


# --- parse_height ---------------------------------------------------------------------


class TestParseHeight:
    @pytest.mark.parametrize("raw", ["100", "170", "250"])
    def test_valid_height_inclusive_bounds(self, raw: str) -> None:
        assert parse_height(raw) == float(raw)

    def test_height_accepts_comma_decimal(self) -> None:
        assert parse_height("170,5") == pytest.approx(170.5)

    @pytest.mark.parametrize("raw", ["99.9", "250.1", "0", "-170"])
    def test_height_out_of_range_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_height(raw)

    @pytest.mark.parametrize("raw", ["tinggi", "", "170 cm"])
    def test_height_non_numeric_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_height(raw)


# --- parse_activity ---------------------------------------------------------------------


class TestParseActivity:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", ActivityLevel.Sedentari),
            ("2", ActivityLevel.Ringan),
            ("3", ActivityLevel.Sedang),
            ("4", ActivityLevel.Berat),
            ("5", ActivityLevel.SangatBerat),
            ("Sedentari", ActivityLevel.Sedentari),
            ("sedentari", ActivityLevel.Sedentari),
            ("ringan", ActivityLevel.Ringan),
            ("sedang", ActivityLevel.Sedang),
            ("berat", ActivityLevel.Berat),
            ("sangat berat", ActivityLevel.SangatBerat),
            ("sangat berat sekali", ActivityLevel.SangatBerat),
            ("tidak aktif", ActivityLevel.Sedentari),
            ("santai", ActivityLevel.Sedang),
        ],
    )
    def test_valid_activity_inputs(self, raw: str, expected: ActivityLevel) -> None:
        assert parse_activity(raw) == expected

    @pytest.mark.parametrize("raw", ["6", "0", "olahraga", "", "numpang lewat"])
    def test_invalid_activity_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_activity(raw)


# --- parse_gram ---------------------------------------------------------------------------


class TestParseGram:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("150", 150.0),
            ("150.5", 150.5),
            ("150,5", 150.5),
            (" 150 ", 150.0),
            ("150g", 150.0),
            ("150 g", 150.0),
            ("150gram", 150.0),
            ("150 gram", 150.0),
            ("150gr", 150.0),
            ("150 gr.", 150.0),
        ],
    )
    def test_valid_gram_inputs(self, raw: str, expected: float) -> None:
        assert parse_gram(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["0", "0g", "-150", "150.0.0"])
    def test_non_positive_or_malformed_gram_raises(self, raw: str) -> None:
        with pytest.raises(ValidationError):
            parse_gram(raw)

    @pytest.mark.parametrize("raw", ["1 piring", "2 butir", "secukupnya"])
    def test_non_mass_portion_raises_for_clarification(self, raw: str) -> None:
        """Non-gram portions must trigger the AWAIT_GRAM_CLARIFY flow (SPEC 5.5)."""
        with pytest.raises(ValidationError):
            parse_gram(raw)
