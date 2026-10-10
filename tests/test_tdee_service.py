"""Unit tests for BMR/TDEE Mifflin-St Jeor formulas (SPEC section 5.3, §6.2).

Branch coverage target: 100% for ``app/services/tdee_service.py``.
Reference values: PRD section 4 (8 simulation profiles, TDEE acuan).
"""

import pytest

from app.models.models import ActivityLevel, Gender
from app.services.tdee_service import (
    PAL_OPTIONS,
    calculate_bmr,
    calculate_tdee,
    pal_for_activity,
)

# --- PAL mapping ------------------------------------------------------------


class TestPalForActivity:
    """PRD section 3/4: closed PAL set 1,2 / 1,375 / 1,55 / 1,725 / 1,9."""

    @pytest.mark.parametrize(
        ("activity", "pal"),
        [
            (ActivityLevel.Sedentari, 1.2),
            (ActivityLevel.Ringan, 1.375),
            (ActivityLevel.Sedang, 1.55),
            (ActivityLevel.Berat, 1.725),
            (ActivityLevel.SangatBerat, 1.9),
        ],
    )
    def test_pal_for_each_activity_level(self, activity: ActivityLevel, pal: float) -> None:
        assert pal_for_activity(activity) == pal

    def test_pal_options_is_complete_closed_set(self) -> None:
        assert set(PAL_OPTIONS) == set(ActivityLevel)
        assert set(PAL_OPTIONS.values()) == {1.2, 1.375, 1.55, 1.725, 1.9}


# --- BMR --------------------------------------------------------------------


class TestCalculateBmr:
    """BMR = 10*BB + 6,25*TB - 5*usia + 5 [L] / - 161 [P] (Mifflin-St Jeor)."""

    def test_male_bmr_adds_five(self) -> None:
        # 10*70 + 6.25*170 - 5*30 + 5 = 700 + 1062.5 - 150 + 5 = 1617.5 (PRD P1)
        assert calculate_bmr(70, 170, 30, Gender.LakiLaki) == pytest.approx(1617.50)

    def test_female_bmr_subtracts_161(self) -> None:
        # 10*60 + 6.25*160 - 5*28 - 161 = 600 + 1000 - 140 - 161 = 1299.0 (PRD P2)
        assert calculate_bmr(60, 160, 28, Gender.Perempuan) == pytest.approx(1299.00)

    def test_elderly_male_bmr(self) -> None:
        # 10*95 + 6.25*170 - 5*65 + 5 = 950 + 1062.5 - 325 + 5 = 1692.5 (PRD P3)
        assert calculate_bmr(95, 170, 65, Gender.LakiLaki) == pytest.approx(1692.50)

    def test_elderly_female_bmr(self) -> None:
        # 10*85 + 6.25*155 - 5*63 - 161 = 850 + 968.75 - 315 - 161 = 1342.75 (PRD P4)
        assert calculate_bmr(85, 155, 63, Gender.Perempuan) == pytest.approx(1342.75)


# --- TDEE -------------------------------------------------------------------


class TestCalculateTdee:
    """TDEE = round(BMR * PAL, 2)."""

    def test_tdee_rounds_to_two_decimals(self) -> None:
        # P1: 1617.5 * 1.375 = 2224.0625 -> 2224.06
        assert calculate_tdee(1617.50, 1.375) == pytest.approx(2224.06)

    def test_tdee_sedentary_female(self) -> None:
        # P4: 1342.75 * 1.2 = 1611.30
        assert calculate_tdee(1342.75, 1.2) == pytest.approx(1611.30)

    def test_tdee_rounding_half_even_boundary(self) -> None:
        # 1000.005 * 1.0 -> float representation makes this round half-even;
        # the contract is only that the result is rounded to 2 decimals.
        assert calculate_tdee(1000.005, 1.0) == pytest.approx(1000.0, abs=0.01)

    @pytest.mark.parametrize(
        ("bmr", "pal", "expected"),
        [
            (1617.50, 1.375, 2224.06),
            (1692.50, 1.2, 2031.00),
            (1435.00, 1.55, 2224.25),
            (1938.75, 1.2, 2326.50),
        ],
    )
    def test_tdee_against_prd_references(self, bmr: float, pal: float, expected: float) -> None:
        assert calculate_tdee(bmr, pal) == pytest.approx(expected)


# --- Combined pipeline --------------------------------------------------------


class TestBmrTdeePipeline:
    """End-to-end: profile values -> BMR -> TDEE for PRD section 4 profiles."""

    @pytest.mark.parametrize(
        ("gender", "weight", "height", "age", "activity", "expected_tdee"),
        [
            (Gender.LakiLaki, 70, 170, 30, ActivityLevel.Ringan, 2224.06),  # P1
            (Gender.Perempuan, 60, 160, 28, ActivityLevel.Ringan, 1786.13),  # P2
            (Gender.LakiLaki, 95, 170, 65, ActivityLevel.Sedentari, 2031.00),  # P3
            (Gender.Perempuan, 85, 155, 63, ActivityLevel.Sedentari, 1611.30),  # P4
            (Gender.LakiLaki, 100, 175, 32, ActivityLevel.Sedentari, 2326.50),  # P5
            (Gender.Perempuan, 90, 158, 30, ActivityLevel.Sedentari, 1891.80),  # P6
            (Gender.LakiLaki, 68, 168, 60, ActivityLevel.Sedang, 2224.25),  # P7
            (Gender.Perempuan, 58, 152, 58, ActivityLevel.Sedang, 1672.45),  # P8
        ],
    )
    def test_pipeline_matches_prd_reference_tdee(
        self,
        gender: Gender,
        weight: float,
        height: float,
        age: int,
        activity: ActivityLevel,
        expected_tdee: float,
    ) -> None:
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, pal_for_activity(activity))
        # ≤0.5% tolerance per PRD AC3
        assert tdee == pytest.approx(expected_tdee, rel=0.005)
