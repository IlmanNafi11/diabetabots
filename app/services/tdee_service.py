"""Deterministic BMR/TDEE calculations — Mifflin-St Jeor + PAL (SPEC section 5.3)."""

from app.models.models import ActivityLevel, Gender

PAL_OPTIONS: dict[ActivityLevel, float] = {
    ActivityLevel.Sedentari: 1.2,
    ActivityLevel.Ringan: 1.375,
    ActivityLevel.Sedang: 1.55,
    ActivityLevel.Berat: 1.725,
    ActivityLevel.SangatBerat: 1.9,
}


def pal_for_activity(activity: ActivityLevel) -> float:
    """Return the PAL multiplier for a closed-set activity level."""
    return PAL_OPTIONS[activity]


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: Gender) -> float:
    """Return Mifflin-St Jeor BMR in kkal.

    Male: ``10*weight + 6.25*height - 5*age + 5``;
    female: same core with ``- 161``.
    """
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return bmr + 5 if gender == Gender.LakiLaki else bmr - 161


def calculate_tdee(bmr: float, pal: float) -> float:
    """Return TDEE in kkal, rounded to two decimals."""
    return round(bmr * pal, 2)
