"""Deterministic nutrition portion + glycemic load math (SPEC section 5.3, D4/D5).

All numbers are computed here in pure Python — the LLM never produces
nutrition, GI, or GL values (SPEC section 7.3).
"""

GI_LOW, GI_MED, GI_HIGH = 55, 69, 70
GL_LOW, GL_MED, GL_HIGH = 10, 19, 20


def nutrition_for_gram(per_100g: float, gram: float, bdd: float = 100.0) -> float:
    """Scale a per-100 g nutrient value to the eaten portion.

    ``bdd`` (Bahan Degree of Digestibility, %) reduces the edible portion for
    raw foods; defaults to 100 (no adjustment).
    """
    return round(per_100g * gram * bdd / 100 / 100, 2)


def glycemic_load(gl_value: float, gram: float, serving_size_g: float) -> float:
    """Scale a per-serving GL source value to the requested portion in grams.

    Raises :class:`ValueError` for non-positive ``serving_size_g`` — the
    source data must always provide a serving size.
    """
    if serving_size_g <= 0:
        raise ValueError(f"serving_size_g must be positive, got {serving_size_g}")
    return round(gl_value * gram / serving_size_g, 2)


def gi_category(gi: int | None) -> str:
    """Categorise GI: <= 55 rendah, 56-69 sedang, >= 70 tinggi."""
    if gi is None:
        return "tidak tersedia"
    if gi <= GI_LOW:
        return "rendah"
    if gi <= GI_MED:
        return "sedang"
    return "tinggi"


def gl_category(gl: float | None) -> str:
    """Categorise GL: <= 10 rendah, 11-19 sedang, >= 20 tinggi."""
    if gl is None:
        return "tidak tersedia"
    if gl <= GL_LOW:
        return "rendah"
    if gl <= GL_MED:
        return "sedang"
    return "tinggi"
