"""Unit tests for response format §3.5 (SPEC section 5.4, §6.2).

Covers :mod:`app.bot.responses`: GI/GL category thresholds mirrored in the
formatter and the exact ANALYSIS_TEMPLATE rendering.
"""

import pytest

from app.bot.responses import AnalysisResponse, format_analysis


def _response(**overrides: object) -> AnalysisResponse:
    defaults: dict[str, object] = {
        "nama": "Nasi Putih",
        "kategori": "Karbohidrat",
        "gram": 150.0,
        "energi": 195.0,
        "energi_100g": 130.0,
        "protein": 3.9,
        "lemak": 0.45,
        "karbo": 42.75,
        "serat": 0.6,
        "gi": 73,
        "gl": 27.4,
        "tdee": 2224.06,
        "sisa": 1849.06,
        "feasibility": "Aman dikonsumsi hari ini.",
    }
    defaults.update(overrides)
    return AnalysisResponse(**defaults)  # type: ignore[arg-type]


class TestFormatAnalysisStructure:
    def test_header_contains_name_category_and_source(self) -> None:
        text = format_analysis(_response())
        first_line = text.splitlines()[0]
        assert "🍎 Nasi Putih — Karbohidrat (sumber: TKPI 2020)" in first_line

    def test_portion_line(self) -> None:
        assert "Untuk 150.0 gram:" in format_analysis(_response())

    def test_nutrient_lines(self) -> None:
        text = format_analysis(_response())
        assert "• Energi: 195.0 kkal (dari 130.0 kkal/100 g)" in text
        assert "• Protein: 3.9 g · Lemak: 0.45 g · Karbohidrat: 42.75 g · Serat: 0.6 g" in text

    def test_tdee_and_sisa_lines(self) -> None:
        text = format_analysis(_response())
        assert "⚡ Kebutuhan energi harian (TDEE): 2224.06 kkal" in text
        assert "⚡ Sisa energi hari ini: 1849.06 kkal" in text

    def test_feasibility_and_sources_footer(self) -> None:
        text = format_analysis(_response())
        assert "Kelayakan konsumsi: Aman dikonsumsi hari ini." in text
        assert "Sumber data: TKPI 2020 + Atkinson dkk. (2021)" in text


class TestGiCategoriesInOutput:
    @pytest.mark.parametrize(
        ("gi", "expected"),
        [
            (55, "rendah"),
            (56, "sedang"),
            (69, "sedang"),
            (70, "tinggi"),
        ],
    )
    def test_gi_category_rendered(self, gi: int, expected: str) -> None:
        text = format_analysis(_response(gi=gi))
        assert f"Indeks Glikemik: {gi} (kategori: {expected})" in text

    def test_gi_none_renders_placeholder(self) -> None:
        text = format_analysis(_response(gi=None, gl=None))
        assert "Indeks Glikemik: None (kategori: tidak tersedia)" in text


class TestGlCategoriesInOutput:
    @pytest.mark.parametrize(
        ("gl", "expected"),
        [
            (10.0, "rendah"),
            (10.5, "sedang"),
            (19.0, "sedang"),
            (19.5, "tinggi"),
        ],
    )
    def test_gl_category_rendered(self, gl: float, expected: str) -> None:
        text = format_analysis(_response(gl=gl))
        assert f"Beban Glikemik aktual: {gl} (kategori: {expected})" in text

    def test_gl_none_renders_tidak_tersedia(self) -> None:
        text = format_analysis(_response(gl=None))
        assert "Beban Glikemik aktual: tidak tersedia" in text
