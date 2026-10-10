"""Structured response formatter for Telegram analysis messages (§3.5 PRD)."""

from dataclasses import dataclass

GI_LOW, GI_MED, GI_HIGH = 55, 69, 70
GL_LOW, GL_MED, GL_HIGH = 10, 19, 20


def _gi_category(gi: int | None) -> str:
    if gi is None:
        return "tidak tersedia"
    if gi <= GI_LOW:
        return "rendah"
    if gi <= GI_MED:
        return "sedang"
    return "tinggi"


def _gl_category(gl: float | None) -> str:
    if gl is None:
        return "tidak tersedia"
    if gl <= GL_LOW:
        return "rendah"
    if gl <= GL_MED:
        return "sedang"
    return "tinggi"


@dataclass(slots=True)
class AnalysisResponse:
    """All fields required to render an analysis message (§3.5 PRD)."""

    nama: str
    kategori: str
    gram: float
    energi: float
    energi_100g: float
    protein: float
    lemak: float
    karbo: float
    serat: float
    gi: int | None
    gl: float | None
    tdee: float
    sisa: float
    feasibility: str


def format_analysis(response: AnalysisResponse) -> str:
    """Render an :class:`AnalysisResponse` as a Telegram-friendly text message.

    Mirrors the ``ANALYSIS_TEMPLATE`` defined in SPEC section 5.4.
    """
    gi_cat = _gi_category(response.gi)
    gl_cat = _gl_category(response.gl)
    gl_text = "tidak tersedia" if response.gl is None else f"{response.gl} (kategori: {gl_cat})"

    return (
        f"🍎 {response.nama} — {response.kategori} (sumber: TKPI 2020)\n"
        f"Untuk {response.gram} gram:\n"
        f"\n"
        f"• Energi: {response.energi} kkal (dari {response.energi_100g} kkal/100 g)\n"
        f"• Protein: {response.protein} g · Lemak: {response.lemak} g · "
        f"Karbohidrat: {response.karbo} g · Serat: {response.serat} g\n"
        f"• Indeks Glikemik: {response.gi} (kategori: {gi_cat}) — atau:\n"
        f'  "Data IG tidak tersedia pada sumber; kualitas karbohidrat tidak dapat dinilai."\n'
        f"• Beban Glikemik aktual: {gl_text}\n"
        f"\n"
        f"⚡ Kebutuhan energi harian (TDEE): {response.tdee} kkal\n"
        f"⚡ Sisa energi hari ini: {response.sisa} kkal\n"
        f"\n"
        f"Kelayakan konsumsi: {response.feasibility}\n"
        f"Sumber data: TKPI 2020 + Atkinson dkk. (2021)\n"
    )
