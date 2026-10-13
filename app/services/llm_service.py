"""LLM analysis service — task-10.

Composes the 4-part prompt (PERKENI 2021 clinical rules, retrieved food
context with pre-computed nutrition/GL values, user profile, user question)
and calls the LLM via LangChain ChatOpenAI.

All nutrition, GI, and GL numbers are computed deterministically in
``app.services.nutrition_service`` before being injected into the prompt.
The LLM narrates; it never produces the numbers.

Public API
----------
``LLMService`` — holds LLM client; exposes:
  ``analyze_food(food_name, gram, retrieved_foods, user_profile,
                 remaining_energy_kkal, user_question) -> NutritionResult``

``build_system_prompt()`` — 4-part system prompt with PERKENI rules.

``build_user_prompt(context, question)`` — user-side prompt injection.

``format_food_context(foods, gram)`` — renders RetrievedFood list with
computed nutrition for the requested gram portion.

``NutritionResult`` — result dataclass with ``narrative: str``.

Dependencies
-----------
- ``langchain_openai.ChatOpenAI`` (OpenAI-compatible).
- ``Settings`` from ``app.config``.
- ``RetrievedFood`` from ``app.services.rag_service``.
- ``nutrition_for_gram``, ``glycemic_load``, ``gi_category``, ``gl_category``
  from ``app.services.nutrition_service``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.services.nutrition_service import (
    gi_category,
    gl_category,
    glycemic_load,
    nutrition_for_gram,
)

if TYPE_CHECKING:
    from app.services.rag_service import RetrievedFood

__all__ = [
    "LLMService",
    "NoFoodRetrievedError",
    "NutritionResult",
    "build_system_prompt",
    "build_user_prompt",
    "format_food_context",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NutritionResult:
    """Structured result of a food analysis call.

    Attributes
    ----------
    narrative:
        The LLM's free-form response in Bahasa Indonesia, grounded in the
        retrieved food documents and formatted per PRD §3.5.
    """

    narrative: str


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class NoFoodRetrievedError(Exception):
    """Raised when ``analyze_food`` is called with an empty retrieved list."""

    def __init__(self, food_name: str) -> None:
        self.food_name = food_name
        super().__init__(f"No food documents retrieved for '{food_name}'. Cannot perform analysis.")


# ---------------------------------------------------------------------------
# Prompt builders (pure — no I/O)
# ---------------------------------------------------------------------------

# Clinical reference text embedded directly in the system prompt (PRD §7.1).
_PERKENI_CLINICAL_RULES = """Pedoman PERKENI 2021 — Terapi Nutrisi Medis:
• Asupan karbohidrat: 45-65% dari total energi harian.
• Asupan lemak: 20-25% dari total energi harian (maksimum 30%).
• Karbohidrat berserat tinggi lebih diutamakan.
• Pola makan bersifat individual: sesuaikan dengan kondisi klinis, preferensi, dan budaya pasien."""


def build_system_prompt() -> str:
    """Return the 4-part system prompt for food analysis.

    Sections (PRD §7.1):
    (a) System instructions + PERKENI 2021 clinical rules.
    (b) Placeholder for retrieved food documents (injected at call time).
    (c) Placeholder for user profile (injected at call time).
    (d) Instruction to follow the §3.5 response format.
    """
    return f"""Anda adalah asisten gizi klinis Indonesia yang didukung data TKPI 2020 dan Atkinson dkk. (2021).

{_PERKENI_CLINICAL_RULES}

--- DATASET ---
Konteks pangan berikut berasal dari basis data TKPI 2020 (komposisi zat gizi) dan Atkinson dkk. 2021 (indeks glikemik):
KONTEKS_PANGAN

--- PROFIL PENGGUNA ---
PROFIL_PENGGUNA

--- ATURAN ANALISIS ---
1. Semua nilai nutrisi, IG, dan BG HARUS berasal dari dokumen konteks yang diberikan. JANGAN mengarang angka.
2. Gunakan nilai porsi yang sudah dihitung sistem: pada konteks pangan, angka setelah panah "→" (contoh: "→ 47,22 kkal untuk 100 g siap konsumsi") adalah nilai porsi OTORITATIF. JANGAN menghitung ulang secara proporsional dari nilai per 100 g: pada pangan berlabel BDD < 100%, nilai per 100 g adalah nilai pangan mentah (belum dikurangi bagian tidak dapat dimakan), sedangkan nilai setelah panah sudah dikoreksi BDD.
3. BG aktual = gl_value * gram_konsumsi / serving_size_g (jika gl_value tersedia).
4. Jika data IG tidak tersedia dalam konteks, nyatakan secara eksplisit: "Data IG tidak tersedia pada sumber; kualitas karbohidrat tidak dapat dinilai."
5. Nilai sisa energi harian dan TDEE disisipkan oleh sistem. Gunakan untuk menilai kelayakan konsumsi.

--- FORMAT RESPONS (§3.5 PRD) ---
Gunakan format berikut untuk respons Anda:

🍎 [Nama Pangan] — [Kategori] (sumber: TKPI 2020)
Untuk [X gram]:

• Energi: [E] kkal untuk [X gram] (sistem telah menghitung nilai ini; JANGAN hitung ulang) [opsional: — dari [E/100g] kkal/100 g BDD [B]%]
• Protein: [P] g · Lemak: [L] g · Karbohidrat: [KH] g · Serat: [S] g
• Indeks Glikemik: [GI] (kategori: rendah/sedang/tinggi) — atau:
  "Data IG tidak tersedia pada sumber; kualitas karbohidrat tidak dapat dinilai."
• Beban Glikemik aktual: [GL] (kategori: rendah ≤10 / sedang 11-19 / tinggi ≥20)

⚡ Kebutuhan energi harian (TDEE): [TDEE] kkal
⚡ Sisa energi hari ini: [sisa] kkal

Kelayakan konsumsi: [penilaian berdasarkan sisa energi, kategori IG/BG, dan acuan PERKENI 2021]
Sumber data: TKPI 2020 + Atkinson dkk. (2021)"""


def format_food_context(foods: list[RetrievedFood], gram: float) -> str:
    """Render a list of RetrievedFood as a human-readable food context string.

    Computes per-portion nutrition values for the given ``gram`` before
    formatting, so the LLM receives accurate numbers.

    Parameters
    ----------
    foods:
        Retrieved food documents (top-k results).
    gram:
        The portion size in grams requested by the user.

    Returns
    -------
    str
        A newline-separated block of formatted food entries.
    """
    lines: list[str] = []
    for food in foods:
        bdd = food.bdd if food.bdd is not None else 100.0
        # BDD (Berat Dapat Dimakan) < 100 means the per-100 g values describe the
        # raw food; only the BDD fraction is edible, so portion values below are
        # BDD-adjusted. The context labels both operands so the LLM never has to
        # guess which number is authoritative (scenario 10 lesson, PRD §7.3 #10).
        bdd_note = f" (mentah, BDD {bdd:g}%)" if bdd < 100 else ""
        # Compute per-gram values; fall back to "tidak tersedia" if metadata missing
        # (graceful degradation for entries indexed before nutrition metadata was added)
        if food.energi_per_100g is not None:
            energi = nutrition_for_gram(food.energi_per_100g, gram, bdd)  # type: ignore[arg-type]
            energi_per_100g = food.energi_per_100g
        else:
            energi = "tidak tersedia"
            energi_per_100g = "tidak tersedia"
        if food.protein_per_100g is not None:
            protein = nutrition_for_gram(food.protein_per_100g, gram, bdd)  # type: ignore[arg-type]
        else:
            protein = "tidak tersedia"
        if food.lemak_per_100g is not None:
            lemak = nutrition_for_gram(food.lemak_per_100g, gram, bdd)  # type: ignore[arg-type]
        else:
            lemak = "tidak tersedia"
        if food.karbohidrat_per_100g is not None:
            karbo = nutrition_for_gram(food.karbohidrat_per_100g, gram, bdd)  # type: ignore[arg-type]
        else:
            karbo = "tidak tersedia"
        if food.serat_per_100g is not None:
            serat = nutrition_for_gram(food.serat_per_100g, gram, bdd)  # type: ignore[arg-type]
        else:
            serat = "tidak tersedia"

        gi_str = (
            f"IG: {int(food.gi_value)} (kategori: {gi_category(int(food.gi_value))})"
            if food.gi_value is not None
            else "IG: tidak tersedia"
        )

        gl_parts: list[str] = []
        if food.gl_value is not None and food.serving_size_g is not None:
            try:
                actual_gl = glycemic_load(food.gl_value, gram, food.serving_size_g)  # type: ignore[arg-type]
                gl_parts.append(f"BG aktual: {actual_gl} (kategori: {gl_category(actual_gl)})")
            except ValueError:
                gl_parts.append("BG: tidak tersedia")
        else:
            gl_parts.append("BG: tidak tersedia (tidak ada data ukuran porsi)")

        line = (
            f"- {food.name} ({food.category}): "
            f"Energi {energi_per_100g} kkal/100 g{bdd_note} "
            f"→ {energi} kkal untuk {gram} g siap konsumsi, "
            f"Protein {protein}g, Lemak {lemak}g, Karbohidrat {karbo}g, Serat {serat}g; "
            f"{gi_str}; {', '.join(gl_parts)}"
        )
        lines.append(line)

    return "\n".join(lines)


def build_user_prompt(
    food_context: str,
    user_question: str,
    profile_block: str = "",
) -> str:
    """Return the user-side prompt with food context, profile, and the question.

    Parameters
    ----------
    food_context:
        Output of ``format_food_context``.
    user_question:
        The user's original question in Bahasa Indonesia.
    profile_block:
        Formatted user profile + energy state (TDEE, sisa energi).
        Part (c) of the 4-part prompt (PRD §7.1).

    Returns
    -------
    str
        The assembled user prompt.
    """
    profile_section = f"Profil pengguna:\n{profile_block}\n\n" if profile_block else ""
    return f"""{profile_section}Analisis konsumsi pangan berdasarkan konteks berikut:

{"-" * 60}
{food_context}
{"-" * 60}

Pertanyaan pengguna: {user_question}

Berikan analisis dalam format §3.5 menggunakan data di atas."""


# ---------------------------------------------------------------------------
# User profile type (passed in by the caller — no ORM dependency here)
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    """Minimal user profile passed to ``analyze_food``."""

    telegram_id: int
    jenis_kelamin: str  # "Laki-laki" or "Perempuan"
    usia_tahun: int
    berat_kg: float
    tinggi_cm: float
    tingkat_aktivitas: str
    tdee_kkal: float


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------


class LLMService:
    """LLM-backed food analysis service using LangChain + OpenAI-compatible API.

    Initialises once on construction; the LLM client is created lazily on first
    ``analyze_food`` call so that the service can be instantiated before
    credentials are available (graceful degradation).

    Raises
    ------
    RuntimeError
        When the API key is still a sentinel ``"your-*"`` value.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._check_credentials()
        self._llm: ChatOpenAI | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_food(
        self,
        food_name: str,
        gram: float,
        retrieved_foods: list[RetrievedFood],
        user_profile: UserProfile | Any,  # accept duck-typed profiles
        remaining_energy_kkal: float,
        user_question: str,
    ) -> NutritionResult:
        """Perform food analysis via LLM with 4-part grounded prompt.

        Parameters
        ----------
        food_name:
            The food being queried (used in the no-results error).
        gram:
            Portion size in grams requested by the user.
        retrieved_foods:
            Top-k results from the RAG retrieval pipeline.
        user_profile:
            Any duck-typed object with the UserProfile fields (tdee_kkal,
            jenis_kelamin, usia_tahun, berat_kg, tinggi_cm,
            tingkat_aktivitas, telegram_id).
        remaining_energy_kkal:
            The user's remaining daily energy budget (from energy_service).
        user_question:
            The original free-text question in Bahasa Indonesia.

        Returns
        -------
        NutritionResult
            LLM narrative response grounded in the retrieved documents.

        Raises
        ------
        NoFoodRetrievedError
            When ``retrieved_foods`` is empty.
        """
        if not retrieved_foods:
            raise NoFoodRetrievedError(food_name)

        # Format food context with pre-computed per-gram values
        food_context = format_food_context(retrieved_foods, gram)

        # Build user profile block (part c of 4-part prompt — in user message)
        profile_block = (
            f"Telegram ID: {user_profile.telegram_id}\n"
            f"Jenis kelamin: {user_profile.jenis_kelamin}\n"
            f"Usia: {user_profile.usia_tahun} tahun\n"
            f"Berat badan: {user_profile.berat_kg} kg\n"
            f"Tinggi badan: {user_profile.tinggi_cm} cm\n"
            f"Tingkat aktivitas: {user_profile.tingkat_aktivitas}\n"
            f"TDEE (kebutuhan energi harian): {user_profile.tdee_kkal:.2f} kkal\n"
            f"Sisa energi hari ini: {remaining_energy_kkal:.2f} kkal"
        )

        # Assemble 4-part prompts
        # Part (a): system prompt with PERKENI rules + placeholders (filled with data)
        system_prompt = (
            build_system_prompt()
            .replace("PROFIL_PENGGUNA", profile_block)
            .replace("KONTEKS_PANGAN", food_context)
        )

        # Parts (c) + (d): profile + food context + question in user message
        user_prompt = build_user_prompt(food_context, user_question, profile_block)

        # Call LLM
        narrative = await self._call_llm(system_prompt, user_prompt)
        return NutritionResult(narrative=narrative)

    # ------------------------------------------------------------------
    # Internal helpers (overridable for testing)
    # ------------------------------------------------------------------

    def _make_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key,
            model=self._settings.chat_model_name,
            temperature=self._settings.llm_temperature,
            # Live gateway measurements (2026-10-13): the real system prompt
            # completes in 34-38 s at temperature 0; 30 s aborted requests the
            # provider would have answered. 90 s covers that tail latency and
            # still expires on a hung connection.
            request_timeout=90.0,
        )

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = self._make_llm()
        return self._llm

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Invoke the configured LLM and return the response text."""
        llm = self._get_llm()
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return response.content if hasattr(response, "content") else str(response)

    # ------------------------------------------------------------------
    # Credentials check
    # ------------------------------------------------------------------

    def _check_credentials(self) -> None:
        sentinel = self._settings.llm_api_key.startswith("your-")
        if sentinel:
            raise RuntimeError(
                "LLMService: API key is a sentinel value. Set LLM_API_KEY in your .env before use."
            )
