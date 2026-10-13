"""Unit tests for app/services/llm_service.py (task-10).

The LLM analysis service composes the 4-part prompt (system instructions
with PERKENI 2021 clinical rules, retrieved food context with pre-computed
nutrition/GL values, user profile, and the user's question) and calls the
LLM via LangChain.

Tests here cover:
- Construction: sentinel credentials raise RuntimeError.
- ``build_system_prompt()``: contains PERKENI rules, food context
  placeholder, profile placeholder.
- ``build_user_prompt()``: formats food context + user query.
- ``format_food_context()``: renders food list with computed nutrition for gram.
- ``analyze_food()``: calls LLM with correctly injected computed values;
  returns NutritionResult.

Real LLM endpoint calls are never made in unit tests. Integration with
real credentials is covered by ``tests/test_integration/test_nutrition_accuracy.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.services.llm_service import (
    LLMService,
    NoFoodRetrievedError,
    NutritionResult,
    build_system_prompt,
    build_user_prompt,
    format_food_context,
)
from app.services.rag_service import RetrievedFood

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings() -> MagicMock:
    """Minimal settings matching the real Settings fields used by LLMService."""
    s = MagicMock()
    s.llm_base_url = "https://api.example.com/v1"
    s.llm_api_key = "test-key"
    s.chat_model_name = "gpt-4o-mini"
    s.llm_temperature = 0.0
    return s


def make_retrieved_food(**overrides: Any) -> RetrievedFood:
    """Factory for test RetrievedFood instances.

    Field names match the real dataclass (record_id, not id).
    """
    defaults: dict[str, Any] = {
        "record_id": "food-0001",
        "name": "Nasi Putih, Segar",
        "category": "Nasi dan produk beras",
        "gi_value": 73.0,
        "distance": 0.12,
        "text_snippet": "Nasi Putih text snippet",
        "energi_per_100g": 130.0,
        "protein_per_100g": 2.5,
        "lemak_per_100g": 0.3,
        "karbohidrat_per_100g": 28.2,
        "serat_per_100g": 0.4,
        "gl_value": 30.0,
        "serving_size_g": 100.0,
        "bdd": 100.0,
    }
    defaults.update(overrides)
    return RetrievedFood(**defaults)


def make_user_profile(**overrides: Any) -> Any:
    """Factory for duck-typed user profile (any object with the required fields)."""
    defaults: dict[str, Any] = {
        "telegram_id": 123456789,
        "jenis_kelamin": "Laki-laki",
        "usia_tahun": 30,
        "berat_kg": 70.0,
        "tinggi_cm": 170.0,
        "tingkat_aktivitas": "Ringan",
        "tdee_kkal": 2224.06,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------


def test_module_importable() -> None:
    """llm_service module must be importable without errors."""
    from app.services import llm_service  # noqa: F401


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLLMServiceConstruction:
    """LLMService raises on sentinel credentials at construction time."""

    def test_sentinel_api_key_raises(self) -> None:
        """A sentinel 'your-*' API key must raise RuntimeError at construction."""
        sentinel = MagicMock()
        sentinel.llm_base_url = "https://api.example.com/v1"
        sentinel.llm_api_key = "your-api-key"  # sentinel
        sentinel.chat_model_name = "gpt-4o-mini"
        sentinel.llm_temperature = 0.0
        with pytest.raises(RuntimeError, match="sentinel"):
            LLMService(settings=sentinel)  # type: ignore[arg-type]

    def test_valid_credentials_do_not_raise(self, mock_settings: MagicMock) -> None:
        """Valid credentials must allow construction (LLM not called yet)."""
        with patch.object(LLMService, "_make_llm"):
            service = LLMService(settings=mock_settings)  # type: ignore[arg-type]
            assert service is not None


# ---------------------------------------------------------------------------
# build_system_prompt — pure function
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """The system prompt contains all four mandatory sections."""

    def test_contains_perkeni_rules(self) -> None:
        """System prompt must embed PERKENI 2021 clinical rules."""
        prompt = build_system_prompt()
        assert "PERKENI" in prompt
        assert "karbohidrat" in prompt.lower() and "45" in prompt

    def test_contains_food_context_placeholder(self) -> None:
        """System prompt must have a placeholder for the retrieved food documents."""
        prompt = build_system_prompt()
        assert "KONTEKS_PANGAN" in prompt

    def test_contains_profile_placeholder(self) -> None:
        """System prompt must have a placeholder for the user profile."""
        prompt = build_system_prompt()
        assert "PROFIL_PENGGUNA" in prompt

    def test_contains_response_format_reminder(self) -> None:
        """System prompt must instruct the model to follow the §3.5 response format."""
        prompt = build_system_prompt()
        assert "format" in prompt.lower()


# ---------------------------------------------------------------------------
# build_user_prompt — pure function
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """build_user_prompt combines food context and the user's question."""

    def test_includes_food_context(self) -> None:
        """The formatted food context must appear verbatim in the user prompt."""
        context = "Nasi Putih: 130 kkal/100g, GI 73"
        question = "Berapa gizi nasi putih 150 gram?"
        prompt = build_user_prompt(context, question)
        assert context in prompt

    def test_includes_user_question(self) -> None:
        """The user's question must appear verbatim in the user prompt."""
        context = "Some food context"
        question = "Berapa gizi nasi putih 150 gram?"
        prompt = build_user_prompt(context, question)
        assert question in prompt


# ---------------------------------------------------------------------------
# format_food_context — pure function
# ---------------------------------------------------------------------------


class TestFormatFoodContext:
    """format_food_context renders retrieved foods with computed nutrition values."""

    def test_includes_food_name(self) -> None:
        """Formatted context must contain the food name."""
        foods = [make_retrieved_food()]
        context = format_food_context(foods, gram=150)
        assert "Nasi Putih" in context

    def test_includes_energy_for_gram(self) -> None:
        """Formatted context must contain the scaled energy value for the portion."""
        foods = [make_retrieved_food()]
        context = format_food_context(foods, gram=150)
        # 130 kkal/100g * 150g / 100 = 195 kkal
        assert "→ 195.0 kkal untuk 150 g siap konsumsi" in context

    def test_distinguishes_per_100g_source_from_bdd_adjusted_portion(self) -> None:
        """Source operand must be BDD-labelled and portion operand must be BDD-adjusted."""
        foods = [
            make_retrieved_food(
                name="Semangka, segar",
                energi_per_100g=28.0,
                bdd=46.0,
            )
        ]

        context = format_food_context(foods, gram=200)

        expected = "Energi 28.0 kkal/100 g (mentah, BDD 46%) → 25.76 kkal untuk 200 g siap konsumsi"
        assert expected in context

    def test_omits_bdd_note_when_bdd_is_full(self) -> None:
        """BDD = 100 must not add a raw-food note to the energy operand."""
        foods = [make_retrieved_food(energi_per_100g=130.0, bdd=100.0)]

        context = format_food_context(foods, gram=100)

        assert "BDD" not in context
        assert "Energi 130.0 kkal/100 g → 130.0 kkal untuk 100 g siap konsumsi" in context

    def test_includes_gi_when_available(self) -> None:
        """GI value must appear in context when the food has it."""
        foods = [make_retrieved_food(gi_value=73.0)]
        context = format_food_context(foods, gram=150)
        assert "73" in context or "IG" in context

    def test_includes_gl_when_available(self) -> None:
        """GL value must appear in context when the food has it."""
        foods = [make_retrieved_food(gl_value=30.0, serving_size_g=100.0)]
        context = format_food_context(foods, gram=150)
        # GL = 30 * 150 / 100 = 45
        assert "45" in context

    def test_graceful_when_gi_missing(self) -> None:
        """Food without GI must still produce a readable context."""
        foods = [
            make_retrieved_food(
                record_id="food-no-gi",
                name="Telur Ayam, Segar",
                category="Telur",
                energi_per_100g=143.0,
                protein_per_100g=12.6,
                lemak_per_100g=9.5,
                karbohidrat_per_100g=0.7,
                serat_per_100g=0.0,
                gi_value=None,
                gl_value=None,
                serving_size_g=None,
                bdd=100.0,
            )
        ]
        context = format_food_context(foods, gram=100)
        assert "Telur Ayam" in context
        assert "143" in context  # energy for 100g

    def test_graceful_when_gl_missing(self) -> None:
        """Food without GL (no serving size) must still produce a readable context."""
        foods = [
            make_retrieved_food(
                record_id="food-no-gl",
                name="Kangkung, Segar",
                category="Sayuran",
                energi_per_100g=24.0,
                protein_per_100g=2.6,
                lemak_per_100g=0.4,
                karbohidrat_per_100g=4.3,
                serat_per_100g=1.6,
                gi_value=None,
                gl_value=None,
                serving_size_g=None,
                bdd=100.0,
            )
        ]
        context = format_food_context(foods, gram=100)
        assert "Kangkung" in context
        assert "24" in context  # energy for 100g

    def test_graceful_when_nutrition_metadata_missing(self) -> None:
        """Food without nutrition metadata still produces a readable context."""
        foods = [
            make_retrieved_food(
                record_id="food-no-meta",
                name="Beras giling",
                category="serealia",
                gi_value=None,
                energi_per_100g=None,
                protein_per_100g=None,
                lemak_per_100g=None,
                karbohidrat_per_100g=None,
                serat_per_100g=None,
                gl_value=None,
                serving_size_g=None,
                bdd=None,
            )
        ]
        context = format_food_context(foods, gram=100)
        assert "Beras giling" in context
        assert "tidak tersedia" in context

    def test_multiple_foods_all_present(self) -> None:
        """Multiple foods are all included in the formatted context."""
        foods = [
            make_retrieved_food(record_id="f1", name="Nasi Putih"),
            make_retrieved_food(record_id="f2", name="Nasi Merah"),
        ]
        context = format_food_context(foods, gram=100)
        assert "Nasi Putih" in context
        assert "Nasi Merah" in context


# ---------------------------------------------------------------------------
# analyze_food — service method
# ---------------------------------------------------------------------------


class TestAnalyzeFood:
    """analyze_food composes the 4-part prompt and calls the LLM."""

    @pytest.fixture()
    def service(self, mock_settings: MagicMock) -> LLMService:
        """Build an LLMService with LLM creation patched."""
        with patch.object(LLMService, "_make_llm"):
            return LLMService(settings=mock_settings)  # type: ignore[arg-type]

    @pytest.mark.asyncio()
    async def test_returns_nutrition_result(self, service: LLMService) -> None:
        """analyze_food must return a NutritionResult with a narrative."""
        foods = [make_retrieved_food()]
        with patch.object(service, "_call_llm", return_value="Mocked narrative response."):
            result = await service.analyze_food(
                food_name="Nasi Putih",
                gram=150,
                retrieved_foods=foods,
                user_profile=make_user_profile(),
                remaining_energy_kkal=1500.0,
                user_question="Berapa gizi nasi putih 150 gram?",
            )
        assert isinstance(result, NutritionResult)
        assert result.narrative == "Mocked narrative response."

    @pytest.mark.asyncio()
    async def test_calls_llm_with_system_and_user_prompts(self, service: LLMService) -> None:
        """analyze_food must invoke _call_llm with both system and user prompts."""
        foods = [make_retrieved_food()]
        with patch.object(service, "_call_llm", return_value="OK") as mock_call:
            await service.analyze_food(
                food_name="Nasi Putih",
                gram=150,
                retrieved_foods=foods,
                user_profile=make_user_profile(),
                remaining_energy_kkal=1500.0,
                user_question="Berapa gizi nasi putih 150 gram?",
            )
            mock_call.assert_called_once()
            system_prompt_arg = mock_call.call_args[0][0]
            user_prompt_arg = mock_call.call_args[0][1]
            assert "PERKENI" in system_prompt_arg
            assert "Nasi Putih" in user_prompt_arg
            assert "150 gram" in user_prompt_arg

    @pytest.mark.asyncio()
    async def test_injects_remaining_energy_in_prompt(self, service: LLMService) -> None:
        """The remaining energy value must appear in the user prompt."""
        foods = [make_retrieved_food()]
        with patch.object(service, "_call_llm", return_value="OK") as mock_call:
            await service.analyze_food(
                food_name="Nasi Putih",
                gram=150,
                retrieved_foods=foods,
                user_profile=make_user_profile(),
                remaining_energy_kkal=1500.0,
                user_question="Berapa gizi nasi putih 150 gram?",
            )
            user_prompt = mock_call.call_args[0][1]
            assert "1500" in user_prompt or "1500.0" in user_prompt

    @pytest.mark.asyncio()
    async def test_injects_tdee_in_prompt(self, service: LLMService) -> None:
        """The user's TDEE must appear in the user prompt."""
        foods = [make_retrieved_food()]
        with patch.object(service, "_call_llm", return_value="OK") as mock_call:
            await service.analyze_food(
                food_name="Nasi Putih",
                gram=150,
                retrieved_foods=foods,
                user_profile=make_user_profile(tdee_kkal=2224.06),
                remaining_energy_kkal=1500.0,
                user_question="Berapa gizi nasi putih 150 gram?",
            )
            user_prompt = mock_call.call_args[0][1]
            assert "2224" in user_prompt

    @pytest.mark.asyncio()
    async def test_raises_when_no_foods_retrieved(self, service: LLMService) -> None:
        """analyze_food must raise NoFoodRetrievedError when the list is empty."""

        with pytest.raises(NoFoodRetrievedError):
            await service.analyze_food(
                food_name="Sayur Aster",
                gram=100,
                retrieved_foods=[],
                user_profile=make_user_profile(),
                remaining_energy_kkal=2000.0,
                user_question="Gizi sayur aster?",
            )
