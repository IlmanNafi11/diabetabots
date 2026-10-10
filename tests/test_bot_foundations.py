"""Behavioral tests for bot conversation state, validation, and responses."""

import pytest

from app.bot.conversation import ConversationManager, ConversationState
from app.bot.responses import AnalysisResponse, format_analysis
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


def test_conversation_manager_tracks_independent_chat_state() -> None:
    manager = ConversationManager()

    first = manager.start_registration(101)
    second = manager.get(202)

    assert first.state is ConversationState.AWAIT_GENDER
    assert second.state is ConversationState.IDLE


def test_conversation_manager_advances_registration_and_stores_values() -> None:
    manager = ConversationManager()
    manager.start_registration(101)

    manager.accept_gender(101, Gender.Perempuan)
    manager.accept_age(101, 35)
    manager.accept_weight(101, 60.5)
    manager.accept_height(101, 165.0)
    conversation = manager.accept_activity(101, ActivityLevel.Sedang)

    assert conversation.state is ConversationState.CONFIRMED
    assert conversation.profile.gender is Gender.Perempuan
    assert conversation.profile.age == 35
    assert conversation.profile.weight_kg == 60.5
    assert conversation.profile.height_cm == 165.0
    assert conversation.profile.activity is ActivityLevel.Sedang


def test_conversation_manager_rejects_out_of_order_transition() -> None:
    manager = ConversationManager()

    with pytest.raises(ValueError, match="AWAIT"):
        manager.accept_age(101, 35)


def test_conversation_manager_preserves_query_during_gram_clarification() -> None:
    manager = ConversationManager()
    manager.start_registration(101)
    manager.accept_gender(101, Gender.LakiLaki)
    manager.accept_age(101, 30)
    manager.accept_weight(101, 70.0)
    manager.accept_height(101, 175.0)
    manager.accept_activity(101, ActivityLevel.Ringan)

    conversation = manager.request_gram_clarification(101, "1 piring nasi putih")
    assert conversation.state is ConversationState.AWAIT_GRAM_CLARIFY
    assert conversation.pending_food_query == "1 piring nasi putih"

    completed = manager.accept_gram(101, 150.0)
    assert completed.state is ConversationState.CONFIRMED
    assert completed.pending_food_query is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("laki-laki", Gender.LakiLaki), ("PEREMPUAN", Gender.Perempuan)],
)
def test_parse_gender_accepts_supported_values(raw: str, expected: Gender) -> None:
    assert parse_gender(raw) is expected


def test_registration_numeric_validators_accept_boundaries() -> None:
    assert parse_age("18") == 18
    assert parse_age("100") == 100
    assert parse_weight("25") == 25.0
    assert parse_weight("300") == 300.0
    assert parse_height("100") == 100.0
    assert parse_height("250") == 250.0


@pytest.mark.parametrize(
    ("validator", "raw"),
    [
        (parse_age, "17"),
        (parse_age, "101"),
        (parse_age, "tiga puluh"),
        (parse_weight, "24.9"),
        (parse_weight, "301"),
        (parse_height, "99"),
        (parse_height, "251"),
    ],
)
def test_registration_numeric_validators_reject_invalid_values(validator, raw: str) -> None:
    with pytest.raises(ValidationError):
        validator(raw)


def test_parse_activity_accepts_number_and_label() -> None:
    assert parse_activity("3") is ActivityLevel.Sedang
    assert parse_activity("sangat berat") is ActivityLevel.SangatBerat


def test_parse_gram_accepts_numeric_gram_values() -> None:
    assert parse_gram("150") == 150.0
    assert parse_gram("150 gram") == 150.0
    assert parse_gram("150,5 g") == 150.5


@pytest.mark.parametrize("raw", ["1 piring", "2 butir", "1 mangkuk", "0 gram", "-10 gram"])
def test_parse_gram_rejects_non_gram_or_non_positive_portions(raw: str) -> None:
    with pytest.raises(ValidationError):
        parse_gram(raw)


def test_format_analysis_includes_available_glycemic_data() -> None:
    response = AnalysisResponse(
        nama="Nasi putih",
        kategori="Serealia",
        gram=150,
        energi=262.5,
        energi_100g=175,
        protein=4.8,
        lemak=0.45,
        karbo=60.0,
        serat=0.3,
        gi=73,
        gl=31.2,
        tdee=2100,
        sisa=1837.5,
        feasibility="Dapat dikonsumsi dengan porsi terbatas.",
    )

    text = format_analysis(response)

    assert "Indeks Glikemik: 73 (kategori: tinggi)" in text
    assert "Beban Glikemik aktual: 31.2 (kategori: tinggi)" in text
    assert "Sumber data: TKPI 2020 + Atkinson dkk. (2021)" in text


def test_format_analysis_explains_missing_glycemic_data() -> None:
    response = AnalysisResponse(
        nama="Contoh pangan",
        kategori="Lainnya",
        gram=100,
        energi=100,
        energi_100g=100,
        protein=1,
        lemak=1,
        karbo=20,
        serat=2,
        gi=None,
        gl=None,
        tdee=1800,
        sisa=1700,
        feasibility="Data glikemik belum cukup untuk penilaian lengkap.",
    )

    text = format_analysis(response)

    assert "Data IG tidak tersedia pada sumber" in text
    assert "Beban Glikemik aktual: tidak tersedia" in text
