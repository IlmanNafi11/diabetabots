"""Unit tests for Telegram bot handlers (SPEC section 5.5, task-12).

Tests the TelegramUpdateHandler which wires:
- ConversationManager (registration state machine)
- RegistrationService (DB upsert)
- RAGService (food retrieval)
- LLMService (analysis)
- EnergyService (daily energy tracking)
- TelegramService (outbound sends)

Covers:
- /start → registration flow
- registration step-by-step (gender, age, weight, height, activity)
- food query at CONFIRMED state → analysis → Ya/Tidak button
- callback_query "Ya" → energy reduction + confirmation
- callback_query "Tidak" → no-op
- re-registration at CONFIRMED state
- gram clarification flow
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from app.bot.conversation import ConversationManager, ConversationState
from app.bot.handlers import TelegramUpdateHandler
from app.models.models import ActivityLevel, Gender
from app.services.energy_service import ConfirmationStatus
from app.services.llm_service import NutritionResult
from app.services.rag_service import RetrievedFood
from app.services.telegram_service import TelegramUpdate

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockUser:
    telegram_id: int = 12345
    jenis_kelamin: Gender = Gender.LakiLaki
    usia_tahun: int = 25
    berat_kg: float = 70.0
    tinggi_cm: float = 175.0
    tingkat_aktivitas: ActivityLevel = ActivityLevel.Ringan
    pal: float = 1.56
    bmr_kkal: float = 1666.5
    tdee_kkal: float = 2600.0


def _make_message_update(
    chat_id: int = 12345,
    user_id: int = 12345,
    text: str = "/start",
    update_id: int = 1,
) -> TelegramUpdate:
    return TelegramUpdate.from_dict(
        {
            "update_id": update_id,
            "message": {
                "message_id": 10,
                "date": 1704067200,
                "chat": {"id": chat_id, "type": "private"},
                "from": {"id": user_id},
                "text": text,
            },
        }
    )


def _make_callback_update(
    chat_id: int = 12345,
    user_id: int = 12345,
    data: str = "confirm_ya",
    update_id: int = 2,
    message_id: int = 11,
) -> TelegramUpdate:
    return TelegramUpdate.from_dict(
        {
            "update_id": update_id,
            "callback_query": {
                "id": "cb-001",
                "from": {"id": user_id, "is_bot": False},
                "chat_instance": "abc",
                "data": data,
                "message": {
                    "message_id": message_id,
                    "chat": {"id": chat_id, "type": "private"},
                },
            },
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture()
def mock_telegram_svc() -> AsyncMock:
    svc = AsyncMock()
    svc.send_message = AsyncMock(return_value={"ok": True, "result": {"message_id": 100}})
    svc.send_message_with_inline_buttons = AsyncMock(
        return_value={"ok": True, "result": {"message_id": 101}}
    )
    svc.edit_message_reply_markup = AsyncMock(return_value={"ok": True, "result": {}})
    return svc


@pytest.fixture()
def mock_reg_svc(mock_session: AsyncMock) -> AsyncMock:
    svc = AsyncMock()
    svc.register_profile = AsyncMock(return_value=MockUser())
    svc.get_user = AsyncMock(return_value=MockUser())
    return svc


@pytest.fixture()
def mock_energy_svc() -> AsyncMock:
    svc = AsyncMock()
    svc.create_pending = AsyncMock(return_value=MagicMock(id=UUID(int=1)))
    svc.confirm_consumption = AsyncMock(
        return_value=(ConfirmationStatus.CONFIRMED, MagicMock(sisa_energi_kkal=2000.0), False)
    )
    svc.decline_consumption = AsyncMock(return_value=ConfirmationStatus.DECLINED)
    return svc


@pytest.fixture()
def mock_rag_svc() -> MagicMock:
    svc = MagicMock()
    svc.retrieve_async = AsyncMock(
        return_value=[
            RetrievedFood(
                record_id="doc-1",
                name="Nasi Putih",
                category="Nasi dan produk beras",
                gi_value=73,
                energi_per_100g=130.0,
                protein_per_100g=2.6,
                lemak_per_100g=0.3,
                karbohidrat_per_100g=28.0,
                serat_per_100g=0.4,
                distance=0.01,
                text_snippet="Nasi Putih: energi 130 kkal/100g",
            )
        ]
    )
    return svc


@pytest.fixture()
def mock_llm_svc() -> AsyncMock:
    svc = AsyncMock()
    svc.analyze_food = AsyncMock(
        return_value=NutritionResult(narrative="Nasi Putih memiliki GL sedang.")
    )
    return svc


@pytest.fixture()
def handler(
    mock_telegram_svc: AsyncMock,
    mock_reg_svc: AsyncMock,
    mock_energy_svc: AsyncMock,
    mock_rag_svc: MagicMock,
    mock_llm_svc: AsyncMock,
) -> TelegramUpdateHandler:
    return TelegramUpdateHandler(
        telegram_svc=mock_telegram_svc,
        conversation_manager=ConversationManager(),
        registration_svc=mock_reg_svc,
        energy_svc=mock_energy_svc,
        rag_svc=mock_rag_svc,
        llm_svc=mock_llm_svc,
    )


@pytest.fixture()
def handler_with_cap(
    mock_telegram_svc: AsyncMock,
    mock_reg_svc: AsyncMock,
    mock_energy_svc: AsyncMock,
    mock_rag_svc: MagicMock,
    mock_llm_svc: AsyncMock,
) -> TelegramUpdateHandler:
    return TelegramUpdateHandler(
        telegram_svc=mock_telegram_svc,
        conversation_manager=ConversationManager(),
        registration_svc=mock_reg_svc,
        energy_svc=mock_energy_svc,
        rag_svc=mock_rag_svc,
        llm_svc=mock_llm_svc,
        max_user_question_chars=20,
    )


# ---------------------------------------------------------------------------
# /start — registration entry
# ---------------------------------------------------------------------------


class TestStartCommand:
    @pytest.mark.asyncio()
    async def test_start_sends_registration_prompt(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        update = _make_message_update(text="/start")
        await handler.handle_update(update)

        mock_telegram_svc.send_message.assert_awaited()
        args = mock_telegram_svc.send_message.call_args
        assert args is not None
        assert "jenis kelamin" in args.kwargs["text"].lower()
        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.AWAIT_GENDER

    @pytest.mark.asyncio()
    async def test_start_idle_resets_to_await_gender(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        # Pre-seed a confirmed conversation
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)
        handler._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)

        # /start mid-registration resets
        update = _make_message_update(text="/start")
        await handler.handle_update(update)

        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.AWAIT_GENDER


# ---------------------------------------------------------------------------
# Registration flow
# ---------------------------------------------------------------------------


class TestRegistrationFlow:
    @pytest.mark.asyncio()
    async def test_accept_gender_moves_to_await_age(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        update = _make_message_update(text="Laki-laki", update_id=2)
        await handler.handle_update(update)

        mock_telegram_svc.send_message.assert_awaited()
        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.AWAIT_AGE

    @pytest.mark.asyncio()
    async def test_accept_invalid_gender_reprompts(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        update = _make_message_update(text="invalid", update_id=2)
        await handler.handle_update(update)

        # Should send error + re-prompt (2 calls)
        assert mock_telegram_svc.send_message.call_count == 2
        # Last call is the re-prompt (not the error message)
        last_text = mock_telegram_svc.send_message.call_args_list[-1].kwargs["text"].lower()
        assert "jenis kelamin" in last_text or "laki" in last_text
        assert handler._conversation_manager.get(12345).state is ConversationState.AWAIT_GENDER

    @pytest.mark.asyncio()
    async def test_accept_age_moves_to_await_weight(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        update = _make_message_update(text="25", update_id=3)
        await handler.handle_update(update)

        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.AWAIT_WEIGHT

    @pytest.mark.asyncio()
    async def test_accept_invalid_age_reprompts(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        update = _make_message_update(text="xyz", update_id=3)
        await handler.handle_update(update)

        # Should send error + re-prompt (2 calls)
        assert mock_telegram_svc.send_message.call_count == 2
        last_text = mock_telegram_svc.send_message.call_args_list[-1].kwargs["text"].lower()
        assert "usia" in last_text or "tahun" in last_text
        assert handler._conversation_manager.get(12345).state is ConversationState.AWAIT_AGE

    @pytest.mark.asyncio()
    async def test_accept_weight_moves_to_await_height(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        update = _make_message_update(text="70", update_id=4)
        await handler.handle_update(update)

        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.AWAIT_HEIGHT

    @pytest.mark.asyncio()
    async def test_accept_height_moves_to_await_activity(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        update = _make_message_update(text="175", update_id=5)
        await handler.handle_update(update)

        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.AWAIT_ACTIVITY

    @pytest.mark.asyncio()
    async def test_accept_activity_completes_registration(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_reg_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)

        update = _make_message_update(text="Ringan", update_id=6)
        await handler.handle_update(update)

        # Should call registration service
        mock_reg_svc.register_profile.assert_awaited_once()
        # Should send confirmation
        mock_telegram_svc.send_message.assert_awaited()
        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.CONFIRMED

    @pytest.mark.asyncio()
    async def test_invalid_activity_reprompts(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)
        update = _make_message_update(text="invalid_activity", update_id=6)
        await handler.handle_update(update)

        # Should send error + re-prompt (2 calls)
        assert mock_telegram_svc.send_message.call_count == 2
        last_text = mock_telegram_svc.send_message.call_args_list[-1].kwargs["text"].lower()
        assert "aktivitas" in last_text or "sedentari" in last_text
        assert handler._conversation_manager.get(12345).state is ConversationState.AWAIT_ACTIVITY


# ---------------------------------------------------------------------------
# Food query at CONFIRMED
# ---------------------------------------------------------------------------


class TestFoodQuery:
    @pytest.mark.asyncio()
    async def test_food_query_returns_analysis_with_buttons(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_llm_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
        mock_reg_svc: AsyncMock,
    ) -> None:
        # Pre-seed CONFIRMED state
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)
        handler._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)

        # Mock energy to return today's record with remaining
        mock_energy_record = MagicMock()
        mock_energy_record.sisa_energi_kkal = 2000.0
        mock_energy_svc.ensure_today = AsyncMock(return_value=mock_energy_record)
        mock_energy_svc.get_today = AsyncMock(return_value=mock_energy_record)

        update = _make_message_update(text="nasi goreng 150g", update_id=7)
        await handler.handle_update(update)

        # RAG should be queried
        mock_rag_svc.retrieve_async.assert_awaited()
        # LLM should be called
        mock_llm_svc.analyze_food.assert_awaited()
        # Should send message with inline buttons
        mock_telegram_svc.send_message_with_inline_buttons.assert_awaited()
        # State stays CONFIRMED
        assert handler._conversation_manager.get(12345).state is ConversationState.CONFIRMED

    @pytest.mark.asyncio()
    async def test_energy_persisted_applies_bdd_edible_portion(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        """Stored energy must scale by BDD (edible portion), not per-100g alone.

        Scenario #10: TKPI declares "Buah naga merah, segar" at 71.0 kkal/100g
        with BDD 66.5%, so 100 g eaten yields round(71.0 * 100 * 66.5 / 100 / 100, 2)
        = 47.22 kkal — not 71.0.
        """
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)
        handler._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)

        mock_rag_svc.retrieve_async.return_value = [
            RetrievedFood(
                record_id="doc-bdd-1",
                name="Buah naga merah, segar",
                category="Buah",
                gi_value=None,
                energi_per_100g=71.0,
                bdd=66.5,
                distance=0.02,
                text_snippet="Buah naga merah: energi 71 kkal/100g",
            )
        ]
        mock_energy_record = MagicMock()
        mock_energy_record.sisa_energi_kkal = 2000.0
        mock_energy_svc.ensure_today = AsyncMock(return_value=mock_energy_record)

        await handler.handle_update(_make_message_update(text="buah naga merah 100g", update_id=13))

        mock_energy_svc.create_pending.assert_awaited_once()
        assert mock_energy_svc.create_pending.call_args.kwargs["energi_kkal"] == pytest.approx(
            47.22
        )

    @pytest.mark.asyncio()
    async def test_gram_clarify_for_non_numeric_portion(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        # Pre-seed CONFIRMED
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)
        handler._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)

        # Food query with "1 piring" — validator can't parse grams
        update = _make_message_update(text="nasi 1 piring", update_id=8)
        await handler.handle_update(update)

        # Should send a gram clarification prompt
        mock_telegram_svc.send_message.assert_awaited()
        args_list = mock_telegram_svc.send_message.call_args_list
        last_text = args_list[-1].kwargs["text"]
        # Clarification message asks for gram
        assert "gram" in last_text.lower()
        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.AWAIT_GRAM_CLARIFY

    @pytest.mark.asyncio()
    async def test_unavailable_food_sends_not_found_without_analysis(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_llm_svc: AsyncMock,
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)
        handler._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)
        mock_rag_svc.retrieve_async.return_value = []

        await handler.handle_update(_make_message_update(text="sayur aster 100g", update_id=9))

        text = mock_telegram_svc.send_message.call_args.kwargs["text"]
        assert "Tidak ditemukan" in text
        mock_llm_svc.analyze_food.assert_not_awaited()
        mock_telegram_svc.send_message_with_inline_buttons.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_overlong_question_rejected_without_analysis(
        self,
        handler_with_cap: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_llm_svc: AsyncMock,
    ) -> None:
        handler_with_cap._conversation_manager.start_registration(12345)
        handler_with_cap._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler_with_cap._conversation_manager.accept_age(12345, 25)
        handler_with_cap._conversation_manager.accept_weight(12345, 70.0)
        handler_with_cap._conversation_manager.accept_height(12345, 175.0)
        handler_with_cap._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)

        await handler_with_cap.handle_update(_make_message_update(text="a" * 21, update_id=10))

        mock_rag_svc.retrieve_async.assert_not_awaited()
        mock_llm_svc.analyze_food.assert_not_awaited()
        mock_telegram_svc.send_message_with_inline_buttons.assert_not_awaited()
        text = mock_telegram_svc.send_message.call_args.kwargs["text"]
        assert "terlalu panjang" in text

    @pytest.mark.asyncio()
    async def test_boundary_length_question_still_analyzed(
        self,
        handler_with_cap: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_llm_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
        mock_reg_svc: AsyncMock,
    ) -> None:
        handler_with_cap._conversation_manager.start_registration(12345)
        handler_with_cap._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler_with_cap._conversation_manager.accept_age(12345, 25)
        handler_with_cap._conversation_manager.accept_weight(12345, 70.0)
        handler_with_cap._conversation_manager.accept_height(12345, 175.0)
        handler_with_cap._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)
        mock_energy_record = MagicMock()
        mock_energy_record.sisa_energi_kkal = 2000.0
        mock_energy_svc.ensure_today = AsyncMock(return_value=mock_energy_record)

        await handler_with_cap.handle_update(
            _make_message_update(text="nasi goreng 150 gram", update_id=11)
        )

        mock_rag_svc.retrieve_async.assert_awaited()
        mock_llm_svc.analyze_food.assert_awaited()

    @pytest.mark.asyncio()
    async def test_hapus_data_clears_profile_and_state(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_reg_svc: AsyncMock,
    ) -> None:
        handler._conversation_manager.start_registration(12345)
        mock_reg_svc.delete_user_data = AsyncMock(return_value=True)

        await handler.handle_update(_make_message_update(text="/hapus_data", update_id=12))

        mock_reg_svc.delete_user_data.assert_awaited_once_with(12345)
        text = mock_telegram_svc.send_message.call_args.kwargs["text"]
        assert "dihapus" in text.lower()
        conv = handler._conversation_manager.get(12345)
        assert conv.state is ConversationState.IDLE

    @pytest.mark.asyncio()
    async def test_hapus_data_unregistered_sends_guidance(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_reg_svc: AsyncMock,
    ) -> None:
        mock_reg_svc.delete_user_data = AsyncMock(return_value=False)

        await handler.handle_update(_make_message_update(text="/hapus_data", update_id=13))

        text = mock_telegram_svc.send_message.call_args.kwargs["text"]
        assert "tidak menemukan" in text.lower()

    @pytest.mark.asyncio()
    async def test_rate_limited_chat_gets_slowdown_reply(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_llm_svc: AsyncMock,
    ) -> None:
        from app.services.update_limiter import UpdateLimiter

        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)
        handler._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)
        handler._update_limiter = UpdateLimiter(window_seconds=60.0, max_calls=1)  # type: ignore[attr-defined]
        mock_energy_record = MagicMock()
        mock_energy_record.sisa_energi_kkal = 2000.0
        handler._energy_svc.ensure_today = AsyncMock(return_value=mock_energy_record)

        await handler.handle_update(_make_message_update(text="nasi goreng 150g", update_id=14))
        await handler.handle_update(_make_message_update(text="ayam goreng 100g", update_id=15))

        # First analysis runs; the second is dropped with a slow-down reply.
        assert mock_rag_svc.retrieve_async.await_count == 1
        mock_llm_svc.analyze_food.assert_awaited_once()
        texts = [c.kwargs["text"] for c in mock_telegram_svc.send_message.call_args_list]
        assert any("sebentar" in t.lower() or "lambat" in t.lower() for t in texts)


# ---------------------------------------------------------------------------
# Callback query — Ya / Tidak
# ---------------------------------------------------------------------------


class TestCallbackQueries:
    @pytest.mark.asyncio()
    async def test_callback_ya_confirms_pending_and_acknowledges(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        pending_id = UUID(int=1)
        update = _make_callback_update(data=f"confirm_ya:{pending_id}", update_id=9, message_id=11)

        await handler.handle_update(update)

        mock_telegram_svc.edit_message_reply_markup.assert_awaited_once_with(
            12345, 11, inline_keyboard=[]
        )
        mock_telegram_svc.send_message.assert_awaited()

    @pytest.mark.asyncio()
    async def test_callback_tidak_declines_pending_without_reducing(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        pending_id = UUID(int=1)
        update = _make_callback_update(
            data=f"confirm_tidak:{pending_id}", update_id=10, message_id=12
        )

        await handler.handle_update(update)

        mock_energy_svc.decline_consumption.assert_awaited_once_with(12345, pending_id)
        mock_energy_svc.confirm_consumption.assert_not_awaited()
        _, text = mock_telegram_svc.send_message.call_args.args
        mock_telegram_svc.edit_message_reply_markup.assert_awaited_once_with(
            12345, 12, inline_keyboard=[]
        )
        assert "tidak dicatat" in text.lower()

    @pytest.mark.asyncio()
    async def test_callback_already_resolved_is_ignored(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        mock_energy_svc.confirm_consumption = AsyncMock(
            return_value=(ConfirmationStatus.ALREADY_RESOLVED, None, False)
        )
        pending_id = UUID(int=1)
        update = _make_callback_update(data=f"confirm_ya:{pending_id}", update_id=11, message_id=13)

        await handler.handle_update(update)

        mock_energy_svc.confirm_consumption.assert_awaited_once_with(12345, pending_id)
        _, text = mock_telegram_svc.send_message.call_args.args
        assert "sudah" in text.lower()

    @pytest.mark.asyncio()
    async def test_callback_stale_is_rejected(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        mock_energy_svc.confirm_consumption = AsyncMock(
            return_value=(ConfirmationStatus.STALE, None, False)
        )
        pending_id = UUID(int=1)
        update = _make_callback_update(data=f"confirm_ya:{pending_id}", update_id=12)

        await handler.handle_update(update)

        _, text = mock_telegram_svc.send_message.call_args.args
        assert "tidak berlaku" in text.lower()


# ---------------------------------------------------------------------------
# Idle / unregistered
# ---------------------------------------------------------------------------


class TestIdleState:
    @pytest.mark.asyncio()
    async def test_idle_non_command_prompts_registration(
        self, handler: TelegramUpdateHandler, mock_telegram_svc: AsyncMock
    ) -> None:
        # Chat is idle (never registered)
        update = _make_message_update(text="halo", update_id=12)
        await handler.handle_update(update)

        mock_telegram_svc.send_message.assert_awaited()
        args = mock_telegram_svc.send_message.call_args
        text = args.kwargs["text"]
        assert "/start" in text or "registrasi" in text.lower() or "jenis kelamin" in text.lower()


# ---------------------------------------------------------------------------
# Handler session wiring
# ---------------------------------------------------------------------------


class TestHandlerSessionWiring:
    @pytest.mark.asyncio()
    async def test_handler_accepts_session_kwarg(
        self,
        mock_telegram_svc: AsyncMock,
        mock_reg_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_llm_svc: AsyncMock,
        mock_session: AsyncMock,
    ) -> None:
        handler = TelegramUpdateHandler(
            telegram_svc=mock_telegram_svc,
            conversation_manager=ConversationManager(),
            registration_svc=mock_reg_svc,
            energy_svc=mock_energy_svc,
            rag_svc=mock_rag_svc,
            llm_svc=mock_llm_svc,
            session=mock_session,
        )
        assert handler._session is mock_session

    @pytest.mark.asyncio()
    async def test_handler_sessionless_instantiation_works(
        self,
        mock_telegram_svc: AsyncMock,
        mock_reg_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_llm_svc: AsyncMock,
    ) -> None:
        handler = TelegramUpdateHandler(
            telegram_svc=mock_telegram_svc,
            conversation_manager=ConversationManager(),
            registration_svc=mock_reg_svc,
            energy_svc=mock_energy_svc,
            rag_svc=mock_rag_svc,
            llm_svc=mock_llm_svc,
        )
        assert handler._session is None


# ---------------------------------------------------------------------------
# Handler failure paths (user-visible error strings)
# ---------------------------------------------------------------------------


class TestHandlerFailurePaths:
    """Every user-facing failure branch must surface its exact message."""

    @staticmethod
    def _confirm(handler: TelegramUpdateHandler) -> None:
        handler._conversation_manager.start_registration(12345)
        handler._conversation_manager.accept_gender(12345, Gender.LakiLaki)
        handler._conversation_manager.accept_age(12345, 25)
        handler._conversation_manager.accept_weight(12345, 70.0)
        handler._conversation_manager.accept_height(12345, 175.0)
        handler._conversation_manager.accept_activity(12345, ActivityLevel.Ringan)

    @staticmethod
    def _sent_text(mock_telegram_svc: AsyncMock) -> str:
        call = mock_telegram_svc.send_message.call_args
        if "text" in call.kwargs:
            return call.kwargs["text"]
        return call.args[1]

    @pytest.mark.asyncio()
    async def test_analysis_failure_sends_generic_error(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        self._confirm(handler)
        mock_rag_svc.retrieve_async.side_effect = Exception("ollama down")

        await handler.handle_update(_make_message_update(text="nasi 100g", update_id=20))

        assert "Maaf, analisis gagal sementara" in self._sent_text(mock_telegram_svc)

    @pytest.mark.asyncio()
    async def test_confirm_failure_sends_save_error(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        mock_energy_svc.confirm_consumption.side_effect = SQLAlchemyError("db down")
        pending_id = UUID(int=1)
        update = _make_callback_update(data=f"confirm_ya:{pending_id}", update_id=21)

        await handler.handle_update(update)

        assert "Maaf, gagal menyimpan" in self._sent_text(mock_telegram_svc)

    @pytest.mark.asyncio()
    async def test_decline_failure_sends_process_error(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        mock_energy_svc.decline_consumption.side_effect = SQLAlchemyError("db down")
        pending_id = UUID(int=1)
        update = _make_callback_update(data=f"confirm_tidak:{pending_id}", update_id=22)

        await handler.handle_update(update)

        assert "Maaf, gagal memproses" in self._sent_text(mock_telegram_svc)

    @pytest.mark.asyncio()
    async def test_missing_profile_sends_profile_not_found(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
        mock_reg_svc: AsyncMock,
    ) -> None:
        self._confirm(handler)
        mock_reg_svc.get_user.return_value = None

        await handler.handle_update(_make_message_update(text="nasi 100g", update_id=23))

        text = mock_telegram_svc.send_message.call_args.kwargs["text"]
        assert "Profil tidak ditemukan" in text

    @pytest.mark.asyncio()
    async def test_missing_energy_sends_unavailable(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_rag_svc: MagicMock,
    ) -> None:
        self._confirm(handler)
        # Retrieved food with no energi_per_100g → cannot record consumption
        mock_rag_svc.retrieve_async.return_value = [
            RetrievedFood(
                record_id="doc-0",
                name="Makanan misterius",
                category="Lainnya",
                gi_value=None,
                energi_per_100g=None,
                distance=0.01,
                text_snippet="Makanan misterius",
            )
        ]

        await handler.handle_update(
            _make_message_update(text="makanan misterius 100g", update_id=24)
        )

        text = mock_telegram_svc.send_message.call_args.kwargs["text"]
        assert "Energi pangan tidak tersedia" in text

    @pytest.mark.asyncio()
    async def test_stale_callback_sends_not_valid_anymore(
        self,
        handler: TelegramUpdateHandler,
        mock_telegram_svc: AsyncMock,
        mock_energy_svc: AsyncMock,
    ) -> None:
        mock_energy_svc.confirm_consumption = AsyncMock(
            return_value=(ConfirmationStatus.STALE, None, False)
        )
        pending_id = UUID(int=1)
        update = _make_callback_update(data=f"confirm_ya:{pending_id}", update_id=25)

        await handler.handle_update(update)

        text = self._sent_text(mock_telegram_svc)
        assert "Analisis sudah tidak berlaku" in text
