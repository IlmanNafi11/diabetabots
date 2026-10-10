"""Unit tests for conversation state machine (SPEC section 5.5, §6.2).

Covers :class:`app.bot.conversation.ConversationManager`:
- full registration happy path IDLE → … → CONFIRMED
- gram clarification flow (AWAIT_GRAM_CLARIFY, pending query preserved)
- invalid-transition rejection
- per-chat isolation and clear()
"""

import pytest

from app.bot.conversation import (
    ConversationManager,
    ConversationState,
)
from app.models.models import ActivityLevel, Gender


@pytest.fixture()
def manager() -> ConversationManager:
    return ConversationManager()


def _complete_registration(manager: ConversationManager, chat_id: int) -> None:
    manager.start_registration(chat_id)
    manager.accept_gender(chat_id, Gender.LakiLaki)
    manager.accept_age(chat_id, 30)
    manager.accept_weight(chat_id, 70.0)
    manager.accept_height(chat_id, 170.0)
    manager.accept_activity(chat_id, ActivityLevel.Ringan)


# --- Registration happy path -------------------------------------------------


class TestRegistrationFlow:
    def test_start_registration_sets_await_gender(self, manager: ConversationManager) -> None:
        conversation = manager.start_registration(1)
        assert conversation.state is ConversationState.AWAIT_GENDER

    def test_full_flow_transitions_to_confirmed(self, manager: ConversationManager) -> None:
        _complete_registration(manager, 1)
        conversation = manager.get(1)
        assert conversation.state is ConversationState.CONFIRMED
        assert conversation.profile.gender is Gender.LakiLaki
        assert conversation.profile.age == 30
        assert conversation.profile.weight_kg == pytest.approx(70.0)
        assert conversation.profile.height_cm == pytest.approx(170.0)
        assert conversation.profile.activity is ActivityLevel.Ringan

    def test_restart_registration_resets_profile(self, manager: ConversationManager) -> None:
        _complete_registration(manager, 1)
        conversation = manager.start_registration(1)
        assert conversation.state is ConversationState.AWAIT_GENDER
        assert conversation.profile.gender is None
        assert conversation.profile.age is None

    def test_get_creates_idle_conversation_for_unknown_chat(
        self, manager: ConversationManager
    ) -> None:
        conversation = manager.get(42)
        assert conversation.state is ConversationState.IDLE
        assert conversation.profile.gender is None
        assert conversation.pending_food_query is None


# --- Invalid transitions -------------------------------------------------------


class TestInvalidTransitions:
    def test_accept_age_before_gender_raises(self, manager: ConversationManager) -> None:
        manager.start_registration(1)
        with pytest.raises(ValueError, match="AWAIT_AGE"):
            manager.accept_age(1, 30)

    def test_accept_gender_from_idle_raises(self, manager: ConversationManager) -> None:
        with pytest.raises(ValueError, match="AWAIT_GENDER"):
            manager.accept_gender(1, Gender.LakiLaki)

    def test_gram_clarify_requires_confirmed_state(self, manager: ConversationManager) -> None:
        manager.start_registration(1)
        with pytest.raises(ValueError, match="CONFIRMED"):
            manager.request_gram_clarification(1, "nasi goreng")


# --- Gram clarification ----------------------------------------------------------


class TestGramClarification:
    def test_request_preserves_query_and_moves_state(self, manager: ConversationManager) -> None:
        _complete_registration(manager, 1)
        conversation = manager.request_gram_clarification(1, "nasi goreng")
        assert conversation.state is ConversationState.AWAIT_GRAM_CLARIFY
        assert conversation.pending_food_query == "nasi goreng"

    def test_accept_gram_returns_to_confirmed_and_clears_query(
        self, manager: ConversationManager
    ) -> None:
        _complete_registration(manager, 1)
        manager.request_gram_clarification(1, "nasi goreng")
        conversation = manager.accept_gram(1, 200.0)
        assert conversation.state is ConversationState.CONFIRMED
        assert conversation.pending_food_query is None

    def test_accept_gram_rejects_non_positive(self, manager: ConversationManager) -> None:
        _complete_registration(manager, 1)
        manager.request_gram_clarification(1, "nasi goreng")
        with pytest.raises(ValueError, match="positive"):
            manager.accept_gram(1, 0.0)


# --- Chat isolation ----------------------------------------------------------------


class TestChatIsolation:
    def test_conversations_are_independent(self, manager: ConversationManager) -> None:
        manager.start_registration(1)
        manager.accept_gender(1, Gender.LakiLaki)
        manager.start_registration(2)
        assert manager.get(1).state is ConversationState.AWAIT_AGE
        assert manager.get(2).state is ConversationState.AWAIT_GENDER

    def test_clear_forgets_chat(self, manager: ConversationManager) -> None:
        _complete_registration(manager, 1)
        manager.clear(1)
        assert manager.get(1).state is ConversationState.IDLE

    def test_clear_unknown_chat_is_noop(self, manager: ConversationManager) -> None:
        manager.clear(999)  # must not raise
