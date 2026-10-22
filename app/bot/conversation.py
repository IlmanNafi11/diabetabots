"""In-memory Telegram conversation state for registration and gram clarification."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from app.models.models import ActivityLevel, Gender


class ConversationState(StrEnum):
    """States in the bot conversation flow (SPEC section 5.5)."""

    IDLE = "IDLE"
    AWAIT_GENDER = "AWAIT_GENDER"
    AWAIT_AGE = "AWAIT_AGE"
    AWAIT_WEIGHT = "AWAIT_WEIGHT"
    AWAIT_HEIGHT = "AWAIT_HEIGHT"
    AWAIT_ACTIVITY = "AWAIT_ACTIVITY"
    CONFIRMED = "CONFIRMED"
    AWAIT_GRAM_CLARIFY = "AWAIT_GRAM_CLARIFY"


@dataclass(slots=True)
class RegistrationProfile:
    """Validated physiological values collected during registration."""

    gender: Gender | None = None
    age: int | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
    activity: ActivityLevel | None = None


@dataclass(slots=True)
class Conversation:
    """Mutable state for one Telegram chat."""

    state: ConversationState = ConversationState.IDLE
    profile: RegistrationProfile = field(default_factory=RegistrationProfile)
    pending_food_query: str | None = None


class ConversationManager:
    """Track independent state machines by Telegram chat ID.

    The store is bounded to keep memory flat under many one-off chatters:
    inactive chats are pruned lazily on access after ``_ttl_seconds``, and the
    dictionary is capped at ``_max_entries`` (oldest touched evicted first).
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 3600.0,
        max_entries: int = 10000,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._conversations: dict[int, Conversation] = {}
        self._last_touched: dict[int, float] = {}
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._now = now or time.monotonic

    def get(self, chat_id: int) -> Conversation:
        """Return a chat's current conversation, creating an idle one if needed."""
        self._prune()
        conversation = self._conversations.setdefault(chat_id, Conversation())
        self._last_touched[chat_id] = self._now()
        self._enforce_cap()
        return conversation

    def _enforce_cap(self) -> None:
        """Evict the least-recently-touched conversations past the cap."""
        overflow = len(self._conversations) - self._max_entries
        if overflow > 0:
            for chat_id in sorted(self._last_touched, key=self._last_touched.get)[:overflow]:
                self._conversations.pop(chat_id, None)
                self._last_touched.pop(chat_id, None)

    def _prune(self) -> None:
        """Evict expired conversations."""
        now = self._now()
        expired = [
            chat_id
            for chat_id, touched in self._last_touched.items()
            if now - touched > self._ttl_seconds
        ]
        for chat_id in expired:
            self._conversations.pop(chat_id, None)
            self._last_touched.pop(chat_id, None)


    def clear(self, chat_id: int) -> None:
        """Forget all transient state for a chat."""
        self._conversations.pop(chat_id, None)
        self._last_touched.pop(chat_id, None)

    def start_registration(self, chat_id: int) -> Conversation:
        """Start or restart registration from the gender prompt."""
        self._prune()
        conversation = Conversation(state=ConversationState.AWAIT_GENDER)
        self._conversations[chat_id] = conversation
        self._last_touched[chat_id] = self._now()
        self._enforce_cap()
        return conversation

    def accept_gender(self, chat_id: int, gender: Gender) -> Conversation:
        conversation = self._require_state(chat_id, ConversationState.AWAIT_GENDER)
        conversation.profile.gender = gender
        conversation.state = ConversationState.AWAIT_AGE
        return conversation

    def accept_age(self, chat_id: int, age: int) -> Conversation:
        conversation = self._require_state(chat_id, ConversationState.AWAIT_AGE)
        conversation.profile.age = age
        conversation.state = ConversationState.AWAIT_WEIGHT
        return conversation

    def accept_weight(self, chat_id: int, weight_kg: float) -> Conversation:
        conversation = self._require_state(chat_id, ConversationState.AWAIT_WEIGHT)
        conversation.profile.weight_kg = weight_kg
        conversation.state = ConversationState.AWAIT_HEIGHT
        return conversation

    def accept_height(self, chat_id: int, height_cm: float) -> Conversation:
        conversation = self._require_state(chat_id, ConversationState.AWAIT_HEIGHT)
        conversation.profile.height_cm = height_cm
        conversation.state = ConversationState.AWAIT_ACTIVITY
        return conversation

    def accept_activity(self, chat_id: int, activity: ActivityLevel) -> Conversation:
        conversation = self._require_state(chat_id, ConversationState.AWAIT_ACTIVITY)
        conversation.profile.activity = activity
        conversation.state = ConversationState.CONFIRMED
        return conversation

    def request_gram_clarification(self, chat_id: int, food_query: str) -> Conversation:
        """Preserve a food query while asking the user for its mass in grams."""
        conversation = self._require_state(chat_id, ConversationState.CONFIRMED)
        conversation.pending_food_query = food_query
        conversation.state = ConversationState.AWAIT_GRAM_CLARIFY
        return conversation

    def accept_gram(self, chat_id: int, gram: float) -> Conversation:
        """Complete gram clarification; the handler consumes the preserved query."""
        if gram <= 0:
            raise ValueError("gram must be positive")
        conversation = self._require_state(chat_id, ConversationState.AWAIT_GRAM_CLARIFY)
        conversation.pending_food_query = None
        conversation.state = ConversationState.CONFIRMED
        return conversation

    def _require_state(self, chat_id: int, expected: ConversationState) -> Conversation:
        conversation = self.get(chat_id)
        if conversation.state is not expected:
            raise ValueError(
                f"chat {chat_id} must be in {expected.value}, not {conversation.state.value}"
            )
        return conversation
