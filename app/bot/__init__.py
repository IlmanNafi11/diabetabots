"""Bot package — Telegram conversation, validation, and response rendering."""

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

__all__ = [
    "AnalysisResponse",
    "ConversationManager",
    "ConversationState",
    "ValidationError",
    "format_analysis",
    "parse_activity",
    "parse_age",
    "parse_gender",
    "parse_gram",
    "parse_height",
    "parse_weight",
]
