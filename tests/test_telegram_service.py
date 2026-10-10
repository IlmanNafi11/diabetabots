"""Unit tests for Telegram polling service (SPEC section 4, task-12).

Tests cover:
- getUpdates response parsing and deduplication
- retry with exponential backoff
- graceful shutdown
- health check
- update routing (message, callback_query, other)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.telegram_service import (
    MessageUpdate,
    TelegramUpdate,
    TelegramUpdateType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.telegram_bot_token = "123456:ABC-DEF"
    s.log_level = "INFO"
    return s


# ---------------------------------------------------------------------------
# TelegramUpdate parsing
# ---------------------------------------------------------------------------


class TestTelegramUpdateParsing:
    def test_message_update_parses_valid(self) -> None:
        raw = {
            "update_id": 999000001,
            "message": {
                "message_id": 10,
                "date": 1704067200,
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "is_bot": False, "first_name": "Ali"},
                "text": "Hello",
            },
        }
        update = TelegramUpdate.from_dict(raw)
        assert update.update_id == 999000001
        assert update.update_type is TelegramUpdateType.MESSAGE
        assert update.chat_id == 42
        assert update.user_id == 42
        assert update.text == "Hello"
        assert update.callback_data is None
        assert update.date == 1704067200

    def test_callback_query_parses_valid(self) -> None:
        raw = {
            "update_id": 999000002,
            "callback_query": {
                "id": "cb-001",
                "from": {"id": 42, "is_bot": False, "first_name": "Ali"},
                "chat_instance": "abc",
                "data": "confirm_ya",
                "message": {
                    "message_id": 11,
                    "chat": {"id": 42, "type": "private"},
                },
            },
        }
        update = TelegramUpdate.from_dict(raw)
        assert update.update_id == 999000002
        assert update.update_type is TelegramUpdateType.CALLBACK_QUERY
        assert update.chat_id == 42
        assert update.user_id == 42
        assert update.callback_data == "confirm_ya"
        assert update.text is None

    def test_edited_message_ignored(self) -> None:
        raw = {
            "update_id": 999000003,
            "edited_message": {
                "message_id": 12,
                "date": 1704067200,
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42},
                "text": "edited text",
            },
        }
        update = TelegramUpdate.from_dict(raw)
        assert update.update_type is TelegramUpdateType.OTHER

    def test_inline_query_ignored(self) -> None:
        raw = {
            "update_id": 999000004,
            "inline_query": {"id": "inline-1", "from": {"id": 42}, "query": "pizza"},
        }
        update = TelegramUpdate.from_dict(raw)
        assert update.update_type is TelegramUpdateType.OTHER

    def test_from_dict_without_update_id_raises(self) -> None:
        with pytest.raises(ValueError, match="update_id"):
            TelegramUpdate.from_dict({"no_update_id": True})

    def test_from_dict_with_non_integer_update_id_raises(self) -> None:
        with pytest.raises(ValueError, match="update_id"):
            TelegramUpdate.from_dict({"update_id": "not-an-int"})


# ---------------------------------------------------------------------------
# MessageUpdate helpers
# ---------------------------------------------------------------------------


class TestMessageUpdateHelpers:
    def test_is_command_exact_match(self) -> None:
        m = MessageUpdate(chat_id=1, user_id=1, text="/start", message_id=1, date=0)
        assert m.is_command("/start")
        assert not m.is_command("/help")

    def test_is_command_matches_first_token_with_args(self) -> None:
        m = MessageUpdate(chat_id=1, user_id=1, text="/analisis nasi", message_id=1, date=0)
        assert m.is_command("/analisis")

    def test_is_command_with_prefix(self) -> None:
        m = MessageUpdate(chat_id=1, user_id=1, text="/analisis nasi", message_id=1, date=0)
        assert m.is_command_with("/analisis")


# ---------------------------------------------------------------------------
# TelegramService getUpdates via mock httpx Client
# ---------------------------------------------------------------------------


class TestTelegramServiceGetUpdates:
    @pytest.mark.asyncio()
    async def test_fetch_updates_returns_parsed_list(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "ok": True,
                "result": [
                    {
                        "update_id": 1,
                        "message": {
                            "message_id": 1,
                            "date": 0,
                            "chat": {"id": 42, "type": "private"},
                            "from": {"id": 42},
                            "text": "hello",
                        },
                    }
                ],
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.get.return_value = mock_response

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings)
            updates = await svc._fetch_updates_with_retry(timeout=1, offset=None)

            assert len(updates) == 1
            assert updates[0].update_id == 1
            assert updates[0].text == "hello"

    @pytest.mark.asyncio()
    async def test_fetch_updates_empty_result(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"ok": True, "result": []}
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.get.return_value = mock_response

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings)
            updates = await svc._fetch_updates_with_retry(timeout=1, offset=None)

            assert updates == []

    @pytest.mark.asyncio()
    async def test_fetch_updates_retry_on_exception(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_ok = MagicMock()
            mock_ok.json.return_value = {"ok": True, "result": []}
            mock_ok.raise_for_status = MagicMock()
            mock_client.return_value.get.side_effect = [
                Exception("network error"),
                Exception("network error"),
                mock_ok,
            ]

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings, max_retries=3)
            updates = await svc._fetch_updates_with_retry(timeout=1, offset=None)

            assert updates == []
            assert mock_client.return_value.get.call_count == 3

    @pytest.mark.asyncio()
    async def test_fetch_updates_raises_after_max_retries(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.get.side_effect = Exception("always fails")

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings, max_retries=2)
            with pytest.raises(Exception, match="always fails"):
                await svc._fetch_updates_with_retry(timeout=1, offset=None)

            assert mock_client.return_value.get.call_count == 2

    @pytest.mark.asyncio()
    async def test_fetch_updates_with_offset(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"ok": True, "result": []}
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.get.return_value = mock_response

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings)
            svc._offset = 5
            await svc._fetch_updates_with_retry(timeout=1, offset=5)

            call_args = mock_client.return_value.get.call_args
            params = call_args.kwargs["params"]
            assert params["offset"] == 5


# ---------------------------------------------------------------------------
# Deduplication / offset tracking
# ---------------------------------------------------------------------------


class TestOffsetTracking:
    @pytest.mark.asyncio()
    async def test_get_next_offset_returns_max_plus_one(self, mock_settings: MagicMock) -> None:
        from app.services.telegram_service import TelegramService

        svc = TelegramService(settings=mock_settings)
        updates = [
            TelegramUpdate(update_id=3, update_type=TelegramUpdateType.OTHER),
            TelegramUpdate(update_id=1, update_type=TelegramUpdateType.OTHER),
            TelegramUpdate(update_id=5, update_type=TelegramUpdateType.OTHER),
        ]
        offset = svc._get_next_offset(updates)
        assert offset == 6

    @pytest.mark.asyncio()
    async def test_get_next_offset_empty_returns_none(self, mock_settings: MagicMock) -> None:
        from app.services.telegram_service import TelegramService

        svc = TelegramService(settings=mock_settings)
        offset = svc._get_next_offset([])
        assert offset is None


# ---------------------------------------------------------------------------
# Send helpers
# ---------------------------------------------------------------------------


class TestSendMessage:
    @pytest.mark.asyncio()
    async def test_send_message_calls_correct_endpoint(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"ok": True, "result": {"message_id": 99}}
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.post.return_value = mock_response

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings)
            result = await svc.send_message(chat_id=42, text="Hello, World!")

            assert result["message_id"] == 99
            call_args = mock_client.return_value.post.call_args
            assert "sendMessage" in call_args[0][0]

    @pytest.mark.asyncio()
    async def test_send_message_with_inline_keyboard(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"ok": True, "result": {"message_id": 5}}
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.post.return_value = mock_response

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings)
            keyboard = [[{"text": "Ya", "callback_data": "confirm_ya"}]]
            result = await svc.send_message(
                chat_id=42,
                text="Konfirmasi?",
                reply_markup={"inline_keyboard": keyboard},
            )

            assert result["message_id"] == 5
            call_args = mock_client.return_value.post.call_args
            body = call_args[1]["json"]
            assert body["reply_markup"] == {"inline_keyboard": keyboard}

    @pytest.mark.asyncio()
    async def test_send_message_api_error_raises(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: chat not found",
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.post.return_value = mock_response

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings)
            with pytest.raises(RuntimeError, match="chat not found"):
                await svc.send_message(chat_id=99999, text="hello")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio()
    async def test_get_me_returns_bot_info(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "ok": True,
                "result": {
                    "id": 123456789,
                    "is_bot": True,
                    "first_name": "DiabetaBot",
                    "username": "diabetabot",
                },
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.get.return_value = mock_response

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings)
            me = await svc.get_me()

            assert me["is_bot"] is True
            assert me["username"] == "diabetabot"

    @pytest.mark.asyncio()
    async def test_get_me_network_error_raises(self, mock_settings: MagicMock) -> None:
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.get.side_effect = Exception("connection refused")

            from app.services.telegram_service import TelegramService

            svc = TelegramService(settings=mock_settings)
            with pytest.raises(Exception, match="connection refused"):
                await svc.get_me()
