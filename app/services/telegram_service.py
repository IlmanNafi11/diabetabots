"""Telegram long-polling engine (SPEC section 4, task-12).

Long-polls ``getUpdates`` using ``httpx`` (sync client, no ``python-telegram-bot``).
Retry with exponential backoff (1 s → 2 s → 4 s → 8 s → 16 s; max 5 retries).
Deduplicates by tracking ``update_id`` offsets. Shutdown on SIGTERM/SIGINT.

Public API
----------
:class:`TelegramService` — main entry point for send + poll
:func:`TelegramUpdate.from_dict` — parse a raw getUpdates dict into a typed update
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings

log = logging.getLogger(__name__)

# Exponential backoff delays in seconds (spec open question: max 5 retries)
_BACKOFF_DELAYS = (1.0, 2.0, 4.0, 8.0, 16.0)


# ---------------------------------------------------------------------------
# Update types
# ---------------------------------------------------------------------------


class TelegramUpdateType:
    """Discriminants for the kind of update received."""

    MESSAGE = "message"
    CALLBACK_QUERY = "callback_query"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Parsed update objects
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class MessageUpdate:
    """Extracted fields from a ``message`` or ``edited_message`` object."""

    chat_id: int
    user_id: int
    text: str | None
    message_id: int
    date: int  # Unix timestamp

    def is_command(self, command: str) -> bool:
        """Return True if this message is exactly ``command`` (no args)."""
        if self.text is None:
            return False
        return self.text.split()[0] == command


@dataclass(slots=True, frozen=True)
class TelegramUpdate:
    """A parsed Telegram update with all relevant fields extracted.

    One of ``message``, ``callback_query``, or neither is present.
    ``callback_data`` is extracted from the callback query for convenience.
    """

    update_id: int
    update_type: str  # TelegramUpdateType value
    chat_id: int | None = None
    user_id: int | None = None
    text: str | None = None
    callback_data: str | None = None
    callback_query_id: str | None = None
    message_id: int | None = None  # from the message inside callback_query
    date: int = 0  # Unix timestamp from the message

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TelegramUpdate:
        """Parse a raw getUpdates ``update`` dict.

        Raises
        ------
        ValueError
            When ``update_id`` is missing from the dict.
        """
        try:
            update_id = int(raw["update_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("TelegramUpdate must have an integer 'update_id'") from exc

        if "message" in raw:
            msg = raw["message"]
            return cls(
                update_id=update_id,
                update_type=TelegramUpdateType.MESSAGE,
                chat_id=_int(msg, "chat", "id"),
                user_id=_int(msg, "from", "id"),
                text=msg.get("text"),
                message_id=_int(msg, "message_id"),
                date=_int(msg, "date") or 0,
            )

        if "callback_query" in raw:
            cq = raw["callback_query"]
            return cls(
                update_id=update_id,
                update_type=TelegramUpdateType.CALLBACK_QUERY,
                chat_id=_int(cq, "message", "chat", "id"),
                user_id=_int(cq, "from", "id"),
                callback_data=cq.get("data"),
                callback_query_id=cq.get("id"),
                message_id=_int(cq, "message", "message_id"),
            )

        return cls(update_id=update_id, update_type=TelegramUpdateType.OTHER)


def _int(d: dict, *keys: str) -> int | None:
    """Navigate nested dict keys and return int or None."""
    try:
        val = d
        for k in keys:
            val = val[k]
        return int(val) if val is not None else None
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# TelegramService
# ---------------------------------------------------------------------------


class TelegramService:
    """Sync HTTP client for the Telegram Bot API (long-polling via httpx).

    The polling loop is started with :meth:`start_polling` and runs until
    :attr:`_running` is set to ``False`` (SIGTERM / SIGINT / :meth:`stop`).

    Parameters
    ----------
    settings:
        ``pydantic-settings`` application config. Falls back to ``Settings()``.
    max_retries:
        Maximum retry attempts per HTTP error before re-raising.
    """

    def __repr__(self) -> str:
        """Never leak the bot token when the service is logged or repr'd."""
        return (
            f"TelegramService(token={'<redacted>'}, running={self._running}, offset={self._offset})"
        )

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        max_retries: int = 5,
    ) -> None:
        self._settings = settings or Settings()
        self._base_url = f"https://api.telegram.org/bot{self._settings.telegram_bot_token}"
        self._max_retries = max_retries
        # Long-polling holds the connection open for up to ``timeout`` seconds
        # (default 55), so the HTTP read timeout must exceed that or every poll
        # aborts with a client-side read timeout before Telegram replies.
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=70.0, write=10.0, pool=10.0)
        )
        self._running = False
        self._offset: int | None = None

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def start_polling(
        self,
        on_update: Any,  # async callable(Update) — provided by caller (handler)
        *,
        timeout: int = 55,
    ) -> None:
        """Run the long-poll loop, calling ``on_update`` for each new update.

        Runs synchronously in a worker thread. A single event loop lives for
        the whole polling lifetime (created once, not per update) so update
        handling and getUpdates share one loop instead of churning a new one
        per message.
        """
        self._running = True
        log.info("Telegram polling started (offset=%s)", self._offset)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while self._running:
                try:
                    updates = loop.run_until_complete(self._poll_loop(timeout=timeout))
                    for update in updates:
                        self._offset = update.update_id + 1
                        try:
                            loop.run_until_complete(on_update(update))
                        except Exception:
                            log.error("on_update handler failed")
                except Exception:
                    log.error("Polling loop failed; retrying")
                    continue
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        log.info("Telegram polling stopped.")

    def stop(self) -> None:
        """Signal the polling loop to stop after the current iteration."""
        self._running = False
        log.info("Telegram polling stop requested.")

    async def _poll_loop(self, timeout: int) -> list[TelegramUpdate]:
        """Single poll iteration with retry."""
        try:
            return await self._fetch_updates_with_retry(timeout=timeout, offset=self._offset)
        except Exception:
            log.error("Poll failed after retries")
            return []

    async def _fetch_updates_with_retry(
        self,
        timeout: int,
        offset: int | None,
    ) -> list[TelegramUpdate]:
        """Fetch updates, retrying on transient HTTP errors with backoff."""
        for attempt in range(self._max_retries):
            try:
                return await self._fetch_updates(timeout=timeout, offset=offset)
            except Exception:
                delay = (
                    _BACKOFF_DELAYS[attempt]
                    if attempt < len(_BACKOFF_DELAYS)
                    else _BACKOFF_DELAYS[-1]
                )
                log.warning(
                    "getUpdates attempt %d/%d failed; retrying in %.0f s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise
        return []  # unreachable

    async def _fetch_updates(
        self,
        timeout: int,
        offset: int | None,
    ) -> list[TelegramUpdate]:
        """Call getUpdates and return parsed updates."""
        params: dict[str, int | None] = {"timeout": timeout, "offset": offset}
        response = self._client.get(
            f"{self._base_url}/getUpdates",
            params={k: v for k, v in params.items() if v is not None},
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(f"Telegram API error: {body.get('description', body)}")
        return [TelegramUpdate.from_dict(u) for u in body.get("result", [])]

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a text message to ``chat_id``.

        Parameters
        ----------
        chat_id:
            Telegram chat ID.
        text:
            Message body (max 4096 chars).
        reply_markup:
            Optional Telegram ``reply_markup`` dict, e.g. ``{"inline_keyboard": [[…]]}``.

        Returns
        -------
        dict
            The Telegram API result dict.
        """
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        response = self._client.post(
            f"{self._base_url}/sendMessage",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(
                f"Telegram API error {body.get('error_code')}: {body.get('description', body)}"
            )
        return body.get("result", {})

    async def send_message_with_inline_buttons(
        self,
        chat_id: int,
        text: str,
        *,
        inline_keyboard: list[list[dict[str, str]]],
    ) -> dict[str, Any]:
        """Convenience wrapper for ``send_message`` with inline keyboard buttons.

        Parameters
        ----------
        inline_keyboard:
            e.g. ``[[{"text": "Ya", "callback_data": "confirm_ya"},
                     {"text": "Tidak", "callback_data": "confirm_tidak"}]]``
        """
        return await self.send_message(
            chat_id,
            text,
            reply_markup={"inline_keyboard": inline_keyboard},
        )

    async def edit_message_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        *,
        inline_keyboard: list[list[dict[str, str]]],
    ) -> dict[str, Any]:
        """Replace a message's inline keyboard (pass ``[]`` to clear buttons).

        Used after a "Ya"/"Tidak" tap so the resolved analysis message no longer
        offers a second tap. Telegram rejects a no-op edit with
        ``message is not modified``; callers may ignore that specific error.
        """
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": {"inline_keyboard": inline_keyboard},
        }
        response = self._client.post(
            f"{self._base_url}/editMessageReplyMarkup",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(
                f"Telegram API error {body.get('error_code')}: {body.get('description', body)}"
            )
        return body.get("result", {})

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def get_me(self) -> dict[str, Any]:
        """Call ``getMe`` and return the bot user object."""
        response = self._client.get(f"{self._base_url}/getMe")
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(f"getMe failed: {body.get('description')}")
        return body.get("result", {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> TelegramService:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
