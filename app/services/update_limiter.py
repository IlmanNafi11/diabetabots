"""Per-chat rate limiting for the analysis pipeline (availability guard).

The bot processes Telegram updates sequentially, so a single flooding chat
can starve every other user for minutes (each analysis blocks on Ollama
embeddings up to 30s and the LLM up to 90s). ``UpdateLimiter`` is a small
sliding-window counter per chat: when a chat exceeds ``max_calls`` in
``window_seconds`` its next updates are dropped (the handler replies with a
short "slow down" message instead of starting a heavy analysis).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable


class UpdateLimiter:
    """Sliding-window per-chat limiter with O(1)-amortized bookkeeping."""
    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        max_calls: int = 2,
        max_tracked_chats: int = 10000,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_seconds = window_seconds
        self._max_calls = max_calls
        self._max_tracked_chats = max_tracked_chats
        self._now = now
        self._calls: dict[int, deque[float]] = defaultdict(deque)

    def allow(self, chat_id: int) -> bool:
        """Return ``True`` if ``chat_id`` may start another analysis now."""
        now = self._now()
        if chat_id not in self._calls:
            self._evict_stale(now)
        queue = self._calls[chat_id]
        while queue and now - queue[0] > self._window_seconds:
            queue.popleft()
        if len(queue) >= self._max_calls:
            return False
        queue.append(now)
        return True

    def _evict_stale(self, now: float) -> None:
        """Drop chats whose last call is outside the window to bound memory."""
        if len(self._calls) < self._max_tracked_chats:
            return
        stale = [
            chat_id
            for chat_id, queue in self._calls.items()
            if not queue or now - queue[-1] > self._window_seconds
        ]
        for chat_id in stale:
            del self._calls[chat_id]