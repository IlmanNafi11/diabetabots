"""Tests for app/services/update_limiter.py — per-chat rate limiting.

Protects the sequential analysis pipeline from a single flooding chat
(availability DoS): a chat may start one heavy analysis per window; further
messages receive a short "slow down" reply instead of queueing behind the
LLM/embedding timeouts.
"""

import pytest

from app.services.update_limiter import UpdateLimiter


@pytest.fixture()
def limiter() -> UpdateLimiter:
    clock = {"now": 100.0}
    return UpdateLimiter(window_seconds=60.0, max_calls=2, now=lambda: clock["now"])


def test_allows_requests_within_window(limiter: UpdateLimiter) -> None:
    assert limiter.allow(1) is True
    assert limiter.allow(1) is True


def test_rejects_request_over_window_cap(limiter: UpdateLimiter) -> None:
    assert limiter.allow(1) is True
    assert limiter.allow(1) is True
    assert limiter.allow(1) is False


def test_window_slides_after_time_passes(limiter: UpdateLimiter) -> None:
    assert limiter.allow(1) is True
    assert limiter.allow(1) is True
    # After the window, the chat is allowed again.
    limiter._now = lambda: 161.0  # type: ignore[attr-defined]
    assert limiter.allow(1) is True


def test_chats_are_isolated(limiter: UpdateLimiter) -> None:
    assert limiter.allow(1) is True
    assert limiter.allow(1) is True
    assert limiter.allow(2) is True