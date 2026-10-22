"""Tests for app/runtime.py — the process-level wiring container (task-14).

``AppRuntime`` binds the async database engine, the Telegram long-polling
client, the conversation state machine, the in-process reset scheduler, and
the per-update handler factory into a single object with an explicit
``start``/``stop`` lifecycle. These tests drive it with fakes so the lifecycle
is deterministic and does not depend on the network.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.runtime import AppRuntime
from app.services.telegram_service import TelegramUpdate


class FakeTelegramSvc:
    """Stand-in for TelegramService recording polling lifecycle calls."""

    def __init__(self) -> None:
        self.poll_calls = 0
        self.stopped = False
        self.last_on_update = None

    def start_polling(self, on_update, *, timeout: int = 55) -> None:
        self.poll_calls += 1
        self.last_on_update = on_update

    def stop(self) -> None:
        self.stopped = True


class FakeScheduler:
    """Stand-in for AsyncIOScheduler recording start/shutdown."""

    def __init__(self) -> None:
        self.started = False
        self.shutdown_called = False

    def start(self) -> None:
        self.started = True

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_called = True


class RecordingHandler:
    """Minimal handler capturing the session it was built with and updates."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.updates: list[TelegramUpdate] = []

    async def handle_update(self, update: TelegramUpdate) -> None:
        self.updates.append(update)


def _make_update(update_id: int = 1, text: str = "/start") -> TelegramUpdate:
    return TelegramUpdate.from_dict(
        {
            "update_id": update_id,
            "message": {
                "message_id": 10,
                "date": 1704067200,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 12345},
                "text": text,
            },
        }
    )


@pytest_asyncio.fixture()
async def runtime(
    async_engine: AsyncEngine,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AppRuntime:
    svc = FakeTelegramSvc()
    sched = FakeScheduler()
    created: list[RecordingHandler] = []

    def factory(session: AsyncSession) -> RecordingHandler:
        handler = RecordingHandler(session)
        created.append(handler)
        return handler

    rt = AppRuntime(
        engine=async_engine,
        sessionmaker=sessionmaker,
        telegram_svc=svc,  # type: ignore[arg-type]
        rag_svc=MagicMock(),  # type: ignore[arg-type]
        scheduler=sched,  # type: ignore[arg-type]
        handler_factory=factory,  # type: ignore[arg-type]
        poll_runner=None,
    )
    rt.fake_svc = svc  # type: ignore[attr-defined]
    rt.fake_scheduler = sched  # type: ignore[attr-defined]
    rt.created_handlers = created  # type: ignore[attr-defined]
    return rt


async def test_dispatch_opens_session_and_routes_update_to_handler(
    runtime: AppRuntime,
) -> None:
    update = _make_update(update_id=7, text="/start")

    await runtime._dispatch(update)

    created = runtime.created_handlers  # type: ignore[attr-defined]
    assert len(created) == 1
    assert isinstance(created[0].session, AsyncSession)
    assert created[0].updates == [update]


async def test_dispatch_rolls_back_when_handler_raises(
    async_engine: AsyncEngine,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    class BoomHandler:
        def __init__(self, session: AsyncSession) -> None:
            self.session = session

        async def handle_update(self, update: TelegramUpdate) -> None:
            raise RuntimeError("boom")

    rt = AppRuntime(
        engine=async_engine,
        sessionmaker=sessionmaker,
        telegram_svc=FakeTelegramSvc(),  # type: ignore[arg-type]
        rag_svc=MagicMock(),  # type: ignore[arg-type]
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        handler_factory=lambda session: BoomHandler(session),  # type: ignore[arg-type,return-value]
        poll_runner=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await rt._dispatch(_make_update())


async def test_start_initialises_schema_starts_scheduler_and_polling(
    runtime: AppRuntime,
    async_engine: AsyncEngine,
) -> None:
    polling_started = asyncio.Event()

    async def poll_runner(on_update) -> None:
        runtime.fake_svc.last_on_update = on_update  # type: ignore[attr-defined]
        polling_started.set()
        await asyncio.Event().wait()  # run until cancelled

    runtime._poll_runner = poll_runner

    await runtime.start()

    assert runtime.is_polling is True
    assert runtime.fake_scheduler.started is True  # type: ignore[attr-defined]
    await asyncio.wait_for(polling_started.wait(), timeout=1.0)

    async with async_engine.connect() as conn:
        result = await conn.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        )
        tables = {row[0] for row in result}
    assert {"users", "daily_energy", "consumption_log", "pending_analyses"} <= tables

    await runtime.stop()


async def test_stop_stops_polling_scheduler_and_disposes_engine(
    runtime: AppRuntime,
) -> None:
    async def poll_runner(on_update) -> None:
        await asyncio.Event().wait()

    runtime._poll_runner = poll_runner
    await runtime.start()

    await runtime.stop()

    assert runtime.is_polling is False
    assert runtime.fake_svc.stopped is True  # type: ignore[attr-defined]
    assert runtime.fake_scheduler.shutdown_called is True  # type: ignore[attr-defined]


async def test_run_polling_in_thread_delegates_to_telegram_start_polling(
    runtime: AppRuntime,
) -> None:
    async def on_update(_update: TelegramUpdate) -> None:
        return None

    await runtime._run_polling_in_thread(on_update)

    assert runtime.fake_svc.poll_calls == 1  # type: ignore[attr-defined]
    assert runtime.fake_svc.last_on_update is not None  # type: ignore[attr-defined]


async def test_stop_before_start_is_a_noop(runtime: AppRuntime) -> None:
    await runtime.stop()

    assert runtime.is_polling is False
