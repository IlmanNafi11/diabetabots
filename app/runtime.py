"""Process-level wiring for DiabetaBot (task-14).

``AppRuntime`` is the single place where the long-lived collaborators are
constructed and torn down:

- the async SQLAlchemy engine + session factory (:mod:`app.database`);
- the Telegram long-polling client (:mod:`app.services.telegram_service`);
- the in-process midnight-WIB reset scheduler (:mod:`app.services.scheduler`);
- a per-update handler built from a fresh DB session
  (:class:`app.bot.handlers.TelegramUpdateHandler`).

It is started from the FastAPI lifespan in :mod:`app.main`, so a single
``uvicorn app.main:app`` process serves both the ``/health`` endpoint and the
Telegram polling loop (PRD §8.1, D12). The runtime is injected so the lifespan
can be exercised with a fake in tests; the default construction path lives in
:func:`build_runtime`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.bot.conversation import ConversationManager
from app.bot.handlers import TelegramUpdateHandler
from app.config import Settings
from app.database import create_async_engine, create_sessionmaker, init_db
from app.services.energy_service import EnergyService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.registration_service import RegistrationService
from app.services.scheduler import build_reset_scheduler
from app.services.telegram_service import TelegramService, TelegramUpdate

log = logging.getLogger(__name__)

#: An async callable that runs the blocking poll loop until cancelled.
PollRunner = Callable[[Callable[[TelegramUpdate], Awaitable[None]]], Awaitable[None]]

#: Builds a handler bound to a per-update session.
HandlerFactory = Callable[[AsyncSession], TelegramUpdateHandler]


class AppRuntime:
    """Owns the process-level collaborators and their start/stop lifecycle."""

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        sessionmaker: async_sessionmaker[AsyncSession],
        telegram_svc: TelegramService,
        scheduler: Any,
        handler_factory: HandlerFactory,
        poll_runner: PollRunner | None = None,
    ) -> None:
        self._engine = engine
        self._sessionmaker = sessionmaker
        self._telegram_svc = telegram_svc
        self._scheduler = scheduler
        self._handler_factory = handler_factory
        self._poll_runner: PollRunner | None = poll_runner
        self._poll_task: asyncio.Task[None] | None = None
        self._stopped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_polling(self) -> bool:
        """True while the polling task is alive."""
        return self._poll_task is not None and not self._poll_task.done()

    async def start(self) -> None:
        """Create tables, start the reset scheduler, and begin polling."""
        await init_db(self._engine)
        self._scheduler.start()
        log.info("Midnight-WIB reset scheduler started")
        runner = self._poll_runner or self._run_polling_in_thread
        self._poll_task = asyncio.create_task(runner(self._dispatch))
        log.info("Telegram polling task started")

    async def stop(self) -> None:
        """Stop polling, shut the scheduler down, and dispose the engine."""
        if self._stopped:
            return
        self._stopped = True

        self._telegram_svc.stop()
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

        self._scheduler.shutdown(wait=False)
        await self._engine.dispose()
        log.info("AppRuntime stopped")

    # ------------------------------------------------------------------
    # Update dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, update: TelegramUpdate) -> None:
        """Handle one update in its own session; commit or roll back."""
        async with self._sessionmaker() as session:
            handler = self._handler_factory(session)
            try:
                await handler.handle_update(update)
                await session.commit()
            except Exception:
                await session.rollback()
                log.exception("Failed to handle update %s", update.update_id)
                raise

    # ------------------------------------------------------------------
    # Default poll runner (blocking httpx client moved off the loop)
    # ------------------------------------------------------------------

    async def _run_polling_in_thread(
        self,
        on_update: Callable[[TelegramUpdate], Awaitable[None]],
    ) -> None:
        """Run the blocking ``TelegramService.start_polling`` in a worker thread.

        ``on_update`` is called from the worker thread; it schedules the
        coroutine on the main event loop via :func:`asyncio.run_coroutine_threadsafe`.
        """
        loop = asyncio.get_running_loop()

        async def schedule(update: TelegramUpdate) -> None:
            future = asyncio.run_coroutine_threadsafe(on_update(update), loop)
            await asyncio.wrap_future(future)

        await asyncio.to_thread(self._telegram_svc.start_polling, schedule)


def build_runtime(settings: Settings | None = None) -> AppRuntime:
    """Construct the production :class:`AppRuntime` from settings.

    Raises
    ------
    RuntimeError
        When required credentials are still sentinel ``"your-*"`` values
        (propagated from :class:`~app.services.rag_service.RAGService`).
    """
    settings = settings or Settings()
    engine = create_async_engine(settings)
    sessionmaker = create_sessionmaker(engine)
    telegram_svc = TelegramService(settings)
    rag_svc = RAGService(settings)
    llm_svc = LLMService(settings)
    conversation = ConversationManager()
    scheduler = build_reset_scheduler(sessionmaker)

    def handler_factory(session: AsyncSession) -> TelegramUpdateHandler:
        return TelegramUpdateHandler(
            telegram_svc=telegram_svc,
            conversation_manager=conversation,
            registration_svc=RegistrationService(session),
            energy_svc=EnergyService(session),
            rag_svc=rag_svc,
            llm_svc=llm_svc,
            session=session,
        )

    return AppRuntime(
        engine=engine,
        sessionmaker=sessionmaker,
        telegram_svc=telegram_svc,
        scheduler=scheduler,
        handler_factory=handler_factory,
    )
