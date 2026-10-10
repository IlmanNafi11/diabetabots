"""In-process APScheduler driving the midnight-WIB energy reset (task-13).

A ``cron`` job fires at 00:00 Asia/Jakarta and calls
:meth:`EnergyService.reset_all_for_date`, which pre-seeds every registered
user's ``daily_energy`` row for the new WIB date. Seeding is idempotent by
``(telegram_id, tanggal_wib)``, so:

- a tick that fires twice creates nothing the second time;
- a user's first activity of the day (``ensure_today``) remains the primary,
  lazy path — a missed tick (e.g. the process was down at midnight) can never
  lose energy data because the row is seeded on first use anyway.

The scheduler owns no session: each tick opens a fresh session from the
injected ``sessionmaker`` and closes it, so a failing tick cannot leak a
connection or corrupt the long-lived polling loop.
"""

import logging
from collections.abc import Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.services.energy_service import EnergyService

__all__ = ["JOB_ID", "build_reset_scheduler", "run_midnight_reset"]

log = logging.getLogger(__name__)

#: Stable job id so a restart replaces (rather than duplicates) the tick.
JOB_ID = "energy_midnight_reset"

#: WIB is UTC+7 year-round (no DST), so a fixed-offset cron is exact.
_WIB_TIMEZONE = "Asia/Jakarta"


async def run_midnight_reset(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> int:
    """Open a session, pre-seed today's WIB rows, and return the row count."""
    async with sessionmaker() as session:
        created = await EnergyService(session).reset_all_for_date()
    log.info("Midnight WIB reset pre-seeded %d daily_energy row(s)", created)
    return created


def build_reset_scheduler(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    job: Callable[[], Awaitable[object]] | None = None,
) -> AsyncIOScheduler:
    """Build (but do not start) an ``AsyncIOScheduler`` with the reset job.

    The caller owns the event loop: start it from the running loop with
    :meth:`~apscheduler.schedulers.asyncio.AsyncIOScheduler.start` after the
    Telegram polling loop is up, and ``shutdown(wait=False)`` on exit.
    """
    scheduler = AsyncIOScheduler(timezone=_WIB_TIMEZONE)
    tick = job or (lambda: run_midnight_reset(sessionmaker))
    scheduler.add_job(
        tick,
        trigger=CronTrigger(hour=0, minute=0, timezone=_WIB_TIMEZONE),
        id=JOB_ID,
        replace_existing=True,
        coalesce=True,  # collapse ticks missed while suspended into one
        misfire_grace_time=3600,  # still seed within the first hour after 00:00
    )
    return scheduler
