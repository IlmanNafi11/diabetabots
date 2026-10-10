"""Unit tests for the midnight-WIB reset scheduler (SPEC energy-tracking).

The reset is idempotent *by* ``tanggal_wib``; the scheduler is a belt-and-braces
pre-seed that fires at 00:00 Asia/Jakarta so a user's first activity of the day
finds a freshly seeded row. Lazy seeding (``ensure_today``) remains the primary
path, so a missed tick can never lose energy data.
"""

from datetime import date

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.models import DailyEnergy, User
from app.services.energy_service import EnergyService
from app.services.scheduler import JOB_ID, build_reset_scheduler, run_midnight_reset


def _make_user(telegram_id: int, tdee: float = 2000.0) -> User:
    return User(
        telegram_id=telegram_id,
        jenis_kelamin="Perempuan",
        usia_tahun=28,
        berat_kg=60,
        tinggi_cm=160,
        tingkat_aktivitas="Ringan",
        pal=1.375,
        bmr_kkal=1299.0,
        tdee_kkal=tdee,
    )


@pytest_asyncio.fixture()
async def users(async_session: AsyncSession) -> list[User]:
    people = [_make_user(101, 2000.0), _make_user(102, 2500.0)]
    async_session.add_all(people)
    await async_session.commit()
    return people


class TestResetAllForDate:
    async def test_seeds_every_user_on_target_date(
        self, async_session: AsyncSession, users: list[User]
    ) -> None:
        service = EnergyService(async_session)
        target = date(2026, 10, 11)

        count = await service.reset_all_for_date(wib_date=target)

        assert count == 2
        for user in users:
            record = await async_session.get(DailyEnergy, (user.telegram_id, target))
            assert record is not None
            assert record.sisa_energi_kkal == pytest.approx(user.tdee_kkal)

    async def test_does_not_overwrite_existing_balance(
        self, async_session: AsyncSession, users: list[User]
    ) -> None:
        service = EnergyService(async_session)
        target = date(2026, 10, 11)
        await service.reset_all_for_date(wib_date=target)
        record = await async_session.get(DailyEnergy, (users[0].telegram_id, target))
        record.sisa_energi_kkal = 1200.0
        await async_session.commit()

        await service.reset_all_for_date(wib_date=target)

        again = await async_session.get(DailyEnergy, (users[0].telegram_id, target))
        assert again.sisa_energi_kkal == pytest.approx(1200.0)

    async def test_defaults_to_today_wib(
        self, async_session: AsyncSession, users: list[User]
    ) -> None:
        service = EnergyService(async_session)
        await service.reset_all_for_date()
        today = await service.get_today(users[0].telegram_id)
        assert today is not None


class TestRunMidnightReset:
    async def test_tick_seeds_and_returns_count(
        self, sessionmaker: async_sessionmaker[AsyncSession], users: list[User]
    ) -> None:
        created = await run_midnight_reset(sessionmaker)
        assert created == 2


class TestSchedulerConstruction:
    def test_registers_single_midnight_job_in_jakarta(
        self, sessionmaker: async_sessionmaker[AsyncSession]
    ) -> None:
        scheduler = build_reset_scheduler(sessionmaker)

        jobs = scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == JOB_ID

        trigger = jobs[0].trigger
        assert str(trigger.timezone) == "Asia/Jakarta"
        fields = {f.name: str(f) for f in trigger.fields}
        assert fields["hour"] == "0"
        assert fields["minute"] == "0"

    def test_scheduler_is_not_running_until_started(
        self, sessionmaker: async_sessionmaker[AsyncSession]
    ) -> None:
        scheduler = build_reset_scheduler(sessionmaker)
        assert scheduler.running is False
