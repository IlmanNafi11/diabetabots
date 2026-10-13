"""Unit tests for WIB midnight reset idempotency (SPEC section 6.2).

The reset is idempotent *by* ``tanggal_wib`` (PRD Iterasi 4): a user's row for
a given WIB date is created at most once, and re-running the reset for the
same date never changes an existing balance.
"""

from datetime import UTC, date, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import User
from app.services.energy_service import EnergyService, wib_now, wib_today


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
async def service(async_session: AsyncSession) -> EnergyService:
    return EnergyService(async_session)


@pytest_asyncio.fixture()
async def user_2(async_session: AsyncSession) -> User:
    user = _make_user(2)
    async_session.add(user)
    await async_session.commit()
    return user


class TestWibClock:
    def test_wib_now_is_utc_plus_seven(self) -> None:
        utc_dt = datetime(2026, 10, 10, 17, 0, 0, tzinfo=UTC)
        assert wib_now(utc_dt).hour == 0  # 17:00 UTC = 00:00 WIB next day
        assert wib_now(utc_dt).date() == date(2026, 10, 11)

    def test_wib_today_follows_wib_clock(self) -> None:
        utc_dt = datetime(2026, 10, 10, 16, 59, 0, tzinfo=UTC)
        assert wib_today(utc_dt) == date(2026, 10, 10)
        utc_dt = datetime(2026, 10, 10, 17, 0, 0, tzinfo=UTC)
        assert wib_today(utc_dt) == date(2026, 10, 11)


class TestMidnightReset:
    """All clocks pinned to 2026-10-13 17:00 WIB via the service ``now=`` API.

    The service derives every WIB date from an injectable ``now``; passing it
    explicitly makes these tests independent of the host timezone. Deriving
    test dates from ``date.today()`` (host clock) previously made
    ``test_past_date_row_untouched_by_reset`` collide with today's WIB row
    whenever the host date was one day ahead of the WIB date (host = UTC+8,
    WIB = UTC+7, between 17:00 UTC and midnight WIB).
    """

    NOW = datetime(2026, 10, 13, 10, 0, 0, tzinfo=UTC)  # 17:00 WIB
    WIB_TODAY = date(2026, 10, 13)

    async def test_new_wib_date_resets_to_tdee(self, service: EnergyService, user_2: User) -> None:
        await service.ensure_today(user_2.telegram_id, user_2.tdee_kkal, now=self.NOW)
        await service.reduce_energy(user_2.telegram_id, 1800.0, now=self.NOW)

        tomorrow = self.WIB_TODAY + timedelta(days=1)
        record = await service.reset_for_date(
            user_2.telegram_id, user_2.tdee_kkal, wib_date=tomorrow
        )
        assert record.tanggal_wib == tomorrow
        assert record.sisa_energi_kkal == pytest.approx(2000.0)

    async def test_reset_is_idempotent_by_tanggal_wib(
        self, service: EnergyService, user_2: User
    ) -> None:
        tomorrow = self.WIB_TODAY + timedelta(days=1)
        first = await service.reset_for_date(
            user_2.telegram_id, user_2.tdee_kkal, wib_date=tomorrow
        )
        await service.reduce_energy(user_2.telegram_id, 250.0, now=self.NOW)
        again = await service.reset_for_date(
            user_2.telegram_id, user_2.tdee_kkal, wib_date=tomorrow
        )
        assert again.tanggal_wib == first.tanggal_wib
        assert again.sisa_energi_kkal == first.sisa_energi_kkal

    async def test_past_date_row_untouched_by_reset(
        self, service: EnergyService, user_2: User
    ) -> None:
        await service.ensure_today(user_2.telegram_id, user_2.tdee_kkal, now=self.NOW)
        await service.reduce_energy(user_2.telegram_id, 700.0, now=self.NOW)

        yesterday = self.WIB_TODAY - timedelta(days=1)
        past = await service.reset_for_date(
            user_2.telegram_id, user_2.tdee_kkal, wib_date=yesterday
        )
        # yesterday has no row yet -> created fresh at TDEE
        assert past.sisa_energi_kkal == pytest.approx(2000.0)
        # today's reduced row is untouched
        today = await service.ensure_today(user_2.telegram_id, user_2.tdee_kkal, now=self.NOW)
        assert today.sisa_energi_kkal == pytest.approx(1300.0)

    async def test_ensure_today_uses_wib_date_not_utc(
        self,
        service: EnergyService,
        user_2: User,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 17:00 UTC is already "tomorrow" in WIB
        utc_late = datetime(2026, 10, 10, 17, 30, 0, tzinfo=UTC)
        record = await service.ensure_today(user_2.telegram_id, user_2.tdee_kkal, now=utc_late)
        assert record.tanggal_wib == date(2026, 10, 11)
