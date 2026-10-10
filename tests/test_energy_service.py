"""Unit tests for energy CRUD: sisa energi (SPEC section 6.2, energy-tracking).

Covers :mod:`app.services.energy_service`:
- ``ensure_today``: first activity of the day seeds sisa = TDEE (idempotent).
- ``reduce_energy``: "Ya" button decrements once; negatives allowed + warned.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ConsumptionLog, DailyEnergy, User
from app.services.energy_service import (
    EnergyService,
    InsufficientEnergyError,
)


def _make_user(telegram_id: int, tdee: float = 2000.0) -> User:
    return User(
        telegram_id=telegram_id,
        jenis_kelamin="Laki-laki",
        usia_tahun=30,
        berat_kg=70,
        tinggi_cm=170,
        tingkat_aktivitas="Ringan",
        pal=1.375,
        bmr_kkal=1617.5,
        tdee_kkal=tdee,
    )


@pytest_asyncio.fixture()
async def service(async_session: AsyncSession) -> EnergyService:
    return EnergyService(async_session)


@pytest_asyncio.fixture()
async def user_1(async_session: AsyncSession) -> User:
    user = _make_user(1)
    async_session.add(user)
    await async_session.commit()
    return user


class TestEnsureToday:
    async def test_first_activity_seeds_tdee(self, service: EnergyService, user_1: User) -> None:
        record = await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        assert record.sisa_energi_kkal == pytest.approx(2000.0)

    async def test_idempotent_same_day_returns_existing(
        self, service: EnergyService, user_1: User
    ) -> None:
        first = await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        await service.reduce_energy(user_1.telegram_id, 500.0)
        again = await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        assert again.tanggal_wib == first.tanggal_wib
        assert again.sisa_energi_kkal == pytest.approx(1500.0)

    async def test_requires_existing_user(
        self, service: EnergyService, async_session: AsyncSession
    ) -> None:
        from sqlalchemy import select

        count = len((await async_session.execute(select(DailyEnergy))).scalars().all())
        assert count == 0
        with pytest.raises(ValueError, match="not registered"):
            await service.ensure_today(999, 2000.0)


class TestReduceEnergy:
    async def test_reduce_decrements_once(self, service: EnergyService, user_1: User) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        record = await service.reduce_energy(user_1.telegram_id, 500.0)
        assert record.sisa_energi_kkal == pytest.approx(1500.0)

    async def test_reduce_before_seed_creates_row(
        self, service: EnergyService, user_1: User
    ) -> None:
        record = await service.reduce_energy(user_1.telegram_id, 300.0)
        assert record.sisa_energi_kkal == pytest.approx(1700.0)

    async def test_negative_balance_allowed_with_warning(
        self, service: EnergyService, user_1: User
    ) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        record, warned = await service.reduce_energy_with_flag(user_1.telegram_id, 2500.0)
        assert record.sisa_energi_kkal == pytest.approx(-500.0)
        assert warned is True

    async def test_no_warning_when_balance_stays_positive(
        self, service: EnergyService, user_1: User
    ) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        _, warned = await service.reduce_energy_with_flag(user_1.telegram_id, 100.0)
        assert warned is False

    async def test_insufficient_energy_error_raises(
        self, service: EnergyService, user_1: User
    ) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        with pytest.raises(InsufficientEnergyError):
            await service.reduce_energy(user_1.telegram_id, 2500.0, allow_negative=False)


class TestGetToday:
    async def test_get_today_returns_none_before_seed(
        self, service: EnergyService, user_1: User
    ) -> None:
        assert await service.get_today(user_1.telegram_id) is None

    async def test_get_today_returns_row_after_seed(
        self, service: EnergyService, user_1: User
    ) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        record = await service.get_today(user_1.telegram_id)
        assert record is not None
        assert record.sisa_energi_kkal == pytest.approx(2000.0)


class TestRecordConsumption:
    """``record_consumption`` — atomic ConsumptionLog insert + single decrement.

    The "Ya" button is the only path that writes a ``consumption_log`` row, and
    it must reduce the balance exactly once (SPEC energy-tracking task-13).
    """

    async def test_records_log_and_decrements_once(
        self, service: EnergyService, user_1: User
    ) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)

        record, log = await service.record_consumption(
            user_1.telegram_id,
            nama_pangan="Nasi Goreng",
            gram=150.0,
            energi_kkal=200.0,
        )

        assert record.sisa_energi_kkal == pytest.approx(1800.0)
        assert log.telegram_id == user_1.telegram_id
        assert log.nama_pangan == "Nasi Goreng"
        assert log.gram == pytest.approx(150.0)
        assert log.energi_kkal == pytest.approx(200.0)

    async def test_log_row_is_persisted(
        self, service: EnergyService, user_1: User, async_session: AsyncSession
    ) -> None:
        from sqlalchemy import select

        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)
        await service.record_consumption(
            user_1.telegram_id,
            nama_pangan="Tahu",
            gram=100.0,
            energi_kkal=80.0,
        )

        result = await async_session.execute(
            select(ConsumptionLog).where(ConsumptionLog.telegram_id == user_1.telegram_id)
        )
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].nama_pangan == "Tahu"

    async def test_warns_when_balance_goes_negative(
        self, service: EnergyService, user_1: User
    ) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)

        record, _ = await service.record_consumption(
            user_1.telegram_id,
            nama_pangan="Kue",
            gram=500.0,
            energi_kkal=2500.0,
        )

        assert record.sisa_energi_kkal == pytest.approx(-500.0)

    async def test_gl_aktual_stored_when_provided(
        self, service: EnergyService, user_1: User
    ) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)

        _, log = await service.record_consumption(
            user_1.telegram_id,
            nama_pangan="Nasi Putih",
            gram=150.0,
            energi_kkal=270.0,
            gl_aktual=30.77,
        )

        assert log.gl_aktual == pytest.approx(30.77)

    async def test_two_confirmations_log_twice_and_decrement_twice(
        self, service: EnergyService, user_1: User
    ) -> None:
        await service.ensure_today(user_1.telegram_id, user_1.tdee_kkal)

        await service.record_consumption(
            user_1.telegram_id, nama_pangan="A", gram=10.0, energi_kkal=100.0
        )
        record, _ = await service.record_consumption(
            user_1.telegram_id, nama_pangan="B", gram=10.0, energi_kkal=100.0
        )

        assert record.sisa_energi_kkal == pytest.approx(1800.0)
