"""Unit tests for DB-backed pending_analyses confirmation (SPEC energy-tracking, task-13).

The inline "Ya"/"Tidak" flow is idempotent via ``pending_analyses.resolved``
(PRD Iterasi 4 / D14) and rejects cross-day (stale) callbacks so a tap from a
previous day never mutates today's balance.
"""

from datetime import UTC, date, datetime
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ConsumptionLog, PendingAnalyses, User
from app.services.energy_service import ConfirmationStatus, EnergyService


def _make_user(telegram_id: int, tdee: float = 2000.0) -> User:
    return User(
        telegram_id=telegram_id,
        jenis_kelamin="Laki-laki",
        usia_tahun=30,
        berat_kg=70.0,
        tinggi_cm=175.0,
        tingkat_aktivitas="Ringan",
        pal=1.55,
        bmr_kkal=1666.0,
        tdee_kkal=tdee,
    )


@pytest_asyncio.fixture()
async def service(async_session: AsyncSession) -> EnergyService:
    return EnergyService(async_session)


@pytest_asyncio.fixture()
async def user_3(async_session: AsyncSession) -> User:
    user = _make_user(3)
    async_session.add(user)
    await async_session.commit()
    return user


class TestCreatePending:
    async def test_create_pending_persists_row(
        self, service: EnergyService, user_3: User, async_session: AsyncSession
    ) -> None:
        pending = await service.create_pending(
            user_3.telegram_id,
            nama_pangan="Nasi Goreng",
            gram=150.0,
            energi_kkal=200.0,
        )

        result = await async_session.execute(
            select(PendingAnalyses).where(PendingAnalyses.id == pending.id)
        )
        row = result.scalar_one()
        assert row.nama_pangan == "Nasi Goreng"
        assert row.gram == pytest.approx(150.0)
        assert row.energi_kkal == pytest.approx(200.0)
        assert row.resolved is False

    async def test_create_pending_stamps_today_wib(
        self, service: EnergyService, user_3: User
    ) -> None:
        utc_late = datetime(2026, 10, 10, 17, 30, 0, tzinfo=UTC)  # 00:30 WIB next day
        pending = await service.create_pending(
            user_3.telegram_id,
            nama_pangan="Tahu",
            gram=100.0,
            energi_kkal=80.0,
            now=utc_late,
        )
        assert pending.tanggal_wib == date(2026, 10, 11)


class TestConfirmConsumption:
    async def test_confirm_reduces_and_logs(
        self, service: EnergyService, user_3: User, async_session: AsyncSession
    ) -> None:
        await service.ensure_today(user_3.telegram_id, user_3.tdee_kkal)
        pending = await service.create_pending(
            user_3.telegram_id,
            nama_pangan="Nasi Goreng",
            gram=150.0,
            energi_kkal=200.0,
        )

        status, record, warned = await service.confirm_consumption(user_3.telegram_id, pending.id)

        assert status is ConfirmationStatus.CONFIRMED
        assert record is not None
        assert record.sisa_energi_kkal == pytest.approx(1800.0)
        assert warned is False
        logs = (
            (
                await async_session.execute(
                    select(ConsumptionLog).where(ConsumptionLog.telegram_id == user_3.telegram_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(logs) == 1

    async def test_confirm_marks_pending_resolved(
        self, service: EnergyService, user_3: User
    ) -> None:
        await service.ensure_today(user_3.telegram_id, user_3.tdee_kkal)
        pending = await service.create_pending(
            user_3.telegram_id,
            nama_pangan="Tahu",
            gram=100.0,
            energi_kkal=80.0,
        )

        await service.confirm_consumption(user_3.telegram_id, pending.id)

        refreshed = await service.get_pending(user_3.telegram_id, pending.id)
        assert refreshed is not None
        assert refreshed.resolved is True

    async def test_double_confirm_reduces_once(
        self, service: EnergyService, user_3: User, async_session: AsyncSession
    ) -> None:
        await service.ensure_today(user_3.telegram_id, user_3.tdee_kkal)
        pending = await service.create_pending(
            user_3.telegram_id,
            nama_pangan="Tahu",
            gram=100.0,
            energi_kkal=80.0,
        )

        first, record, _ = await service.confirm_consumption(user_3.telegram_id, pending.id)
        second, record2, _ = await service.confirm_consumption(user_3.telegram_id, pending.id)

        assert first is ConfirmationStatus.CONFIRMED
        assert second is ConfirmationStatus.ALREADY_RESOLVED
        assert record2 is None
        assert record is not None
        assert record.sisa_energi_kkal == pytest.approx(1920.0)
        logs = (
            (
                await async_session.execute(
                    select(ConsumptionLog).where(ConsumptionLog.telegram_id == user_3.telegram_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(logs) == 1

    async def test_confirm_unknown_pending_is_not_found(
        self, service: EnergyService, user_3: User
    ) -> None:
        import uuid

        status, record, warned = await service.confirm_consumption(user_3.telegram_id, uuid.uuid4())
        assert status is ConfirmationStatus.NOT_FOUND
        assert record is None
        assert warned is False

    async def test_confirm_stale_callback_rejected(
        self, service: EnergyService, user_3: User, async_session: AsyncSession
    ) -> None:
        yesterday = datetime(2026, 10, 10, 3, 0, 0, tzinfo=UTC)  # 10:00 WIB Oct 10
        pending = await service.create_pending(
            user_3.telegram_id,
            nama_pangan="Tahu",
            gram=100.0,
            energi_kkal=80.0,
            now=yesterday,
        )
        today = datetime(2026, 10, 11, 3, 0, 0, tzinfo=UTC)  # Oct 11 WIB

        status, record, _ = await service.confirm_consumption(
            user_3.telegram_id, pending.id, now=today
        )

        assert status is ConfirmationStatus.STALE
        assert record is None
        logs = (
            (
                await async_session.execute(
                    select(ConsumptionLog).where(ConsumptionLog.telegram_id == user_3.telegram_id)
                )
            )
            .scalars()
            .all()
        )
        assert logs == []

    async def test_confirm_warns_when_negative(self, service: EnergyService, user_3: User) -> None:
        await service.ensure_today(user_3.telegram_id, user_3.tdee_kkal)
        pending = await service.create_pending(
            user_3.telegram_id,
            nama_pangan="Kue",
            gram=500.0,
            energi_kkal=2500.0,
        )

        status, record, warned = await service.confirm_consumption(user_3.telegram_id, pending.id)

        assert status is ConfirmationStatus.CONFIRMED
        assert record is not None
        assert record.sisa_energi_kkal == pytest.approx(-500.0)
        assert warned is True


class TestDeclineConsumption:
    async def test_decline_marks_resolved_without_recording(
        self, service: EnergyService, user_3: User, async_session: AsyncSession
    ) -> None:
        await service.ensure_today(user_3.telegram_id, user_3.tdee_kkal)
        pending = await service.create_pending(
            user_3.telegram_id,
            nama_pangan="Nasi Goreng",
            gram=150.0,
            energi_kkal=200.0,
        )

        status = await service.decline_consumption(user_3.telegram_id, pending.id)

        assert status is ConfirmationStatus.DECLINED
        today = await service.get_today(user_3.telegram_id)
        assert today is not None
        assert today.sisa_energi_kkal == pytest.approx(2000.0)
        logs = (
            (
                await async_session.execute(
                    select(ConsumptionLog).where(ConsumptionLog.telegram_id == user_3.telegram_id)
                )
            )
            .scalars()
            .all()
        )
        assert logs == []
        refreshed = await service.get_pending(user_3.telegram_id, pending.id)
        assert refreshed is not None and refreshed.resolved is True

    async def test_decline_unknown_pending_is_not_found(
        self, service: EnergyService, user_3: User
    ) -> None:
        import uuid

        status = await service.decline_consumption(user_3.telegram_id, uuid.uuid4())
        assert status is ConfirmationStatus.NOT_FOUND


class TestDeleteUserData:
    """User-initiated data deletion (PDP compliance)."""

    async def test_delete_user_data_removes_all_related_rows(
        self,
        async_session: AsyncSession,
        user_3: User,
    ) -> None:
        pending_id = UUID(int=77)
        async_session.add(
            PendingAnalyses(
                id=pending_id,
                telegram_id=user_3.telegram_id,
                nama_pangan="nasi goreng",
                gram=100.0,
                energi_kkal=250.0,
                gl_aktual=20.0,
                tanggal_wib=date(2026, 10, 22),
            )
        )
        await async_session.commit()

        from app.services.registration_service import RegistrationService

        reg_svc = RegistrationService(async_session)
        deleted = await reg_svc.delete_user_data(user_3.telegram_id)

        assert deleted is True
        remaining = await async_session.execute(
            select(PendingAnalyses).where(PendingAnalyses.telegram_id == user_3.telegram_id)
        )
        assert remaining.scalar_one_or_none() is None

    async def test_delete_user_data_unknown_user_returns_false(
        self,
        async_session: AsyncSession,
    ) -> None:
        from app.services.registration_service import RegistrationService

        reg_svc = RegistrationService(async_session)
        deleted = await reg_svc.delete_user_data(999999)

        assert deleted is False
