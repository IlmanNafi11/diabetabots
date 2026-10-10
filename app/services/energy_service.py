"""Sisa energi (remaining daily energy) CRUD + WIB reset (SPEC energy-tracking).

Idempotency contract (PRD Iterasi 4): a ``DailyEnergy`` row exists at most
once per ``(telegram_id, tanggal_wib)``. The first activity of a WIB day
seeds ``sisa = TDEE``; re-running the same day never re-seeds.
"""

from datetime import UTC, date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ConsumptionLog, DailyEnergy, User

WIB = timezone(timedelta(hours=7), name="Asia/Jakarta")


class EnergyServiceError(Exception):
    """Base class for energy service failures."""


class UserNotRegisteredError(EnergyServiceError, ValueError):
    """Raised when the telegram_id has no ``users`` row."""


class InsufficientEnergyError(EnergyServiceError):
    """Raised when a reduction is rejected because it would overdraw."""


def wib_now(now: datetime | None = None) -> datetime:
    """Return ``now`` expressed in Asia/Jakarta (UTC+7)."""
    return (now or datetime.now(UTC)).astimezone(WIB)


def wib_today(now: datetime | None = None) -> date:
    """Return the current calendar date in WIB."""
    return wib_now(now).date()


class EnergyService:
    """Async CRUD over ``DailyEnergy`` with WIB-midnight idempotent reset."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ensure_today(
        self,
        telegram_id: int,
        tdee_kkal: float,
        *,
        now: datetime | None = None,
    ) -> DailyEnergy:
        """Return today's (WIB) row, seeding ``sisa = TDEE`` only if absent."""
        return await self._ensure_for_date(telegram_id, tdee_kkal, wib_today(now))

    async def reduce_energy(
        self,
        telegram_id: int,
        amount_kkal: float,
        *,
        allow_negative: bool = True,
        now: datetime | None = None,
    ) -> DailyEnergy:
        """Decrement today's balance once (inline button "Ya").

        With ``allow_negative=False`` (the "strict" mode used by validation
        paths), raises :class:`InsufficientEnergyError` instead of overdrawing.
        """
        record, _ = await self.reduce_energy_with_flag(
            telegram_id, amount_kkal, allow_negative=allow_negative, now=now
        )
        return record

    async def reduce_energy_with_flag(
        self,
        telegram_id: int,
        amount_kkal: float,
        *,
        allow_negative: bool = True,
        now: datetime | None = None,
    ) -> tuple[DailyEnergy, bool]:
        """Decrement and report whether the balance went negative.

        Returns ``(record, warned)``; ``warned`` is ``True`` only when the
        resulting balance is below zero (PRD: negatives allowed with warning).
        """
        user = await self._require_user(telegram_id)
        record = await self._ensure_for_date(telegram_id, user.tdee_kkal, wib_today(now))
        new_balance = round(record.sisa_energi_kkal - amount_kkal, 2)
        if new_balance < 0 and not allow_negative:
            raise InsufficientEnergyError(
                f"reduce of {amount_kkal} kkal would leave {new_balance} kkal "
                f"for telegram_id {telegram_id}"
            )
        record.sisa_energi_kkal = new_balance
        await self._session.commit()
        return record, new_balance < 0

    async def get_today(
        self, telegram_id: int, *, now: datetime | None = None
    ) -> DailyEnergy | None:
        """Return today's WIB row, or ``None`` if the user has none yet."""
        result = await self._session.execute(
            select(DailyEnergy).where(
                DailyEnergy.telegram_id == telegram_id,
                DailyEnergy.tanggal_wib == wib_today(now),
            )
        )
        return result.scalar_one_or_none()

    async def reset_for_date(
        self,
        telegram_id: int,
        tdee_kkal: float,
        *,
        wib_date: date,
    ) -> DailyEnergy:
        """Idempotently (re)seed a given WIB date's row at ``tdee_kkal``.

        Never modifies an existing balance — idempotent by ``tanggal_wib``.
        """
        return await self._ensure_for_date(telegram_id, tdee_kkal, wib_date)

    async def _ensure_for_date(
        self, telegram_id: int, tdee_kkal: float, wib_date: date
    ) -> DailyEnergy:
        result = await self._session.execute(
            select(DailyEnergy).where(
                DailyEnergy.telegram_id == telegram_id,
                DailyEnergy.tanggal_wib == wib_date,
            )
        )
        record = result.scalar_one_or_none()
        if record is not None:
            return record

        await self._require_user(telegram_id)
        record = DailyEnergy(
            telegram_id=telegram_id, tanggal_wib=wib_date, sisa_energi_kkal=tdee_kkal
        )
        self._session.add(record)
        await self._session.commit()
        return record

    async def _require_user(self, telegram_id: int) -> User:
        result = await self._session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise UserNotRegisteredError(f"telegram_id {telegram_id} is not registered")
        return user

    async def record_consumption(
        self,
        telegram_id: int,
        *,
        nama_pangan: str,
        gram: float,
        energi_kkal: float,
        gl_aktual: float | None = None,
        analysis_message_id: int | None = None,
        now: datetime | None = None,
    ) -> tuple[DailyEnergy, ConsumptionLog]:
        """Record a confirmed consumption ("Ya") and decrement the balance once.

        Writes a :class:`ConsumptionLog` row and applies the single balance
        reduction in one commit, so a confirmed consumption is never logged
        without its energy being subtracted (SPEC energy-tracking task-13).
        Callers gate this behind ``pending_analyses.resolved`` so a double tap
        cannot record twice.
        """
        user = await self._require_user(telegram_id)
        record = await self._ensure_for_date(telegram_id, user.tdee_kkal, wib_today(now))

        record.sisa_energi_kkal = round(record.sisa_energi_kkal - energi_kkal, 2)
        log = ConsumptionLog(
            telegram_id=telegram_id,
            nama_pangan=nama_pangan,
            gram=round(gram, 2),
            energi_kkal=round(energi_kkal, 2),
            gl_aktual=gl_aktual,
            analysis_message_id=analysis_message_id,
        )
        self._session.add(log)
        await self._session.commit()
        return record, log
