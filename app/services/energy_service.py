"""Sisa energi (remaining daily energy) CRUD + WIB reset (SPEC energy-tracking).

Idempotency contract (PRD Iterasi 4): a ``DailyEnergy`` row exists at most
once per ``(telegram_id, tanggal_wib)``. The first activity of a WIB day
seeds ``sisa = TDEE``; re-running the same day never re-seeds.
"""

from datetime import UTC, date, datetime, timedelta, timezone
from enum import StrEnum
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ConsumptionLog, DailyEnergy, PendingAnalyses, User


class ConfirmationStatus(StrEnum):
    """Outcome of an inline "Ya"/"Tidak" confirmation callback.

    ``STALE`` is a callback from a previous WIB day (PRD §8.1: cross-day
    callbacks are rejected); ``ALREADY_RESOLVED`` covers a double tap.
    """

    CONFIRMED = "confirmed"
    DECLINED = "declined"
    ALREADY_RESOLVED = "already_resolved"
    STALE = "stale"
    NOT_FOUND = "not_found"


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

    async def reset_all_for_date(self, *, wib_date: date | None = None) -> int:
        """Pre-seed a WIB date's row for every registered user (scheduler tick).

        Idempotent by ``(telegram_id, tanggal_wib)``: users who already have a
        row for ``wib_date`` are left untouched, so a tick that fires twice (or
        a lazy ``ensure_today`` that beat the scheduler) never re-seeds a
        balance. Returns the number of rows created.
        """
        target = wib_date or wib_today()
        result = await self._session.execute(select(User))
        users = result.scalars().all()

        created = 0
        for user in users:
            existing = await self._session.execute(
                select(DailyEnergy).where(
                    DailyEnergy.telegram_id == user.telegram_id,
                    DailyEnergy.tanggal_wib == target,
                )
            )
            if existing.scalar_one_or_none() is not None:
                continue
            self._session.add(
                DailyEnergy(
                    telegram_id=user.telegram_id,
                    tanggal_wib=target,
                    sisa_energi_kkal=user.tdee_kkal,
                )
            )
            created += 1

        if created:
            await self._session.commit()
        return created

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
        commit: bool = True,
    ) -> tuple[DailyEnergy, ConsumptionLog]:
        """Record a confirmed consumption ("Ya") and decrement the balance once.

        Writes a :class:`ConsumptionLog` row and applies the single balance
        reduction; by default commits in one call. Pass ``commit=False`` when
        the caller bundles the write together with the pending claim so both
        land in a single transaction.
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
        if commit:
            await self._session.commit()
        return record, log

    # ------------------------------------------------------------------
    # Inline "Ya"/"Tidak" confirmation (task-13)
    # ------------------------------------------------------------------

    async def create_pending(
        self,
        telegram_id: int,
        *,
        nama_pangan: str,
        gram: float,
        energi_kkal: float,
        gl_aktual: float | None = None,
        now: datetime | None = None,
    ) -> PendingAnalyses:
        """Persist the verified payload behind an analysis message's buttons.

        The row's ``id`` is embedded in the callback data so a tap always
        resolves against the numbers that were shown, never a re-derivation
        (PRD §8.1: "nilai terverifikasi"). ``tanggal_wib`` stamps the day the
        analysis was produced so cross-day callbacks can be rejected.
        """
        await self._require_user(telegram_id)
        pending = PendingAnalyses(
            telegram_id=telegram_id,
            nama_pangan=nama_pangan,
            gram=round(gram, 2),
            energi_kkal=round(energi_kkal, 2),
            gl_aktual=gl_aktual,
            tanggal_wib=wib_today(now),
            resolved=False,
        )
        self._session.add(pending)
        await self._session.commit()
        return pending

    async def get_pending(self, telegram_id: int, pending_id: UUID) -> PendingAnalyses | None:
        """Return a pending analysis row for ``telegram_id``, or ``None``."""
        result = await self._session.execute(
            select(PendingAnalyses).where(
                PendingAnalyses.id == pending_id,
                PendingAnalyses.telegram_id == telegram_id,
            )
        )
        return result.scalar_one_or_none()

    async def confirm_consumption(
        self,
        telegram_id: int,
        pending_id: UUID,
        *,
        now: datetime | None = None,
    ) -> tuple[ConfirmationStatus, DailyEnergy | None, bool]:
        """Resolve a "Ya" tap: log the consumption and reduce the balance once.

        Returns ``(status, record, warned)``. ``record`` is ``None`` unless the
        status is ``CONFIRMED``. Cross-day callbacks return ``STALE`` without
        touching the balance.

        Race safety: the pending row is claimed with a conditional
        ``UPDATE ... WHERE resolved = false`` (compare-and-swap) *before* the
        balance is reduced, so two taps racing through the same await points
        can never both pass the resolved check and double-reduce.
        """
        pending = await self.get_pending(telegram_id, pending_id)
        if pending is None:
            return ConfirmationStatus.NOT_FOUND, None, False
        if pending.tanggal_wib != wib_today(now):
            return ConfirmationStatus.STALE, None, False
        if not await self._claim_pending(telegram_id, pending_id):
            return ConfirmationStatus.ALREADY_RESOLVED, None, False

        record, _log = await self.record_consumption(
            telegram_id,
            nama_pangan=pending.nama_pangan,
            gram=pending.gram,
            energi_kkal=pending.energi_kkal,
            gl_aktual=pending.gl_aktual,
            now=now,
            commit=False,
        )
        await self._session.commit()
        return ConfirmationStatus.CONFIRMED, record, record.sisa_energi_kkal < 0

    async def decline_consumption(
        self,
        telegram_id: int,
        pending_id: UUID,
        *,
        now: datetime | None = None,
    ) -> ConfirmationStatus:
        """Resolve a "Tidak" tap: mark resolved without any balance change."""
        pending = await self.get_pending(telegram_id, pending_id)
        if pending is None:
            return ConfirmationStatus.NOT_FOUND
        if pending.tanggal_wib != wib_today(now):
            return ConfirmationStatus.STALE
        if not await self._claim_pending(telegram_id, pending_id):
            return ConfirmationStatus.ALREADY_RESOLVED
        await self._session.commit()
        return ConfirmationStatus.DECLINED

    async def _claim_pending(self, telegram_id: int, pending_id: UUID) -> bool:
        """Atomically claim a pending row if it is still unresolved.

        Runs a conditional ``UPDATE ... SET resolved = true WHERE resolved =
        false`` inside the current transaction; only the first caller gets
        ``rowcount == 1``, every racing caller sees zero rows updated.
        """
        result = await self._session.execute(
            update(PendingAnalyses)
            .where(
                PendingAnalyses.id == pending_id,
                PendingAnalyses.telegram_id == telegram_id,
                PendingAnalyses.resolved.is_(False),
            )
            .values(resolved=True)
        )
        return result.rowcount == 1
