"""Registration persistence: conversation state → BMR/TDEE → User row upsert.

This service is the bridge between the conversation state machine
(``app.bot.conversation``) and the database layer (``app.models.models.User``).
It is called by the Telegram handler when the registration flow reaches
``ConversationState.CONFIRMED``.

The upsert is idempotent: calling ``register_profile`` for an existing
``telegram_id`` updates the row rather than creating a duplicate.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import ConsumptionLog, DailyEnergy, PendingAnalyses, User
from app.services.tdee_service import calculate_bmr, calculate_tdee, pal_for_activity


class RegistrationService:
    """Async service for persisting and updating user profiles."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def register_profile(
        self,
        telegram_id: int,
        gender,
        age: int,
        weight_kg: float,
        height_cm: float,
        activity,
    ) -> User:
        """Upsert a User row with validated profile data and derived BMR/TDEE.

        Parameters
        ----------
        telegram_id:
            Telegram chat ID used as the primary key.
        gender:
            ``Gender`` enum member.
        age:
            Age in whole years.
        weight_kg:
            Body weight in kilograms.
        height_cm:
            Height in centimetres.
        activity:
            ``ActivityLevel`` enum member.

        Returns
        -------
        User
            The upserted (or newly created) user row.
        """
        pal = pal_for_activity(activity)
        bmr = calculate_bmr(weight_kg, height_cm, age, gender)
        tdee = calculate_tdee(bmr, pal)

        # Attempt to find an existing row
        result = await self._session.execute(select(User).where(User.telegram_id == telegram_id))
        existing: User | None = result.scalar_one_or_none()

        if existing is not None:
            existing.jenis_kelamin = gender
            existing.usia_tahun = age
            existing.berat_kg = weight_kg
            existing.tinggi_cm = height_cm
            existing.tingkat_aktivitas = activity
            existing.pal = pal
            existing.bmr_kkal = bmr
            existing.tdee_kkal = tdee
            user = existing
        else:
            user = User(
                telegram_id=telegram_id,
                jenis_kelamin=gender,
                usia_tahun=age,
                berat_kg=weight_kg,
                tinggi_cm=height_cm,
                tingkat_aktivitas=activity,
                pal=pal,
                bmr_kkal=bmr,
                tdee_kkal=tdee,
            )
            self._session.add(user)

        await self._session.flush()
        return user

    async def get_user(self, telegram_id: int) -> User | None:
        """Return the User row for ``telegram_id``, or ``None`` if not registered."""
        result = await self._session.execute(select(User).where(User.telegram_id == telegram_id))
        return result.scalar_one_or_none()

    async def delete_user_data(self, telegram_id: int) -> bool:
        """Delete every row owned by ``telegram_id`` (PDP right-to-erasure).

        Removes the user's profile and all derived records (daily energy,
        consumption log, pending analyses) in dependency order. Returns
        ``False`` when the user does not exist; the caller commits the
        transaction.
        """
        user = await self.get_user(telegram_id)
        if user is None:
            return False

        await self._session.execute(
            delete(PendingAnalyses).where(PendingAnalyses.telegram_id == telegram_id)
        )
        await self._session.execute(
            delete(ConsumptionLog).where(ConsumptionLog.telegram_id == telegram_id)
        )
        await self._session.execute(
            delete(DailyEnergy).where(DailyEnergy.telegram_id == telegram_id)
        )
        await self._session.execute(delete(User).where(User.telegram_id == telegram_id))
        await self._session.flush()
        return True
