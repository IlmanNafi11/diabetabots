"""Integration test: 8 PRD simulation profiles — TDEE ≤0.5% from reference (PRD §1.3 AC3, §4).

End-to-end pipeline: conversation state → registration profile → BMR/TDEE
calculation → User row upsert → TDEE accuracy verified.

Reference TDEE values are calculated from the Mifflin-St Jeor formula (BMR =
10·BB + 6.25·TB - 5·Usia + 5 [L] / -161 [P]; TDEE = round(BMR x PAL, 2))
and match the values documented in PRD §4.

This test requires a real (or test) PostgreSQL database; it uses the same
async fixtures as the rest of the integration suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.conversation import ConversationManager, ConversationState
from app.models.models import ActivityLevel, Gender, User
from app.services.registration_service import RegistrationService
from app.services.tdee_service import calculate_bmr, calculate_tdee, pal_for_activity

# ---------------------------------------------------------------------------
# 8 PRD simulation profiles (PRD §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Profile:
    label: str
    gender: Gender
    age: int
    weight_kg: float
    height_cm: float
    activity: ActivityLevel
    reference_tdee: float


PROFILES: list[Profile] = [
    # P1
    Profile("P1", Gender.LakiLaki, 30, 70.0, 170.0, ActivityLevel.Ringan, 2224.06),
    # P2
    Profile("P2", Gender.Perempuan, 28, 60.0, 160.0, ActivityLevel.Ringan, 1786.13),
    # P3
    Profile("P3", Gender.LakiLaki, 65, 95.0, 170.0, ActivityLevel.Sedentari, 2031.0),
    # P4
    Profile("P4", Gender.Perempuan, 63, 85.0, 155.0, ActivityLevel.Sedentari, 1611.30),
    # P5
    Profile("P5", Gender.LakiLaki, 32, 100.0, 175.0, ActivityLevel.Sedentari, 2326.50),
    # P6
    Profile("P6", Gender.Perempuan, 30, 90.0, 158.0, ActivityLevel.Sedentari, 1891.80),
    # P7
    Profile("P7", Gender.LakiLaki, 60, 68.0, 168.0, ActivityLevel.Sedang, 2224.25),
    # P8
    Profile("P8", Gender.Perempuan, 58, 58.0, 152.0, ActivityLevel.Sedang, 1672.45),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def registration_service(async_session: AsyncSession) -> RegistrationService:
    return RegistrationService(async_session)


@pytest_asyncio.fixture()
def conversation_manager() -> ConversationManager:
    return ConversationManager()


# ---------------------------------------------------------------------------
# Accuracy check helper
# ---------------------------------------------------------------------------

TDEE_TOLERANCE = 0.005  # 0.5% relative tolerance


def tdee_error_pct(computed: float, reference: float) -> float:
    """Return absolute percentage error of computed vs reference."""
    return abs(computed - reference) / reference * 100


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
class TestTdee8Profiles:
    """Verify TDEE accuracy for all 8 PRD simulation profiles."""

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.label)
    async def test_tdee_within_0_5_percent(
        self,
        async_session: AsyncSession,
        registration_service: RegistrationService,
        profile: Profile,
    ) -> None:
        """TDEE must be ≤0.5% different from the PRD reference value."""
        # Deterministic math check (proves service functions are correct)
        pal = pal_for_activity(profile.activity)
        bmr = calculate_bmr(profile.weight_kg, profile.height_cm, profile.age, profile.gender)
        tdee = calculate_tdee(bmr, pal)

        error = tdee_error_pct(tdee, profile.reference_tdee)
        assert error <= 0.5, (
            f"[{profile.label}] computed TDEE={tdee} vs ref={profile.reference_tdee} "
            f"→ error={error:.4f}% (limit 0.5%)"
        )

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.label)
    async def test_registration_persists_user_row(
        self,
        async_session: AsyncSession,
        registration_service: RegistrationService,
        profile: Profile,
    ) -> None:
        """Registering a profile must create a User row with correct fields."""
        telegram_id = int(profile.label[1:])  # P1 → 1, P2 → 2, …

        # Run the registration pipeline
        await registration_service.register_profile(
            telegram_id=telegram_id,
            gender=profile.gender,
            age=profile.age,
            weight_kg=profile.weight_kg,
            height_cm=profile.height_cm,
            activity=profile.activity,
        )

        # Verify the row exists and fields match
        result = await async_session.execute(select(User).where(User.telegram_id == telegram_id))
        user: User | None = result.scalar_one_or_none()
        assert user is not None, (
            f"[{profile.label}] User row not found for telegram_id={telegram_id}"
        )

        assert user.jenis_kelamin == profile.gender
        assert user.usia_tahun == profile.age
        assert user.berat_kg == pytest.approx(profile.weight_kg)
        assert user.tinggi_cm == pytest.approx(profile.height_cm)
        assert user.tingkat_aktivitas == profile.activity
        assert user.pal == pytest.approx(pal_for_activity(profile.activity))

        expected_bmr = calculate_bmr(
            profile.weight_kg, profile.height_cm, profile.age, profile.gender
        )
        expected_tdee = calculate_tdee(expected_bmr, pal_for_activity(profile.activity))
        assert user.bmr_kkal == pytest.approx(expected_bmr)
        assert user.tdee_kkal == pytest.approx(expected_tdee)

    @pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.label)
    async def test_registration_upsert_idempotent(
        self,
        async_session: AsyncSession,
        registration_service: RegistrationService,
        profile: Profile,
    ) -> None:
        """Calling register_profile twice for the same telegram_id must not duplicate rows."""
        telegram_id = int(profile.label[1:]) + 100  # offset to avoid collision

        await registration_service.register_profile(
            telegram_id=telegram_id,
            gender=profile.gender,
            age=profile.age,
            weight_kg=profile.weight_kg,
            height_cm=profile.height_cm,
            activity=profile.activity,
        )
        await async_session.commit()

        # Second call — upsert, not insert
        await registration_service.register_profile(
            telegram_id=telegram_id,
            gender=profile.gender,
            age=profile.age,
            weight_kg=profile.weight_kg,
            height_cm=profile.height_cm,
            activity=profile.activity,
        )
        await async_session.commit()

        result = await async_session.execute(select(User).where(User.telegram_id == telegram_id))
        users = result.scalars().all()
        assert len(users) == 1, (
            f"[{profile.label}] Expected 1 User row after upsert, got {len(users)}"
        )

    async def test_conversation_to_tdee_pipeline(
        self,
        async_session: AsyncSession,
        conversation_manager: ConversationManager,
        registration_service: RegistrationService,
    ) -> None:
        """Simulate the Telegram registration flow: state machine → TDEE → DB.

        This mirrors how the Telegram handler will call the system:
        1. /start → start_registration
        2. Sequential accept_* calls
        3. On CONFIRMED → register_profile persists the row
        """
        telegram_id = 9999
        profile = PROFILES[0]  # P1

        # Step 1: begin registration
        conv = conversation_manager.start_registration(telegram_id)
        assert conv.state == ConversationState.AWAIT_GENDER

        # Step 2: collect all fields
        conv = conversation_manager.accept_gender(telegram_id, profile.gender)
        assert conv.state == ConversationState.AWAIT_AGE

        conv = conversation_manager.accept_age(telegram_id, profile.age)
        assert conv.state == ConversationState.AWAIT_WEIGHT

        conv = conversation_manager.accept_weight(telegram_id, profile.weight_kg)
        assert conv.state == ConversationState.AWAIT_HEIGHT

        conv = conversation_manager.accept_height(telegram_id, profile.height_cm)
        assert conv.state == ConversationState.AWAIT_ACTIVITY

        conv = conversation_manager.accept_activity(telegram_id, profile.activity)
        assert conv.state == ConversationState.CONFIRMED

        # Step 3: persist via registration service
        rp = conv.profile
        await registration_service.register_profile(
            telegram_id=telegram_id,
            gender=rp.gender,
            age=rp.age,
            weight_kg=rp.weight_kg,
            height_cm=rp.height_cm,
            activity=rp.activity,
        )

        # Verify
        result = await async_session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        assert user is not None
        assert user.tdee_kkal == pytest.approx(profile.reference_tdee, rel=0.005)
