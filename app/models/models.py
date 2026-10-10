"""SQLAlchemy ORM models for DiabetaBot (SPEC section 5.2, PRD section 8.2).

Four tables back the chat experience:

- ``users``            — one physiological profile per Telegram account.
- ``daily_energy``     — remaining daily energy, keyed by (telegram_id, date).
- ``consumption_log``  — confirmed consumption events.
- ``pending_analyses`` — source of truth for the inline Ya/Tidak button payload.
"""

from datetime import date, datetime
from enum import Enum, StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


class Gender(StrEnum):
    """Biological sex used by the Mifflin-St Jeor BMR formula.

    Members are PascalCase per the naming convention in SPEC section 5.2.
    """

    LakiLaki = "Laki-laki"
    Perempuan = "Perempuan"


class ActivityLevel(StrEnum):
    """Closed set of activity levels mapped to the five PAL multipliers."""

    Sedentari = "Sedentari"
    Ringan = "Ringan"
    Sedang = "Sedang"
    Berat = "Berat"
    SangatBerat = "Sangat Berat"


def _enum_values(enum_cls: type[Enum]) -> list[str]:
    """Return enum *values* (not member names) for a PostgreSQL enum type."""
    return [member.value for member in enum_cls]


class User(Base):
    """A Telegram account's physiological profile and derived energy targets."""

    __tablename__ = "users"

    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    jenis_kelamin: Mapped[Gender] = mapped_column(
        SAEnum(Gender, name="gender", values_callable=_enum_values)
    )
    usia_tahun: Mapped[int] = mapped_column(Integer)
    berat_kg: Mapped[float] = mapped_column(Float)
    tinggi_cm: Mapped[float] = mapped_column(Float)
    tingkat_aktivitas: Mapped[ActivityLevel] = mapped_column(
        SAEnum(ActivityLevel, name="activity_level", values_callable=_enum_values)
    )
    pal: Mapped[float] = mapped_column(Float)
    bmr_kkal: Mapped[float] = mapped_column(Float)
    tdee_kkal: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DailyEnergy(Base):
    """Remaining energy for a user on a given WIB date.

    Set to the user's TDEE on the first activity of the day and reset at
    midnight WIB. The composite primary key makes the reset idempotent.
    """

    __tablename__ = "daily_energy"

    telegram_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.telegram_id"), primary_key=True
    )
    tanggal_wib: Mapped[date] = mapped_column(Date, primary_key=True)
    sisa_energi_kkal: Mapped[float] = mapped_column(Float)


class ConsumptionLog(Base):
    """A confirmed consumption event (only recorded when the user taps "Ya")."""

    __tablename__ = "consumption_log"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    telegram_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.telegram_id"), index=True
    )
    nama_pangan: Mapped[str] = mapped_column(Text)
    gram: Mapped[float] = mapped_column(Float)
    energi_kkal: Mapped[float] = mapped_column(Float)
    gl_aktual: Mapped[float | None] = mapped_column(Float, nullable=True)
    consumed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    analysis_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class PendingAnalyses(Base):
    """Verified payload behind an analysis message's inline Ya/Tidak button.

    Idempotency against double taps is enforced by the ``resolved`` flag.
    """

    __tablename__ = "pending_analyses"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    telegram_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.telegram_id"), index=True
    )
    nama_pangan: Mapped[str] = mapped_column(Text)
    gram: Mapped[float] = mapped_column(Float)
    energi_kkal: Mapped[float] = mapped_column(Float)
    gl_aktual: Mapped[float | None] = mapped_column(Float, nullable=True)
    tanggal_wib: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
