"""Tests for app/models/models.py — SQLAlchemy ORM models."""

from uuid import UUID

from sqlalchemy import BigInteger, Enum, inspect
from sqlalchemy.types import Date, DateTime

from app.models.models import (
    Base,
    ConsumptionLog,
    DailyEnergy,
    Gender,
    PendingAnalyses,
    User,
)


def test_user_table_name() -> None:
    assert User.__tablename__ == "users"


def test_user_columns() -> None:
    mapper = inspect(User)
    column_names = {c.name for c in mapper.columns}

    expected = {
        "telegram_id",
        "jenis_kelamin",
        "usia_tahun",
        "berat_kg",
        "tinggi_cm",
        "tingkat_aktivitas",
        "pal",
        "bmr_kkal",
        "tdee_kkal",
        "created_at",
    }
    assert expected <= column_names


def test_user_telegram_id_is_primary_key() -> None:
    mapper = inspect(User)
    pk_columns = [c.name for c in mapper.primary_key]
    assert pk_columns == ["telegram_id"]


def test_user_telegram_id_type_is_bigint() -> None:
    col = inspect(User).columns["telegram_id"]
    assert isinstance(col.type, BigInteger)


def test_user_jenis_kelamin_is_enum() -> None:
    col = inspect(User).columns["jenis_kelamin"]
    assert isinstance(col.type, Enum)
    assert set(col.type.enums) == {"Laki-laki", "Perempuan"}


def test_user_tingkat_aktivitas_is_enum() -> None:
    col = inspect(User).columns["tingkat_aktivitas"]
    assert isinstance(col.type, Enum)
    assert set(col.type.enums) == {"Sedentari", "Ringan", "Sedang", "Berat", "Sangat Berat"}


def test_user_pal_is_float() -> None:
    col = inspect(User).columns["pal"]
    assert col.type.python_type is float


def test_user_bmr_tdee_are_floats() -> None:
    mapper = inspect(User)
    assert mapper.columns["bmr_kkal"].type.python_type is float
    assert mapper.columns["tdee_kkal"].type.python_type is float


def test_user_created_at_is_datetime() -> None:
    col = inspect(User).columns["created_at"]
    assert isinstance(col.type, DateTime)


def test_daily_energy_table_name() -> None:
    assert DailyEnergy.__tablename__ == "daily_energy"


def test_daily_energy_composite_pk() -> None:
    mapper = inspect(DailyEnergy)
    pk_columns = [c.name for c in mapper.primary_key]
    assert pk_columns == ["telegram_id", "tanggal_wib"]


def test_daily_energy_tanggal_wib_is_date() -> None:
    col = inspect(DailyEnergy).columns["tanggal_wib"]
    assert isinstance(col.type, Date)


def test_daily_energy_sisa_energi_is_float() -> None:
    col = inspect(DailyEnergy).columns["sisa_energi_kkal"]
    assert col.type.python_type is float


def test_consumption_log_table_name() -> None:
    assert ConsumptionLog.__tablename__ == "consumption_log"


def test_consumption_log_columns() -> None:
    mapper = inspect(ConsumptionLog)
    column_names = {c.name for c in mapper.columns}

    expected = {
        "id",
        "telegram_id",
        "nama_pangan",
        "gram",
        "energi_kkal",
        "gl_aktual",
        "consumed_at",
        "analysis_message_id",
    }
    assert expected <= column_names


def test_consumption_log_id_is_primary_key() -> None:
    mapper = inspect(ConsumptionLog)
    pk_columns = [c.name for c in mapper.primary_key]
    assert pk_columns == ["id"]


def test_consumption_log_consumed_at_is_datetime() -> None:
    col = inspect(ConsumptionLog).columns["consumed_at"]
    assert isinstance(col.type, DateTime)


def test_pending_analyses_table_name() -> None:
    assert PendingAnalyses.__tablename__ == "pending_analyses"


def test_pending_analyses_columns() -> None:
    mapper = inspect(PendingAnalyses)
    column_names = {c.name for c in mapper.columns}

    expected = {
        "id",
        "telegram_id",
        "nama_pangan",
        "gram",
        "energi_kkal",
        "gl_aktual",
        "tanggal_wib",
        "created_at",
        "resolved",
    }
    assert expected <= column_names


def test_pending_analyses_id_is_uuid() -> None:
    col = inspect(PendingAnalyses).columns["id"]
    assert col.type.python_type is UUID


def test_pending_analyses_resolved_defaults_false() -> None:
    col = inspect(PendingAnalyses).columns["resolved"]
    assert col.default is not None
    assert col.default.arg is False


def test_pending_analyses_tanggal_wib_is_date() -> None:
    col = inspect(PendingAnalyses).columns["tanggal_wib"]
    assert isinstance(col.type, Date)


def test_gender_enum_values() -> None:
    assert Gender.LakiLaki.value == "Laki-laki"
    assert Gender.Perempuan.value == "Perempuan"


def test_all_models_registered_on_base() -> None:
    table_names = set(Base.metadata.tables.keys())
    assert {"users", "daily_energy", "consumption_log", "pending_analyses"} <= table_names
