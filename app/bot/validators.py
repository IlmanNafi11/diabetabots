"""Input validation for bot registration and gram parsing."""

import re

from app.models.models import ActivityLevel, Gender


class ValidationError(ValueError):
    """Raised when a user-supplied value cannot be parsed or is out of range."""


def _strip(text: str) -> str:
    return text.strip()


def parse_gender(raw: str) -> Gender:
    """Parse gender from free-text input.

    Accepts Laki-laki / laki variants and the equivalent Perempuan forms,
    case-insensitively.  Raises :exc:`ValidationError` for unrecognised input.
    """
    text = _strip(raw).lower()
    if text in {"laki-laki", "laki", "pria", "l"}:
        return Gender.LakiLaki
    if text in {"perempuan", "wanita", "p"}:
        return Gender.Perempuan
    raise ValidationError(f"'{raw}' bukan pilihan yang valid. Ketik 'Laki-laki' atau 'Perempuan'.")


def parse_age(raw: str) -> int:
    """Parse age from a string; accepts only integer values 18-100 (inclusive)."""
    text = _strip(raw)
    if not re.fullmatch(r"\d{1,3}", text):
        raise ValidationError(f"Usia '{raw}' harus berupa angka bulat.")
    value = int(text)
    if not (18 <= value <= 100):
        raise ValidationError(f"Usia harus antara 18-100 tahun. Anda memasukkan {value}.")
    return value


def parse_weight(raw: str) -> float:
    """Parse body weight in kg from a string; accepts values 25-300 kg (inclusive)."""
    text = _strip(raw).replace(",", ".")
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        raise ValidationError(f"Berat '{raw}' harus berupa angka.")
    value = float(text)
    if not (25.0 <= value <= 300.0):
        raise ValidationError(f"Berat harus antara 25-300 kg. Anda memasukkan {value} kg.")
    return value


def parse_height(raw: str) -> float:
    """Parse height in cm from a string; accepts values 100-250 cm (inclusive)."""
    text = _strip(raw).replace(",", ".")
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        raise ValidationError(f"Tinggi '{raw}' harus berupa angka.")
    value = float(text)
    if not (100.0 <= value <= 250.0):
        raise ValidationError(f"Tinggi harus antara 100-250 cm. Anda memasukkan {value} cm.")
    return value


def parse_activity(raw: str) -> ActivityLevel:
    """Parse activity level from a 1-5 number or free-text keyword."""
    text = _strip(raw).lower()

    # Numeric shortcuts
    num_map = {
        "1": ActivityLevel.Sedentari,
        "2": ActivityLevel.Ringan,
        "3": ActivityLevel.Sedang,
        "4": ActivityLevel.Berat,
        "5": ActivityLevel.SangatBerat,
    }
    if text in num_map:
        return num_map[text]

    # Keyword matching (partial)
    keywords: dict[str, ActivityLevel] = {
        "sangat berat": ActivityLevel.SangatBerat,
        "berat": ActivityLevel.Berat,
        "sedang": ActivityLevel.Sedang,
        "ringan": ActivityLevel.Ringan,
        "santai": ActivityLevel.Sedang,
        "sedentari": ActivityLevel.Sedentari,
        "tidak aktif": ActivityLevel.Sedentari,
    }
    for keyword, level in keywords.items():
        if keyword in text:
            return level

    raise ValidationError(
        f"Tingkat aktivitas '{raw}' tidak dikenali. "
        "Ketik angka 1-5 atau pilih: Sedentari, Ringan, Sedang, Berat, Sangat Berat."
    )


def parse_gram(raw: str) -> float:
    """Parse a gram value from free-text food quantity.

    Accepts bare numbers or numbers with 'g'/'gram'/'gr' suffix.
    Raises :exc:`ValidationError` for non-numeric input or non-positive values.
    For portion descriptions like '1 piring' or '2 butir' that cannot be resolved
    to a gram value, raises :exc:`ValidationError` so the caller can trigger
    the gram-clarification flow.
    """
    text = _strip(raw).lower().replace(",", ".")

    # Strip trailing unit suffixes
    stripped = re.sub(r"\s*(g|gram|gr|gr\.?)\s*$", "", text)

    # Accept integer or decimal - digits optionally followed by decimal and digits
    if not re.fullmatch(r"\d+(\.\d+)?", stripped):
        raise ValidationError(
            f"'{raw}' tidak dapat dikonversi ke gram. Ketik jumlah dalam gram, contoh: 150."
        )

    value = float(stripped)
    if value <= 0:
        raise ValidationError("Jumlah gram harus lebih dari 0.")
    return value
