"""Parser for the TKPI+GI nutrition dataset (task-07).

Converts each JSONL line of ``data/tkpi-gi-final.jsonl`` into a strongly typed
:class:`FoodRecord` while preserving the one-record-per-line invariant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NUTRIENT_FIELDS = (
    "Energi",
    "Protein",
    "Air",
    "Karbohidrat",
    "Lemak",
    "Serat",
    "Abu",
    "Kalsium",
    "Besi",
    "Kalium",
    "Seng",
    "Fosfor",
    "Natrium",
    "Tembaga",
    "Retinol",
    "Karoten Total",
    "Beta Karoten",
    "Thiamin",
    "Riboflavin",
    "Niasin",
    "Vitamin C",
)


@dataclass(frozen=True)
class FoodRecord:
    """One source row from the TKPI+GI dataset."""

    record_id: str
    name: str
    category: str
    nutrients: dict[str, float] = field(default_factory=dict)
    gi_value: float | None = None
    gl_value: float | None = None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_records(path: Path) -> list[FoodRecord]:
    """Load JSONL data with exactly one record per non-empty line.

    Record IDs derive from the raw line number (``food-NNNN``) so they stay
    stable even when blank lines appear between entries.
    """
    records: list[FoodRecord] = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            raw = json.loads(line)
            nutrients = {
                field_name: float(raw[field_name])
                for field_name in NUTRIENT_FIELDS
                if raw.get(field_name) not in (None, "")
            }
            records.append(
                FoodRecord(
                    record_id=f"food-{index:04d}",
                    name=str(raw["Nama Bahan Pangan"]).replace("\n", " ").strip(),
                    category=str(raw["Kategori"]),
                    nutrients=nutrients,
                    gi_value=_optional_float(raw.get("gi_value")),
                    gl_value=_optional_float(raw.get("gl_value")),
                )
            )
    return records
