"""Tests for the TKPI+GI JSONL parser (task-07 data indexing)."""

from pathlib import Path

import pytest

from app.data.parser import load_records

# Each fixture row is one physical JSON line, matching the dataset format.
RAW_LINE = (
    '{"Nama Bahan Pangan": "Beras giling,\\nmentah", "Kategori": "serealia",'
    ' "Per Amount": 100.0, "Per Unit": "g", "Energi": 357.0, "Protein": 8.4,'
    ' "Air": 12.0, "Karbohidrat": 77.1, "Lemak": 1.7, "Serat": 0.2, "Abu": 0.8,'
    ' "Kalsium": 147.0, "Besi": 1.8, "Kalium": 71.0, "Seng": 0.5, "Fosfor": 81.0,'
    ' "Natrium": 27.0, "Tembaga": 0.1, "Retinol": 0.0, "Karoten Total": 0.0,'
    ' "Beta Karoten": 0.0, "Thiamin": 0.2, "Riboflavin": 0.08, "Niasin": 2.6,'
    ' "Vitamin C": 0.0, "BDD": 100.0, "matched_gi_id": "Tidak ada",'
    ' "matched_gi_name": "Tidak ada", "gi_value": 64.0, "gi_sem": "sedang",'
    ' "gl_value": 49.3, "available_carbohydrate_raw": null, "serving_size_raw": null,'
    ' "serving_size_g": null, "Kategori IG": null, "Kategori BG": null}'
)

RAW_LINE_NO_GI = (
    '{"Nama Bahan Pangan": "Bayam rebus", "Kategori": "sayuran",'
    ' "Per Amount": 100.0, "Per Unit": "g", "Energi": 37.0, "Protein": 3.0,'
    ' "Air": 90.0, "Karbohidrat": 4.4, "Lemak": 0.5, "Serat": 2.2, "Abu": 1.9,'
    ' "Kalsium": 146.0, "Besi": 3.0, "Kalium": 490.0, "Seng": 0.6, "Fosfor": 51.0,'
    ' "Natrium": 74.0, "Tembaga": 0.1, "Retinol": 0.0, "Karoten Total": 2939.0,'
    ' "Beta Karoten": 1764.0, "Thiamin": 0.05, "Riboflavin": 0.16, "Niasin": 0.5,'
    ' "Vitamin C": 20.0, "BDD": 100.0, "matched_gi_id": "Tidak ada",'
    ' "matched_gi_name": "Tidak ada", "gi_value": null, "gi_sem": null,'
    ' "gl_value": null, "available_carbohydrate_raw": null, "serving_size_raw": null,'
    ' "serving_size_g": null, "Kategori IG": null, "Kategori BG": null}'
)


@pytest.fixture()
def sample_jsonl(tmp_path: Path) -> Path:
    """Two food records separated by a blank line.

    The blank line produces no record, but record IDs track the physical
    line number so IDs remain stable regardless of surrounding blank lines.
    """
    source = tmp_path / "foods.jsonl"
    source.write_text(RAW_LINE + "\n\n" + RAW_LINE_NO_GI + "\n", encoding="utf-8")
    return source


class TestLoadRecords:
    def test_returns_one_record_per_jsonl_line(self, sample_jsonl: Path) -> None:
        records = load_records(sample_jsonl)

        assert len(records) == 2
        assert [record.name for record in records] == [
            "Beras giling, mentah",
            "Bayam rebus",
        ]

    def test_normalizes_newlines_in_food_name(self, sample_jsonl: Path) -> None:
        records = load_records(sample_jsonl)

        assert "\n" not in records[0].name
        assert records[0].name == "Beras giling, mentah"

    def test_maps_nutrient_fields_to_floats(self, sample_jsonl: Path) -> None:
        records = load_records(sample_jsonl)

        nutrients = records[0].nutrients
        assert nutrients["Energi"] == pytest.approx(357.0)
        assert nutrients["Protein"] == pytest.approx(8.4)
        assert nutrients["Vitamin C"] == pytest.approx(0.0)

    def test_missing_gi_fields_default_to_none(self, sample_jsonl: Path) -> None:
        records = load_records(sample_jsonl)

        assert records[0].gi_value == pytest.approx(64.0)
        assert records[0].gl_value == pytest.approx(49.3)
        assert records[1].gi_value is None
        assert records[1].gl_value is None

    def test_record_ids_track_physical_line_numbers(self, sample_jsonl: Path) -> None:
        records = load_records(sample_jsonl)

        assert [record.record_id for record in records] == [
            "food-0000",
            "food-0002",
        ]

    def test_skips_blank_lines_without_shifting_first_id(self, tmp_path: Path) -> None:
        source = tmp_path / "foods.jsonl"
        source.write_text("\n" + RAW_LINE + "\n\n", encoding="utf-8")

        records = load_records(source)

        assert len(records) == 1
        assert records[0].record_id == "food-0001"

    def test_raises_keyerror_for_missing_required_field(self, tmp_path: Path) -> None:
        source = tmp_path / "foods.jsonl"
        source.write_text('{"Kategori": "serealia"}\n', encoding="utf-8")

        with pytest.raises(KeyError):
            load_records(source)

    def test_parses_real_dataset_with_expected_entry_count(self) -> None:
        dataset = Path("data/tkpi-gi-final.jsonl")
        if not dataset.exists():
            pytest.skip("real dataset not present in this checkout")

        records = load_records(dataset)

        assert len(records) == 1146
