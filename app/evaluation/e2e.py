"""End-to-end scenario evaluation harness (task-15, PRD §7.3/§7.4).

This module holds the *pure* parts of the ten-scenario evaluation:

- the explicit catalogue of the ten PRD §7.3 Telegram scenarios;
- a transcript evaluator that turns recorded outbound calls plus a few
  database facts into a per-scenario pass/fail :class:`ScenarioResult`;
- markdown report + raw-log renderers for the §7.4 audit artifacts.

The live driver (real Telegram handler, PostgreSQL, ChromaDB, LLM) lives in
``scripts/run_e2e.py`` so that this module stays importable without any
network, database, or credential dependency.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

#: PRD §4 simulation profiles: label → (gender, age, weight_kg, height_cm, activity_key)
PROFILES: dict[str, tuple[str, int, float, float, str]] = {
    "P1": ("Laki-laki", 30, 70.0, 170.0, "Ringan"),
    "P2": ("Perempuan", 28, 60.0, 160.0, "Ringan"),
    "P3": ("Laki-laki", 65, 95.0, 170.0, "Sedentari"),
    "P4": ("Perempuan", 63, 85.0, 155.0, "Sedentari"),
    "P5": ("Laki-laki", 32, 100.0, 175.0, "Sedentari"),
    "P6": ("Perempuan", 30, 90.0, 158.0, "Sedentari"),
    "P7": ("Laki-laki", 60, 68.0, 168.0, "Sedang"),
    "P8": ("Perempuan", 58, 58.0, 152.0, "Sedang"),
}

#: Activity keyword → the numeric shortcut accepted by ``parse_activity``.
_ACTIVITY_INPUT = {
    "Sedentari": "1",
    "Ringan": "2",
    "Sedang": "3",
    "Berat": "4",
    "SangatBerat": "5",
}


@dataclass(frozen=True, slots=True)
class Turn:
    """One inbound Telegram event in a scenario script.

    ``kind`` is ``"message"`` for a text message, or ``"callback_ya"`` /
    ``"callback_tidak"`` for an inline-button tap. Callback turns carry no
    text — the driver resolves the pending row id from the button payload.
    """

    kind: str
    text: str = ""


def message(text: str) -> Turn:
    return Turn(kind="message", text=text)


CONFIRM_YA = Turn(kind="callback_ya")
CONFIRM_TIDAK = Turn(kind="callback_tidak")


@dataclass(frozen=True, slots=True)
class Scenario:
    """A PRD §7.3 end-to-end scenario with its expected behaviours."""

    number: int
    category: str
    profile: str
    description: str
    turns: tuple[Turn, ...]
    expects: tuple[str, ...]
    expected_top_term: str = ""

    @property
    def food_query(self) -> str:
        """The first free-text food query the user sends (for retrieval facts)."""
        for turn in self.turns:
            if turn.kind == "message" and turn.text.strip() not in {"", "/start"}:
                return turn.text.strip()
        return ""


E2E_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        1,
        "Pokok",
        "P1",
        "Tanya gizi nasi putih 150 g (tersedia, ber-GI)",
        (message("nasi putih 150g"),),
        ("registration", "analysis", "no_error"),
    ),
    Scenario(
        2,
        "Pokok",
        "P3",
        "Makan makaroni 1 mangkuk → klarifikasi gram → konfirmasi konsumsi",
        (message("makan makaroni 1 mangkuk"), message("200"), CONFIRM_YA),
        ("registration", "gram_clarify", "analysis", "confirm_reduces_once", "no_error"),
    ),
    Scenario(
        3,
        "Lauk hewani",
        "P2",
        "Telur rebus 2 butir → klarifikasi gram per butir",
        (message("telur rebus 2 butir"), message("100")),
        ("registration", "gram_clarify", "analysis", "no_error"),
    ),
    Scenario(
        4,
        "Lauk hewani",
        "P4",
        "Ikan kembung 100 g → konfirmasi konsumsi",
        (message("ikan kembung 100g"), CONFIRM_YA),
        ("registration", "analysis", "correct_food", "confirm_reduces_once", "no_error"),
        expected_top_term="kembung",
    ),
    Scenario(
        5,
        "Lauk nabati",
        "P5",
        "Tahu 100 g → konfirmasi konsumsi",
        (message("tahu 100g"), CONFIRM_YA),
        ("registration", "analysis", "confirm_reduces_once", "no_error"),
    ),
    Scenario(
        6,
        "Lauk nabati",
        "P6",
        "Oncom (tersedia, tanpa IG → respons transparan)",
        (message("oncom 100g"),),
        ("registration", "analysis", "gi_unavailable_source", "no_error"),
    ),
    Scenario(
        7,
        "Sayuran",
        "P7",
        "Kangkung 100 g (tersedia, tanpa IG → respons transparan)",
        (message("kangkung 100g"),),
        ("registration", "analysis", "gi_unavailable_source", "no_error"),
    ),
    Scenario(
        8,
        "Sayuran",
        "P8",
        "Sayur aster (tidak tersedia di TKPI → respons jelas)",
        (message("sayur aster 100g"),),
        ("registration", "not_found", "no_error"),
    ),
    Scenario(
        9,
        "Buah",
        "P1",
        "Semangka 200 g (tersedia & ber-GI → GL aktual dihitung)",
        (message("semangka 200g"),),
        ("registration", "analysis", "gl_computed", "no_error"),
    ),
    Scenario(
        10,
        "Buah",
        "P3",
        "Buah naga merah (tersedia, tanpa IG, BDD 66,5%)",
        (message("buah naga merah 100g"),),
        ("registration", "analysis", "gi_unavailable_source", "no_error"),
    ),
)


def registration_turns(profile: str) -> tuple[str, ...]:
    """Return the registration message texts for a PRD §4 profile label."""
    gender, age, weight, height, activity = PROFILES[profile]
    return ("/start", gender, str(age), str(weight), str(height), _ACTIVITY_INPUT[activity])


# ---------------------------------------------------------------------------
# Recorded transcript + evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutboundCall:
    """One recorded call the handler made to the Telegram client."""

    method: str
    chat_id: int
    text: str = ""
    inline_keyboard: list[list[dict[str, str]]] = field(default_factory=list)
    message_id: int | None = None


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Outcome of one evaluated scenario."""

    number: int
    profile: str
    category: str
    description: str
    passed: bool
    checks: tuple[str, ...]
    messages: tuple[str, ...]
    error: str | None = None


#: Human-readable labels for each expectation key.
_CHECK_LABELS = {
    "registration": "registrasi profil selesai",
    "gram_clarify": "klarifikasi gram diminta",
    "analysis": "analisis terkirim dengan tombol Ya/Tidak",
    "not_found": "respons 'tidak tersedia' terkirim",
    "correct_food": "pangan teratas sesuai kata kunci skenario",
    "confirm_reduces_once": "tombol Ya mengurangi tepat sekali (idempoten)",
    "gl_computed": "GL aktual dihitung dari data TKPI",
    "gi_unavailable_source": "sumber teratas tanpa data IG (jalur transparan)",
    "no_error": "tidak ada pesan galat",
}


def evaluate_scenario(
    scenario: Scenario,
    calls: Sequence[OutboundCall],
    facts: dict[str, Any],
) -> ScenarioResult:
    """Evaluate one scenario's recorded transcript against its expectations.

    ``facts`` carries database/retrieval observations the transcript alone
    cannot show (balance before/after a confirmation, the pending row's GL,
    the top retrieved food's GI, …).
    """
    texts = [call.text for call in calls if call.text]
    checks: list[str] = []

    def check(key: str, passed: bool) -> None:
        checks.append(f"{'PASS' if passed else 'FAIL'} — {_CHECK_LABELS[key]}")

    for key in scenario.expects:
        if key == "registration":
            check(key, any("Registrasi selesai" in text for text in texts))
        elif key == "gram_clarify":
            check(key, any("Berapa gram" in text for text in texts))
        elif key == "analysis":
            check(
                key,
                any(call.method == "send_message_with_inline_buttons" for call in calls),
            )
        elif key == "not_found":
            check(key, any("Tidak ditemukan" in text for text in texts))
        elif key == "confirm_reduces_once":
            before = facts.get("balance_before")
            after = facts.get("balance_after")
            energy = facts.get("energy_kkal")
            double_after = facts.get("balance_after_double")
            single_ok = False
            if None not in (before, after, energy):
                single_ok = abs(round(before - energy, 2) - after) < 1e-6
            idempotent = (
                double_after is not None
                and after is not None
                and (abs(double_after - after) < 1e-6)
            )
            check(key, bool(single_ok and idempotent))
        elif key == "correct_food":
            term = scenario.expected_top_term.casefold()
            top_name = str(facts.get("top1_name", "")).casefold()
            check(key, bool(term and term in top_name))
        elif key == "gl_computed":
            check(key, facts.get("pending_gl") is not None)
        elif key == "gi_unavailable_source":
            check(key, facts.get("top1_gi") is None)
        elif key == "no_error":
            errored = any("analisis gagal" in text.lower() for text in texts)
            check(key, not errored and facts.get("error") is None)
        else:  # pragma: no cover - guards a typo in the catalogue
            raise ValueError(f"unknown expectation key: {key!r}")

    passed = all(entry.startswith("PASS") for entry in checks)
    return ScenarioResult(
        number=scenario.number,
        profile=scenario.profile,
        category=scenario.category,
        description=scenario.description,
        passed=passed,
        checks=tuple(checks),
        messages=tuple(texts),
        error=facts.get("error"),
    )


# ---------------------------------------------------------------------------
# Artifact rendering (§7.4)
# ---------------------------------------------------------------------------


def scenario_log_name(number: int) -> str:
    """Return the raw-log filename for a scenario, e.g. ``e2e-scenario-03.log``."""
    return f"e2e-scenario-{number:02d}.log"


def render_raw_log(
    scenario: Scenario,
    calls: Sequence[OutboundCall],
    facts: dict[str, Any],
) -> str:
    """Render a scenario's transcript as a JSONL raw log (unedited evidence)."""
    lines: list[str] = [
        json.dumps(
            {
                "scenario": scenario.number,
                "category": scenario.category,
                "profile": scenario.profile,
                "description": scenario.description,
                "turns": [{"kind": turn.kind, "text": turn.text} for turn in scenario.turns],
                "expects": list(scenario.expects),
            },
            ensure_ascii=False,
        )
    ]
    for call in calls:
        lines.append(
            json.dumps(
                {
                    "method": call.method,
                    "chat_id": call.chat_id,
                    "text": call.text,
                    "inline_keyboard": call.inline_keyboard,
                    "message_id": call.message_id,
                },
                ensure_ascii=False,
            )
        )
    lines.append(json.dumps({"facts": facts}, ensure_ascii=False, default=str))
    return "\n".join(lines) + "\n"


def render_e2e_report(
    results: Sequence[ScenarioResult],
    *,
    generated_at: str,
    environment: dict[str, Any] | None = None,
) -> str:
    """Render the §7.4 markdown report summarising all ten scenarios."""
    env = environment or {}
    total = len(results)
    passed = sum(1 for result in results if result.passed)

    lines: list[str] = [
        "# Laporan Evaluasi E2E 10 Skenario Percakapan Telegram",
        "",
        f"**Tanggal:** {generated_at} · **Plan ref:** task-15 · "
        "**SPEC:** SPEC.md §8 (telegram-integration)",
        "",
        "## Apa yang Diuji",
        "",
        "Menjalankan 10 skenario percakapan Telegram end-to-end (5 kategori pangan, "
        "kondisi tersedia/tidak tersedia) melalui handler produksi "
        "(`TelegramUpdateHandler`), dengan basis data PostgreSQL nyata, koleksi "
        "ChromaDB hasil indeks TKPI+GI, dan LLM nyata sesuai konfigurasi `.env`.",
        "",
        "**Kriteria lolos:** seluruh 10 skenario berjalan tanpa galat, dengan perilaku "
        "terverifikasi (klarifikasi gram, konfirmasi Ya idempoten, jalur tanpa IG, "
        "pangan tidak tersedia, GL aktual).",
        "",
        "## Bahan Pengujian",
        "",
    ]

    if env:
        lines.append("| Parameter | Nilai |")
        lines.append("|---|---|")
        for key, value in env.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    lines += [
        "- **Bukti mentah (tidak diedit):** `reports/logs/e2e-scenario-01.log` … "
        "`reports/logs/e2e-scenario-10.log`.",
        "",
        "## Hasil Pengujian",
        "",
        "| # | Kategori | Profil | Skenario | Hasil | Bukti |",
        "|---|---|---|---|---|---|",
    ]

    for result in results:
        status = "✅ Lulus" if result.passed else "❌ Gagal"
        lines.append(
            f"| {result.number} | {result.category} | {result.profile} | "
            f"{result.description} | {status} | "
            f"`reports/logs/{scenario_log_name(result.number)}` |"
        )

    lines += [
        "",
        f"**Ringkasan: {'✅ Lulus' if passed == total else '❌ Gagal'}: {passed}/{total}**",
        "",
        "### Detail pemeriksaan per skenario",
        "",
    ]

    for result in results:
        lines.append(f"**Skenario {result.number} — {result.description}**")
        lines.append("")
        for entry in result.checks:
            lines.append(f"- {entry}")
        if result.error:
            lines.append(f"- ⚠️ galat: {result.error}")
        lines.append("")

    lines += [
        "## Kesimpulan & Keputusan",
        "",
    ]

    failures = [result for result in results if not result.passed]
    if not failures:
        lines.append(
            "- **AC PRD §1.3 #5 & #6 terpenuhi:** seluruh 10 skenario berjalan tanpa "
            "galat dan artefak bukti tersimpan (log mentah per skenario + laporan ini)."
        )
    else:
        lines.append(
            f"- **{len(failures)} skenario belum lulus:** "
            + ", ".join(f"#{result.number}" for result in failures)
            + ". Lihat detail di atas dan log mentah terkait."
        )
    lines.append("")

    return "\n".join(lines)
