"""Run the ten PRD §7.3 Telegram scenarios end-to-end (task-15).

This is an *audit runner*, not a pytest test (SPEC.md §6.4 excludes the E2E
scenarios from the suite). It drives the real production handler
(:class:`app.bot.handlers.TelegramUpdateHandler`) with synthetic Telegram
updates parsed through the real ``TelegramUpdate.from_dict`` path, against a
real PostgreSQL database, the real ChromaDB ``tkpi_gi`` collection, and the
real configured LLM/embedding endpoint. Outbound Telegram calls are captured by
a recording double instead of hitting the Telegram API.

Artifacts (§7.4):
    reports/logs/e2e-scenario-01.log … e2e-scenario-10.log   (raw JSONL)
    reports/laporan-e2e-10-scenarios-YYYY-MM-DD.md           (readable report)

Usage:
    uv run python -m scripts.run_e2e [--date YYYY-MM-DD] [--scenario N]…
        [--database-url URL] [--report-dir DIR] [--log-dir DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.bot.conversation import ConversationManager
from app.bot.handlers import TelegramUpdateHandler
from app.config import Settings
from app.database import create_async_engine, create_sessionmaker, init_db
from app.evaluation.e2e import (
    E2E_SCENARIOS,
    OutboundCall,
    Scenario,
    ScenarioResult,
    evaluate_scenario,
    registration_turns,
    render_e2e_report,
    render_raw_log,
    scenario_log_name,
)
from app.models.models import ConsumptionLog, DailyEnergy, PendingAnalyses, User
from app.services.energy_service import EnergyService, wib_today
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.registration_service import RegistrationService
from app.services.telegram_service import TelegramUpdate

#: Telegram chat ids used for the evaluation, one per scenario.
CHAT_ID_BASE = 9_000_000

#: Default database: the local test database (same host as production config).
DEFAULT_E2E_DATABASE_URL = (
    "postgresql+asyncpg://diabetabot:diabetabot@localhost:5432/diabetabot_test"
)


class RecordingTelegramClient:
    """Captures outbound handler calls instead of calling the Telegram API."""

    def __init__(self) -> None:
        self.calls: list[OutboundCall] = []
        self._next_message_id = 10_000

    def _new_message_id(self) -> int:
        self._next_message_id += 1
        return self._next_message_id

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        keyboard = (reply_markup or {}).get("inline_keyboard", []) if reply_markup else []
        message_id = self._new_message_id()
        self.calls.append(
            OutboundCall(
                method="send_message",
                chat_id=chat_id,
                text=text,
                inline_keyboard=keyboard,
                message_id=message_id,
            )
        )
        return {"ok": True, "result": {"message_id": message_id}}

    async def send_message_with_inline_buttons(
        self,
        chat_id: int,
        text: str,
        *,
        inline_keyboard: list[list[dict[str, str]]],
    ) -> dict[str, Any]:
        message_id = self._new_message_id()
        self.calls.append(
            OutboundCall(
                method="send_message_with_inline_buttons",
                chat_id=chat_id,
                text=text,
                inline_keyboard=inline_keyboard,
                message_id=message_id,
            )
        )
        return {"ok": True, "result": {"message_id": message_id}}

    async def edit_message_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        *,
        inline_keyboard: list[list[dict[str, str]]],
    ) -> dict[str, Any]:
        self.calls.append(
            OutboundCall(
                method="edit_message_reply_markup",
                chat_id=chat_id,
                text="",
                inline_keyboard=inline_keyboard,
                message_id=message_id,
            )
        )
        return {"ok": True, "result": {}}

    async def answer_callback_query(
        self, callback_query_id: str, *, text: str | None = None
    ) -> dict[str, Any]:
        self.calls.append(OutboundCall(method="answer_callback_query", chat_id=0, text=text or ""))
        return {"ok": True, "result": {}}

    def last_pending_id(self) -> str | None:
        """Return the pending UUID from the most recent inline keyboard."""
        for call in reversed(self.calls):
            for row in call.inline_keyboard:
                for button in row:
                    data = button.get("callback_data", "")
                    if data.startswith("confirm_ya:"):
                        return data.split(":", 1)[1]
        return None


# ---------------------------------------------------------------------------
# Update builders (exercise the real TelegramUpdate.from_dict parsing path)
# ---------------------------------------------------------------------------


def _message_update(update_id: int, chat_id: int, text: str) -> TelegramUpdate:
    return TelegramUpdate.from_dict(
        {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": 1_760_000_000,
                "chat": {"id": chat_id, "type": "private"},
                "from": {"id": chat_id, "is_bot": False},
                "text": text,
            },
        }
    )


def _callback_update(update_id: int, chat_id: int, data: str) -> TelegramUpdate:
    return TelegramUpdate.from_dict(
        {
            "update_id": update_id,
            "callback_query": {
                "id": f"cb-{update_id}",
                "from": {"id": chat_id, "is_bot": False},
                "chat_instance": "e2e",
                "data": data,
                "message": {
                    "message_id": update_id,
                    "chat": {"id": chat_id, "type": "private"},
                },
            },
        }
    )


# ---------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------


class ScenarioDriver:
    """Runs one scenario through the real handler and collects observations."""

    def __init__(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        rag_svc: RAGService,
        llm_svc: LLMService,
        telegram_client: RecordingTelegramClient,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._rag_svc = rag_svc
        self._llm_svc = llm_svc
        self._telegram = telegram_client

    async def _handle(self, update: TelegramUpdate) -> None:
        """Handle one update in its own session, mirroring AppRuntime._dispatch."""
        async with self._sessionmaker() as session:
            handler = TelegramUpdateHandler(
                telegram_svc=self._telegram,  # type: ignore[arg-type]
                conversation_manager=self._conversation,
                registration_svc=RegistrationService(session),
                energy_svc=EnergyService(session),
                rag_svc=self._rag_svc,
                llm_svc=self._llm_svc,
                session=session,
            )
            await handler.handle_update(update)
            await session.commit()

    async def _clean(self, chat_id: int) -> None:
        async with self._sessionmaker() as session:
            for table in (ConsumptionLog, PendingAnalyses, DailyEnergy, User):
                await session.execute(delete(table).where(table.telegram_id == chat_id))
            await session.commit()

    async def _balance(self, chat_id: int) -> float | None:
        async with self._sessionmaker() as session:
            record = (
                await session.execute(
                    select(DailyEnergy).where(
                        DailyEnergy.telegram_id == chat_id,
                        DailyEnergy.tanggal_wib == wib_today(),
                    )
                )
            ).scalar_one_or_none()
            return None if record is None else record.sisa_energi_kkal

    async def _latest_pending(self, chat_id: int) -> PendingAnalyses | None:
        async with self._sessionmaker() as session:
            return (
                await session.execute(
                    select(PendingAnalyses)
                    .where(PendingAnalyses.telegram_id == chat_id)
                    .order_by(PendingAnalyses.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

    async def run(self, scenario: Scenario) -> tuple[list[OutboundCall], dict[str, Any]]:
        chat_id = CHAT_ID_BASE + scenario.number
        self._telegram.calls.clear()
        self._conversation = ConversationManager()
        await self._clean(chat_id)

        facts: dict[str, Any] = {}
        error: str | None = None
        update_id = scenario.number * 100

        # 1. Registration (five steps) — always exercised end-to-end.
        for text in registration_turns(scenario.profile):
            update_id += 1
            await self._handle(_message_update(update_id, chat_id, text))

        # 2. Scenario-specific turns.
        try:
            for turn in scenario.turns:
                update_id += 1
                if turn.kind == "message":
                    await self._handle(_message_update(update_id, chat_id, turn.text))
                elif turn.kind == "callback_ya":
                    facts["balance_before"] = await self._balance(chat_id)
                    pending_id = self._telegram.last_pending_id()
                    if pending_id is None:
                        raise RuntimeError("no pending analysis produced a Ya button")
                    await self._handle(
                        _callback_update(update_id, chat_id, f"confirm_ya:{pending_id}")
                    )
                    facts["balance_after"] = await self._balance(chat_id)
                    # Idempotency probe: a second tap must not reduce again.
                    update_id += 1
                    await self._handle(
                        _callback_update(update_id, chat_id, f"confirm_ya:{pending_id}")
                    )
                    facts["balance_after_double"] = await self._balance(chat_id)
                elif turn.kind == "callback_tidak":
                    pending_id = self._telegram.last_pending_id()
                    if pending_id is None:
                        raise RuntimeError("no pending analysis produced a Tidak button")
                    await self._handle(
                        _callback_update(update_id, chat_id, f"confirm_tidak:{pending_id}")
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        # 3. Retrieval + persistence facts.
        if scenario.food_query:
            try:
                retrieved = self._rag_svc.retrieve(scenario.food_query, top_k=5)
                if retrieved:
                    facts["top1_name"] = retrieved[0].name
                    facts["top1_gi"] = retrieved[0].gi_value
                    facts["top1_distance"] = retrieved[0].distance
            except Exception as exc:
                facts["retrieval_error"] = f"{type(exc).__name__}: {exc}"

        pending = await self._latest_pending(chat_id)
        if pending is not None:
            facts["pending_nama"] = pending.nama_pangan
            facts["pending_gram"] = pending.gram
            facts["energy_kkal"] = pending.energi_kkal
            facts["pending_gl"] = pending.gl_aktual
            facts["pending_resolved"] = pending.resolved

        facts["error"] = error
        return list(self._telegram.calls), facts


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _safe_environment(settings: Settings, doc_count: int) -> dict[str, Any]:
    """Report configuration without leaking credentials."""
    from urllib.parse import urlparse

    parsed = urlparse(settings.database_url)
    return {
        "Basis data": f"{parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}",
        "Koleksi ChromaDB": settings.collection_name,
        "Jumlah dokumen terindeks": doc_count,
        "Model embedding": settings.embedding_model_name,
        "Dimensi embedding": settings.embedding_dim,
        "Model chat (LLM)": settings.chat_model_name,
        "Strategi chunking": settings.chunking_strategy,
        "RAG top-k": settings.rag_top_k,
        "RAG kandidat (name-grounding)": settings.rag_candidate_k,
    }


def _collection_count(settings: Settings) -> int:
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        return client.get_collection(name=settings.collection_name).count()
    except Exception:
        return 0


async def run_all(
    settings: Settings,
    scenarios: list[Scenario],
    *,
    report_dir: Path,
    log_dir: Path,
    generated_at: str,
) -> tuple[list[ScenarioResult], int]:
    engine: AsyncEngine = create_async_engine(settings)
    sessionmaker = create_sessionmaker(engine)
    results: list[ScenarioResult] = []
    try:
        await init_db(engine)
        rag_svc = RAGService(settings)
        llm_svc = LLMService(settings)
        doc_count = _collection_count(settings)
        telegram_client = RecordingTelegramClient()
        driver = ScenarioDriver(
            sessionmaker=sessionmaker,
            rag_svc=rag_svc,
            llm_svc=llm_svc,
            telegram_client=telegram_client,
        )

        log_dir.mkdir(parents=True, exist_ok=True)
        for scenario in scenarios:
            calls, facts = await driver.run(scenario)
            result = evaluate_scenario(scenario, calls, facts)
            results.append(result)
            (log_dir / scenario_log_name(scenario.number)).write_text(
                render_raw_log(scenario, calls, facts), encoding="utf-8"
            )
            status = "PASS" if result.passed else "FAIL"
            print(f"scenario {scenario.number:02d}: {status} — {scenario.description}")

        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"laporan-e2e-10-scenarios-{generated_at[:10]}.md"
        report_path.write_text(
            render_e2e_report(
                results,
                generated_at=generated_at,
                environment=_safe_environment(settings, doc_count),
            ),
            encoding="utf-8",
        )
        print(f"report: {report_path}")
    finally:
        await engine.dispose()

    passed = sum(1 for result in results if result.passed)
    return results, passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.now(UTC).astimezone().strftime("%Y-%m-%d"),
        help="Report date stamp (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        action="append",
        dest="scenarios",
        help="Run only these scenario numbers (repeatable).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override the E2E database URL (defaults to the local test database).",
    )
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--log-dir", default="reports/logs")
    args = parser.parse_args(argv)

    settings = Settings()
    if args.database_url:
        settings = settings.model_copy(update={"database_url": args.database_url})
    elif not settings.database_url.endswith("/diabetabot_test"):
        settings = settings.model_copy(update={"database_url": DEFAULT_E2E_DATABASE_URL})

    selected = [
        scenario
        for scenario in E2E_SCENARIOS
        if not args.scenarios or scenario.number in args.scenarios
    ]

    generated_at = args.date
    results, passed = asyncio.run(
        run_all(
            settings,
            selected,
            report_dir=Path(args.report_dir),
            log_dir=Path(args.log_dir),
            generated_at=generated_at,
        )
    )
    total = len(results)
    print(json.dumps({"passed": passed, "total": total}, ensure_ascii=False))
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
