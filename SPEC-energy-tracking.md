# Spec: energy-tracking

**Status:** Implemented; task-13 Telegram button wiring + midnight scheduler
complete (unit-verified) · **Plan ref:** task-04/task-05 (service + unit),
task-13 (integration) · **SPEC.md §8:** energy-tracking

## Objective

Track the user's remaining daily energy (`sisa_energi_kkal`), seeded at TDEE each WIB
day, reduced once per confirmed consumption ("Ya" button), with idempotent midnight-WIB
reset and negative balances allowed with a warning.

## Public API

- `app/services/energy_service.py`
  - `WIB = timezone(timedelta(hours=7))`; `wib_now(now=None) -> datetime`;
    `wib_today(now=None) -> date`
  - Errors: `EnergyServiceError` (base), `UserNotRegisteredError`, `InsufficientEnergyError`
  - `EnergyService(session)`:
    - `ensure_today(telegram_id, tdee_kkal, *, now=None) -> DailyEnergy` — seed
      `sisa = TDEE` only if today's row absent
    - `reduce_energy(telegram_id, amount_kkal, *, allow_negative=True, now=None)`
      — decrement once ("Ya"); `allow_negative=False` raises `InsufficientEnergyError`
    - `reduce_energy_with_flag(...) -> tuple[DailyEnergy, bool]` — decrement plus
      `warned = (new_balance < 0)`
    - `get_today(telegram_id, *, now=None) -> DailyEnergy | None`
    - `reset_for_date(telegram_id, tdee_kkal, *, wib_date) -> DailyEnergy` —
      never modifies an existing balance
    - `record_consumption(...) -> (DailyEnergy, ConsumptionLog)` — write the
      `consumption_log` row and apply the single balance decrement in one commit
    - `reset_all_for_date(*, wib_date=None) -> int` — pre-seed the target WIB
      date for every registered user (scheduler tick); idempotent by PK
  - `ConfirmationStatus(StrEnum)` — `CONFIRMED`, `DECLINED`, `ALREADY_RESOLVED`,
    `STALE`, `NOT_FOUND`
  - `create_pending(...) -> PendingAnalyses`; `get_pending(telegram_id, id)`
  - `confirm_consumption(telegram_id, id, *, now=None) -> (status, record, warned)`
  - `decline_consumption(telegram_id, id, *, now=None) -> status`
- Inline button flow (task-13): callback data is `confirm_ya:<uuid>` /
  `confirm_tidak:<uuid>`; "Ya" → `record_consumption`; "Tidak" → no balance
  change; both idempotent via `pending_analyses.resolved`
- Scheduler (task-13): in-process `AsyncIOScheduler` (`app/services/scheduler.py`),
  Asia/Jakarta cron at 00:00, idempotent by `tanggal_wib`

## Dependencies

- `database-schema` (`daily_energy`, `users`), `tdee-calculation` (TDEE value)
- `telegram-integration` delivers button callbacks (Phase 3)

## Data Flow

```
first activity of WIB day → ensure_today → row seeded sisa = TDEE
analysis → create_pending (deterministic energy/GL + tanggal_wib)
"Ya" → confirm_consumption → record_consumption (log + balance, one commit)
"Tidak" → decline_consumption (resolved only; no balance/log mutation)
midnight WIB → scheduler reset_all_for_date → fresh rows when absent
missed scheduler tick → next ensure_today lazily seeds the new date
```

## Edge Cases

- Double "Ya" (double tap) must reduce only once — idempotency via `pending_analyses.resolved`
- Reset is lazy by `(telegram_id, tanggal_wib)` composite PK: re-seeding an existing
  day never overwrites the balance
- Negative balance allowed with warning (PRD Iterasi 4); strict mode available for
  validation paths
- Unregistered `telegram_id` → `UserNotRegisteredError`
- Amounts rounded to 2 decimals before write
- A callback whose `pending.tanggal_wib` is before today (WIB) is `STALE`:
  no balance or log mutation, and the user is told to send a new food
- Unknown `pending_id` (or one owned by another user) is `NOT_FOUND`: no mutation

## Acceptance Criteria

- [x] Energy CRUD + idempotency unit-tested (`tests/test_energy_service.py`)
- [x] WIB midnight reset idempotency unit-tested (`tests/test_energy_reset.py`)
- [x] Button "Ya" reduces exactly once; "Tidak" does nothing
      (`tests/test_pending_confirmation.py`)
- [x] Midnight-WIB reset verified with an APScheduler job on a real async run
      loop (`tests/test_energy_scheduler.py`)
- [x] Resolved callback buttons cleared after the outcome is sent
      (`tests/test_telegram_handlers.py`)

## Boundaries (module-specific)

**Always:** compute the WIB date via `wib_today`; commit per mutation.
**Ask first:** changing reset semantics (lazy vs scheduler-only) or negative-balance policy.
**Never:** re-seed an existing day's balance; record consumption without an explicit "Ya".
