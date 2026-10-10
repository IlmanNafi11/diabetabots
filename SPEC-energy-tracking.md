# Spec: energy-tracking

**Status:** Service implemented; Telegram button wiring in Phase 3 (task-13) ·
**Plan ref:** task-04/task-05 (service + unit), task-13 (integration) ·
**SPEC.md §8:** energy-tracking

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
- Inline button flow (task-13): "Ya" → `reduce_energy` + `consumption_log` insert;
  "Tidak" → no balance change; both idempotent via `pending_analyses.resolved`
- Scheduler (Phase 3): in-process APScheduler, Asia/Jakarta, idempotent by `tanggal_wib`

## Dependencies

- `database-schema` (`daily_energy`, `users`), `tdee-calculation` (TDEE value)
- `telegram-integration` delivers button callbacks (Phase 3)

## Data Flow

```
first activity of WIB day → ensure_today → row seeded sisa = TDEE
"Ya" → reduce_energy (−energi_kkal, round 2) → commit → [warned if balance < 0]
"Tidak" → no-op on balance
midnight WIB → next ensure_today targets new tanggal_wib → fresh seed at TDEE
```

## Edge Cases

- Double "Ya" (double tap) must reduce only once — idempotency via `pending_analyses.resolved`
- Reset is lazy by `(telegram_id, tanggal_wib)` composite PK: re-seeding an existing
  day never overwrites the balance
- Negative balance allowed with warning (PRD Iterasi 4); strict mode available for
  validation paths
- Unregistered `telegram_id` → `UserNotRegisteredError`
- Amounts rounded to 2 decimals before write

## Acceptance Criteria

- [x] Energy CRUD + idempotency unit-tested (`tests/test_energy_service.py`)
- [x] WIB midnight reset idempotency unit-tested (`tests/test_energy_reset.py`)
- [ ] Button "Ya" reduces exactly once; "Tidak" does nothing (task-13, Checkpoint 3)
- [ ] Midnight-WIB reset verified with scheduler in run loop (task-13)

## Boundaries (module-specific)

**Always:** compute the WIB date via `wib_today`; commit per mutation.
**Ask first:** changing reset semantics (lazy vs scheduler-only) or negative-balance policy.
**Never:** re-seed an existing day's balance; record consumption without an explicit "Ya".
