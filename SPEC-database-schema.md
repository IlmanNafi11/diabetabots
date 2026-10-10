# Spec: database-schema

**Status:** Implemented · **Plan ref:** task-02 · **SPEC.md §8:** database-schema

## Objective

Define the four PostgreSQL tables backing user registration, daily energy tracking,
consumption logging, and button idempotency, via SQLAlchemy 2.0 async ORM.

## Public API (database operations)

- `app/database.py`: async `engine` + `async_session` (asyncpg), `get_db()` dependency
- `app/models/models.py`: `Base` (DeclarativeBase), models below; enums `Gender`,
  `ActivityLevel` (StrEnum, stored by **value** via `values_callable`)

### Tables

| Table | PK | Columns |
|---|---|---|
| `users` | `telegram_id` (BigInteger) | `jenis_kelamin` (enum gender), `usia_tahun` (int), `berat_kg`, `tinggi_cm`, `tingkat_aktivitas` (enum activity_level), `pal`, `bmr_kkal`, `tdee_kkal` (float), `created_at` (timestamptz, server now) |
| `daily_energy` | composite (`telegram_id` FK→users, `tanggal_wib` date) | `sisa_energi_kkal` (float) |
| `consumption_log` | `id` (UUID v4) | `telegram_id` (FK, indexed), `nama_pangan` (text), `gram`, `energi_kkal`, `gl_aktual` (nullable float), `consumed_at` (timestamptz, server now), `analysis_message_id` (nullable BigInteger) |
| `pending_analyses` | `id` (UUID v4) | `telegram_id` (FK, indexed), `nama_pangan`, `gram`, `energi_kkal`, `gl_aktual` (nullable), `tanggal_wib` (date), `created_at` (server now), `resolved` (bool, default false) |

## Dependencies

- PostgreSQL 15+ (Mode A: compose service; Mode B: external per D13)
- `sqlalchemy[asyncio]`, `asyncpg`; connection config via `app/config.py` (pydantic-settings)

## Data Flow

- Registration completes → one `users` row with persisted `pal`/`bmr_kkal`/`tdee_kkal`
- First activity of a WIB day → `daily_energy` row seeded `sisa = TDEE`
  (composite PK makes reset idempotent)
- Button "Ya" → insert `consumption_log`, mark matching `pending_analyses.resolved = true`
- Button "Tidak" → mark `pending_analyses.resolved = true`, no log row

## Edge Cases

- Re-seeding an existing `(telegram_id, tanggal_wib)` never overwrites `sisa_energi_kkal`
- Double-tap on an inline button is absorbed by the `resolved` flag
- `gl_aktual` nullable: foods without GL data log energy only
- Enum values are stored as their string values (`"Laki-laki"`, `"Sangat Berat"`), not names

## Acceptance Criteria

- [x] All 4 tables creatable via `Base.metadata.create_all`
- [x] Composite PK on `daily_energy` enforces one row per user per WIB date
- [x] FKs from `daily_energy`, `consumption_log`, `pending_analyses` → `users.telegram_id`
- [x] Migration tooling decision recorded (create_all at startup for thesis scope; no
      schema changes after task-02 without Ask First per SPEC.md §7.2)

## Boundaries (module-specific)

**Ask first:** any add/rename column, new table, or type change (SPEC.md §7.2).
**Never:** store credentials in the DB or connection string in code.
