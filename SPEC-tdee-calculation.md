# Spec: tdee-calculation

**Status:** Implemented · **Plan ref:** task-04/task-05 (unit), task-11 (8-profile
integration) · **SPEC.md §8:** tdee-calculation

## Objective

Deterministic (no LLM) BMR and TDEE from the user's physiological profile using
Mifflin-St Jeor + PAL, as the energy budget consumed by `energy-tracking`.

## Public API

- `app/services/tdee_service.py`
  - `PAL_OPTIONS: dict[ActivityLevel, float]` —
    Sedentari 1.2 · Ringan 1.375 · Sedang 1.55 · Berat 1.725 · SangatBerat 1.9
  - `pal_for_activity(activity: ActivityLevel) -> float`
  - `calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: Gender) -> float`
    — male: `10w + 6.25h − 5a + 5`; female: `10w + 6.25h − 5a − 161`
  - `calculate_tdee(bmr: float, pal: float) -> float` — `round(bmr * pal, 2)`
- Registration persists `pal`, `bmr_kkal`, `tdee_kkal` on the `users` row

## Dependencies

- `database-schema` (enum types + users columns)
- Consumed by `user-registration` (persist) and `energy-tracking` (seed `sisa_energi_kkal`)

## Data Flow

```
validated profile → calculate_bmr → calculate_tdee(bmr, pal_for_activity(activity))
    → users.tdee_kkal → daily_energy seed / "Sisa energi" responses
```

## Edge Cases

- BMR is **not** rounded; TDEE is rounded to 2 decimals (matches reference values)
- Activity is a closed enum — no unknown multipliers possible
- Validation ranges (age 18–100, weight 25–300, height 100–250) are enforced upstream
  by `validators`, not here

## Acceptance Criteria

- [x] Formula unit-tested with branch coverage on gender (`tests/test_tdee_service.py`,
      `tests/test_gl_calculation.py` for related math)
- [ ] 8 simulation profiles (P1–P8, PRD §4) match reference ≤0.5% —
      `tests/test_integration/test_tdee_8profiles.py` (task-11)
- [ ] Run log + report stored (`reports/logs/tdee-raw.log`, laporan) — task-11

## Boundaries (module-specific)

**Always:** keep the math in pure Python functions, never delegate to the LLM.
**Never:** change the formula or PAL table without updating SPEC.md §5.3 first.
