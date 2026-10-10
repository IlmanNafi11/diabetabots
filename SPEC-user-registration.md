# Spec: user-registration

**Status:** Bot foundations implemented; DB persistence wired in Phase 2 (task-11) ·
**Plan ref:** task-04 / task-11 · **SPEC.md §8:** user-registration

## Objective

Five-step interactive Telegram onboarding (gender → age → weight → height → activity)
that validates each input, computes BMR/TDEE deterministically, and persists the
profile in `users`.

## Public API

- `app/bot/conversation.py`
  - `ConversationState` (StrEnum): `IDLE, AWAIT_GENDER, AWAIT_AGE, AWAIT_WEIGHT,
    AWAIT_HEIGHT, AWAIT_ACTIVITY, CONFIRMED, AWAIT_GRAM_CLARIFY`
  - `RegistrationProfile` — collected values before persistence
  - `ConversationManager`: `get, clear, start_registration, accept_gender, accept_age,
    accept_weight, accept_height, accept_activity, request_gram_clarification, accept_gram`
- `app/bot/validators.py`
  - `parse_gender(raw) -> Gender` — accepts laki-laki/laki/pria/l, perempuan/wanita/p
  - `parse_age(raw) -> int` — integer, **18–100 inclusive**
  - `parse_weight(raw) -> float` — **25–300 kg**
  - `parse_height(raw) -> float` — **100–250 cm**
  - `parse_activity(raw) -> ActivityLevel` — 1–5 or free-text keyword
  - `parse_gram(raw) -> float` — positive grams (mass portions only)
  - all raise `ValidationError` (subclass of `ValueError`) with Indonesian user-facing messages
- Persistence (Phase 2, task-11): upsert `users` row; derive `pal`, `bmr_kkal`, `tdee_kkal`
  via `app/services/tdee_service.py`

## Dependencies

- `database-schema` (users table), `tdee-calculation` (BMR/TDEE math)
- Telegram wiring (`telegram-integration`, Phase 3) supplies messages; this module owns
  state + validation only

## Data Flow

```
/start → AWAIT_GENDER → AWAIT_AGE → AWAIT_WEIGHT → AWAIT_HEIGHT → AWAIT_ACTIVITY
      → CONFIRMED (+ users row upserted with derived pal/bmr/tdee)
Invalid input → ValidationError → re-prompt, state unchanged
```

## Edge Cases

- Invalid input at any step re-prompts without advancing state
- `/start` during registration restarts the flow from gender
- Unordered input (e.g. age while `AWAIT_GENDER`) → `ValueError` from `_require_state`
- Registration is a fresh gate: a query from an unregistered chat re-enters onboarding
- Gram clarification (`AWAIT_GRAM_CLARIFY`) is a separate branch from registration and
  requires `gram > 0`

## Acceptance Criteria

- [ ] 5-step interactive onboarding works end-to-end in Telegram (verified in Phase 3)
- [ ] Profile data persists in `users` (verified by task-11 integration test)
- [x] State machine transitions unit-tested (`tests/test_conversation.py`, 177-test
      foundation suite green)
- [x] All validation ranges unit-tested (`tests/test_validators.py`)

## Boundaries (module-specific)

**Always:** validate before persisting; keep prompts in Indonesian.
**Never:** assume a gram value for non-mass portions — clarify (SPEC.md §7.3);
generate medical advice during onboarding.
