# Spec: telegram-integration

**Status:** Implemented (polling + registration + task-13 confirmation) ·
**Plan ref:** task-12/task-13 · **SPEC.md §8:** telegram-integration

## Objective

Connect Telegram long polling to the DiabetaBot conversation state machine,
registration services, grounded nutrition analysis, and DB-backed energy
confirmation. Every user-facing string is Bahasa Indonesia. The callback flow
must be safe under retries and double taps.

## Public API

  - `TelegramUpdate.from_dict(raw)` — parses `message` and `callback_query`
    updates, extracting `chat_id`, `user_id`, `text`, `callback_data`,
    `callback_query_id`, and `message_id`.
  - `TelegramService.start_polling` — runs the long-polling loop on a single
    worker-thread event loop for its whole lifetime; per-update handlers are
    awaited on that same loop. Retry/backoff and update-id offset tracking
    included.
  - `TelegramService.__repr__` — redacted; never leaks the bot token.
  - `send_message`, `send_message_with_inline_buttons`, and
    `edit_message_reply_markup` — outbound text, keyboard, and keyboard-clear
    calls.
- `app/bot/handlers.py`
  - `TelegramUpdateHandler.handle_update` — routes callbacks vs messages.
  - `build_confirm_keyboard(pending_id)` — builds the Ya/Tidak keyboard with
    `confirm_ya:<uuid>` / `confirm_tidak:<uuid>` callback data.
  - `_analyze_and_send` — retrieval → narrative → deterministic payload →
    `pending_analyses` insert → message with buttons.
  - `_handle_confirm_ya` / `_handle_confirm_tidak` — resolve the pending row,
    branch on `ConfirmationStatus`, and clear the resolved keyboard.
- `app/bot/conversation.py` — state machine, including `AWAIT_GRAM_CLARIFY`.

## Dependencies

- `llm-analysis` for the grounded narrative and deterministic payload.
- `energy-tracking` for `pending_analyses`, `daily_energy`, and
  `consumption_log`.
- `user-registration` and `tdee-calculation` for profile persistence and TDEE.
- `database-schema` for the async session factory used per request/tick.

## Data Flow

```
getUpdates → TelegramUpdate
  ├─ message  → state machine → validate → service call → prompt / analysis
  └─ callback → parse "confirm_ya:<uuid>" → EnergyService confirmation
                                            → balance/log mutation + ack
                                            → clear resolved keyboard
```

Registration advances `IDLE → AWAIT_GENDER → AWAIT_AGE → AWAIT_WEIGHT →
AWAIT_HEIGHT → AWAIT_ACTIVITY → CONFIRMED`; non-gram portions route through
`AWAIT_GRAM_CLARIFY` before analysis.

## Edge Cases

- Unknown, malformed, or foreign callback data → rejected with a Bahasa
  Indonesia notice; no DB mutation.
- Double tap → `pending_analyses.resolved` yields `ALREADY_RESOLVED`; the
  balance is reduced exactly once.
- Cross-day callback → `STALE`; no balance change.
- Unknown pending row → `NOT_FOUND`; no balance change.
- Declined ("Tidak") → row marked resolved with no `consumption_log` row.
- Database failure during confirmation → logged, user told to retry; no partial
  balance/log state (single commit per confirmation).
- Polling/HTTP errors → retry with backoff; a failed update never aborts the
  loop.
- A chat exceeding the per-chat rate limit (default 10 analysis/min) receives
  a Bahasa Indonesia slow-down reply; no analysis or DB work is started.
- Over-long questions (above `MAX_USER_QUESTION_CHARS`, default 512) are
  rejected with a short refusal; the LLM is never called.
- Logs and reprs never contain the bot token (`TelegramService.__repr__` is
  redacted).

## Acceptance Criteria

- [x] Long-polling engine with offset tracking and retry/backoff.
- [x] Registration flow and `/start` restart handled.
- [x] "Ya" reduces exactly once and logs the consumption; "Tidak" does nothing.
- [x] Callback idempotency via `pending_analyses.resolved` (task-13).
- [x] Resolved analysis messages no longer offer a second tap.
- [ ] 10 E2E scenarios (task-15) and the final verification report (task-16).

## Boundaries

**Always:** answer every callback with an explicit Bahasa Indonesia outcome;
resolve a tap against the persisted pending payload; keep one commit per
confirmation.

**Ask first:** adding Telegram API methods, changing callback-data encoding, or
altering the state machine's states/transitions.

**Never:** act on a callback without a matching pending row; mutate the balance
on "Tidak"; process the same resolved callback twice.
