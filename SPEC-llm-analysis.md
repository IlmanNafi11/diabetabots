# Spec: llm-analysis

**Status:** Implemented · **Plan ref:** task-10 · **SPEC.md §8:** llm-analysis

## Objective

Produce a Bahasa Indonesia nutrition analysis grounded in the retrieved TKPI 2020
and Atkinson et al. (2021) food records. The LLM supplies the narrative only;
energy, nutrition scaling, glycemic load, categories, and the daily-energy
values are deterministic application data.

## Public API

- `app/services/llm_service.py`
  - `NutritionResult(narrative)` — immutable LLM result.
  - `UserProfile` — the profile fields injected into the prompt.
  - `build_system_prompt() -> str` — four-part system prompt with PERKENI rules,
    dataset/profile placeholders, and the §3.5 response contract.
  - `build_user_prompt(food_context, user_profile, remaining_energy_kkal,
    user_question) -> str` — prompt containing the retrieved context and
    remaining budget.
  - `LLMService(settings).analyze_food(food_name, gram, retrieved_foods,
    user_profile, remaining_energy_kkal, user_question) -> NutritionResult` —
    grounded async analysis; an empty retrieval raises `NoFoodRetrievedError`.
- `app/services/nutrition_service.py`
  - `nutrition_for_gram(value_per_100g, gram) -> float` — proportional scaling,
    rounded to two decimals.
  - `glycemic_load(gl_value, gram, serving_size_g) -> float` — proportional GL,
    rounded to two decimals.
  - `gi_category` and `gl_category` — threshold classification.
- `app/bot/responses.py`
  - `format_analysis(AnalysisResponse) -> str` — renders the stable §3.5 message.

## Dependencies

- `rag-retrieval` supplies `RetrievedFood` records and their source values.
- `tdee-calculation` supplies the user's TDEE.
- `energy-tracking` supplies the current WIB daily balance and stores the
  verified energy/GL payload behind confirmation callbacks.
- The configured OpenAI-compatible chat endpoint is used only for narrative
  generation; credentials are validated at service construction.

## Data Flow

```
food query + gram
  → RAG top-k RetrievedFood records
  → deterministic nutrition/GL values from TKPI fields
  → four-part grounded prompt (dataset + profile + remaining energy)
  → LLM narrative in Bahasa Indonesia
  → Telegram analysis message + DB-backed Ya/Tidak confirmation
```

The handler persists `pending_analyses` from the deterministic first retrieved
food before sending the keyboard. A callback therefore uses the exact values
shown for that analysis and never recalculates them from a later retrieval.

## Edge Cases

- No retrieved food → `NoFoodRetrievedError`/user-facing not-found response;
  the LLM is not called.
- Missing energy source value → the handler does not create a confirmable
  pending row.
- Missing GI or serving size → GL is `None`; the prompt/message states that
  the source does not provide enough information.
- LLM credentials are absent or sentinel values → construction fails clearly;
  no fake analysis is returned.
- Narrative numbers are not trusted for balance arithmetic.

## Acceptance Criteria

- [x] Four-part grounded prompt and §3.5 response format implemented.
- [x] Nutrition and GL formulas are deterministic and unit-tested.
- [x] Handler stores deterministic energy/GL values in `pending_analyses`.
- [x] Integration nutrition accuracy test covers TKPI-grounded output (network
      credentials required; excluded from the deterministic unit gate).

## Boundaries

**Always:** use retrieved source values for numeric nutrition and GL; pass the
remaining WIB energy explicitly; keep all user-facing narrative in Bahasa
Indonesia.

**Ask first:** changing the §3.5 response contract, clinical rules, retrieval
source, or deterministic formula semantics.

**Never:** let the LLM invent numeric nutrition values, record consumption
without an explicit confirmation, or silently substitute an ungrounded food
record.
