"""Contract tests for the task-15 E2E evaluation harness."""

from app.evaluation.e2e import (
    E2E_SCENARIOS,
    OutboundCall,
    ScenarioResult,
    evaluate_scenario,
    render_e2e_report,
)


def test_evaluator_requires_exactly_once_confirmation_and_no_error() -> None:
    scenario = E2E_SCENARIOS[1]
    calls = [
        OutboundCall(
            method="send_message",
            chat_id=1,
            text="✅ Registrasi selesai!",
        ),
        OutboundCall(
            method="send_message",
            chat_id=1,
            text="Berapa gram untuk makanan?",
        ),
        OutboundCall(
            method="send_message_with_inline_buttons",
            chat_id=1,
            text="Analisis",
        ),
        OutboundCall(method="send_message", chat_id=1, text="✅ Tercatat."),
    ]
    result = evaluate_scenario(
        scenario,
        calls,
        {
            "balance_before": 1000.0,
            "balance_after": 800.0,
            "balance_after_double": 800.0,
            "energy_kkal": 200.0,
            "error": None,
        },
    )

    assert result.passed
    assert all(check.startswith("PASS") for check in result.checks)


def test_evaluator_rejects_handler_error_even_when_analysis_exists() -> None:
    scenario = E2E_SCENARIOS[0]
    result = evaluate_scenario(
        scenario,
        [
            OutboundCall(
                method="send_message",
                chat_id=1,
                text="✅ Registrasi selesai!",
            ),
            OutboundCall(
                method="send_message_with_inline_buttons",
                chat_id=1,
                text="Analisis",
            ),
        ],
        {"error": "RuntimeError: LLM unavailable"},
    )

    assert not result.passed
    assert any(check.startswith("FAIL") and "galat" in check for check in result.checks)


def test_evaluator_rejects_wrong_top_food_for_grounded_scenario() -> None:
    scenario = E2E_SCENARIOS[3]
    calls = [
        OutboundCall(method="send_message", chat_id=1, text="✅ Registrasi selesai!"),
        OutboundCall(method="send_message_with_inline_buttons", chat_id=1, text="Analisis"),
    ]
    result = evaluate_scenario(
        scenario,
        calls,
        {
            "top1_name": "Kool kembang",
            "balance_before": 1000.0,
            "balance_after": 975.0,
            "balance_after_double": 975.0,
            "energy_kkal": 25.0,
            "error": None,
        },
    )

    assert not result.passed
    assert any(check.startswith("FAIL") and "pangan teratas" in check for check in result.checks)


def test_e2e_catalog_matches_the_ten_prd_scenarios() -> None:
    """The audit runner must keep all ten PRD §7.3 cases explicit."""
    assert [scenario.number for scenario in E2E_SCENARIOS] == list(range(1, 11))
    assert len({scenario.category for scenario in E2E_SCENARIOS}) == 5
    assert all(
        scenario.profile in {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"}
        for scenario in E2E_SCENARIOS
    )


def test_e2e_report_contains_raw_log_links_and_result_table() -> None:
    """The markdown artifact must identify every scenario and its raw evidence."""
    results = [
        ScenarioResult(
            number=scenario.number,
            profile=scenario.profile,
            category=scenario.category,
            description=scenario.description,
            passed=True,
            checks=("registration", "analysis"),
            messages=("✅ Registrasi selesai!",),
            error=None,
        )
        for scenario in E2E_SCENARIOS
    ]

    report = render_e2e_report(results, generated_at="2026-10-11T00:00:00+00:00")

    assert "10 skenario" in report
    assert "| 1 |" in report
    assert "| 10 |" in report
    assert "reports/logs/e2e-scenario-01.log" in report
    assert "✅ Lulus: 10/10" in report
