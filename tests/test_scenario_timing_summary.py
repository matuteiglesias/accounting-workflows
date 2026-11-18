from __future__ import annotations

from accounting.diagnostics.summarize_scenario_timings import render_html, summarize


def test_summary_keeps_stage_and_module_timings_separate() -> None:
    data = {
        "schema": "accounting.scenario.perf.v1", "scenario_id": "pilot",
        "total_seconds": 34,
        "variants": [{
            "variant": "base", "run_id": "EXAMPLE", "total_seconds": 34,
            "events": [
                {"kind": "make_stage", "stage": "run-debt", "elapsed_seconds": 30, "exit_code": 0},
                {"kind": "python_subprocess", "command": ["-m", "accounting.debt.balance_views"],
                 "elapsed_seconds": 28, "exit_code": 0},
                {"kind": "renderer", "stage": "treasury_html", "elapsed_seconds": 2},
                {"kind": "renderer", "stage": "treasury_pdf", "elapsed_seconds": 2},
            ],
        }],
    }
    stages, commands = summarize(data)
    assert len(stages) == 3 and len(commands) == 1
    assert commands[0]["operation"] == "accounting.debt.balance_views"
    html = render_html(data)
    assert "run-debt" in html
    assert "accounting.debt.balance_views" in html
    assert "do not add both totals" in html
    assert "<h1>Accounting scenario" in html
