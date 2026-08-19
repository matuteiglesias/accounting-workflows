from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.professional import drilldown as professional


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "diagnostics" / "derived_metric_authority_inventory_20260819.csv"


def test_inventory_covers_wave5_derived_authority_surface() -> None:
    inventory = pd.read_csv(INVENTORY)
    expected = {
        "statement.net_operating",
        "statement.coverage_after_draws",
        "statement.treasury_fx_net",
        "annual.savings_rate",
        "professional.operating_margin",
        "professional.opex_to_rent",
        "professional.draws_to_operating_result",
        "professional.coverage_after_funding_and_draws",
        "professional.annual_net_operating_rebuild",
        "professional.diagnostic_box_level",
        "annual.debt_net_pm_position",
        "professional.net_flow_observed",
    }
    assert set(inventory["authority_id"]) == expected
    assert set(inventory["pr17_readiness"]) <= {"READY", "BLOCKED", "DEFER"}
    assert set(inventory["decision_required"]) <= {"yes", "no"}


def test_professional_formula_identity_is_currently_bound_to_four_human_labels() -> None:
    expected = {
        "Margen operativo": (
            "operating_margin",
            ("IS.NET.OPERATING", "IS.REVENUE.OPERATING"),
        ),
        "OPEX / renta": (
            "opex_to_rent",
            ("IS.OPEX.PROPERTY", "IS.REVENUE.OPERATING"),
        ),
        "Retiros / resultado operativo": (
            "draws_to_operating_result",
            ("DIST.DRAWS.PERSONAL", "IS.NET.OPERATING"),
        ),
        "Cobertura después de funding y retiros": (
            "coverage_after_funding_and_draws",
            (
                "COV.NET.AFTER_DRAWS",
                "IS.NET.OPERATING",
                "FUND.CONTRIB.TOTAL",
                "DIST.DRAWS.PERSONAL",
            ),
        ),
    }
    for label, (formula_id, components) in expected.items():
        spec = professional._annual_formula_spec(pd.Series({"metric": label}))
        assert spec is not None
        assert spec.formula_id == formula_id
        assert spec.component_metric_ids == components

    assert professional._annual_formula_spec(pd.Series({"metric": "future stable metric"})) is None


def test_professional_ratio_zero_denominator_is_characterized_as_zero_today() -> None:
    annual = pd.DataFrame(
        [
            {"metric_id": "IS.NET.OPERATING", "period": "2026", "Currency": "ARS", "value": 0.0},
            {"metric_id": "IS.REVENUE.OPERATING", "period": "2026", "Currency": "ARS", "value": 0.0},
        ]
    )
    result = professional._build_annual_formula_cell(
        table_id="overview_balance_dashboard",
        row=pd.Series({"metric": "Margen operativo", "Currency": "ARS"}),
        period="2026",
        currency="ARS",
        display_value=0.0,
        annual=annual,
        tolerance=1e-6,
    )
    assert result is not None
    assert result[0] == "ok"
    assert result[1] == 0.0
    assert professional._safe_div(100.0, 0.0) == 0.0


def test_professional_coverage_prefers_source_metric_then_recomputes_with_zero_defaults() -> None:
    row = pd.Series(
        {"metric": "Cobertura después de funding y retiros", "Currency": "ARS"}
    )
    partial = pd.DataFrame(
        [
            {"metric_id": "IS.NET.OPERATING", "period": "2026", "Currency": "ARS", "value": 100.0},
            {"metric_id": "DIST.DRAWS.PERSONAL", "period": "2026", "Currency": "ARS", "value": 30.0},
        ]
    )
    recomputed = professional._build_annual_formula_cell(
        table_id="overview_balance_dashboard",
        row=row,
        period="2026",
        currency="ARS",
        display_value=70.0,
        annual=partial,
        tolerance=1e-6,
    )
    assert recomputed is not None
    assert recomputed[1] == 70.0  # missing funding silently contributes 0 today

    source_first = pd.concat(
        [
            partial,
            pd.DataFrame(
                [
                    {
                        "metric_id": "COV.NET.AFTER_DRAWS",
                        "period": "2026",
                        "Currency": "ARS",
                        "value": 55.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    sourced = professional._build_annual_formula_cell(
        table_id="overview_balance_dashboard",
        row=row,
        period="2026",
        currency="ARS",
        display_value=55.0,
        annual=source_first,
        tolerance=1e-6,
    )
    assert sourced is not None
    assert sourced[1] == 55.0


def test_annual_savings_rate_zero_denominator_is_not_applicable(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    metrics_dir = tmp_path / "metrics"
    run_root.mkdir()
    pd.DataFrame(
        [
            {"period": "2026-01", "Currency": "ARS", "statement_line": "net_operating", "amount": 0.0},
            {"period": "2026-01", "Currency": "ARS", "statement_line": "coverage_after_draws", "amount": 100.0},
        ]
    ).to_csv(run_root / "monthly_operating_statement.csv", index=False)

    paths = build_annual_balance_dashboard(run_root, metrics_dir, "fixture", "2026-01-31")
    metrics = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    savings = metrics[
        metrics["metric_id"].eq("COV.SAVINGS_RATE")
        & metrics["period"].astype(str).str.replace(r"\.0$", "", regex=True).eq("2026")
        & metrics["Currency"].eq("ARS")
    ]
    assert len(savings) == 1
    assert savings.iloc[0]["value_status"] == "not_applicable"
    assert pd.isna(savings.iloc[0]["value"])


def test_diagnostic_box_level_currently_uses_blank_party_fallback_and_zero_missing_prior() -> None:
    cash = pd.DataFrame(
        [
            {
                "period": "2026-02",
                "Currency": "ARS",
                "Box": "Property Management",
                "party": "",
                "position_type": "inferred_box_motor",
                "source_type": "inferred_box_motor",
                "close_amount": 100.0,
            },
            {
                "period": "2026-02",
                "Currency": "ARS",
                "Box": "Property Management",
                "party": "",
                "position_type": "cash_close",
                "source_type": "bank_statement",
                "close_amount": 100.0,
            },
        ]
    )
    result = professional._build_derived_cell(
        table_id="monthly_tables_diagnostic_box_level_matrix",
        row=pd.Series(
            {
                "Currency": "ARS",
                "Box": "Property Management",
                "metric": "diagnostic_box_level",
            }
        ),
        period="2026-02",
        display_value=200.0,
        split=pd.DataFrame(),
        audit=pd.DataFrame(),
        stmt=pd.DataFrame(),
        annual=pd.DataFrame(),
        cash_close=cash,
        debt_activity=pd.DataFrame(),
        debt_position=pd.DataFrame(),
        tolerance=1e-6,
    )
    assert result[0] == "ok"
    assert result[1] == 200.0
    # There is no 2026-01 row. The current implementation therefore treats the
    # previous position as zero, and the blank-party fallback also admits the
    # validated cash row alongside the inferred control row.
    assert result[5]["previous_period"] == "2026-01"


def test_upstream_derived_lines_remain_authoritative_after_pr17_contract() -> None:
    semantic = (ROOT / "accounting" / "marts" / "semantic.py").read_text(encoding="utf-8")
    assert "net_operating = float(op_rev - opex)" in semantic
    assert "coverage_after_draws = float(net_operating + funding - draws)" in semantic
    assert 'add_row(base, "net_operating"' in semantic
    assert 'add_row(base, "coverage_after_draws"' in semantic

    # PR17 adds a declarative contract but no production consumer. The source
    # statement authorities characterized by PR16 remain unchanged.
    contract = ROOT / "accounting" / "contracts" / "derived_metrics.py"
    assert contract.exists()
    consumers: list[str] = []
    for path in (ROOT / "accounting").rglob("*.py"):
        if path == contract:
            continue
        text = path.read_text(encoding="utf-8")
        if "contracts.derived_metrics" in text or "DerivedMetricSpec" in text:
            consumers.append(str(path.relative_to(ROOT)))
    assert consumers == []
