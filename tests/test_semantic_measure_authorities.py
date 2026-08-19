from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from accounting.management.usd_ccl_flows import _measure_direction
from accounting.marts.semantic import build_monthly_operating_statement_from_split
from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.professional.drilldown import (
    _cash_bridge_line_spec,
    _fx_treasury_measure_for_row,
    _semantic_filter_for_statement_line,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "semantic_measure_authorities_20260819.csv"


def _matrix() -> dict[str, dict[str, str]]:
    with MATRIX.open(encoding="utf-8", newline="") as handle:
        return {row["semantic_concept"]: row for row in csv.DictReader(handle)}


def _split() -> pd.DataFrame:
    concepts = [
        ("rent", "operating_revenue", "rent", 11, 0, 11, 11, False),
        ("opex_taxes", "property_opex", "taxes", 0, 12, -12, 12, False),
        ("opex_services", "property_opex", "services", 0, 13, -13, 13, False),
        ("opex_maintenance", "property_opex", "maintenance", 0, 14, -14, 14, False),
        ("opex_legal", "property_opex", "legal", 0, 15, -15, 15, False),
        ("funding", "funding_contribution", "family_or_tenant_contribution", 16, 0, 16, 16, False),
        ("withdrawals", "family_withdrawal_candidate", "personal_expense", 0, 17, -17, 17, False),
        ("debt_principal", "debt_movement", "principal", 18, 0, 18, 18, False),
        ("debt_repayment", "debt_movement", "repayment", 0, 19, -19, 19, False),
        ("internal_transfer", "internal_transfer", "transfer", 0, 0, 0, 20, False),
        ("fx_proceeds", "treasury_fx", "fx_conversion_proceeds", 21, 0, 21, 21, False),
        ("fx_outflow", "treasury_fx", "fx_conversion_outflow", 0, 22, -22, 22, False),
        ("fx_cost", "treasury_fx", "fx_cost_or_spread", 0, 23, -23, 23, False),
        ("unknown_fx", "treasury_fx", "unapproved_future_fx", 0, 24, -24, 24, False),
        ("review_required", "unknown", "ambiguous", 0, 25, -25, 25, True),
    ]
    return pd.DataFrame(
        [
            {
                "case": case,
                "period": "2026-01",
                "period_end": "2026-01-31",
                "Currency": "ARS",
                "semantic_bucket": bucket,
                "semantic_subbucket": subbucket,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "net_amount": net_amount,
                "amount_abs": amount_abs,
                "n_tx": 1,
                "review_required": review,
            }
            for case, bucket, subbucket, amount_in, amount_out, net_amount, amount_abs, review in concepts
        ]
    )


def test_matrix_has_requested_cases_and_measured_edit_distance() -> None:
    rows = _matrix()
    assert set(rows) == {
        "rent",
        "opex_taxes",
        "opex_services",
        "opex_maintenance",
        "opex_legal",
        "funding",
        "withdrawals",
        "debt_principal",
        "debt_repayment",
        "internal_transfer",
        "fx_proceeds",
        "fx_outflow",
        "fx_cost",
        "unknown_fx",
        "review_required",
    }
    assert all(
        row["semantic_edit_distance"] == row["production_authority_count"]
        for row in rows.values()
    )
    assert {rows[name]["semantic_edit_distance"] for name in [
        "rent", "opex_taxes", "opex_services", "opex_maintenance",
        "opex_legal", "funding", "withdrawals", "fx_proceeds",
        "fx_outflow", "fx_cost",
    ]} == {"1"}


def test_native_statement_measures_and_review_fallback_are_frozen() -> None:
    statement, _ = build_monthly_operating_statement_from_split(_split())
    by_line = statement.set_index("statement_line")

    expected = {
        "rent_revenue": (11, "amount_in"),
        "taxes": (12, "amount_out"),
        "services": (13, "amount_out"),
        "maintenance": (14, "amount_out"),
        "legal": (15, "amount_out"),
        "funding_contributions": (16, "amount_in"),
        "family_draws_or_distributions": (17, "amount_out"),
        "debt_movements": (37, "amount_abs"),
        "internal_transfers": (20, "amount_abs"),
        "treasury_fx_conversion_in": (21, "amount_in"),
        "treasury_fx_conversion_out": (22, "amount_out"),
        "treasury_fx_cost": (23, "amount_out"),
        "unknown_or_ambiguous_outflows": (25, "amount_out"),
    }
    for line, (amount, measure) in expected.items():
        assert by_line.loc[line, "amount"] == amount
        assert measure in by_line.loc[line, "source_filter"]
    assert by_line.loc["treasury_fx_net", "amount"] == -48

    fallback = _split()
    fallback.loc[fallback["case"].eq("review_required"), "amount_out"] = 0
    fallback.loc[fallback["case"].eq("review_required"), "net_amount"] = 25
    fallback_statement, _ = build_monthly_operating_statement_from_split(fallback)
    fallback_by_line = fallback_statement.set_index("statement_line")
    assert fallback_by_line.loc["unknown_or_ambiguous_outflows", "amount"] == 25


def test_management_and_professional_selectors_match_characterization() -> None:
    rows = _matrix()
    for concept in ["rent", "opex_taxes", "opex_services", "opex_maintenance", "opex_legal", "funding", "withdrawals", "fx_proceeds", "fx_outflow", "fx_cost"]:
        row = rows[concept]
        semantic = {
            "semantic_bucket": row["semantic_bucket"],
            "semantic_subbucket": row["semantic_subbucket"].replace("*", ""),
        }
        assert _measure_direction(semantic) == row["management_measure"].removeprefix("amount_")

    assert _measure_direction({"semantic_bucket": "debt_movement", "semantic_subbucket": "principal"}) == ""
    assert _measure_direction({"semantic_bucket": "internal_transfer", "semantic_subbucket": "transfer"}) == ""
    assert _measure_direction({"semantic_bucket": "treasury_fx", "semantic_subbucket": "unapproved_future_fx"}) == ""

    line_measures = {
        "rent_revenue": "amount_in",
        "property_opex_true": "amount_out",
        "funding_contributions": "amount_in",
        "family_draws_or_distributions": "amount_out",
        "treasury_fx_conversion_in": "amount_in",
        "treasury_fx_conversion_out": "amount_out",
        "treasury_fx_cost": "amount_out",
        "treasury_fx_net": "net_amount",
    }
    for line, expected in line_measures.items():
        spec = _semantic_filter_for_statement_line(line)
        assert spec is not None and spec[0] == expected

    assert _semantic_filter_for_statement_line("debt_movements") is None
    assert _semantic_filter_for_statement_line("internal_transfers") is None
    assert _semantic_filter_for_statement_line("unknown_or_ambiguous_outflows")[0] == "amount_abs"

    for metric, expected in {
        "fx_conversion_proceeds": "amount_in",
        "fx_conversion_outflow": "amount_out",
        "fx_cost_or_spread": "amount_out",
        "future_unknown_fx": "",
    }.items():
        assert _fx_treasury_measure_for_row(
            "monthly_tables_fx_treasury_all_measures", pd.Series({"metric": metric})
        ) == expected

    for line, expected in {
        "movimiento_neto_deuda": "net_amount",
        "renta": "amount_in",
        "funding_contribuciones": "amount_in",
        "opex_propiedad": "amount_out",
        "retiros_gasto_familiar": "amount_out",
    }.items():
        spec = _cash_bridge_line_spec(line)
        assert spec is not None and spec[0] == expected


def test_annual_metrics_delegate_atomic_measures_to_upstream_or_contract() -> None:
    source = (ROOT / "accounting" / "metrics" / "annual.py").read_text(encoding="utf-8")
    assert "resolve_semantic_measure" in source
    assert 'rows[measure]' in source
    assert '"IS.RENT.TOTAL",s.semantic_bucket.eq("operating_revenue")&s.semantic_subbucket.eq("rent"),"amount_in"' not in source
    assert '"IS.OPEX.BY_CATEGORY",s.semantic_bucket.eq("property_opex"),"amount_out"' not in source
    assert '"FUND.CONTRIB.BY_ACTOR",s.semantic_bucket.eq("funding_contribution"),"amount_in"' not in source
    assert '"DIST.DRAWS.BY_TYPE",s.semantic_bucket.eq("family_withdrawal_candidate"),"amount_out"' not in source
    assert 'groupby(["period","Currency",dim],dropna=False)["net_amount"]' in source
    assert '"treasury_fx_conversion_in": "TR.FX.CONVERSION.IN"' in source
    assert '"treasury_fx_conversion_out": "TR.FX.CONVERSION.OUT"' in source
    assert '"treasury_fx_cost": "TR.FX.COST.OUT"' in source
    assert '"treasury_fx_net": "TR.FX.NET"' in source
    assert '"ID.DEBT.ACTIVITY.NEW_CLAIMS"' in source
    assert '"ID.DEBT.ACTIVITY.REPAYMENTS"' in source


def test_annual_atomic_detail_matches_characterized_monthly_measures(
    tmp_path: Path,
) -> None:
    split = _split()
    split["Box"] = "Property Management"
    split["Lugar"] = "CABA"
    split["actor"] = "Matías"
    statement, _ = build_monthly_operating_statement_from_split(split)
    run_root = tmp_path / "run"
    metrics_dir = tmp_path / "metrics"
    run_root.mkdir()
    split.to_csv(run_root / "monthly_flow_semantic_split.csv", index=False)
    statement.to_csv(run_root / "monthly_operating_statement.csv", index=False)

    paths = build_annual_balance_dashboard(
        run_root, metrics_dir, run_id="semantic-parity", as_of_date="2026-08-19"
    )
    annual = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    available = annual[annual["value_status"].eq("available")]

    def values(metric_id: str) -> dict[str, float]:
        rows = available[available["metric_id"].eq(metric_id)]
        return dict(zip(rows["dimension_value"].fillna(""), rows["value"].astype(float)))

    assert values("IS.RENT.TOTAL")[""] == 11
    assert values("IS.RENT.BY_PROPERTY")["CABA"] == 11
    assert values("IS.OPEX.BY_CATEGORY") == {
        "legal": 15,
        "maintenance": 14,
        "services": 13,
        "taxes": 12,
    }
    assert values("FUND.CONTRIB.BY_ACTOR")["Matías"] == 16
    assert values("DIST.DRAWS.BY_TYPE")["personal_expense"] == 17
    assert values("TR.FX.BY_BOX")["Property Management"] == -48
    assert values("TR.FX.BY_TYPE") == {
        "fx_conversion_outflow": -22,
        "fx_conversion_proceeds": 21,
        "fx_cost_or_spread": -23,
        "unapproved_future_fx": -24,
    }
