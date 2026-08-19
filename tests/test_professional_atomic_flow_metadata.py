from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from accounting.contracts.atomic_flow_drilldowns import resolve_flow_cell_spec
from accounting.professional.table_contracts import enrich_professional_table


def test_fuzzy_compatibility_inference_never_opts_row_into_governed_flow() -> None:
    df = pd.DataFrame(
        [
            {"Currency": "ARS", "metric": "Funding / aportes", "2026": 100},
            {
                "Currency": "ARS",
                "metric": "Inquilinos directo a pagar impuestos",
                "2026": 30,
            },
        ]
    )

    out = enrich_professional_table(df, "overview_balance_dashboard")

    assert out["metric_id"].tolist() == [
        "FUND.CONTRIB.TOTAL",
        "FUND.CONTRIB.BY_CHANNEL",
    ]
    assert out["drilldown_cell_id"].tolist() == ["", ""]


def test_explicit_stable_metric_ids_propagate_only_governed_atomic_flows() -> None:
    df = pd.DataFrame(
        [
            {"metric_id": "IS.RENT.TOTAL", "Currency": "ARS", "2026": 100},
            {"metric_id": "IS.OPEX.PROPERTY", "Currency": "ARS", "2026": 40},
            {
                "metric_id": "FUND.CONTRIB.BY_FUNDING_ACTOR",
                "Currency": "ARS",
                "funding_actor": "Matías",
                "2026": 20,
            },
            {"metric_id": "DIST.DRAWS.PERSONAL", "Currency": "ARS", "2026": 10},
            {"metric_id": "TR.FX.CONVERSION.IN", "Currency": "ARS", "2026": 5},
            {
                "metric_id": "FUND.CONTRIB.DIRECT_OBLIGATION",
                "Currency": "ARS",
                "2026": 7,
            },
            {"metric_id": "TR.FX.NET", "Currency": "ARS", "2026": 3},
            {"metric_id": "DIST.DIVIDENDS", "Currency": "ARS", "2026": 1},
        ]
    )

    out = enrich_professional_table(df, "income_operating_statement")

    got = dict(zip(out["metric_id"], out["drilldown_cell_id"], strict=True))
    assert got["IS.RENT.TOTAL"] == "flow.rent.total"
    assert got["IS.OPEX.PROPERTY"] == "flow.property_opex.total"
    assert (
        got["FUND.CONTRIB.BY_FUNDING_ACTOR"]
        == "flow.funding_contribution.by_actor"
    )
    assert got["DIST.DRAWS.PERSONAL"] == "flow.family_draws_or_distributions.total"
    assert got["TR.FX.CONVERSION.IN"] == "flow.fx.conversion_proceeds"

    assert got["FUND.CONTRIB.DIRECT_OBLIGATION"] == ""
    assert got["TR.FX.NET"] == ""
    assert got["DIST.DIVIDENDS"] == ""

    for cell_id in out["drilldown_cell_id"]:
        if cell_id:
            assert resolve_flow_cell_spec(cell_id) is not None


def test_explicit_statement_lines_propagate_without_label_inference() -> None:
    df = pd.DataFrame(
        [
            {
                "statement_line": "operating_revenue",
                "Currency": "ARS",
                "2026-06": 100,
            },
            {
                "statement_line": "property_opex_true",
                "Currency": "ARS",
                "2026-06": 40,
            },
            {
                "statement_line": "family_draws_or_distributions",
                "Currency": "ARS",
                "2026-06": 10,
            },
            {
                "statement_line": "treasury_fx_conversion_out",
                "Currency": "ARS",
                "2026-06": 5,
            },
            {
                "statement_line": "net_operating",
                "Currency": "ARS",
                "2026-06": 60,
            },
        ]
    )

    out = enrich_professional_table(df, "monthly_tables_operating_statement_matrix")
    got = dict(
        zip(out["statement_line"], out["drilldown_cell_id"], strict=True)
    )

    assert got["operating_revenue"] == "flow.operating_revenue"
    assert got["property_opex_true"] == "flow.property_opex.total"
    assert (
        got["family_draws_or_distributions"]
        == "flow.family_draws_or_distributions.total"
    )
    assert got["treasury_fx_conversion_out"] == "flow.fx.conversion_outflow"
    assert got["net_operating"] == ""


@pytest.mark.parametrize(
    ("table_id", "expected"),
    [
        ("monthly_tables_draws_by_box_amount_out", "flow.draws.by_box"),
        ("monthly_tables_draws_by_type_amount_out", "flow.draws.by_type"),
    ],
)
def test_unambiguous_producer_table_contracts_can_supply_cell_family(
    table_id: str,
    expected: str,
) -> None:
    df = pd.DataFrame(
        [{"Currency": "ARS", "Box": "Family Business", "2026-06": 10}]
    )

    out = enrich_professional_table(df, table_id)

    assert out.loc[0, "drilldown_cell_id"] == expected
    assert resolve_flow_cell_spec(expected) is not None


def test_conflicting_structured_metadata_fails_closed_to_blank_id() -> None:
    df = pd.DataFrame(
        [
            {
                "metric_id": "IS.RENT.TOTAL",
                "statement_line": "property_opex_true",
                "Currency": "ARS",
                "2026": 100,
            }
        ]
    )

    out = enrich_professional_table(df, "income_operating_statement")

    assert out.loc[0, "drilldown_cell_id"] == ""


def test_curated_repair_of_stale_metric_id_does_not_reuse_stale_flow_identity() -> None:
    df = pd.DataFrame(
        [
            {
                "Currency": "ARS",
                "metric": "Cobertura después de funding y retiros",
                "metric_id": "FUND.CONTRIB.TOTAL",
                "2026": 150,
            }
        ]
    )

    out = enrich_professional_table(df, "overview_balance_dashboard")

    assert out.loc[0, "metric_id"] == "COV.NET.AFTER_DRAWS"
    assert out.loc[0, "drilldown_cell_id"] == ""


def test_explicit_producer_cell_id_is_validated_and_cannot_conflict() -> None:
    good = pd.DataFrame(
        [
            {
                "drilldown_cell_id": "flow.rent.total",
                "metric_id": "IS.RENT.TOTAL",
                "Currency": "ARS",
                "2026": 100,
            }
        ]
    )
    out = enrich_professional_table(good, "income_operating_statement")
    assert out.loc[0, "drilldown_cell_id"] == "flow.rent.total"

    unknown = good.copy()
    unknown.loc[0, "drilldown_cell_id"] = "flow.future.unknown"
    with pytest.raises(ValueError, match="Unknown atomic-flow drilldown_cell_id"):
        enrich_professional_table(unknown, "income_operating_statement")

    conflict = good.copy()
    conflict.loc[0, "drilldown_cell_id"] = "flow.property_opex.total"
    with pytest.raises(ValueError, match="conflicts with stable producer metadata"):
        enrich_professional_table(conflict, "income_operating_statement")


def test_debt_table_does_not_gain_atomic_flow_metadata() -> None:
    df = pd.DataFrame(
        [
            {
                "Currency": "USD",
                "measure": "open_total",
                "pair": "PM → MI",
                "2025-03": 8726.2,
            }
        ]
    )

    out = enrich_professional_table(df, "monthly_tables_debt_position_matrix")

    assert list(out.columns) == list(df.columns)
    assert "drilldown_cell_id" not in out.columns


def test_pr10b_does_not_wire_professional_executor_to_flow_registry() -> None:
    root = Path(__file__).resolve().parents[1]
    drilldown = (
        root / "accounting" / "professional" / "drilldown.py"
    ).read_text(encoding="utf-8")

    assert "contracts.atomic_flow_drilldowns" not in drilldown
