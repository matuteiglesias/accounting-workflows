from __future__ import annotations

import pandas as pd
import pytest

from accounting.professional import drilldown
from accounting.professional.drilldown import _fx_treasury_measure_for_row
from accounting.professional.fx_drilldown_authority import (
    FX_MEASURES,
    FX_TREASURY_TABLE_IDS,
    _fx_treasury_measure_for_row as authority_measure_for_row,
    resolve_fx_drilldown,
)


@pytest.mark.parametrize(
    ("table_id", "row", "expected"),
    [
        ("monthly_tables_fx_treasury_all_measures", {"measure": "amount_abs"}, "amount_abs"),
        ("monthly_tables_fx_treasury_compact", {"measure": "amount_in"}, "amount_in"),
        ("monthly_tables_fx_treasury_amount_in", {}, "amount_in"),
        ("monthly_tables_fx_treasury_amount_out", {}, "amount_out"),
        (
            "monthly_tables_fx_treasury_amount_out",
            {"metric": "fx_conversion_proceeds"},
            "amount_out",
        ),
        ("monthly_tables_fx_treasury_net_amount", {}, "net_amount"),
        ("monthly_tables_fx_treasury_all_measures", {"metric": "fx_conversion_proceeds"}, "amount_in"),
        ("monthly_tables_fx_treasury_all_measures", {"metric": "fx_conversion_outflow"}, "amount_out"),
        ("monthly_tables_fx_treasury_all_measures", {"metric": "fx_cost_or_spread"}, "amount_out"),
        ("monthly_tables_fx_treasury_all_measures", {"metric": "fx_net"}, "net_amount"),
        ("monthly_tables_fx_treasury_all_measures", {"metric": "amount_abs"}, "amount_abs"),
        ("monthly_tables_fx_treasury_compact", {}, ""),
        ("monthly_tables_fx_treasury_all_measures", {}, ""),
        ("monthly_tables_fx_treasury_all_measures", {"metric": "future_fx_metric"}, ""),
    ],
)
def test_effective_fx_measure_resolution_is_characterized(
    table_id: str, row: dict[str, str], expected: str
) -> None:
    assert _fx_treasury_measure_for_row(table_id, pd.Series(row, dtype=object)) == expected


def test_explicit_measure_precedes_table_and_metric_fallbacks() -> None:
    row = pd.Series(
        {
            "measure": "amount_abs",
            "metric": "fx_conversion_proceeds",
        }
    )
    assert _fx_treasury_measure_for_row(
        "monthly_tables_fx_treasury_amount_out", row
    ) == "amount_abs"


def test_currency_total_and_box_currency_grains_are_explicit() -> None:
    total = resolve_fx_drilldown(
        "monthly_tables_fx_treasury_amount_in",
        pd.Series({"Currency": "ARS"}),
    )
    assert total is not None and total.supported
    assert (total.grain, total.measure, total.currency, total.box) == (
        "currency_total",
        "amount_in",
        "ARS",
        "",
    )

    by_box = resolve_fx_drilldown(
        "monthly_tables_fx_treasury_all_measures",
        pd.Series(
            {
                "Currency": "USD",
                "Box": "Property Management",
                "measure": "amount_out",
            }
        ),
    )
    assert by_box is not None and by_box.supported
    assert (by_box.grain, by_box.measure, by_box.currency, by_box.box) == (
        "box_currency",
        "amount_out",
        "USD",
        "Property Management",
    )


def test_box_required_contract_without_box_fails_closed() -> None:
    resolution = resolve_fx_drilldown(
        "monthly_tables_fx_treasury_all_measures",
        pd.Series(
            {
                "Currency": "ARS",
                "drilldown_cell_id": "flow.fx.conversion_proceeds",
            }
        ),
    )
    assert resolution is not None and not resolution.supported
    assert resolution.measure == "amount_in"
    assert resolution.grain == "box_currency"
    assert "missing Box" in resolution.unsupported_reason


def test_compact_row_without_measure_is_unsupported_not_net() -> None:
    resolution = resolve_fx_drilldown(
        "monthly_tables_fx_treasury_compact",
        pd.Series({"Currency": "ARS"}),
    )
    assert resolution is not None and not resolution.supported
    assert resolution.measure == ""
    assert "no explicit recognized measure" in resolution.unsupported_reason


def test_public_drilldown_uses_the_single_fx_runtime_authority() -> None:
    assert drilldown._fx_treasury_measure_for_row is authority_measure_for_row
    assert drilldown.FX_TREASURY_TABLE_IDS is FX_TREASURY_TABLE_IDS
    assert drilldown.FX_MEASURES is FX_MEASURES
