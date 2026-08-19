from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from accounting.professional.drilldown import _fx_treasury_measure_for_row


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "accounting" / "professional" / "drilldown.py"


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
        ("monthly_tables_fx_treasury_compact", {}, "net_amount"),
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


def test_shadowed_fx_resolver_and_constants_are_gone() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fx_treasury_measure_for_row"
    ]
    assigned_names = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]

    assert len(functions) == 1
    assert assigned_names.count("FX_TREASURY_TABLE_IDS") == 1
    assert assigned_names.count("FX_MEASURES") == 1
