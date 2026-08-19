from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.marts.semantic import build_monthly_operating_statement_from_split
from accounting.professional.drilldown import _semantic_filter_for_statement_line


ROOT = Path(__file__).resolve().parents[1]
SPLIT_FIXTURE = ROOT / "fixtures" / "semantic_measure_statement_input.csv"

ATOMIC_STATEMENT_LINES = {
    "operating_revenue",
    "rent_revenue",
    "property_opex_true",
    "funding_contributions",
    "family_draws_or_distributions",
    "treasury_fx_conversion_in",
    "treasury_fx_conversion_out",
    "treasury_fx_cost",
}


def test_atomic_drilldowns_reconcile_exactly_to_native_statement_values() -> None:
    split = pd.read_csv(SPLIT_FIXTURE)
    statement, _ = build_monthly_operating_statement_from_split(split)
    atomic_statement = statement.loc[
        statement["statement_line"].isin(ATOMIC_STATEMENT_LINES)
    ]

    for row in atomic_statement.itertuples(index=False):
        spec = _semantic_filter_for_statement_line(row.statement_line)
        assert spec is not None
        measure, membership = spec
        group = split.loc[
            split["period"].eq(row.period) & split["Currency"].eq(row.Currency)
        ]
        matched = pd.to_numeric(
            group.loc[membership(group), measure], errors="coerce"
        ).fillna(0.0).sum()
        assert matched == row.amount, (
            row.period,
            row.Currency,
            row.statement_line,
            measure,
            matched,
            row.amount,
        )
