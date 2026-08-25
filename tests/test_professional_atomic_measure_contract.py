from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.contracts.atomic_flow_drilldowns import resolve_flow_cell_spec
from accounting.contracts.semantic_measures import resolve_semantic_measure
from accounting.marts.semantic import build_monthly_operating_statement_from_split
from accounting.professional.table_contracts import EXPLICIT_STATEMENT_LINE_FLOW_CELL_IDS


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


def _members_mask(frame: pd.DataFrame, cell_id: str) -> tuple[pd.Series, str]:
    spec = resolve_flow_cell_spec(cell_id)
    assert spec is not None
    measure = resolve_semantic_measure(*spec.measure_ref)
    assert measure is not None
    bucket = frame["semantic_bucket"].fillna("").astype(str)
    subbucket = frame["semantic_subbucket"].fillna("").astype(str)
    mask = pd.Series(False, index=frame.index)
    for member_bucket, member_subbucket in spec.semantic_members:
        member = bucket.eq(member_bucket)
        if member_subbucket:
            member &= subbucket.eq(member_subbucket)
        mask |= member
    return mask, measure


def test_atomic_statement_values_reconcile_to_governed_membership_and_measure() -> None:
    split = pd.read_csv(SPLIT_FIXTURE)
    statement, _ = build_monthly_operating_statement_from_split(split)
    atomic_statement = statement.loc[statement["statement_line"].isin(ATOMIC_STATEMENT_LINES)]

    for row in atomic_statement.itertuples(index=False):
        cell_id = EXPLICIT_STATEMENT_LINE_FLOW_CELL_IDS[row.statement_line]
        membership, measure = _members_mask(split, cell_id)
        group = split.loc[
            split["period"].eq(row.period)
            & split["Currency"].eq(row.Currency)
            & membership
        ]
        matched = pd.to_numeric(group[measure], errors="coerce").fillna(0.0).sum()
        assert matched == row.amount, (
            row.period,
            row.Currency,
            row.statement_line,
            cell_id,
            measure,
            matched,
            row.amount,
        )
