from __future__ import annotations

import pandas as pd

from accounting.professional.drilldown import (
    STATUS_OK,
    STATUS_UNSUPPORTED,
    _build_derived_cell,
    _governed_flow_resolution,
    _spec_for_cell,
)
from accounting.professional.table_contracts import enrich_professional_table


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def _derived(
    table_id: str,
    row: pd.Series,
    period: str,
    display_value: float,
    split: pd.DataFrame,
    audit: pd.DataFrame | None = None,
):
    return _build_derived_cell(
        table_id=table_id,
        row=row,
        period=period,
        display_value=display_value,
        split=split,
        audit=_empty() if audit is None else audit,
        stmt=_empty(),
        annual=_empty(),
        cash_close=_empty(),
        debt_activity=_empty(),
        debt_position=_empty(),
        tolerance=1e-6,
    )


def test_governed_direct_spec_uses_declared_membership_grain_and_measure() -> None:
    row = enrich_professional_table(
        pd.DataFrame(
            [{"Currency": "ARS", "Box": "Household", "2026-01": 10}]
        ),
        "monthly_tables_draws_by_box_amount_out",
    ).iloc[0]

    split = pd.DataFrame(
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Household",
                "semantic_bucket": "family_withdrawal_candidate",
                "semantic_subbucket": "personal_expense",
                "amount_out": 10,
            },
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Family Business",
                "semantic_bucket": "family_withdrawal_candidate",
                "semantic_subbucket": "personal_expense",
                "amount_out": 99,
            },
            {
                "period": "2026-01",
                "Currency": "ARS",
                "Box": "Household",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "amount_out": 77,
            },
        ]
    )

    spec = _spec_for_cell("monthly_tables_draws_by_box_amount_out", row)
    assert spec is not None
    assert spec.measure == "amount_out"
    selected = split.loc[spec.filter_func(split, row)]
    assert selected["amount_out"].sum() == 10
    assert set(selected["semantic_bucket"]) == {"family_withdrawal_candidate"}
    assert set(selected["Box"]) == {"Household"}


def test_governed_union_reconciles_both_withdrawal_buckets() -> None:
    raw = pd.DataFrame(
        [
            {
                "statement_line": "family_draws_or_distributions",
                "Currency": "ARS",
                "2026-01": 14,
            }
        ]
    )
    row = enrich_professional_table(
        raw, "monthly_tables_operating_statement_matrix"
    ).iloc[0]
    assert row["drilldown_cell_id"] == "flow.family_draws_or_distributions.total"

    split = pd.DataFrame(
        [
            {
                "period": "2026-01",
                "Currency": "ARS",
                "semantic_bucket": "family_withdrawal_candidate",
                "semantic_subbucket": "personal_expense",
                "amount_out": 10,
                "source_tx_ids_sample": "a",
            },
            {
                "period": "2026-01",
                "Currency": "ARS",
                "semantic_bucket": "family_withdrawal",
                "semantic_subbucket": "distribution",
                "amount_out": 4,
                "source_tx_ids_sample": "b",
            },
            {
                "period": "2026-01",
                "Currency": "ARS",
                "semantic_bucket": "property_opex",
                "semantic_subbucket": "services",
                "amount_out": 100,
                "source_tx_ids_sample": "c",
            },
        ]
    )
    audit = pd.DataFrame(
        [
            {"tx_id": "a", "period": "2026-01", "Currency": "ARS", "semantic_bucket": "family_withdrawal_candidate", "semantic_subbucket": "personal_expense"},
            {"tx_id": "b", "period": "2026-01", "Currency": "ARS", "semantic_bucket": "family_withdrawal", "semantic_subbucket": "distribution"},
            {"tx_id": "c", "period": "2026-01", "Currency": "ARS", "semantic_bucket": "property_opex", "semantic_subbucket": "services"},
        ]
    )

    status, matched, residual, lineage, _, filters, _, detail, _ = _derived(
        "monthly_tables_operating_statement_matrix",
        row,
        "2026-01",
        14,
        split,
        audit,
    )
    assert status == STATUS_OK
    assert matched == 14
    assert residual == 0
    assert lineage.startswith("governed_atomic_flow:")
    assert filters["measure"] == "amount_out"
    assert filters["drilldown_cell_id"] == "flow.family_draws_or_distributions.total"
    assert set(detail["tx_id"]) == {"a", "b"}


def test_governed_annual_flow_uses_year_membership_not_one_month() -> None:
    row = enrich_professional_table(
        pd.DataFrame(
            [{"metric_id": "IS.RENT.TOTAL", "Currency": "ARS", "2026": 30}]
        ),
        "income_operating_statement",
    ).iloc[0]
    split = pd.DataFrame(
        [
            {"period": "2026-01", "Currency": "ARS", "semantic_bucket": "operating_revenue", "semantic_subbucket": "rent", "amount_in": 10},
            {"period": "2026-06", "Currency": "ARS", "semantic_bucket": "operating_revenue", "semantic_subbucket": "rent", "amount_in": 20},
            {"period": "2025-12", "Currency": "ARS", "semantic_bucket": "operating_revenue", "semantic_subbucket": "rent", "amount_in": 999},
        ]
    )

    status, matched, residual, lineage, *_ = _derived(
        "income_operating_statement", row, "2026", 30, split
    )
    assert status == STATUS_OK
    assert matched == 30
    assert residual == 0
    assert lineage.startswith("governed_atomic_flow:")


def test_missing_required_grain_fails_closed_instead_of_broadening() -> None:
    row = enrich_professional_table(
        pd.DataFrame(
            [{"metric_id": "IS.RENT.BY_PROPERTY", "Currency": "ARS", "2026": 30}]
        ),
        "income_operating_statement",
    ).iloc[0]
    assert row["drilldown_cell_id"] == "flow.rent.by_property"

    split = pd.DataFrame(
        [
            {"period": "2026-01", "Currency": "ARS", "Lugar": "CABA", "semantic_bucket": "operating_revenue", "semantic_subbucket": "rent", "amount_in": 30},
        ]
    )
    status, matched, residual, lineage, _, filters, *_ = _derived(
        "income_operating_statement", row, "2026", 30, split
    )
    assert status == STATUS_UNSUPPORTED
    assert matched == 0
    assert residual == -30
    assert lineage == "unsupported"
    assert "missing governed grain dimensions" in filters["reason"]
    assert "Lugar" in filters["reason"]


def test_multi_semantic_funding_support_ids_remain_on_legacy_path() -> None:
    row = enrich_professional_table(
        pd.DataFrame(
            [
                {
                    "metric_id": "FUND.CONTRIB.BY_FUNDING_ACTOR",
                    "Currency": "ARS",
                    "funding_actor": "Matías",
                    "2026": 20,
                }
            ]
        ),
        "income_operating_statement",
    ).iloc[0]
    assert row["drilldown_cell_id"] == "flow.funding_contribution.by_actor"
    assert _governed_flow_resolution(row) is None


def test_all_measures_diagnostic_remains_legacy_and_ungoverned() -> None:
    row = pd.Series(
        {
            "measure": "net_amount",
            "Currency": "ARS",
            "Box": "Property Management",
            "semantic_bucket": "property_opex",
        }
    )
    spec = _spec_for_cell("monthly_tables_flow_bucket_all_measures", row)
    assert spec is not None
    assert spec.measure == "net_amount"
    assert not row.get("drilldown_cell_id", "")
