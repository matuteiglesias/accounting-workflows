from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.contracts.annual_flow_membership import (
    ANNUAL_FLOW_MEMBERSHIP_VERSION,
    AnnualFlowMembershipSpec,
    build_annual_flow_membership,
)
from accounting.marts.semantic import build_monthly_operating_statement_from_split
from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.professional.drilldown import build_professional_flow_drilldowns


def _split() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"period":"2026-01","period_end":"2026-01-31","Currency":"ARS","Box":"Family Business","Lugar":"CABA","semantic_bucket":"operating_revenue","semantic_subbucket":"rent","amount_in":70.0,"amount_out":0.0,"net_amount":70.0,"amount_abs":70.0,"n_tx":1,"source_tx_ids_sample":"rent-jan-ars","review_required":False},
            {"period":"2026-03","period_end":"2026-03-31","Currency":"ARS","Box":"Family Business","Lugar":"CABA","semantic_bucket":"operating_revenue","semantic_subbucket":"rent","amount_in":30.0,"amount_out":0.0,"net_amount":30.0,"amount_abs":30.0,"n_tx":1,"source_tx_ids_sample":"rent-mar-ars","review_required":False},
            {"period":"2026-01","period_end":"2026-01-31","Currency":"USD","Box":"Family Business","Lugar":"CABA","semantic_bucket":"operating_revenue","semantic_subbucket":"rent","amount_in":10.0,"amount_out":0.0,"net_amount":10.0,"amount_abs":10.0,"n_tx":1,"source_tx_ids_sample":"rent-jan-usd","review_required":False},
            {"period":"2026-01","period_end":"2026-01-31","Currency":"ARS","Box":"Property Management","Lugar":"CABA","semantic_bucket":"property_opex","semantic_subbucket":"services","amount_in":0.0,"amount_out":100.0,"net_amount":-100.0,"amount_abs":100.0,"n_tx":1,"source_tx_ids_sample":"opex-services","review_required":False},
            {"period":"2026-03","period_end":"2026-03-31","Currency":"ARS","Box":"Property Management","Lugar":"CABA","semantic_bucket":"property_opex","semantic_subbucket":"taxes","amount_in":0.0,"amount_out":40.0,"net_amount":-40.0,"amount_abs":40.0,"n_tx":1,"source_tx_ids_sample":"opex-taxes","review_required":False},
            {"period":"2026-01","period_end":"2026-01-31","Currency":"ARS","Box":"Household","Lugar":"Home","semantic_bucket":"household_expense","semantic_subbucket":"services","amount_in":0.0,"amount_out":999.0,"net_amount":-999.0,"amount_abs":999.0,"n_tx":1,"source_tx_ids_sample":"household-expense","review_required":False},
            {"period":"2026-03","period_end":"2026-03-31","Currency":"ARS","Box":"Household","Lugar":"Home","semantic_bucket":"family_withdrawal_candidate","semantic_subbucket":"personal_expense","amount_in":0.0,"amount_out":20.0,"net_amount":-20.0,"amount_abs":20.0,"n_tx":1,"source_tx_ids_sample":"draw-personal","review_required":False},
        ]
    )


def _row(lineage: pd.DataFrame, metric_id: str, currency: str, dimension_value: str = "") -> pd.Series:
    rows = lineage[
        lineage["metric_id"].eq(metric_id)
        & lineage["period"].astype(str).eq("2026")
        & lineage["Currency"].eq(currency)
        & lineage["dimension_value"].fillna("").astype(str).eq(dimension_value)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_annual_flow_membership_sums_only_governed_monthly_cells() -> None:
    lineage = build_annual_flow_membership(_split())

    rent_ars = _row(lineage, "IS.RENT.TOTAL", "ARS")
    assert rent_ars["value"] == 100.0
    assert rent_ars["member_months"] == "2026-01;2026-03"
    assert "2026-02" not in rent_ars["monthly_governed_cell_ids"]
    assert set(rent_ars["source_member_ids"].split(";")) == {"rent-jan-ars", "rent-mar-ars"}
    assert rent_ars["measure_id"] == "amount_in"

    rent_usd = _row(lineage, "IS.RENT.TOTAL", "USD")
    assert rent_usd["value"] == 10.0
    assert rent_usd["source_member_ids"] == "rent-jan-usd"

    opex = _row(lineage, "IS.OPEX.PROPERTY", "ARS")
    assert opex["value"] == 140.0
    assert set(opex["source_member_ids"].split(";")) == {"opex-services", "opex-taxes"}
    assert "household-expense" not in opex["source_member_ids"]

    services = _row(lineage, "IS.OPEX.BY_CATEGORY", "ARS", "services")
    taxes = _row(lineage, "IS.OPEX.BY_CATEGORY", "ARS", "taxes")
    assert services["value"] == 100.0
    assert taxes["value"] == 40.0

    draws = _row(lineage, "DIST.DRAWS.PERSONAL", "ARS")
    assert draws["value"] == 20.0
    assert draws["source_member_ids"] == "draw-personal"
    assert set(lineage["lineage_version"]) == {ANNUAL_FLOW_MEMBERSHIP_VERSION}
    assert set(lineage["aggregation_rule"]) == {"sum_monthly_governed_values"}


def test_annual_flow_membership_contract_rejects_stock_semantics() -> None:
    with pytest.raises(ValueError, match="flow-only"):
        AnnualFlowMembershipSpec(
            annual_cell_family="annual.bad.stock",
            metric_id="BS.BAD.STOCK",
            monthly_flow_cell_id="flow.rent.total",
            flow_or_stock="stock",  # type: ignore[arg-type]
        )


def test_annual_metrics_materialize_lineage_and_reconcile_values(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    split = _split()
    split.to_csv(run / "monthly_flow_semantic_split.csv", index=False)
    statement, _ = build_monthly_operating_statement_from_split(split)
    statement.to_csv(run / "monthly_operating_statement.csv", index=False)

    paths = build_annual_balance_dashboard(
        run, run, run_id="annual-lineage-fixture", as_of_date="2026-03-31"
    )
    assert paths["annual_flow_membership"] == run / "annual_flow_membership.csv"
    lineage = pd.read_csv(paths["annual_flow_membership"], keep_default_na=False)
    metrics = pd.read_csv(paths["annual_balance_dashboard_metrics"])

    for metric_id, currency, dim_value, expected in [
        ("IS.RENT.TOTAL", "ARS", "", 100.0),
        ("IS.RENT.TOTAL", "USD", "", 10.0),
        ("IS.OPEX.PROPERTY", "ARS", "", 140.0),
        ("IS.OPEX.BY_CATEGORY", "ARS", "services", 100.0),
        ("DIST.DRAWS.PERSONAL", "ARS", "", 20.0),
    ]:
        lin = _row(lineage, metric_id, currency, dim_value)
        candidates = metrics[
            metrics["metric_id"].eq(metric_id)
            & pd.to_numeric(metrics["period"], errors="coerce").eq(2026)
            & metrics["Currency"].astype(str).eq(currency)
            & metrics["value_status"].eq("available")
        ]
        if dim_value:
            candidates = candidates[
                candidates["dimension_value"].fillna("").astype(str).eq(dim_value)
            ]
        else:
            candidates = candidates[
                candidates["dimension_name"].fillna("").astype(str).eq("")
            ]
        assert len(candidates) == 1
        assert float(candidates.iloc[0]["value"]) == expected
        assert float(lin["value"]) == expected


def test_professional_annual_flow_consumes_declared_lineage_not_monthly_reclassification(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    split = _split()
    split.to_csv(run / "monthly_flow_semantic_split.csv", index=False)
    statement, _ = build_monthly_operating_statement_from_split(split)
    statement.to_csv(run / "monthly_operating_statement.csv", index=False)
    build_annual_balance_dashboard(run, run, "annual-prof", "2026-03-31")

    # Mutate the monthly source after lineage materialization. If professional
    # annual execution independently reclassified monthly rows, this would leak
    # into the drilldown. The declared annual lineage remains the authority.
    poisoned = split.copy()
    poisoned.loc[poisoned["semantic_bucket"].eq("household_expense"), "semantic_bucket"] = "property_opex"
    poisoned.loc[poisoned["source_tx_ids_sample"].eq("household-expense"), "amount_out"] = 99999.0
    poisoned.to_csv(run / "monthly_flow_semantic_split.csv", index=False)

    repo = tmp_path / "repo"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {"metric_id":"IS.RENT.TOTAL","Currency":"ARS","2026":100.0},
            {"metric_id":"IS.OPEX.PROPERTY","Currency":"ARS","2026":140.0},
            {"metric_id":"DIST.DRAWS.PERSONAL","Currency":"ARS","2026":20.0},
            {"metric_id":"IS.OPEX.BY_CATEGORY","dimension_name":"semantic_subbucket","dimension_value":"services","Currency":"ARS","2026":100.0},
        ]
    ).to_csv(tables / "overview_balance_dashboard.csv", index=False)

    paths = build_professional_flow_drilldowns(repo, pack, run_root=run)
    index = pd.read_csv(paths["index"])
    governed = index[
        index["table_id"].eq("overview_balance_dashboard")
        & index["period"].astype(str).eq("2026")
        & index["Currency"].astype(str).eq("ARS")
    ]
    assert len(governed) == 4
    assert set(governed["status"]) == {"ok"}
    assert set(governed["source_artifact"]) == {"annual_flow_membership.csv"}
    assert set(governed["lineage_level"]) == {"annual_governed_membership"}
    assert set(pd.to_numeric(governed["residual"], errors="coerce")) == {0.0}

    for _, row in governed.iterrows():
        filters = json.loads(row["filter_json"])
        assert filters["lineage_version"] == ANNUAL_FLOW_MEMBERSHIP_VERSION
        assert filters["aggregation_rule"] == "sum_monthly_governed_values"
        assert "household-expense" not in filters["source_member_ids"]
