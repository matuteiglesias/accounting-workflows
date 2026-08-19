from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from accounting.professional.drilldown import build_professional_flow_drilldowns


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_professional_build_routes_monthly_and_annual_debt_activity_through_contract(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_debt_activity.csv",
        [
            {
                "period": "2025-03",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayment",
                "new_principal": 0,
                "interest_accrued": 0,
                "repayments": 180,
                "adjustments": 0,
                "net_change": 0,
            },
            {
                "period": "2025-04",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayment",
                "new_principal": 0,
                "interest_accrued": 0,
                "repayments": 170,
                "adjustments": 0,
                "net_change": 0,
            },
            {
                "period": "2025-04",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "closing_balance",
                "new_principal": 0,
                "interest_accrued": 0,
                "repayments": 0,
                "adjustments": 0,
                "net_change": 0,
            },
        ],
    )
    _write(
        tables / "monthly_tables_debt_activity_matrix.csv",
        [
            {
                "measure": "repayments",
                "Currency": "USD",
                "pair": "PM → MI",
                "2025-03": 180,
                "2025-04": 170,
            }
        ],
    )
    _write(
        tables / "annual_debt_activity_by_pair_wide.csv",
        [
            {
                "metric_id": "DEBT.ACTIVITY.REPAYMENT.BY_PAIR",
                "line_id": "debt.repayment",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "pair": "PM → MI",
                "activity_type": "repayments",
                "2025": 350,
            }
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    monthly = index[
        index["table_id"].eq("monthly_tables_debt_activity_matrix")
    ].sort_values("period")
    annual = index[index["table_id"].eq("annual_debt_activity_by_pair_wide")]

    assert list(monthly["status"]) == ["ok", "ok"]
    assert list(monthly["matched_value_sum"]) == [180.0, 170.0]
    assert set(monthly["lineage_level"]) == {"governed_debt_activity:monthly"}
    assert len(annual) == 1
    assert annual.iloc[0]["status"] == "ok"
    assert float(annual.iloc[0]["matched_value_sum"]) == 350.0
    assert annual.iloc[0]["lineage_level"] == "governed_debt_activity:annual"

    for _, output_row in pd.concat([monthly, annual]).iterrows():
        filters = json.loads(output_row["filter_json"])
        assert filters["spec_id"] == "debt.activity.repayment"
        assert filters["measure"] == "repayments"
        assert filters["activity_type"] == "repayment"
        assert filters["aggregation"] == "sum_flow"
        assert filters["executor"] == "governed_debt_activity_v1"
