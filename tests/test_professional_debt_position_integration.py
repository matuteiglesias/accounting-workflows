from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from accounting.professional.drilldown import build_professional_flow_drilldowns


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_professional_builder_routes_monthly_and_annual_debt_stock_through_contract(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    run = repo / "out" / "run" / "accounting" / "latest"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"

    _write(
        run / "monthly_debt_position.csv",
        [
            {
                "period": "2025-03",
                "as_of_date": "2025-03-31",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "principal",
                "open_amount": 850.0,
                "open_principal": 850.0,
                "open_interest": 20.0,
                "open_total": 870.0,
            },
            {
                "period": "2025-04",
                "as_of_date": "not-a-date-z",
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "principal",
                "open_amount": 700.0,
                "open_principal": 700.0,
                "open_interest": 0.0,
                "open_total": 700.0,
            },
        ],
    )

    _write(
        tables / "monthly_tables_debt_position_matrix.csv",
        [
            {
                "measure": "open_principal",
                "Currency": "USD",
                "pair": "PM → MI",
                "2025-03": 850.0,
                "2025-04": 700.0,
            }
        ],
    )
    _write(
        tables / "annual_debt_stock_by_pair_wide.csv",
        [
            {
                "Currency": "USD",
                "debtor": "PM",
                "creditor": "MI",
                "component": "open_principal",
                "2025": 700.0,
            }
        ],
    )

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])

    march = index[
        index["table_id"].eq("monthly_tables_debt_position_matrix")
        & index["period"].astype(str).eq("2025-03")
    ].iloc[0]
    assert march["status"] == "ok"
    assert march["matched_value_sum"] == 850.0
    march_filters = json.loads(march["filter_json"])
    assert march_filters["spec_id"] == "debt.position.principal"
    assert march_filters["executor"] == "governed_debt_position_v1"
    assert march_filters["selected_as_of_date"] == "2025-03-31"

    april = index[
        index["table_id"].eq("monthly_tables_debt_position_matrix")
        & index["period"].astype(str).eq("2025-04")
    ].iloc[0]
    assert april["status"] == "unavailable"
    assert april["matched_value_sum"] == 0.0
    april_filters = json.loads(april["filter_json"])
    assert april_filters["availability_status"] == "unavailable"
    assert april_filters["valid_as_of_rows"] == 0

    annual = index[index["table_id"].eq("annual_debt_stock_by_pair_wide")].iloc[0]
    assert annual["status"] == "unavailable"
    assert annual["matched_value_sum"] == 0.0
    annual_filters = json.loads(annual["filter_json"])
    assert annual_filters["selected_period"] == "2025-04"
    assert annual_filters["invalid_as_of_policy"] == "unavailable"
    assert "prior periods are not substituted" in annual_filters["reason"]

    # Unavailable is evidence-bearing, not a silent zero: detail files still
    # exist and preserve the invalid candidate rows for review.
    assert (pack / april["detail_csv_relpath"]).exists()
    assert (pack / annual["detail_csv_relpath"]).exists()
