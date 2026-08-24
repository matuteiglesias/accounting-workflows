from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.marts.debt import build_monthly_debt_position
from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.professional.drilldown import build_professional_flow_drilldowns


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build(tmp_path: Path) -> tuple[Path, Path]:
    debt_dir = tmp_path / "debt"
    run = tmp_path / "run"
    _write(
        debt_dir / "debt_balance_monthly.csv",
        [
            {"as_of_date":"2025-03-19","period":"2025-03","debtor":"PM","creditor":"MI","currency":"usd","open_principal":880,"open_interest":20,"open_total":900},
            {"as_of_date":"2025-03-31","period":"2025-03","debtor":"PM","creditor":"MI","currency":"usd","open_principal":850,"open_interest":20,"open_total":870},
            {"as_of_date":"not-a-date-a","period":"2025-04","debtor":"PM","creditor":"MI","currency":"usd","open_principal":800,"open_interest":0,"open_total":800},
            {"as_of_date":"not-a-date-z","period":"2025-04","debtor":"PM","creditor":"MI","currency":"usd","open_principal":700,"open_interest":0,"open_total":700},
        ],
    )
    _write(
        debt_dir / "debt_repayment_events.csv",
        [
            {"repayment_date":"2025-03-20","debtor":"PM","creditor":"MI","currency":"usd","allocated_amount":180},
            {"repayment_date":"2025-04-20","debtor":"PM","creditor":"MI","currency":"usd","allocated_amount":170},
        ],
    )
    build_monthly_debt_position(debt_dir, run)
    return debt_dir, run


def test_invalid_closing_period_stays_unavailable_through_annual_metrics(tmp_path: Path) -> None:
    _, run = _build(tmp_path)
    position = pd.read_csv(run / "monthly_debt_position.csv")
    april = position[position["period"].eq("2025-04")]
    assert set(april["position_status"]) == {"unavailable"}
    assert april["open_amount"].isna().all()

    metrics_dir = tmp_path / "metrics"
    paths = build_annual_balance_dashboard(run, metrics_dir, "debt-authority-fixture", "2025-04-30")
    annual = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    stock_ids = {
        "ID.DEBT.TOTAL.OPEN",
        "ID.DEBT.PRINCIPAL.OPEN",
        "ID.DEBT.INTEREST.OPEN",
        "ID.DEBT.NET_PM_POSITION",
    }
    stock = annual[
        annual["metric_id"].isin(stock_ids)
        & annual["period"].astype(str).eq("2025")
        & annual["Currency"].astype(str).eq("USD")
    ]
    assert set(stock["metric_id"]) == stock_ids
    assert set(stock["value_status"]) == {"unavailable"}
    assert stock["value"].isna().all()
    assert stock["calculation_rule"].astype(str).str.contains("latest valid as_of_date").all()

    repayments = annual[
        annual["metric_id"].eq("ID.DEBT.ACTIVITY.REPAYMENTS")
        & annual["period"].astype(str).eq("2025")
        & annual["Currency"].astype(str).eq("USD")
    ]
    assert len(repayments) == 1
    assert repayments.iloc[0]["value_status"] == "available"
    assert float(repayments.iloc[0]["value"]) == 350.0


def test_professional_position_uses_same_fail_closed_authority_after_mart(tmp_path: Path) -> None:
    _, run = _build(tmp_path)
    repo = tmp_path / "repo"
    pack = repo / "out" / "professional_pack" / "latest"
    tables = pack / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"measure":"open_principal","Currency":"USD","pair":"PM → MI","2025-03":850,"2025-04":700}]
    ).to_csv(tables / "monthly_tables_debt_position_matrix.csv", index=False)

    paths = build_professional_flow_drilldowns(repo, pack, run)
    index = pd.read_csv(paths["index"])
    april = index[
        index["table_id"].eq("monthly_tables_debt_position_matrix")
        & index["period"].astype(str).eq("2025-04")
        & index["Currency"].astype(str).eq("USD")
    ]
    assert len(april) == 1
    row = april.iloc[0]
    assert row["status"] == "unavailable"
    assert float(row["matched_value_sum"]) == 0.0
    assert "no valid as_of_date" in str(row["filter_reason"])
