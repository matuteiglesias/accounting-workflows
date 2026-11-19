from pathlib import Path

import pandas as pd
import pytest

from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.metrics.cash_control_adapter import annual_rows, read_source
from accounting.metrics.frontier import build_metrics_frontier


def _write_source(root: Path) -> None:
    pd.DataFrame(
        [
            {"period": "2026-01", "period_end": "2026-01-31", "Box": "A", "Currency": "ARS", "net_cash_flow": 80, "closing_control": 80, "opening_control": 0, "opening_balance_status": "unknown"},
            {"period": "2026-02", "period_end": "2026-02-28", "Box": "A", "Currency": "ARS", "net_cash_flow": -30, "closing_control": 50, "opening_control": 80, "opening_balance_status": "unknown"},
            {"period": "2026-01", "period_end": "2026-01-31", "Box": "B", "Currency": "ARS", "net_cash_flow": 20, "closing_control": 20, "opening_control": 0, "opening_balance_status": "unknown"},
            {"period": "2026-01", "period_end": "2026-01-31", "Box": "A", "Currency": "USD", "net_cash_flow": 45, "closing_control": 45, "opening_control": 0, "opening_balance_status": "unknown"},
        ]
    ).to_csv(root / "monthly_cash_accountability.csv", index=False)


def test_frontier_and_annual_emit_source_backed_controls_without_cross_scope(tmp_path: Path) -> None:
    _write_source(tmp_path)
    monthly = tmp_path / "monthly"
    annual = tmp_path / "annual"

    frontier_paths = build_metrics_frontier(tmp_path, monthly, "fixture", "2026-02-28")
    series = pd.read_csv(frontier_paths["frontend_metric_series"])
    controls = series[series.metric_id.str.startswith("CTRL.BOX")]
    assert len(controls) == 8
    assert set(controls.dimension_value) == {"A", "B"}
    assert set(controls.Currency) == {"ARS", "USD"}
    assert controls.loc[controls.metric_id.eq("CTRL.BOX.NET_MOVEMENT"), "value"].tolist() == [80.0, -30.0, 20.0, 45.0]
    assert controls.loc[controls.metric_id.eq("CTRL.BOX.CUM_NET"), "value"].tolist() == [80.0, 50.0, 20.0, 45.0]
    assert controls.source_table.eq("monthly_cash_accountability.csv").all()
    assert controls.frontend_suitability.eq("safe_with_caveat").all()
    assert controls.caveat.str.contains("not actual cash or liquidity").all()

    annual_paths = build_annual_balance_dashboard(tmp_path, annual, "fixture", "2026-02-28")
    metrics = pd.read_csv(annual_paths["annual_balance_dashboard_metrics"])
    controls = metrics[metrics.metric_id.str.startswith("CTRL.BOX")]
    annual_values = controls.set_index(["metric_id", "dimension_value", "Currency"])["value"]
    assert annual_values["CTRL.BOX.NET_MOVEMENT", "A", "ARS"] == 50.0
    assert annual_values["CTRL.BOX.NET_MOVEMENT", "B", "ARS"] == 20.0
    assert annual_values["CTRL.BOX.NET_MOVEMENT", "A", "USD"] == 45.0
    assert annual_values["CTRL.BOX.CUM_NET", "A", "ARS"] == 50.0
    assert controls.dimension_name.eq("Box").all()
    snapshots = annual_rows(read_source(tmp_path), run_id="fixture", as_of_date="2026-02-28")
    end_by_scope = {(r["metric_id"], r["Box"], r["Currency"]): r["period_end"] for r in snapshots}
    # Annual year-to-date stocks must not claim a future December 31 snapshot.
    assert end_by_scope[("CTRL.BOX.CUM_NET", "A", "ARS")] == "2026-02-28"
    assert end_by_scope[("CTRL.BOX.CUM_NET", "B", "ARS")] == "2026-01-31"


def test_absent_source_is_explicitly_unavailable_and_invalid_source_fails_closed(tmp_path: Path) -> None:
    paths = build_annual_balance_dashboard(tmp_path, tmp_path / "annual", "fixture", "2026-02-28")
    metrics = pd.read_csv(paths["annual_balance_dashboard_metrics"])
    controls = metrics[metrics.metric_id.str.startswith("CTRL.BOX")]
    assert set(controls.value_status) == {"unavailable"}
    assert controls.caveat.str.contains("Source absent").all()

    _write_source(tmp_path)
    source = pd.read_csv(tmp_path / "monthly_cash_accountability.csv")
    source["net_cash_flow"] = source["net_cash_flow"].astype(object)
    source.loc[1, "net_cash_flow"] = "not-a-number"
    source.to_csv(tmp_path / "monthly_cash_accountability.csv", index=False)
    with pytest.raises(ValueError, match="invalid numeric net_cash_flow"):
        read_source(tmp_path)

    _write_source(tmp_path)
    source = pd.read_csv(tmp_path / "monthly_cash_accountability.csv")
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    source.to_csv(tmp_path / "monthly_cash_accountability.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_source(tmp_path)
