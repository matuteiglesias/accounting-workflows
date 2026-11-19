from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd
import pytest

from accounting.contracts.cash_control import (
    CASH_CONTROL_SPECS_V1,
    CASH_CONTROL_SPECS_VERSION,
    resolve_cash_control_spec,
)


REFERENCE = Path("/home/matias/repos/accounting-workflows/out/run/accounting/20261118T210059Z_FBPM/monthly_cash_accountability.csv")
REFERENCE_SHA256 = "c15b6af3f02fa061d1ee70bdc954fa403e04dd5098c721879e3a76cf9c395ec1"


def test_replacement_specs_are_additive_and_pin_the_existing_columns() -> None:
    assert CASH_CONTROL_SPECS_VERSION == "cash_control_specs_v1"
    assert set(CASH_CONTROL_SPECS_V1) == {
        "CTRL.BOX.NET_MOVEMENT",
        "CTRL.BOX.CUM_NET",
    }
    movement = resolve_cash_control_spec("CTRL.BOX.NET_MOVEMENT")
    cumulative = resolve_cash_control_spec("CTRL.BOX.CUM_NET")
    assert movement is not None and cumulative is not None
    assert movement.value_ref == "net_cash_flow"
    assert movement.annualization == "sum_period_flows"
    assert cumulative.value_ref == "closing_control"
    assert cumulative.annualization == "latest_month_end"
    assert movement.liquidity_claim == cumulative.liquidity_claim == "not_actual_cash_or_liquidity"


def test_synthetic_controls_conserve_movement_by_box_and_native_currency() -> None:
    # Transfer legs are deliberately represented separately: they conserve
    # across the pair without introducing a consolidated Box metric.
    frame = pd.DataFrame(
        [
            {"period": "2026-01", "period_end": "2026-01-31", "Box": "A", "Currency": "ARS", "total_cash_in": 100, "total_cash_out": 20, "opening_control": 0, "opening_balance_status": "unknown"},
            {"period": "2026-02", "period_end": "2026-02-28", "Box": "A", "Currency": "ARS", "total_cash_in": 0, "total_cash_out": 30, "opening_control": 80, "opening_balance_status": "unknown"},
            {"period": "2026-01", "period_end": "2026-01-31", "Box": "B", "Currency": "ARS", "total_cash_in": 20, "total_cash_out": 0, "opening_control": 0, "opening_balance_status": "unknown"},
            {"period": "2026-01", "period_end": "2026-01-31", "Box": "A", "Currency": "USD", "total_cash_in": 50, "total_cash_out": 5, "opening_control": 0, "opening_balance_status": "unknown"},
        ]
    )
    frame["net_cash_flow"] = frame["total_cash_in"] - frame["total_cash_out"]
    frame["closing_control"] = frame.groupby(["Box", "Currency"])["net_cash_flow"].cumsum()

    assert frame.loc[(frame.Box == "A") & (frame.Currency == "ARS"), "net_cash_flow"].tolist() == [80, -30]
    assert frame.loc[(frame.Box == "A") & (frame.Currency == "ARS"), "closing_control"].tolist() == [80, 50]
    assert frame.loc[(frame.Box == "B") & (frame.Currency == "ARS"), "closing_control"].item() == 20
    assert frame.loc[(frame.Box == "A") & (frame.Currency == "USD"), "closing_control"].item() == 45
    # The transfer pair (A out 20, B in 20) conserves in ARS, while each Box
    # keeps its own control and currencies never mix.
    assert frame.loc[frame.Currency.eq("ARS"), "net_cash_flow"].sum() == 70
    assert frame.loc[frame.Currency.eq("USD"), "net_cash_flow"].sum() == 45
    assert frame.loc[frame.Currency.eq("ARS"), "opening_balance_status"].eq("unknown").all()


def test_reference_fingerprint_is_skip_safe_and_stock_rule_is_explicit() -> None:
    if not REFERENCE.exists():
        pytest.skip("pinned private reference run is not available")
    frame = pd.read_csv(REFERENCE)
    assert len(frame) == 137
    assert set(["period", "period_end", "Box", "Currency", "net_cash_flow", "closing_control"]).issubset(frame.columns)
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256

    ordered = frame.sort_values(["Box", "Currency", "period"])
    assert (ordered["opening_control"] + ordered["net_cash_flow"] - ordered["closing_control"]).abs().max() <= 0.01
    annual = ordered.assign(year=ordered["period"].astype(str).str[:4])
    for _, group in annual.groupby(["year", "Box", "Currency"]):
        latest = group.sort_values("period").iloc[-1]
        assert latest["closing_control"] != pytest.approx(group["closing_control"].sum()) or len(group) == 1
