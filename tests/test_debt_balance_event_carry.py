from __future__ import annotations

import pandas as pd

from accounting.debt.balance_views import build_debt_balance_daily


def test_event_dates_recompute_only_when_stock_can_change(monkeypatch) -> None:
    items = pd.DataFrame([
        {"debt_id": "a", "opened_at": "2026-01-01", "closed_at": "",
         "debtor": "A", "creditor": "PM", "currency": "USD",
         "item_type": "Prestamo", "original_amount": 100.0},
        {"debt_id": "b", "opened_at": "2026-02-10", "closed_at": "",
         "debtor": "A", "creditor": "PM", "currency": "USD",
         "item_type": "Prestamo", "original_amount": 50.0},
    ])
    allocations = pd.DataFrame([
        {"target_debt_id": "a", "allocation_date": "2026-03-15", "allocated_amount": 30.0},
    ])

    # Every event recomputes the known/active merge exactly once. The
    # final detailed-output merge adds one more; days without events do not.
    calls = []
    original_merge = pd.DataFrame.merge

    def counted_merge(frame, *args, **kwargs):
        calls.append(1)
        return original_merge(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "merge", counted_merge)
    result = build_debt_balance_daily(items, allocations=allocations, end_date="2026-03-31")

    assert len(calls) == 4  # 3 event dates + 1 final detail join
    assert set(result["as_of_date"]) == set(pd.date_range("2026-01-01", "2026-03-31").strftime("%Y-%m-%d"))
    assert result.loc[result.as_of_date.eq("2026-02-09"), "open_total"].iloc[0] == 100
    assert result.loc[result.as_of_date.eq("2026-02-10"), "open_total"].iloc[0] == 150
    assert result.loc[result.as_of_date.eq("2026-03-14"), "open_total"].iloc[0] == 150
    assert result.loc[result.as_of_date.eq("2026-03-15"), "open_total"].iloc[0] == 120
    assert result.loc[result.as_of_date.eq("2026-03-31"), "open_total"].iloc[0] == 120


def test_initial_carried_stock_and_same_day_closing_without_allocations() -> None:
    items = pd.DataFrame([
        {"opened_at": "2026-01-01", "closed_at": "2026-02-10",
         "debtor": "A", "creditor": "PM", "currency": "USD",
         "item_type": "Prestamo", "original_amount": 100.0},
        {"opened_at": "2026-01-01", "closed_at": "",
         "debtor": "A", "creditor": "PM", "currency": "USD",
         "item_type": "Interes", "original_amount": 15.0},
    ])
    result = build_debt_balance_daily(items, start_date="2026-02-01", end_date="2026-02-28")
    assert result["as_of_date"].min() == "2026-02-01"
    assert result["as_of_date"].max() == "2026-02-28"
    principal = result[result.item_type.eq("Prestamo")]
    assert principal.loc[principal.as_of_date.eq("2026-02-09"), "open_amount"].iloc[0] == 100
    assert principal.loc[principal.as_of_date.eq("2026-02-10"), "open_amount"].iloc[0] == 0
    assert principal.loc[principal.as_of_date.eq("2026-02-28"), "open_amount"].iloc[0] == 0
    interest = result[result.item_type.eq("Interes")]
    assert interest["open_amount"].eq(15).all()
