from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from accounting.cutoff import (
    CUTOFF_RULE,
    CUTOFF_VERSION,
    cutoff_metadata,
    load_run_cutoff,
    resolve_run_as_of_date,
)
from accounting.debt.balance_views import (
    _last_snapshot_by_period,
    build_debt_balance_daily,
    resolve_effective_end_date,
)
from accounting.marts.cash import build_monthly_cash_close
from accounting.support.latest import (
    assert_latest_target_publishable,
    update_scoped_latest,
)


def _write_cutoff_manifest(run_root: Path, cutoff_date: str = "2026-07-31") -> None:
    meta = run_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "A.ingest",
        "params": cutoff_metadata(cutoff_date),
    }
    (meta / "stage_A_ingest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_stage_a_cutoff_is_the_single_as_of_authority(tmp_path):
    _write_cutoff_manifest(tmp_path)
    cutoff = load_run_cutoff(tmp_path)
    assert cutoff.date == "2026-07-31"
    assert cutoff.rule == CUTOFF_RULE
    assert cutoff.version == CUTOFF_VERSION

    assert resolve_run_as_of_date(tmp_path) == "2026-07-31"
    assert resolve_run_as_of_date(tmp_path, "2026-07-31") == "2026-07-31"
    with pytest.raises(ValueError, match="conflicts with immutable Stage A cutoff"):
        resolve_run_as_of_date(tmp_path, "2026-08-20")


def test_debt_open_stock_is_carried_through_cutoff_without_new_events(tmp_path):
    out_root = tmp_path / "out"
    run_id = "20260820T160000Z_FBPM"
    run_root = out_root / "run" / "accounting" / run_id
    debt_dir = out_root / "debt_resolution" / run_id
    debt_dir.mkdir(parents=True)
    _write_cutoff_manifest(run_root)

    end_date = resolve_effective_end_date(debt_dir, None)
    assert end_date == "2026-07-31"
    with pytest.raises(ValueError, match="Debt end-date conflicts"):
        resolve_effective_end_date(debt_dir, "2026-08-01")

    open_items = pd.DataFrame(
        [
            {
                "opened_at": "2026-02-10",
                "closed_at": "",
                "debtor": "Property Management",
                "creditor": "MI",
                "currency": "USD",
                "item_type": "Prestamo",
                "original_amount": 100.0,
                "open_amount": 100.0,
            }
        ]
    )
    daily = build_debt_balance_daily(open_items, end_date=end_date)
    assert daily["as_of_date"].max() == "2026-07-31"
    assert daily.loc[daily["as_of_date"].eq("2026-07-31"), "open_total"].max() == 100.0

    monthly = _last_snapshot_by_period(daily, "M", "M")
    july = monthly.loc[monthly["period"].eq("2026-07")]
    assert not july.empty
    assert july["as_of_date"].max().date().isoformat() == "2026-07-31"
    assert july["open_total"].max() == 100.0


def _validated_cash_row(as_of_date: str) -> dict[str, object]:
    period = as_of_date[:7]
    period_end = pd.Period(period, freq="M").end_time.date().isoformat()
    return {
        "period": period,
        "period_end": period_end,
        "as_of_date": as_of_date,
        "Box": "Property Management",
        "account_id": "bank-usd",
        "account_name": "Bank USD",
        "Currency": "USD",
        "close_amount": 250.0,
        "source_type": "account_snapshot",
        "source_reference": "fixture",
        "validation_status": "validated",
        "validated_by": "controller",
        "notes": "fixture",
    }


def test_validated_cash_cannot_reintroduce_future_evidence(tmp_path):
    _write_cutoff_manifest(tmp_path)
    pd.DataFrame([_validated_cash_row("2026-08-01")]).to_csv(
        tmp_path / "validated_cash_close.csv", index=False
    )
    with pytest.raises(ValueError, match="after immutable Stage A cutoff"):
        build_monthly_cash_close(tmp_path)


def test_validated_cash_at_cutoff_remains_frontend_safe(tmp_path):
    _write_cutoff_manifest(tmp_path)
    pd.DataFrame([_validated_cash_row("2026-07-31")]).to_csv(
        tmp_path / "validated_cash_close.csv", index=False
    )
    paths = build_monthly_cash_close(tmp_path)
    cash = pd.read_csv(paths["monthly_cash_close"])
    selected = cash.loc[cash["account_id"].astype(str).eq("bank-usd")]
    assert len(selected) == 1
    assert selected.iloc[0]["as_of_date"] == "2026-07-31"
    assert bool(selected.iloc[0]["is_frontend_safe"]) is True


def test_cutoff_run_cannot_silently_replace_latest_pointers(tmp_path):
    out_root = tmp_path / "out"
    run_base = out_root / "run" / "accounting"
    metrics_base = out_root / "metrics"
    old_id = "20260820T120000Z_FBPM"
    cutoff_id = "20260820T160000Z_FBPM"

    (run_base / old_id).mkdir(parents=True)
    cutoff_root = run_base / cutoff_id
    _write_cutoff_manifest(cutoff_root)
    (metrics_base / cutoff_id).mkdir(parents=True)

    update_scoped_latest(run_base, old_id, "FBPM")
    before = (run_base / "latest_FBPM").readlink()

    with pytest.raises(ValueError, match="Refusing to move latest pointers"):
        assert_latest_target_publishable([run_base, metrics_base], cutoff_id)

    assert (run_base / "latest_FBPM").readlink() == before
    assert_latest_target_publishable(
        [run_base, metrics_base],
        cutoff_id,
        allow_cutoff_latest=True,
    )
