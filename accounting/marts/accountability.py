from __future__ import annotations

"""Governed six-month accountability cycles and Box-specific control views."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


CYCLE_FREQUENCY_MONTHS = 6
CYCLE_ANCHOR_MONTH = 3
CYCLE_ANCHOR_DAY = 1
TOLERANCE = 0.01

CYCLE_COLUMNS = [
    "cycle_id", "cycle_start", "cycle_end", "view_type", "as_of_date", "Box",
    "Currency", "opening_accountability_balance", "accountable_receipts",
    "documented_distributions", "supported_uses", "documented_transfers_out",
    "closing_accountability_balance", "validated_cash", "validated_cash_status",
    "validated_cash_as_of_date", "other_documented_custody", "accountability_gap",
    "accountability_gap_status", "n_months", "n_tx", "source_table", "policy_id",
]

HOUSEHOLD_COLUMNS = [
    "period", "period_end", "Box", "Currency", "opening_household_balance",
    "effective_funding_contributions", "other_effective_receipts",
    "domestic_uses", "documented_transfers_out", "closing_household_balance",
    "position_label", "n_tx", "source_table",
]


def cycle_bounds(value: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    date = pd.Timestamp(value).normalize()
    anchor_year = date.year if date.month >= CYCLE_ANCHOR_MONTH else date.year - 1
    months_since_anchor = (date.year - anchor_year) * 12 + date.month - CYCLE_ANCHOR_MONTH
    start_month_offset = (months_since_anchor // CYCLE_FREQUENCY_MONTHS) * CYCLE_FREQUENCY_MONTHS
    start = pd.Timestamp(anchor_year, CYCLE_ANCHOR_MONTH, CYCLE_ANCHOR_DAY) + pd.DateOffset(months=start_month_offset)
    end = start + pd.DateOffset(months=CYCLE_FREQUENCY_MONTHS) - pd.Timedelta(days=1)
    return start, end


def _number(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.Series(0.0, index=frame.index)
    return frame[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


def build_family_business_accountability_cycles(
    monthly: pd.DataFrame, *, as_of_date: str | None = None
) -> pd.DataFrame:
    work = monthly.loc[monthly["Box"].astype(str).eq("Family Business")].copy()
    if work.empty:
        return pd.DataFrame(columns=CYCLE_COLUMNS)
    work["period_date"] = pd.to_datetime(work["period"].astype(str) + "-01", errors="coerce")
    work = work.loc[work["period_date"].notna()].copy()
    bounds = work["period_date"].map(cycle_bounds)
    work["cycle_start"] = bounds.map(lambda pair: pair[0])
    work["cycle_end"] = bounds.map(lambda pair: pair[1])
    work["documented_distributions"] = _number(
        work, ["personal_draws_out", "dividends_out"]
    )
    work["documented_transfers_out"] = _number(
        work, ["internal_transfer_out", "fx_out"]
    )
    work["supported_uses"] = (
        pd.to_numeric(work["total_cash_out"], errors="coerce").fillna(0.0)
        - work["documented_distributions"]
        - work["documented_transfers_out"]
    )
    work["accountable_receipts"] = pd.to_numeric(
        work["total_cash_in"], errors="coerce"
    ).fillna(0.0)
    cutoff = pd.Timestamp(as_of_date).normalize() if as_of_date else pd.Timestamp.today().normalize()

    rows: list[dict[str, Any]] = []
    for (currency, start, end), group in work.groupby(
        ["Currency", "cycle_start", "cycle_end"], dropna=False, sort=True
    ):
        group = group.sort_values("period")
        completed = end <= cutoff
        final = group.iloc[-1]
        cash_ok = (
            str(final.get("validated_cash_status", "")) == "available"
            and not str(final.get("validated_cash_reason", "")).strip()
            and pd.notna(pd.to_numeric(pd.Series([final.get("validated_cash_close")]), errors="coerce").iloc[0])
        )
        opening = float(pd.to_numeric(pd.Series([group.iloc[0].get("opening_control")]), errors="coerce").fillna(0).iloc[0])
        receipts = float(group["accountable_receipts"].sum())
        distributions = float(group["documented_distributions"].sum())
        uses = float(group["supported_uses"].sum())
        transfers = float(group["documented_transfers_out"].sum())
        closing = opening + receipts - distributions - uses - transfers
        validated_cash = float(final["validated_cash_close"]) if cash_ok else pd.NA
        custody = 0.0
        gap = closing - validated_cash - custody if cash_ok else pd.NA
        rows.append({
            "cycle_id": f"{start.date().isoformat()}_{end.date().isoformat()}",
            "cycle_start": start.date().isoformat(), "cycle_end": end.date().isoformat(),
            "view_type": "completed_cycle" if completed else "current_since_last_cut",
            "as_of_date": min(cutoff, end).date().isoformat(), "Box": "Family Business",
            "Currency": currency, "opening_accountability_balance": opening,
            "accountable_receipts": receipts, "documented_distributions": distributions,
            "supported_uses": uses, "documented_transfers_out": transfers,
            "closing_accountability_balance": closing, "validated_cash": validated_cash,
            "validated_cash_status": "available" if cash_ok else "unavailable",
            "validated_cash_as_of_date": final.get("validated_as_of_date", "") if cash_ok else "",
            "other_documented_custody": custody, "accountability_gap": gap,
            "accountability_gap_status": "available" if cash_ok else "unavailable_no_validated_cash",
            "n_months": len(group), "n_tx": int(pd.to_numeric(group.get("n_tx"), errors="coerce").fillna(0).sum()),
            "source_table": "monthly_cash_accountability.csv",
            "policy_id": "fb_accountability_cycle_6m_anchor_03_01_v1",
        })
    current_start, current_end = cycle_bounds(cutoff)
    for currency in sorted(work["Currency"].astype(str).unique()):
        if any(row["Currency"] == currency and row["cycle_start"] == current_start.date().isoformat() for row in rows):
            continue
        prior = [row for row in rows if row["Currency"] == currency and pd.Timestamp(row["cycle_end"]) < current_start]
        opening = float(prior[-1]["closing_accountability_balance"]) if prior else 0.0
        rows.append({
            "cycle_id": f"{current_start.date().isoformat()}_{current_end.date().isoformat()}",
            "cycle_start": current_start.date().isoformat(), "cycle_end": current_end.date().isoformat(),
            "view_type": "current_since_last_cut", "as_of_date": cutoff.date().isoformat(),
            "Box": "Family Business", "Currency": currency,
            "opening_accountability_balance": opening, "accountable_receipts": 0.0,
            "documented_distributions": 0.0, "supported_uses": 0.0,
            "documented_transfers_out": 0.0, "closing_accountability_balance": opening,
            "validated_cash": pd.NA, "validated_cash_status": "unavailable",
            "validated_cash_as_of_date": "", "other_documented_custody": 0.0,
            "accountability_gap": pd.NA,
            "accountability_gap_status": "unavailable_no_validated_cash",
            "n_months": 0, "n_tx": 0, "source_table": "monthly_cash_accountability.csv",
            "policy_id": "fb_accountability_cycle_6m_anchor_03_01_v1",
        })
    return pd.DataFrame(rows, columns=CYCLE_COLUMNS).sort_values(["Currency", "cycle_start"]).reset_index(drop=True)


def build_household_monthly_control(monthly: pd.DataFrame) -> pd.DataFrame:
    work = monthly.loc[monthly["Box"].astype(str).eq("Household")].copy()
    if work.empty:
        return pd.DataFrame(columns=HOUSEHOLD_COLUMNS)
    work = work.sort_values(["Currency", "period"])
    contribution = pd.to_numeric(work.get("funding_cash_in"), errors="coerce").fillna(0.0)
    total_in = pd.to_numeric(work.get("total_cash_in"), errors="coerce").fillna(0.0)
    total_out = pd.to_numeric(work.get("total_cash_out"), errors="coerce").fillna(0.0)
    transfers = _number(work, ["internal_transfer_out", "fx_out"])
    uses = total_out - transfers
    net = total_in - total_out
    opening = net.groupby(work["Currency"], dropna=False).cumsum() - net
    closing = opening + net
    out = pd.DataFrame({
        "period": work["period"], "period_end": work["period_end"], "Box": "Household",
        "Currency": work["Currency"], "opening_household_balance": opening,
        "effective_funding_contributions": contribution,
        "other_effective_receipts": total_in - contribution,
        "domestic_uses": uses, "documented_transfers_out": transfers,
        "closing_household_balance": closing,
        "position_label": closing.map(lambda value: "surplus" if value > TOLERANCE else ("deficit" if value < -TOLERANCE else "balanced")),
        "n_tx": pd.to_numeric(work.get("n_tx"), errors="coerce").fillna(0).astype(int),
        "source_table": "monthly_cash_accountability.csv",
    })
    return out[HOUSEHOLD_COLUMNS]


def build_accountability_views(run_root: Path, *, as_of_date: str | None = None) -> dict[str, Path]:
    run_root = Path(run_root)
    monthly = pd.read_csv(run_root / "monthly_cash_accountability.csv")
    fb = build_family_business_accountability_cycles(monthly, as_of_date=as_of_date)
    hh = build_household_monthly_control(monthly)
    fb_path = run_root / "family_business_accountability_cycles.csv"
    hh_path = run_root / "household_monthly_control.csv"
    fb.to_csv(fb_path, index=False)
    hh.to_csv(hh_path, index=False)
    return {"family_business_accountability_cycles": fb_path, "household_monthly_control": hh_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--as-of-date")
    args = parser.parse_args()
    for path in build_accountability_views(args.run_root, as_of_date=args.as_of_date).values():
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
