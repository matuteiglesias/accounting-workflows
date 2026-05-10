from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLS = [
    "opened_at",
    "debtor",
    "creditor",
    "currency",
    "item_type",
    "original_amount",
]


def _normalize_open_items(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in debt_open_items.csv: {missing}")

    out = df.copy()

    out["opened_at"] = pd.to_datetime(out["opened_at"], errors="coerce").dt.normalize()
    if "closed_at" not in out.columns:
        out["closed_at"] = pd.NaT
    else:
        out["closed_at"] = pd.to_datetime(out["closed_at"], errors="coerce").dt.normalize()

    out["debtor"] = out["debtor"].astype(str).str.strip()
    out["creditor"] = out["creditor"].astype(str).str.strip()
    out["currency"] = out["currency"].astype(str).str.strip().str.upper()
    out["item_type"] = out["item_type"].astype(str).str.strip()
    out["original_amount"] = pd.to_numeric(out["original_amount"], errors="coerce")
    out["open_amount"] = pd.to_numeric(out.get("open_amount", pd.NA), errors="coerce")

    out = out.dropna(subset=["opened_at", "debtor", "creditor", "currency", "item_type", "original_amount"]).copy()

    # If an item is still open, we want the full original amount to count historically.
    # If it has been closed, it contributes full original amount up to the day before closed_at.
    out["closed_at"] = pd.to_datetime(out["closed_at"], errors="coerce")

    return out.reset_index(drop=True)


def _date_span(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DatetimeIndex:
    min_open = df["opened_at"].min()
    max_close = df["closed_at"].dropna().max()

    start = pd.to_datetime(start_date).normalize() if start_date else min_open
    if pd.isna(start):
        raise ValueError("Could not infer start date from open items")

    if end_date:
        end = pd.to_datetime(end_date).normalize()
    else:
        # If there are open debts, run through today-like max open horizon from data:
        # use latest closed_at if all closed, else latest opened_at among still-open items
        if df["closed_at"].isna().any():
            end = max(df.loc[df["closed_at"].isna(), "opened_at"].max(), max_close if pd.notna(max_close) else pd.Timestamp.min)
        else:
            end = max_close

    if pd.isna(end):
        end = start

    if end < start:
        raise ValueError(f"End date {end.date()} is before start date {start.date()}")

    return pd.date_range(start=start, end=end, freq="D")


def build_debt_balance_daily(
    open_items: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df = _normalize_open_items(open_items)
    days = _date_span(df, start_date, end_date)

    rows: list[dict] = []

    for as_of_date in days:
        active = df.loc[
            (df["opened_at"] <= as_of_date)
            & (
                df["closed_at"].isna()
                | (df["closed_at"] > as_of_date)
            )
        ].copy()

        if active.empty:
            continue

        grouped = (
            active.groupby(["debtor", "creditor", "currency", "item_type"], dropna=False)["original_amount"]
            .sum()
            .reset_index()
            .rename(columns={"original_amount": "open_amount"})
        )

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "as_of_date": as_of_date.date().isoformat(),
                    "period_grain": "D",
                    "period": as_of_date.date().isoformat(),
                    "debtor": row["debtor"],
                    "creditor": row["creditor"],
                    "currency": row["currency"],
                    "item_type": row["item_type"],
                    "open_amount": float(row["open_amount"]),
                }
            )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(
            columns=[
                "as_of_date",
                "period_grain",
                "period",
                "debtor",
                "creditor",
                "currency",
                "item_type",
                "open_amount",
                "open_principal",
                "open_interest",
                "open_total",
            ]
        )

    pivot = (
        daily.pivot_table(
            index=["as_of_date", "period_grain", "period", "debtor", "creditor", "currency"],
            columns="item_type",
            values="open_amount",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )

    if "Prestamo" not in pivot.columns:
        pivot["Prestamo"] = 0.0
    if "Interes" not in pivot.columns:
        pivot["Interes"] = 0.0

    pivot = pivot.rename(
        columns={
            "Prestamo": "open_principal",
            "Interes": "open_interest",
        }
    )
    pivot["open_total"] = pivot["open_principal"] + pivot["open_interest"]

    detailed = daily.merge(
        pivot[["as_of_date", "debtor", "creditor", "currency", "open_principal", "open_interest", "open_total"]],
        on=["as_of_date", "debtor", "creditor", "currency"],
        how="left",
    )

    return detailed.sort_values(
        ["as_of_date", "debtor", "creditor", "currency", "item_type"]
    ).reset_index(drop=True)


def _last_snapshot_by_period(daily: pd.DataFrame, freq: str, grain_label: str) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()

    work = daily.copy()
    work["as_of_date"] = pd.to_datetime(work["as_of_date"], errors="coerce")
    work["period_obj"] = work["as_of_date"].dt.to_period(freq)
    work["period"] = work["period_obj"].astype(str)

    # Keep last available daily snapshot within each period / key / item_type
    idx = (
        work.groupby(["period", "debtor", "creditor", "currency", "item_type"], dropna=False)["as_of_date"]
        .idxmax()
    )
    out = work.loc[idx].copy()
    out["period_grain"] = grain_label
    return out.drop(columns=["period_obj"]).sort_values(
        ["period", "debtor", "creditor", "currency", "item_type"]
    ).reset_index(drop=True)


def write_outputs(
    daily: pd.DataFrame,
    *,
    write_dir: Path,
    source_path: str,
    start_date: str | None,
    end_date: str | None,
) -> None:
    write_dir.mkdir(parents=True, exist_ok=True)

    monthly = _last_snapshot_by_period(daily, "M", "M")
    quarterly = _last_snapshot_by_period(daily, "Q", "Q")
    yearly = _last_snapshot_by_period(daily, "Y", "Y")

    daily.to_csv(write_dir / "debt_balance_daily.csv", index=False)
    monthly.to_csv(write_dir / "debt_balance_monthly.csv", index=False)
    quarterly.to_csv(write_dir / "debt_balance_quarterly.csv", index=False)
    yearly.to_csv(write_dir / "debt_balance_yearly.csv", index=False)

    manifest = {
        "source_open_items": source_path,
        "start_date": start_date,
        "end_date": end_date,
        "n_daily_rows": int(len(daily)),
        "n_monthly_rows": int(len(monthly)),
        "n_quarterly_rows": int(len(quarterly)),
        "n_yearly_rows": int(len(yearly)),
        "currencies": sorted(daily["currency"].astype(str).unique().tolist()) if not daily.empty else [],
        "debtors": sorted(daily["debtor"].astype(str).unique().tolist()) if not daily.empty else [],
        "creditors": sorted(daily["creditor"].astype(str).unique().tolist()) if not daily.empty else [],
    }
    (write_dir / "debt_balance_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical debt balance views from debt_open_items.csv")
    p.add_argument("--open-items", required=True, help="Path to debt_open_items.csv")
    p.add_argument("--write-dir", required=True, help="Directory where debt balance CSVs will be written")
    p.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="Optional YYYY-MM-DD")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    open_items_path = Path(args.open_items)
    write_dir = Path(args.write_dir)

    open_items = pd.read_csv(open_items_path)
    daily = build_debt_balance_daily(
        open_items,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    write_outputs(
        daily,
        write_dir=write_dir,
        source_path=str(open_items_path),
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"Wrote: {write_dir / 'debt_balance_daily.csv'}")
    print(f"Wrote: {write_dir / 'debt_balance_monthly.csv'}")
    print(f"Wrote: {write_dir / 'debt_balance_quarterly.csv'}")
    print(f"Wrote: {write_dir / 'debt_balance_yearly.csv'}")
    print(f"Wrote: {write_dir / 'debt_balance_manifest.json'}")


if __name__ == "__main__":
    main()