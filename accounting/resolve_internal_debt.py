from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd

from .ingest import build_ledger_base


VALID_DEBT_TYPES = {"Prestamo", "Interes"}
VALID_REPAYMENT_TYPE = "Repago"
DEFAULT_STATUS = "pagado"


@dataclass
class OpenItem:
    debt_id: str
    source_tx_id: str
    opened_at: str
    debtor: str
    creditor: str
    currency: str
    item_type: str
    original_amount: float
    open_amount: float
    detalle: str
    lugar: str
    status: str
    closed_at: str = ""


@dataclass
class Allocation:
    allocation_id: str
    repayment_tx_id: str
    allocation_date: str
    debtor: str
    creditor: str
    currency: str
    target_debt_id: str
    target_item_type: str
    target_opened_at: str
    allocated_amount: float
    rule_version: str
    note: str = ""


@dataclass
class RepaymentEvent:
    repayment_tx_id: str
    repayment_date: str
    debtor: str
    creditor: str
    currency: str
    repayment_amount: float
    allocated_amount: float
    leftover_amount: float
    n_allocations: int
    rule_version: str


def _parse_list_arg(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [x.strip() for x in s.replace(",", " ").split() if x.strip()] or None


def load_debt_ledger(args: argparse.Namespace) -> pd.DataFrame:
    df = build_ledger_base(
        fixture_path=args.fixture or None,
        sheet_url=args.sheet_url or None,
        service_account_file=args.service_account or None,
        sheet_name=args.sheet_name,
        exclude_household=bool(args.exclude_household),
        only_status=_parse_list_arg(args.include_statuses),
        require_tx_id=True,
    ).copy()

    # keep only debt slice
    df = df.loc[df["Tipo"].isin(VALID_DEBT_TYPES | {VALID_REPAYMENT_TYPE})].copy()

    # optional: exclude Household again defensively if Box survived as a weird label
    if "Box" in df.columns:
        df = df.loc[df["Box"].astype(str).str.strip().str.lower() != "household"].copy()

    # enrich expected convenience fields
    if "Detalle" not in df.columns:
        df["Detalle"] = ""
    if "Lugar" not in df.columns:
        df["Lugar"] = ""
    if "notes" not in df.columns:
        df["notes"] = ""

    return df.sort_values(["Date", "tx_id"]).reset_index(drop=True)


def build_open_items(df: pd.DataFrame) -> List[OpenItem]:
    debt_rows = df.loc[df["Tipo"].isin(VALID_DEBT_TYPES)].copy()
    items: List[OpenItem] = []

    for _, row in debt_rows.iterrows():
        item_type = str(row["Tipo"])
        creditor = str(row["payer"])
        debtor = str(row["receiver"])

        items.append(
            OpenItem(
                debt_id=f"{item_type.lower()}::{row['tx_id']}",
                source_tx_id=str(row["tx_id"]),
                opened_at=pd.to_datetime(row["Date"]).date().isoformat(),
                debtor=debtor,
                creditor=creditor,
                currency=str(row["Currency"]),
                item_type=item_type,
                original_amount=float(row["amount"]),
                open_amount=float(row["amount"]),
                detalle=str(row.get("Detalle", "")),
                lugar=str(row.get("Lugar", "")),
                status="open",
            )
        )

    return items


def sort_open_items(items: List[OpenItem]) -> List[OpenItem]:
    type_rank = {"Interes": 0, "Prestamo": 1}
    return sorted(
        items,
        key=lambda x: (
            x.debtor,
            x.creditor,
            x.currency,
            type_rank.get(x.item_type, 99),
            x.opened_at,
            x.debt_id,
        ),
    )


def build_repayments(df: pd.DataFrame) -> pd.DataFrame:
    rep = df.loc[df["Tipo"] == VALID_REPAYMENT_TYPE].copy()
    rep["debtor"] = rep["payer"].astype(str)
    rep["creditor"] = rep["receiver"].astype(str)
    rep["repayment_amount"] = pd.to_numeric(rep["amount"], errors="coerce")
    return rep.sort_values(["Date", "tx_id"]).reset_index(drop=True)


def resolve_repayments(
    open_items: List[OpenItem],
    repayments: pd.DataFrame,
    full_only: bool = True,
    rule_version: str = "interest_first_fifo_full_only_skip_if_insufficient_v1",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items = sort_open_items(open_items)
    allocations: List[Allocation] = []
    repayment_events: List[RepaymentEvent] = []

    by_key: Dict[Tuple[str, str, str], List[OpenItem]] = {}
    for item in items:
        by_key.setdefault((item.debtor, item.creditor, item.currency), []).append(item)

    alloc_counter = 1

    for _, rep in repayments.iterrows():
        repayment_tx_id = str(rep["tx_id"])
        repayment_date = pd.to_datetime(rep["Date"]).date().isoformat()
        debtor = str(rep["debtor"])
        creditor = str(rep["creditor"])
        currency = str(rep["Currency"])
        remaining = float(rep["repayment_amount"])

        candidates = by_key.get((debtor, creditor, currency), [])

        allocated_before = remaining
        n_allocs = 0

        for item in candidates:
            if remaining <= 0:
                break
            if item.status == "closed" or item.open_amount <= 0:
                continue

            needed = item.open_amount

            if full_only and remaining < needed:
                continue

            alloc_amt = needed if full_only else min(remaining, needed)

            item.open_amount -= alloc_amt
            remaining -= alloc_amt
            n_allocs += 1

            if abs(item.open_amount) < 1e-9:
                item.open_amount = 0.0
                item.status = "closed"
                item.closed_at = repayment_date

            allocations.append(
                Allocation(
                    allocation_id=f"alloc::{alloc_counter}",
                    repayment_tx_id=repayment_tx_id,
                    allocation_date=repayment_date,
                    debtor=debtor,
                    creditor=creditor,
                    currency=currency,
                    target_debt_id=item.debt_id,
                    target_item_type=item.item_type,
                    target_opened_at=item.opened_at,
                    allocated_amount=alloc_amt,
                    rule_version=rule_version,
                    note="full cancellation" if full_only else "partial or full cancellation",
                )
            )
            alloc_counter += 1

        repayment_events.append(
            RepaymentEvent(
                repayment_tx_id=repayment_tx_id,
                repayment_date=repayment_date,
                debtor=debtor,
                creditor=creditor,
                currency=currency,
                repayment_amount=float(rep["repayment_amount"]),
                allocated_amount=allocated_before - remaining,
                leftover_amount=remaining,
                n_allocations=n_allocs,
                rule_version=rule_version,
            )
        )

    return (
        pd.DataFrame([asdict(x) for x in items]),
        pd.DataFrame([asdict(x) for x in allocations]),
        pd.DataFrame([asdict(x) for x in repayment_events]),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve internal debt chronologically from canonical ingest.")
    p.add_argument("--fixture", default=None)
    p.add_argument("--sheet-url", default=None)
    p.add_argument("--service-account", default=None)
    p.add_argument("--sheet-name", default="C. Long Ledger")
    p.add_argument("--write-dir", required=True)
    p.add_argument("--include-statuses", default=DEFAULT_STATUS)
    p.add_argument("--exclude-household", action="store_true")
    p.add_argument("--full-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    df = load_debt_ledger(args)
    open_items = build_open_items(df)
    repayments = build_repayments(df)

    open_items_df, allocations_df, repayment_events_df = resolve_repayments(
        open_items=open_items,
        repayments=repayments,
        full_only=args.full_only,
    )

    open_items_df.to_csv(write_dir / "debt_open_items.csv", index=False)
    allocations_df.to_csv(write_dir / "debt_allocations.csv", index=False)
    repayment_events_df.to_csv(write_dir / "debt_repayment_events.csv", index=False)

    print(f"Wrote: {write_dir / 'debt_open_items.csv'}")
    print(f"Wrote: {write_dir / 'debt_allocations.csv'}")
    print(f"Wrote: {write_dir / 'debt_repayment_events.csv'}")


if __name__ == "__main__":
    main()