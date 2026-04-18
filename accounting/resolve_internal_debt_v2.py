from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .ingest import build_ledger_base


VALID_DEBT_TYPES = {"Prestamo", "Interes"}
VALID_REPAYMENT_TYPE = "Repago"
DEFAULT_REPAYMENT_STATUSES = "pagado"
RULE_VERSION = "interest_first_fifo_full_only_skip_if_insufficient_v2"


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
    issuer: str
    ledger_status: str
    engine_status: str
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


@dataclass
class TimelineEvent:
    event_date: str
    event_kind: str
    tx_id: str
    debt_id: str
    debtor: str
    creditor: str
    currency: str
    item_type: str
    amount: float
    related_tx_id: str = ""
    note: str = ""


@dataclass
class StatusReconciliation:
    debt_id: str
    source_tx_id: str
    debtor: str
    creditor: str
    currency: str
    item_type: str
    ledger_status: str
    engine_status: str
    original_amount: float
    open_amount: float
    closed_at: str
    reconciliation_note: str


def _parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return [token.strip() for token in text.replace(",", " ").split() if token.strip()] or None


def _coalesce_first_present(df: pd.DataFrame, candidates: list[str], target: str) -> pd.DataFrame:
    """
    Build exactly one canonical column `target` from the first non-empty value
    across candidate columns, then drop the extra aliases.
    """
    present = [c for c in candidates if c in df.columns]
    if not present:
        return df

    base = None
    for col in present:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.copy()
        if base is None:
            base = s
        else:
            base = base.where(base.notna(), s)

    df = df.drop(columns=present, errors="ignore")
    df[target] = base
    return df


def _normalize_ledger_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # 1. Canonicalize aliases without creating duplicate labels
    work = _coalesce_first_present(work, ["tx_id", "transaction_id"], "tx_id")
    work = _coalesce_first_present(work, ["amount", "monto"], "amount")
    work = _coalesce_first_present(work, ["Date"], "Date")
    work = _coalesce_first_present(work, ["payer"], "payer")
    work = _coalesce_first_present(work, ["receiver"], "receiver")
    work = _coalesce_first_present(work, ["Currency"], "Currency")
    work = _coalesce_first_present(work, ["Tipo"], "Tipo")
    work = _coalesce_first_present(work, ["Flujo"], "Flujo")
    work = _coalesce_first_present(work, ["Detalle"], "Detalle")
    work = _coalesce_first_present(work, ["Lugar"], "Lugar")
    work = _coalesce_first_present(work, ["Issuer"], "Issuer")
    work = _coalesce_first_present(work, ["status"], "status")
    work = _coalesce_first_present(work, ["Box"], "Box")

    # 2. Defensive: if any duplicate labels somehow remain, keep first occurrence only
    work = work.loc[:, ~work.columns.duplicated()].copy()

    required = ["tx_id", "Date", "payer", "receiver", "amount", "Currency", "Tipo", "Flujo"]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise KeyError(f"Ledger missing required columns: {missing}")

    for col in ["Detalle", "Lugar", "Issuer", "status", "Box"]:
        if col not in work.columns:
            work[col] = ""

    # 3. Cast after canonicalization
    work["tx_id"] = work["tx_id"].astype(str).str.strip()
    work["payer"] = work["payer"].astype(str).str.strip()
    work["receiver"] = work["receiver"].astype(str).str.strip()
    work["Currency"] = work["Currency"].astype(str).str.strip()
    work["Tipo"] = work["Tipo"].astype(str).str.strip()
    work["Flujo"] = work["Flujo"].astype(str).str.strip()
    work["Detalle"] = work["Detalle"].fillna("").astype(str).str.strip()
    work["Lugar"] = work["Lugar"].fillna("").astype(str).str.strip()
    work["Issuer"] = work["Issuer"].fillna("").astype(str).str.strip()
    work["Box"] = work["Box"].fillna("").astype(str).str.strip()
    work["status"] = work["status"].fillna("").astype(str).str.strip().str.lower()

    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")

    work = work.loc[work["Date"].notna() & work["amount"].notna()].copy()
    return work.sort_values(["Date", "tx_id"]).reset_index(drop=True)



def load_debt_ledger(args: argparse.Namespace) -> pd.DataFrame:
    if args.ledger_csv:
        raw = pd.read_csv(args.ledger_csv)
    else:
        raw = build_ledger_base(
            fixture_path=args.fixture or None,
            sheet_url=args.sheet_url or None,
            service_account_file=args.service_account or None,
            sheet_name=args.sheet_name,
            exclude_household=bool(args.exclude_household),
            only_status=None,
            require_tx_id=True,
        )

    df = _normalize_ledger_columns(raw)

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns remain after normalization: {dupes}")

    df = df.loc[df["Tipo"].isin(VALID_DEBT_TYPES | {VALID_REPAYMENT_TYPE})].copy()

    if args.exclude_household and "Box" in df.columns:
        df = df.loc[df["Box"].astype(str).str.strip().str.lower() != "household"].copy()

    allowed_currencies = _parse_list_arg(args.currencies)
    if allowed_currencies:
        allowed = {c.upper() for c in allowed_currencies}
        df = df.loc[df["Currency"].str.upper().isin(allowed)].copy()

    return df.sort_values(["Date", "tx_id"]).reset_index(drop=True)


def build_open_items(df: pd.DataFrame, verbose: bool = False) -> List[OpenItem]:
    def vprint(*args):
        if verbose:
            print(*args)

    debt_rows = df.loc[df["Tipo"].isin(VALID_DEBT_TYPES)].copy()
    vprint(f"[build_open_items] total_rows={len(df)} debt_rows={len(debt_rows)}")

    if verbose and not df.empty:
        vprint("[build_open_items] Tipo counts:")
        print(df["Tipo"].value_counts(dropna=False).to_string())

        cols = [c for c in ["Date", "tx_id", "payer", "receiver", "Currency", "Tipo", "Flujo", "amount", "status"] if c in df.columns]
        vprint("[build_open_items] filtered ledger preview:")
        print(df[cols].head(20).to_string(index=False))

    items: List[OpenItem] = []

    for _, row in debt_rows.iterrows():
        item_type = str(row["Tipo"])
        debtor = str(row["payer"])
        creditor = str(row["receiver"])

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
                issuer=str(row.get("Issuer", "")),
                ledger_status=str(row.get("status", "")),
                engine_status="open",
            )
        )

    vprint(f"[build_open_items] built_open_items={len(items)}")
    return items

def sort_open_items(items: Sequence[OpenItem]) -> List[OpenItem]:
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


def build_repayments(df: pd.DataFrame, repayment_statuses: Optional[List[str]]) -> pd.DataFrame:
    rep = df.loc[df["Tipo"] == VALID_REPAYMENT_TYPE].copy()
    if repayment_statuses:
        allowed = {status.strip().lower() for status in repayment_statuses}
        rep = rep.loc[rep["status"].isin(allowed)].copy()

    rep["debtor"] = rep["payer"].astype(str)
    rep["creditor"] = rep["receiver"].astype(str)
    rep["repayment_amount"] = pd.to_numeric(rep["amount"], errors="coerce")
    rep = rep.loc[rep["repayment_amount"].notna() & (rep["repayment_amount"] > 0)].copy()
    # return rep.sort_values(["Date", "tx_id"]).reset_index(drop=True)
    return rep.sort_values(["Date"]).reset_index(drop=True)

def resolve_repayments(
    open_items: List[OpenItem],
    repayments: pd.DataFrame,
    full_only: bool = True,
    rule_version: str = RULE_VERSION,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def vprint(*args):
        if verbose:
            print(*args)

    items = sort_open_items(open_items)
    allocations: List[Allocation] = []
    repayment_events: List[RepaymentEvent] = []
    timeline: List[TimelineEvent] = []

    vprint("\n[resolve] ===== START =====")
    vprint(f"[resolve] open_items_in={len(open_items)} sorted_items={len(items)} repayments_in={len(repayments)}")
    vprint(f"[resolve] rule_version={rule_version} full_only={full_only}")

    if items:
        preview = [
            {
                "debt_id": x.debt_id,
                "opened_at": x.opened_at,
                "debtor": x.debtor,
                "creditor": x.creditor,
                "currency": x.currency,
                "item_type": x.item_type,
                "open_amount": x.open_amount,
                "ledger_status": x.ledger_status,
            }
            for x in items[:10]
        ]
        vprint("[resolve] first open items preview:")
        for row in preview:
            vprint("  ", row)
    else:
        vprint("[resolve] WARNING: no open items were provided to resolver")

    by_key: Dict[Tuple[str, str, str], List[OpenItem]] = {}
    for item in items:
        key = (item.debtor, item.creditor, item.currency)
        by_key.setdefault(key, []).append(item)
        timeline.append(
            TimelineEvent(
                event_date=item.opened_at,
                event_kind="debt_opened",
                tx_id=item.source_tx_id,
                debt_id=item.debt_id,
                debtor=item.debtor,
                creditor=item.creditor,
                currency=item.currency,
                item_type=item.item_type,
                amount=item.original_amount,
                note=f"ledger_status={item.ledger_status or 'na'}",
            )
        )

    vprint(f"[resolve] candidate key count={len(by_key)}")
    if by_key:
        vprint("[resolve] candidate keys and counts:")
        for key, vals in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            total_open = sum(float(x.open_amount) for x in vals if x.engine_status != "closed")
            vprint(f"  key={key} n_items={len(vals)} total_open={total_open:.2f}")

    alloc_counter = 1

    for _, rep in repayments.iterrows():
        repayment_tx_id = str(rep["tx_id"])
        repayment_date = pd.to_datetime(rep["Date"]).date().isoformat()
        debtor = str(rep["debtor"])
        creditor = str(rep["creditor"])
        currency = str(rep["Currency"])
        remaining = float(rep["repayment_amount"])

        key = (debtor, creditor, currency)
        candidates = by_key.get(key, [])
        initial_amount = remaining
        n_allocs = 0

        vprint("\n[resolve] --- repayment start ---")
        vprint(
            f"[resolve] repayment_tx_id={repayment_tx_id} "
            f"date={repayment_date} debtor={debtor} creditor={creditor} "
            f"currency={currency} amount={initial_amount:.2f}"
        )
        vprint(f"[resolve] lookup_key={key} candidate_count={len(candidates)}")

        timeline.append(
            TimelineEvent(
                event_date=repayment_date,
                event_kind="repayment_event",
                tx_id=repayment_tx_id,
                debt_id="",
                debtor=debtor,
                creditor=creditor,
                currency=currency,
                item_type=VALID_REPAYMENT_TYPE,
                amount=initial_amount,
                note=f"status={rep.get('status', '')}",
            )
        )

        if not candidates:
            vprint("[resolve] no candidates found for this repayment key")
            repayment_events.append(
                RepaymentEvent(
                    repayment_tx_id=repayment_tx_id,
                    repayment_date=repayment_date,
                    debtor=debtor,
                    creditor=creditor,
                    currency=currency,
                    repayment_amount=initial_amount,
                    allocated_amount=0.0,
                    leftover_amount=initial_amount,
                    n_allocations=0,
                    rule_version=rule_version,
                )
            )
            continue

        for item in candidates:
            if remaining <= 0:
                vprint("[resolve] repayment fully consumed, break")
                break

            vprint(
                f"[resolve] inspecting debt_id={item.debt_id} "
                f"type={item.item_type} opened_at={item.opened_at} "
                f"open_amount={float(item.open_amount):.2f} "
                f"engine_status={item.engine_status}"
            )

            if item.engine_status == "closed":
                vprint("[resolve] skip: already closed")
                continue

            if item.open_amount <= 0:
                vprint("[resolve] skip: non-positive open_amount")
                continue

            needed = float(item.open_amount)

            if full_only and remaining < needed:
                vprint(
                    f"[resolve] skip: full_only=True and remaining={remaining:.2f} < needed={needed:.2f}"
                )
                continue

            alloc_amt = needed if full_only else min(remaining, needed)
            vprint(
                f"[resolve] allocate: alloc_amt={alloc_amt:.2f} "
                f"remaining_before={remaining:.2f}"
            )

            item.open_amount = float(item.open_amount - alloc_amt)
            remaining = float(remaining - alloc_amt)
            n_allocs += 1

            if abs(item.open_amount) < 1e-9:
                item.open_amount = 0.0
                item.engine_status = "closed"
                item.closed_at = repayment_date
                vprint(
                    f"[resolve] debt closed: debt_id={item.debt_id} closed_at={repayment_date}"
                )
            else:
                vprint(
                    f"[resolve] debt partially reduced: debt_id={item.debt_id} "
                    f"open_amount_now={item.open_amount:.2f}"
                )

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
            timeline.append(
                TimelineEvent(
                    event_date=repayment_date,
                    event_kind="allocation",
                    tx_id=repayment_tx_id,
                    debt_id=item.debt_id,
                    debtor=debtor,
                    creditor=creditor,
                    currency=currency,
                    item_type=item.item_type,
                    amount=alloc_amt,
                    related_tx_id=item.source_tx_id,
                    note="applied_to_debt",
                )
            )
            alloc_counter += 1

        vprint(
            f"[resolve] repayment end: allocated={initial_amount - remaining:.2f} "
            f"leftover={remaining:.2f} n_allocations={n_allocs}"
        )

        repayment_events.append(
            RepaymentEvent(
                repayment_tx_id=repayment_tx_id,
                repayment_date=repayment_date,
                debtor=debtor,
                creditor=creditor,
                currency=currency,
                repayment_amount=initial_amount,
                allocated_amount=initial_amount - remaining,
                leftover_amount=remaining,
                n_allocations=n_allocs,
                rule_version=rule_version,
            )
        )

    open_items_df = pd.DataFrame([asdict(x) for x in items])
    allocations_df = pd.DataFrame([asdict(x) for x in allocations])
    repayment_events_df = pd.DataFrame([asdict(x) for x in repayment_events])
    timeline_df = pd.DataFrame([asdict(x) for x in timeline]).sort_values(
        ["event_date", "event_kind", "tx_id", "debt_id"],
        ignore_index=True,
    )

    reconciliation_rows: List[StatusReconciliation] = []
    for item in items:
        ledger_closed = item.ledger_status == "cerrado"
        engine_closed = item.engine_status == "closed"
        if ledger_closed == engine_closed:
            note = "aligned"
        elif ledger_closed and not engine_closed:
            note = "ledger says cerrado but engine still leaves balance open"
        elif (not ledger_closed) and engine_closed:
            note = "engine closed via repayments but ledger is not marked cerrado"
        else:
            note = "status mismatch"

        reconciliation_rows.append(
            StatusReconciliation(
                debt_id=item.debt_id,
                source_tx_id=item.source_tx_id,
                debtor=item.debtor,
                creditor=item.creditor,
                currency=item.currency,
                item_type=item.item_type,
                ledger_status=item.ledger_status,
                engine_status=item.engine_status,
                original_amount=item.original_amount,
                open_amount=item.open_amount,
                closed_at=item.closed_at,
                reconciliation_note=note,
            )
        )

    reconciliation_df = pd.DataFrame([asdict(x) for x in reconciliation_rows])

    vprint("\n[resolve] ===== FINISH =====")
    vprint(
        f"[resolve] final shapes: open_items={len(open_items_df)} "
        f"allocations={len(allocations_df)} repayments={len(repayment_events_df)} "
        f"timeline={len(timeline_df)} reconciliation={len(reconciliation_df)}"
    )

    return open_items_df, allocations_df, repayment_events_df, timeline_df, reconciliation_df

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resolve internal debt chronologically from canonical ingest or an existing ledger CSV."
    )
    p.add_argument("--ledger-csv", default=None, help="Existing canonical ledger CSV to read directly")
    p.add_argument("--fixture", default=None)
    p.add_argument("--sheet-url", default=None)
    p.add_argument("--service-account", default=None)
    p.add_argument("--sheet-name", default="C. Long Ledger")
    p.add_argument("--write-dir", required=True)
    p.add_argument(
        "--repayment-statuses",
        default=DEFAULT_REPAYMENT_STATUSES,
        help="Statuses that count as effective repayment events, default: pagado",
    )
    p.add_argument("--exclude-household", action="store_true")
    p.add_argument("--full-only", action="store_true", help="Require full cancellation of each matched item")
    p.add_argument(
        "--currencies",
        default="USD",
        help="Optional currency filter for the debt-resolution slice, default: USD",
    )
    return p.parse_args()

def main() -> None:
    print("[DEBUG] entered main")
    args = parse_args()
    print(f"[DEBUG] args={args}")

    write_dir = Path(args.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] write_dir={write_dir}")

    df = load_debt_ledger(args)
    print(f"[DEBUG] loaded ledger rows={len(df)} cols={list(df.columns)}")

    open_items = build_open_items(df, verbose=True)
    print(f"[DEBUG] open_items={len(open_items)}")

    repayments = build_repayments(df, repayment_statuses=_parse_list_arg(args.repayment_statuses))
    print(f"[DEBUG] repayments={len(repayments)}")

    open_items_df, allocations_df, repayment_events_df, timeline_df, reconciliation_df = resolve_repayments(
        open_items=open_items,
        repayments=repayments,
        full_only=args.full_only,
        verbose=True,
    )

    print(
        "[DEBUG] outputs shapes:",
        len(open_items_df),
        len(allocations_df),
        len(repayment_events_df),
        len(timeline_df),
        len(reconciliation_df),
    )

    open_items_df.to_csv(write_dir / "debt_open_items.csv", index=False)
    allocations_df.to_csv(write_dir / "debt_allocations.csv", index=False)
    repayment_events_df.to_csv(write_dir / "debt_repayment_events.csv", index=False)
    timeline_df.to_csv(write_dir / "debt_resolution_timeline.csv", index=False)
    reconciliation_df.to_csv(write_dir / "debt_status_reconciliation.csv", index=False)

    print(f"Wrote: {write_dir / 'debt_open_items.csv'}")
    print(f"Wrote: {write_dir / 'debt_allocations.csv'}")
    print(f"Wrote: {write_dir / 'debt_repayment_events.csv'}")
    print(f"Wrote: {write_dir / 'debt_resolution_timeline.csv'}")
    print(f"Wrote: {write_dir / 'debt_status_reconciliation.csv'}")


if __name__ == "__main__":
    main()