#!/usr/bin/env python3
"""Compare two canonical all-status ledgers by stable source transaction id."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IGNORED = {"tx_id", "ingest_ts", "source_row"}


def _change_class(changed: set[str]) -> str:
    if changed <= {"notes", "Detalle", "Lugar", "issuer", "account_id", "tag"}:
        return "actor grooming" if changed & {"notes", "Detalle"} else "descriptive grooming"
    if changed == {"Box"}:
        return "Box correction"
    if changed == {"status"}:
        return "status correction"
    if changed & {"amount", "amount_cents", "Date", "Currency"}:
        return "amount/date correction"
    if changed & {"payer", "receiver"}:
        return "actor grooming"
    if changed & {"Flujo", "Tipo"}:
        return "source correction"
    return "other"


def build_diff(old_path: Path, new_path: Path) -> pd.DataFrame:
    old = pd.read_csv(old_path, dtype=str).fillna("")
    new = pd.read_csv(new_path, dtype=str).fillna("")
    key = "transaction_id"
    if key not in old or key not in new or old[key].duplicated().any() or new[key].duplicated().any():
        raise ValueError("transaction_id must be present and unique in both ledgers")
    common = [c for c in old.columns if c in new.columns and c not in IGNORED | {key}]
    merged = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        membership = row["_merge"]
        changed = {c for c in common if membership != "both" or row.get(f"{c}_old", "") != row.get(f"{c}_new", "")}
        if membership == "both" and not changed:
            continue
        classification = "added" if membership == "right_only" else "removed" if membership == "left_only" else _change_class(changed)
        old_amount = pd.to_numeric(pd.Series([row.get("amount_old", 0)]), errors="coerce").fillna(0).iloc[0]
        new_amount = pd.to_numeric(pd.Series([row.get("amount_new", 0)]), errors="coerce").fillna(0).iloc[0]
        rows.append({
            "transaction_id": row[key], "old_tx_id": row.get("tx_id_old", ""), "new_tx_id": row.get("tx_id_new", ""),
            "old_source_row": row.get("source_row_old", ""), "new_source_row": row.get("source_row_new", ""),
            "change_class": classification, "changed_fields": ";".join(sorted(changed)),
            "old_date": row.get("Date_old", ""), "new_date": row.get("Date_new", ""),
            "old_box": row.get("Box_old", ""), "new_box": row.get("Box_new", ""),
            "old_status": row.get("status_old", ""), "new_status": row.get("status_new", ""),
            "old_amount": old_amount, "new_amount": new_amount, "amount_delta": new_amount-old_amount,
            "old_currency": row.get("Currency_old", ""), "new_currency": row.get("Currency_new", ""),
            "old_payer": row.get("payer_old", ""), "new_payer": row.get("payer_new", ""),
            "old_receiver": row.get("receiver_old", ""), "new_receiver": row.get("receiver_new", ""),
            "old_flujo": row.get("Flujo_old", ""), "new_flujo": row.get("Flujo_new", ""),
            "old_tipo": row.get("Tipo_old", ""), "new_tipo": row.get("Tipo_new", ""),
            "old_detalle": row.get("Detalle_old", ""), "new_detalle": row.get("Detalle_new", ""),
            "old_lugar": row.get("Lugar_old", ""), "new_lugar": row.get("Lugar_new", ""),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_diff(args.old, args.new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"rows={len(result)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
