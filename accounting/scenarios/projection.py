"""Select one scenario's analytic ledger without changing historical evidence.

The existing accounting producers consume this ledger unchanged. Only the
scenario execution is allowed to include forward-looking `planeado` rows.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path


def build_projection_ledger(
    paid_path: Path, all_status_path: Path, output: Path, *,
    as_of_date: str, through_date: str,
) -> dict[str, int]:
    start, end = date.fromisoformat(as_of_date), date.fromisoformat(through_date)
    if end < start:
        raise ValueError("projection through_date precedes as_of_date")
    with Path(paid_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or not {"tx_id", "Date", "status"}.issubset(fields):
            raise ValueError("invalid canonical paid ledger schema")
        paid = list(reader)
    if any(x["status"].strip().casefold() != "pagado" for x in paid):
        raise ValueError("base canonical ledger must contain paid rows only")
    if any(date.fromisoformat(x["Date"][:10]) > start for x in paid):
        raise ValueError("a future paid entry cannot be treated as observed")
    with Path(all_status_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or ()) != set(fields):
            raise ValueError("canonical and all-status schemas differ")
        selected = [
            x for x in reader
            if x["status"].strip().casefold() == "planeado"
            and start <= date.fromisoformat(x["Date"][:10]) <= end
        ]
    tx_ids = [x["tx_id"] for x in paid + selected]
    if len(set(tx_ids)) != len(tx_ids):
        raise ValueError("duplicate tx_id in projected analytic ledger")
    with Path(output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paid + selected)
    return {"paid": len(paid), "planned": len(selected), "projected_total": len(tx_ids)}
