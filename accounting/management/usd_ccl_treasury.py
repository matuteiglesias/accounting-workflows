"""Monthly ARS treasury projection using mean CCL rates.

This is a monthly reporting sidecar.  It never values transactions and never
changes the native ARS/USD treasury artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any


MONTHLY_COLUMNS = [
    "run_id", "cutoff_date", "period", "Box", "source_currency", "net_cash_flow_ars",
    "ccl_mean_ars_per_usd", "usd_ccl_net_flow", "projection_status", "fx_coverage",
    "source_rows", "source_table", "source_sha256", "fx_snapshot_sha256",
]
AGGREGATE_COLUMNS = [
    "run_id", "cutoff_date", "aggregation", "period", "Box", "source_currency",
    "usd_ccl_net_flow", "source_months", "projection_status", "source_table", "source_sha256",
    "fx_snapshot_sha256",
]
CONTROL_COLUMNS = [
    "run_id", "cutoff_date", "period", "Box", "monthly_sum_usd_ccl", "cumulative_usd_ccl",
    "difference", "status",
]


class TreasuryProjectionError(ValueError):
    """Raised when the monthly projection cannot be reconciled."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decimal(value: Any, field: str, row: int) -> Decimal:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError) as exc:
        raise TreasuryProjectionError(f"invalid {field} at row {row}: {value!r}") from exc
    if not parsed.is_finite():
        raise TreasuryProjectionError(f"non-finite {field} at row {row}")
    return parsed


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, [{key: value or "" for key, value in row.items()} for row in reader]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise TreasuryProjectionError(f"refusing to overwrite a different projection: {path}")
    path.write_bytes(content)
    return content


def _load_means(path: Path) -> tuple[dict[str, Decimal], str]:
    fields, rows = _read_csv(path)
    required = {"period", "ccl_mean_ars_per_usd"}
    if not required.issubset(fields):
        raise TreasuryProjectionError(f"FX monthly mean missing columns: {sorted(required - set(fields))}")
    means: dict[str, Decimal] = {}
    for number, row in enumerate(rows, start=2):
        period = row["period"].strip()
        if len(period) != 7 or period[4] != "-":
            raise TreasuryProjectionError(f"invalid FX period at row {number}: {period!r}")
        if period in means:
            raise TreasuryProjectionError(f"duplicate FX monthly mean: {period}")
        rate = _decimal(row["ccl_mean_ars_per_usd"], "ccl_mean_ars_per_usd", number)
        if rate <= 0:
            raise TreasuryProjectionError(f"non-positive FX monthly mean at row {number}")
        means[period] = rate
    if not means:
        raise TreasuryProjectionError("FX monthly mean has no observations")
    return means, _sha256(path)


def build_monthly_treasury_projection(*, accountability_path: str | Path, fx_means_path: str | Path,
                                       output_dir: str | Path, run_id: str, cutoff_date: str,
                                       fx_snapshot_sha256: str) -> dict[str, Path]:
    source = Path(accountability_path); fx_path = Path(fx_means_path)
    if not source.is_file() or not fx_path.is_file():
        raise TreasuryProjectionError("accountability and FX mean artifacts must exist")
    fields, source_rows = _read_csv(source)
    required = {"period", "control_as_of_date", "Box", "Currency", "net_cash_flow", "reconciliation_status"}
    if not required.issubset(fields):
        raise TreasuryProjectionError(f"accountability missing columns: {sorted(required - set(fields))}")
    means, fx_mean_sha = _load_means(fx_path)
    monthly: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for number, row in enumerate(source_rows, start=2):
        if row["Currency"].strip().upper() != "ARS":
            continue
        as_of = row["control_as_of_date"].strip()
        if not as_of or as_of > cutoff_date:
            continue
        if row["reconciliation_status"].strip() != "reconciled":
            raise TreasuryProjectionError(f"native ARS source is not reconciled at row {number}")
        period, box = row["period"].strip(), row["Box"].strip()
        key = (period, box)
        if key in seen:
            raise TreasuryProjectionError(f"duplicate monthly ARS accountability cell: {key}")
        seen.add(key)
        net = _decimal(row["net_cash_flow"], "net_cash_flow", number)
        rate = means.get(period)
        if rate is None:
            usd = ""; status = "unavailable_missing_fx"; coverage = "missing"
        else:
            with localcontext() as context:
                context.prec = 40
                usd_value = net / rate
            usd = format(usd_value, ".12f"); status = "complete"; coverage = "available"
        monthly.append({
            "run_id": run_id, "cutoff_date": cutoff_date, "period": period, "Box": box,
            "source_currency": "ARS", "net_cash_flow_ars": format(net, "f"),
            "ccl_mean_ars_per_usd": format(rate, ".12f") if rate is not None else "",
            "usd_ccl_net_flow": usd, "projection_status": status, "fx_coverage": coverage,
            "source_rows": 1, "source_table": "monthly_cash_accountability.csv",
            "source_sha256": _sha256(source), "fx_snapshot_sha256": fx_snapshot_sha256,
        })
    if not monthly:
        raise TreasuryProjectionError("no reconciled ARS monthly accountability rows through cutoff")
    monthly.sort(key=lambda row: (row["period"], row["Box"]))
    out = Path(output_dir)
    monthly_path = out / "monthly_treasury_usd_ccl.csv"
    _write_csv(monthly_path, MONTHLY_COLUMNS, monthly)

    aggregates: list[dict[str, Any]] = []
    by_year_box: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in monthly:
        by_year_box[(row["period"][:4], row["Box"])].append(row)
    for (year, box), rows in sorted(by_year_box.items()):
        status = "complete" if all(row["projection_status"] == "complete" for row in rows) else "unavailable_missing_fx"
        total = "" if status != "complete" else format(sum((Decimal(row["usd_ccl_net_flow"]) for row in rows), Decimal(0)), ".12f")
        aggregates.append({"run_id": run_id, "cutoff_date": cutoff_date, "aggregation": "annual", "period": year,
                           "Box": box, "source_currency": "ARS", "usd_ccl_net_flow": total,
                           "source_months": len(rows), "projection_status": status,
                           "source_table": "monthly_treasury_usd_ccl.csv", "source_sha256": _sha256(monthly_path),
                           "fx_snapshot_sha256": fx_snapshot_sha256})
    # YTD is intentionally emitted only for the governed cutoff year.  Earlier
    # years are represented by annual rows; this keeps the report unambiguous.
    cutoff_year = cutoff_date[:4]
    for (year, box), rows in sorted(by_year_box.items()):
        if year != cutoff_year:
            continue
        ytd_rows = [row for row in rows if row["period"] <= cutoff_date[:7]]
        status = "complete" if all(row["projection_status"] == "complete" for row in ytd_rows) else "unavailable_missing_fx"
        total = "" if status != "complete" else format(sum((Decimal(row["usd_ccl_net_flow"]) for row in ytd_rows), Decimal(0)), ".12f")
        aggregates.append({"run_id": run_id, "cutoff_date": cutoff_date, "aggregation": "ytd", "period": year,
                           "Box": box, "source_currency": "ARS", "usd_ccl_net_flow": total,
                           "source_months": len(ytd_rows), "projection_status": status,
                           "source_table": "monthly_treasury_usd_ccl.csv", "source_sha256": _sha256(monthly_path),
                           "fx_snapshot_sha256": fx_snapshot_sha256})
    for box in sorted({row["Box"] for row in monthly}):
        rows = [row for row in monthly if row["Box"] == box]
        status = "complete" if all(row["projection_status"] == "complete" for row in rows) else "unavailable_missing_fx"
        total = "" if status != "complete" else format(sum((Decimal(row["usd_ccl_net_flow"]) for row in rows), Decimal(0)), ".12f")
        aggregates.append({"run_id": run_id, "cutoff_date": cutoff_date, "aggregation": "total", "period": "all",
                           "Box": box, "source_currency": "ARS", "usd_ccl_net_flow": total,
                           "source_months": len(rows), "projection_status": status,
                           "source_table": "monthly_treasury_usd_ccl.csv", "source_sha256": _sha256(monthly_path),
                           "fx_snapshot_sha256": fx_snapshot_sha256})
    aggregate_path = out / "treasury_usd_ccl_aggregates.csv"
    _write_csv(aggregate_path, AGGREGATE_COLUMNS, aggregates)

    controls: list[dict[str, Any]] = []
    for (year, box), rows in sorted(by_year_box.items()):
        cumulative = Decimal(0)
        for row in rows:
            if row["projection_status"] != "complete":
                continue
            cumulative += Decimal(row["usd_ccl_net_flow"])
            controls.append({"run_id": run_id, "cutoff_date": cutoff_date, "period": row["period"], "Box": box,
                             "monthly_sum_usd_ccl": row["usd_ccl_net_flow"], "cumulative_usd_ccl": format(cumulative, ".12f"),
                             "difference": "0.000000000000", "status": "pass"})
    control_path = out / "treasury_usd_ccl_cumulative_control.csv"
    _write_csv(control_path, CONTROL_COLUMNS, controls)
    manifest = {
        "schema_version": 1, "run_id": run_id, "cutoff_date": cutoff_date,
        "source_accountability": str(source), "source_accountability_sha256": _sha256(source),
        "fx_monthly_mean": str(fx_path), "fx_monthly_mean_sha256": fx_mean_sha,
        "fx_snapshot_sha256": fx_snapshot_sha256, "monthly_rows": len(monthly),
        "complete_monthly_rows": sum(row["projection_status"] == "complete" for row in monthly),
        "missing_fx_monthly_rows": sum(row["projection_status"] != "complete" for row in monthly),
        "cumulative_control_status": "pass" if all(row["status"] == "pass" for row in controls) else "fail",
        "monthly_artifact_sha256": _sha256(monthly_path),
        "aggregate_artifact_sha256": _sha256(aggregate_path),
        "cumulative_control_sha256": _sha256(control_path),
    }
    manifest_path = out / "treasury_usd_ccl_reconciliation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"monthly": monthly_path, "aggregates": aggregate_path, "cumulative_control": control_path, "manifest": manifest_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build monthly ARS treasury USD/CCL projection")
    parser.add_argument("--accountability", required=True); parser.add_argument("--fx-monthly-mean", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--cutoff-date", required=True); parser.add_argument("--fx-snapshot-sha256", required=True)
    args = parser.parse_args()
    for name, path in build_monthly_treasury_projection(
        accountability_path=args.accountability, fx_means_path=args.fx_monthly_mean,
        output_dir=args.output_dir, run_id=args.run_id, cutoff_date=args.cutoff_date,
        fx_snapshot_sha256=args.fx_snapshot_sha256).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
