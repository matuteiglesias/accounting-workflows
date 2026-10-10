"""Read-only, source-bound CCL snapshot export.

This module is deliberately independent from transaction valuation.  It exports
the authorized CCL CANON observations needed by monthly treasury projections,
with a content hash and fail-closed date/rate/provenance validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from calendar import monthrange
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterable


SNAPSHOT_COLUMNS = [
    "rate_date", "ars_per_usd_ccl", "rate_source", "rate_series", "source_reference",
]


class CCLSnapshotError(ValueError):
    """Raised when CCL source observations cannot be safely bound."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> bytes:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SNAPSHOT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise CCLSnapshotError(f"refusing to overwrite a different snapshot: {path}")
    path.write_bytes(content)
    return content


def build_snapshot(
    rows: Iterable[dict[str, Any]], *, cutoff_date: str, source_name: str,
    source_reference: str, output_dir: str | Path,
) -> dict[str, Path]:
    """Validate and write rows from CCL CANON through the governed cutoff."""
    try:
        cutoff = date.fromisoformat(cutoff_date)
    except ValueError as exc:
        raise CCLSnapshotError(f"invalid cutoff_date: {cutoff_date!r}") from exc
    accepted: list[dict[str, str]] = []
    excluded_after_cutoff = 0
    seen: set[str] = set()
    sources: set[str] = set()
    series: set[str] = set()
    raw_count = 0
    for row_number, raw in enumerate(rows, start=2):
        values = {key: str(raw.get(key, "") or "").strip() for key in SNAPSHOT_COLUMNS}
        if not any(values.values()):
            continue
        raw_count += 1
        try:
            parsed_date = date.fromisoformat(values["rate_date"])
        except ValueError as exc:
            raise CCLSnapshotError(f"invalid rate_date at source row {row_number}: {values['rate_date']!r}") from exc
        try:
            rate = Decimal(values["ars_per_usd_ccl"])
        except InvalidOperation as exc:
            raise CCLSnapshotError(f"invalid rate at source row {row_number}: {values['ars_per_usd_ccl']!r}") from exc
        if not rate.is_finite() or rate <= 0:
            raise CCLSnapshotError(f"rate must be positive and finite at source row {row_number}")
        if not values["rate_source"] or not values["rate_series"] or not values["source_reference"]:
            raise CCLSnapshotError(f"missing provenance at source row {row_number}")
        if parsed_date > cutoff:
            excluded_after_cutoff += 1
            continue
        key = parsed_date.isoformat()
        if key in seen:
            raise CCLSnapshotError(f"duplicate CCL observation for rate_date={key}")
        seen.add(key)
        values["rate_date"] = key
        values["ars_per_usd_ccl"] = format(rate, "f")
        accepted.append(values)
        sources.add(values["rate_source"])
        series.add(values["rate_series"])
    if not accepted:
        raise CCLSnapshotError("no CCL observations on or before cutoff")
    if len(sources) != 1 or len(series) != 1:
        raise CCLSnapshotError("CCL snapshot must have exactly one source and one series")
    accepted.sort(key=lambda row: row["rate_date"])
    out = Path(output_dir)
    snapshot = out / "ccl_rates_snapshot.csv"
    content = _write_csv(snapshot, accepted)
    monthly: list[dict[str, str]] = []
    non_business_day_observations = 0
    buckets: dict[str, list[Decimal]] = {}
    for row in accepted:
        buckets.setdefault(row["rate_date"][:7], []).append(Decimal(row["ars_per_usd_ccl"]))
    for period, values in sorted(buckets.items()):
        period_rows = [row for row in accepted if row["rate_date"].startswith(period)]
        business_rows = [row for row in period_rows if date.fromisoformat(row["rate_date"]).weekday() < 5]
        non_business_day_observations += len(period_rows) - len(business_rows)
        values = [Decimal(row["ars_per_usd_ccl"]) for row in business_rows]
        if not values:
            raise CCLSnapshotError(f"no business-day CCL observations for {period}")
        with localcontext() as context:
            context.prec = 40
            mean = sum(values, Decimal(0)) / Decimal(len(values))
        period_dates = [row["rate_date"] for row in business_rows]
        period_end = monthrange(int(period[:4]), int(period[5:7]))[1]
        is_partial = period == cutoff_date[:7] and int(max(period_dates)[8:10]) < period_end
        monthly.append({
            "period": period,
            "observations": str(len(values)),
            "rate_min_date": min(row["rate_date"] for row in accepted if row["rate_date"].startswith(period)),
            "rate_max_date": max(row["rate_date"] for row in accepted if row["rate_date"].startswith(period)),
            "ccl_mean_ars_per_usd": format(mean, ".12f"),
            "coverage_status": "partial" if is_partial else "complete",
        })
    monthly_path = out / "monthly_ccl_mean.csv"
    monthly_content = _write_simple_csv(monthly_path, list(monthly[0]), monthly)
    manifest = {
        "schema_version": 1,
        "source_name": source_name,
        "source_reference": source_reference,
        "source_reference_sha256": _sha256_bytes(source_reference.encode()),
        "cutoff_date": cutoff.isoformat(),
        "raw_observation_count": raw_count,
        "excluded_after_cutoff": excluded_after_cutoff,
        "accepted_observation_count": len(accepted),
        "non_business_day_observations_excluded_from_mean": non_business_day_observations,
        "rate_source": next(iter(sources)),
        "rate_series": next(iter(series)),
        "rate_min_date": accepted[0]["rate_date"],
        "rate_max_date": accepted[-1]["rate_date"],
        "snapshot_sha256": _sha256_bytes(content),
        "monthly_mean_sha256": _sha256_bytes(monthly_content),
    }
    manifest_path = out / "ccl_snapshot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"snapshot": snapshot, "monthly_mean": monthly_path, "manifest": manifest_path}


def _write_simple_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> bytes:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    if path.exists() and path.read_bytes() != content:
        raise CCLSnapshotError(f"refusing to overwrite a different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    return content


def export_sheet_snapshot(*, sheet_url: str, service_account: str, sheet_name: str,
                          cutoff_date: str, output_dir: str | Path) -> dict[str, Path]:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    match = re.search(r"/spreadsheets/d/([^/]+)", sheet_url)
    if not match:
        raise CCLSnapshotError("sheet_url does not contain a spreadsheet id")
    service = build("sheets", "v4", credentials=Credentials.from_service_account_file(
        service_account, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    ), cache_discovery=False)
    values = service.spreadsheets().values().get(
        spreadsheetId=match.group(1), range=f"'{sheet_name}'!M1:Q"
    ).execute().get("values", [])
    if not values:
        raise CCLSnapshotError(f"sheet tab has no values: {sheet_name}")
    header = values[0]
    if header != SNAPSHOT_COLUMNS:
        raise CCLSnapshotError(f"unexpected CCL CANON header: {header!r}")
    rows = [dict(zip(header, list(row) + [""] * (len(header) - len(row)))) for row in values[1:]]
    return build_snapshot(rows, cutoff_date=cutoff_date, source_name=sheet_name,
                          source_reference=f"google-sheet:{match.group(1)}:{sheet_name}!M:Q",
                          output_dir=output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a private CCL CANON snapshot")
    parser.add_argument("--sheet-url", required=True); parser.add_argument("--service-account", required=True)
    parser.add_argument("--sheet-name", default="CCL CANON"); parser.add_argument("--cutoff-date", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for name, path in export_sheet_snapshot(sheet_url=args.sheet_url, service_account=args.service_account,
                                             sheet_name=args.sheet_name, cutoff_date=args.cutoff_date,
                                             output_dir=args.output_dir).items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
