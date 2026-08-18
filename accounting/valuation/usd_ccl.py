"""Build a deterministic, fixture-safe USD/CCL valuation sidecar.

This stage is deliberately separate from canonical ingest.  It accepts only
local files, reads the canonical ledger without modifying it, and implements the
minimal v1 policy: native USD identity, exact-date ARS conversion, and no
fallback for missing dates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from io import StringIO
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


SCHEMA_VERSION = 1
IMPLEMENTATION_ID = "usd_ccl_exact_date_sidecar_v1"
SIDECAR_COLUMNS = [
    "tx_id",
    "valuation_basis",
    "valuation_currency",
    "amount_usd_ccl",
    "fx_rate_to_usd_ccl",
    "fx_rate_date",
    "fx_rate_source",
    "fx_rate_policy",
    "fx_conversion_status",
    "fx_rate_age_days",
    "fx_rate_source_reference",
]
RATE_COLUMNS = [
    "rate_date",
    "ars_per_usd_ccl",
    "rate_source",
    "rate_series",
    "fx_rate_source_reference",
]
NATIVE_FINGERPRINT_COLUMNS = [
    "tx_id",
    "Date",
    "amount",
    "amount_cents",
    "Currency",
    "payer",
    "receiver",
    "Flujo",
    "Tipo",
    "status",
    "Box",
    "source_file",
    "source_row",
    "notes",
]
COUNTER_FIELDS = [
    "native_usd_identity_rows",
    "exact_matches",
    "missing_rates",
    "unsupported_currency_rows",
    "invalid_native_rows",
]


class ValuationContractError(ValueError):
    """Raised when an input violates the fixture-only valuation contract."""


@dataclass(frozen=True)
class RateObservation:
    rate_date: str
    ars_per_usd_ccl: Decimal
    rate_source: str
    rate_series: str
    source_reference: str


def _require_local_file(value: str | Path, *, name: str) -> Path:
    raw = str(value)
    parsed = urlparse(raw)
    if parsed.scheme or parsed.netloc or "://" in raw:
        raise ValuationContractError(f"{name} must be a local filesystem path, not a URL/URI: {raw}")
    path = Path(raw)
    if not path.is_file():
        raise ValuationContractError(f"{name} does not exist or is not a file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValuationContractError(f"CSV has no header: {path}")
        rows = [{key: (value or "") for key, value in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def _parse_iso_date(value: str, *, field: str, row_number: int) -> str:
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValuationContractError(f"invalid {field} at row {row_number}: {value!r}") from exc


def _parse_decimal(value: str, *, field: str, row_number: int) -> Decimal:
    try:
        parsed = Decimal(value.strip())
    except (AttributeError, InvalidOperation) as exc:
        raise ValuationContractError(f"invalid {field} at row {row_number}: {value!r}") from exc
    if not parsed.is_finite():
        raise ValuationContractError(f"non-finite {field} at row {row_number}: {value!r}")
    return parsed


def _load_policy(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        policy = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValuationContractError(f"invalid policy JSON: {path}") from exc
    required = {
        "valuation_policy_id": str,
        "valuation_basis": str,
        "valuation_currency": str,
        "matching_policy": str,
        "ars_quote_convention": str,
        "amount_decimal_places": int,
        "rate_decimal_places": int,
    }
    for field, expected_type in required.items():
        if field not in policy or not isinstance(policy[field], expected_type):
            raise ValuationContractError(f"policy field {field!r} must be {expected_type.__name__}")
    expected = {
        "valuation_basis": "usd_ccl",
        "valuation_currency": "USD",
        "matching_policy": "exact_date_only",
        "ars_quote_convention": "ars_per_usd_ccl",
    }
    for field, expected_value in expected.items():
        if policy[field] != expected_value:
            raise ValuationContractError(
                f"unsupported policy {field}={policy[field]!r}; expected {expected_value!r}"
            )
    for field in ["amount_decimal_places", "rate_decimal_places"]:
        if not 0 <= policy[field] <= 18:
            raise ValuationContractError(f"policy field {field!r} must be between 0 and 18")
    return policy, hashlib.sha256(raw).hexdigest()


def _load_rates(path: Path) -> tuple[dict[str, RateObservation], dict[str, Any]]:
    columns, rows = _read_csv(path)
    missing = [column for column in RATE_COLUMNS if column not in columns]
    if missing:
        raise ValuationContractError(f"rate snapshot missing columns: {missing}")
    if not rows:
        raise ValuationContractError("rate snapshot has no observations")

    observations: dict[str, RateObservation] = {}
    sources: set[str] = set()
    series: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        rate_date = _parse_iso_date(row["rate_date"], field="rate_date", row_number=row_number)
        rate = _parse_decimal(
            row["ars_per_usd_ccl"], field="ars_per_usd_ccl", row_number=row_number
        )
        if rate <= 0:
            raise ValuationContractError(
                f"ars_per_usd_ccl must be positive at row {row_number}: {row['ars_per_usd_ccl']!r}"
            )
        source = row["rate_source"].strip()
        rate_series = row["rate_series"].strip()
        reference = row["fx_rate_source_reference"].strip()
        if not source or not rate_series or not reference:
            raise ValuationContractError(f"blank rate provenance at row {row_number}")
        if rate_date in observations:
            raise ValuationContractError(
                f"duplicate canonical rate observation key rate_date={rate_date!r}"
            )
        observations[rate_date] = RateObservation(
            rate_date=rate_date,
            ars_per_usd_ccl=rate,
            rate_source=source,
            rate_series=rate_series,
            source_reference=reference,
        )
        sources.add(source)
        series.add(rate_series)

    if len(sources) != 1 or len(series) != 1:
        raise ValuationContractError(
            "rate snapshot must contain exactly one rate_source and one rate_series"
        )
    ordered_dates = sorted(observations)
    return observations, {
        "rate_source": next(iter(sources)),
        "rate_series": next(iter(series)),
        "rate_raw_observation_count": len(rows),
        "rate_observation_count": len(observations),
        "rate_rejected_observation_count": 0,
        "rate_min_date": ordered_dates[0],
        "rate_max_date": ordered_dates[-1],
    }


def _validate_ledger(columns: list[str], rows: list[dict[str, str]]) -> None:
    required = ["tx_id", "Date", "amount", "Currency"]
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValuationContractError(f"canonical ledger missing columns: {missing}")
    if not rows:
        raise ValuationContractError("canonical ledger has no rows")
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        tx_id = row["tx_id"].strip()
        if not tx_id:
            raise ValuationContractError(f"blank tx_id at ledger row {row_number}")
        if tx_id in seen:
            raise ValuationContractError(f"duplicate tx_id in canonical ledger: {tx_id!r}")
        seen.add(tx_id)


def _native_business_fingerprint(columns: list[str], rows: list[dict[str, str]]) -> str:
    selected = [column for column in NATIVE_FINGERPRINT_COLUMNS if column in columns]
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=selected, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: item["tx_id"]):
        writer.writerow({column: row.get(column, "") for column in selected})
    return _sha256_text(buffer.getvalue())


def _quantum(decimal_places: int) -> Decimal:
    return Decimal(1).scaleb(-decimal_places)


def _format_decimal(value: Decimal, decimal_places: int) -> str:
    return format(value.quantize(_quantum(decimal_places), rounding=ROUND_HALF_EVEN), "f")


def _sidecar_rows(
    ledger_rows: list[dict[str, str]],
    rates: dict[str, RateObservation],
    policy: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    counters = {field: 0 for field in COUNTER_FIELDS}
    output: list[dict[str, str]] = []
    amount_places = int(policy["amount_decimal_places"])
    rate_places = int(policy["rate_decimal_places"])

    for row in sorted(ledger_rows, key=lambda item: item["tx_id"]):
        tx_id = row["tx_id"].strip()
        currency = row["Currency"].strip().upper()
        tx_date = row["Date"].strip()
        base = {
            "tx_id": tx_id,
            "valuation_basis": "usd_ccl",
            "valuation_currency": "USD",
            "amount_usd_ccl": "",
            "fx_rate_to_usd_ccl": "",
            "fx_rate_date": "",
            "fx_rate_source": "",
            "fx_rate_policy": policy["valuation_policy_id"],
            "fx_conversion_status": "",
            "fx_rate_age_days": "",
            "fx_rate_source_reference": "",
        }
        try:
            amount = _parse_decimal(row["amount"], field="amount", row_number=0)
        except ValuationContractError:
            base["fx_conversion_status"] = "invalid_native_amount"
            counters["invalid_native_rows"] += 1
            output.append(base)
            continue

        if currency in {"USD", "ARS"}:
            try:
                tx_date = _parse_iso_date(tx_date, field="Date", row_number=0)
            except ValuationContractError:
                base["fx_conversion_status"] = "invalid_native_amount"
                counters["invalid_native_rows"] += 1
                output.append(base)
                continue

        if currency == "USD":
            base.update(
                {
                    "amount_usd_ccl": _format_decimal(amount, amount_places),
                    "fx_rate_to_usd_ccl": _format_decimal(Decimal(1), rate_places),
                    "fx_rate_date": tx_date,
                    "fx_rate_source": "native_identity",
                    "fx_conversion_status": "identity_native_usd",
                    "fx_rate_age_days": "0",
                    "fx_rate_source_reference": "native_identity",
                }
            )
            counters["native_usd_identity_rows"] += 1
        elif currency == "ARS":
            observation = rates.get(tx_date)
            if observation is None:
                base["fx_conversion_status"] = "unavailable_missing_rate"
                counters["missing_rates"] += 1
            else:
                with localcontext() as context:
                    context.prec = max(36, amount_places + rate_places + 12)
                    projected = amount / observation.ars_per_usd_ccl
                    rate_to_usd = Decimal(1) / observation.ars_per_usd_ccl
                base.update(
                    {
                        "amount_usd_ccl": _format_decimal(projected, amount_places),
                        "fx_rate_to_usd_ccl": _format_decimal(rate_to_usd, rate_places),
                        "fx_rate_date": observation.rate_date,
                        "fx_rate_source": observation.rate_source,
                        "fx_conversion_status": "converted_exact_date",
                        "fx_rate_age_days": "0",
                        "fx_rate_source_reference": observation.source_reference,
                    }
                )
                counters["exact_matches"] += 1
        else:
            base["fx_conversion_status"] = "unsupported_currency"
            counters["unsupported_currency_rows"] += 1
        output.append(base)
    return output, counters


def _csv_bytes(columns: list[str], rows: Iterable[dict[str, Any]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_csv_immutable(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _csv_bytes(columns, rows)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise ValuationContractError(
            f"refusing to overwrite existing valuation artifact with different bytes: {path}"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_identity(repo_root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def build_usd_ccl_valuation(
    *,
    ledger_path: str | Path,
    rates_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    source_scope_tag: str,
    expected_rate_sha256: str | None = None,
) -> dict[str, Path]:
    ledger = _require_local_file(ledger_path, name="ledger")
    rates_file = _require_local_file(rates_path, name="rates")
    policy_file = _require_local_file(policy_path, name="policy")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger_sha = _sha256(ledger)
    rate_sha = _sha256(rates_file)
    if expected_rate_sha256 and rate_sha != expected_rate_sha256:
        raise ValuationContractError(
            f"rate artifact SHA mismatch: expected={expected_rate_sha256}; actual={rate_sha}"
        )
    ledger_columns, ledger_rows = _read_csv(ledger)
    _validate_ledger(ledger_columns, ledger_rows)
    policy, policy_sha = _load_policy(policy_file)
    rates, rate_metadata = _load_rates(rates_file)
    sidecar_rows, counters = _sidecar_rows(ledger_rows, rates, policy)

    if len(sidecar_rows) != len(ledger_rows):
        raise AssertionError("valuation row count does not match canonical ledger")
    source_ids = {row["tx_id"].strip() for row in ledger_rows}
    sidecar_ids = {row["tx_id"] for row in sidecar_rows}
    if len(sidecar_ids) != len(sidecar_rows) or source_ids != sidecar_ids:
        raise AssertionError("valuation sidecar failed 1:1 tx_id coverage")
    if sum(counters.values()) != len(sidecar_rows):
        raise AssertionError("valuation status counters do not reconcile to valuation rows")

    sidecar_path = out_dir / "ledger_valuation_usd_ccl.csv"
    validation_path = out_dir / "valuation_validation.json"
    manifest_path = out_dir / "valuation_manifest.json"

    _write_csv_immutable(sidecar_path, SIDECAR_COLUMNS, sidecar_rows)
    sidecar_sha = _sha256(sidecar_path)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "checks": [
            {"name": "source_tx_id_non_null_unique", "status": "pass"},
            {"name": "sidecar_tx_id_non_null_unique", "status": "pass"},
            {"name": "ledger_to_sidecar_anti_join_empty", "status": "pass"},
            {"name": "sidecar_to_ledger_anti_join_empty", "status": "pass"},
            {"name": "status_counters_reconcile", "status": "pass"},
            {"name": "rate_observation_keys_unique", "status": "pass"},
            {"name": "network_lookup_disabled", "status": "pass"},
        ],
        "valuation_rows": len(sidecar_rows),
        **counters,
    }
    _write_json_atomic(validation_path, validation)

    repo_root = Path(__file__).resolve().parents[2]
    code_revision, code_dirty = _git_identity(repo_root)
    identity_payload = "|".join(
        [ledger_sha, rate_sha, policy_sha, str(SCHEMA_VERSION), IMPLEMENTATION_ID]
    )
    valuation_id = _sha256_text(identity_payload)
    manifest = {
        "stage": "V.usd_ccl_valuation",
        "mode": "smoke",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "valuation_schema_version": SCHEMA_VERSION,
        "valuation_policy_id": policy["valuation_policy_id"],
        "valuation_policy_sha256": policy_sha,
        "source_ledger_artifact": str(ledger),
        "source_ledger_sha256": ledger_sha,
        "source_ledger_business_fingerprint": _native_business_fingerprint(
            ledger_columns, ledger_rows
        ),
        "source_ledger_row_count": len(ledger_rows),
        "source_scope_tag": source_scope_tag,
        "source_population": "recognized",
        "rate_artifact": str(rates_file),
        "rate_artifact_sha256": rate_sha,
        **rate_metadata,
        "valuation_rows": len(sidecar_rows),
        **counters,
        "valuation_artifact": str(sidecar_path),
        "valuation_artifact_sha256": sidecar_sha,
        "valuation_validation_artifact": str(validation_path),
        "valuation_validation_sha256": _sha256(validation_path),
        "valuation_id": valuation_id,
        "code_revision": code_revision,
        "code_dirty": code_dirty,
        "implementation_id": IMPLEMENTATION_ID,
        "generated_by_network_access": False,
    }
    _write_json_atomic(manifest_path, manifest)
    return {
        "sidecar": sidecar_path,
        "manifest": manifest_path,
        "validation": validation_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline exact-date USD/CCL valuation sidecar"
    )
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--rates", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="smoke-usd-ccl-valuation")
    parser.add_argument("--source-scope-tag", default="FBPM")
    parser.add_argument("--expected-rate-sha256", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_usd_ccl_valuation(
        ledger_path=args.ledger,
        rates_path=args.rates,
        policy_path=args.policy,
        output_dir=args.output_dir,
        run_id=args.run_id,
        source_scope_tag=args.source_scope_tag,
        expected_rate_sha256=args.expected_rate_sha256 or None,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
