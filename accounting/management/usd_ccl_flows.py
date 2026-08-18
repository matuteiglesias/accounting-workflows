"""Build fixture-safe USD/CCL management-flow eligibility evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any, Iterable


AUDIT_COLUMNS = [
    "tx_id", "Date", "period", "Box", "Currency", "amount", "direction",
    "semantic_bucket", "semantic_subbucket", "measure_direction", "measure_inclusion",
    "amount_usd_ccl", "fx_conversion_status", "valuation_basis", "valuation_currency",
    "valuation_policy_id", "management_eligibility", "eligibility_reason",
]
COMPONENT_COLUMNS = [
    "period", "Box", "semantic_bucket", "semantic_subbucket", "measure_direction",
    "valuation_basis", "valuation_currency", "valuation_policy_id",
    "value_usd_ccl", "reportable_value_usd_ccl", "available_value_usd_ccl",
    "projection_status", "source_rows", "contributing_rows", "eligible_rows",
    "review_required_rows", "missing_valuation_rows", "negative_amount_rows",
    "fx_overlap_rows", "measure_direction_excluded_rows", "excluded_not_approved_rows",
]
VALUED_STATUSES = {
    "identity_native_usd",
    "converted_exact_date",
    "converted_previous_available",
}
APPROVED_BUCKETS = {
    "operating_revenue",
    "property_opex",
    "funding_contribution",
    "family_withdrawal_candidate",
    "family_withdrawal",
    "treasury_fx",
}
FX_OVERLAP_BUCKETS = {
    "operating_revenue",
    "property_opex",
    "funding_contribution",
    "family_withdrawal_candidate",
    "family_withdrawal",
    "debt_movement",
}
MEASURE_DIRECTIONS = {
    "operating_revenue": "in",
    "property_opex": "out",
    "funding_contribution": "in",
    "family_withdrawal_candidate": "out",
    "family_withdrawal": "out",
}
TREASURY_MEASURE_DIRECTIONS = {
    "fx_conversion_proceeds": "in",
    "fx_conversion_outflow": "out",
    "fx_cost_or_spread": "out",
}
MANAGEMENT_IMPLEMENTATION_ID = "usd_ccl_semantic_measures_v2"


class ManagementProjectionContractError(ValueError):
    """Raised when projection inputs cannot be reconciled safely."""


def _local_file(value: str | Path, name: str) -> Path:
    raw = str(value)
    if "://" in raw:
        raise ManagementProjectionContractError(f"{name} must be a local filesystem path")
    path = Path(raw)
    if not path.is_file():
        raise ManagementProjectionContractError(f"{name} does not exist: {path}")
    return path


def _read_keyed(path: Path, required: set[str], name: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(required - columns)
        if missing:
            raise ManagementProjectionContractError(f"{name} missing columns: {missing}")
        keyed: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            tx_id = (row.get("tx_id") or "").strip()
            if not tx_id:
                raise ManagementProjectionContractError(f"blank tx_id in {name} row {row_number}")
            if tx_id in keyed:
                raise ManagementProjectionContractError(f"duplicate tx_id in {name}: {tx_id!r}")
            keyed[tx_id] = {key: value or "" for key, value in row.items()}
    if not keyed:
        raise ManagementProjectionContractError(f"{name} has no rows")
    return keyed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_bindings(
    ledger_path: Path,
    ledger: dict[str, dict[str, str]],
    semantic: dict[str, dict[str, str]],
    sidecar_path: Path,
    manifest_path: Path,
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManagementProjectionContractError(f"invalid valuation manifest: {manifest_path}") from exc
    if manifest.get("source_ledger_sha256") != _sha256(ledger_path):
        raise ManagementProjectionContractError("valuation manifest is not bound to the supplied ledger SHA")
    if manifest.get("valuation_artifact_sha256") != _sha256(sidecar_path):
        raise ManagementProjectionContractError("valuation manifest is not bound to the supplied sidecar SHA")
    for tx_id, native in ledger.items():
        classified = semantic[tx_id]
        for field in ["Date", "Currency", "Box"]:
            if native[field].strip() != classified[field].strip():
                raise ManagementProjectionContractError(
                    f"semantic audit {field} does not match ledger for tx_id={tx_id!r}"
                )
        if _decimal(native["amount"], "amount", tx_id) != _decimal(
            classified["amount"], "semantic amount", tx_id
        ):
            raise ManagementProjectionContractError(
                f"semantic audit amount does not match ledger for tx_id={tx_id!r}"
            )


def _decimal(value: str, field: str, tx_id: str) -> Decimal:
    try:
        parsed = Decimal(value.strip())
    except (AttributeError, InvalidOperation) as exc:
        raise ManagementProjectionContractError(
            f"invalid {field} for tx_id={tx_id!r}: {value!r}"
        ) from exc
    if not parsed.is_finite():
        raise ManagementProjectionContractError(
            f"non-finite {field} for tx_id={tx_id!r}: {value!r}"
        )
    return parsed


def _has_fx_evidence(row: dict[str, str]) -> bool:
    payer = row.get("payer", "").strip().casefold()
    receiver = row.get("receiver", "").strip().casefold()
    blob = " ".join(
        row.get(field, "")
        for field in ["Flujo", "Tipo", "Detalle", "notes", "cash_path"]
    ).casefold()
    return payer == "fx" or receiver == "fx" or "cambio:fx" in blob


def _measure_direction(semantic: dict[str, str]) -> str:
    bucket = semantic.get("semantic_bucket", "").strip()
    if bucket == "treasury_fx":
        return TREASURY_MEASURE_DIRECTIONS.get(
            semantic.get("semantic_subbucket", "").strip(), ""
        )
    return MEASURE_DIRECTIONS.get(bucket, "")


def _measure_inclusion(semantic: dict[str, str]) -> tuple[str, str]:
    expected = _measure_direction(semantic)
    if not expected:
        return "", "excluded_not_approved_v1"
    if semantic.get("direction", "").strip() != expected:
        return expected, "excluded_direction"
    return expected, "selected"


def _eligibility(
    ledger: dict[str, str], semantic: dict[str, str], valuation: dict[str, str]
) -> tuple[str, str]:
    tx_id = ledger["tx_id"]
    bucket = semantic["semantic_bucket"].strip()
    valuation_status = valuation["fx_conversion_status"].strip()
    if valuation_status not in VALUED_STATUSES or not valuation["amount_usd_ccl"].strip():
        return "unavailable_valuation", valuation_status or "missing_valuation_status"
    amount = _decimal(ledger["amount"], "amount", tx_id)
    if amount < 0:
        return "review_required", "negative_native_amount"
    if _has_fx_evidence(ledger) and bucket in FX_OVERLAP_BUCKETS:
        return "review_required", "fx_semantic_overlap"
    if (
        semantic.get("review_required", "").strip().casefold() in {"true", "1", "yes"}
        or semantic.get("classification_status", "").strip() == "review_required"
        or bucket == "unknown"
    ):
        return "review_required", "ambiguous_native_semantics"
    if bucket not in APPROVED_BUCKETS:
        return "excluded_not_approved_v1", "semantic_bucket_not_approved_v1"
    if bucket == "treasury_fx" and not _measure_direction(semantic):
        return "excluded_not_approved_v1", "treasury_subbucket_not_approved_v1"
    return "eligible", "eligible"


def _csv_bytes(columns: list[str], rows: Iterable[dict[str, Any]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_immutable(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    content = _csv_bytes(columns, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise ManagementProjectionContractError(
            f"refusing to overwrite projected management artifact with different bytes: {path}"
        )
    path.write_bytes(content)


def build_usd_ccl_management_flows(
    *, ledger_path: str | Path, semantic_audit_path: str | Path,
    valuation_sidecar_path: str | Path, valuation_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Join three tx-grain artifacts and fail closed at component-cell grain."""
    ledger_file = _local_file(ledger_path, "ledger")
    sidecar_file = _local_file(valuation_sidecar_path, "valuation sidecar")
    manifest_file = _local_file(valuation_manifest_path, "valuation manifest")
    ledger = _read_keyed(
        ledger_file,
        {"tx_id", "Date", "amount", "Currency", "Box", "payer", "receiver", "Flujo", "Tipo"},
        "ledger",
    )
    semantic = _read_keyed(
        _local_file(semantic_audit_path, "semantic audit"),
        {"tx_id", "Date", "amount", "Currency", "Box", "direction", "semantic_bucket", "semantic_subbucket", "classification_status", "review_required"},
        "semantic audit",
    )
    valuation = _read_keyed(
        sidecar_file,
        {"tx_id", "amount_usd_ccl", "fx_conversion_status", "valuation_basis", "valuation_currency", "fx_rate_policy"},
        "valuation sidecar",
    )
    ledger_ids = set(ledger)
    if set(semantic) != ledger_ids or set(valuation) != ledger_ids:
        raise ManagementProjectionContractError(
            "ledger, semantic audit, and valuation sidecar must have identical tx_id coverage"
        )
    _validate_bindings(ledger_file, ledger, semantic, sidecar_file, manifest_file)

    audit_rows: list[dict[str, str]] = []
    for tx_id in sorted(ledger):
        native = ledger[tx_id]
        classified = semantic[tx_id]
        valued = valuation[tx_id]
        eligibility, reason = _eligibility(native, classified, valued)
        measure_direction, measure_inclusion = _measure_inclusion(classified)
        audit_rows.append({
            "tx_id": tx_id,
            "Date": native["Date"],
            "period": native["Date"][:7],
            "Box": native["Box"],
            "Currency": native["Currency"],
            "amount": native["amount"],
            "direction": classified["direction"],
            "semantic_bucket": classified["semantic_bucket"],
            "semantic_subbucket": classified["semantic_subbucket"],
            "measure_direction": measure_direction,
            "measure_inclusion": measure_inclusion,
            "amount_usd_ccl": valued["amount_usd_ccl"],
            "fx_conversion_status": valued["fx_conversion_status"],
            "valuation_basis": valued["valuation_basis"],
            "valuation_currency": valued["valuation_currency"],
            "valuation_policy_id": valued["fx_rate_policy"],
            "management_eligibility": eligibility,
            "eligibility_reason": reason,
        })

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    group_fields = [
        "period", "Box", "semantic_bucket", "semantic_subbucket",
        "valuation_basis", "valuation_currency", "valuation_policy_id",
    ]
    for row in audit_rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    component_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        selected = [row for row in rows if row["measure_inclusion"] == "selected"]
        eligible = [row for row in selected if row["management_eligibility"] == "eligible"]
        review_required = [
            row for row in selected if row["management_eligibility"] == "review_required"
        ]
        missing_valuation = [
            row for row in selected if row["management_eligibility"] == "unavailable_valuation"
        ]
        excluded_not_approved = [
            row for row in rows
            if row["management_eligibility"] == "excluded_not_approved_v1"
            or row["measure_inclusion"] == "excluded_not_approved_v1"
        ]
        available = sum(
            (_decimal(row["amount_usd_ccl"], "amount_usd_ccl", row["tx_id"]) for row in eligible),
            Decimal(0),
        )
        if len(excluded_not_approved) == len(rows):
            projection_status = "excluded_not_approved_v1"
            reportable = ""
        elif review_required:
            projection_status = "incomplete_review_required"
            reportable = ""
        elif missing_valuation:
            projection_status = "incomplete_unavailable_valuation"
            reportable = ""
        else:
            projection_status = "complete"
            reportable = format(available, "f")
        component_rows.append({
            **dict(zip(group_fields, key)),
            "measure_direction": rows[0]["measure_direction"],
            "value_usd_ccl": reportable,
            "reportable_value_usd_ccl": reportable,
            "available_value_usd_ccl": format(available, "f"),
            "projection_status": projection_status,
            "source_rows": len(rows),
            "contributing_rows": len(selected),
            "eligible_rows": len(eligible),
            "review_required_rows": len(review_required),
            "missing_valuation_rows": len(missing_valuation),
            "negative_amount_rows": sum(row["eligibility_reason"] == "negative_native_amount" for row in selected),
            "fx_overlap_rows": sum(row["eligibility_reason"] == "fx_semantic_overlap" for row in selected),
            "measure_direction_excluded_rows": sum(row["measure_inclusion"] == "excluded_direction" for row in rows),
            "excluded_not_approved_rows": len(excluded_not_approved),
        })

    out_dir = Path(output_dir)
    paths = {
        "audit": out_dir / "management_usd_ccl_flow_audit.csv",
        "components": out_dir / "monthly_management_usd_ccl_components.csv",
    }
    _write_immutable(paths["audit"], AUDIT_COLUMNS, audit_rows)
    _write_immutable(paths["components"], COMPONENT_COLUMNS, component_rows)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fixture-safe USD/CCL flow eligibility")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--semantic-audit", required=True)
    parser.add_argument("--valuation-sidecar", required=True)
    parser.add_argument("--valuation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = build_usd_ccl_management_flows(
        ledger_path=args.ledger,
        semantic_audit_path=args.semantic_audit,
        valuation_sidecar_path=args.valuation_sidecar,
        valuation_manifest_path=args.valuation_manifest,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
