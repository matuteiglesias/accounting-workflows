from __future__ import annotations

"""Build the exact-run ``acct.docs-input@1`` factual bridge.

This module is deliberately downstream of accounting authority. It selects and
packages existing governed facts, explicit evidence relations, and an optional
prospective commitments sidecar. It never reclassifies ledger rows, creates debt,
infers legal responsibility, or manufactures distributable balances.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from accounting.docs_input import DOCS_INPUT_SCHEMA, TREASURY_COMMITMENTS_SCHEMA
from accounting.evidence.relations import (
    EvidenceContractError,
    TransactionEvidenceIndex,
    load_transaction_evidence,
)
from accounting.professional.drilldown import INDEX_FILENAME
from accounting.reports.common import (
    atomic_write_json,
    atomic_write_text,
    ensure_relative_bundle_path,
    sha256_file,
)
from accounting.scope import load_run_scope_if_present


EVIDENCE_COVERAGE_COLUMNS = [
    "population_id",
    "population_label",
    "source_detail_path",
    "coverage_status",
    "coverage_reason",
    "Currency",
    "Box",
    "property",
    "actor",
    "transaction_rows",
    "distinct_tx_ids",
    "approved_evidence_rows",
    "candidate_evidence_rows",
    "missing_evidence_rows",
    "coverage_pct",
    "source_tx_ids",
]

ACTOR_PROPERTY_COST_COLUMNS = [
    "settlement_case_id",
    "source_tx_id",
    "leg_role",
    "Date",
    "period",
    "Currency",
    "property",
    "Box",
    "obligation_box",
    "expense_category",
    "gross_cost",
    "stakeholder_actor",
    "actor_role",
    "recognized_support",
    "settlement_mode",
    "cash_path",
    "physical_payment_id",
    "physical_payer",
    "physical_payee",
    "payment_method",
    "evidence_ref",
    "evidence_status",
    "allocation_status",
    "allocation_basis",
    "underlying_participant",
    "underlying_allocated_amount",
    "obligation_period",
    "settlement_period",
    "funding_status",
    "debt_origin",
]

UNRESOLVED_ALLOCATION_COLUMNS = [
    "source_tx_id",
    "Date",
    "period",
    "Currency",
    "amount",
    "property",
    "description",
    "status",
    "economic_scope",
    "accounting_nature",
    "debt_effect",
    "allocation_status",
    "asserted_bearer",
    "source_file",
    "source_row",
    "evidence_status",
]

COMMITMENT_REQUIRED_COLUMNS = [
    "commitment_id",
    "due_date",
    "property",
    "Box",
    "Currency",
    "expected_amount",
    "amount_status",
    "commitment_status",
    "obligation_category",
    "source_ref",
    "notes",
]

RESERVE_SCHEDULE_COLUMNS = [
    "commitment_id",
    "due_date",
    "period",
    "property",
    "Box",
    "Currency",
    "expected_amount",
    "amount_status",
    "commitment_status",
    "obligation_category",
    "reserve_included",
    "required_reserve_amount",
    "cumulative_required_reserve",
    "source_ref",
    "notes",
]

RESERVE_QA_COLUMNS = ["check", "status", "severity", "Currency", "amount", "detail"]

SUPPORTED_AMOUNT_STATUSES = frozenset({"known", "estimated"})
SUPPORTED_COMMITMENT_STATUSES = frozenset(
    {"contracted", "known_due", "approved_plan", "scenario"}
)
RESERVE_INCLUDED_STATUSES = frozenset({"contracted", "known_due", "approved_plan"})
SUPPORT_LEG_ROLES = frozenset({"stakeholder_support", "stakeholder_direct_expense"})


@dataclass(frozen=True, slots=True)
class EvidenceState:
    status: str
    reason: str
    index: TransactionEvidenceIndex | None
    documents_path: Path | None
    relations_path: Path | None


@dataclass(frozen=True, slots=True)
class OutputArtifact:
    logical_name: str
    path: Path
    grain: str
    authority: str
    caveat: str
    status: str = "available"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required docs-input columns: {missing}")


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    atomic_write_text(path, frame.to_csv(index=False))
    return path


def _resolve_inside(root: Path, relpath: str) -> Path:
    safe = ensure_relative_bundle_path(relpath)
    root_resolved = root.resolve(strict=True)
    candidate = (root_resolved / safe).resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"docs-input source path escapes bundle root: {relpath!r}") from exc
    return candidate


def _source_csv(run_root: Path, filename: str) -> pd.DataFrame:
    path = run_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"docs-input source artifact missing: {path}")
    return pd.read_csv(path)


def _evidence_state(
    *,
    run_root: Path,
    evidence_documents_path: Path | None,
    transaction_evidence_path: Path | None,
) -> EvidenceState:
    documents = Path(evidence_documents_path) if evidence_documents_path else run_root / "evidence_documents.csv"
    relations = Path(transaction_evidence_path) if transaction_evidence_path else run_root / "transaction_evidence.csv"

    explicit_documents = evidence_documents_path is not None
    explicit_relations = transaction_evidence_path is not None
    if explicit_documents != explicit_relations:
        raise EvidenceContractError(
            "docs-input explicit evidence paths are incomplete: both evidence documents "
            "and transaction evidence paths are required"
        )

    documents_exists = documents.is_file()
    relations_exists = relations.is_file()
    if documents_exists != relations_exists:
        raise EvidenceContractError(
            "optional transaction evidence is incomplete: both evidence_documents.csv "
            "and transaction_evidence.csv must be present"
        )
    if not documents_exists:
        return EvidenceState(
            status="sidecar_absent",
            reason="no complete acct.transaction-evidence@1 sidecar was supplied",
            index=None,
            documents_path=None,
            relations_path=None,
        )

    return EvidenceState(
        status="available",
        reason="validated acct.transaction-evidence@1 sidecar loaded",
        index=load_transaction_evidence(documents, relations),
        documents_path=documents,
        relations_path=relations,
    )


def _singular_dimension(frame: pd.DataFrame, candidates: Iterable[str]) -> str:
    for column in candidates:
        if column not in frame.columns:
            continue
        values = sorted(
            {
                _clean_text(value)
                for value in frame[column]
                if _clean_text(value)
            }
        )
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            return ""
    return ""


def _population_label(index_row: pd.Series, detail_relpath: str) -> str:
    for column in ("label", "metric_label", "metric_id", "table_id", "row_id"):
        value = _clean_text(index_row.get(column))
        if value:
            return value
    return Path(detail_relpath).stem


def build_evidence_coverage(
    *,
    pack_dir: Path | None,
    evidence: EvidenceState,
) -> tuple[pd.DataFrame, str, str]:
    """Project evidence coverage over existing professional detail populations."""

    if pack_dir is None:
        return (
            pd.DataFrame(columns=EVIDENCE_COVERAGE_COLUMNS),
            "unavailable",
            "professional pack not supplied; no drilldown populations were inspected",
        )

    pack_root = Path(pack_dir).resolve(strict=True)
    index_path = pack_root / "drilldown" / INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(f"professional drilldown index not found: {index_path}")
    index = pd.read_csv(index_path)
    _require_columns(index, {"detail_csv_relpath"}, INDEX_FILENAME)

    rows: list[dict[str, Any]] = []
    for _, index_row in index.iterrows():
        detail_relpath = _clean_text(index_row.get("detail_csv_relpath"))
        if not detail_relpath:
            continue
        detail_path = _resolve_inside(pack_root, detail_relpath)
        detail = pd.read_csv(detail_path)
        if "tx_id" in detail.columns:
            tx_rows = [
                _clean_text(value)
                for value in detail["tx_id"].tolist()
                if _clean_text(value)
            ]
        else:
            tx_rows = []

        approved = candidate = missing = 0
        if evidence.index is None:
            missing = len(tx_rows)
        else:
            for tx_id in tx_rows:
                if evidence.index.links_for_tx(tx_id, status="approved"):
                    approved += 1
                elif evidence.index.has_candidate_for_tx(tx_id):
                    candidate += 1
                else:
                    missing += 1
        if approved + candidate + missing != len(tx_rows):
            raise AssertionError("evidence coverage partition failed")

        coverage_pct = round(100.0 * approved / len(tx_rows), 1) if tx_rows else 0.0
        rows.append(
            {
                "population_id": ensure_relative_bundle_path(detail_relpath),
                "population_label": _population_label(index_row, detail_relpath),
                "source_detail_path": ensure_relative_bundle_path(detail_relpath),
                "coverage_status": evidence.status,
                "coverage_reason": evidence.reason,
                "Currency": _singular_dimension(detail, ("Currency", "currency"))
                or _clean_text(index_row.get("currency")),
                "Box": _singular_dimension(detail, ("Box", "target_box", "obligation_box")),
                "property": _singular_dimension(detail, ("Lugar", "property")),
                "actor": _singular_dimension(
                    detail,
                    ("actor", "funding_actor", "stakeholder_actor", "recipient"),
                ),
                "transaction_rows": len(tx_rows),
                "distinct_tx_ids": len(set(tx_rows)),
                "approved_evidence_rows": approved,
                "candidate_evidence_rows": candidate,
                "missing_evidence_rows": missing,
                "coverage_pct": coverage_pct,
                "source_tx_ids": ";".join(sorted(set(tx_rows))),
            }
        )

    return (
        pd.DataFrame(rows, columns=EVIDENCE_COVERAGE_COLUMNS),
        "available",
        "professional drilldown populations inspected",
    )


def build_actor_property_cost_support_detail(
    *,
    run_root: Path,
) -> pd.DataFrame:
    """Project governed settlement legs with explicit property/source dimensions."""

    detail = _source_csv(run_root, "stakeholder_settlement_detail.csv")
    audit = _source_csv(run_root, "classification_audit.csv")

    required_detail = {
        "settlement_case_id",
        "source_tx_id",
        "Date",
        "Currency",
        "gross_amount",
        "expense_category",
        "allocation_status",
        "allocation_basis",
        "stakeholder_actor",
        "actor_role",
        "allocated_amount",
        "settlement_mode",
        "cash_path",
        "physical_payment_id",
        "physical_payer",
        "physical_payee",
        "payment_method",
        "evidence_ref",
        "evidence_status",
        "leg_role",
        "obligation_box",
        "underlying_participant",
        "underlying_allocated_amount",
        "obligation_period",
        "settlement_period",
        "funding_status",
        "debt_origin",
    }
    _require_columns(detail, required_detail, "stakeholder_settlement_detail.csv")
    _require_columns(audit, {"tx_id", "Lugar", "Box"}, "classification_audit.csv")

    nonempty_audit = audit.loc[audit["tx_id"].map(_clean_text).ne("")].copy()
    duplicate_ids = sorted(
        set(
            nonempty_audit.loc[
                nonempty_audit["tx_id"].astype(str).duplicated(keep=False), "tx_id"
            ].astype(str)
        )
    )
    referenced = {
        _clean_text(value)
        for value in detail["source_tx_id"]
        if _clean_text(value)
    }
    conflicting = [tx_id for tx_id in duplicate_ids if tx_id in referenced]
    if conflicting:
        raise ValueError(
            "classification_audit.csv is not unique for referenced source tx_id: "
            f"{conflicting}"
        )

    audit_lookup = {
        _clean_text(row["tx_id"]): row
        for _, row in nonempty_audit.iterrows()
        if _clean_text(row["tx_id"])
    }

    work = detail.copy()
    dates = pd.to_datetime(work["Date"], errors="coerce")
    invalid_date = work["Date"].map(_clean_text).ne("") & dates.isna()
    if invalid_date.any():
        raise ValueError(
            "stakeholder_settlement_detail.csv contains invalid Date values for docs-input"
        )
    work["period"] = dates.dt.to_period("M").astype(str).replace("NaT", "")
    work["gross_amount"] = pd.to_numeric(work["gross_amount"], errors="raise")
    work["allocated_amount"] = pd.to_numeric(work["allocated_amount"], errors="raise")
    work["underlying_allocated_amount"] = pd.to_numeric(
        work["underlying_allocated_amount"], errors="coerce"
    ).fillna(0.0)

    properties: list[str] = []
    boxes: list[str] = []
    for raw_tx_id in work["source_tx_id"]:
        tx_id = _clean_text(raw_tx_id)
        source = audit_lookup.get(tx_id)
        properties.append(_clean_text(source.get("Lugar")) if source is not None else "")
        boxes.append(_clean_text(source.get("Box")) if source is not None else "")

    out = pd.DataFrame(
        {
            "settlement_case_id": work["settlement_case_id"].map(_clean_text),
            "source_tx_id": work["source_tx_id"].map(_clean_text),
            "leg_role": work["leg_role"].map(_clean_text),
            "Date": work["Date"].map(_clean_text),
            "period": work["period"],
            "Currency": work["Currency"].map(_clean_text),
            "property": properties,
            "Box": boxes,
            "obligation_box": work["obligation_box"].map(_clean_text),
            "expense_category": work["expense_category"].map(_clean_text),
            "gross_cost": work["gross_amount"].astype(float),
            "stakeholder_actor": work["stakeholder_actor"].map(_clean_text),
            "actor_role": work["actor_role"].map(_clean_text),
            "recognized_support": work["allocated_amount"].where(
                work["leg_role"].map(_clean_text).isin(SUPPORT_LEG_ROLES), 0.0
            ).astype(float),
            "settlement_mode": work["settlement_mode"].map(_clean_text),
            "cash_path": work["cash_path"].map(_clean_text),
            "physical_payment_id": work["physical_payment_id"].map(_clean_text),
            "physical_payer": work["physical_payer"].map(_clean_text),
            "physical_payee": work["physical_payee"].map(_clean_text),
            "payment_method": work["payment_method"].map(_clean_text),
            "evidence_ref": work["evidence_ref"].map(_clean_text),
            "evidence_status": work["evidence_status"].map(_clean_text),
            "allocation_status": work["allocation_status"].map(_clean_text),
            "allocation_basis": work["allocation_basis"].map(_clean_text),
            "underlying_participant": work["underlying_participant"].map(_clean_text),
            "underlying_allocated_amount": work["underlying_allocated_amount"].astype(float),
            "obligation_period": work["obligation_period"].map(_clean_text),
            "settlement_period": work["settlement_period"].map(_clean_text),
            "funding_status": work["funding_status"].map(_clean_text),
            "debt_origin": work["debt_origin"].map(_clean_text),
        },
        columns=ACTOR_PROPERTY_COST_COLUMNS,
    )
    if len(out) != len(detail):
        raise AssertionError("actor/property/cost docs-input changed settlement-leg row count")
    forbidden = {"legal_obligor", "legal_debtor", "legal_creditor", "legal_owner"}
    if forbidden.intersection(out.columns):
        raise AssertionError("docs-input must not infer legal party fields")
    return out


def _evidence_status_for_tx(tx_id: object, evidence: EvidenceState) -> str:
    text = _clean_text(tx_id)
    if not text:
        return "no_source_tx_id"
    if evidence.index is None:
        return "sidecar_absent"
    if evidence.index.links_for_tx(text, status="approved"):
        return "approved"
    if evidence.index.has_candidate_for_tx(text):
        return "candidate"
    return "missing"


def build_unresolved_allocation_detail(
    *,
    run_root: Path,
    evidence: EvidenceState,
) -> pd.DataFrame:
    source = _source_csv(run_root, "cost_allocation_gaps.csv")
    required = {
        "source_tx_id",
        "Date",
        "period",
        "Currency",
        "amount",
        "Lugar",
        "description",
        "status",
        "economic_scope",
        "accounting_nature",
        "debt_effect",
        "allocation_status",
        "asserted_bearer",
        "source_file",
        "source_row",
    }
    _require_columns(source, required, "cost_allocation_gaps.csv")
    amounts = pd.to_numeric(source["amount"], errors="raise")
    out = pd.DataFrame(
        {
            "source_tx_id": source["source_tx_id"].map(_clean_text),
            "Date": source["Date"].map(_clean_text),
            "period": source["period"].map(_clean_text),
            "Currency": source["Currency"].map(_clean_text),
            "amount": amounts.astype(float),
            "property": source["Lugar"].map(_clean_text),
            "description": source["description"].map(_clean_text),
            "status": source["status"].map(_clean_text),
            "economic_scope": source["economic_scope"].map(_clean_text),
            "accounting_nature": source["accounting_nature"].map(_clean_text),
            "debt_effect": source["debt_effect"].map(_clean_text),
            "allocation_status": source["allocation_status"].map(_clean_text),
            "asserted_bearer": source["asserted_bearer"].map(_clean_text),
            "source_file": source["source_file"].map(_clean_text),
            "source_row": source["source_row"].map(_clean_text),
            "evidence_status": [
                _evidence_status_for_tx(tx_id, evidence)
                for tx_id in source["source_tx_id"]
            ],
        },
        columns=UNRESOLVED_ALLOCATION_COLUMNS,
    )
    if len(out) != len(source):
        raise AssertionError("unresolved-allocation docs-input changed source row count")
    if not out.empty and not out["debt_effect"].eq("none").all():
        raise ValueError(
            "cost_allocation_gaps.csv contains non-none debt_effect; docs-input refuses to promote or reinterpret gaps"
        )
    if not out.empty and out["asserted_bearer"].map(_clean_text).ne("").any():
        raise ValueError(
            "cost_allocation_gaps.csv contains asserted_bearer; docs-input requires explicit review before exposing bearer assertions"
        )

    source_totals = source.assign(__amount=amounts).groupby("Currency", dropna=False)["__amount"].sum()
    out_totals = out.groupby("Currency", dropna=False)["amount"].sum()
    if not source_totals.equals(out_totals):
        raise AssertionError("unresolved-allocation native-currency totals did not reconcile")
    return out


def build_required_reserve_schedule(
    commitments_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(commitments_path, dtype=str).fillna("")
    _require_columns(source, COMMITMENT_REQUIRED_COLUMNS, commitments_path.name)

    work = source[COMMITMENT_REQUIRED_COLUMNS].copy()
    work["commitment_id"] = work["commitment_id"].map(_clean_text)
    if work["commitment_id"].eq("").any():
        raise ValueError("treasury commitments require non-empty commitment_id")
    duplicated = sorted(
        work.loc[work["commitment_id"].duplicated(keep=False), "commitment_id"].unique()
    )
    if duplicated:
        raise ValueError(f"duplicate treasury commitment_id values: {duplicated}")

    due = pd.to_datetime(work["due_date"], errors="coerce")
    if due.isna().any():
        bad = work.loc[due.isna(), "commitment_id"].tolist()
        raise ValueError(f"invalid treasury commitment due_date for: {bad}")
    amount = pd.to_numeric(work["expected_amount"], errors="coerce")
    if amount.isna().any() or amount.lt(0).any():
        bad = work.loc[amount.isna() | amount.lt(0), "commitment_id"].tolist()
        raise ValueError(f"invalid non-negative expected_amount for commitments: {bad}")

    amount_status = work["amount_status"].map(_clean_text)
    bad_amount_status = sorted(set(amount_status) - SUPPORTED_AMOUNT_STATUSES)
    if bad_amount_status:
        raise ValueError(f"unsupported treasury commitment amount_status: {bad_amount_status}")
    commitment_status = work["commitment_status"].map(_clean_text)
    bad_commitment_status = sorted(set(commitment_status) - SUPPORTED_COMMITMENT_STATUSES)
    if bad_commitment_status:
        raise ValueError(
            f"unsupported treasury commitment commitment_status: {bad_commitment_status}"
        )

    currency = work["Currency"].map(_clean_text).str.upper()
    if currency.eq("").any():
        raise ValueError("treasury commitments require explicit native Currency")

    out = pd.DataFrame(
        {
            "commitment_id": work["commitment_id"],
            "due_date": due.dt.date.astype(str),
            "period": due.dt.to_period("M").astype(str),
            "property": work["property"].map(_clean_text),
            "Box": work["Box"].map(_clean_text),
            "Currency": currency,
            "expected_amount": amount.astype(float),
            "amount_status": amount_status,
            "commitment_status": commitment_status,
            "obligation_category": work["obligation_category"].map(_clean_text),
            "reserve_included": commitment_status.isin(RESERVE_INCLUDED_STATUSES),
            "required_reserve_amount": amount.where(
                commitment_status.isin(RESERVE_INCLUDED_STATUSES), 0.0
            ).astype(float),
            "cumulative_required_reserve": 0.0,
            "source_ref": work["source_ref"].map(_clean_text),
            "notes": work["notes"].map(_clean_text),
        },
        columns=RESERVE_SCHEDULE_COLUMNS,
    )
    out = out.sort_values(["Currency", "due_date", "commitment_id"], kind="stable").reset_index(drop=True)
    out["cumulative_required_reserve"] = out.groupby("Currency", sort=False)[
        "required_reserve_amount"
    ].cumsum()

    if out.loc[out["commitment_status"].eq("scenario"), "reserve_included"].any():
        raise AssertionError("scenario commitments entered required reserve")

    qa_rows: list[dict[str, Any]] = [
        {
            "check": "unique_commitment_id",
            "status": "pass",
            "severity": "error",
            "Currency": "",
            "amount": 0.0,
            "detail": f"rows={len(out)}",
        },
        {
            "check": "scenario_excluded_from_required_reserve",
            "status": "pass",
            "severity": "error",
            "Currency": "",
            "amount": float(
                out.loc[out["commitment_status"].eq("scenario"), "required_reserve_amount"].sum()
            ),
            "detail": "scenario rows remain visible but contribute zero required reserve",
        },
    ]
    for curr, group in out.groupby("Currency", sort=True):
        included = float(
            group.loc[group["reserve_included"], "expected_amount"].sum()
        )
        reserve = float(group["required_reserve_amount"].sum())
        ok = abs(included - reserve) <= 0.01
        qa_rows.append(
            {
                "check": "required_reserve_reconciles",
                "status": "pass" if ok else "fail",
                "severity": "error",
                "Currency": curr,
                "amount": reserve - included,
                "detail": f"included_expected={included}; required_reserve={reserve}",
            }
        )
    qa = pd.DataFrame(qa_rows, columns=RESERVE_QA_COLUMNS)
    if qa["status"].eq("fail").any():
        raise ValueError(f"required reserve schedule QA failed: {qa.to_dict('records')}")
    return out, qa


def _artifact_record(artifact: OutputArtifact, *, out_dir: Path) -> dict[str, Any]:
    relative = artifact.path.resolve().relative_to(out_dir.resolve()).as_posix()
    relative = ensure_relative_bundle_path(relative)
    rows: int | None = None
    if artifact.path.suffix.lower() == ".csv":
        rows = int(len(pd.read_csv(artifact.path)))
    return {
        "name": artifact.logical_name,
        "path": relative,
        "rows": rows,
        "sha256": sha256_file(artifact.path),
        "grain": artifact.grain,
        "authority": artifact.authority,
        "caveat": artifact.caveat,
        "status": artifact.status,
    }


def _prospective_provenance(commitments_path: Path | None) -> dict[str, Any]:
    if commitments_path is None:
        return {
            "present": False,
            "schema_version": TREASURY_COMMITMENTS_SCHEMA,
            "row_count": 0,
            "sha256": None,
        }
    path = Path(commitments_path)
    return {
        "present": True,
        "schema_version": TREASURY_COMMITMENTS_SCHEMA,
        "row_count": int(len(pd.read_csv(path, dtype=str))),
        "sha256": sha256_file(path),
    }


def _evidence_provenance(evidence: EvidenceState) -> dict[str, Any]:
    if evidence.index is None:
        return {
            "status": evidence.status,
            "reason": evidence.reason,
            "documents": 0,
            "relations": 0,
            "approved_relations": 0,
            "documents_sha256": None,
            "relations_sha256": None,
        }
    assert evidence.documents_path is not None
    assert evidence.relations_path is not None
    return {
        "status": evidence.status,
        "reason": evidence.reason,
        "documents": evidence.index.document_count,
        "relations": evidence.index.relation_count,
        "approved_relations": evidence.index.approved_relation_count,
        "documents_sha256": sha256_file(evidence.documents_path),
        "relations_sha256": sha256_file(evidence.relations_path),
    }


def build_docs_input_bundle(
    *,
    run_root: Path,
    out_dir: Path,
    pack_dir: Path | None = None,
    evidence_documents_path: Path | None = None,
    transaction_evidence_path: Path | None = None,
    commitments_path: Path | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Path]:
    """Build docs-input artifacts from one exact accounting run.

    Existing accounting artifacts are read-only. Optional prospective commitments
    are kept outside the historical accounting spine and only affect the reserve
    schedule artifacts emitted here.
    """

    run_root = Path(run_root).resolve(strict=True)
    if run_root.name.startswith("latest"):
        raise ValueError("docs-input requires an exact run, not a latest pointer")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence = _evidence_state(
        run_root=run_root,
        evidence_documents_path=evidence_documents_path,
        transaction_evidence_path=transaction_evidence_path,
    )

    artifacts: list[OutputArtifact] = []
    paths: dict[str, Path] = {}

    coverage, coverage_status, coverage_reason = build_evidence_coverage(
        pack_dir=pack_dir,
        evidence=evidence,
    )
    coverage_path = _write_csv(out_dir / "accounting_evidence_coverage.csv", coverage)
    paths["accounting_evidence_coverage"] = coverage_path
    artifacts.append(
        OutputArtifact(
            logical_name="accounting_evidence_coverage",
            path=coverage_path,
            grain="professional detail population x source row evidence status",
            authority="professional drilldown membership + acct.transaction-evidence@1",
            caveat=(
                "coverage measures linked supporting evidence only; it is not transaction validity, "
                "legal proof, or authority to retain/disburse funds"
            ),
            status=coverage_status,
        )
    )

    actor_detail = build_actor_property_cost_support_detail(run_root=run_root)
    actor_path = _write_csv(out_dir / "actor_property_cost_support_detail.csv", actor_detail)
    paths["actor_property_cost_support_detail"] = actor_path
    artifacts.append(
        OutputArtifact(
            logical_name="actor_property_cost_support_detail",
            path=actor_path,
            grain="governed stakeholder settlement leg",
            authority="stakeholder_settlement_detail.csv + explicit tx_id lookup in classification_audit.csv",
            caveat=(
                "physical payer/support/property dimensions are factual accounting metadata; they do not assign legal liability, "
                "ownership, reimbursement rights, or final economic entitlement"
            ),
        )
    )

    unresolved = build_unresolved_allocation_detail(run_root=run_root, evidence=evidence)
    unresolved_path = _write_csv(out_dir / "unresolved_allocation_detail.csv", unresolved)
    paths["unresolved_allocation_detail"] = unresolved_path
    artifacts.append(
        OutputArtifact(
            logical_name="unresolved_allocation_detail",
            path=unresolved_path,
            grain="one governed unresolved cost-allocation source row",
            authority="cost_allocation_gaps.csv",
            caveat=(
                "unresolved allocation is not established debt or legal bearer; docs-input preserves debt_effect=none"
            ),
        )
    )

    commitments = Path(commitments_path).resolve(strict=True) if commitments_path is not None else None
    if commitments is not None:
        reserve, reserve_qa = build_required_reserve_schedule(commitments)
        reserve_path = _write_csv(out_dir / "required_reserve_schedule.csv", reserve)
        reserve_qa_path = _write_csv(out_dir / "required_reserve_schedule_qa.csv", reserve_qa)
        paths["required_reserve_schedule"] = reserve_path
        paths["required_reserve_schedule_qa"] = reserve_qa_path
        artifacts.extend(
            [
                OutputArtifact(
                    logical_name="required_reserve_schedule",
                    path=reserve_path,
                    grain="prospective commitment",
                    authority=TREASURY_COMMITMENTS_SCHEMA,
                    caveat=(
                        "prospective reserve input is outside the historical ledger and does not establish a distributable balance, "
                        "legal withholding authority, OPEX recognition, debt, or cash availability"
                    ),
                ),
                OutputArtifact(
                    logical_name="required_reserve_schedule_qa",
                    path=reserve_qa_path,
                    grain="validation check / native currency",
                    authority="required reserve schedule reconciliation",
                    caveat="QA validates projection arithmetic only; it does not validate the legal basis of a commitment",
                ),
            ]
        )

    scope = load_run_scope_if_present(run_root)
    generated_at_utc = generated_at_utc or datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": DOCS_INPUT_SCHEMA,
        "source_run_id": run_root.name,
        "scope_tag": scope.tag if scope is not None else "",
        "generated_at_utc": generated_at_utc,
        "artifacts": [_artifact_record(item, out_dir=out_dir) for item in artifacts],
        "evidence_input": _evidence_provenance(evidence),
        "prospective_input": _prospective_provenance(commitments),
        "coverage_status": coverage_status,
        "coverage_reason": coverage_reason,
        "accounting_authority_changed": False,
        "legal_interpretation_included": False,
        "distribution_capacity_included": False,
    }
    manifest_path = out_dir / "docs_input_manifest.json"
    atomic_write_json(manifest_path, manifest)
    paths["manifest"] = manifest_path
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the exact-run acct.docs-input@1 factual bridge"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pack-dir", type=Path)
    parser.add_argument("--evidence-documents", type=Path)
    parser.add_argument("--transaction-evidence", type=Path)
    parser.add_argument("--commitments", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = build_docs_input_bundle(
        run_root=args.run_root,
        out_dir=args.out_dir,
        pack_dir=args.pack_dir,
        evidence_documents_path=args.evidence_documents,
        transaction_evidence_path=args.transaction_evidence,
        commitments_path=args.commitments,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in sorted(paths.items())},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
