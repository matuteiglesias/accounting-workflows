from __future__ import annotations

"""Compile one property-scoped formal-rendition dataset from governed run artifacts.

This module is a downstream projection only. It never reclassifies source
transactions, infers title/creditor status, converts currencies, or turns a
property activity result into a legal or physical-cash balance.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from accounting.rendition.scope import PropertyRegistry, PropertyScope
from accounting.rendition.spec import RenditionSpec, load_rendition_spec
from accounting.rendition.validation import validate_rendition_context


TOLERANCE = 0.01
MANDATORY_ACTIVITY_CAVEAT = (
    "El resultado de actividad no constituye por sí mismo determinación de deuda, "
    "saldo jurídicamente exigible, titularidad de fondos ni caja física disponible."
)

CASH_REQUIRED = {
    "tx_id",
    "Date",
    "Box",
    "Currency",
    "Lugar",
    "amount_in",
    "amount_out",
    "net_amount",
    "payer",
    "receiver",
    "Detalle",
    "semantic_bucket",
    "semantic_subbucket",
    "source_file",
    "source_row",
    "rule_id",
    "review_required",
    "movement_basis",
    "cash_direction",
}
AUDIT_REQUIRED = {
    "tx_id",
    "Date",
    "Box",
    "Currency",
    "Lugar",
    "amount",
    "payer",
    "receiver",
    "Detalle",
    "semantic_bucket",
    "semantic_subbucket",
    "cash_effect",
    "debt_effect",
    "rule_id",
    "review_required",
    "source_file",
    "source_row",
}
ACCOUNTABILITY_REQUIRED = {
    "period",
    "Box",
    "Currency",
    "closing_control",
    "validated_cash_status",
}

CASH_OUTPUT_COLUMNS = [
    "rendition_line_id",
    "tx_id",
    "Date",
    "property_id",
    "property_display_name",
    "Box",
    "Currency",
    "amount_in",
    "amount_out",
    "net_amount",
    "payer",
    "receiver",
    "Detalle",
    "semantic_bucket",
    "semantic_subbucket",
    "cash_direction",
    "movement_basis",
    "source_file",
    "source_row",
    "rule_id",
    "review_required",
]
NON_CASH_OUTPUT_COLUMNS = [
    "rendition_line_id",
    "tx_id",
    "Date",
    "property_id",
    "property_display_name",
    "Box",
    "Currency",
    "amount",
    "direction",
    "payer",
    "receiver",
    "Detalle",
    "semantic_bucket",
    "semantic_subbucket",
    "cash_effect",
    "debt_effect",
    "funding_actor",
    "funding_channel",
    "settlement_case_id",
    "settlement_mode",
    "leg_role",
    "source_file",
    "source_row",
    "rule_id",
    "review_required",
]
SUMMARY_COLUMNS = [
    "property_id",
    "property_display_name",
    "Currency",
    "basis",
    "row_role",
    "statement_side",
    "account_line_id",
    "account_line",
    "amount",
    "n_transactions",
    "source_tx_ids",
]
OPENING_COLUMNS = [
    "opening_basis_type",
    "grain",
    "property_id",
    "property_display_name",
    "Box",
    "Currency",
    "opening_as_of_period",
    "opening_amount",
    "status",
    "validated_cash_status",
    "note",
]
VALIDATION_COLUMNS = ["check", "status", "severity", "amount", "detail"]


def _read_required(path: Path, required: set[str], name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing rendition source {name}: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    return frame


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truth(value: object) -> bool:
    return _text(value).casefold() in {"true", "1", "yes", "y"}


def _selector_key(box: object, lugar: object) -> tuple[str, str]:
    return (_text(box).casefold(), _text(lugar).casefold())


def _selected_lookup(selected: tuple[PropertyScope, ...]) -> dict[tuple[str, str], PropertyScope]:
    return {
        _selector_key(entry.box_selector, entry.lugar_selector): entry
        for entry in selected
    }


def _prepare_dates(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if out["Date"].isna().any():
        bad = out.loc[out["Date"].isna(), "tx_id"].astype(str).tolist()
        raise ValueError(f"{name} has invalid Date rows: {bad[:10]}")
    return out


def _ensure_unique_tx(frame: pd.DataFrame, *, name: str) -> None:
    duplicated = frame["tx_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        bad = sorted(frame.loc[duplicated, "tx_id"].astype(str).unique())
        raise ValueError(f"{name} tx_id is not unique: {bad[:10]}")


def _attach_property(
    frame: pd.DataFrame,
    selected: tuple[PropertyScope, ...],
) -> pd.DataFrame:
    lookup = _selected_lookup(selected)
    out = frame.copy()
    resolved = [
        lookup.get(_selector_key(box, lugar))
        for box, lugar in zip(out["Box"], out["Lugar"])
    ]
    out["property_id"] = [entry.property_id if entry is not None else "" for entry in resolved]
    out["property_display_name"] = [
        entry.display_name if entry is not None else "" for entry in resolved
    ]
    return out.loc[out["property_id"].ne("")].copy()


def _period_mask(frame: pd.DataFrame, spec: RenditionSpec) -> pd.Series:
    start = pd.Timestamp(spec.period.from_date)
    through = pd.Timestamp(spec.period.through_date)
    return frame["Date"].ge(start) & frame["Date"].le(through)


def _validate_treasury_source_qa(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing rendition source treasury QA: {path}")
    qa = pd.read_csv(path)
    if "status" not in qa.columns:
        raise ValueError("box_treasury_transaction_detail_qa.csv missing status")
    failures = qa.loc[qa["status"].astype(str).str.casefold().eq("fail")]
    if not failures.empty:
        raise ValueError(
            "source treasury transaction detail QA contains failures: "
            f"{failures.head(10).to_dict('records')}"
        )


def _property_population_exists(
    cash_full: pd.DataFrame,
    audit_full: pd.DataFrame,
    selected: tuple[PropertyScope, ...],
) -> None:
    available = {
        _selector_key(box, lugar)
        for frame in (cash_full, audit_full)
        for box, lugar in zip(frame["Box"], frame["Lugar"])
    }
    missing = [
        entry.property_id
        for entry in selected
        if _selector_key(entry.box_selector, entry.lugar_selector) not in available
    ]
    if missing:
        raise ValueError(
            "selected property selector has no source population in the exact run: "
            f"{missing}"
        )


def _build_cash_tape(
    source: pd.DataFrame,
    spec: RenditionSpec,
    selected: tuple[PropertyScope, ...],
) -> pd.DataFrame:
    scoped = _attach_property(source, selected)
    scoped = scoped.loc[_period_mask(scoped, spec)].copy()
    for column in ("amount_in", "amount_out", "net_amount"):
        scoped[column] = pd.to_numeric(scoped[column], errors="coerce")
    if scoped[["amount_in", "amount_out", "net_amount"]].isna().any().any():
        raise ValueError("rendition cash population has non-numeric cash amounts")
    arithmetic_gap = (scoped["amount_in"] - scoped["amount_out"] - scoped["net_amount"]).abs()
    if not arithmetic_gap.empty and float(arithmetic_gap.max()) > TOLERANCE:
        raise ValueError(
            "rendition cash population breaks governed cash arithmetic: "
            f"max_gap={float(arithmetic_gap.max())}"
        )
    bad_basis = scoped.loc[
        ~scoped["movement_basis"].astype(str).eq("actual_cash")
        | ~scoped["cash_direction"].astype(str).isin(["in", "out"])
    ]
    if not bad_basis.empty:
        raise ValueError("rendition cash tape contains non-actual-cash source rows")

    scoped["rendition_line_id"] = (
        spec.rendition_id + ":cash:" + scoped["tx_id"].astype(str)
    )
    scoped["Date"] = scoped["Date"].dt.date.astype(str)
    scoped["review_required"] = scoped["review_required"].map(_truth)
    return scoped[CASH_OUTPUT_COLUMNS].sort_values(
        ["property_id", "Currency", "Date", "tx_id"]
    ).reset_index(drop=True)


def _non_cash_mask(frame: pd.DataFrame, cash_tx_ids: set[str]) -> pd.Series:
    cash_effect = frame["cash_effect"].map(_text).str.casefold()
    debt_effect = frame["debt_effect"].map(_text).str.casefold()
    settlement_mode = frame.get("settlement_mode", pd.Series("", index=frame.index)).map(_text).str.casefold()
    non_cash = (
        cash_effect.str.startswith("no_cash_")
        | cash_effect.eq("non_cash_support")
        | settlement_mode.isin({"constructive", "offset"})
        | (~debt_effect.isin({"", "none", "nan", "n/a"}))
    )
    return ~frame["tx_id"].astype(str).isin(cash_tx_ids) & non_cash


def _build_non_cash_events(
    source: pd.DataFrame,
    spec: RenditionSpec,
    selected: tuple[PropertyScope, ...],
    cash_tx_ids: set[str],
) -> pd.DataFrame:
    scoped = _attach_property(source, selected)
    scoped = scoped.loc[_period_mask(scoped, spec)].copy()
    scoped = scoped.loc[_non_cash_mask(scoped, cash_tx_ids)].copy()
    scoped["amount"] = pd.to_numeric(scoped["amount"], errors="coerce")
    if scoped["amount"].isna().any():
        raise ValueError("rendition non-cash population has non-numeric amount")
    for column in (
        "direction",
        "funding_actor",
        "funding_channel",
        "settlement_case_id",
        "settlement_mode",
        "leg_role",
    ):
        if column not in scoped.columns:
            scoped[column] = ""
    scoped["rendition_line_id"] = (
        spec.rendition_id + ":noncash:" + scoped["tx_id"].astype(str)
    )
    scoped["Date"] = scoped["Date"].dt.date.astype(str)
    scoped["review_required"] = scoped["review_required"].map(_truth)
    return scoped[NON_CASH_OUTPUT_COLUMNS].sort_values(
        ["property_id", "Currency", "Date", "tx_id"]
    ).reset_index(drop=True)


_CASH_LINE_LABELS = {
    "operating_revenue": "Fondos recibidos",
    "funding_contribution": "Aportes",
    "property_opex": "Gastos patrimoniales",
    "family_withdrawal_candidate": "Distribuciones / retiros candidatos",
    "family_withdrawal": "Distribuciones / retiros",
    "internal_transfer": "Transferencias internas",
    "debt_movement": "Movimientos de deuda",
    "treasury_fx": "Movimientos de tesorería FX",
    "cost_allocation_gap": "Asignación de costos pendiente",
    "unknown": "Otros / revisión requerida",
}


def _cash_account_line(row: pd.Series) -> tuple[str, str, str]:
    direction = _text(row["cash_direction"])
    bucket = _text(row["semantic_bucket"]) or "unknown"
    side = "debe" if direction == "in" else "haber"
    line_id = f"{side}.{bucket}"
    label = _CASH_LINE_LABELS.get(bucket, bucket.replace("_", " ").strip().title())
    return side, line_id, label


def _source_ids(series: pd.Series) -> str:
    return ";".join(sorted(set(series.astype(str))))


def _build_summary(cash: pd.DataFrame, non_cash: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not cash.empty:
        work = cash.copy()
        metadata = work.apply(_cash_account_line, axis=1, result_type="expand")
        metadata.columns = ["statement_side", "account_line_id", "account_line"]
        work = pd.concat([work, metadata], axis=1)
        work["summary_amount"] = work["amount_in"].where(
            work["cash_direction"].eq("in"), work["amount_out"]
        )
        keys = [
            "property_id",
            "property_display_name",
            "Currency",
            "statement_side",
            "account_line_id",
            "account_line",
        ]
        for key, group in work.groupby(keys, dropna=False, sort=True):
            rows.append(
                {
                    "property_id": key[0],
                    "property_display_name": key[1],
                    "Currency": key[2],
                    "basis": "cash",
                    "row_role": "component",
                    "statement_side": key[3],
                    "account_line_id": key[4],
                    "account_line": key[5],
                    "amount": float(group["summary_amount"].sum()),
                    "n_transactions": int(group["tx_id"].nunique()),
                    "source_tx_ids": _source_ids(group["tx_id"]),
                }
            )

        for key, group in cash.groupby(
            ["property_id", "property_display_name", "Currency"],
            dropna=False,
            sort=True,
        ):
            rows.append(
                {
                    "property_id": key[0],
                    "property_display_name": key[1],
                    "Currency": key[2],
                    "basis": "cash",
                    "row_role": "result",
                    "statement_side": "result",
                    "account_line_id": "activity_result",
                    "account_line": "Resultado de actividad",
                    "amount": float(group["net_amount"].sum()),
                    "n_transactions": int(group["tx_id"].nunique()),
                    "source_tx_ids": _source_ids(group["tx_id"]),
                }
            )

    if not non_cash.empty:
        work = non_cash.copy()
        work["bucket"] = work["semantic_bucket"].map(_text).replace("", "unknown")
        for key, group in work.groupby(
            ["property_id", "property_display_name", "Currency", "bucket"],
            dropna=False,
            sort=True,
        ):
            bucket = str(key[3])
            rows.append(
                {
                    "property_id": key[0],
                    "property_display_name": key[1],
                    "Currency": key[2],
                    "basis": "non_cash_event",
                    "row_role": "component",
                    "statement_side": "separate_annex",
                    "account_line_id": f"non_cash.{bucket}",
                    "account_line": f"Evento administrado sin tránsito por caja: {bucket}",
                    "amount": float(group["amount"].sum()),
                    "n_transactions": int(group["tx_id"].nunique()),
                    "source_tx_ids": _source_ids(group["tx_id"]),
                }
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        ["property_id", "Currency", "basis", "row_role", "account_line_id"]
    ).reset_index(drop=True)


def _active_property_currencies(
    cash: pd.DataFrame,
    non_cash: pd.DataFrame,
    selected: tuple[PropertyScope, ...],
) -> list[tuple[PropertyScope, str]]:
    by_id = {entry.property_id: entry for entry in selected}
    pairs: set[tuple[str, str]] = set()
    for frame in (cash, non_cash):
        if frame.empty:
            continue
        pairs |= {
            (str(row.property_id), str(row.Currency))
            for row in frame[["property_id", "Currency"]].itertuples(index=False)
        }
    return [(by_id[property_id], currency) for property_id, currency in sorted(pairs)]


def _has_prior_scoped_activity(
    cash_full: pd.DataFrame,
    audit_full: pd.DataFrame,
    selected: tuple[PropertyScope, ...],
    from_date: str,
) -> bool:
    cutoff = pd.Timestamp(from_date)
    for frame in (cash_full, audit_full):
        scoped = _attach_property(frame, selected)
        if not scoped.empty and scoped["Date"].lt(cutoff).any():
            return True
    return False


def _build_opening_basis(
    spec: RenditionSpec,
    selected: tuple[PropertyScope, ...],
    cash: pd.DataFrame,
    non_cash: pd.DataFrame,
    cash_full: pd.DataFrame,
    audit_full: pd.DataFrame,
    accountability: pd.DataFrame,
) -> pd.DataFrame:
    basis = spec.opening_basis.type
    rows: list[dict[str, Any]] = []
    active_pairs = _active_property_currencies(cash, non_cash, selected)

    if basis == "zero_origin_from_management_start":
        if _has_prior_scoped_activity(cash_full, audit_full, selected, spec.period.from_date):
            raise ValueError(
                "zero_origin_from_management_start conflicts with earlier selected-property "
                "activity in the exact run"
            )
        for entry, currency in active_pairs:
            rows.append(
                {
                    "opening_basis_type": basis,
                    "grain": "property_currency",
                    "property_id": entry.property_id,
                    "property_display_name": entry.display_name,
                    "Box": entry.box_selector,
                    "Currency": currency,
                    "opening_as_of_period": "",
                    "opening_amount": 0.0,
                    "status": "available",
                    "validated_cash_status": "not_applicable",
                    "note": "Zero origin asserted only after confirming no earlier scoped activity in the exact run.",
                }
            )

    elif basis == "governed_prior_close":
        work = accountability.copy()
        work["period"] = work["period"].astype(str)
        work["period_end"] = pd.PeriodIndex(work["period"], freq="M").end_time.normalize()
        work = work.loc[work["period_end"].lt(pd.Timestamp(spec.period.from_date))].copy()
        for box in sorted({entry.box_selector for entry in selected}):
            box_rows = work.loc[work["Box"].astype(str).eq(box)]
            for currency in sorted(box_rows["Currency"].dropna().astype(str).unique()):
                group = box_rows.loc[box_rows["Currency"].astype(str).eq(currency)].sort_values("period")
                if group.empty:
                    continue
                last = group.iloc[-1]
                rows.append(
                    {
                        "opening_basis_type": basis,
                        "grain": "box_currency",
                        "property_id": "",
                        "property_display_name": "",
                        "Box": box,
                        "Currency": currency,
                        "opening_as_of_period": str(last["period"]),
                        "opening_amount": float(last["closing_control"]),
                        "status": "available_box_level_not_property_attributable",
                        "validated_cash_status": _text(last.get("validated_cash_status")),
                        "note": (
                            "Governed prior Box control close. It is pooled treasury control and "
                            "is not attributed to any selected property or asserted as physical cash."
                        ),
                    }
                )

    else:
        status = (
            "pending_external_verification"
            if basis == "verified_external"
            else "unavailable"
        )
        note = (
            "External opening evidence is not ingested in wave 2; amount deliberately left unavailable."
            if basis == "verified_external"
            else "No opening amount asserted."
        )
        for entry, currency in active_pairs:
            rows.append(
                {
                    "opening_basis_type": basis,
                    "grain": "property_currency",
                    "property_id": entry.property_id,
                    "property_display_name": entry.display_name,
                    "Box": entry.box_selector,
                    "Currency": currency,
                    "opening_as_of_period": "",
                    "opening_amount": pd.NA,
                    "status": status,
                    "validated_cash_status": "unavailable",
                    "note": note,
                }
            )

    return pd.DataFrame(rows, columns=OPENING_COLUMNS)


def _validation_rows(
    cash: pd.DataFrame,
    non_cash: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail: str, amount: float = 0.0) -> None:
        rows.append(
            {
                "check": check,
                "status": "pass" if ok else "fail",
                "severity": "error",
                "amount": amount,
                "detail": detail,
            }
        )

    duplicate_cash = int(cash["tx_id"].duplicated().sum()) if not cash.empty else 0
    duplicate_non_cash = int(non_cash["tx_id"].duplicated().sum()) if not non_cash.empty else 0
    overlap = set(cash["tx_id"].astype(str)) & set(non_cash["tx_id"].astype(str))
    add("cash_tx_id_unique", duplicate_cash == 0, f"duplicates={duplicate_cash}", duplicate_cash)
    add(
        "non_cash_tx_id_unique",
        duplicate_non_cash == 0,
        f"duplicates={duplicate_non_cash}",
        duplicate_non_cash,
    )
    add("cash_non_cash_populations_disjoint", not overlap, f"overlap={sorted(overlap)[:10]}", len(overlap))

    cash_components = summary.loc[
        summary["basis"].eq("cash") & summary["row_role"].eq("component")
    ]
    cash_results = summary.loc[
        summary["basis"].eq("cash") & summary["row_role"].eq("result")
    ]
    for key, group in cash.groupby(["property_id", "Currency"], dropna=False, sort=True):
        expected_in = float(group["amount_in"].sum())
        expected_out = float(group["amount_out"].sum())
        expected_net = float(group["net_amount"].sum())
        component = cash_components.loc[
            cash_components["property_id"].eq(key[0])
            & cash_components["Currency"].eq(key[1])
        ]
        actual_in = float(component.loc[component["statement_side"].eq("debe"), "amount"].sum())
        actual_out = float(component.loc[component["statement_side"].eq("haber"), "amount"].sum())
        result = cash_results.loc[
            cash_results["property_id"].eq(key[0])
            & cash_results["Currency"].eq(key[1])
        ]
        actual_net = float(result["amount"].sum())
        gap = max(
            abs(actual_in - expected_in),
            abs(actual_out - expected_out),
            abs(actual_net - expected_net),
        )
        add(
            "summary_matches_cash_tape",
            gap <= TOLERANCE,
            (
                f"property={key[0]}; currency={key[1]}; "
                f"expected=({expected_in},{expected_out},{expected_net}); "
                f"actual=({actual_in},{actual_out},{actual_net})"
            ),
            gap,
        )

    failures = [row for row in rows if row["status"] == "fail" and row["severity"] == "error"]
    if failures:
        raise ValueError(f"rendition compiler validation failed: {failures[:10]}")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def compile_rendition(
    *,
    spec: RenditionSpec,
    registry: PropertyRegistry,
    run_root: str | Path,
    out_base: str | Path,
) -> dict[str, Path]:
    run_root = Path(run_root)
    validate_rendition_context(spec, registry, run_root)
    selected = registry.resolve(spec.properties)

    cash_path = run_root / "box_treasury_transaction_detail.csv"
    cash_qa_path = run_root / "box_treasury_transaction_detail_qa.csv"
    audit_path = run_root / "classification_audit.csv"
    accountability_path = run_root / "monthly_cash_accountability.csv"

    _validate_treasury_source_qa(cash_qa_path)
    cash_full = _prepare_dates(
        _read_required(cash_path, CASH_REQUIRED, cash_path.name),
        name=cash_path.name,
    )
    audit_full = _prepare_dates(
        _read_required(audit_path, AUDIT_REQUIRED, audit_path.name),
        name=audit_path.name,
    )
    accountability = _read_required(
        accountability_path,
        ACCOUNTABILITY_REQUIRED,
        accountability_path.name,
    )
    _ensure_unique_tx(cash_full, name=cash_path.name)
    _ensure_unique_tx(audit_full, name=audit_path.name)
    _property_population_exists(cash_full, audit_full, selected)

    cash = _build_cash_tape(cash_full, spec, selected)
    non_cash = _build_non_cash_events(
        audit_full,
        spec,
        selected,
        set(cash["tx_id"].astype(str)),
    )
    summary = _build_summary(cash, non_cash)
    opening = _build_opening_basis(
        spec,
        selected,
        cash,
        non_cash,
        cash_full,
        audit_full,
        accountability,
    )
    validation = _validation_rows(cash, non_cash, summary)

    out_dir = Path(out_base) / spec.rendition_id / "compiled"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cash_tape": out_dir / "rendition_cash_tape.csv",
        "non_cash_events": out_dir / "rendition_non_cash_events.csv",
        "summary": out_dir / "rendition_summary.csv",
        "opening_basis": out_dir / "rendition_opening_basis.csv",
        "validation": out_dir / "rendition_validation.csv",
        "context": out_dir / "rendition_context.json",
    }
    cash.to_csv(paths["cash_tape"], index=False)
    non_cash.to_csv(paths["non_cash_events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    opening.to_csv(paths["opening_basis"], index=False)
    validation.to_csv(paths["validation"], index=False)
    context = {
        "schema": "accounting.rendition.v1",
        "phase": "wave2_compiled_account",
        "rendition_id": spec.rendition_id,
        "source_run_id": spec.source_run_id,
        "rendidor_display_name": spec.rendidor_display_name,
        "period": {
            "from": spec.period.from_date,
            "through": spec.period.through_date,
        },
        "properties": list(spec.properties),
        "boxes": list(spec.boxes),
        "recipients": list(spec.recipients),
        "opening_basis": spec.opening_basis.type,
        "mandatory_activity_caveat": MANDATORY_ACTIVITY_CAVEAT,
        "accounting_authority": (
            "cash values are projections of box_treasury_transaction_detail.csv; "
            "non-cash events are projections of classification_audit.csv"
        ),
    }
    paths["context"].write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile transaction-level datasets for accounting.rendition.v1"
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--property-registry", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-base", required=True, type=Path)
    args = parser.parse_args()

    spec = load_rendition_spec(args.spec)
    registry = PropertyRegistry.from_csv(args.property_registry)
    paths = compile_rendition(
        spec=spec,
        registry=registry,
        run_root=args.run_root,
        out_base=args.out_base,
    )
    print(
        json.dumps(
            {key: str(path) for key, path in paths.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
