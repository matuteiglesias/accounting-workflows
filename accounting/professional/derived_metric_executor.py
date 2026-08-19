from __future__ import annotations

"""Governed professional execution for Wave 5 derived metrics.

The executor consumes only stable ``derived_metric_id`` metadata and the closed
``DerivedMetricSpec`` registry. It never reclassifies ledger rows and never
selects formulas from presentation labels.
"""

from typing import Any

import pandas as pd

from accounting.cash_authority import (
    inferred_control_schema_supported,
    select_inferred_box_control_period,
)
from accounting.contracts.derived_metrics import DerivedMetricSpec, resolve_derived_metric_spec


ANNUAL_REQUIRED_COLUMNS = {"metric_id", "period", "Currency", "value", "value_status"}


def _norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def _currency(row: pd.Series) -> str:
    return _norm(row.get("Currency"))


def annual_derived_schema_supported(annual: pd.DataFrame) -> bool:
    return annual is not None and ANNUAL_REQUIRED_COLUMNS.issubset(annual.columns)


def _annual_scalar_rows(
    annual: pd.DataFrame,
    *,
    metric_id: str,
    period: str,
    currency: str,
) -> pd.DataFrame:
    if annual.empty:
        return annual.copy()
    mask = annual["metric_id"].fillna("").astype(str).str.strip().eq(metric_id)
    mask &= annual["period"].map(_norm).eq(_norm(period))
    mask &= annual["Currency"].fillna("").astype(str).str.strip().eq(currency)
    rows = annual.loc[mask].copy()
    if "dimension_name" in rows.columns:
        rows = rows.loc[rows["dimension_name"].fillna("").astype(str).str.strip().eq("")].copy()
    return rows


def _metric_ref_id(ref: str) -> str:
    if not ref.startswith("metric:"):
        raise ValueError(f"Expected governed metric ref, got {ref!r}")
    return ref.split(":", 1)[1]


def _available_scalar(
    annual: pd.DataFrame,
    *,
    ref: str,
    period: str,
    currency: str,
) -> tuple[float | None, str, pd.DataFrame]:
    metric_id = _metric_ref_id(ref)
    rows = _annual_scalar_rows(
        annual,
        metric_id=metric_id,
        period=period,
        currency=currency,
    )
    if rows.empty:
        return None, f"missing_component:{metric_id}", rows
    if len(rows) != 1:
        return None, f"duplicate_scalar_component:{metric_id}", rows
    status = _norm(rows.iloc[0].get("value_status")).lower()
    if status != "available":
        return None, f"component_not_available:{metric_id}:{status or 'blank'}", rows
    value = pd.to_numeric(rows.iloc[0].get("value"), errors="coerce")
    if pd.isna(value):
        return None, f"component_value_not_numeric:{metric_id}", rows
    return float(value), "", rows


def _compose(spec: DerivedMetricSpec, values: list[float]) -> float:
    if spec.operation == "subtract":
        return float(values[0] - values[1])
    if spec.operation == "add_subtract":
        return float(values[0] + values[1] - values[2])
    if spec.operation == "ratio":
        return float(values[0] / values[1])
    raise ValueError(f"Scalar composition does not support {spec.operation!r}")


def _unsupported_result(
    *,
    spec: DerivedMetricSpec,
    display_value: float,
    source_artifact: str,
    reason: str,
    period: str,
    currency: str,
    evidence: pd.DataFrame,
    sections: list[tuple[str, pd.DataFrame]],
):
    return (
        "unsupported",
        0.0,
        -display_value,
        "governed_derived_metric",
        source_artifact,
        {
            "derived_metric_id": spec.spec_id,
            "executor": "governed_derived_metric_v1",
            "period": period,
            "Currency": currency,
            "selection_status": "unavailable",
            "selection_reason": reason,
            "missing_component_policy": spec.missing_component_policy,
        },
        "Derived metric failed closed; missing or non-applicable governed evidence was not converted to zero.",
        evidence,
        sections,
    )


def _execute_annual_scalar(
    *,
    spec: DerivedMetricSpec,
    row: pd.Series,
    period: str,
    display_value: float,
    annual: pd.DataFrame,
    tolerance: float,
):
    currency = _currency(row)
    if not currency:
        return _unsupported_result(
            spec=spec,
            display_value=display_value,
            source_artifact="annual_balance_dashboard_metrics.csv",
            reason="missing_currency",
            period=period,
            currency=currency,
            evidence=pd.DataFrame(),
            sections=[],
        )

    component_values: list[float] = []
    component_frames: list[pd.DataFrame] = []
    for ref in spec.component_refs:
        value, reason, rows = _available_scalar(
            annual,
            ref=ref,
            period=period,
            currency=currency,
        )
        component_frames.append(rows)
        if value is None:
            evidence = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
            return _unsupported_result(
                spec=spec,
                display_value=display_value,
                source_artifact="annual_balance_dashboard_metrics.csv",
                reason=reason,
                period=period,
                currency=currency,
                evidence=evidence,
                sections=[("Governed component rows", evidence)],
            )
        component_values.append(value)

    component_rows = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()

    if spec.operation == "ratio" and abs(component_values[1]) <= spec.tolerance:
        return _unsupported_result(
            spec=spec,
            display_value=display_value,
            source_artifact="annual_balance_dashboard_metrics.csv",
            reason="zero_denominator:not_applicable",
            period=period,
            currency=currency,
            evidence=component_rows,
            sections=[("Governed component rows", component_rows)],
        )

    composed = _compose(spec, component_values)

    if spec.authority_mode == "source_value_with_formula_reconciliation":
        assert spec.source_value_ref is not None
        source_value, source_reason, source_rows = _available_scalar(
            annual,
            ref=spec.source_value_ref,
            period=period,
            currency=currency,
        )
        if source_value is None:
            evidence = pd.concat([source_rows, component_rows], ignore_index=True)
            return _unsupported_result(
                spec=spec,
                display_value=display_value,
                source_artifact="annual_balance_dashboard_metrics.csv",
                reason=f"source_authority_{source_reason}",
                period=period,
                currency=currency,
                evidence=evidence,
                sections=[("Governed source metric row", source_rows), ("Governed component rows", component_rows)],
            )
        matched = source_value
        formula_residual = composed - source_value
        display_residual = matched - display_value
        status = "ok" if abs(display_residual) <= tolerance and abs(formula_residual) <= spec.tolerance else "residual_warning"
        formula_rows = pd.DataFrame([
            {
                "derived_metric_id": spec.spec_id,
                "authority_mode": spec.authority_mode,
                "operation": spec.operation,
                "source_value": source_value,
                "composed_value": composed,
                "formula_residual": formula_residual,
                "displayed_value": display_value,
                "display_residual": display_residual,
            }
        ])
        return (
            status,
            matched,
            display_residual,
            "governed_source_value_with_formula_reconciliation",
            "annual_balance_dashboard_metrics.csv",
            {
                "derived_metric_id": spec.spec_id,
                "executor": "governed_derived_metric_v1",
                "authority_mode": spec.authority_mode,
                "source_value_ref": spec.source_value_ref,
                "component_refs": list(spec.component_refs),
                "operation": spec.operation,
                "formula_residual": formula_residual,
                "period": period,
                "Currency": currency,
            },
            "Upstream metric is authoritative; formula composition is independent reconciliation/explanation only.",
            source_rows,
            [("Governed source metric row", source_rows), ("Formula reconciliation", formula_rows), ("Governed component rows", component_rows)],
        )

    matched = composed
    residual = matched - display_value
    status = "ok" if abs(residual) <= tolerance else "residual_warning"
    formula_rows = pd.DataFrame([
        {
            "derived_metric_id": spec.spec_id,
            "operation": spec.operation,
            "component_refs": ";".join(spec.component_refs),
            "matched_value": matched,
            "displayed_value": display_value,
            "residual": residual,
        }
    ])
    return (
        status,
        matched,
        residual,
        "governed_derived_formula",
        "annual_balance_dashboard_metrics.csv",
        {
            "derived_metric_id": spec.spec_id,
            "executor": "governed_derived_metric_v1",
            "authority_mode": spec.authority_mode,
            "component_refs": list(spec.component_refs),
            "operation": spec.operation,
            "zero_denominator_policy": spec.zero_denominator_policy,
            "period": period,
            "Currency": currency,
        },
        "Computed only from governed annual scalar metric authorities; no ledger or semantic membership is rediscovered here.",
        component_rows,
        [("Formula", formula_rows), ("Governed component rows", component_rows)],
    )


def _previous_month(period: str) -> str:
    try:
        return str(pd.Period(str(period), freq="M") - 1)
    except Exception:
        return ""


def _execute_diagnostic_delta(
    *,
    spec: DerivedMetricSpec,
    row: pd.Series,
    period: str,
    display_value: float,
    cash_close: pd.DataFrame,
    tolerance: float,
):
    currency, box = _currency(row), _norm(row.get("Box"))
    previous_period = _previous_month(period)
    if not previous_period:
        return _unsupported_result(
            spec=spec,
            display_value=display_value,
            source_artifact="monthly_cash_close.csv",
            reason="invalid_month_period",
            period=period,
            currency=currency,
            evidence=pd.DataFrame(),
            sections=[],
        )
    current = select_inferred_box_control_period(
        cash_close, period=period, currency=currency, box=box
    )
    previous = select_inferred_box_control_period(
        cash_close, period=previous_period, currency=currency, box=box
    )
    sections = [
        ("Current inferred box-control snapshot", current.selected),
        ("Previous inferred box-control snapshot", previous.selected),
        ("Current inferred candidates", current.candidates),
        ("Previous inferred candidates", previous.candidates),
    ]
    if not current.available or not previous.available:
        reason = (
            f"current:{current.status}:{current.reason};"
            f"previous:{previous.status}:{previous.reason}"
        )
        evidence = pd.concat([current.selected, previous.selected], ignore_index=True)
        return _unsupported_result(
            spec=spec,
            display_value=display_value,
            source_artifact="monthly_cash_close.csv",
            reason=reason,
            period=period,
            currency=currency,
            evidence=evidence,
            sections=sections,
        )
    matched = float(current.value) - float(previous.value)
    residual = matched - display_value
    status = "ok" if abs(residual) <= tolerance else "residual_warning"
    formula_rows = pd.DataFrame([
        {
            "derived_metric_id": spec.spec_id,
            "period": period,
            "previous_period": previous_period,
            "Currency": currency,
            "Box": box,
            "current_inferred_control": float(current.value),
            "previous_inferred_control": float(previous.value),
            "matched_value": matched,
            "displayed_value": display_value,
            "residual": residual,
        }
    ])
    return (
        status,
        matched,
        residual,
        "governed_inferred_box_control_period_delta",
        "monthly_cash_close.csv",
        {
            "derived_metric_id": spec.spec_id,
            "executor": "governed_derived_metric_v1",
            "operation": "period_delta",
            "component_ref": "cash.control.inferred_box_motor",
            "period": period,
            "previous_period": previous_period,
            "Currency": currency,
            "Box": box,
            "missing_component_policy": spec.missing_component_policy,
            "validated_cash_fallback": "never",
        },
        "Diagnostic box level is the month-over-month delta of governed inferred-box control only; validated cash and internal balances cannot enter this formula.",
        formula_rows,
        [("Diagnostic formula", formula_rows), *sections],
    )


def execute_derived_metric(
    *,
    table_id: str,
    row: pd.Series,
    period: str,
    display_value: float,
    annual: pd.DataFrame,
    cash_close: pd.DataFrame,
    tolerance: float,
):
    spec_id = _norm(row.get("derived_metric_id"))
    if not spec_id:
        return None
    spec = resolve_derived_metric_spec(spec_id)
    if spec is None:
        raise ValueError(f"Unknown governed derived_metric_id: {spec_id!r}")

    if spec.operation == "period_delta":
        if cash_close is not None and not cash_close.empty and not inferred_control_schema_supported(cash_close):
            return None
        return _execute_diagnostic_delta(
            spec=spec,
            row=row,
            period=period,
            display_value=display_value,
            cash_close=cash_close,
            tolerance=tolerance,
        )

    # Historical/minimal annual artifacts remain on compatibility. Current
    # production annual metrics always include value_status and use the governed
    # path, where missing evidence then fails closed per the contract.
    if not annual_derived_schema_supported(annual):
        return None
    if "Y" not in spec.period_grains:
        return _unsupported_result(
            spec=spec,
            display_value=display_value,
            source_artifact="annual_balance_dashboard_metrics.csv",
            reason="period_grain_not_supported",
            period=period,
            currency=_currency(row),
            evidence=pd.DataFrame(),
            sections=[],
        )
    return _execute_annual_scalar(
        spec=spec,
        row=row,
        period=period,
        display_value=display_value,
        annual=annual,
        tolerance=tolerance,
    )
