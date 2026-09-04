from __future__ import annotations

"""Small governed pie-chart primitive for professional reports.

Charts consume already-materialized professional views.  This module contains
no ledger classification and no client-side aggregation.
"""

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable
from xml.sax.saxutils import escape

import pandas as pd


TOLERANCE = 0.01
PALETTE = ("#173b6c", "#315f98", "#6288b7", "#91abc9", "#b8c9dc", "#d0dbe8", "#718096", "#a0aec0")


@dataclass(frozen=True)
class PieSpec:
    chart_id: str
    source_metric: str
    measure: str
    slice_dimension: str
    currency: str
    scope: str
    period_basis: str
    period: str
    title: str
    subtitle: str = ""
    max_slices: int = 8


def validate_pie_population(spec: PieSpec, rows: pd.DataFrame, denominator: float) -> pd.DataFrame:
    required = {spec.slice_dimension, spec.measure, "Currency", "scope", "period_basis", "period"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"pie source missing required fields: {missing}")
    if rows.empty:
        raise ValueError(f"pie source is empty: {spec.chart_id}")
    if set(rows["Currency"].astype(str)) != {spec.currency}:
        raise ValueError(f"pie mixes currency or has wrong currency: {spec.chart_id}")
    if set(rows["scope"].astype(str)) != {spec.scope}:
        raise ValueError(f"pie mixes scope: {spec.chart_id}")
    if set(rows["period_basis"].astype(str)) != {spec.period_basis}:
        raise ValueError(f"pie mixes period basis: {spec.chart_id}")
    if set(rows["period"].astype(str)) != {str(spec.period)}:
        raise ValueError(f"pie mixes period: {spec.chart_id}")
    values = pd.to_numeric(rows[spec.measure], errors="coerce")
    if values.isna().any():
        raise ValueError(f"pie has unavailable values: {spec.chart_id}")
    if (values < -TOLERANCE).any():
        raise ValueError(f"pie has negative values: {spec.chart_id}")
    if rows[spec.slice_dimension].astype(str).str.strip().eq("").any():
        raise ValueError(f"pie has blank slice identity: {spec.chart_id}")
    if rows[spec.slice_dimension].astype(str).duplicated().any():
        raise ValueError(f"pie has duplicate slice identity: {spec.chart_id}")
    if len(rows) > spec.max_slices:
        raise ValueError(f"pie has too many slices: {spec.chart_id}")
    if abs(float(values.sum()) - float(denominator)) > TOLERANCE:
        raise ValueError(
            f"pie denominator mismatch {spec.chart_id}: slices={values.sum()} denominator={denominator}"
        )
    return rows.assign(_value=values.astype(float), _share=values.astype(float) / float(denominator))


def render_pie_svg(spec: PieSpec, rows: pd.DataFrame, denominator: float, *, width: int = 520, height: int = 245) -> tuple[str, pd.DataFrame]:
    prepared = validate_pie_population(spec, rows, denominator)
    cx, cy, radius = 102, 112, 78
    start = -pi / 2
    paths: list[str] = []
    trace_rows: list[dict[str, object]] = []
    for index, row in prepared.reset_index(drop=True).iterrows():
        value = float(row["_value"])
        share = float(row["_share"])
        end = start + share * 2 * pi
        large = 1 if share > 0.5 else 0
        x1, y1 = cx + radius * cos(start), cy + radius * sin(start)
        x2, y2 = cx + radius * cos(end), cy + radius * sin(end)
        d = f"M {cx},{cy} L {x1:.2f},{y1:.2f} A {radius},{radius} 0 {large},1 {x2:.2f},{y2:.2f} Z"
        paths.append(f'<path d="{d}" fill="{PALETTE[index % len(PALETTE)]}" stroke="#fff" stroke-width="1"/>')
        label = str(row[spec.slice_dimension])
        trace_rows.append({
            "chart_id": spec.chart_id, "metric_id": spec.source_metric,
            "slice_id": f"{spec.chart_id}:{label}", "slice_key": label,
            "display_label": label, "Currency": spec.currency, "scope": spec.scope,
            "period_basis": spec.period_basis, "period": spec.period,
            "value": value, "denominator": float(denominator), "share": share,
            "source_table": row.get("source_table", ""), "source_filter": row.get("source_filter", ""),
            "calculation_rule": row.get("calculation_rule", ""),
            "reporting_group_mapping_hash": row.get("reporting_group_mapping_hash", ""),
        })
        start = end

    def amount(value: float) -> str:
        if abs(value - round(value)) < TOLERANCE:
            return f"{value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    legend = []
    for index, row in prepared.reset_index(drop=True).iterrows():
        label = escape(str(row[spec.slice_dimension]))
        share = float(row["_share"]) * 100
        legend.append(
            f'<g transform="translate(210,{28 + index * 25})"><rect width="9" height="9" rx="2" fill="{PALETTE[index % len(PALETTE)]}"/>'
            f'<text x="16" y="9" class="pie-label">{label}</text><text x="16" y="21" class="pie-value">{escape(spec.currency)} {amount(float(row["_value"]))} · {share:.1f}%</text></g>'
        )
    svg = (
        f'<svg class="governed-pie" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(spec.title)}">'
        '<style>.pie-label{font:600 11px Inter,Arial,sans-serif;fill:#172033}.pie-value{font:10px Inter,Arial,sans-serif;fill:#667085}</style>'
        + "".join(paths)
        + ''.join(legend)
        + f'<text x="210" y="222" class="pie-total">Total: {escape(spec.currency)} {amount(float(denominator))}</text></svg>'
    )
    return svg, pd.DataFrame(trace_rows)


def professional_distribution_view(semantic_audit: pd.DataFrame, governed_totals: pd.DataFrame, *, scope: str = "FBPM") -> pd.DataFrame:
    """Build recipient slices from governed semantic distribution membership."""
    frame = semantic_audit.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    mask = frame["semantic_bucket"].astype(str).isin({"family_withdrawal_candidate", "family_withdrawal"})
    frame = frame.loc[mask & frame["Box"].astype(str).isin({"Family Business", "Property Management"})].copy()
    frame["recipient"] = frame["receiver"].astype(str).str.strip()
    frame["value"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.loc[frame["recipient"].ne("") & frame["value"].ge(0)]
    frame["year"] = frame["Date"].dt.year.astype("Int64").astype(str)
    rows = []
    for (year, currency, recipient), group in frame.groupby(["year", "Currency", "recipient"], sort=True):
        rows.append({
            "metric_id": "DIST.BY_RECIPIENT", "line_id": f"DIST.BY_RECIPIENT|{year}|{currency}|{recipient}",
            "period": year, "period_basis": "annual", "Currency": currency, "scope": scope,
            "recipient": recipient, "value": float(group["value"].sum()),
            "source_table": "classification_audit.csv", "source_filter": "semantic_bucket in family_withdrawal_candidate,family_withdrawal; receiver=recipient",
            "calculation_rule": "annual governed distribution membership summed by recipient; no ledger reclassification",
        })
    return pd.DataFrame(rows)


def professional_support_view(stakeholder_support: pd.DataFrame, *, scope: str = "FBPM") -> pd.DataFrame:
    frame = stakeholder_support.copy()
    frame = frame.loc[frame["target_box"].astype(str).eq("Property Management")].copy()
    frame["value"] = pd.to_numeric(frame["recognized_amount"], errors="coerce")
    frame["period_basis"] = "settlement"
    frame["period"] = frame["period"].astype(str).str[:4]
    rows = []
    for (period, currency, actor), group in frame.groupby(["period", "Currency", "funding_actor"], dropna=False, sort=True):
        actor = str(actor)
        rows.append({
            "metric_id": "SUPPORT.BY_ACTOR", "line_id": f"SUPPORT.BY_ACTOR|{period}|{currency}|{actor}",
            "period": period, "period_basis": "annual", "Currency": currency, "scope": scope,
            "funding_actor": actor, "reporting_group": str(group["reporting_group"].iloc[0]),
            "value": float(group["value"].sum()), "source_table": "monthly_stakeholder_support.csv",
            "source_filter": "target_box=Property Management; period_basis=settlement; funding_actor=actor",
            "calculation_rule": "annual recognized stakeholder support summed from governed mart",
        })
    return pd.DataFrame(rows)
