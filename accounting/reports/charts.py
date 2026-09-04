from __future__ import annotations

"""Small governed pie-chart primitive for professional reports.

Charts consume already-materialized professional views. This module contains
no ledger classification and no client-side aggregation.
"""

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Mapping
from xml.sax.saxutils import escape

import pandas as pd


TOLERANCE = 0.01
PALETTE = (
    "#173b6c", "#315f98", "#6288b7", "#91abc9", "#b8c9dc", "#d0dbe8",
    "#718096", "#a0aec0", "#234f82", "#4b78a8", "#7f9fbe", "#c4d2e0",
)
ACTOR_IDENTITY_ALIASES = {"Hector": "Héctor"}
PAYMENT_DISPLAY_ALIASES = {
    "Family Business": "Caja FB",
    "Property Management": "Caja PM",
    "Inquilino": "Inquilino no identificado",
}


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


def _canonical_actor_identity(value: object) -> str:
    text = str(value or "").strip()
    return ACTOR_IDENTITY_ALIASES.get(text, text)


def _amount(value: float) -> str:
    if abs(value - round(value)) < TOLERANCE:
        return f"{value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_stable_color_map(
    rows: pd.DataFrame,
    slice_dimension: str,
    measure: str = "value",
) -> dict[str, str]:
    """Assign one deterministic family-level color per canonical slice identity."""
    if rows.empty:
        return {}
    if slice_dimension not in rows.columns or measure not in rows.columns:
        raise ValueError(f"color domain missing fields: {slice_dimension}, {measure}")
    work = rows[[slice_dimension, measure]].copy()
    work["_identity"] = work[slice_dimension].astype(str).str.strip()
    work["_value"] = pd.to_numeric(work[measure], errors="coerce")
    if work["_value"].isna().any() or work["_identity"].eq("").any():
        raise ValueError("color domain contains unavailable value or blank identity")
    ranked = (
        work.groupby("_identity", as_index=False, sort=False)["_value"].sum()
        .sort_values(["_value", "_identity"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return {
        str(row["_identity"]): PALETTE[index % len(PALETTE)]
        for index, (_, row) in enumerate(ranked.iterrows())
    }


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
    if float(denominator) <= TOLERANCE:
        raise ValueError(f"pie denominator must be positive: {spec.chart_id}")
    return rows.assign(_value=values.astype(float), _share=values.astype(float) / float(denominator))


def render_pie_svg(
    spec: PieSpec,
    rows: pd.DataFrame,
    denominator: float,
    *,
    width: int = 520,
    height: int = 245,
    color_map: Mapping[str, str] | None = None,
) -> tuple[str, pd.DataFrame]:
    prepared = validate_pie_population(spec, rows, denominator).copy()
    prepared["_slice_identity"] = prepared[spec.slice_dimension].astype(str).str.strip()
    prepared = prepared.sort_values(
        ["_value", "_slice_identity"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)

    def color_for(identity: str, index: int) -> str:
        if color_map and identity in color_map:
            return str(color_map[identity])
        return PALETTE[index % len(PALETTE)]

    def display_for(row: pd.Series) -> str:
        candidate = str(row.get("display_label", "") or "").strip()
        return candidate or str(row["_slice_identity"])

    trace_rows: list[dict[str, object]] = []
    for index, row in prepared.iterrows():
        identity = str(row["_slice_identity"])
        display_label = display_for(row)
        color = color_for(identity, index)
        trace_rows.append({
            "chart_id": spec.chart_id,
            "metric_id": spec.source_metric,
            "slice_id": f"{spec.chart_id}:{identity}",
            "slice_key": identity,
            "display_label": display_label,
            "Currency": spec.currency,
            "scope": spec.scope,
            "period_basis": spec.period_basis,
            "period": spec.period,
            "value": float(row["_value"]),
            "denominator": float(denominator),
            "share": float(row["_share"]),
            "color": color,
            "source_table": row.get("source_table", ""),
            "source_filter": row.get("source_filter", ""),
            "calculation_rule": row.get("calculation_rule", ""),
            "reporting_group_mapping_hash": row.get("reporting_group_mapping_hash", ""),
        })

    if len(prepared) == 1:
        row = prepared.iloc[0]
        identity = str(row["_slice_identity"])
        display_label = escape(display_for(row))
        color = color_for(identity, 0)
        single_height = 94
        svg = (
            f'<svg class="governed-pie pie-single" viewBox="0 0 {width} {single_height}" role="img" '
            f'aria-label="{escape(spec.title)}">'
            '<style>.pie-single-label{font:600 13px Inter,Arial,sans-serif;fill:#172033}'
            '.pie-single-value{font:11px Inter,Arial,sans-serif;fill:#667085}'
            '.pie-total{font:700 11px Inter,Arial,sans-serif;fill:#102a56}</style>'
            f'<rect x="18" y="19" width="12" height="12" rx="3" fill="{color}"/>'
            f'<text x="40" y="30" class="pie-single-label">{display_label}</text>'
            f'<text x="40" y="50" class="pie-single-value">{escape(spec.currency)} {_amount(float(row["_value"]))} · 100.0%</text>'
            f'<text x="40" y="74" class="pie-total">Total: {escape(spec.currency)} {_amount(float(denominator))}</text>'
            '</svg>'
        )
        return svg, pd.DataFrame(trace_rows)

    cx, cy, radius = 102, 112, 78
    legend_top = 28
    legend_step = 25
    legend_text_bottom = legend_top + (len(prepared) - 1) * legend_step + 21
    total_y = legend_text_bottom + 28
    dynamic_height = max(height, total_y + 18)
    start = -pi / 2
    paths: list[str] = []
    legend: list[str] = []

    for index, row in prepared.iterrows():
        value = float(row["_value"])
        share = float(row["_share"])
        end = start + share * 2 * pi
        large = 1 if share > 0.5 else 0
        x1, y1 = cx + radius * cos(start), cy + radius * sin(start)
        x2, y2 = cx + radius * cos(end), cy + radius * sin(end)
        d = f"M {cx},{cy} L {x1:.2f},{y1:.2f} A {radius},{radius} 0 {large},1 {x2:.2f},{y2:.2f} Z"
        identity = str(row["_slice_identity"])
        color = color_for(identity, index)
        paths.append(f'<path d="{d}" fill="{color}" stroke="#fff" stroke-width="1"/>')
        label = escape(display_for(row))
        share_pct = share * 100
        legend.append(
            f'<g transform="translate(210,{legend_top + index * legend_step})"><rect width="9" height="9" rx="2" fill="{color}"/>'
            f'<text x="16" y="9" class="pie-label">{label}</text>'
            f'<text x="16" y="21" class="pie-value">{escape(spec.currency)} {_amount(value)} · {share_pct:.1f}%</text></g>'
        )
        start = end

    svg = (
        f'<svg class="governed-pie" viewBox="0 0 {width} {dynamic_height}" role="img" aria-label="{escape(spec.title)}">'
        '<style>.pie-label{font:600 11px Inter,Arial,sans-serif;fill:#172033}'
        '.pie-value{font:10px Inter,Arial,sans-serif;fill:#667085}</style>'
        + "".join(paths)
        + "".join(legend)
        + f'<text x="210" y="{total_y}" class="pie-total">Total: {escape(spec.currency)} {_amount(float(denominator))}</text></svg>'
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
    frame["funding_actor"] = frame["funding_actor"].map(_canonical_actor_identity)
    if "reporting_group" in frame.columns:
        frame["reporting_group"] = frame["reporting_group"].map(_canonical_actor_identity)
    frame["value"] = pd.to_numeric(frame["recognized_amount"], errors="coerce")
    frame["period_basis"] = "settlement"
    frame["period"] = frame["period"].astype(str).str[:4]
    rows = []
    for (period, currency, actor), group in frame.groupby(["period", "Currency", "funding_actor"], dropna=False, sort=True):
        actor = str(actor)
        reporting_group = str(group["reporting_group"].iloc[0]) if "reporting_group" in group.columns else actor
        rows.append({
            "metric_id": "SUPPORT.BY_ACTOR", "line_id": f"SUPPORT.BY_ACTOR|{period}|{currency}|{actor}",
            "period": period, "period_basis": "annual", "Currency": currency, "scope": scope,
            "funding_actor": actor, "reporting_group": reporting_group,
            "value": float(group["value"].sum()), "source_table": "monthly_stakeholder_support.csv",
            "source_filter": "target_box=Property Management; period_basis=settlement; funding_actor=actor",
            "calculation_rule": "annual recognized stakeholder support summed from governed mart",
        })
    return pd.DataFrame(rows)


def professional_fb_receipts_view(semantic_audit: pd.DataFrame, *, scope: str = "FBPM") -> pd.DataFrame:
    """Transaction-backed Family Business cash receipts, grouped by nature.

    The view is deliberately not called distributions: it is the governed
    physical cash-in population for the FB Box, with no claim about custody or
    final destination.
    """
    frame = semantic_audit.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.loc[
        frame["Box"].astype(str).eq("Family Business")
        & frame["direction"].astype(str).eq("in")
        & frame["cash_effect"].astype(str).eq("cash_in_box")
    ].copy()
    frame["value"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.loc[frame["value"].ge(0)]
    frame["period"] = frame["Date"].dt.year.astype("Int64").astype(str)
    frame["receipt_nature"] = frame["semantic_subbucket"].astype(str).replace({"nan": "Sin clasificar"})
    rows = []
    for (period, currency, nature), group in frame.groupby(["period", "Currency", "receipt_nature"], sort=True):
        rows.append({
            "metric_id": "FB.CASH_RECEIPTS.BY_NATURE",
            "line_id": f"FB.CASH_RECEIPTS.BY_NATURE|{period}|{currency}|{nature}",
            "period": period, "period_basis": "annual", "Currency": currency, "scope": scope,
            "receipt_nature": nature, "value": float(group["value"].sum()),
            "source_table": "classification_audit.csv",
            "source_filter": "Box=Family Business; direction=in; cash_effect=cash_in_box",
            "calculation_rule": "annual physical cash receipts grouped by governed semantic subbucket",
        })
    return pd.DataFrame(rows)


def professional_rent_receipts_view(semantic_audit: pd.DataFrame, *, scope: str = "FBPM") -> pd.DataFrame:
    """Governed rent cash receipts split by accounting Box."""
    frame = semantic_audit.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.loc[
        frame["Box"].astype(str).isin({"Family Business", "Property Management"})
        & frame["semantic_subbucket"].astype(str).eq("rent")
        & frame["direction"].astype(str).eq("in")
        & frame["cash_effect"].astype(str).eq("cash_in_box")
    ].copy()
    frame["value"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame["period"] = frame["Date"].dt.year.astype("Int64").astype(str)
    rows = []
    for (period, currency, box), group in frame.groupby(["period", "Currency", "Box"], sort=True):
        rows.append({
            "metric_id": "RENT.CASH_RECEIPTS.BY_BOX",
            "line_id": f"RENT.CASH_RECEIPTS.BY_BOX|{period}|{currency}|{box}",
            "period": period, "period_basis": "annual", "Currency": currency, "scope": scope,
            "box": box, "value": float(group["value"].sum()),
            "source_table": "classification_audit.csv",
            "source_filter": "semantic_subbucket=rent; direction=in; cash_effect=cash_in_box",
            "calculation_rule": "annual governed rent cash receipts grouped by accounting Box",
        })
    return pd.DataFrame(rows)


def professional_tax_service_support_view(stakeholder_support: pd.DataFrame, *, scope: str = "FBPM") -> pd.DataFrame:
    """Recognized taxes/services applied by actors to PM, not PM cash-out."""
    frame = stakeholder_support.copy()
    frame = frame.loc[
        frame["target_box"].astype(str).eq("Property Management")
        & frame["obligation_category"].astype(str).isin({"taxes", "services"})
    ].copy()
    frame["funding_actor"] = frame["funding_actor"].map(_canonical_actor_identity)
    frame["value"] = pd.to_numeric(frame["recognized_amount"], errors="coerce")
    frame["period_basis"] = "settlement"
    frame["period"] = frame["period"].astype(str).str[:4]
    rows = []
    for (period, currency, actor), group in frame.groupby(["period", "Currency", "funding_actor"], dropna=False, sort=True):
        actor = str(actor)
        rows.append({
            "metric_id": "SUPPORT.TAXES_SERVICES.BY_ACTOR",
            "line_id": f"SUPPORT.TAXES_SERVICES.BY_ACTOR|{period}|{currency}|{actor}",
            "period": period, "period_basis": "annual", "Currency": currency, "scope": scope,
            "funding_actor": actor, "value": float(group["value"].sum()),
            "source_table": "monthly_stakeholder_support.csv",
            "source_filter": "target_box=Property Management; obligation_category in taxes,services",
            "calculation_rule": "annual recognized taxes/services support grouped by funding actor",
        })
    return pd.DataFrame(rows)


def professional_tax_service_payment_view(semantic_audit: pd.DataFrame, *, scope: str = "FBPM") -> pd.DataFrame:
    """All governed tax/service applications, by identified payer/supporter.

    Economic-expense legs are not counted when a separate direct-support leg
    exists. Box cash expenses remain attributed to the paying Box.
    """
    frame = semantic_audit.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.loc[
        frame["semantic_subbucket"].astype(str).isin({"taxes", "services"})
        & frame["Box"].astype(str).isin({"Family Business", "Property Management"})
        & frame["cash_effect"].astype(str).isin({"cash_out_box", "no_cash_in_box_direct_payment", "no_cash_out_box_direct_payment"})
        & ~((frame["cash_effect"].astype(str).eq("no_cash_out_box_direct_payment")) & frame["leg_role"].astype(str).eq("economic_expense"))
    ].copy()
    frame["payer_actor"] = frame["funding_actor"].where(
        frame["funding_actor"].notna() & frame["funding_actor"].astype(str).ne(""),
        frame["Box"],
    )
    frame["payer_actor"] = frame["payer_actor"].map(_canonical_actor_identity)
    frame["value"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame["period"] = frame["Date"].dt.year.astype("Int64").astype(str)
    rows = []
    for (period, currency, actor), group in frame.groupby(["period", "Currency", "payer_actor"], sort=True):
        actor = str(actor)
        rows.append({
            "metric_id": "TAX_SERVICE.PAYMENTS.BY_ACTOR",
            "line_id": f"TAX_SERVICE.PAYMENTS.BY_ACTOR|{period}|{currency}|{actor}",
            "period": period,
            "period_basis": "annual",
            "Currency": currency,
            "scope": scope,
            "funding_actor": actor,
            "display_label": PAYMENT_DISPLAY_ALIASES.get(actor, actor),
            "value": float(group["value"].sum()),
            "source_table": "classification_audit.csv",
            "source_filter": "semantic_subbucket in taxes,services; cash/payment leg; economic_expense direct mirror excluded",
            "calculation_rule": "annual identified tax/service applications grouped by funding actor or paying Box",
        })
    return pd.DataFrame(rows)
