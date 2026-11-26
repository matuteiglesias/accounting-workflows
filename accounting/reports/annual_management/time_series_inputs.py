from __future__ import annotations

"""Certified adapter from existing monthly operating statement to two charts.

Like-for-like Jan..last fully closed month for each year, native ARS. Both
charts consume the same selected population; no ledger rescan is permitted.
"""
from datetime import datetime

import pandas as pd

from accounting.reports.time_series import (
    TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)

COMPONENTS = (
    ("taxes", "Impuestos"),
    ("services", "Servicios"),
    ("maintenance", "Mantenimiento"),
    ("legal", "Gastos legales"),
    ("other_property_opex", "Otros OPEX"),
)
SOURCE = "monthly_operating_statement.csv"


def build_operating_chart_pair(
    monthly: pd.DataFrame, *, as_of_date: str,
) -> tuple[str, str, pd.DataFrame]:
    cutoff = pd.Timestamp(as_of_date)
    # Month-grain data: do not imply a partly completed calendar month is whole.
    completed_months = 12 if cutoff.is_month_end and cutoff.month == 12 else (
        cutoff.month if cutoff.is_month_end else cutoff.month - 1
    )
    if completed_months < 1:
        raise ValueError("no complete month available to compare")
    required = {"period", "Currency", "statement_line", "amount",
                "classification_coverage_ratio", "unknown_amount", "review_required_amount"}
    if required - set(monthly):
        raise ValueError(f"monthly operating source missing {sorted(required - set(monthly))}")
    ar = monthly.loc[
        monthly["Currency"].eq("ARS") & monthly["period"].astype(str).str.fullmatch(r"20\d\d-(0[1-9]|1[0-2])")
    ].copy()
    ar["year"] = ar["period"].str[:4]
    ar["month"] = ar["period"].str[-2:].astype(int)
    ar = ar.loc[
        ar["month"].le(completed_months) & ar["year"].le(str(cutoff.year))
    ]
    needed = {n for n, _ in COMPONENTS} | {"rent_revenue", "property_opex_true"}
    ar = ar.loc[ar["statement_line"].isin(needed)].copy()
    if ar.duplicated(["period", "statement_line"]).any():
        raise ValueError("duplicate operating source month and statement line")
    ar["amount"] = pd.to_numeric(ar["amount"], errors="raise")
    if ar["amount"].isna().any() or ar["amount"].lt(0).any():
        raise ValueError("nonfinite or negative source amounts")
    for col in ("classification_coverage_ratio", "unknown_amount", "review_required_amount"):
        ar[col] = pd.to_numeric(ar[col], errors="raise")
    # A known nonzero unknown/review component means the charts are not safe to
    # present as complete cost membership. Do not turn it into a zero bar.
    if ar[["unknown_amount", "review_required_amount"]].gt(.01).any().any():
        raise ValueError("unclassified operating amounts require review")
    if ar["classification_coverage_ratio"].lt(.9999).any():
        raise ValueError("operating classification incomplete")

    years = sorted(ar["year"].unique())
    if len(years) < 2:
        raise ValueError("two or more comparable years required")
    cost_points, share_points = [], []
    for year in years:
        group = ar.loc[ar["year"].eq(year)]
        periods = {f"{year}-{m:02d}" for m in range(1, completed_months + 1)}
        if set(group.period) != periods:
            raise ValueError(f"incomplete operating time coverage: {year}")
        costs = {}
        for category in needed:
            found = group.loc[group["statement_line"].eq(category)]
            if len(found) != completed_months:
                raise ValueError(f"missing source component {category}: {year}")
            costs[category] = float(found.amount.sum())
        amount = sum(costs[key] for key, _ in COMPONENTS)
        if abs(costs["property_opex_true"] - amount) > .02:
            raise ValueError(f"operating components do not reconcile for {year}")
        for key, _ in COMPONENTS:
            cost_points.append({
                "period": year, "series_id": key, "value": costs[key],
                "unit": "ARS", "scope": "FBPM", "phase": "observed",
                "source_table": SOURCE,
            })
        if costs["rent_revenue"] > .01:
            share_points.append({
                "period": year, "series_id": "opex_to_rent",
                "value": costs["property_opex_true"] / costs["rent_revenue"],
                "unit": "%", "scope": "FBPM", "phase": "observed",
                "source_table": SOURCE,
            })
    if len(share_points) < 2:
        raise ValueError("no comparable rent denominators for cost ratio")
    period_description = f"enero–{['', 'enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre'][completed_months]} de cada año"
    note = (
        f"Ventana homogénea: {period_description}. ARS nominales, sin deflactar. "
        f"Corte {cutoff.strftime('%d/%m/%Y')}; el mes en curso no se incluye."
    )
    cost_spec = TimeSeriesSpec(
        chart_id="ts01_operating_costs", title="¿Cómo evolucionaron los costos de los inmuebles?",
        frequency="Y", mark="stacked_bars", unit="ARS", scope="FBPM",
        series_ids=tuple(key for key, _ in COMPONENTS),
        series_labels=tuple(label for _, label in COMPONENTS),
        subtitle="Evolución anual comparable · ARS nominales",
        note=note, max_points=12,
    )
    share_spec = TimeSeriesSpec(
        chart_id="ts02_opex_to_rent", title="¿Qué proporción de la renta absorbieron los costos?",
        frequency="Y", mark="line", unit="%", scope="FBPM",
        series_ids=("opex_to_rent",), series_labels=("Costos / renta",),
        subtitle="Costos y renta en ARS; no incluye USD",
        note=f"Misma ventana: {period_description}. No mide causalidad ni eficiencia.",
        max_points=12,
    )
    costs_svg, costs_trace = render_timeseries_svg(
        cost_spec, pd.DataFrame(cost_points), width=940, height=266
    )
    share_svg, share_trace = render_timeseries_svg(
        share_spec, pd.DataFrame(share_points), width=520, height=252
    )
    return (
        timeseries_figure(cost_spec, costs_svg),
        timeseries_figure(share_spec, share_svg),
        pd.concat((costs_trace, share_trace), ignore_index=True),
    )
