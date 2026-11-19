from __future__ import annotations

"""TS06: governed monthly operating result versus withdrawals (not cash savings)."""
import pandas as pd

from accounting.reports.time_series import (
    HISTORY_START_MONTH, TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)


def operating_vs_distributions_chart(
    monthly: pd.DataFrame, *, as_of_date: str,
) -> tuple[str, pd.DataFrame]:
    required = {
        "period", "Currency", "statement_line", "amount",
        "classification_coverage_ratio", "unknown_amount", "review_required_amount",
    }
    if missing := required - set(monthly):
        raise ValueError(f"TS06 missing governed source fields {sorted(missing)}")
    cutoff = pd.Timestamp(as_of_date)
    last_closed = cutoff.to_period("M") if cutoff.is_month_end else cutoff.to_period("M") - 1
    categories = ("net_operating", "family_draws_or_distributions")
    d = monthly.loc[
        monthly["Currency"].eq("ARS") &
        monthly["statement_line"].isin(categories) &
        monthly["period"].astype(str).le(str(last_closed))
    ].copy()
    periods = sorted(p for p in d.period.astype(str).unique() if p >= HISTORY_START_MONTH)
    if len(periods) < 2:
        return "", pd.DataFrame()
    if len(periods) >= 2 and pd.period_range(periods[0], periods[-1], freq="M").size != len(periods):
        raise ValueError("incomplete monthly TS06 calendar")
    d = d.loc[d.period.isin(periods)]
    if d.duplicated(["period", "statement_line"]).any():
        raise ValueError("duplicate TS06 statement line and month")
    if not all(set(group.statement_line) == set(categories) for _, group in d.groupby("period")):
        raise ValueError("TS06 requires both governed measures each month")
    for col in ("amount", "classification_coverage_ratio", "unknown_amount", "review_required_amount"):
        d[col] = pd.to_numeric(d[col], errors="raise")
    if d[["amount", "classification_coverage_ratio"]].isna().any().any():
        raise ValueError("TS06 source has missing numerical values")
    if d[["unknown_amount", "review_required_amount"]].gt(.01).any().any() or d["classification_coverage_ratio"].lt(.9999).any():
        raise ValueError("TS06 operating classification requires review")
    if d.loc[d.statement_line.eq("family_draws_or_distributions"), "amount"].lt(0).any():
        raise ValueError("negative TS06 withdrawal source")
    points = pd.DataFrame({
        "period": d["period"], "series_id": d["statement_line"],
        "value": d["amount"], "unit": "ARS", "scope": "FBPM",
        "phase": "observed", "source_table": "monthly_operating_statement.csv",
    })
    spec = TimeSeriesSpec(
        chart_id="ts06_operating_vs_distributions",
        title="¿Cómo evolucionaron el resultado operativo y los retiros?",
        frequency="M", mark="line", unit="ARS", scope="FBPM",
        series_ids=categories,
        series_labels=("Resultado operativo", "Retiros / distribuciones"),
        subtitle=f"{HISTORY_START_MONTH} a {str(last_closed)} · meses completos · ARS nominales",
        note=(
            "Comparación de flujos mensuales registrados. Los retiros son una categoría "
            "administrativa, no prueba de un derecho individual. La diferencia entre "
            "las curvas NO constituye ahorro bancario ni saldo disponible."
        ), max_points=120,
        period_domain=tuple(pd.period_range(HISTORY_START_MONTH, last_closed, freq="M").astype(str)),
    )
    svg, trace = render_timeseries_svg(spec, points, height=275)
    return timeseries_figure(spec, svg), trace
