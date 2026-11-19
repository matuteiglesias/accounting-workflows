from __future__ import annotations

"""TS03: direct chart of the already-governed monthly rent view."""
import pandas as pd

from accounting.reports.time_series import (
    TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)


def monthly_rent_chart(
    governed_months: pd.DataFrame, *, currency: str, as_of_date: str,
) -> tuple[str, pd.DataFrame]:
    """One scope-level rental series, not an invented collector/Box split."""
    required = {"period", "Currency", "value", "scope", "source_table", "period_basis"}
    if missing := required - set(governed_months):
        raise ValueError(f"monthly rent chart view missing {sorted(missing)}")
    cutoff = pd.Timestamp(as_of_date)
    last_closed = str(
        cutoff.to_period("M") if cutoff.is_month_end
        else cutoff.to_period("M") - 1
    )
    rows = governed_months.loc[
        governed_months["Currency"].eq(currency) &
        governed_months["period"].astype(str).le(last_closed)
    ].copy()
    if len(rows) < 2:
        return "", pd.DataFrame()
    if not rows["period_basis"].eq("monthly").all():
        raise ValueError("rent evolution needs monthly governed values")
    if set(rows.scope) != {"FBPM"} or rows.duplicated(["period"]).any():
        raise ValueError("rent evolution needs unique FBPM month totals")
    rows["value"] = pd.to_numeric(rows["value"], errors="raise")
    if rows["value"].isna().any() or rows["value"].lt(0).any():
        raise ValueError("invalid rent values")
    points = pd.DataFrame({
        "period": rows["period"].astype(str), "series_id": "rent",
        "value": rows["value"], "unit": currency, "scope": "FBPM",
        "phase": "observed", "source_table": rows["source_table"],
    })
    spec = TimeSeriesSpec(
        chart_id=f"ts03_monthly_rent_{currency.lower()}",
        title="¿Cómo evolucionó la renta reconocida mes a mes?",
        frequency="M", mark="line", unit=currency, scope="FBPM",
        series_ids=("rent",), series_labels=("Renta reconocida",),
        subtitle=f"Family Business + Property Management · {currency} · meses completos",
        note=(
            "Serie operativa gobernada; no identifica a quien cobró ni reconoce "
            "derecho a los frutos. El mes parcial puede figurar en la tabla, "
            "pero no en este gráfico."
        ), max_points=24,
    )
    svg, trace = render_timeseries_svg(spec, points, height=268)
    return timeseries_figure(spec, svg), trace
