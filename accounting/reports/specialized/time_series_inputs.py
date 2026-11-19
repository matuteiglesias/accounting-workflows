from __future__ import annotations

"""TS03: direct chart of the already-governed monthly rent view."""
import pandas as pd

from accounting.reports.time_series import (
    HISTORY_START_MONTH, TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)


def monthly_rent_chart(
    governed_months: pd.DataFrame, *, currency: str, as_of_date: str,
    metric: str = "rent",
) -> tuple[str, pd.DataFrame]:
    """One scope-level rental series, not an invented collector/Box split."""
    required = {"period", "Currency", "value", "scope", "source_table", "period_basis"}
    if missing := required - set(governed_months):
        raise ValueError(f"monthly rent chart view missing {sorted(missing)}")
    if metric not in ("rent", "opex"):
        raise ValueError("unrecognized monthly history metric")
    cutoff = pd.Timestamp(as_of_date)
    last_closed = str(
        cutoff.to_period("M") if cutoff.is_month_end
        else cutoff.to_period("M") - 1
    )
    rows = governed_months.loc[
        governed_months["Currency"].eq(currency) &
        governed_months["period"].astype(str).le(last_closed) &
        governed_months["period"].astype(str).ge(HISTORY_START_MONTH)
    ].copy()
    if len(rows) < 2:
        return "", pd.DataFrame()
    if not rows["period_basis"].eq("monthly").all():
        raise ValueError("rent evolution needs monthly governed values")
    if set(rows.scope) != {"FBPM"} or rows.duplicated(["period"]).any():
        raise ValueError("rent evolution needs unique FBPM month totals")
    rows["value"] = pd.to_numeric(rows["value"], errors="raise")
    if rows["value"].isna().any() or rows["value"].lt(0).any():
        raise ValueError("invalid governed monthly series values")
    points = pd.DataFrame({
        "period": rows["period"].astype(str), "series_id": metric,
        "value": rows["value"], "unit": currency, "scope": "FBPM",
        "phase": "observed", "source_table": rows["source_table"],
    })
    rent = metric == "rent"
    spec = TimeSeriesSpec(
        chart_id=f"{'ts03_monthly_rent' if rent else 'ts11_monthly_opex'}_{currency.lower()}",
        title=("¿Cómo evolucionó la renta reconocida mes a mes?" if rent
               else "¿Cómo evolucionaron los costos operativos mes a mes?"),
        frequency="M", mark="line", unit=currency, scope="FBPM",
        series_ids=(metric,), series_labels=("Renta reconocida" if rent else "OPEX de inmuebles",),
        subtitle=f"Family Business + Property Management · {currency} · desde enero 2022",
        note=(
            "Serie operativa gobernada; no identifica a quien cobró ni reconoce "
            "derecho a los frutos. El mes parcial puede figurar en la tabla, "
            "pero no en este gráfico."
            if rent else
            "Costos de propiedad clasificados. Meses sin valor registrado no se "
            "interpretan como costo cero; el mes parcial se detalla solo en tabla."
        ), max_points=120,
        period_domain=tuple(pd.period_range(HISTORY_START_MONTH, last_closed, freq="M").astype(str)),
    )
    svg, trace = render_timeseries_svg(spec, points, height=268)
    return timeseries_figure(spec, svg), trace
