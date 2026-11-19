from __future__ import annotations

"""TS04: gross physical monthly Box inflows/outflows, not operating income."""
import pandas as pd

from accounting.reports.time_series import (
    HISTORY_START_MONTH, TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)


def cash_inout_chart(
    group: pd.DataFrame, *, as_of_date: str,
) -> tuple[str, pd.DataFrame]:
    required = {
        "period", "Box", "Currency", "total_cash_in", "total_cash_out",
        "reconciliation_status",
    }
    if missing := required - set(group):
        raise ValueError(f"treasury chart missing {sorted(missing)}")
    if group.empty:
        return "", pd.DataFrame()
    if group["Box"].nunique() != 1 or group["Currency"].nunique() != 1:
        raise ValueError("TS04 cannot mix Boxes or native currencies")
    currency, box = str(group["Currency"].iloc[0]), str(group["Box"].iloc[0])
    rows = group.copy()
    if not as_of_date:
        raise ValueError("TS04 requires an explicit accounting cutoff")
    cutoff = pd.Timestamp(as_of_date)
    last = str(cutoff.to_period("M") if cutoff.is_month_end else cutoff.to_period("M") - 1)
    rows = rows.loc[rows["period"].astype(str).le(last)]
    rows = rows.loc[rows["period"].astype(str).ge(HISTORY_START_MONTH)].sort_values("period")
    if len(rows) < 2:
        return "", pd.DataFrame()
    if rows["period"].duplicated().any():
        raise ValueError("duplicate Box-month")
    if not rows["reconciliation_status"].isin(("reconciled", "calendar_zero")).all():
        raise ValueError("treasury inflow/outflow chart requires reconciled rows")
    for key in ("total_cash_in", "total_cash_out"):
        rows[key] = pd.to_numeric(rows[key], errors="raise")
        if rows[key].isna().any() or rows[key].lt(0).any():
            raise ValueError("invalid Box cash flows")
    if not rows[["total_cash_in", "total_cash_out"]].gt(.01).any().any():
        return "", pd.DataFrame()
    pairs = (
        ("cash_in", "total_cash_in", "Entradas físicas"),
        ("cash_out", "total_cash_out", "Salidas físicas"),
    )
    nonempty_source = rows.loc[rows["reconciliation_status"].eq("reconciled")]
    if nonempty_source.empty:
        return "", pd.DataFrame()
    first_recorded = str(nonempty_source["period"].iloc[0])
    points = []
    for row in rows.loc[rows["period"].astype(str).ge(first_recorded)].to_dict("records"):
        for series_id, field, _ in pairs:
            points.append({
                "period": str(row["period"]), "series_id": series_id,
                "value": float(row[field]), "unit": currency, "scope": box,
                "phase": "observed", "source_table": "monthly_cash_accountability.csv",
            })
    spec = TimeSeriesSpec(
        chart_id=f"ts04_cash_inout_{box.lower().replace(' ', '_')}_{currency.lower()}",
        title="¿Cuánto dinero entró y salió por mes?",
        frequency="M", mark="bars", unit=currency, scope=box,
        series_ids=tuple(p[0] for p in pairs),
        series_labels=tuple(p[2] for p in pairs),
        subtitle=f"{box} · {currency} · movimientos físicos registrados",
        note=(
            "Entradas y salidas de caja física registradas; incluyen movimientos no operativos "
            "y no equivalen a renta, ahorro o saldo bancario. "
            "El eje comienza en enero de 2022; meses previos al primer movimiento "
            "del Box quedan sin dato. Sólo meses completos hasta el corte."
        ), max_points=120,
        period_domain=tuple(pd.period_range(HISTORY_START_MONTH, last, freq="M").astype(str)),
    )
    svg, trace = render_timeseries_svg(spec, pd.DataFrame(points), height=278)
    return timeseries_figure(spec, svg), trace
