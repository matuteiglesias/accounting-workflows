from __future__ import annotations

"""Plot certified debt *positions* at each month-end; never sum stock over time."""
import pandas as pd

from accounting.reports.time_series import (
    HISTORY_START_MONTH, TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)


def build_debt_position_chart(
    positions: pd.DataFrame, *, currency: str = "USD",
) -> tuple[str, pd.DataFrame]:
    required = {
        "period", "debtor", "creditor", "Currency", "component",
        "position_status", "open_amount",
    }
    if missing := required - set(positions.columns):
        raise ValueError(f"debt chart missing source fields: {sorted(missing)}")
    frame = positions.loc[
        positions["component"].eq("total") &
        positions["position_status"].eq("available") &
        positions["Currency"].eq(currency),
        list(required),
    ].copy()
    if frame.empty:
        return "", pd.DataFrame()
    if frame.duplicated(["period", "debtor", "creditor", "Currency"]).any():
        raise ValueError("duplicated total debt position per relationship/month")
    frame["open_amount"] = pd.to_numeric(frame["open_amount"], errors="raise")
    if frame["open_amount"].isna().any() or frame["open_amount"].lt(-.01).any():
        raise ValueError("invalid monthly debt position")
    periods = sorted(p for p in frame["period"].astype(str).unique() if p >= HISTORY_START_MONTH)
    if len(periods) < 2:
        return "", pd.DataFrame()
    current = frame.loc[frame["period"].eq(periods[-1])].copy()
    current["relation"] = current["debtor"].astype(str) + " → " + current["creditor"].astype(str)
    # Top two by latest stock. Others remain in the total, not hidden in
    # an apparent zero or silently included in these two relationship lines.
    leaders = current.sort_values(
        ["open_amount", "relation"], ascending=[False, True], kind="stable"
    ).loc[lambda x: x["open_amount"].gt(.01), "relation"].head(2).tolist()
    subset = frame.loc[frame["period"].isin(periods)].copy()
    subset["relation"] = subset["debtor"].astype(str) + " → " + subset["creditor"].astype(str)
    data = []
    for period in periods:
        group = subset.loc[subset["period"].eq(period)]
        data.append({
            "period": period, "series_id": "total_open",
            "value": float(group["open_amount"].sum()),
            "unit": currency, "scope": "FBPM", "phase": "observed",
            "source_table": "monthly_debt_position.csv",
        })
        for leader in leaders:
            matching = group.loc[group["relation"].eq(leader)]
            if len(matching) == 1:
                data.append({
                    "period": period, "series_id": leader,
                    "value": float(matching["open_amount"].iloc[0]),
                    "unit": currency, "scope": "FBPM", "phase": "observed",
                    "source_table": "monthly_debt_position.csv",
                })
    spec = TimeSeriesSpec(
        chart_id=f"ts07_open_debt_{currency.lower()}",
        title="¿Cómo evolucionó el saldo abierto de las obligaciones registradas?",
        frequency="M", mark="line", unit=currency, scope="FBPM",
        series_ids=("total_open", *leaders),
        series_labels=("Total abierto", *leaders),
        subtitle=f"Posición mensual · {HISTORY_START_MONTH} a {periods[-1]} · {currency}",
        note=(
            "Stocks de deuda contable: el total suma relaciones de un mismo mes; "
            "nunca suma saldos entre meses. Se destacan hasta dos relaciones por saldo final. "
            "No determina exigibilidad ni responsabilidad jurídica. "
            "Tramos anteriores al primer registro quedan sin dato, no deuda cero."
        ),
        period_domain=tuple(pd.period_range(HISTORY_START_MONTH, periods[-1], freq="M").astype(str)),
        max_points=120,
    )
    svg, trace = render_timeseries_svg(spec, pd.DataFrame(data), width=940, height=310)
    return timeseries_figure(spec, svg), trace


def monthly_debt_position_table(positions: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Display monthly stocks from Jan-2022 with explicit unavailable months.

    This calendar is a rendering projection of known stock observations, NOT a
    reconstruction of liabilities before the debt subsystem has evidence.
    """
    required = {"period", "debtor", "creditor", "Currency", "component", "position_status", "open_amount"}
    if missing := required - set(positions):
        raise ValueError(f"debt history missing fields: {sorted(missing)}")
    frame = positions.loc[
        positions["component"].eq("total") &
        positions["position_status"].eq("available") &
        positions["period"].astype(str).ge(HISTORY_START_MONTH)
    ].copy()
    if frame.empty:
        return "", pd.DataFrame()
    if frame.duplicated(["period", "Currency", "debtor", "creditor"]).any():
        raise ValueError("duplicate monthly relationship debt stock")
    frame["open_amount"] = pd.to_numeric(frame["open_amount"], errors="raise")
    if frame.open_amount.isna().any() or frame.open_amount.lt(-0.01).any():
        raise ValueError("invalid available debt balance")
    tables, trace = [], []
    for currency, current in frame.groupby("Currency", sort=True):
        month_end = str(current["period"].astype(str).max())
        totals = current.groupby("period")["open_amount"].sum().to_dict()
        rows = []
        for month in pd.period_range(HISTORY_START_MONTH, month_end, freq="M").astype(str):
            if month in totals:
                value = float(totals[month])
                text_value = (f"{value:,.2f}").replace(",", "X").replace(".", ",").replace("X", ".")
                amount_label = f"{currency} {text_value}"
                status = "Observado"
                trace.append({"period": month, "Currency": currency, "open_total": value,
                              "status": "observed", "source_table": "monthly_debt_position.csv"})
            else:
                amount_label, status = "Sin registro", "Sin dato"
                trace.append({"period": month, "Currency": currency, "open_total": None,
                              "status": "unavailable", "source_table": "monthly_debt_position.csv"})
            rows.append(f'<tr class="{status.replace(" ", "-").lower()}">'
                        f'<td>{month}</td><td class="num">{amount_label}</td><td>{status}</td></tr>')
        tables.append(
            f'<h3>Deuda abierta mensual · {currency} · {HISTORY_START_MONTH}–{month_end}</h3>'
            '<p class="note">Saldos de cierre por mes, no sumables en el tiempo. '
            '«Sin registro» no significa deuda cero.</p>'
            '<div class="table-wrap"><table class="monthly-debt"><thead><tr><th>Mes</th>'
            '<th class="num">Deuda abierta</th><th>Cobertura</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>'
        )
    return "".join(tables), pd.DataFrame(trace)
