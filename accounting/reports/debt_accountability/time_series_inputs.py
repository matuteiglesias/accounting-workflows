from __future__ import annotations

"""Plot certified debt *positions* at each month-end; never sum stock over time."""
import pandas as pd

from accounting.reports.time_series import (
    TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)


def build_debt_position_chart(
    positions: pd.DataFrame, *, currency: str = "USD", max_months: int = 24,
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
    periods = sorted(frame["period"].astype(str).unique())[-max_months:]
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
        subtitle=f"Posición al cierre de cada mes · {currency} · últimos {len(periods)} meses",
        note=(
            "Stocks de deuda contable: el total suma relaciones de un mismo mes; "
            "nunca suma saldos entre meses. Se destacan hasta dos relaciones por saldo final. "
            "No determina exigibilidad ni responsabilidad jurídica."
        ),
    )
    svg, trace = render_timeseries_svg(spec, pd.DataFrame(data), width=940, height=310)
    return timeseries_figure(spec, svg), trace
