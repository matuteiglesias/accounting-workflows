from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd

from accounting.reports.treasury_accountability.spec import (
    ALWAYS_SHOW,
    BOX_ORDER,
    CURRENCY_ORDER,
    GROUPS,
    LABELS,
    START_PERIOD,
)
from accounting.reports.charts import (
    PieSpec,
    professional_distribution_view,
    professional_support_view,
    render_pie_svg,
)

TOL = 0.01


def _h(value: object) -> str:
    return html.escape(str(value))


def _fmt_num(value: object, *, zero_dash: bool = True) -> str:
    if value is None or pd.isna(value):
        return "No disponible"
    number = float(value)
    if abs(number) < 0.005:
        return "—" if zero_dash else "0"
    negative = number < 0
    number = abs(number)
    if abs(number - round(number)) < 0.005:
        rendered = f"{number:,.0f}"
    else:
        rendered = f"{number:,.2f}"
    rendered = rendered.replace(",", "X").replace(".", ",").replace("X", ".")
    return ("−" if negative else "") + rendered


def _compact(value: object) -> str:
    if value is None or pd.isna(value):
        return "No disponible"
    number = float(value)
    absolute = abs(number)
    sign = "−" if number < 0 else ""
    if absolute >= 1_000_000:
        rendered = f"{absolute / 1_000_000:.2f} M"
    elif absolute >= 1_000:
        rendered = f"{absolute / 1_000:.1f} k"
    else:
        rendered = f"{absolute:.2f}" if abs(absolute - round(absolute)) > 0.005 else f"{absolute:.0f}"
    return sign + rendered.replace(".", ",")


def complete_calendar(group: pd.DataFrame, start_period: str, max_period: str) -> pd.DataFrame:
    periods = pd.period_range(start_period, max_period, freq="M").astype(str)
    out = pd.DataFrame({"period": periods}).merge(group.copy(), on="period", how="left")
    out["Box"] = group["Box"].iloc[0]
    out["Currency"] = group["Currency"].iloc[0]
    out["period_end"] = pd.PeriodIndex(out["period"], freq="M").end_time.date.astype(str)

    numeric_exclusions = {
        "period", "period_end", "control_as_of_date", "Box", "Currency",
        "reconciliation_status", "validated_cash_status", "validated_cash_reason",
        "validated_as_of_date", "anchor_alignment_status", "debt_reconciliation_status",
    }
    for col in [col for col in group.columns if col not in numeric_exclusions]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    movement_cols = [
        col
        for _, cols in GROUPS
        for col in cols
        if col not in {"opening_control", "closing_control"}
    ]
    for col in set(
        movement_cols
        + ["box_motor_net", "box_flow_net", "reconciliation_gap", "n_tx", "n_review_required"]
    ):
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    # The report convention is explicitly zero-origin. It does not manufacture
    # validated liquidity; it merely carries the source net movement over the
    # completed calendar.
    out["closing_control"] = out["net_cash_flow"].fillna(0).cumsum()
    out["opening_control"] = out["closing_control"] - out["net_cash_flow"].fillna(0)
    out["reconciliation_status"] = out["reconciliation_status"].fillna("calendar_zero")
    out["validated_cash_status"] = out["validated_cash_status"].fillna("unavailable")
    out["validated_cash_reason"] = out["validated_cash_reason"].fillna("no_source_row")
    out["n_review_required"] = out["n_review_required"].fillna(0)
    return out


def _active_columns(group: pd.DataFrame) -> list[tuple[str, list[str]]]:
    groups = []
    for group_label, cols in GROUPS:
        active = []
        for col in cols:
            if col not in group.columns:
                continue
            total_abs = pd.to_numeric(group[col], errors="coerce").fillna(0).abs().sum()
            if col in ALWAYS_SHOW or total_abs > TOL:
                active.append(col)
        if active:
            groups.append((group_label, active))
    return groups


def _svg_chart(group: pd.DataFrame, currency: str) -> str:
    width, height = 1180, 275
    left, right, top, bottom = 58, 20, 18, 38
    plot_width, plot_height = width - left - right, height - top - bottom
    closing = group["closing_control"].fillna(0).astype(float).tolist()
    net = group["net_cash_flow"].fillna(0).astype(float).tolist()
    n = len(group)
    values = closing + net + [0]
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        vmin -= 1
        vmax += 1
    padding = (vmax - vmin) * 0.08
    vmin -= padding
    vmax += padding

    def y(value: float) -> float:
        return top + (vmax - value) / (vmax - vmin) * plot_height

    def x(index: int) -> float:
        return left + (index / max(n - 1, 1)) * plot_width

    zero_y = y(0)
    parts = [
        f'<svg class="cash-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Evolución mensual del control acumulado y flujo neto {_h(currency)}">'
    ]
    for tick in [vmin + (vmax - vmin) * i / 4 for i in range(5)]:
        yy = y(tick)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" '
            'stroke="#e6e9ef" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-7}" y="{yy+3:.1f}" text-anchor="end" font-size="8" '
            f'fill="#7b8493">{_h(_compact(tick))}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" '
        'stroke="#aeb8c7" stroke-width="1"/>'
    )

    step = plot_width / max(n, 1)
    bar_width = max(2.0, min(11.0, step * 0.55))
    for index, value in enumerate(net):
        xx = x(index) - bar_width / 2
        yy = y(value)
        rect_top = min(yy, zero_y)
        rect_height = max(1.0, abs(zero_y - yy))
        fill = "#aeb8c7" if value >= 0 else "#6f7a89"
        parts.append(
            f'<rect x="{xx:.1f}" y="{rect_top:.1f}" width="{bar_width:.1f}" '
            f'height="{rect_height:.1f}" fill="{fill}" opacity=".62"/>'
        )

    path = " ".join(
        ("M" if index == 0 else "L") + f"{x(index):.1f},{y(value):.1f}"
        for index, value in enumerate(closing)
    )
    parts.append(f'<path d="{path}" fill="none" stroke="#315f98" stroke-width="2.3"/>')
    if closing:
        parts.append(
            f'<circle cx="{x(n-1):.1f}" cy="{y(closing[-1]):.1f}" r="3.5" fill="#102a56"/>'
        )

    years_seen: list[str] = []
    for index, period in enumerate(group["period"].astype(str)):
        year = period[:4]
        if year in years_seen:
            continue
        years_seen.append(year)
        parts.append(
            f'<text x="{x(index):.1f}" y="{height-14}" font-size="8" fill="#667085">{year}</text>'
        )
        parts.append(
            f'<line x1="{x(index):.1f}" y1="{top}" x2="{x(index):.1f}" y2="{height-bottom}" '
            'stroke="#eef0f4" stroke-width="1"/>'
        )

    parts.append(
        f'<line x1="{width-260}" y1="13" x2="{width-238}" y2="13" '
        'stroke="#315f98" stroke-width="2.3"/>'
    )
    parts.append(
        f'<text x="{width-232}" y="16" font-size="8" fill="#667085">control acumulado</text>'
    )
    parts.append(
        f'<rect x="{width-130}" y="7" width="8" height="8" fill="#aeb8c7" opacity=".7"/>'
    )
    parts.append(
        f'<text x="{width-117}" y="16" font-size="8" fill="#667085">flujo neto</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _table_html(group: pd.DataFrame) -> str:
    groups = _active_columns(group)
    flat_cols = [col for _, cols in groups for col in cols]
    first_of_group = {cols[0] for _, cols in groups if cols}
    parts = ['<div class="table-wrap"><table class="treasury">']
    parts.append('<thead><tr class="group-head"><th class="month" rowspan="2">Mes</th>')
    for label, cols in groups:
        parts.append(f'<th colspan="{len(cols)}">{_h(label)}</th>')
    parts.append('</tr><tr class="col-head">')
    for col in flat_cols:
        css_class = "group-divider" if col in first_of_group else ""
        parts.append(f'<th class="{css_class}">{_h(LABELS.get(col, col))}</th>')
    parts.append("</tr></thead><tbody>")

    prior_year = None
    for _, row in group.iterrows():
        period = str(row["period"])
        year = period[:4]
        row_class = "year-start" if prior_year is not None and year != prior_year else ""
        prior_year = year
        parts.append(f'<tr class="{row_class}"><td class="month">{_h(period)}</td>')
        for col in flat_cols:
            value = row.get(col, 0)
            display = _fmt_num(value, zero_dash=col not in {"opening_control", "closing_control"})
            classes = []
            if col in first_of_group:
                classes.append("group-divider")
            if col in {
                "direct_tax_support_non_cash",
                "direct_service_support_non_cash",
                "other_non_cash_support",
            }:
                classes.append("non-cash")
            if col == "net_cash_flow":
                classes.append("net-pos" if float(value) >= 0 else "net-neg")
            if col == "closing_control":
                classes.append("close")
            if display == "—":
                classes.append("zero")
            parts.append(f'<td class="{" ".join(classes)}">{_h(display)}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _status_summary(group: pd.DataFrame) -> tuple[bool, float, int, int]:
    source_rows = group.loc[~group["reconciliation_status"].eq("calendar_zero")].copy()
    reconciled = source_rows.empty or source_rows["reconciliation_status"].astype(str).eq("reconciled").all()
    max_gap = (
        pd.to_numeric(source_rows["reconciliation_gap"], errors="coerce").abs().max()
        if not source_rows.empty
        else 0.0
    )
    reviews = (
        int(pd.to_numeric(source_rows["n_review_required"], errors="coerce").fillna(0).sum())
        if not source_rows.empty
        else 0
    )
    validated = (
        source_rows.loc[source_rows["validated_cash_status"].astype(str).eq("available")]
        if not source_rows.empty
        else source_rows
    )
    return bool(reconciled), float(max_gap or 0.0), reviews, len(validated)


def _series_html(group: pd.DataFrame) -> str:
    box = str(group["Box"].iloc[0])
    currency = str(group["Currency"].iloc[0])
    slug = box.lower().replace(" ", "-") + "-" + currency.lower()
    latest = group.iloc[-1]
    total_in = float(group["total_cash_in"].sum())
    total_out = float(group["total_cash_out"].sum())
    reconciled, max_gap, reviews, validated_count = _status_summary(group)
    validated_text = "disponible" if validated_count else "no disponible"
    period_range = f'{group["period"].iloc[0]} → {group["period"].iloc[-1]}'

    return f'''<section class="series-section" id="{_h(slug)}">
      <div class="series-title-row"><div class="series-title">{_h(currency)}</div><div class="series-range">{_h(period_range)}</div></div>
      <div class="kpis">
        <div class="kpi"><div class="kpi-label">Control contable acumulado</div><div class="kpi-value navy">{_h(_compact(latest["closing_control"]))}</div><div class="kpi-note">No equivale a caja validada.</div></div>
        <div class="kpi"><div class="kpi-label">Último flujo neto</div><div class="kpi-value">{_h(_compact(latest["net_cash_flow"]))}</div><div class="kpi-note">{_h(latest["period"])}</div></div>
        <div class="kpi"><div class="kpi-label">Entradas acumuladas</div><div class="kpi-value">{_h(_compact(total_in))}</div><div class="kpi-note">movimientos registrados</div></div>
        <div class="kpi"><div class="kpi-label">Salidas acumuladas</div><div class="kpi-value">{_h(_compact(total_out))}</div><div class="kpi-note">movimientos registrados</div></div>
        <div class="kpi"><div class="kpi-label">Caja validada</div><div class="kpi-value">{_h(validated_text)}</div><div class="kpi-note">snapshot externo / aprobado</div></div>
      </div>
      <div class="status-strip">
        <span class="status-pill">Movimientos físicos: {"reconciliados" if reconciled else "revisar"}</span>
        <span class="status-pill">Gap máx.: {_h(_fmt_num(max_gap, zero_dash=False))}</span>
        <span class="status-pill">Review-required: {reviews}</span>
        <span class="status-pill">Moneda: {_h(currency)}</span>
      </div>
      <div class="chart-card">
        <div class="chart-title">Control acumulado y flujo neto mensual · {_h(currency)}</div>
        <div class="chart-subtitle">La línea es el control acumulado de origen cero; las barras muestran el movimiento neto que explica cada cambio.</div>
        {_svg_chart(group, currency)}
      </div>
      {_table_html(group)}
      <div class="table-note">“Apoyo directo sin movimiento de caja” conserva obligaciones/apoyos económicos que no movieron físicamente fondos del Box. Las rayas (—) significan cero movimiento. Apertura y cierre muestran cero explícito cuando corresponde.</div>
    </section>'''


def _cycle_html(cycles: pd.DataFrame) -> str:
    if cycles.empty:
        return ""
    rows = []
    for _, row in cycles.sort_values(["Currency", "cycle_start"]).reset_index(drop=True).iterrows():
        closing = pd.to_numeric(pd.Series([row.get("closing_accountability_balance")]), errors="coerce").iloc[0]
        gap = row.get("accountability_gap") if str(row.get("accountability_gap_status")) == "available" else pd.NA
        rows.append(
            "<tr>"
            f"<td>{_h(row.get('cycle_start'))} → {_h(row.get('cycle_end'))}</td>"
            f"<td>{_h(row.get('view_type'))}</td><td>{_h(row.get('Currency'))}</td>"
            f"<td>{_h(_fmt_num(row.get('opening_accountability_balance')))}</td>"
            f"<td>{_h(_fmt_num(row.get('accountable_receipts')))}</td>"
            f"<td>{_h(_fmt_num(row.get('documented_distributions')))}</td>"
            f"<td>{_h(_fmt_num(row.get('supported_uses')))}</td>"
            f"<td>{_h(_fmt_num(row.get('documented_transfers_out')))}</td>"
            f"<td>{_h(_fmt_num(row.get('closing_accountability_balance')))}</td>"
            f"<td>{_h(_fmt_num(gap))}</td></tr>"
        )
    return f'''<section class="box-section" id="fb-accountability-cycles"><header class="box-head"><h2 class="box-title">FAMILY BUSINESS · CICLOS DE RENDICIÓN</h2><div class="box-subtitle">Cortes determinísticos Mar–Ago / Sep–Feb. Son fechas de revisión, no fechas legales de distribución.</div></header>
      <div class="chart-card"><div class="chart-title">Saldo contable acumulado sujeto a rendición</div><div class="chart-subtitle">Cierres históricos por ciclo plenamente cubierto. No representa caja real.</div></div>
      <div class="table-wrap"><table class="treasury"><thead><tr><th>Ciclo</th><th>Vista</th><th>Moneda</th><th>Apertura</th><th>Fondos recibidos</th><th>Distribuciones</th><th>Usos</th><th>Transferencias</th><th>Saldo bajo rendición</th><th>Gap vs caja</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
      <div class="table-note">El gap contra liquidez se muestra únicamente cuando existe una observación de caja validada y alineada; de lo contrario figura No disponible.</div></section>'''


def _stakeholder_support_html(support: pd.DataFrame) -> str:
    if support.empty:
        return ""
    work = support.loc[support["target_box"].astype(str).eq("Property Management")].copy()
    if work.empty:
        return ""
    work["recognized_amount"] = pd.to_numeric(work["recognized_amount"], errors="coerce").fillna(0.0)
    role_labels = {
        "tenant": "Inquilino/a", "tenant_family": "Familia inquilina",
        "family_funder": "Aportante familiar", "other": "Otro",
        "unavailable": "No disponible",
    }
    rows = []
    for (currency, actor, role), group in work.groupby(
        ["Currency", "funding_actor", "actor_role"], dropna=False, sort=True
    ):
        categories = group.groupby("obligation_category")["recognized_amount"].sum()
        taxes = float(categories.get("taxes", 0.0))
        services = float(categories.get("services", 0.0))
        total = float(group["recognized_amount"].sum())
        other = total - taxes - services
        rows.append(
            f"<tr><td>{_h(actor)}</td><td>{_h(role_labels.get(str(role), str(role)))}</td>"
            f"<td>{_h(currency)}</td><td>{_h(_fmt_num(taxes))}</td>"
            f"<td>{_h(_fmt_num(services))}</td><td>{_h(_fmt_num(other))}</td>"
            f"<td>{_h(_fmt_num(total))}</td></tr>"
        )
    return f'''<section class="box-section" id="pm-stakeholder-support"><header class="box-head"><h2 class="box-title">PROPERTY MANAGEMENT · PAGOS Y APORTES APLICADOS POR ACTORES</h2><div class="box-subtitle">Apoyo reconocido para obligaciones de PM sin afirmar que el dinero haya ingresado a su caja.</div></header>
      <div class="table-wrap"><table class="treasury"><thead><tr><th>Actor</th><th>Rol</th><th>Moneda</th><th>Impuestos</th><th>Servicios</th><th>Otros</th><th>Total reconocido</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
      <div class="table-note">Estas aplicaciones son constructivas y están siempre identificadas por Box objetivo. No se suman como eventos físicos independientes ni establecen por sí solas un derecho legal de reintegro.</div></section>'''


def _governed_pie_card(spec: PieSpec, rows: pd.DataFrame, denominator: float) -> tuple[str, pd.DataFrame]:
    svg, trace = render_pie_svg(spec, rows, denominator)
    return (
        f'<div class="chart-card pie-card"><div class="chart-title">{_h(spec.title)}</div>'
        f'<div class="chart-subtitle">{_h(spec.subtitle)} · Total: {_h(spec.currency)} {_h(_fmt_num(denominator, zero_dash=False))}</div>{svg}</div>',
        trace,
    )


def _stakeholder_charts_html(
    *,
    semantic_audit: pd.DataFrame,
    stakeholder_support: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    out_dir: Path,
) -> str:
    distribution = professional_distribution_view(semantic_audit, annual_metrics)
    support = professional_support_view(stakeholder_support)
    cards: list[str] = []
    traces: list[pd.DataFrame] = []

    def metric_total(metric: str, period: str, currency: str) -> float:
        rows = annual_metrics.loc[
            annual_metrics["metric_id"].eq(metric)
            & annual_metrics["period"].astype(str).str.removesuffix(".0").eq(str(period))
            & annual_metrics["Currency"].astype(str).eq(currency)
        ]
        if len(rows) != 1:
            raise ValueError(f"governed chart denominator is not singular: {metric} {period} {currency}")
        value = pd.to_numeric(rows["value"], errors="coerce").iloc[0]
        if pd.isna(value) or float(value) < 0:
            raise ValueError(f"governed chart denominator unavailable: {metric} {period} {currency}")
        return float(value)

    available = []
    for period in sorted(set(distribution.get("period", pd.Series(dtype=str)).astype(str))):
        available.append(("distribution", period))
    for period in sorted(set(support.get("period", pd.Series(dtype=str)).astype(str))):
        available.append(("support", period))
    # Current report: latest completed annual period for historical context,
    # plus 2026 YTD where present. A compact cumulative card completes the view.
    years = sorted({p for _, p in available if p})
    selected = [y for y in years if y in {"2025", "2026"}] or (years[-1:] if years else [])
    for family, frame, metric, dim, title in [
        ("distribution", distribution, "DIST.DRAWS.PERSONAL", "recipient", "Distribuciones registradas por receptor"),
        ("support", support, "SUPPORT.BY_ACTOR", "funding_actor", "Pagos y aportes reconocidos por actor"),
    ]:
        for period in selected:
            for currency in sorted(set(frame.loc[frame["period"].eq(period), "Currency"].astype(str))):
                subset = frame.loc[(frame["period"].eq(period)) & frame["Currency"].astype(str).eq(currency)].copy()
                if subset.empty:
                    continue
                denominator = metric_total(metric, period, currency) if family == "distribution" else float(subset["value"].sum())
                spec = PieSpec(
                    chart_id=f"{family}_by_{dim}_{period}_{currency}", source_metric=metric,
                    measure="value", slice_dimension=dim, currency=currency, scope="FBPM",
                    period_basis="annual", period=period, title=f"{title} · {period}{' YTD · corte 31/08/2026' if period == '2026' else ''}",
                    subtitle="Acumulado anual nominal" if currency == "ARS" else "Total anual nominal",
                )
                card, trace = _governed_pie_card(spec, subset, denominator); cards.append(card); traces.append(trace)

        # Cumulative is a separate governed population, never a pie of mixed currencies.
        # Keep the cumulative support population in the internal chart authority;
        # the first human-facing v1 stays compact and renders cumulative
        # distributions plus annual/YTD support views without an orphan page.
        if family == "support":
            continue
        for currency in sorted(set(frame["Currency"].astype(str))):
            subset = frame.loc[frame["Currency"].astype(str).eq(currency)].groupby(dim, as_index=False)["value"].sum()
            subset["Currency"] = currency; subset["scope"] = "FBPM"; subset["period_basis"] = "cumulative"; subset["period"] = "all"
            if family == "distribution":
                denominators = annual_metrics.loc[annual_metrics.metric_id.eq(metric) & annual_metrics.Currency.astype(str).eq(currency), "value"]
                denominator = float(pd.to_numeric(denominators, errors="coerce").fillna(0).sum())
            else:
                denominator = float(subset["value"].sum())
            if denominator <= 0 or subset.empty:
                continue
            spec = PieSpec(
                chart_id=f"{family}_by_{dim}_cumulative_{currency}", source_metric=metric,
                measure="value", slice_dimension=dim, currency=currency, scope="FBPM",
                period_basis="cumulative", period="all", title=f"{title} · acumulado", subtitle="Acumulado nominal ARS" if currency == "ARS" else "Acumulado nominal",
            )
            card, trace = _governed_pie_card(spec, subset, denominator); cards.append(card); traces.append(trace)

    trace_path = out_dir / "chart_trace.csv"
    trace_frame = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    trace_frame.to_csv(trace_path, index=False)
    note = ('<div class="table-note chart-note">Los gráficos describen universos contables diferentes y no constituyen por sí solos una liquidación o neteo jurídico entre actores. El apoyo se mide por período de settlement y por Box objetivo; 2026 es YTD al corte 31/08/2026.</div>')
    return '<section class="box-section chart-section" id="governed-stakeholder-charts"><header class="box-head"><h2 class="box-title">QUIÉN RECIBIÓ · QUIÉN APORTÓ</h2><div class="box-subtitle">Vistas profesionales sobre autoridades gobernadas; las monedas permanecen separadas.</div></header><div class="pie-grid">' + ''.join(cards) + '</div>' + note + '</section>'


def build_validation(
    source: pd.DataFrame,
    source_qa: pd.DataFrame,
    completed: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(check: str, status: str, detail: str, severity: str = "error") -> None:
        rows.append({"check": check, "status": status, "severity": severity, "detail": detail})

    required = {
        "period", "Box", "Currency", "opening_control", "total_cash_in",
        "total_cash_out", "net_cash_flow", "closing_control", "reconciliation_gap",
        "reconciliation_status", "validated_cash_status", "n_review_required",
    }
    add(
        "input_schema",
        "pass" if required.issubset(source.columns) else "fail",
        f"missing={sorted(required - set(source.columns))}",
    )
    duplicates = source.duplicated(["period", "Box", "Currency"], keep=False)
    add("source_key_unique", "pass" if not duplicates.any() else "fail", f"duplicate_rows={int(duplicates.sum())}")

    max_identity = float(
        (source["opening_control"] + source["net_cash_flow"] - source["closing_control"])
        .abs()
        .max()
    )
    add(
        "source_opening_plus_net_equals_closing",
        "pass" if max_identity <= TOL else "fail",
        f"max_gap={max_identity}",
    )
    max_reconciliation = float(
        pd.to_numeric(source["reconciliation_gap"], errors="coerce").fillna(0).abs().max()
    )
    add(
        "source_components_equal_box_motor",
        "pass" if max_reconciliation <= TOL else "fail",
        f"max_gap={max_reconciliation}",
    )
    non_reconciled = source.loc[~source["reconciliation_status"].astype(str).eq("reconciled")]
    add(
        "source_rows_reconciled",
        "pass" if non_reconciled.empty else "fail",
        f"non_reconciled={len(non_reconciled)}",
    )

    if not source_qa.empty:
        failed = source_qa.loc[source_qa["status"].astype(str).str.lower().eq("fail")]
        warned = source_qa.loc[source_qa["status"].astype(str).str.lower().eq("warn")]
        add("source_qa_no_fail", "pass" if failed.empty else "fail", f"fails={len(failed)}")
        add(
            "source_qa_warnings",
            "pass" if warned.empty else "warn",
            f"warnings={len(warned)}",
            "warning",
        )

    for (box, currency), group in completed.items():
        max_gap = float(
            (group["opening_control"] + group["net_cash_flow"] - group["closing_control"])
            .abs()
            .max()
        )
        add(
            f"render_calendar_identity::{box}::{currency}",
            "pass" if max_gap <= TOL else "fail",
            f"max_gap={max_gap}",
        )

    unknown = float(
        source.get("unknown_cash_in", pd.Series(dtype=float)).fillna(0).sum()
        + source.get("unknown_cash_out", pd.Series(dtype=float)).fillna(0).sum()
    )
    add(
        "unknown_actual_cash_visible",
        "pass" if abs(unknown) <= TOL else "warn",
        f"amount={unknown}",
        "warning",
    )
    reviews = int(pd.to_numeric(source["n_review_required"], errors="coerce").fillna(0).sum())
    add(
        "review_required_visible",
        "pass" if reviews == 0 else "warn",
        f"n={reviews}",
        "warning",
    )
    validated = int(source["validated_cash_status"].astype(str).eq("available").sum())
    add(
        "validated_cash_anchor_availability",
        "info",
        f"available_rows={validated}; zero-origin control remains distinct from validated liquidity",
        "info",
    )
    return pd.DataFrame(rows)


def render_report(
    *,
    accountability_path: Path,
    qa_path: Path | None,
    cycles_path: Path | None = None,
    stakeholder_support_path: Path | None = None,
    semantic_audit_path: Path | None = None,
    annual_metrics_path: Path | None = None,
    out_dir: Path,
    start_period: str = START_PERIOD,
) -> dict[str, Path]:
    source = pd.read_csv(accountability_path)
    source_qa = pd.read_csv(qa_path) if qa_path and qa_path.exists() else pd.DataFrame()
    cycles = pd.read_csv(cycles_path) if cycles_path and cycles_path.exists() else pd.DataFrame()
    stakeholder_support = (
        pd.read_csv(stakeholder_support_path)
        if stakeholder_support_path and stakeholder_support_path.exists()
        else pd.DataFrame()
    )
    semantic_audit = pd.read_csv(semantic_audit_path) if semantic_audit_path and semantic_audit_path.exists() else pd.DataFrame()
    annual_metrics = pd.read_csv(annual_metrics_path) if annual_metrics_path and annual_metrics_path.exists() else pd.DataFrame()

    non_numeric = {
        "period", "period_end", "control_as_of_date", "Box", "Currency",
        "reconciliation_status", "validated_cash_status", "validated_cash_reason",
        "validated_as_of_date", "anchor_alignment_status", "debt_reconciliation_status",
    }
    for col in [col for col in source.columns if col not in non_numeric]:
        source[col] = pd.to_numeric(source[col], errors="coerce")

    max_period = source["period"].astype(str).max()
    cutoff_label = pd.Period(max_period, freq="M").end_time.strftime("%d/%m/%Y")
    if not cycles.empty:
        cutoff = pd.Period(max_period, freq="M").end_time.normalize()
        starts = pd.to_datetime(cycles["cycle_start"], errors="coerce")
        ends = pd.to_datetime(cycles["cycle_end"], errors="coerce")
        cycles = cycles.loc[starts.ge(pd.Timestamp("2022-03-01")) & ends.le(cutoff)].copy()
    completed: dict[tuple[str, str], pd.DataFrame] = {}
    for (box, currency), group in source.groupby(["Box", "Currency"], sort=False):
        completed[(str(box), str(currency))] = complete_calendar(
            group.sort_values("period"), start_period, max_period
        )

    validation = build_validation(source, source_qa, completed)
    failures = validation.loc[validation["status"].eq("fail")]

    sections = []
    toc = []
    for box in BOX_ORDER:
        series = []
        for currency in CURRENCY_ORDER:
            key = (box, currency)
            if key not in completed:
                continue
            series.append(_series_html(completed[key]))
            slug = box.lower().replace(" ", "-") + "-" + currency.lower()
            toc.append((f"{box} · {currency}", f"#{slug}"))
        if series:
            sections.append(
                f'''<section class="box-section"><header class="box-head"><h2 class="box-title">{_h(box.upper())}</h2><div class="box-subtitle">Rendición mensual de movimientos efectivos y control acumulado por moneda.</div></header>{"".join(series)}</section>'''
            )

    qa_items = "".join(
        f'<div class="qa-item"><strong>{_h(row["check"])} · {_h(str(row["status"]).upper())}</strong><span>{_h(row["detail"])}</span></div>'
        for _, row in validation.iterrows()
    )
    css = Path(__file__).with_name("report.css").read_text(encoding="utf-8")
    html_document = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rendición mensual de tesorería · {_h(max_period)}</title><style>{css}</style></head><body><div class="report-shell">
      <header class="hero"><div class="eyebrow">Family Business / Property Management</div><h1>Rendición mensual de tesorería</h1><div class="hero-sub">Movimientos efectivos de caja, aplicación mensual y evolución del control acumulado por Box y moneda. Cada fila explica cómo el flujo lleva de la apertura al cierre.</div><div class="meta-row"><span>Corte contable: {_h(cutoff_label)}</span><span>Período: {_h(start_period)} → {_h(max_period)}</span><span>Fuente: {_h(accountability_path.name)}</span><span>Convención: control de origen cero</span></div><nav class="toc">{"".join(f'<a href="{href}">{_h(label)}</a>' for label, href in toc)}</nav><div class="method-note"><strong>Base de control:</strong> 0 al inicio de {_h(start_period)}. Apertura y cierre representan el saldo acumulado de movimientos físicos registrados en el motor del Box. No constituyen saldos bancarios o de efectivo validados cuando <em>validated_cash_status</em> está unavailable. ARS y USD se mantienen siempre separados.</div></header>
      {_cycle_html(cycles)}
      {_stakeholder_support_html(stakeholder_support)}
      {(_stakeholder_charts_html(semantic_audit=semantic_audit, stakeholder_support=stakeholder_support, annual_metrics=annual_metrics, out_dir=out_dir) if not semantic_audit.empty and not annual_metrics.empty and not stakeholder_support.empty else '')}
      {"".join(sections)}
      <section class="qa-card"><h2>Control de integridad del reporte</h2><div class="qa-grid">{qa_items}</div></section>
      <footer class="footer"><span>Fuente contable: monthly_cash_accountability.csv · movimientos reconciliados contra box motor.</span><span>Generación reproducible; ver report_validation.csv.</span></footer>
    </div></body></html>'''

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "report.html"
    validation_path = out_dir / "report_validation.csv"
    summary_path = out_dir / "report_series_summary.csv"
    html_path.write_text(html_document, encoding="utf-8")
    validation.to_csv(validation_path, index=False)

    summary_rows = []
    for (box, currency), group in completed.items():
        latest = group.iloc[-1]
        summary_rows.append({
            "Box": box,
            "Currency": currency,
            "period_start": start_period,
            "period_end": max_period,
            "closing_control": latest["closing_control"],
            "latest_net_cash_flow": latest["net_cash_flow"],
            "total_cash_in": group["total_cash_in"].sum(),
            "total_cash_out": group["total_cash_out"].sum(),
            "validated_cash_available_rows": int(
                group["validated_cash_status"].astype(str).eq("available").sum()
            ),
            "n_review_required": int(
                pd.to_numeric(group["n_review_required"], errors="coerce").fillna(0).sum()
            ),
        })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    if not failures.empty:
        raise ValueError(
            f"treasury report validation failed with {len(failures)} hard failure(s); inspect report_validation.csv"
        )
    return {"html": html_path, "validation": validation_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render monthly treasury accountability HTML from the governed accountability mart."
    )
    parser.add_argument("--accountability", required=True, type=Path)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--cycles", type=Path)
    parser.add_argument("--stakeholder-support", type=Path)
    parser.add_argument("--semantic-audit", type=Path)
    parser.add_argument("--annual-metrics", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--start-period", default=START_PERIOD)
    args = parser.parse_args()
    outputs = render_report(
        accountability_path=args.accountability,
        qa_path=args.qa,
        cycles_path=args.cycles,
        stakeholder_support_path=args.stakeholder_support,
        semantic_audit_path=args.semantic_audit,
        annual_metrics_path=args.annual_metrics,
        out_dir=args.out_dir,
        start_period=args.start_period,
    )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
