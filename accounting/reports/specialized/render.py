from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from accounting.reports.charts import PieSpec, render_pie_svg
from accounting.reports.pdf import render_pdf
from accounting.reports.specialized.spec import REPORT_SPECS
from accounting.reports.specialized.views import SpecializedViewResult, build_specialized_view


def _money(value: float, currency: str) -> str:
    return f"{currency} {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _scope_label(scope: str) -> str:
    return {
        "FBPM": "Family Business + Property Management",
        "Property Management": "Property Management",
        "Family Business": "Family Business",
    }.get(scope, scope)


def _period_selection(frame: pd.DataFrame, policy: str, as_of_date: str) -> tuple[pd.DataFrame, str]:
    if frame.empty:
        return frame.copy(), "No disponible"
    cutoff = pd.Timestamp(as_of_date)
    year = str(cutoff.year)
    periods = frame["period"].astype(str)
    if policy == "latest_year":
        selected = frame.loc[periods.eq(year)].copy()
        if selected.empty:
            fallback = max((p[:4] for p in periods if len(p) >= 4), default=year)
            selected = frame.loc[periods.str[:4].eq(fallback)].copy()
            year = fallback
        label = year + (f" YTD · corte {cutoff.strftime('%d/%m/%Y')}" if year == str(cutoff.year) else "")
        return selected, label
    if policy == "latest_year_months":
        selected = frame.loc[periods.str[:4].eq(year)].copy()
        if selected.empty:
            fallback = max((p[:4] for p in periods if len(p) >= 4), default=year)
            selected = frame.loc[periods.str[:4].eq(fallback)].copy()
            year = fallback
        label = year + (f" YTD · corte {cutoff.strftime('%d/%m/%Y')}" if year == str(cutoff.year) else "")
        return selected.sort_values("period"), label
    raise ValueError(f"unsupported specialized period policy: {policy}")


def _table(frame: pd.DataFrame, columns: tuple[tuple[str, str], ...]) -> str:
    if frame.empty:
        return '<p class="muted">No disponible para este corte.</p>'
    available = [(column, label) for column, label in columns if column in frame.columns]
    view = frame[[column for column, _ in available]].copy()
    for column, _ in available:
        if column == "value":
            view[column] = [
                _money(float(value), str(currency))
                for value, currency in zip(view[column], frame["Currency"])
            ]
    labels = [label for _, label in available]
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + ''.join(f'<th>{html.escape(label)}</th>' for label in labels)
        + '</tr></thead><tbody>'
        + ''.join(
            '<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in row) + '</tr>'
            for row in view.itertuples(index=False, name=None)
        )
        + '</tbody></table></div>'
    )


def _summary_html(frame: pd.DataFrame, currency: str, dimension: str) -> str:
    total = float(pd.to_numeric(frame["value"], errors="coerce").sum())
    members = int(frame[dimension].astype(str).nunique()) if dimension in frame.columns else len(frame)
    return f'''<div class="metric-grid">
      <div class="metric"><span>Total reconocido</span><strong>{html.escape(_money(total, currency))}</strong></div>
      <div class="metric"><span>Componentes visibles</span><strong>{members}</strong></div>
    </div>'''


def _bars_html(frame: pd.DataFrame, dimension: str, currency: str, *, chronological: bool) -> str:
    if frame.empty:
        return '<p class="muted">No disponible para este corte.</p>'
    work = frame.copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.sort_values(dimension) if chronological else work.sort_values(["value", dimension], ascending=[False, True])
    maximum = float(work["value"].max()) if not work.empty else 0.0
    rows = []
    for _, row in work.iterrows():
        value = float(row["value"])
        width = 0.0 if maximum <= 0 else value / maximum * 100.0
        label = html.escape(str(row[dimension]))
        rows.append(
            '<div class="bar-row">'
            f'<div class="bar-label">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div>'
            f'<div class="bar-value">{html.escape(_money(value, currency))}</div>'
            '</div>'
        )
    return '<div class="bar-list">' + ''.join(rows) + '</div>'


def _comparison_html(frame: pd.DataFrame, dimension: str, currency: str) -> str:
    cards = []
    for _, row in frame.sort_values(["value", dimension], ascending=[False, True]).iterrows():
        cards.append(
            '<div class="metric">'
            f'<span>{html.escape(str(row[dimension]))}</span>'
            f'<strong>{html.escape(_money(float(row["value"]), currency))}</strong>'
            '</div>'
        )
    return '<div class="metric-grid">' + ''.join(cards) + '</div>' + _bars_html(
        frame, dimension, currency, chronological=False
    )


def _plain_trace(report_id: str, result: SpecializedViewResult, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "report_id": report_id,
            "metric_id": result.metric_id,
            "line_id": row.get("line_id", ""),
            "slice_key": row.get(result.dimension, ""),
            "display_label": row.get("display_label", row.get(result.dimension, "")),
            "Currency": row.get("Currency", ""),
            "scope": row.get("scope", ""),
            "period_basis": row.get("period_basis", ""),
            "period": row.get("period", ""),
            "value": row.get("value", ""),
            "source_table": row.get("source_table", ""),
            "source_filter": row.get("source_filter", ""),
            "calculation_rule": row.get("calculation_rule", ""),
        })
    return pd.DataFrame(rows)


def render_specialized(
    *,
    report_id: str,
    run_root: Path,
    metrics_dir: Path,
    out_dir: Path,
    as_of_date: str,
    browser_bin: str | Path | None = None,
    require_pdf: bool = True,
) -> dict[str, Path]:
    spec = next(s for s in REPORT_SPECS if s.report_id == report_id)
    result = build_specialized_view(
        spec.view_key,
        run_root=run_root,
        metrics_dir=metrics_dir,
        scope=spec.scope,
    )
    selected, period_label = _period_selection(result.frame, spec.period_policy, as_of_date)
    if selected.empty:
        raise ValueError(f"specialized report has no governed rows for selected period: {report_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    css = Path(__file__).with_name("report.css").read_text(encoding="utf-8")
    traces = []
    currency_sections = []

    currencies = sorted(selected["Currency"].astype(str).dropna().unique())
    if spec.currency_policy != "separate_native":
        raise ValueError(f"unsupported specialized currency policy: {spec.currency_policy}")

    for currency in currencies:
        current = selected.loc[selected["Currency"].astype(str).eq(currency)].copy()
        if current.empty:
            continue
        blocks = []
        if "summary" in spec.section_plan:
            blocks.append(_summary_html(current, currency, result.dimension))
        if "pie" in spec.section_plan:
            if current["period_basis"].astype(str).nunique() != 1 or current["period"].astype(str).nunique() != 1:
                raise ValueError(f"pie report requires one governed period: {report_id} {currency}")
            denominator = float(pd.to_numeric(current["value"], errors="coerce").sum())
            pie_spec = PieSpec(
                chart_id=f"{report_id}_{currency}_{current['period'].iloc[0]}",
                source_metric=result.metric_id,
                measure="value",
                slice_dimension=result.dimension,
                currency=currency,
                scope=spec.scope,
                period_basis=str(current["period_basis"].iloc[0]),
                period=str(current["period"].iloc[0]),
                title=f"{spec.title} · {period_label} · {currency}",
                subtitle="Población gobernada; moneda nativa separada.",
                max_slices=12,
            )
            svg, pie_trace = render_pie_svg(pie_spec, current, denominator)
            pie_trace.insert(0, "report_id", report_id)
            traces.append(pie_trace)
            blocks.append(f'<div class="chart">{svg}</div>')
        elif "bars" in spec.section_plan:
            traces.append(_plain_trace(report_id, result, current))
            blocks.append(_bars_html(current, result.dimension, currency, chronological=True))
        elif "comparison" in spec.section_plan:
            traces.append(_plain_trace(report_id, result, current))
            blocks.append(_comparison_html(current, result.dimension, currency))
        else:
            traces.append(_plain_trace(report_id, result, current))

        if "table" in spec.section_plan:
            blocks.append(_table(current, result.table_columns))

        currency_sections.append(
            f'''<section class="currency-section">
              <div class="currency-head"><h2>{html.escape(currency)}</h2><span>{html.escape(period_label)}</span></div>
              {''.join(blocks)}
            </section>'''
        )

    trace = pd.concat(traces, ignore_index=True, sort=False) if traces else pd.DataFrame()
    trace_path = out_dir / "internal_trace.csv"
    trace.to_csv(trace_path, index=False)

    cutoff_label = pd.Timestamp(as_of_date).strftime("%d/%m/%Y")
    body = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(spec.title)}</title><style>{css}</style></head><body><main>
      <header><p class="eyebrow">INFORME ESPECIALIZADO · {html.escape(spec.family.upper())}</p><h1>{html.escape(spec.title)}</h1><p class="subtitle">Corte contable: {html.escape(cutoff_label)} · Scope: {html.escape(_scope_label(spec.scope))}</p></header>
      <section class="question"><h2>{html.escape(spec.question)}</h2><p>{html.escape(spec.establishes)}</p></section>
      {''.join(currency_sections)}
      <section class="method"><h2>Alcance y método</h2><p><strong>Qué establece:</strong> {html.escape(spec.establishes)}</p><p><strong>Qué no establece:</strong> {html.escape(spec.caveat)}</p><p>Fuente: vista profesional gobernada del backend. El documento no reconstruye movimientos ni reclasifica transacciones.</p></section>
    </main></body></html>'''
    html_path = out_dir / "report.html"
    html_path.write_text(body, encoding="utf-8")
    outputs = {"html": html_path, "trace": trace_path}
    if require_pdf:
        outputs["pdf"] = render_pdf(html_path, out_dir / "report.pdf", browser_bin=browser_bin)
    return outputs
