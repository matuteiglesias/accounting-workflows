from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from accounting.reports.charts import (
    PieSpec,
    professional_distribution_view,
    professional_support_view,
    professional_tax_service_payment_view,
    render_pie_svg,
)
from accounting.reports.pdf import render_pdf
from accounting.reports.specialized.spec import REPORT_SPECS, SpecializedReportSpec


def _money(value: float, currency: str) -> str:
    return f"{currency} {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _table(frame: pd.DataFrame, label: str) -> str:
    if frame.empty:
        return '<p class="muted">No disponible para este corte.</p>'
    cols = [c for c in frame.columns if c in {"period", "funding_actor", "recipient", "obligation_category", "value", "Currency"}]
    view = frame[cols].copy()
    view = view.rename(columns={"period": "Período", "funding_actor": "Actor", "recipient": "Receptor", "obligation_category": "Categoría", "value": "Importe", "Currency": "Moneda"})
    if "Importe" in view:
        view["Importe"] = view.apply(lambda r: _money(float(r["Importe"]), str(r.get("Moneda", ""))), axis=1)
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in view.columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>' for row in view.itertuples(index=False, name=None)) + '</tbody></table>'


def render_specialized(*, report_id: str, run_root: Path, metrics_dir: Path, out_dir: Path, browser_bin: str | Path | None = None, require_pdf: bool = True) -> dict[str, Path]:
    spec = next(s for s in REPORT_SPECS if s.report_id == report_id)
    audit = pd.read_csv(run_root / "classification_audit.csv")
    support = pd.read_csv(run_root / "monthly_stakeholder_support.csv")
    metrics = pd.read_csv(metrics_dir / "annual_balance_dashboard_metrics.csv")
    if spec.family == "distributions":
        view = professional_distribution_view(audit, metrics)
        dim = "recipient"; metric = "DIST.BY_RECIPIENT"; heading = "Receptor"
    elif spec.family == "support":
        view = professional_support_view(support)
        dim = "funding_actor"; metric = "SUPPORT.BY_ACTOR"; heading = "Actor"
    else:
        view = professional_tax_service_payment_view(audit)
        category = "taxes" if spec.family == "costs" and "tax" in report_id else "services"
        # The payment view is transaction-governed; specialize only its category.
        rows = audit.loc[audit["semantic_subbucket"].astype(str).eq(category)].copy()
        view = professional_tax_service_payment_view(rows)
        dim = "funding_actor"; metric = "TAX_SERVICE.PAYMENTS.BY_ACTOR"; heading = "Actor"
    if view.empty:
        raise ValueError(f"specialized report has no governed rows: {report_id}")
    year = str(view["period"].astype(str).max())
    current = view.loc[view["period"].astype(str).eq(year)].copy()
    currency = str(current["Currency"].iloc[0])
    current = current.loc[current["Currency"].astype(str).eq(currency)].copy()
    current["scope"] = "FBPM"; current["period_basis"] = "annual"; current["period"] = year
    denominator = float(current["value"].sum())
    chart_spec = PieSpec(report_id + "_annual", metric, "value", dim, currency, "FBPM", "annual", year, spec.title + " · " + year, "Período anual gobernado", max_slices=12)
    svg, trace = render_pie_svg(chart_spec, current, denominator)
    out_dir.mkdir(parents=True, exist_ok=True)
    css_path = out_dir / "report.css"
    css_path.write_text((Path(__file__).with_name("report.css")).read_text(encoding="utf-8"), encoding="utf-8")
    trace.to_csv(out_dir / "internal_trace.csv", index=False)
    rows_html = _table(current, heading)
    year_label = year + (" YTD · corte 31/08/2026" if year == "2026" else "")
    css = css_path.read_text(encoding="utf-8")
    body = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{html.escape(spec.title)}</title><style>{css}</style></head><body><main><header><p class="eyebrow">INFORME ESPECIALIZADO · {html.escape(spec.family.upper())}</p><h1>{html.escape(spec.title)}</h1><p class="subtitle">Corte contable: 31/08/2026 · {html.escape(year_label)}</p></header><section><h2>{html.escape(spec.question)}</h2><div class="metric"><span>Total reconocido</span><strong>{_money(denominator, currency)}</strong></div><div class="chart">{svg}</div>{rows_html}</section><section class="method"><h2>Alcance y método</h2><p>{html.escape(spec.caveat)}</p><p>Fuente: autoridad profesional gobernada; movimientos y saldos no se reconstruyen desde este documento.</p></section></main></body></html>'''
    html_path = out_dir / "report.html"; html_path.write_text(body, encoding="utf-8")
    outputs = {"html": html_path, "css": css_path, "trace": out_dir / "internal_trace.csv"}
    if require_pdf:
        outputs["pdf"] = render_pdf(html_path, out_dir / "report.pdf", browser_bin=browser_bin)
    return outputs
