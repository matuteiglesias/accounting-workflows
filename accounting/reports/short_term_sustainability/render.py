from __future__ import annotations

"""TS09 private report: 6 historical months; current as forward month 1."""
from html import escape
import hashlib
import json
from pathlib import Path

import pandas as pd
from accounting.reports.pdf import render_pdf
from accounting.reports.time_series import (
    TIMESERIES_CSS, TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _money(value, unit: str) -> str:
    if value is None or pd.isna(value):
        return "Sin plan"
    value = 0.0 if abs(float(value)) < 0.005 else float(value)
    return (f"{unit} {value:,.0f}").replace(",", "X").replace(".", ",").replace("X", ".")


def _history_forward(as_of: pd.Timestamp) -> tuple[list[str], list[str]]:
    now = as_of.to_period("M")
    return ([str(now - i) for i in range(6, 0, -1)],
            [str(now + i) for i in range(6)])


def _build_rows(observed: pd.DataFrame, policy: pd.DataFrame,
                box: str, unit: str, history: list[str], future: list[str]) -> list[dict]:
    rows = []
    for period in history + future:
        if period in history:
            d = observed.loc[
                observed.period.eq(period) & observed.Box.eq(box) &
                observed.Currency.eq(unit)
            ]
            if len(d) != 1 or not bool(d.iloc[0].materialized_month) or d.iloc[0].temporal_status != "observed":
                raise ValueError(f"no certified historical point: {box}/{unit}/{period}")
            point, phase, coverage, src = d.iloc[0], "observed", "observado", "monthly_policy_inputs.csv"
            available = True
        else:
            d = policy.loc[
                policy.period.eq(period) & policy.Box.eq(box) &
                policy.Currency.eq(unit)
            ]
            if len(d) != 1:
                raise ValueError(f"missing policy-input month: {box}/{unit}/{period}")
            point = d.iloc[0]
            available = bool(point.materialized_month)
            current = period == future[0]
            if current and (not available or point.temporal_status != "partial_month"):
                raise ValueError("current month must be partial and recorded")
            if not current and available and point.temporal_status != "projected":
                raise ValueError("unexpected projected status")
            phase = "current" if current else "projected"
            coverage = "actual parcial" if current else ("plan registrado" if available else "sin plan")
            src = "monthly_policy_inputs.csv"
        rows.append({
            "period": period, "Box": box, "Currency": unit, "phase": phase,
            "coverage": coverage, "source_table": src,
            "cash_in": point.total_cash_in if available else None,
            "cash_out": point.total_cash_out if available else None,
            "net_cash": point.net_cash_flow if available else None,
            "control": point.closing_control if available else None,
            "opex_due": (
                (point.planned_cash_operating_cost_due_this_month
                 if period in future[1:] else point.actual_or_projected_operating_costs_out)
                if available else None
            ),
        })
    return rows


def _reserve_from_next_month(rows: list[dict], future: list[str]) -> tuple[float | None, list[str]]:
    # November's already-collected rent is not new liquidity available after
    # the current cutoff. Evaluate December onward from an unknown opening cash.
    running, trough, valid_count = 0.0, 0.0, 0
    missing = []
    for row in rows:
        if row["period"] not in future[1:]:
            continue
        if row["coverage"] == "sin plan":
            missing.append(row["period"])
            continue
        if missing:
            raise ValueError("forecast resumes after first unplanned month")
        running += float(row["net_cash"])
        trough = min(trough, running)
        valid_count += 1
    return (round(-trough, 2) if valid_count else None), missing


def _chart(
    *, box: str, unit: str, chart: str, title: str,
    months: list[str], rows: list[dict], fields: tuple[tuple[str, str], ...],
    mark: str, note: str,
) -> tuple[str, pd.DataFrame]:
    values = []
    for row in rows:
        if row["coverage"] == "sin plan":
            continue
        for field, _ in fields:
            if not pd.isna(row[field]):
                values.append({
                    "period": row["period"], "series_id": field,
                    "value": float(row[field]), "unit": unit, "scope": box,
                    "phase": row["phase"], "source_table": row["source_table"],
                })
    slug = box.lower().replace(" ", "_")
    spec = TimeSeriesSpec(
        chart_id=f"ts09_{chart}_{slug}_{unit.lower()}",
        title=title, frequency="M", mark=mark, unit=unit, scope=box,
        series_ids=tuple(key for key, _ in fields),
        series_labels=tuple(label for _, label in fields),
        period_domain=tuple(months), max_points=12,
        subtitle="Mayo–octubre histórico · noviembre actual · diciembre–abril según planes",
        note=note,
    )
    svg, trace = render_timeseries_svg(spec, pd.DataFrame(values), height=260)
    return timeseries_figure(spec, svg), trace


STYLES = """
:root{--ink:#172033;--navy:#102a56;--muted:#667085;--line:#d7dde7}
*{box-sizing:border-box}
body{margin:0;background:#edf1f5;color:var(--ink);font:13px/1.4 Inter,system-ui,Arial,sans-serif}
main{max-width:1120px;margin:auto;background:#fff;padding:30px 38px}
h1{color:var(--navy);font-size:29px;margin:8px 0}
h2{color:var(--navy);font-size:19px}
.subtitle{color:var(--muted);font-size:11px}
.warning{background:#fff5e5;border:1px solid #e9d2a1;border-radius:7px;padding:12px;margin:12px 0}
.scope{margin:24px 0 40px;break-before:page;page-break-before:always}
.scope:first-of-type{break-before:auto;page-break-before:auto}
table{width:100%;border-collapse:collapse;font-size:10px;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid var(--line);padding:6px}
th{background:#f2f5f9;color:var(--navy);text-align:left}
td.num{text-align:right}
tr.current{background:#f0f5fb}tr.projected{background:#f7f8fc}
tr.absent{background:#fff5e5}
.ts-figure{margin:9px 0}
footer{padding-top:10px;margin-top:15px;border-top:1px solid var(--line);color:var(--muted);font-size:9px}
@media print{@page{size:A4 landscape;margin:11mm}
body{background:#fff}main{padding:0;max-width:none}
.scope{break-before:page;page-break-before:always}
.ts-figure .governed-timeseries{max-height:142px}
.ts-figure{break-inside:avoid}table{font-size:8px}}
"""


def render_report(
    *, run_root: Path, policy_root: Path, out_dir: Path,
    as_of_date: str, require_pdf: bool = True, browser_bin: str | None = None,
) -> dict[str, Path]:
    base, policy, out = map(Path, (run_root, policy_root, out_dir))
    as_of = pd.Timestamp(as_of_date)
    historical, forward = _history_forward(as_of)
    months = historical + forward
    manifest = json.loads((policy / "policy_inputs_manifest.json").read_text())
    if manifest.get("schema") != "accounting.policy_inputs.v1":
        raise ValueError("unknown policy inputs schema")
    if manifest.get("base_run_id") != base.name or manifest.get("base_all_status_sha256") != _sha(base / "ledger_canonical_all_status.csv"):
        raise ValueError("policy inputs are not pinned to this exact run")
    if manifest.get("as_of_date") != str(as_of.date()) or str(pd.Timestamp(manifest["through_date"]).to_period("M")) != forward[-1]:
        raise ValueError("policy inputs mismatch report's six-month window")
    features = pd.read_csv(policy / "monthly_policy_inputs.csv")
    observed = features  # same isolated, pinned materialization; never mix accounting runs
    required = {"period", "Box", "Currency", "temporal_status", "materialized_month",
                "total_cash_in", "total_cash_out", "net_cash_flow", "closing_control",
                "planned_cash_operating_cost_due_this_month", "actual_or_projected_operating_costs_out"}
    if required - set(features):
        raise ValueError(f"missing policy fields: {sorted(required - set(features))}")
    if features.duplicated(["period", "Box", "Currency"]).any():
        raise ValueError("duplicate month/Box/currency")
    scopes = [("Property Management", "ARS"), ("Family Business", "ARS")]
    if (observed.Box.eq("Property Management") & observed.Currency.eq("USD") & observed.period.isin(historical) & observed.materialized_month).sum() == 6:
        scopes.append(("Property Management", "USD"))
    panels, all_rows, traces, gap_info = [], [], [], []
    for box, unit in scopes:
        rows = _build_rows(observed, features, box, unit, historical, forward)
        stress, missing = _reserve_from_next_month(rows, forward)
        gap_info.append({"Box": box, "Currency": unit, "minimum_dec_through_last_planned": stress,
                         "missing_plan_months": missing, "complete_6m": not missing})
        charts = []
        for chart, title, fields, mark, note in (
            ("flows", "¿Cuánto entra y sale físicamente en cada mes?",
             (("cash_in", "Entradas"), ("cash_out", "Salidas")), "bars",
             "Noviembre es parcial; desde diciembre los importes dependen de planes; abril no tiene planes registrados."),
            ("net", "¿Qué meses aportan o consumen flujo neto?",
             (("net_cash", "Flujo neto"),), "line",
             "No equivale a caja bancarizada. La ausencia de un mes no significa un flujo cero."),
            ("control", "¿Cómo cambia el control contable acumulado?",
             (("control", "Control de origen cero"),), "line",
             "Control contable desde origen cero, no saldo de efectivo validado."),
        ):
            html, trace = _chart(
                box=box, unit=unit, chart=chart, title=title,
                months=months, rows=rows, fields=fields, mark=mark, note=note,
            )
            charts.append(html)
            traces.append(trace)
        tab = []
        for row in rows:
            klass = ("current" if row["coverage"] == "actual parcial"
                     else "projected" if row["coverage"] == "plan registrado"
                     else "absent" if row["coverage"] == "sin plan" else "")
            tab.append(
                f'<tr class="{klass}"><td>{escape(row["period"])}</td>'
                f'<td>{escape(row["coverage"])}</td>'
                + "".join(
                    f'<td class="num">{escape(_money(row[field], unit))}</td>'
                    for field in ("cash_in", "cash_out", "opex_due", "net_cash", "control")
                )
                + "</tr>"
            )
            all_rows.append(row)
        headline = (
            f"Mínimo de liquidez desde diciembre hasta último mes planificado continuo: "
            f"<strong>{escape(_money(stress, unit))}</strong> (sujeto a supuestos de cobros y pagos). "
            f"Meses sin plan: <strong>{escape(', '.join(missing) if missing else 'ninguno')}</strong>."
        )
        panels.append(
            f'<section class="scope"><h2>{escape(box)} · {escape(unit)}</h2>'
            f'<p class="subtitle">{headline}</p>' + "".join(charts) +
            '<table><thead><tr><th>Mes</th><th>Estado</th><th>Entradas</th><th>Salidas</th>'
            '<th>OPEX caja</th><th>Neto</th><th>Control contable</th></tr></thead><tbody>'
            + "".join(tab) + '</tbody></table></section>'
        )
    page = (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Continuidad financiera · 6+6 meses</title><style>' +
        STYLES + TIMESERIES_CSS + '</style></head><body><main>'
        '<header><div class="subtitle">INFORME PRIVADO · ESCENARIO BASE</div>'
        '<h1>Continuidad financiera: 6 meses atrás + 6 meses hacia adelante</h1>'
        f'<p>Histórico: <strong>{historical[0]}–{historical[-1]}</strong>. '
        f'<strong>{forward[0]}</strong> es el primer mes prospectivo (actual/parcial). '
        f'Horizonte: <strong>{forward[-1]}</strong>. Corte: {escape(str(as_of.date()))}.</p>'
        '<div class="warning"><strong>Cobertura limitada:</strong> noviembre incluye solo '
        'transacciones registradas al corte, no un pronóstico del mes entero. '
        'Los planes vigentes terminan en marzo; abril carece de cobertura. '
        '<strong>No puede certificarse sostenibilidad de seis meses.</strong> '
        'Saldos bancarios reales y facturas vencidas todavía no están disponibles.</div></header>'
        + "".join(panels) +
        '<section class="warning"><h2>Facturas vencidas pendientes</h2>'
        '<p>Sin cartera de facturas vencidas validada: no las computamos ni suponemos '
        'que su importe sea cero. Su incorporación futura será otra capa del reporte.</p>'
        '<p>Los importes de reserva modelan déficits mensuales del escenario de planes '
        'desde diciembre, con saldo incremental inicial cero; no verifican que existan '
        'los fondos, ni consideran el orden de pagos dentro de cada mes.</p></section>'
        f'<footer>Run exacto: {escape(base.name)} · report privado, '
        'no publicado en Accounting Viewer.</footer></main></body></html>'
    )
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "report.html"
    trace_path = out / "time_series_trace.csv"
    table_path = out / "monthly_sustainability_table.csv"
    info_path = out / "report_manifest.json"
    html_path.write_text(page, encoding="utf-8")
    pd.concat(traces, ignore_index=True).to_csv(trace_path, index=False)
    pd.DataFrame(all_rows).to_csv(table_path, index=False)
    info_path.write_text(json.dumps({
        "schema": "accounting.short_term_sustainability.v1",
        "source_run_id": base.name, "as_of_date": str(as_of.date()),
        "source_policy_sha256": _sha(policy / "monthly_policy_inputs.csv"),
        "historical_months": historical, "forward_months": forward,
        "first_forward_month_partial": True,
        "verified_opening_cash": False, "unpaid_bills_supported": False,
        "financial_sustainability_confirmed": False,
        "gap_checks": gap_info,
    }, indent=2) + "\n")
    results = {"html": html_path, "trace": trace_path, "table": table_path, "manifest": info_path}
    if require_pdf:
        results["pdf"] = render_pdf(html_path, out / "report.pdf", browser_bin=browser_bin)
    return results


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Private 6+6 month sustainability report from a pinned isolated run")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--policy-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--as-of-date", required=True)
    p.add_argument("--browser-bin", default=None)
    p.add_argument("--no-pdf", action="store_true")
    a = p.parse_args()
    results = render_report(
        run_root=a.run_root, policy_root=a.policy_root, out_dir=a.out_dir,
        as_of_date=a.as_of_date, require_pdf=not a.no_pdf, browser_bin=a.browser_bin,
    )
    for name, path in results.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
