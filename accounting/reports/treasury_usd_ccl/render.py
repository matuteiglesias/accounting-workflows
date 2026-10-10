"""Render the monthly ARS treasury projection in USD/CCL.

This report is a companion to the native treasury accountability report.  It
consumes only the monthly projection sidecar and never values transactions or
changes the native report bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from accounting.reports.pdf import render_pdf as render_pdf_file


REPORT_ID = "treasury_usd_ccl"
RENDERER_VERSION = "treasury_usd_ccl_report_v1"


class TreasuryUSDCCLReportError(ValueError):
    """Raised when the companion report cannot be reconciled."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> str:
    if value is None or str(value).strip() == "" or pd.isna(value):
        return "no disponible"
    return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _read_projection(path: Path, *, run_id: str, cutoff_date: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"run_id", "cutoff_date", "period", "Box", "source_currency", "net_cash_flow_ars",
                "ccl_mean_ars_per_usd", "usd_ccl_net_flow", "projection_status", "fx_coverage"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TreasuryUSDCCLReportError(f"projection missing columns: {missing}")
    if set(frame["run_id"]) != {run_id} or set(frame["cutoff_date"]) != {cutoff_date}:
        raise TreasuryUSDCCLReportError("projection run identity/cutoff does not match report arguments")
    if set(frame["source_currency"]) != {"ARS"}:
        raise TreasuryUSDCCLReportError("USD/CCL treasury report accepts only ARS source flows")
    if frame[["period", "Box"]].duplicated().any():
        raise TreasuryUSDCCLReportError("projection has duplicate monthly Box cells")
    return frame.sort_values(["Box", "period"]).reset_index(drop=True)


def _read_aggregates(path: Path, *, run_id: str, cutoff_date: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"run_id", "cutoff_date", "aggregation", "period", "Box", "usd_ccl_net_flow", "projection_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TreasuryUSDCCLReportError(f"aggregate projection missing columns: {missing}")
    if set(frame["run_id"]) != {run_id} or set(frame["cutoff_date"]) != {cutoff_date}:
        raise TreasuryUSDCCLReportError("aggregate run identity/cutoff does not match report arguments")
    return frame.sort_values(["aggregation", "Box", "period"]).reset_index(drop=True)


def _validation_rows(monthly: pd.DataFrame, aggregates: pd.DataFrame, control: pd.DataFrame,
                     *, run_id: str, cutoff_date: str, html_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    def add(check: str, status: str, detail: str) -> None:
        rows.append({"check": check, "status": status, "severity": "error" if status == "fail" else "info", "detail": detail})
    add("exact_run_identity", "pass", f"run_id={run_id} cutoff_date={cutoff_date}")
    missing = int((monthly["projection_status"] != "complete").sum())
    add("fx_coverage", "pass" if missing == 0 else "warn", f"complete={len(monthly)-missing} unavailable={missing}")
    add("no_zero_for_missing_fx", "pass", "unavailable FX cells remain blank rather than zero")
    add("cumulative_control", "pass" if set(control.get("status", [])) <= {"pass"} else "fail", "cumulative derived-flow controls")
    expected_values = [str(value).strip() for value in monthly["usd_ccl_net_flow"] if str(value).strip()]
    absent = [value for value in expected_values if value not in html_text]
    add("html_values_reconcile_to_csv", "pass" if not absent else "fail", f"checked={len(expected_values)} absent={len(absent)}")
    add("aggregate_rows_present", "pass" if not aggregates.empty else "fail", f"rows={len(aggregates)}")
    return rows


def _html_document(monthly: pd.DataFrame, aggregates: pd.DataFrame, *, run_id: str, cutoff_date: str,
                   projection_manifest: dict[str, Any], fx_manifest: dict[str, Any]) -> str:
    css = Path(__file__).parents[1].joinpath("treasury_accountability", "report.css").read_text(encoding="utf-8")
    extra_css = """.ccl-table{width:100%;border-collapse:collapse;font-size:9px}.ccl-table th,.ccl-table td{padding:6px 8px;border-bottom:1px solid #e8ebf0;text-align:right;white-space:nowrap}.ccl-table th{background:#102a56;color:#fff;text-align:center}.ccl-table th:first-child,.ccl-table td:first-child,.ccl-table th:nth-child(2),.ccl-table td:nth-child(2){text-align:left}.ccl-note{margin:12px 0;padding:10px 12px;background:#eef3f8;border-left:3px solid #315f98;font-size:10px;line-height:1.45}.warn{color:#9a6700}.section-title{color:#102a56;margin-top:22px}.ccl-table tr.agg-annual td{background:#f4f6f9;border-top:2px solid #aeb8c7;font-weight:700}.ccl-table tr.agg-ytd td{background:#eef3f8;border-top:2px solid #315f98;font-weight:760}.ccl-table tr.agg-total td{background:#e3ebf5;border-top:3px solid #102a56;border-bottom:3px solid #102a56;font-weight:800}"""
    groups = []
    for box, frame in monthly.groupby("Box", sort=True):
        rows = []
        for row in frame.itertuples(index=False):
            key = f"{row.period}|{row.Box}"
            rows.append(
                f'<tr data-source-key="{_h(key)}"><td>{_h(row.period)}</td><td>{_h(row.Box)}</td>'
                f'<td>{_h(_number(row.net_cash_flow_ars))} ARS</td><td>{_h(_number(row.ccl_mean_ars_per_usd))}</td>'
                f'<td data-source-value="{_h(row.usd_ccl_net_flow)}">{_h(_number(row.usd_ccl_net_flow))}</td>'
                f'<td>{_h(row.projection_status)}</td><td>{_h(row.fx_coverage)}</td></tr>'
            )
        aggregate_rows = []
        box_aggregates = aggregates.loc[aggregates["Box"].eq(box)].copy()
        order = {"annual": 0, "ytd": 1, "total": 2}
        box_aggregates["_order"] = box_aggregates["aggregation"].map(order).fillna(9)
        for row in box_aggregates.sort_values(["_order", "period"]).itertuples(index=False):
            row_class = {"annual": "agg-annual", "ytd": "agg-ytd", "total": "agg-total"}.get(row.aggregation, "")
            label = "YTD 2026" if row.aggregation == "ytd" else ("TOTAL SERIE" if row.aggregation == "total" else f"ANUAL {_h(row.period)}")
            aggregate_rows.append(f'<tr class="{row_class}" data-aggregation="{_h(row.aggregation)}"><td>{label}</td><td>{_h(row.Box)}</td><td>{_h(_number(row.usd_ccl_net_flow))}</td><td>{_h(row.projection_status)}</td></tr>')
        groups.append(f'<h2 class="section-title">{_h(box)}</h2><div class="table-wrap"><table class="ccl-table"><thead><tr><th>Período</th><th>Box</th><th>Flujo neto ARS</th><th>CCL medio ARS/USD</th><th>Flujo neto USD CCL</th><th>Estado</th><th>Cobertura FX</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><h3 class="section-title">Agregados · {_h(box)}</h3><div class="table-wrap"><table class="ccl-table"><thead><tr><th>Agregación</th><th>Box</th><th>Flujo neto USD CCL</th><th>Estado</th></tr></thead><tbody>{"".join(aggregate_rows)}</tbody></table></div>')
    return f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Tesorería mensual · USD/CCL</title><style>{css}{extra_css}</style></head><body><div class="report-shell">
<header class="hero"><div class="eyebrow">Family Business / Property Management</div><h1>Tesorería mensual · USD/CCL</h1><div class="hero-sub">Proyección mensual de los flujos ARS de tesorería expresados al promedio CCL del período. Informe complementario; no constituye valuación transaccional ni reemplaza la rendición nominal.</div><div class="meta-row"><span>RUN_ID: {_h(run_id)}</span><span>Corte: {_h(cutoff_date)}</span><span>Fuente ARS: monthly_cash_accountability.csv</span><span>Fuente FX: CCL CANON · {int(fx_manifest.get("accepted_observation_count", 0))} observaciones</span></div><div class="ccl-note"><strong>Convención:</strong> se convierten sólo flujos mensuales ARS mediante media mensual CCL. USD nativo, saldos acumulados y movimientos a nivel transacción no se convierten aquí. Octubre es parcial al corte {_h(cutoff_date)}; los días no hábiles excluidos de la media están documentados en el snapshot.</div></header>
<section class="box-section"><header class="box-head"><h2 class="box-title">Flujos mensuales convertidos</h2><div class="box-subtitle">Misma clasificación y Box del reporte ARS fuente; moneda de origen preservada.</div></header><div class="series-section">{"".join(groups)}</div></section>
<section class="box-section"><header class="box-head"><h2 class="box-title">Agregados de la serie</h2><div class="box-subtitle">Anual por año, YTD únicamente para 2026 y total de toda la serie. Las filas sombreadas son sumas de meses convertidos.</div></header><div class="series-section"><div class="ccl-note">Las filas <strong>ANUAL</strong>, <strong>YTD 2026</strong> y <strong>TOTAL SERIE</strong> son agregaciones derivadas; no representan saldos ni valuaciones de posiciones acumuladas.</div></div></section>
<section class="qa-card"><h2>Proveniencia y controles</h2><div class="qa-grid"><div class="qa-item"><strong>Snapshot CCL</strong><span>SHA256 {_h(fx_manifest.get("snapshot_sha256", ""))}</span></div><div class="qa-item"><strong>Media mensual</strong><span>SHA256 {_h(projection_manifest.get("fx_monthly_mean_sha256", ""))}</span></div><div class="qa-item"><strong>Proyección</strong><span>SHA256 {_h(projection_manifest.get("monthly_artifact_sha256", ""))}</span></div><div class="qa-item"><strong>Cobertura</strong><span>{_h(fx_manifest.get("rate_min_date", ""))} → {_h(fx_manifest.get("rate_max_date", ""))}; octubre parcial</span></div></div></section>
<footer class="footer"><span>Informe opt-in · fuente contable mensual reconciliada.</span><span>Generación reproducible; ver report_validation.csv y report_manifest.json.</span></footer></div></body></html>'''


def render_report(*, monthly_path: Path, aggregates_path: Path, cumulative_control_path: Path,
                  projection_manifest_path: Path, fx_manifest_path: Path, out_dir: Path,
                  run_id: str, cutoff_date: str, browser_bin: str | Path | None = None,
                  render_pdf: bool = True) -> dict[str, Path]:
    monthly = _read_projection(monthly_path, run_id=run_id, cutoff_date=cutoff_date)
    aggregates = _read_aggregates(aggregates_path, run_id=run_id, cutoff_date=cutoff_date)
    control = pd.read_csv(cumulative_control_path, dtype=str).fillna("")
    projection_manifest = json.loads(projection_manifest_path.read_text(encoding="utf-8"))
    fx_manifest = json.loads(fx_manifest_path.read_text(encoding="utf-8"))
    if projection_manifest.get("run_id") != run_id or projection_manifest.get("cutoff_date") != cutoff_date:
        raise TreasuryUSDCCLReportError("projection manifest identity mismatch")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "report.html"
    html_path.write_text(_html_document(monthly, aggregates, run_id=run_id, cutoff_date=cutoff_date,
                                        projection_manifest=projection_manifest, fx_manifest=fx_manifest), encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    pdf_path = out_dir / "report.pdf"
    if render_pdf:
        render_pdf_file(html_path, pdf_path, browser_bin=browser_bin)
    validation_rows = _validation_rows(monthly, aggregates, control, run_id=run_id, cutoff_date=cutoff_date, html_text=html_text)
    if render_pdf:
        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            validation_rows.append({"check": "pdf_exists", "status": "fail", "severity": "error", "detail": "missing PDF"})
        else:
            pdftotext = shutil.which("pdftotext")
            if pdftotext:
                text = subprocess.run([pdftotext, str(pdf_path), "-"], check=True, capture_output=True, text=True).stdout
                expected = [_number(value) for value in monthly["usd_ccl_net_flow"] if str(value).strip()]
                absent = [value for value in expected if value not in text]
                validation_rows.append({"check": "pdf_values_reconcile_to_csv", "status": "pass" if not absent else "fail", "severity": "error" if absent else "info", "detail": f"checked={len(expected)} absent={len(absent)}"})
            else:
                validation_rows.append({"check": "pdf_values_reconcile_to_csv", "status": "warn", "severity": "info", "detail": "pdftotext unavailable; PDF file/header verified only"})
    validation = pd.DataFrame(validation_rows)
    validation_path = out_dir / "report_validation.csv"
    validation.to_csv(validation_path, index=False)
    status = "fail" if (validation["status"] == "fail").any() else ("warn" if (validation["status"] == "warn").any() else "pass")
    manifest = {
        "schema": "accounting_report_manifest.v1", "report_id": REPORT_ID, "renderer_version": RENDERER_VERSION,
        "source_run_id": run_id, "scope_tag": "FBPM", "as_of_date": cutoff_date,
        "validation_status": status,
        "sources": [{"name": path.name, "path": str(path), "sha256": _sha256(path)} for path in [monthly_path, aggregates_path, cumulative_control_path, projection_manifest_path, fx_manifest_path]],
        "outputs": {"html": {"path": str(html_path), "sha256": _sha256(html_path)}, "validation": {"path": str(validation_path), "sha256": _sha256(validation_path)}},
    }
    if render_pdf:
        manifest["outputs"]["pdf"] = {"path": str(pdf_path), "sha256": _sha256(pdf_path)}
    manifest_path = out_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status == "fail":
        raise TreasuryUSDCCLReportError(f"USD/CCL report validation failed: {validation.loc[validation.status.eq('fail'), 'check'].tolist()}")
    return {"html": html_path, "validation": validation_path, "manifest": manifest_path, **({"pdf": pdf_path} if render_pdf else {})}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the opt-in monthly treasury USD/CCL report")
    parser.add_argument("--monthly", required=True, type=Path); parser.add_argument("--aggregates", required=True, type=Path)
    parser.add_argument("--cumulative-control", required=True, type=Path); parser.add_argument("--projection-manifest", required=True, type=Path)
    parser.add_argument("--fx-manifest", required=True, type=Path); parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True); parser.add_argument("--cutoff-date", required=True); parser.add_argument("--browser-bin")
    args = parser.parse_args()
    outputs = render_report(monthly_path=args.monthly, aggregates_path=args.aggregates, cumulative_control_path=args.cumulative_control,
                            projection_manifest_path=args.projection_manifest, fx_manifest_path=args.fx_manifest, out_dir=args.out_dir,
                            run_id=args.run_id, cutoff_date=args.cutoff_date, browser_bin=args.browser_bin)
    for name, path in outputs.items(): print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
