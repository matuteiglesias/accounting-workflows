from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from accounting.reports.annual_management.spec import (
    DEBT_RELATIONS,
    FUNDING_ACTORS,
    KPI_SPECS,
    OPERATING_ROWS,
    OPERATING_USD_ROWS,
    QUALITY_METRICS,
    REPORT_META,
    SUMMARY_ROWS,
    YEARS,
)

KEY_COLS = ["metric_id", "period", "Currency", "dimension_name", "dimension_value"]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _esc(value: Any) -> str:
    return html.escape(str(value))


@dataclass(frozen=True)
class Cell:
    value: float | None
    status: str
    display: str
    source_rows: int
    source_table: str = ""


class MetricStore:
    def __init__(self, metrics: pd.DataFrame, contract: pd.DataFrame):
        self.metrics = metrics.copy()
        self.contract = contract.copy()
        for col in [
            "metric_id", "Currency", "dimension_name", "dimension_value",
            "value_status", "source_table", "run_id", "as_of_date",
        ]:
            if col not in self.metrics.columns:
                self.metrics[col] = ""
            self.metrics[col] = self.metrics[col].map(_text)
        self.metrics["period"] = self.metrics["period"].map(_text)
        self.metrics["value"] = pd.to_numeric(self.metrics["value"], errors="coerce")
        self.contract["metric_id"] = self.contract["metric_id"].map(_text)
        self.rendered: list[dict[str, Any]] = []

    def periods(self) -> list[str]:
        observed = [p for p in self.metrics["period"].unique().tolist() if p.isdigit()]
        selected = [p for p in YEARS if p in observed]
        return selected or sorted(observed)

    def as_of_date(self) -> str:
        values = [v for v in self.metrics["as_of_date"].unique().tolist() if v]
        return max(values) if values else ""

    def run_id(self) -> str:
        values = sorted(set(v for v in self.metrics["run_id"].unique().tolist() if v))
        return values[0] if len(values) == 1 else ",".join(values)

    @staticmethod
    def _format(value: float | None, status: str, fmt: str, currency: str) -> str:
        status = _text(status).lower()
        if status in {"not_applicable", "n/a"}:
            return "—"
        if value is None or pd.isna(value) or status not in {
            "available", "ok", "pass", "structural_zero",
        }:
            return "No disponible"
        if fmt == "ratio":
            pct = float(value) * 100.0 if abs(float(value)) <= 10 else float(value)
            return f"{pct:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")
        if fmt == "count":
            return f"{int(round(float(value))):,}".replace(",", ".")
        prefix = "USD " if fmt == "usd" or currency == "USD" else "$ "
        absolute = abs(float(value))
        if absolute >= 1_000_000:
            rendered = f"{float(value)/1_000_000:,.2f} M"
        elif absolute >= 1_000:
            rendered = f"{float(value)/1_000:,.1f} k"
        else:
            rendered = f"{float(value):,.2f}"
        return prefix + rendered.replace(",", "X").replace(".", ",").replace("X", ".")

    def get(
        self,
        metric_id: str,
        period: str,
        currency: str = "",
        dimension_name: str = "",
        dimension_value: str = "",
        *,
        fmt: str = "money",
        missing_policy: str = "unavailable",
        track: dict[str, str] | None = None,
    ) -> Cell:
        df = self.metrics
        mask = df["metric_id"].eq(metric_id) & df["period"].eq(str(period))
        mask &= df["Currency"].eq(currency)
        mask &= df["dimension_name"].eq(dimension_name)
        mask &= df["dimension_value"].eq(dimension_value)
        rows = df.loc[mask]
        if rows.empty and missing_policy == "zero":
            cell = Cell(
                0.0,
                "structural_zero",
                self._format(0.0, "structural_zero", fmt, currency),
                0,
                "sparse additive breakdown",
            )
        elif len(rows) != 1:
            cell = Cell(None, "unavailable", "No disponible", len(rows), "")
        else:
            row = rows.iloc[0]
            status = _text(row.get("value_status")) or "available"
            value = None if pd.isna(row.get("value")) else float(row.get("value"))
            cell = Cell(
                value,
                status,
                self._format(value, status, fmt, currency),
                1,
                _text(row.get("source_table")),
            )
        if track:
            self.rendered.append({
                **track,
                "period": str(period),
                "metric_id": metric_id,
                "derived_id": "",
                "Currency": currency,
                "dimension_name": dimension_name,
                "dimension_value": dimension_value,
                "raw_value": cell.value,
                "value_status": cell.status,
                "display_value": cell.display,
                "source_table": cell.source_table,
                "formula": "",
            })
        return cell

    def derived(
        self,
        derived_id: str,
        period: str,
        *,
        track: dict[str, str] | None = None,
    ) -> Cell:
        if derived_id == "operating_margin":
            numerator = self.get("IS.NET.OPERATING", period, "ARS")
            denominator = self.get("IS.REVENUE.OPERATING", period, "ARS")
            formula = "IS.NET.OPERATING / IS.REVENUE.OPERATING"
        elif derived_id == "opex_to_rent":
            numerator = self.get("IS.OPEX.PROPERTY", period, "ARS")
            denominator = self.get("IS.RENT.TOTAL", period, "ARS")
            formula = "IS.OPEX.PROPERTY / IS.RENT.TOTAL"
        elif derived_id == "draws_to_operating":
            numerator = self.get("DIST.DRAWS.PERSONAL", period, "ARS")
            denominator = self.get("IS.NET.OPERATING", period, "ARS")
            formula = "DIST.DRAWS.PERSONAL / IS.NET.OPERATING"
        else:
            raise KeyError(derived_id)

        if numerator.value is None or denominator.value is None:
            cell = Cell(None, "unavailable", "No disponible", 0, "report-derived")
        elif abs(float(denominator.value)) <= 1e-12:
            cell = Cell(None, "not_applicable", "—", 0, "report-derived")
        else:
            value = float(numerator.value) / float(denominator.value)
            cell = Cell(
                value,
                "available",
                self._format(value, "available", "ratio", "ARS"),
                0,
                "report-derived",
            )

        if track:
            self.rendered.append({
                **track,
                "period": str(period),
                "metric_id": "",
                "derived_id": derived_id,
                "Currency": "ARS",
                "dimension_name": "",
                "dimension_value": "",
                "raw_value": cell.value,
                "value_status": cell.status,
                "display_value": cell.display,
                "source_table": "report-derived from governed annual metrics",
                "formula": formula,
            })
        return cell


def _period_labels(periods: list[str], as_of_date: str) -> list[str]:
    labels = periods.copy()
    if periods:
        last = periods[-1]
        if as_of_date.startswith(f"{last}-06-30"):
            labels[-1] = f"{last} H1"
        elif as_of_date.startswith(last):
            labels[-1] = f"{last} YTD"
    return labels


def _rows(
    store: MetricStore,
    specs: list[dict[str, Any]],
    periods: list[str],
    page: str,
    component: str,
) -> list[dict[str, Any]]:
    output = []
    for spec in specs:
        values = []
        for period in periods:
            track = {
                "page_id": page,
                "component_id": component,
                "row_id": spec["label"],
            }
            if spec.get("derived"):
                cell = store.derived(spec["derived"], period, track=track)
            else:
                cell = store.get(
                    spec["metric_id"],
                    period,
                    spec.get("currency", ""),
                    spec.get("dimension_name", ""),
                    spec.get("dimension_value", ""),
                    fmt=spec.get("format", "money"),
                    missing_policy=spec.get("missing_policy", "unavailable"),
                    track=track,
                )
            values.append(cell.display)
        output.append({
            "label": spec["label"],
            "values": values,
            "role": spec.get("role", ""),
            "indent": spec.get("indent", 0),
        })
    return output


def _table(
    title: str,
    headers: list[str],
    rows: list[dict[str, Any]],
    *,
    compact: bool = False,
) -> str:
    cls = "metric-table compact" if compact else "metric-table"
    head = "".join(f"<th>{_esc(item)}</th>" for item in headers)
    body = []
    for row in rows:
        tr_cls = "major" if row.get("role") == "major" else ""
        values = "".join(f"<td>{_esc(value)}</td>" for value in row["values"])
        body.append(
            f'<tr class="{tr_cls}"><td class="indent-{int(row.get("indent", 0))}">'
            f'{_esc(row["label"])}</td>{values}</tr>'
        )
    return (
        f'<div class="panel"><h2>{_esc(title)}</h2><table class="{cls}">'
        f'<thead><tr><th>Concepto</th>{head}</tr></thead><tbody>{"".join(body)}'
        "</tbody></table></div>"
    )


def _page_header(meta: dict[str, Any], number: int, title: str, question: str) -> str:
    periods = " · ".join(meta["period_columns"])
    return (
        '<section class="report-page"><header class="page-header"><div>'
        f'<div class="brand">{_esc(meta["title"])}</div>'
        f'<div class="subtitle">{_esc(meta["subtitle"])}</div></div>'
        f'<div class="period-box"><strong>Estado al {_esc(meta["as_of_date"] or "No disponible")}</strong>'
        f'<span>{_esc(periods)}</span></div></header><div class="page-body">'
        f'<div class="section-title">{number}. {_esc(title)}</div>'
        f'<div class="section-question">{_esc(question)}</div>'
    )


def _page_footer(meta: dict[str, Any], number: int) -> str:
    return (
        '</div><footer class="page-footer">'
        f'<span>Fuente: accounting backend · Scope: {_esc(meta["scope"])} · '
        f'{_esc(meta["currency_note"])}</span><span>Página {number} de 6</span>'
        "</footer></section>"
    )


def _render_html(model: dict[str, Any], css: str) -> str:
    meta = model["meta"]
    periods = meta["period_columns"]
    pages: list[str] = []

    kpis = "".join(
        f'<div class="kpi"><div class="kpi-label">{_esc(item["label"])}</div>'
        f'<div class="kpi-value">{_esc(item["value"])}</div>'
        f'<div class="kpi-note">{_esc(meta["last_period_label"])}</div></div>'
        for item in model["summary"]["kpis"]
    )
    pages.append(
        _page_header(
            meta,
            1,
            "RESUMEN EJECUTIVO",
            "¿Cómo está el patrimonio y cuáles son las señales principales?",
        )
        + f'<div class="kpi-grid kpi-grid-5">{kpis}</div><div class="page-grid">'
        + f'<div class="span-8">{_table("Tabla resumen", periods, model["summary"]["rows"])}</div>'
        + '<div class="span-4"><div class="panel"><h2>Ratios clave</h2><div class="chart-slot"></div></div>'
        + '<div class="panel compact-note"><h2>Lectura</h2><p>Resumen de generación operativa, aplicación del resultado y posición. ARS y USD se muestran por separado.</p></div></div></div>'
        + _page_footer(meta, 1)
    )

    portfolio_rows = [
        {"label": row["label"], "values": [row["value"], row["share"]]}
        for row in model["operations"]["portfolio"]
    ]
    pages.append(
        _page_header(
            meta,
            2,
            "OPERACIÓN Y PORTFOLIO",
            "¿Qué produce el patrimonio y qué cuesta mantener esa producción?",
        )
        + '<div class="page-grid">'
        + f'<div class="span-7">{_table("Estado operativo · ARS nominales", periods, model["operations"]["rows"])}</div>'
        + '<div class="span-5"><div class="panel"><h2>Composición de rentas</h2><div class="chart-slot tall"></div></div>'
        + _table("Portfolio actual · " + meta["last_period_label"], ["Renta", "%"], portfolio_rows, compact=True)
        + '</div><div class="span-12">'
        + _table("Operación en USD", periods, model["operations"]["usd_rows"], compact=True)
        + '</div></div>'
        + _page_footer(meta, 2)
    )

    pages.append(
        _page_header(
            meta,
            3,
            "APLICACIÓN DEL RESULTADO",
            "¿Qué hacemos con lo que el patrimonio produce?",
        )
        + '<div class="page-grid"><div class="span-7">'
        + _table("Puente de aplicación · ARS", periods, model["application"]["bridge_ars"])
        + '<p class="caveat">Dividendos se muestran como categoría separada y no se restan una segunda vez del resultado post-retiros.</p></div>'
        + '<div class="span-5">'
        + _table("Puente de aplicación · USD", periods, model["application"]["bridge_usd"], compact=True)
        + '<div class="panel"><h2>Ratios de aplicación</h2><div class="chart-slot"></div></div></div>'
        + '<div class="span-7">'
        + _table("Funding por actor / origen", periods, model["application"]["funding"], compact=True)
        + '</div><div class="span-5">'
        + _table("Distribuciones ARS", periods, model["application"]["distributions_ars"], compact=True)
        + _table("Distribuciones USD", periods, model["application"]["distributions_usd"], compact=True)
        + '</div></div>'
        + _page_footer(meta, 3)
    )

    pages.append(
        _page_header(
            meta,
            4,
            "CAJA Y TESORERÍA",
            "¿Qué liquidez verificable tenemos y cómo la administramos?",
        )
        + '<div class="page-grid"><div class="span-7">'
        + _table("Posición de caja validada · ARS", periods, model["treasury"]["cash_ars"], compact=True)
        + _table("Posición de caja validada · USD", periods, model["treasury"]["cash_usd"], compact=True)
        + '</div><div class="span-5"><div class="panel"><h2>Estado de caja</h2>'
        + '<div class="status-big">No disponible si no existe snapshot validado</div>'
        + '<p class="caveat">La ausencia de posición validada nunca se convierte en cero ni usa caja inferida como fallback.</p></div></div>'
        + '<div class="span-7">'
        + _table("Operaciones de tesorería / FX", periods, model["treasury"]["fx"], compact=True)
        + '</div><div class="span-5"><div class="panel"><h2>Espacio de tendencia FX</h2><div class="chart-slot tall"></div></div></div></div>'
        + _page_footer(meta, 4)
    )

    activity_rows = [
        {"label": row["label"], "values": row["values"]}
        for row in model["debt"]["activity"]
    ]
    pages.append(
        _page_header(
            meta,
            5,
            "DEUDA Y CRÉDITOS INTERNOS",
            "¿Quién debe a quién, cuánto y cómo evolucionó?",
        )
        + '<div class="page-grid"><div class="span-7">'
        + _table("Posición al cierre · USD", periods, model["debt"]["positions"], compact=True)
        + '</div><div class="span-5"><div class="panel"><h2>Evolución de principales relaciones</h2><div class="chart-slot tall"></div></div></div>'
        + '<div class="span-12">'
        + _table(
            "Actividad por relación · " + meta["last_period_label"],
            model["debt"]["activity_headers"],
            activity_rows,
            compact=True,
        )
        + '<p class="caveat">Stock y actividad se presentan por separado: los saldos no se obtienen sumando meses.</p></div></div>'
        + _page_footer(meta, 5)
    )

    count_rows = [
        {"label": key, "values": [value]}
        for key, value in model["quality"]["contract_counts"].items()
    ]
    quality_rows = [
        {"label": row["label"], "values": [row["status"], row["detail"]]}
        for row in model["quality"]["quality_rows"]
    ]
    qa_rows = [
        {"label": row["check"], "values": [row["status"], row["severity"], row["detail"]]}
        for row in model["quality"]["qa_rows"]
    ]
    pages.append(
        _page_header(
            meta,
            6,
            "CONTROL Y CALIDAD",
            "¿Qué sabemos con confianza y qué sigue necesitando evidencia?",
        )
        + '<div class="page-grid"><div class="span-5">'
        + _table("Estado del contrato", ["Métricas"], count_rows, compact=True)
        + '</div><div class="span-7">'
        + _table("Indicadores de calidad", ["Estado", "Detalle"], quality_rows, compact=True)
        + '</div><div class="span-12">'
        + _table("QA del annual dashboard", ["Estado", "Severity", "Detalle"], qa_rows, compact=True)
        + '</div></div>'
        + _page_footer(meta, 6)
    )

    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Informe patrimonial y de gestión</title><style>'
        + css
        + '</style></head><body>'
        + "".join(pages)
        + '</body></html>'
    )


def build_report_model(store: MetricStore, qa: pd.DataFrame | None) -> dict[str, Any]:
    periods = store.periods()
    if not periods:
        raise ValueError("annual metrics contain no year periods")
    labels = _period_labels(periods, store.as_of_date())
    last = periods[-1]

    kpis = []
    for spec in KPI_SPECS:
        cell = store.get(
            spec["metric_id"],
            last,
            spec.get("currency", ""),
            fmt=spec.get("format", "money"),
            track={"page_id": "summary", "component_id": "kpis", "row_id": spec["label"]},
        )
        kpis.append({"label": spec["label"], "value": cell.display})

    portfolio = []
    parts = []
    for value in ["CABA", "Tigre 01", "Tigre 28", "Tigre 32"]:
        cell = store.get(
            "IS.RENT.BY_PROPERTY",
            last,
            "ARS",
            "Lugar",
            value,
            missing_policy="zero",
            track={"page_id": "operations", "component_id": "portfolio_latest", "row_id": value},
        )
        parts.append((value, cell))
    total = sum(cell.value or 0 for _, cell in parts)
    for label, cell in parts:
        share = "—" if abs(total) <= 1e-12 else f"{100 * float(cell.value or 0) / total:.1f}%".replace(".", ",")
        portfolio.append({"label": label, "value": cell.display, "share": share})

    bridge_ars = [
        {"label": "Resultado operativo", "metric_id": "IS.NET.OPERATING", "currency": "ARS", "role": "major"},
        {"label": "+ Funding / aportes", "metric_id": "FUND.CONTRIB.TOTAL", "currency": "ARS"},
        {"label": "− Retiros personales", "metric_id": "DIST.DRAWS.PERSONAL", "currency": "ARS"},
        {"label": "de los retiros: dividendos", "metric_id": "DIST.DIVIDENDS", "currency": "ARS", "indent": 1},
        {"label": "= Resultado post-retiros", "metric_id": "COV.NET.AFTER_DRAWS", "currency": "ARS", "role": "major"},
        {"label": "Tasa de ahorro / retención", "metric_id": "COV.SAVINGS_RATE", "currency": "ARS", "format": "ratio"},
        {"label": "Retiros / resultado operativo", "derived": "draws_to_operating", "format": "ratio"},
    ]
    bridge_usd = [
        {"label": "Resultado operativo USD", "metric_id": "IS.NET.OPERATING", "currency": "USD", "format": "usd", "missing_policy": "zero", "role": "major"},
        {"label": "+ Funding / aportes USD", "metric_id": "FUND.CONTRIB.TOTAL", "currency": "USD", "format": "usd", "missing_policy": "zero"},
        {"label": "− Retiros personales USD", "metric_id": "DIST.DRAWS.PERSONAL", "currency": "USD", "format": "usd", "missing_policy": "zero"},
        {"label": "de los retiros: dividendos USD", "metric_id": "DIST.DIVIDENDS", "currency": "USD", "format": "usd", "missing_policy": "zero", "indent": 1},
        {"label": "= Resultado post-retiros USD", "metric_id": "COV.NET.AFTER_DRAWS", "currency": "USD", "format": "usd", "missing_policy": "zero", "role": "major"},
    ]
    funding = [
        {
            "label": actor,
            "metric_id": "FUND.CONTRIB.BY_FUNDING_ACTOR",
            "currency": "ARS",
            "dimension_name": "funding_actor",
            "dimension_value": actor,
            "missing_policy": "zero",
        }
        for actor in FUNDING_ACTORS
    ]
    distributions_ars = [
        {"label": "Retiros personales ARS", "metric_id": "DIST.DRAWS.PERSONAL", "currency": "ARS"},
        {"label": "Dividendos ARS", "metric_id": "DIST.DIVIDENDS", "currency": "ARS"},
    ]
    distributions_usd = [
        {"label": "Retiros personales USD", "metric_id": "DIST.DRAWS.PERSONAL", "currency": "USD", "format": "usd", "missing_policy": "zero"},
        {"label": "Dividendos USD", "metric_id": "DIST.DIVIDENDS", "currency": "USD", "format": "usd", "missing_policy": "zero"},
    ]
    cash_ars = [
        {"label": "Family Business", "metric_id": "BS.CASH.CLOSE.BOX", "currency": "ARS", "dimension_name": "Box", "dimension_value": "Family Business"},
        {"label": "Property Management", "metric_id": "BS.CASH.CLOSE.BOX", "currency": "ARS", "dimension_name": "Box", "dimension_value": "Property Management"},
        {"label": "TOTAL ARS", "metric_id": "BS.CASH.TOTAL", "currency": "ARS", "role": "major"},
    ]
    cash_usd = [
        {"label": "Family Business", "metric_id": "BS.CASH.CLOSE.BOX", "currency": "USD", "dimension_name": "Box", "dimension_value": "Family Business", "format": "usd"},
        {"label": "Property Management", "metric_id": "BS.CASH.CLOSE.BOX", "currency": "USD", "dimension_name": "Box", "dimension_value": "Property Management", "format": "usd"},
        {"label": "TOTAL USD", "metric_id": "BS.CASH.TOTAL", "currency": "USD", "format": "usd", "role": "major"},
    ]
    fx = []
    for metric_id, label in [
        ("TR.FX.CONVERSION.IN", "Conversión - entrada"),
        ("TR.FX.CONVERSION.OUT", "Conversión - salida"),
        ("TR.FX.COST.OUT", "Costo / spread"),
        ("TR.FX.NET", "Movimiento neto FX"),
    ]:
        fx.append({"label": label + " ARS", "metric_id": metric_id, "currency": "ARS", "missing_policy": "zero"})
        fx.append({"label": label + " USD", "metric_id": metric_id, "currency": "USD", "format": "usd", "missing_policy": "zero"})

    debt_specs = [
        {
            "label": relation,
            "metric_id": "ID.DEBT.OPEN.BY_COUNTERPARTY",
            "currency": "USD",
            "dimension_name": "debtor_creditor",
            "dimension_value": relation,
            "format": "usd",
        }
        for relation in DEBT_RELATIONS
    ]
    debt_specs.extend([
        {"label": "TOTAL DEUDA ABIERTA", "metric_id": "ID.DEBT.TOTAL.OPEN", "currency": "USD", "format": "usd", "role": "major"},
        {"label": "Principal abierto", "metric_id": "ID.DEBT.PRINCIPAL.OPEN", "currency": "USD", "format": "usd"},
        {"label": "Interés abierto", "metric_id": "ID.DEBT.INTEREST.OPEN", "currency": "USD", "format": "usd"},
        {"label": "Posición neta PM", "metric_id": "ID.DEBT.NET_PM_POSITION", "currency": "USD", "format": "usd", "role": "major"},
    ])
    activity_map = [
        ("Nuevos claims", "ID.DEBT.ACTIVITY.NEW_CLAIMS"),
        ("Interés", "ID.DEBT.ACTIVITY.INTEREST_ACCRUED"),
        ("Repagos", "ID.DEBT.ACTIVITY.REPAYMENTS"),
        ("Ajustes", "ID.DEBT.ACTIVITY.ADJUSTMENTS"),
        ("Neto", "ID.DEBT.ACTIVITY.NET_CHANGE"),
    ]
    debt_activity = []
    for relation in DEBT_RELATIONS:
        values = [
            store.get(
                metric_id,
                last,
                "USD",
                "debtor_creditor",
                relation,
                fmt="usd",
                track={
                    "page_id": "debt",
                    "component_id": "activity_latest",
                    "row_id": f"{relation}:{label}",
                },
            ).display
            for label, metric_id in activity_map
        ]
        debt_activity.append({"label": relation, "values": values})

    contract_counts = (
        store.contract.get("validation_status", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "blank")
        .value_counts()
        .to_dict()
    )
    quality_rows = []
    for spec in QUALITY_METRICS:
        subset = store.metrics.loc[store.metrics["metric_id"].eq(spec["metric_id"])]
        statuses = sorted(set(item for item in subset["value_status"].tolist() if item)) if not subset.empty else []
        quality_rows.append({
            "label": spec["label"],
            "status": "OK" if "available" in statuses else "No disponible",
            "detail": "status=" + (",".join(statuses) or "sin observaciones"),
        })

    qa_rows = []
    if qa is not None:
        for _, row in qa.iterrows():
            qa_rows.append({
                "check": _text(row.get("check")),
                "status": _text(row.get("status")).upper(),
                "severity": _text(row.get("severity")),
                "detail": _text(row.get("detail")),
            })

    return {
        "meta": {
            **REPORT_META,
            "as_of_date": store.as_of_date(),
            "run_id": store.run_id(),
            "period_columns": labels,
            "last_period": last,
            "last_period_label": labels[-1],
        },
        "summary": {
            "kpis": kpis,
            "rows": _rows(store, SUMMARY_ROWS, periods, "summary", "summary_table"),
        },
        "operations": {
            "rows": _rows(store, OPERATING_ROWS, periods, "operations", "operating_table"),
            "usd_rows": _rows(store, OPERATING_USD_ROWS, periods, "operations", "operating_usd"),
            "portfolio": portfolio,
        },
        "application": {
            "bridge_ars": _rows(store, bridge_ars, periods, "application", "bridge_ars"),
            "bridge_usd": _rows(store, bridge_usd, periods, "application", "bridge_usd"),
            "funding": _rows(store, funding, periods, "application", "funding"),
            "distributions_ars": _rows(store, distributions_ars, periods, "application", "dist_ars"),
            "distributions_usd": _rows(store, distributions_usd, periods, "application", "dist_usd"),
        },
        "treasury": {
            "cash_ars": _rows(store, cash_ars, periods, "treasury", "cash_ars"),
            "cash_usd": _rows(store, cash_usd, periods, "treasury", "cash_usd"),
            "fx": _rows(store, fx, periods, "treasury", "fx"),
        },
        "debt": {
            "positions": _rows(store, debt_specs, periods, "debt", "positions"),
            "activity": debt_activity,
            "activity_headers": [item[0] for item in activity_map],
        },
        "quality": {
            "contract_counts": contract_counts,
            "quality_rows": quality_rows,
            "qa_rows": qa_rows,
        },
    }


def validate_report(store: MetricStore) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check: str, ok: bool, detail: str, severity: str = "error") -> None:
        rows.append({
            "check": check,
            "status": "pass" if ok else ("warn" if severity == "warning" else "fail"),
            "severity": severity,
            "detail": detail,
        })

    required_metrics = {
        "metric_id", "period", "Currency", "value", "value_status",
        "dimension_name", "dimension_value",
    }
    required_contract = {
        "metric_id", "flow_or_stock", "frontend_suitability", "legacy_flag",
        "validation_status",
    }
    add(
        "metrics_schema",
        required_metrics.issubset(store.metrics.columns),
        f"missing={sorted(required_metrics-set(store.metrics.columns))}",
    )
    add(
        "contract_schema",
        required_contract.issubset(store.contract.columns),
        f"missing={sorted(required_contract-set(store.contract.columns))}",
    )
    missing_contract = sorted(set(store.metrics["metric_id"]) - set(store.contract["metric_id"]))
    add("contract_metric_coverage", not missing_contract, f"missing={missing_contract}")
    duplicates = store.metrics.duplicated(KEY_COLS, keep=False)
    add("metric_key_unique", not duplicates.any(), f"duplicate_rows={int(duplicates.sum())}")

    cells = pd.DataFrame(store.rendered)
    if cells.empty:
        add("report_cells_present", False, "no rendered cells")
        return pd.DataFrame(rows)

    bad_zero = cells.loc[
        cells["value_status"].astype(str).isin(["unavailable", "not_applicable"])
        & cells["display_value"].astype(str).str.contains(
            r"^(?:\$ |USD )?0(?:[,\.]0+)?$", regex=True
        )
    ]
    add("no_unavailable_as_zero", bad_zero.empty, f"bad_cells={len(bad_zero)}")

    legacy = set(
        store.contract.loc[
            store.contract["legacy_flag"].astype(str).str.lower().eq("true"),
            "metric_id",
        ]
    )
    public = cells.loc[~cells["page_id"].eq("quality")]
    used_legacy = sorted(set(public["metric_id"]) & legacy)
    add("no_legacy_metrics_on_public_pages", not used_legacy, f"legacy_ids={used_legacy}")

    leaked = cells.loc[
        cells["page_id"].eq("summary")
        & cells["metric_id"].eq("ID.DEBT.TOTAL.OPEN")
    ]
    add("summary_excludes_total_open_debt", leaked.empty, f"cells={len(leaked)}")

    required_usd = {
        "IS.RENT.TOTAL", "IS.NET.OPERATING", "DIST.DIVIDENDS",
        "COV.NET.AFTER_DRAWS",
    }
    usd_ids = set(cells.loc[cells["Currency"].eq("USD"), "metric_id"])
    add(
        "requested_usd_lines_rendered",
        required_usd.issubset(usd_ids),
        f"missing={sorted(required_usd-usd_ids)}",
    )
    return pd.DataFrame(rows)


def render_report(
    *,
    metrics_path: Path,
    contract_path: Path,
    qa_path: Path | None,
    out_dir: Path,
) -> dict[str, Path]:
    metrics = pd.read_csv(metrics_path)
    contract = pd.read_csv(contract_path)
    qa = pd.read_csv(qa_path) if qa_path and qa_path.exists() else None
    out_dir.mkdir(parents=True, exist_ok=True)

    store = MetricStore(metrics, contract)
    model = build_report_model(store, qa)
    validations = validate_report(store)
    cells = pd.DataFrame(store.rendered).drop_duplicates()

    cells_path = out_dir / "report_cells.csv"
    validation_path = out_dir / "report_validation.csv"
    html_path = out_dir / "report.html"
    cells.to_csv(cells_path, index=False)
    validations.to_csv(validation_path, index=False)
    css = Path(__file__).with_name("report.css").read_text(encoding="utf-8")
    html_path.write_text(_render_html(model, css), encoding="utf-8")

    if (validations["status"] == "fail").any():
        raise ValueError(
            "annual management report validation failed; inspect report_validation.csv"
        )
    return {
        "html": html_path,
        "cells": cells_path,
        "validation": validation_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render annual management HTML from governed annual dashboard artifacts."
    )
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = render_report(
        metrics_path=args.metrics,
        contract_path=args.contract,
        qa_path=args.qa,
        out_dir=args.out_dir,
    )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
