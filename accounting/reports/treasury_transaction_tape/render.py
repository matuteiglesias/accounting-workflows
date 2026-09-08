from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


TOLERANCE = 0.01
DETAIL_REQUIRED = {
    "tx_id",
    "Date",
    "period",
    "Box",
    "Currency",
    "movement_basis",
    "cash_direction",
    "cash_category",
    "amount_in",
    "amount_out",
    "net_amount",
    "payer",
    "receiver",
    "Lugar",
    "Detalle",
    "semantic_bucket",
    "semantic_subbucket",
    "direction_source",
    "classification_status",
    "review_required",
    "rule_id",
}
ACCOUNTABILITY_REQUIRED = {
    "period",
    "Box",
    "Currency",
    "total_cash_in",
    "total_cash_out",
    "net_cash_flow",
    "closing_control",
    "validated_cash_status",
}


def _money(value: Any) -> str:
    number = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0])
    text = f"{abs(number):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if abs(number) <= TOLERANCE:
        return "0,00"
    return f"-{text}" if number < 0 else text


def _require(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing treasury transaction tape columns: {missing}")


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _prepare_detail(detail_path: Path, as_of_date: str) -> pd.DataFrame:
    detail = pd.read_csv(detail_path)
    _require(detail, DETAIL_REQUIRED, detail_path.name)
    detail = detail.copy()
    detail["Date"] = pd.to_datetime(detail["Date"], errors="coerce")
    if detail["Date"].isna().any():
        bad = detail.loc[detail["Date"].isna(), "tx_id"].astype(str).tolist()
        raise ValueError(f"treasury transaction tape has invalid Date rows: {bad[:10]}")
    detail = detail.loc[detail["Date"].le(pd.Timestamp(as_of_date))].copy()
    for column in ("amount_in", "amount_out", "net_amount"):
        detail[column] = pd.to_numeric(detail[column], errors="coerce")
    if detail[["amount_in", "amount_out", "net_amount"]].isna().any().any():
        raise ValueError("treasury transaction tape contains non-numeric cash amounts")
    detail["period"] = detail["period"].astype(str)
    detail["Box"] = detail["Box"].fillna("").astype(str).str.strip()
    detail["Currency"] = detail["Currency"].fillna("").astype(str).str.strip()
    detail = detail.sort_values(["Box", "Currency", "Date", "tx_id"]).reset_index(drop=True)
    grouped = detail.groupby(["Box", "Currency"], dropna=False, sort=False)
    detail["accum_in"] = grouped["amount_in"].cumsum()
    detail["accum_out"] = grouped["amount_out"].cumsum()
    detail["accum_net"] = grouped["net_amount"].cumsum()
    detail["Date_display"] = detail["Date"].dt.strftime("%d/%m/%Y")
    return detail


def _prepare_accountability(accountability_path: Path, as_of_date: str) -> pd.DataFrame:
    accountability = pd.read_csv(accountability_path)
    _require(accountability, ACCOUNTABILITY_REQUIRED, accountability_path.name)
    accountability = accountability.copy()
    accountability["period"] = accountability["period"].astype(str)
    cutoff_period = str(pd.Timestamp(as_of_date).to_period("M"))
    accountability = accountability.loc[accountability["period"].le(cutoff_period)].copy()
    if accountability.duplicated(["period", "Box", "Currency"], keep=False).any():
        bad = accountability.loc[
            accountability.duplicated(["period", "Box", "Currency"], keep=False),
            ["period", "Box", "Currency"],
        ].to_dict("records")
        raise ValueError(f"monthly cash accountability is not singular by period/Box/Currency: {bad[:10]}")
    for column in ("total_cash_in", "total_cash_out", "net_cash_flow", "closing_control"):
        accountability[column] = pd.to_numeric(accountability[column], errors="coerce")
    if accountability[["total_cash_in", "total_cash_out", "net_cash_flow", "closing_control"]].isna().any().any():
        raise ValueError("monthly cash accountability contains non-numeric treasury values")
    return accountability


def _validation(
    detail: pd.DataFrame,
    accountability: pd.DataFrame,
    detail_qa_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, ok: bool, detail_text: str, *, box: str = "", currency: str = "", period: str = "", amount: float = 0.0) -> None:
        rows.append(
            {
                "check": check,
                "period": period,
                "Box": box,
                "Currency": currency,
                "amount": amount,
                "status": "pass" if ok else "fail",
                "severity": "error",
                "detail": detail_text,
            }
        )

    source_qa = pd.read_csv(detail_qa_path)
    source_status = set(source_qa.get("status", pd.Series(dtype=str)).astype(str).str.lower())
    add(
        "atomic_treasury_source_qa_passes",
        "fail" not in source_status,
        f"source_qa_statuses={sorted(source_status)}",
    )

    bad_membership = detail.loc[
        ~detail["movement_basis"].astype(str).eq("actual_cash")
        | ~detail["cash_direction"].astype(str).isin(["in", "out"])
        | ~detail["direction_source"].astype(str).eq("box_party_match")
    ]
    add(
        "display_population_is_actual_physical_cash_only",
        bad_membership.empty,
        f"bad_rows={len(bad_membership)}",
        amount=float(len(bad_membership)),
    )

    row_gap = (detail["amount_in"] - detail["amount_out"] - detail["net_amount"]).abs()
    max_row_gap = float(row_gap.max()) if len(row_gap) else 0.0
    add(
        "atomic_in_minus_out_equals_net",
        max_row_gap <= TOLERANCE,
        f"max_gap={max_row_gap}",
        amount=max_row_gap,
    )
    running_gap = (detail["accum_in"] - detail["accum_out"] - detail["accum_net"]).abs()
    max_running_gap = float(running_gap.max()) if len(running_gap) else 0.0
    add(
        "running_in_minus_out_equals_running_net",
        max_running_gap <= TOLERANCE,
        f"max_gap={max_running_gap}",
        amount=max_running_gap,
    )

    atomic = (
        detail.groupby(["period", "Box", "Currency"], dropna=False, as_index=False)
        .agg(
            atomic_in=("amount_in", "sum"),
            atomic_out=("amount_out", "sum"),
            atomic_net=("net_amount", "sum"),
        )
    )
    accountability_view = accountability[
        ["period", "Box", "Currency", "total_cash_in", "total_cash_out", "net_cash_flow"]
    ].copy()
    keys = pd.concat(
        [
            atomic[["period", "Box", "Currency"]],
            accountability_view[["period", "Box", "Currency"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    monthly = (
        keys.merge(atomic, on=["period", "Box", "Currency"], how="left")
        .merge(accountability_view, on=["period", "Box", "Currency"], how="left")
        .fillna(0.0)
    )
    for _, row in monthly.iterrows():
        gaps = {
            "in": float(row["atomic_in"]) - float(row["total_cash_in"]),
            "out": float(row["atomic_out"]) - float(row["total_cash_out"]),
            "net": float(row["atomic_net"]) - float(row["net_cash_flow"]),
        }
        gap = max(abs(value) for value in gaps.values())
        add(
            "atomic_month_matches_cash_accountability",
            gap <= TOLERANCE,
            f"gaps={gaps}",
            period=str(row["period"]),
            box=str(row["Box"]),
            currency=str(row["Currency"]),
            amount=gap,
        )

    for (box, currency), group in detail.groupby(["Box", "Currency"], dropna=False, sort=True):
        latest = accountability.loc[
            accountability["Box"].astype(str).eq(str(box))
            & accountability["Currency"].astype(str).eq(str(currency))
        ].sort_values("period")
        if latest.empty:
            add(
                "final_running_net_matches_closing_control",
                False,
                "no accountability close for tape population",
                box=str(box),
                currency=str(currency),
            )
            continue
        last = latest.iloc[-1]
        tape_close = float(group.iloc[-1]["accum_net"])
        control_close = float(last["closing_control"])
        gap = tape_close - control_close
        add(
            "final_running_net_matches_closing_control",
            abs(gap) <= TOLERANCE,
            f"tape={tape_close}; closing_control={control_close}; gap={gap}",
            period=str(last["period"]),
            box=str(box),
            currency=str(currency),
            amount=gap,
        )

    return pd.DataFrame(rows)


def _cell(value: Any) -> str:
    return html.escape("" if value is None or pd.isna(value) else str(value))


def _amount_cell(value: float, *, blank_zero: bool = False, signed: bool = False) -> str:
    number = float(value)
    if blank_zero and abs(number) <= TOLERANCE:
        return "—"
    text = _money(number)
    if signed and number > TOLERANCE:
        text = "+" + text
    return html.escape(text)


def _table_rows(group: pd.DataFrame) -> str:
    rows: list[str] = []
    for period, month in group.groupby("period", sort=True):
        for _, row in month.iterrows():
            concept = str(row.get("Detalle") or row.get("cash_category") or row.get("semantic_subbucket") or "")
            rows.append(
                "<tr>"
                f"<td class='num in'>{_amount_cell(row['amount_in'], blank_zero=True)}</td>"
                f"<td class='num out'>{_amount_cell(row['amount_out'], blank_zero=True)}</td>"
                f"<td class='num net'>{_amount_cell(row['net_amount'], signed=True)}</td>"
                f"<td class='num accum'>{_amount_cell(row['accum_in'])}</td>"
                f"<td class='num accum'>{_amount_cell(row['accum_out'])}</td>"
                f"<td class='num accum strong'>{_amount_cell(row['accum_net'], signed=True)}</td>"
                f"<td>{_cell(row['Date_display'])}</td>"
                f"<td class='concept'>{_cell(concept)}</td>"
                f"<td>{_cell(row.get('payer', ''))}</td>"
                f"<td>{_cell(row.get('receiver', ''))}</td>"
                f"<td>{_cell(row.get('Lugar', ''))}</td>"
                f"<td class='tx'>{_cell(row.get('tx_id', ''))}</td>"
                "</tr>"
            )
        month_in = float(month["amount_in"].sum())
        month_out = float(month["amount_out"].sum())
        month_net = float(month["net_amount"].sum())
        last = month.iloc[-1]
        rows.append(
            "<tr class='month-close'>"
            f"<td class='num'>{_amount_cell(month_in)}</td>"
            f"<td class='num'>{_amount_cell(month_out)}</td>"
            f"<td class='num'>{_amount_cell(month_net, signed=True)}</td>"
            f"<td class='num'>{_amount_cell(last['accum_in'])}</td>"
            f"<td class='num'>{_amount_cell(last['accum_out'])}</td>"
            f"<td class='num strong'>{_amount_cell(last['accum_net'], signed=True)}</td>"
            f"<td colspan='6'>Cierre {html.escape(str(period))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _section(group: pd.DataFrame, accountability: pd.DataFrame) -> str:
    box = str(group.iloc[0]["Box"])
    currency = str(group.iloc[0]["Currency"])
    total_in = float(group["amount_in"].sum())
    total_out = float(group["amount_out"].sum())
    total_net = float(group["net_amount"].sum())
    latest = accountability.loc[
        accountability["Box"].astype(str).eq(box)
        & accountability["Currency"].astype(str).eq(currency)
    ].sort_values("period")
    last_control = float(latest.iloc[-1]["closing_control"]) if not latest.empty else total_net
    validated_status = str(latest.iloc[-1].get("validated_cash_status", "unavailable")) if not latest.empty else "unavailable"
    validated_value = latest.iloc[-1].get("validated_cash_close", pd.NA) if not latest.empty else pd.NA
    validated_label = (
        f"{currency} {_money(validated_value)}"
        if not pd.isna(validated_value) and validated_status == "available"
        else "No disponible"
    )
    return f"""
    <section class='tape-section'>
      <div class='section-head'>
        <div><p class='eyebrow'>BOX × MONEDA</p><h2>{html.escape(box)} · {html.escape(currency)}</h2></div>
        <span>{len(group)} movimientos físicos</span>
      </div>
      <div class='metric-grid'>
        <div class='metric'><span>Entradas acumuladas</span><strong>{html.escape(currency)} {_money(total_in)}</strong></div>
        <div class='metric'><span>Salidas acumuladas</span><strong>{html.escape(currency)} {_money(total_out)}</strong></div>
        <div class='metric'><span>Saldo de control</span><strong>{html.escape(currency)} {_money(last_control)}</strong></div>
        <div class='metric'><span>Caja validada</span><strong>{html.escape(validated_label)}</strong><small>{html.escape(validated_status)}</small></div>
      </div>
      <div class='table-wrap'>
        <table>
          <thead><tr>
            <th>Entrada</th><th>Salida</th><th>Δ neto</th><th>Σ entradas</th><th>Σ salidas</th><th>Σ neto</th>
            <th>Fecha</th><th>Concepto</th><th>De</th><th>A</th><th>Inmueble</th><th>Tx</th>
          </tr></thead>
          <tbody>{_table_rows(group)}</tbody>
          <tfoot><tr>
            <td class='num'>{_amount_cell(total_in)}</td><td class='num'>{_amount_cell(total_out)}</td>
            <td class='num'>{_amount_cell(total_net, signed=True)}</td><td class='num'>{_amount_cell(total_in)}</td>
            <td class='num'>{_amount_cell(total_out)}</td><td class='num strong'>{_amount_cell(total_net, signed=True)}</td>
            <td colspan='6'>Saldo acumulado al corte</td>
          </tr></tfoot>
        </table>
      </div>
      <div class='control-note'>
        <strong>Lectura administrativa:</strong> el saldo de control es la diferencia acumulada entre entradas y salidas físicas registradas.
        Si la caja validada no está disponible, este informe no afirma que ese saldo exista hoy como efectivo o depósito bancario: requiere
        conciliación externa de custodia o documentación adicional de disposiciones que hayan salido del sistema.
      </div>
    </section>
    """


def render_report(
    *,
    detail_path: Path,
    detail_qa_path: Path,
    accountability_path: Path,
    out_dir: Path,
    as_of_date: str,
) -> dict[str, Path]:
    detail_path = Path(detail_path)
    detail_qa_path = Path(detail_qa_path)
    accountability_path = Path(accountability_path)
    out_dir = Path(out_dir)
    detail = _prepare_detail(detail_path, as_of_date)
    accountability = _prepare_accountability(accountability_path, as_of_date)
    if detail.empty:
        raise ValueError("treasury transaction tape has no governed actual-cash rows through cutoff")

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "internal_trace.csv"
    detail.to_csv(trace_path, index=False)
    validation = _validation(detail, accountability, detail_qa_path)
    validation_path = out_dir / "report_validation.csv"
    validation.to_csv(validation_path, index=False)
    if validation["status"].eq("fail").any():
        failed = validation.loc[validation["status"].eq("fail"), "check"].tolist()
        raise ValueError(f"treasury transaction tape validation failed: {failed[:10]}")

    sections = "".join(
        _section(group, accountability)
        for _, group in detail.groupby(["Box", "Currency"], dropna=False, sort=True)
    )
    cutoff_label = pd.Timestamp(as_of_date).strftime("%d/%m/%Y")
    first_date = detail["Date"].min().strftime("%d/%m/%Y")
    css = """
    :root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;background:#f4f6f9}
    *{box-sizing:border-box}body{margin:0;background:#f4f6f9}main{max-width:1500px;margin:0 auto;padding:42px 28px 72px}
    header{background:#152136;color:white;border-radius:22px;padding:34px 38px;margin-bottom:24px;box-shadow:0 16px 40px rgba(22,31,49,.12)}
    h1{font-size:38px;line-height:1.05;margin:6px 0 10px}h2{font-size:25px;margin:3px 0}.subtitle{color:#cfdae9;max-width:900px}
    .eyebrow{font-size:11px;letter-spacing:.16em;font-weight:800;margin:0;color:#7f93ad}.tape-section{background:white;border-radius:20px;padding:26px;margin:22px 0;box-shadow:0 10px 30px rgba(22,31,49,.07)}
    .section-head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:18px}.section-head span{font-size:13px;color:#627087}
    .metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:20px}.metric{border:1px solid #e4e8ef;background:#fafbfc;border-radius:14px;padding:14px 16px}.metric span{display:block;color:#6b7588;font-size:12px;margin-bottom:5px}.metric strong{font-size:20px}.metric small{display:block;color:#8792a4;margin-top:4px}
    .table-wrap{overflow-x:auto;border:1px solid #dde3eb;border-radius:14px}table{border-collapse:collapse;width:100%;font-size:11px;min-width:1220px}th{position:sticky;top:0;background:#eef2f7;color:#344055;text-align:right;padding:9px 8px;border-bottom:1px solid #dbe2eb;white-space:nowrap}th:nth-child(n+7){text-align:left}
    td{padding:7px 8px;border-bottom:1px solid #edf0f4;vertical-align:top;white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums}.accum{background:#fafbfd}.strong{font-weight:800;color:#14233c}.concept{max-width:290px;white-space:normal}.tx{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:9px;color:#6c7688}
    .month-close td{background:#f1f4f8;font-weight:700;border-top:1px solid #d8dee8}.month-close td:last-child{color:#526078}tfoot td{background:#17243a;color:white;font-weight:800;border:0;padding:10px 8px}
    .control-note{margin-top:16px;padding:14px 16px;border-left:4px solid #7185a2;background:#f5f7fa;color:#4e5a6d;line-height:1.5}.method{margin-top:28px;padding:22px 26px;background:#e9edf3;border-radius:16px;color:#445064;line-height:1.55}
    @media(max-width:900px){main{padding:18px 10px}.metric-grid{grid-template-columns:1fr 1fr}header{padding:26px 22px}h1{font-size:30px}}
    @media print{body{background:white}main{max-width:none;padding:12mm}.tape-section{box-shadow:none;break-before:page}.table-wrap{overflow:visible}.metric-grid{grid-template-columns:repeat(4,1fr)}th{position:static}}
    """
    body = f"""<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Rendición transaccional de tesorería</title><style>{css}</style></head><body><main>
    <header><p class='eyebrow'>INFORME DE TESORERÍA · TRAZABILIDAD TRANSACCIONAL</p><h1>Rendición transaccional de tesorería</h1>
    <p class='subtitle'>Entradas, salidas y saldo de control acumulado por Box y moneda · {html.escape(first_date)} a {html.escape(cutoff_label)}. Cada fila corresponde a un movimiento físico gobernado; las monedas nunca se mezclan.</p></header>
    {sections}
    <section class='method'><h2>Alcance y método</h2>
    <p><strong>Qué establece:</strong> reproduce cronológicamente cada movimiento físico gobernado de tesorería y demuestra, transacción por transacción, cómo entradas menos salidas forman el saldo de control.</p>
    <p><strong>Qué no establece:</strong> el saldo de control no constituye por sí solo efectivo físico validado, faltante, apropiación, deuda jurídica ni obligación personal. La caja validada se mantiene como una observación independiente.</p>
    <p><strong>Autoridad:</strong> la membresía del informe proviene exclusivamente del artefacto atómico de tesorería ya gobernado. Este renderer ordena, acumula y reconcilia; no clasifica ni reinterpreta transacciones.</p></section>
    </main></body></html>"""
    html_path = out_dir / "report.html"
    html_path.write_text(body, encoding="utf-8")
    return {"html": html_path, "trace": trace_path, "validation": validation_path}
