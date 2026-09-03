from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import pandas as pd

from accounting.reports.debt_accountability.spec import SUBTITLE, TITLE, TOLERANCE


def _h(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _n(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])


def _money(value: Any, currency: str = "USD") -> str:
    number = _n(value)
    rendered = f"{abs(number):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{'−' if number < 0 else ''}{currency} {rendered}"


def _event_table(detail: pd.DataFrame, year: str) -> tuple[str, pd.DataFrame]:
    current = detail.loc[detail["period"].astype(str).str.startswith(year)].copy()
    blocks = []
    events = []
    for tx_id, group in current.groupby("repayment_tx_id", sort=False):
        first = group.iloc[0]
        repayment, remainder = _n(first["repayment_amount"]), _n(first["leftover_amount"])
        allocated = pd.to_numeric(group["allocated_amount"], errors="coerce").fillna(0).sum()
        events.append({"repayment_tx_id": tx_id, "repayment_date": first["repayment_date"], "debtor": first["debtor"], "creditor": first["creditor"], "Currency": first["Currency"], "repayment_amount": repayment, "allocated_amount": allocated, "leftover_amount": remainder, "allocation_rows": int(group["target_debt_id"].fillna("").astype(str).str.strip().ne("").sum())})
        allocations = "".join(
            f'<div class="allocation-row"><span>{_h(row.get("target_detail") or row.get("target_debt_id"))}</span><span>{_money(row.get("balance_before"), first["Currency"])}</span><span>{_money(row.get("allocated_amount"), first["Currency"])}</span><span>{_money(row.get("balance_after"), first["Currency"])}</span></div>'
            for _, row in group.iterrows() if str(row.get("target_debt_id", "")).strip()
        )
        blocks.append(f'<tr><td>{_h(first["repayment_date"])}</td><td>{_h(first["debtor"])} → {_h(first["creditor"])}</td><td class="num">{_money(repayment, first["Currency"])}</td><td class="num">{_money(allocated, first["Currency"])}</td><td class="num">{_money(remainder, first["Currency"])}</td><td>{_h(first["allocation_status"])}</td></tr><tr><td colspan="6"><details><summary>Ver obligaciones aplicadas</summary><div class="allocation"><div class="allocation-row"><strong>Obligación</strong><strong>Antes</strong><strong>Aplicado</strong><strong>Después</strong></div>{allocations}</div></details></td></tr>')
    return "".join(blocks), pd.DataFrame(events)


def _activity_model(activity: pd.DataFrame, year: str) -> pd.DataFrame:
    ytd = activity.loc[activity["period"].astype(str).str.startswith(year)].copy()
    rows = []
    for (debtor, creditor, currency), group in ytd.groupby(["debtor", "creditor", "Currency"], sort=True):
        group = group.sort_values("period")
        def total(kind: str, col: str) -> float:
            return pd.to_numeric(group.loc[group["activity_type"].eq(kind), col], errors="coerce").fillna(0).sum()
        openings = group.loc[group["activity_type"].eq("opening_balance")]
        closings = group.loc[group["activity_type"].eq("closing_balance")]
        opening = _n(openings.iloc[0]["opening_total"]) if not openings.empty else 0
        closing = _n(closings.iloc[-1]["closing_total"]) if not closings.empty else 0
        row = {"debtor": debtor, "creditor": creditor, "Currency": currency, "opening": opening, "new": total("new_claim", "new_principal"), "interest": total("interest_accrual", "interest_accrued"), "repayments": total("repayment", "repayments"), "adjustments": total("adjustment", "adjustments"), "closing": closing}
        if any(abs(row[key]) > TOLERANCE for key in ("opening", "new", "interest", "repayments", "adjustments", "closing")):
            rows.append(row)
    return pd.DataFrame(rows)


def build_validation(position: pd.DataFrame, position_qa: pd.DataFrame, activity: pd.DataFrame, activity_qa: pd.DataFrame, detail: pd.DataFrame, gaps: pd.DataFrame, gaps_qa: pd.DataFrame, *, as_of_date: str) -> pd.DataFrame:
    rows = []
    def add(check: str, ok: bool, detail_text: str) -> None:
        rows.append({"check": check, "status": "pass" if ok else "fail", "severity": "error", "detail": detail_text})
    latest = position["period"].astype(str).max()
    close = position.loc[position["period"].astype(str).eq(latest)]
    totals = close.loc[close["component"].eq("total")]
    principal = close.loc[close["component"].eq("principal"), "open_amount"].sum()
    interest = close.loc[close["component"].eq("interest"), "open_amount"].sum()
    gross = totals["open_amount"].sum()
    add("stock_component_mismatch", abs(gross-principal-interest) <= TOLERANCE, f"gross={gross}; principal={principal}; interest={interest}")
    add("gross_headline_matches_relationships", abs(gross-totals["open_amount"].sum()) <= TOLERANCE, f"residual={gross-totals['open_amount'].sum()}")
    model = _activity_model(activity, as_of_date[:4])
    bridge_residual = (model["opening"]+model["new"]+model["interest"]-model["repayments"]+model["adjustments"]-model["closing"]).abs().max() if not model.empty else 0
    add("bridge_reconciles", _n(bridge_residual) <= TOLERANCE, f"max_residual={bridge_residual}")
    material_adjustments = model["adjustments"].abs().gt(TOLERANCE).sum() if not model.empty else 0
    add("no_material_unexplained_adjustment", material_adjustments == 0, f"rows={material_adjustments}")
    event_bad, future, allocated_total = 0, 0, 0.0
    for _, group in detail.groupby("repayment_tx_id", sort=False):
        repayment, remainder = _n(group.iloc[0]["repayment_amount"]), _n(group.iloc[0]["leftover_amount"])
        allocated = pd.to_numeric(group["allocated_amount"], errors="coerce").fillna(0).sum(); allocated_total += allocated
        event_bad += int(abs(repayment-allocated-remainder) > TOLERANCE)
        target = pd.to_datetime(group["target_opened_at"], errors="coerce"); paid = pd.to_datetime(group["repayment_date"], errors="coerce")
        future += int((target > paid).sum())
    activity_repayments = pd.to_numeric(activity.loc[activity["activity_type"].eq("repayment"), "repayments"], errors="coerce").fillna(0).sum()
    add("repayment_equals_allocated_plus_remainder", event_bad == 0, f"bad_events={event_bad}")
    add("repayment_detail_aggregate_matches_activity", abs(allocated_total-activity_repayments) <= TOLERANCE, f"detail={allocated_total}; activity={activity_repayments}")
    add("no_future_dated_repayment_target", future == 0, f"rows={future}")
    add("native_currency_only", position["Currency"].astype(str).str.strip().ne("").all() and activity["Currency"].astype(str).str.strip().ne("").all(), "no cross-currency aggregate")
    gap_ids = set(gaps["source_tx_id"].astype(str)); targets = set(detail["target_source_tx_id"].dropna().astype(str))
    pair_overlap = ((position["debtor"].astype(str).str.casefold()=="costos") & (position["creditor"].astype(str).str.casefold()=="pm")).sum()
    add("cost_allocation_gap_outside_debt", not (gap_ids & targets) and pair_overlap == 0, f"target_overlap={len(gap_ids & targets)}; position_rows={pair_overlap}")
    dates = set(close["as_of_date"].astype(str))
    add("run_cutoff_matches", dates == {as_of_date}, f"expected={as_of_date}; actual={sorted(dates)}")
    source_fail = sum(q["status"].astype(str).str.casefold().eq("fail").sum() for q in (position_qa, activity_qa, gaps_qa))
    add("source_qa_no_fail", source_fail == 0, f"failures={source_fail}")
    add("report_source_residual", abs(gross-principal-interest) <= TOLERANCE and bridge_residual <= TOLERANCE, "all modeled totals reconcile")
    return pd.DataFrame(rows)


def render_report(*, position_path: Path, position_qa_path: Path, activity_path: Path, activity_qa_path: Path, repayment_detail_path: Path, gaps_path: Path, gaps_qa_path: Path, out_dir: Path, as_of_date: str) -> dict[str, Path]:
    position, position_qa = pd.read_csv(position_path), pd.read_csv(position_qa_path)
    activity, activity_qa = pd.read_csv(activity_path), pd.read_csv(activity_qa_path)
    detail, gaps, gaps_qa = pd.read_csv(repayment_detail_path), pd.read_csv(gaps_path), pd.read_csv(gaps_qa_path)
    validation = build_validation(position, position_qa, activity, activity_qa, detail, gaps, gaps_qa, as_of_date=as_of_date)
    out_dir.mkdir(parents=True, exist_ok=True); validation_path = out_dir/"report_validation.csv"; validation.to_csv(validation_path,index=False)
    if validation["status"].eq("fail").any():
        raise ValueError(f"debt report validation failed; inspect {validation_path}")
    latest, year = position["period"].astype(str).max(), as_of_date[:4]
    close = position[(position["period"].astype(str)==latest) & position["position_status"].eq("available")]
    totals = close[close["component"].eq("total")]; gross=_n(totals["open_amount"].sum()); principal=_n(close.loc[close.component.eq("principal"),"open_amount"].sum()); interest=_n(close.loc[close.component.eq("interest"),"open_amount"].sum())
    model = _activity_model(activity, year)
    applied = _n(model["repayments"].sum())
    events_html, events = _event_table(detail, year)
    event_amount=_n(events["repayment_amount"].sum()); remainder=_n(events["leftover_amount"].sum()); allocation_rows=int(events["allocation_rows"].sum())
    relation_rows="".join(f'<tr><td>{_h(r.debtor)} → {_h(r.creditor)}</td><td>{_h(r.Currency)}</td><td class="num">{_money(r.open_principal,r.Currency)}</td><td class="num">{_money(r.open_interest,r.Currency)}</td><td class="num">{_money(r.open_total,r.Currency)}</td></tr>' for r in totals.itertuples() if _n(r.open_amount)>TOLERANCE)
    activity_rows="".join(f'<tr><td>{_h(r.debtor)} → {_h(r.creditor)}</td><td>{_h(r.Currency)}</td><td class="num">{_money(r.opening,r.Currency)}</td><td class="num">{_money(r.new,r.Currency)}</td><td class="num">{_money(r.interest,r.Currency)}</td><td class="num">{_money(r.repayments,r.Currency)}</td><td class="num">{_money(r.adjustments,r.Currency)}</td><td class="num">{_money(r.closing,r.Currency)}</td></tr>' for r in model.itertuples())
    yearly=[]
    for yr in sorted(set(activity["period"].astype(str).str[:4])):
        m=_activity_model(activity,yr)
        if not m.empty: yearly.append(f'<tr><td>{yr}</td>'+''.join(f'<td class="num">{_money(m[c].sum())}</td>' for c in ["opening","new","interest","repayments","adjustments","closing"])+"</tr>")
    gap_total=gaps.groupby("Currency")["amount"].sum().to_dict()
    gap_rows="".join(f'<div class="card outside"><strong>{_money(r.amount,r.Currency)}</strong><div>{_h(r.description)} · {_h(r.Lugar)}</div><div>Ámbito económico: Property Management</div><div>Estado: Pendiente de asignación</div><div>Deudor jurídico: No determinado</div></div>' for r in gaps.itertuples())
    qa_html="".join(f'<div><strong class="status-{_h(r.status)}">{_h(r.status.upper())}</strong> · {_h(r.check)}</div>' for r in validation.itertuples())
    css=Path(__file__).with_name("report.css").read_text()
    doc=f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{TITLE}</title><style>{css}</style></head><body><main class="report"><header class="hero"><div class="eyebrow">Informe contable gobernado</div><h1>{TITLE}</h1><div class="subtitle">{SUBTITLE}</div><div class="meta"><span>Corte: {as_of_date}</span><span>Período actual: {year} YTD</span><span>Monedas sin mezclar</span></div></header>
    <section class="page"><div class="section-head"><h2>1. Posición actual</h2><div class="note">¿Quién debe a quién y cuánto permanece abierto?</div></div><div class="grid"><div class="kpi"><span>Deuda abierta bruta</span><strong>{_money(gross)}</strong></div><div class="kpi"><span>Principal</span><strong>{_money(principal)}</strong></div><div class="kpi"><span>Interés</span><strong>{_money(interest)}</strong></div><div class="kpi"><span>Repagos aplicados {year}</span><strong>{_money(applied)}</strong></div></div><table><thead><tr><th>Relación</th><th>Moneda</th><th class="num">Principal</th><th class="num">Interés</th><th class="num">Total</th></tr></thead><tbody>{relation_rows}</tbody></table><div class="card outside"><span>Costos pendientes de asignación</span><strong>{' · '.join(_money(v,k) for k,v in gap_total.items())}</strong><div>No integran deuda registrada.</div></div></section>
    <section class="page"><h2>2. Cómo cambió</h2><div class="equation">Apertura + nuevas obligaciones + interés − repagos aplicados ± ajustes = cierre</div><table><thead><tr><th>Año</th><th class="num">Apertura</th><th class="num">Nuevas obligaciones</th><th class="num">Interés</th><th class="num">Repagos</th><th class="num">Ajustes</th><th class="num">Cierre</th></tr></thead><tbody>{''.join(yearly)}</tbody></table><p><strong>Los movimientos se suman. Los saldos se observan al cierre.</strong></p></section>
    <section class="page"><h2>3. Actividad por relación · {year} YTD</h2><table><thead><tr><th>Relación</th><th>Moneda</th><th class="num">Apertura</th><th class="num">Nueva</th><th class="num">Interés</th><th class="num">Repagos</th><th class="num">Ajustes</th><th class="num">Cierre</th></tr></thead><tbody>{activity_rows}</tbody></table></section>
    <section class="page"><h2>4. Repagos y trazabilidad</h2><div class="grid"><div class="kpi"><span>Eventos</span><strong>{len(events)}</strong></div><div class="kpi"><span>Monto de eventos</span><strong>{_money(event_amount)}</strong></div><div class="kpi"><span>Aplicado</span><strong>{_money(applied)}</strong></div><div class="kpi"><span>Remanente</span><strong>{_money(remainder)}</strong><small>{allocation_rows} asignaciones</small></div></div><table><thead><tr><th>Fecha</th><th>Relación</th><th class="num">Evento</th><th class="num">Aplicado</th><th class="num">Sin asignar</th><th>Estado</th></tr></thead><tbody>{events_html}</tbody></table><p class="note">El remanente es un repago registrado aún no imputado completamente bajo la regla vigente; no es deuda adicional ni dinero faltante.</p></section>
    <section class="page"><h2>5. Fuera de deuda y controles</h2><div class="gap-grid">{gap_rows}</div><p class="note">Una partida pendiente de asignación permanece visible sin establecer deudor jurídico ni habilitar repago ordinario.</p><div class="qa">{qa_html}</div></section><footer class="footer">Este informe describe obligaciones registradas en el sistema contable. No constituye por sí solo una determinación de responsabilidad jurídica.</footer></main></body></html>'''
    html_path=out_dir/"report.html"; html_path.write_text(doc,encoding="utf-8")
    return {"html":html_path,"validation":validation_path}


def main() -> None:
    parser=argparse.ArgumentParser();
    for name in ("position","position-qa","activity","activity-qa","repayment-detail","gaps","gaps-qa","out-dir"): parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--as-of-date",required=True)
    args=parser.parse_args(); render_report(position_path=args.position,position_qa_path=args.position_qa,activity_path=args.activity,activity_qa_path=args.activity_qa,repayment_detail_path=args.repayment_detail,gaps_path=args.gaps,gaps_qa_path=args.gaps_qa,out_dir=args.out_dir,as_of_date=args.as_of_date)


if __name__ == "__main__": main()
