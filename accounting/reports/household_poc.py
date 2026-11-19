"""HH1 scope-isolated, five-chart proof of concept.

No attempt is made to coerce FBPM-only reports into Household. The full
suite is declared SKIPPED_INCOMPATIBLE, leaving core HH metrics untouched.
The report boundary is an allowlisted manifest with five pre-rendered SVGs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
import math
from pathlib import Path

import pandas as pd

from accounting.scope import load_run_scope

SCHEMA = "household_poc_catalog.v1"
COLORS = ("#376e99", "#df9552", "#549c8a", "#8d77b6", "#b7c47b", "#bd748d")
SPANISH_MONTHS = ("Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic")
DETAILS = (
    ("cash_flow", "Entradas y salidas", "Movimientos mensuales de caja registrados en HH."),
    ("contributions", "Aportes por persona", "Aportes reconocidos en 2025 y 2026, diferenciados por responsable."),
    ("outflows", "Salidas por categoría", "Egresos clasificados en el control de tesorería."),
    ("monthly_costs", "Gastos por mes", "Gastos reconocidos, agrupados por período."),
    ("movement_types", "Déficit virtual", "Magnitud del saldo negativo del libro doméstico, no deuda individual."),
)
CATEGORIES = {
    "funding_contribution": "Aportes",
    "property_opex": "Gastos",
    "debt_movement": "Deuda",
    "internal_transfer": "Transferencias",
    "operating_revenue": "Ingresos operativos",
}
COSTS = {
    "services": "Servicios",
    "maintenance": "Mantenimiento",
    "legal": "Legales",
    "taxes": "Impuestos",
    "other": "Otros",
}
OUTFLOWS = (
    ("services_out", "Servicios"),
    ("maintenance_out", "Mantenimiento"),
    ("legal_out", "Legales"),
    ("taxes_out", "Impuestos"),
    ("debt_repayments_out", "Repagos de deuda"),
    ("debt_principal_out", "Préstamos"),
    ("personal_draws_out", "Retiros"),
    ("internal_transfer_out", "Transferencias"),
    ("other_cash_out", "Otros"),
    ("unknown_cash_out", "Sin clasificar"),
)


def _money(value: float) -> str:
    return f"${value / 1_000_000:,.2f} M" if abs(value) >= 1_000_000 else f"${value / 1000:,.0f} mil"


def _txt(x: object) -> str:
    return html.escape(str(x), quote=True)


def _bar_svg(title: str, labels: list[str], series: list[tuple[str, list[float]]],
             *, unit: str = "ARS", note: str = "") -> str:
    """Standalone accessible grouped bar chart, no runtime JS or external assets."""
    if not labels or not series:
        raise ValueError("chart requires non-empty categories and series")
    if any(len(values) != len(labels) for _, values in series):
        raise ValueError("series and label count mismatch")
    all_values = [float(v) for _, vals in series for v in vals]
    if any(not math.isfinite(v) or v < 0 for v in all_values):
        raise ValueError("HH1 proof only accepts finite nonnegative gross magnitudes")
    width, height = 1080, 470
    x0, x1, top, bottom = 90, 1030, 88, 373
    max_value = max(all_values or [0.0], default=0.0)
    max_value = max(max_value * 1.12, 1.0)
    plot_h = bottom - top
    gap = (x1-x0) / len(labels)
    bar_band = min(gap*.77, 105)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_txt(title)}" viewBox="0 0 {width} {height}">',
        f'<title>{_txt(title)}</title>',
        '<rect x="0" y="0" width="1080" height="470" rx="20" fill="#fff"/>',
        f'<text x="55" y="43" font-family="system-ui,sans-serif" font-size="21" font-weight="700" fill="#203b57">{_txt(title)}</text>',
    ]
    for i in range(5):
        val = max_value*i/4
        y = bottom-(val/max_value)*plot_h
        elements.extend([
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e7edf2" stroke-width="1"/>',
            f'<text x="{x0-9}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="system-ui,sans-serif" fill="#748699">{_txt(_money(val) if unit=="ARS" else f"{val:.0f}")}</text>',
        ])
    series_n=len(series)
    for j,(name,values) in enumerate(series):
        color=COLORS[j%len(COLORS)]
        for i,value in enumerate(values):
            bar_w=bar_band/max(series_n,1)*.85
            center=x0+gap*(i+.5)+(j-(series_n-1)/2)*bar_band/max(series_n,1)
            h=(value/max_value)*plot_h
            elements.append(f'<rect x="{center-bar_w/2:.1f}" y="{bottom-h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
    for i,label in enumerate(labels):
        elements.append(f'<text x="{x0+gap*(i+.5):.1f}" y="397" text-anchor="middle" font-family="system-ui,sans-serif" font-size="{11 if len(labels)>9 else 13}" fill="#41566a">{_txt(label)}</text>')
    for j,(name,_) in enumerate(series):
        x=130+j*245
        elements.extend([
            f'<rect x="{x}" y="418" width="13" height="13" rx="3" fill="{COLORS[j%len(COLORS)]}"/>',
            f'<text x="{x+19}" y="429" font-size="12" font-family="system-ui,sans-serif" fill="#41566a">{_txt(name)}</text>'
        ])
    if note:
        elements.append(f'<text x="540" y="457" font-size="11" text-anchor="middle" font-family="system-ui,sans-serif" fill="#667c8e">{_txt(note)}</text>')
    elements.append('</svg>')
    return "\n".join(elements)


def _horizontal_svg(title: str, values: list[tuple[str, float]], note: str = "", unit: str = "ARS") -> str:
    if not values:
        raise ValueError("horizontal chart requires non-empty values")
    values = sorted(values, key=lambda t: -t[1])[:10]
    if any(not math.isfinite(v) or v < 0 for _,v in values):
        raise ValueError("invalid chart data")
    w,h=1080,490
    scale=max([v for _,v in values]+[1])
    barx,barw=260,650
    parts=[
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_txt(title)}" viewBox="0 0 {w} {h}">',
        f'<title>{_txt(title)}</title>',
        '<rect width="1080" height="490" rx="20" fill="white"/>',
        f'<text x="55" y="48" font-family="system-ui,sans-serif" font-size="22" font-weight="700" fill="#203b57">{_txt(title)}</text>'
    ]
    for i,(label,amount) in enumerate(values):
        y=90+i*36
        parts.append(f'<text x="{barx-16}" y="{y+17}" text-anchor="end" font-size="13" font-family="system-ui,sans-serif" fill="#40576c">{_txt(label)}</text>')
        parts.append(f'<rect x="{barx}" y="{y}" width="{barw}" height="25" rx="4" fill="#edf2f5"/>')
        parts.append(f'<rect x="{barx}" y="{y}" width="{max(0,(amount/scale)*barw):.2f}" height="25" rx="4" fill="{COLORS[i%len(COLORS)]}"/>')
        suffix=_money(amount) if unit=="ARS" else str(int(amount))
        parts.append(f'<text x="{barx+barw+13}" y="{y+18}" font-size="12" font-family="system-ui,sans-serif" fill="#1e3b55">{_txt(suffix)}</text>')
    if note:
        parts.append(f'<text x="540" y="470" text-anchor="middle" font-size="11" font-family="system-ui,sans-serif" fill="#60758a">{_txt(note)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def build_household_poc(run_root: Path, out_dir: Path, *, cutoff: str = "2026-10-31") -> Path:
    run_root=Path(run_root).resolve(strict=True)
    scope=load_run_scope(run_root)
    if scope.tag!="HH" or set(scope.boxes)!={"Household"}:
        raise ValueError("HH1 rejects any run with Box outside Household")
    if not run_root.name.endswith("_HH"):
        raise ValueError("HH1 source run must carry an HH scope suffix")
    cutoff_date=pd.Timestamp(cutoff)
    if cutoff_date.strftime("%Y-%m-%d")!=cutoff:
        raise ValueError("cutoff must be an ISO date")
    if cutoff_date!=cutoff_date.to_period("M").end_time.normalize():
        raise ValueError("HH1 must use a complete-month cutoff")

    cash=pd.read_csv(run_root/"monthly_cash_accountability.csv")
    semantic=pd.read_csv(run_root/"classification_audit.csv")
    if set(cash.Box.dropna())!={"Household"} or set(semantic.Box.dropna())!={"Household"}:
        raise ValueError("Cross-scope rows found in HH reporting inputs")
    if semantic.tx_id.duplicated().any():
        raise ValueError("HH semantic source must be one row per tx_id")
    if pd.to_datetime(semantic.Date).max()>cutoff_date:
        raise ValueError("HH source dates extend past reporting cutoff")
    if any(pd.to_datetime(cash.period.astype(str)+"-01")>cutoff_date):
        raise ValueError("HH cash includes periods after cutoff")
    if not cash.reconciliation_status.eq("reconciled").all():
        raise ValueError("HH cash rows must reconcile before publication")

    year=str(cutoff_date.year)
    months=[f"{year}-{i:02d}" for i in range(1,cutoff_date.month+1)]
    cash_year=cash[(cash.Currency=="ARS")&(cash.period.isin(months))].copy()
    if cash_year.empty or cash_year.period.duplicated().any():
        raise ValueError("HH1 needs unique, complete ARS monthly source rows")
    if set(cash_year.period)!=set(months):
        raise ValueError("HH1 monthly ARS source is incomplete")
    cash_year=cash_year.set_index("period").loc[months]
    categories=[(col,name) for col,name in OUTFLOWS if float(cash_year[col].sum())>0]
    if not categories:
        raise ValueError("HH1 has no outgoing categories")
    expenses=semantic[(semantic.Currency=="ARS")&(semantic.Date.astype(str).str[:4]==year)
        &(semantic.semantic_bucket=="property_opex")
        &(semantic.direction=="out")].copy()
    if expenses.empty:
        raise ValueError("HH1 has no classified spending records")
    expenses["month"]=expenses.Date.astype(str).str[:7]
    expense_by_month=expenses.groupby("month").amount.sum()
    expense_by_type=expenses.groupby("semantic_subbucket").amount.sum()
    # Funding claims are only the source-recognized contributions, never inferred
    # from who executed payments. Show the prior year for the virtual 2025 MI
    # policy. Values remain nominal ARS; the viewer must not imply deflation.
    baseline_year=str(int(year)-1)
    movements=semantic[(semantic.Currency=="ARS")
        &(semantic.Date.astype(str).str[:4]==year)].copy()
    funding=semantic[(semantic.Currency=="ARS")
        &(semantic.semantic_bucket=="funding_contribution")
        &(semantic.Date.astype(str).str[:4].isin([baseline_year,year]))].copy()
    if funding.empty:
        raise ValueError("HH1 needs recognised funding rows for actor chart")
    funding["year"]=funding.Date.astype(str).str[:4]
    actor_totals=funding.groupby(["payer","year"],dropna=False).amount.sum()
    known={"MI":"Matías", "Cande":"Candela", "Alen":"Alén", "Alejandro":"Alejandro", "CA":"Claudia"}
    actors=[k for k in ("MI","Cande","Alen","Alejandro","CA") if k in set(funding.payer)]
    actors.extend(sorted(set(funding.payer)-set(actors)))
    if len(actors)>10:
        raise ValueError("HH1 actor chart needs explicit grouping for more than ten counterparties")
    deficit=[max(0.0,-float(v)) for v in cash_year.closing_control.tolist()]
    if any(not math.isfinite(v) for v in deficit):
        raise ValueError("HH1 requires finite monthly virtual balances")

    charts={
        "cash_flow": _bar_svg("Entradas y salidas registradas · ARS", [SPANISH_MONTHS[i] for i in range(cutoff_date.month)],
             [("Entradas",cash_year.total_cash_in.tolist()),("Salidas",cash_year.total_cash_out.tolist())],
             note="Caja operativa registrada; no implica saldo bancario validado."),
        "contributions": _bar_svg("Aportes reconocidos por persona · ARS nominales", [known.get(k,k) for k in actors],
             [(baseline_year,[float(actor_totals.get((k,baseline_year),0)) for k in actors]),
              (year+" hasta octubre",[float(actor_totals.get((k,year),0)) for k in actors])],
             note="Matías 2025: aportes virtuales. No se agregan aportes declarados aún sin asiento."),
        "outflows":_horizontal_svg("Salidas registradas por tipo · ARS",
             [(name,float(cash_year[col].sum())) for col,name in categories],
             note="Distribución del total_cash_out; no duplicar con los gastos reconocidos."),
        "monthly_costs":_bar_svg("Gastos reconocidos por mes · ARS",
             [SPANISH_MONTHS[i] for i in range(cutoff_date.month)],
             [("Gastos",[float(expense_by_month.get(month,0)) for month in months])],
             note="Gastos semánticos: pueden incluir pagos directos sin caja física."),
        "movement_types":_bar_svg("Déficit de caja virtual acumulado · ARS", [SPANISH_MONTHS[i] for i in range(cutoff_date.month)],
             [("Déficit virtual",deficit)],
             note="Magnitud del saldo negativo contable; NO es banco ni crédito individual reconocido."),
    }
    out_dir=Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("HH1 refuses to overwrite a populated report bundle")
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"charts").mkdir()
    entries=[]
    for report_id,title,description in DETAILS:
        filename=f"charts/{report_id}.svg"
        data=charts[report_id].encode("utf-8")
        (out_dir/filename).write_bytes(data)
        entries.append({
            "report_id":report_id,
            "title":title,
            "description":description,
            "chart":filename,
            "sha256":sha256(data).hexdigest(),
        })

    # Compatibility is an explicit nonfatal decision. Existing FBPM reports
    # are untouched and not executed for HH; HH1 provides minimal scoped POC.
    skipped=[
        {"report_id":name,"status":"SKIPPED_INCOMPATIBLE",
         "reason":"requires FBPM-specific semantics or presentation contract; no HH override approved"}
        for name in ("annual_management","treasury_accountability",
                     "treasury_transaction_tape","debt_accountability")
    ]
    try:
        from accounting.reports.specialized.spec import REPORT_SPECS
        for spec in REPORT_SPECS:
            skipped.append({"report_id":spec.report_id,"status":"SKIPPED_INCOMPATIBLE",
                            "reason":"specialized report view assumes the FBPM governance contract"})
    except ImportError:
        raise RuntimeError("Cannot list compatibility decisions without report registry")

    summary={
        "schema":SCHEMA,
        "source_run_id":run_root.name,
        "scope_tag":scope.tag,
        "scope_boxes":list(scope.boxes),
        "as_of_date":cutoff,
        "selected_year":year,
        "currency":"ARS",
        "status":"POC_COMPLETE",
        "reports":entries,
        "skipped_existing_reports":skipped,
        "data_qa":{
            "unique_scoped_transactions":int(len(semantic)),
            "expense_transactions_2026":int(len(expenses)),
            "movement_transactions_2026":int(len(movements)),
            "cash_months_2026":int(len(cash_year)),
            "annual_cash_in_ars":round(float(cash_year.total_cash_in.sum()),2),
            "annual_cash_out_ars":round(float(cash_year.total_cash_out.sum()),2),
            "recognized_spending_ars":round(float(expenses.amount.sum()),2),
            "virtual_closing_ars":round(float(cash_year.closing_control.iloc[-1]),2),
            "prior_year_recognized_MI_virtual_contributions_ars":round(float(actor_totals.get(("MI",baseline_year),0)),2),
            "current_year_recognized_CA_contributions_ars":round(float(actor_totals.get(("CA",year),0)),2),
            "no_cross_box_rows":True,
            "cash_reconciliation":"reconciled",
            "source_hh_historical_last_date":str(semantic.Date.max()),
            "caveat":"Accountant-classified HH expense and cash numbers are not identical concepts.",
        },
    }
    catalog_path=out_dir/"household_catalog.json"
    catalog_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return catalog_path


def main() -> None:
    parser=argparse.ArgumentParser(description="Create five static Household charts, skipping incompatible FBPM reports.")
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--cutoff",default="2026-10-31")
    args=parser.parse_args()
    print(build_household_poc(args.run_root,args.out_dir,cutoff=args.cutoff))


if __name__=="__main__":
    main()
