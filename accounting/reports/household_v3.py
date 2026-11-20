"""HH1 v3: 2024–2026 provisional November family accounting reports.

Only paid HH ledger transactions through an observed cutoff. Month-end is a
display horizon, NOT an assertion that a future month-end has occurred.
Amounts are nominal ARS; bank custody and individual claims are not inferred.
This intentionally coexists with the historical HH1 v2 producer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from accounting.scope import load_run_scope
from accounting.reports.household_detail import ReportContent, ars, render_report

SCHEMA="household_poc_catalog.v3"
SPANISH=("Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic")
REPORTS=(
    ("virtual_balance","Caja virtual","Entradas, salidas, neto y saldo acumulado desde enero de 2024."),
    ("contributions","Aportes","Aportes y apoyos directos por persona: 2024, 2025 y 2026."),
    ("outflows","Destino de los recursos","Matriz de importes por rubro y año, sin porcentajes."),
    ("monthly_costs","Evolución del gasto","Gasto mensual desde 2024 y promedio de cada año dividido por 12."),
    ("evidence","Comprobantes","Inventario de boletas, statements y pruebas de pago: estado de documentación."),
)
ACTORS=(("MI","Matías"),("Cande","Candela"),("Alen","Alén"),("Alejandro","Alejandro"),("CA","Claudia"))
COST_LABELS=(
    "Luz","Gas","Internet","Agua","ABL","Legales",
    "Mantenimiento · Makro","Mantenimiento · Perro",
    "Mantenimiento · Otros","Servicios sin identificar","Otros gastos",
)


def _svg_text(v: Any) -> str:
    return html.escape(str(v),quote=True)


def _float(v: Any) -> float:
    x=float(v)
    if not math.isfinite(x):
        raise ValueError("nonfinite HH numeric value")
    return x


def _short(x: float) -> str:
    sign="−" if x<0 else ""
    a=abs(x)
    return f"{sign}{a/1_000_000:.1f} M" if a>=1_000_000 else f"{sign}{a/1000:.0f} mil" if a>=1000 else str(round(a))


def _pesos0(x: float) -> str:
    return ("−" if x<0 else "")+"$ "+f"{abs(x):,.0f}".replace(",", ".")


def _svg(parts: list[str], *, width: int, height: int) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 {width} {height}">',
        *parts,"</svg>",
    ])


def _plot_balance(months: list[str], ins: list[float], outs: list[float],
                  nets: list[float], balances: list[float], opening: float) -> str:
    """Signed monthly bars and a separate, negative-direction closing balance line."""
    n=len(months);w,h=1480,690
    x0,x1=90,1430
    band=(x1-x0)/n
    top,bottom,zero=108,325,220
    bound=max(max(ins),max(outs),max(abs(x) for x in nets),1.)*1.18
    lower_top,lower_bottom=389,588
    min_line=min(0.,opening,*balances);max_line=max(0.,opening,*balances)
    margin=max(0.1*(max_line-min_line),5000.)
    min_line-=margin;max_line+=margin
    def ly(x):return lower_bottom-(x-min_line)/(max_line-min_line)*(lower_bottom-lower_top)
    parts=[
        '<rect width="1480" height="690" fill="#fff" rx="16"/>',
        '<text x="70" y="43" font-family="sans-serif" font-size="24" font-weight="700" fill="#17364b">Caja virtual · entradas, salidas, neto y saldo acumulado</text>',
        '<text x="70" y="68" font-family="sans-serif" font-size="14" fill="#617587">ARS nominales · 2024–2026 · barras mensuales y saldo de control</text>',
        '<line x1="90" y1="220" x2="1430" y2="220" stroke="#586c78" stroke-width="1.4"/>',
        '<text x="70" y="213" text-anchor="end" font-size="12" fill="#637787">0</text>',
        '<text x="90" y="363" font-family="sans-serif" font-weight="700" font-size="14" fill="#18394e">Saldo virtual al cierre de cada mes</text>',
    ]
    for j,y in enumerate((2024,2025,2026)):
        idx=j*12
        if idx<n:
            x=x0+(idx+.5)*band
            if j:
                parts.append(f'<line x1="{x-band/2:.1f}" x2="{x-band/2:.1f}" y1="{top}" y2="{lower_bottom+10}" stroke="#c4d5de" stroke-dasharray="5 4"/>')
            parts.append(f'<text x="{x+4:.1f}" y="93" font-family="sans-serif" font-size="13" font-weight="700" fill="#3b6177">{y}</text>')
    for i,(month,in_v,out_v,net_v) in enumerate(zip(months,ins,outs,nets)):
        center=x0+(i+.5)*band
        barwidth=min(8.5,band*.23)
        # Positive green inbound; negative red outgoing below the zero line;
        # signed gray net, never absolute-deficit-upwards.
        for offset,value,col in ((-barwidth-1,in_v,"#23866c"),(0.,-out_v,"#c34c57"),(barwidth+1,net_v,"#84919b")):
            bh=abs(value)/bound*103.
            y=zero-bh if value>=0 else zero
            parts.append(f'<rect x="{center+offset-barwidth/2:.1f}" y="{y:.1f}" width="{barwidth:.1f}" height="{bh:.1f}" fill="{col}" rx="1"/>')
        if month.endswith(("-01","-07")):
            parts.append(f'<text x="{center:.1f}" y="345" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#718699">{SPANISH[int(month[5:])-1]} {month[:4]}</text>')
    for tick in (min_line,0.,max_line):
        y=ly(tick)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e0eaf0"/>')
        parts.append(f'<text x="83" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#6e8393">{_svg_text(_short(tick))}</text>')
    points=[(x0+(i+.5)*band,ly(v)) for i,v in enumerate(balances)]
    poly=" ".join(f"{x:.1f},{y:.1f}" for x,y in points)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#325d88" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    for x,y in points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="#325d88"/>')
    for j,(label,color) in enumerate((("Entradas +","#23866c"),("Salidas −","#c34c57"),("Neto ±","#84919b"),("Saldo acumulado","#325d88"))):
        lx=128+j*314
        parts.append(f'<rect x="{lx}" y="631" width="14" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx+21}" y="642" font-family="sans-serif" font-size="14" fill="#344f63">{label}</text>')
    parts.append('<text x="740" y="674" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#627887">Nov 2026 provisional: incluye solo pagos fechados hasta el corte de datos; no representa efectivo bancario.</text>')
    return _svg(parts,width=w,height=h)


def _plot_contributions(rows: list[tuple[str,list[float]]]) -> str:
    """Five actor groups × three years."""
    w,h=1200,540;left,right=90,1150;top,bottom=118,413
    maxval=max((x for _,vals in rows for x in vals),default=0)*1.14 or 1.
    band=(right-left)/len(rows)
    colors=("#2e779e","#4d9f8c","#c7954f")
    parts=[
        '<rect width="1200" height="540" rx="16" fill="white"/>',
        '<text x="55" y="49" font-size="24" font-family="sans-serif" font-weight="700" fill="#17364b">Aportes reconocidos por persona · 2024 a nov. 2026</text>',
        '<text x="55" y="73" font-size="14" font-family="sans-serif" fill="#677f91">Incluye contrapartidas de gastos pagados directamente; no duplicar transferencias ni cuotas.</text>',
    ]
    for k in range(5):
        val=maxval*k/4
        y=bottom-(val/maxval)*(bottom-top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e3eaee"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" font-size="12" text-anchor="end" fill="#667c8c">{_svg_text(_short(val))}</text>')
    for i,(name,vals) in enumerate(rows):
        center=left+band*(i+.5)
        for j,val in enumerate(vals):
            x=center-33+j*23
            bh=val/maxval*(bottom-top)
            parts.append(f'<rect x="{x:.1f}" y="{bottom-bh:.1f}" width="19" height="{bh:.1f}" rx="2" fill="{colors[j]}"/>')
        parts.append(f'<text x="{center:.1f}" y="438" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#40596e">{_svg_text(name)}</text>')
    for j,(label,col) in enumerate(zip(("2024","2025","2026 hasta nov."),colors)):
        x=295+j*245
        parts.append(f'<rect x="{x}" y="485" width="14" height="14" rx="2" fill="{col}"/>')
        parts.append(f'<text x="{x+22}" y="498" font-size="14" fill="#40596e" font-family="sans-serif">{label}</text>')
    parts.append('<text x="600" y="526" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#657c8e">2026 incluye hasta el 20/11. Noviembre está abierto y no constituye cierre definitivo.</text>')
    return _svg(parts,width=w,height=h)


def _plot_matrix(matrix: dict[str,dict[int,float]], years: list[int]) -> str:
    w,h=1200,750;left=65;top=122
    parts=[
        '<rect width="1200" height="750" fill="white" rx="16"/>',
        '<text x="55" y="47" font-size="24" font-family="sans-serif" font-weight="700" fill="#17364b">Costos de la casa · matriz de importes</text>',
        '<text x="55" y="74" font-size="14" font-family="sans-serif" fill="#677f91">2024, 2025 y 2026 provisional · ARS nominales, sin porcentajes</text>',
    ]
    cols=[(365,"2024"),(570,"2025"),(775,"2026*"),(985,"Total")]
    for x,label in cols:
        parts.append(f'<text x="{x}" y="114" font-size="14" font-family="sans-serif" font-weight="700" text-anchor="end" fill="#2c596f">{label}</text>')
    for i,label in enumerate(COST_LABELS):
        y=137+i*49
        parts.append(f'<rect x="{left}" y="{y-15}" width="1080" height="41" rx="5" fill="{"#f1f7f9" if i%2==0 else "#fff"}"/>')
        parts.append(f'<text x="{left+12}" y="{y+10}" font-size="14" fill="#375367" font-family="sans-serif">{_svg_text(label)}</text>')
        vals=[matrix[label][yr] for yr in years]
        for (x,_),val in zip(cols,vals+[sum(vals)]):
            parts.append(f'<text x="{x}" y="{y+10}" font-family="sans-serif" font-size="14" text-anchor="end" fill="#2f4d62">{_svg_text(_pesos0(val))}</text>')
    y=137+len(COST_LABELS)*49
    parts.append(f'<rect x="{left}" y="{y-15}" width="1080" height="44" rx="5" fill="#e1f0ed"/>')
    parts.append(f'<text x="{left+12}" y="{y+12}" font-size="15" font-family="sans-serif" font-weight="700" fill="#205b61">Total</text>')
    totals=[sum(matrix[l][yr] for l in COST_LABELS) for yr in years]
    for (x,_),val in zip(cols,totals+[sum(totals)]):
        parts.append(f'<text x="{x}" y="{y+12}" font-family="sans-serif" font-size="15" font-weight="700" text-anchor="end" fill="#205b61">{_svg_text(_pesos0(val))}</text>')
    parts.append('<text x="600" y="718" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#5d768a">* 2026 incluye pagos hasta 20/11; el cuadro completo con centavos está en el informe HTML.</text>')
    return _svg(parts,width=w,height=h)


def _plot_costs(months: list[str], costs: list[float], avg: dict[int,float]) -> str:
    w,h=1480,510;left,right=92,1430;top,bottom=101,386;n=len(months)
    maxv=max(max(costs),*avg.values(),1.)*1.1
    gap=(right-left)/n
    parts=[
        '<rect width="1480" height="510" fill="white" rx="16"/>',
        '<text x="60" y="48" font-family="sans-serif" font-size="24" font-weight="700" fill="#17364b">Gasto por mes · 2024–2026</text>',
        '<text x="60" y="72" font-family="sans-serif" font-size="14" fill="#687f90">Barras: gastos pagados · segmentos: promedio anual dividido por 12 (2026 hasta noviembre)</text>',
    ]
    for k in range(5):
        val=maxv*k/4;y=bottom-(val/maxv)*(bottom-top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e4ebef"/>')
        parts.append(f'<text x="{left-7}" y="{y+4:.1f}" font-family="sans-serif" text-anchor="end" font-size="11" fill="#6d8393">{_svg_text(_short(val))}</text>')
    for i,(month,cost) in enumerate(zip(months,costs)):
        x=left+(i+.5)*gap
        bh=cost/maxv*(bottom-top)
        parts.append(f'<rect x="{x-8:.1f}" y="{bottom-bh:.1f}" width="16" height="{bh:.1f}" fill="#5b8faa" rx="2"/>')
        if month.endswith(("-01","-07")):
            parts.append(f'<text x="{x:.1f}" y="405" font-family="sans-serif" font-size="11" text-anchor="middle" fill="#667f90">{SPANISH[int(month[-2:])-1]} {month[:4]}</text>')
    for j,yr in enumerate((2024,2025,2026)):
        start=left+j*12*gap;end=min(right,left+(j+1)*12*gap)
        y=bottom-avg[yr]/maxv*(bottom-top)
        parts.append(f'<line x1="{start}" y1="{y:.1f}" x2="{end}" y2="{y:.1f}" stroke="#c17837" stroke-width="3"/>')
        parts.append(f'<text x="{(start+end)/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#935a28">{yr}: {_svg_text(_short(avg[yr]))}/mes</text>')
        if j:
            parts.append(f'<line x1="{start}" y1="{top}" x2="{start}" y2="{bottom}" stroke="#becfd9" stroke-dasharray="5 4"/>')
    parts.append('<text x="740" y="483" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#667d91">Importes sin ajuste por inflación. El promedio de 2026 usa denominador 12, aunque noviembre esté parcial.</text>')
    return _svg(parts,width=w,height=h)


def _plot_evidence(year_stats: dict[int,dict[str,int]]) -> str:
    w,h=1030,450
    colors={"verificado":"#2e8c74","en_revision":"#cf9551","sin_vinculo":"#a0acb6"}
    labels={"verificado":"Verificado","en_revision":"En revisión","sin_vinculo":"Sin enlace aprobado"}
    parts=['<rect width="1030" height="450" rx="16" fill="white"/>',
           '<text x="50" y="52" font-family="sans-serif" font-size="23" font-weight="700" fill="#17364b">Comprobantes y boletas · estado documental HH</text>',
           '<text x="50" y="78" font-family="sans-serif" font-size="14" fill="#667c8d">Cantidad de operaciones de gasto por estado de evidencia; enlaces sujetos a autorización.</text>']
    total_max=max((sum(m.values()) for m in year_stats.values()),default=1)
    for j,year in enumerate((2024,2025,2026)):
        y=132+j*83;off=0
        parts.append(f'<text x="90" y="{y+25}" text-anchor="end" font-family="sans-serif" font-size="17" font-weight="700" fill="#37546c">{year}</text>')
        for key in ("verificado","en_revision","sin_vinculo"):
            val=year_stats[year][key]
            width=640*val/total_max
            if width>0:
                parts.append(f'<rect x="{124+off:.1f}" y="{y}" width="{width:.1f}" height="39" fill="{colors[key]}"/>')
                if width>45:
                    parts.append(f'<text x="{124+off+width/2:.1f}" y="{y+25}" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="white">{val}</text>')
            off+=width
        parts.append(f'<text x="796" y="{y+25}" font-family="sans-serif" font-size="16" fill="#4a6578">Total {sum(year_stats[year].values())}</text>')
    for j,key in enumerate(colors):
        x=82+j*290
        parts.append(f'<rect x="{x}" y="403" width="13" height="13" fill="{colors[key]}"/>')
        parts.append(f'<text x="{x+20}" y="414" font-family="sans-serif" font-size="12" fill="#4a6578">{labels[key]}</text>')
    return _svg(parts,width=w,height=h)


def _cost_label(r: pd.Series) -> str:
    bucket=str(r.semantic_subbucket).lower()
    composite=" ".join(str(r.get(k,"") or "") for k in ("issuer","Detalle","tag")).lower()
    if bucket=="services":
        if "edenor" in composite: return "Luz"
        if "metrogas" in composite: return "Gas"
        if "internet" in composite or "telecentro" in composite: return "Internet"
        if "aysa" in composite or "agua" in composite: return "Agua"
        return "Servicios sin identificar"
    if bucket=="taxes":return "ABL" if "abl" in composite else "Otros gastos"
    if bucket=="legal":return "Legales"
    if bucket=="maintenance":
        if "perro" in composite or "hh_pet_dog" in composite:return "Mantenimiento · Perro"
        if "makro" in composite or "macro" in composite:return "Mantenimiento · Makro"
        return "Mantenimiento · Otros"
    return "Otros gastos"


def _assert_close(got: float,want: float, label: str) -> None:
    if abs(got-want)>.03:
        raise ValueError(f"HH v3 reconciliation failed: {label}: {got} != {want}")


def _private_evidence_coverage(snapshot: Path, ledger: pd.DataFrame) -> tuple[dict[str,list[tuple[str,str]]],dict[str,int]]:
    """Read old private HH-only metadata, never publish its PDFs or links.

    Historical source run may differ if every referenced tx is byte-equivalent
    in identity, date and ARS amount. Permission flags must explicitly forbid
    publication. This is metadata lineage, not clearance to share evidence.
    """
    root=Path(snapshot).resolve(strict=True)
    m=json.loads((root/"manifest.json").read_text(encoding="utf8"))
    if (m.get("scope_tag")!="HH_EVIDENCE" or m.get("publication_allowed") is not False
            or m.get("stakeholder_publication_allowed") is not False):
        raise ValueError("HH evidence metadata must be private, scoped and nonpublishable")
    required=pd.read_csv(root/"required_transactions.csv",dtype={"tx_id":str})
    relation=pd.read_csv(root/"transaction_evidence.csv",dtype={"tx_id":str})
    docs=pd.read_csv(root/"evidence_documents.csv")
    if required.tx_id.duplicated().any() or docs.evidence_id.duplicated().any():
        raise ValueError("duplicate HH evidence source identifier")
    if set(required.Box)!={"Household"}:
        raise ValueError("cross-Box documentary inventory")
    if not set(relation.tx_id).issubset(set(required.tx_id)) or not set(relation.evidence_id).issubset(set(docs.evidence_id)):
        raise ValueError("private evidence relation contains unknown sources")
    current=ledger.set_index("tx_id")
    for x in required.itertuples(index=False):
        if x.tx_id not in current.index:
            raise ValueError("old HH evidence source is absent in current run")
        row=current.loc[x.tx_id]
        if str(row.Date)!=str(x.Date) or abs(float(row.amount)-float(x.amount))>.001 or str(row.Box)!="Household":
            raise ValueError("source transaction changed since documentary inventory")
    counts={
        "private_docs":len(docs),
        "candidate_relations":int((relation.status=="candidate").sum()),
        "approved_internal_proof_relations":int(((relation.status=="approved")&(relation.relation.isin(["payment_proof","transfer_proof"]))).sum()),
        "linked_transactions":int(relation.tx_id.nunique()),
    }
    if counts["private_docs"]!=m.get("distinct_documents") or counts["approved_internal_proof_relations"]!=m.get("approved_payment_proof_relations"):
        raise ValueError("private evidence manifest and relations disagree")
    index={}
    for row in relation.itertuples(index=False):
        index.setdefault(str(row.tx_id),[]).append((str(row.relation),str(row.status)))
    return index,counts


def build_household_v3(run_root: Path, out_dir: Path, *,
                       data_as_of: str="2026-11-20", month_end: str="2026-11-30",
                       evidence_snapshot: Path | None = None) -> Path:
    run_root=Path(run_root).resolve(strict=True)
    scope=load_run_scope(run_root)
    if scope.tag!="HH" or set(scope.boxes)!={"Household"} or not run_root.name.endswith("_HH"):
        raise ValueError("HH v3 requires exact Household-only run")
    asof=pd.Timestamp(data_as_of)
    end=pd.Timestamp(month_end)
    if str(asof.date())!=data_as_of or str(end.date())!=month_end:
        raise ValueError("cutoff dates must be ISO")
    if asof>pd.Timestamp.now(tz="UTC").tz_localize(None).normalize():
        raise ValueError("data cutoff may not be in the future")
    if asof>end or end!=end.to_period("M").end_time.normalize() or end.year!=2026 or end.month!=11:
        raise ValueError("November 2026 provisional closure requires 2026-11-30 horizon")
    stage=json.loads((run_root/"meta/stage_A_ingest.json").read_text())
    if stage["params"].get("cutoff_date")!=data_as_of:
        raise ValueError("source Stage A cutoff must equal the declared data date")
    c=pd.read_csv(run_root/"monthly_cash_accountability.csv")
    s=pd.read_csv(run_root/"classification_audit.csv")
    ledger=pd.read_csv(run_root/"ledger_canonical.csv",dtype={"tx_id":str})
    if set(c.Box.dropna())!={"Household"} or set(s.Box.dropna())!={"Household"} or set(ledger.Box.dropna())!={"Household"}:
        raise ValueError("HH v3 refuses foreign-Box ledger")
    if s.tx_id.duplicated().any() or ledger.tx_id.duplicated().any() or set(s.tx_id)!=set(ledger.tx_id):
        raise ValueError("HH v3 requires one-to-one semantic and ledger identity")
    if s.Date.astype(str).max()>data_as_of or ledger.Date.astype(str).max()>data_as_of:
        raise ValueError("source contains dates after observed cutoff")
    if not c.reconciliation_status.eq("reconciled").all():
        raise ValueError("HH v3 refuses unreconciled cash rows")
    months=pd.period_range("2024-01","2026-11",freq="M").astype(str).tolist()
    cash=c[(c.Currency=="ARS") & (c.period.isin(months))].copy()
    if cash.period.duplicated().any() or set(cash.period)!=set(months):
        raise ValueError("HH v3 requires every ARS month from 2024-01 to 2026-11")
    cash=cash.set_index("period").loc[months]
    for col in ("total_cash_in","total_cash_out","net_cash_flow","opening_control","closing_control"):
        cash[col]=cash[col].astype(float)
    if abs(float(cash.closing_control.iloc[-1])-(float(cash.opening_control.iloc[0])+float(cash.net_cash_flow.sum())))>.03:
        raise ValueError("HH v3 opening-to-closing bridge does not reconcile")
    for j in range(len(months)):
        _assert_close(float(cash.iloc[j].opening_control+cash.iloc[j].net_cash_flow),
                      float(cash.iloc[j].closing_control),"monthly net")
        if j>0:_assert_close(float(cash.iloc[j].opening_control),float(cash.iloc[j-1].closing_control),"month continuity")
    years=(2024,2025,2026)
    s=s.merge(ledger[["tx_id","issuer"]],on="tx_id",validate="one_to_one",how="left")
    s["year"]=s.Date.str[:4].astype(int)
    s["month"]=s.Date.str[:7]
    base=s[(s.Currency=="ARS")&(s.year.isin(years))].copy()
    expense=base[(base.semantic_bucket=="property_opex")&(base.direction=="out")].copy()
    funding=base[(base.semantic_bucket=="funding_contribution") | (
        (base.semantic_bucket=="property_opex")&(base.direction=="in")
    )].copy()
    if expense.empty or funding.empty:
        raise ValueError("HH v3 needs expenses and recognized funding")
    unknown=set(funding.payer)-{p for p,_ in ACTORS}
    if unknown:raise ValueError(f"Unmapped HH contributors need review: {sorted(unknown)}")
    contrib=funding.groupby(["payer","year"]).amount.sum()
    cash_year={yr:cash[cash.index.str.startswith(str(yr))] for yr in years}
    for yr in years:
        _assert_close(float(funding[funding.year.eq(yr)].amount.sum()),
                      float(cash_year[yr].total_cash_in.sum()),f"contributions {yr}")
        if (expense.amount<0).any():
            raise ValueError("negative expenses require separate reversal accounting")
    # Source issuer is authoritative for supplier identification when legacy
    # semantic categories are coarse. Never rely on freeform guesses alone.
    expense["cost_kind"]=expense.apply(_cost_label,axis=1)
    matrix={label:{yr:float(expense[(expense.cost_kind==label)&(expense.year==yr)].amount.sum()) for yr in years} for label in COST_LABELS}
    for yr in years:_assert_close(sum(matrix[l][yr] for l in COST_LABELS),
                                float(expense[expense.year.eq(yr)].amount.sum()),f"cost matrix {yr}")
    total_cost={yr:sum(matrix[l][yr] for l in COST_LABELS) for yr in years}
    expense_by_month=expense.groupby("month").amount.sum()
    costs=[float(expense_by_month.get(month,0)) for month in months]
    avg={yr:total_cost[yr]/12 for yr in years}
    actor_rows=[(name,[float(contrib.get((code,yr),0)) for yr in years]) for code,name in ACTORS]
    # Do not infer documentary coverage from an accounting payment flag.
    # A separate privately reviewed evidence package can be linked only
    # through the exact-run, audience-authorized optional viewer workflow.
    private_relations={}
    private_counts={"private_docs":0,"candidate_relations":0,"approved_internal_proof_relations":0,"linked_transactions":0}
    if evidence_snapshot is not None:
        private_relations,private_counts=_private_evidence_coverage(Path(evidence_snapshot),ledger)
    statuses={}
    for yr in years:
        subset=expense[expense.year.eq(yr)]
        # ledger evidence columns are typically empty, so no verified links
        # are invented. Approved proof requires a separate documentary audit.
        linked=int(sum(bool(str(x).strip()) for x in subset["evidence_ref"].dropna())) if "evidence_ref" in subset else 0
        if linked:
            raise ValueError("HH v3 found ledger evidence refs: require separate audited document mapping")
        identified=int(subset.tx_id.isin(set(private_relations)).sum())
        statuses[yr]={"verificado":0,"en_revision":identified,"sin_vinculo":len(subset)-identified}
    chart={
        "virtual_balance":_plot_balance(months,cash.total_cash_in.tolist(),cash.total_cash_out.tolist(),
                                        cash.net_cash_flow.tolist(),cash.closing_control.tolist(),
                                        float(cash.opening_control.iloc[0])),
        "contributions":_plot_contributions(actor_rows),
        "outflows":_plot_matrix(matrix,list(years)),
        "monthly_costs":_plot_costs(months,costs,avg),
        "evidence":_plot_evidence(statuses),
    }
    opening=float(cash.opening_control.iloc[0]);closing=float(cash.closing_control.iloc[-1]);net=float(cash.net_cash_flow.sum())
    total_in=float(cash.total_cash_in.sum());total_out=float(cash.total_cash_out.sum())
    month_rows=tuple((m,ars(cash.at[m,"total_cash_in"]),ars(cash.at[m,"total_cash_out"]),
                      ars(cash.at[m,"net_cash_flow"]),ars(cash.at[m,"opening_control"]),
                      ars(cash.at[m,"closing_control"])) for m in months)
    actors_rows=tuple((name,*(ars(v) for v in vals),ars(sum(vals))) for name,vals in actor_rows)
    expense_rows=tuple((label,*(ars(matrix[label][yr]) for yr in years),ars(sum(matrix[label].values())))
                       for label in COST_LABELS)+(("TOTAL",*(ars(total_cost[yr]) for yr in years),ars(sum(total_cost.values()))),)
    costs_rows=tuple((m,ars(v)) for m,v in zip(months,costs))+tuple(
        (f"PROMEDIO {yr} / 12",ars(avg[yr])) for yr in years
    )
    evidence_rows=[]
    for row in expense.sort_values(["Date","tx_id"],ascending=[False,True]).itertuples(index=False):
        links=private_relations.get(str(row.tx_id),[])
        bill=("Candidata · revisión" if any(r=="liability_source" and st=="candidate" for r,st in links)
              else "Aprobada internamente" if any(r in ("liability_source","statement_context") and st=="approved" for r,st in links)
              else "Pendiente")
        proof=("Aprobada internamente" if any(r in ("payment_proof","transfer_proof") and st=="approved" for r,st in links)
               else "Candidata · revisión" if any(r in ("payment_proof","transfer_proof") and st=="candidate" for r,st in links)
               else "Pendiente")
        access=("En inventario privado · sin permiso de publicación" if links else
                "Sin documento vinculado · sin enlace público aprobado")
        evidence_rows.append((str(row.Date),str(row.cost_kind),ars(row.amount),
                              bill,proof,access+" · Ref "+str(row.tx_id)[:12]))
    evidence_rows=tuple(evidence_rows)
    contents={
        "virtual_balance":ReportContent(
            "virtual_balance","01 · Caja virtual: entradas, salidas y saldo",
            "¿Cómo se movió la caja desde enero de 2024 y cuál es el déficit real del libro?",
            f"Al comenzar 2024 figuraban {ars(opening)} de saldo de control, arrastrados desde 2023. "
            f"Entre enero de 2024 y noviembre de 2026, las entradas netas de salidas variaron {ars(net)} "
            f"y la posición virtual quedó en {ars(closing)}. Noviembre es parcial: datos hasta {data_as_of}, "
            f"horizonte de presentación {month_end}, sin pagos planeados.",
            (("Saldo al abrir 2024",ars(opening)),("Neto 2024–nov. 2026",ars(net)),
             ("Saldo virtual actual",ars(closing)),("Noviembre","Provisional")),
            ("Mes","Entradas +","Salidas − (magnitud)","Resultado neto","Saldo inicial","Saldo final"),month_rows,
            ("Las barras verdes representan entradas; las rojas son salidas bajo cero, las grises son el neto firmado.",
             "La línea de puntos representa el saldo virtual acumulado: desciende cuando el déficit aumenta.",
             "El flujo y los saldos incluyen movimientos de deuda en 2024; no equivalen al costo operativo."),
            "El saldo inicial de 2024 NO se reinicia en cero: conserva −$4.179 de 2023. Caja virtual ≠ dinero en cuentas ≠ pasivo legal a favor de una persona.",
            "monthly_cash_accountability.csv / ledger_canonical.csv, scope HH, ARS"),
        "contributions":ReportContent(
            "contributions","02 · Aportes y apoyos por persona",
            "¿Qué pusieron Matías, Candela, Alén, Alejandro y Claudia en cada año?",
            "Se muestran aportes reconocidos y contrapartidas de pagos directos. "
            f"El total 2026 incluye noviembre solo hasta {data_as_of}. "
            "El apoyo del perro aparece una vez en aportes, y su gasto una vez en costos.",
            (("Total aportado 2024",ars(float(cash_year[2024].total_cash_in.sum()))),
             ("Total aportado 2025",ars(float(cash_year[2025].total_cash_in.sum()))),
             ("Total aportado 2026*",ars(float(cash_year[2026].total_cash_in.sum())))),
            ("Persona","2024","2025","2026 hasta nov.*","Total 2024–2026"),actors_rows,
            ("El ledger contiene contribuciones virtuales de Matías en 2025: no son depósitos de efectivo independientes.",
             "También incluye aportes para pagar servicios y perro realizados directamente por residentes, sin duplicar el gasto.",
             "Una transferencia o apoyo reconocido no representa por sí solo una cuota jurídica exigible."),
            "2026 es provisional. La contribución neta de Candela al perro y el reintegro de Alén deben verificarse contra comprobantes individuales antes de fijar saldos entre personas.",
            "classification_audit.csv · funding_contribution y contrapartida property_opex direction=in"),
        "outflows":ReportContent(
            "outflows","03 · Aplicación de los recursos",
            "¿Cuánto costaron luz, gas, internet, agua, ABL, legales, Makro, perro y otros mantenimientos?",
            "Importes de gastos de Household (no deuda ni repagos) por fecha del asiento, en ARS nominales. "
            "Las partidas no identificables se mantienen visibles; ninguna se asigna por intuición.",
            (("Costos 2024",ars(total_cost[2024])),("Costos 2025",ars(total_cost[2025])),
             ("Costos 2026*",ars(total_cost[2026]))),
            ("Categoría","2024","2025","2026 hasta nov.*","Total"),expense_rows,
            ("Luz, gas, internet y agua se separan usando issuer/proveedor verificado en el ledger y etiquetas de servicio.",
             "Mantenimiento se divide en Makro, perro y otros. No se duplica una contribución que financió una factura.",
             "Los repagos y préstamos de 2024 afectan la caja, pero no se suman a este cuadro de costos operativos."),
            "La clasificación no decide quién es responsable de un gasto ni su cuota. Pagos y consumo pueden corresponder a meses distintos; ABL puede corresponder a propietarios.",
            "classification_audit.csv + ledger_canonical.csv/issuer; solo HH property_opex out ARS"),
        "monthly_costs":ReportContent(
            "monthly_costs","04 · Evolución del gasto",
            "¿Cómo evolucionó el costo mensual y cuál es el promedio anual dividido por 12?",
            f"Los costos nominales suman {ars(sum(total_cost.values()))} entre 2024 y noviembre de 2026. "
            f"Promedios de calendario: 2024 {ars(avg[2024])}, 2025 {ars(avg[2025])}, "
            f"2026 parcial {ars(avg[2026])}, siempre usando denominador doce.",
            (("Promedio 2024 / 12",ars(avg[2024])),("Promedio 2025 / 12",ars(avg[2025])),
             ("Promedio 2026* / 12",ars(avg[2026]))),
            ("Mes o indicador","Gasto operativo HH"),costs_rows,
            ("Las barras reflejan gastos reconocidos por fecha de pago, no necesariamente período de consumo.",
             "El promedio del 2026 se divide por doce aun cuando falta completar noviembre y diciembre: no se debe llamar promedio de once meses.",
             "Son pesos nominales; comparaciones reales requieren ajuste de precios y definición compartible de costos."),
            f"Datos observados hasta {data_as_of}. El promedio 2026 disminuirá o aumentará con nuevas partidas; no es una cuota aprobada.",
            "classification_audit.csv, ARS property_opex direction=out"),
        "evidence":ReportContent(
            "evidence","05 · Boletas y comprobantes",
            "¿Qué movimientos tienen boleta, estado de cuenta o comprobante de pago verificable?",
            "Esta edición muestra la cobertura documental de los gastos HH. "
            f"Hay {private_counts['linked_transactions']} operaciones con documentación privada localizada "
            f"({private_counts['private_docs']} archivos; {private_counts['approved_internal_proof_relations']} "
            "relaciones de prueba de pago aprobadas internamente), pero ninguna tiene permiso de publicación. "
            "No se copian PDFs ni se ofrecen enlaces a una hoja completa que también contiene otras cajas.",
            (("Operaciones de gasto 2024–2026",str(len(expense))),
             ("Operaciones con documentos privados",str(private_counts["linked_transactions"])),
             ("Pruebas de pago aprobadas internamente",str(private_counts["approved_internal_proof_relations"])),
             ("Enlaces autorizados en HH1","0")),
            ("Fecha de asiento","Rubro","Importe","Boleta / statement","Prueba de pago","Estado y referencia"),evidence_rows,
            ("Boleta, estado de cuenta y prueba de pago son documentos distintos y pueden faltar de forma independiente.",
             "Un gasto marcado «pagado» en el ledger no prueba que un comprobante pueda compartirse.",
             "Para habilitar los enlaces de los PDFs hace falta vincular cada archivo a tx_id, verificar integridad y autorizar audiencia por documento."),
            "Este informe no afirma que los documentos no existan. Existen archivos privados de revisión; no se publican en un portal familiar con acceso también posible a terceros.",
            "classification_audit.csv y ledger_canonical.csv / evidencia documental pendiente de autorización"),
    }
    out_dir=Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("HH v3 refuses overwriting populated bundle")
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"charts").mkdir();(out_dir/"reports").mkdir()
    entries=[]
    for i,(rid,title,description) in enumerate(REPORTS):
        svg=chart[rid].encode("utf8");chart_name=f"charts/{rid}.svg"
        (out_dir/chart_name).write_bytes(svg)
        report_markup=render_report(contents[rid],run_id=run_root.name,cutoff=data_as_of,
                           previous_id=REPORTS[i-1][0] if i else None,
                           next_id=REPORTS[i+1][0] if i<len(REPORTS)-1 else None)
        zoom_link=(f'<p class="source"><a href="../charts/{rid}.svg" target="_blank" rel="noopener">'
                   'Abrir gráfico ampliado ↗</a></p>')
        if report_markup.count('<figure class="figure">')!=1:
            raise ValueError("HH v3 chart zoom insertion failed")
        body=report_markup.replace('<figure class="figure">',zoom_link+'<figure class="figure">',1).encode("utf8")
        report_name=f"reports/{rid}.html";(out_dir/report_name).write_bytes(body)
        entries.append({"report_id":rid,"title":title,"description":description,
                        "chart":chart_name,"sha256":sha256(svg).hexdigest(),
                        "html":report_name,"html_sha256":sha256(body).hexdigest()})
    from accounting.reports.specialized.spec import REPORT_SPECS
    skip=[{"report_id":name,"status":"SKIPPED_INCOMPATIBLE","reason":"FBPM semantics not supported for HH"} for name in
          ("annual_management","treasury_accountability","treasury_transaction_tape","debt_accountability")]
    skip.extend({"report_id":r.report_id,"status":"SKIPPED_INCOMPATIBLE","reason":"FBPM specialist not applicable to HH"} for r in REPORT_SPECS)
    qa={
        "no_cross_box_rows":True,"cash_reconciliation":"reconciled",
        "html_reports_generated":5,"cash_months_since_2024":len(months),
        "unique_scoped_transactions":len(s),"expense_transactions_2024_2026":len(expense),
        "annual_cash_in_ars":round(float(cash_year[2026].total_cash_in.sum()),2),
        "annual_cash_out_ars":round(float(cash_year[2026].total_cash_out.sum()),2),
        "annual_net_ars":round(float(cash_year[2026].net_cash_flow.sum()),2),
        "virtual_opening_2024_ars":round(opening,2),
        "virtual_opening_2026_ars":round(float(cash_year[2026].opening_control.iloc[0]),2),
        "virtual_closing_ars":round(closing,2),
        "cost_annual_ars":{str(yr):round(total_cost[yr],2) for yr in years},
        "cost_average_divide_12_ars":{str(yr):round(avg[yr],2) for yr in years},
        "contribution_annual_ars":{str(yr):round(float(cash_year[yr].total_cash_in.sum()),2) for yr in years},
        "unpublished_evidence_attachments":True,
        "unapproved_evidence_links":0,
        "private_evidence_metadata":private_counts,
        "closure_provisional":True,
    }
    catalog={"schema":SCHEMA,"source_run_id":run_root.name,"scope_tag":"HH","scope_boxes":["Household"],
             "data_as_of_date":data_as_of,"as_of_date":month_end,"selected_year":"2026",
             "period_start":"2024-01-01","currency":"ARS","status":"POC_COMPLETE",
             "closing_status":"PROVISIONAL_AS_OF","reports":entries,
             "skipped_existing_reports":skip,"data_qa":qa}
    dest=out_dir/"household_catalog.json"
    dest.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return dest


def main()->None:
    p=argparse.ArgumentParser(description="Build HH1 five reports, 2024–November 2026 provisional")
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--data-as-of",default="2026-11-20")
    p.add_argument("--month-end",default="2026-11-30")
    p.add_argument("--evidence-snapshot",type=Path)
    a=p.parse_args()
    print(build_household_v3(a.run_root,a.out_dir,data_as_of=a.data_as_of,
                             month_end=a.month_end,evidence_snapshot=a.evidence_snapshot))


if __name__=="__main__":main()
