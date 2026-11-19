"""Five independently readable, source-owned Household report pages.

Pure HTML rendering: every number is calculated by the HH backend. The viewer
only copies these pages and links to them. No new personal debt ledger and no
inference from filenames or bank balances.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from html import escape
from typing import Iterable


def ars(value: float | int) -> str:
    """Compact Spanish locale for labelled ARS values, preserving cents."""
    val = Decimal(str(value)).quantize(Decimal("0.01"))
    sign = "−" if val < 0 else ""
    a, b = f"{abs(val):,.2f}".split(".")
    return f"{sign}$ {a.replace(',', '.')},{b}"


def text(s: object) -> str:
    return escape(str(s), quote=True)


@dataclass(frozen=True)
class ReportContent:
    report_id: str
    title: str
    question: str
    answer: str
    metric_cards: tuple[tuple[str, str], ...]
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    readout: tuple[str, ...]
    caution: str
    source: str


def render_report(report: ReportContent, *, run_id: str, cutoff: str,
                  previous_id: str | None, next_id: str | None) -> str:
    if not report.rows or not report.headers:
        raise ValueError("A published Household report must contain a reviewable data table")
    if any(len(row) != len(report.headers) for row in report.rows):
        raise ValueError("HH report data table width mismatch")
    if len(report.metric_cards) > 4:
        raise ValueError("Keep the family report digest small")

    kpis = "".join(
        f'<div class="stat"><span>{text(name)}</span><strong>{text(value)}</strong></div>'
        for name, value in report.metric_cards
    )
    headers = "".join(f'<th scope="col">{text(h)}</th>' for h in report.headers)
    rows = "".join(
        "<tr>" + "".join(f"<td>{text(c)}</td>" for c in row) + "</tr>"
        for row in report.rows
    )
    takeaways = "".join(f"<li>{text(item)}</li>" for item in report.readout)
    links = [f'<a href="../index.html">← Volver al recorrido Household</a>']
    if previous_id:
        links.append(f'<a href="{text(previous_id)}.html">← Anterior</a>')
    if next_id:
        links.append(f'<a href="{text(next_id)}.html">Siguiente →</a>')
    nav = " ".join(links)
    # Local URLs are finite chart IDs that already passed producer allowlisting.
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>{text(report.title)} · Household · Balbín</title>
  <style>
    :root{{--ink:#1d344b;--muted:#536a7e;--blue:#2c6488;--bg:#f3f7f9;--line:#dce6ed}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink)}}
    .frame{{max-width:1100px;margin:auto;padding:24px 26px 64px}}
    nav{{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;font-weight:650}}
    a{{color:#18557a}} a:focus-visible{{outline:3px solid #376e99;outline-offset:3px}}
    .label{{font-size:11px;font-weight:800;letter-spacing:.16em;color:var(--blue);text-transform:uppercase}}
    header{{padding:38px 0 17px}} h1{{font-size:clamp(27px,4vw,45px);letter-spacing:-.035em;margin:10px 0 14px}}
    .question{{font-size:20px;font-weight:600;line-height:1.4;max-width:830px}}
    .answer{{font-size:16px;line-height:1.7;color:#2c475e;max-width:880px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px;margin:18px 0 24px}}
    .stat{{background:white;border:1px solid var(--line);border-radius:12px;padding:16px}}
    .stat span{{font-size:12px;line-height:1.4;color:var(--muted);display:block}}
    .stat strong{{display:block;font-size:clamp(18px,2.2vw,25px);margin-top:9px;overflow-wrap:anywhere}}
    section{{background:white;border:1px solid var(--line);border-radius:15px;padding:20px 22px;margin:14px 0}}
    h2{{font-size:20px;letter-spacing:-.02em;margin:3px 0 14px}}
    .figure{{margin:0}} .figure img{{display:block;width:100%;height:auto}}
    figcaption, .source{{font-size:12px;line-height:1.6;color:var(--muted)}}
    .tablebox{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}
    th{{text-align:left;color:#526b80;background:#f0f5f8}} th,td{{padding:10px 12px;border-bottom:1px solid #e4ecf1}}
    tbody tr:nth-child(even){{background:#fafcfd}} tbody tr:hover{{background:#edf5fa}}
    li{{line-height:1.7;margin:5px 0}} .warning{{border-left:4px solid #b17c39;background:#fffaf1}}
    .footer{{font-size:11px;color:#657e91;border-top:1px solid var(--line);padding:17px 0;line-height:1.7}}
    @media(max-width:650px){{.frame{{padding:18px 12px 38px}}section{{padding:15px 12px}}.question{{font-size:17px}}}}
  </style>
</head>
<body>
<div class="frame">
  <nav aria-label="Navegación Household">{nav}</nav>
  <header>
    <span class="label">Balbín · Household · Informe contable de gestión</span>
    <h1>{text(report.title)}</h1>
    <p class="question">{text(report.question)}</p>
    <p class="answer">{text(report.answer)}</p>
    <p class="source">Caja virtual HH · ARS nominales · Corte {text(cutoff)}</p>
  </header>
  <div class="grid">{kpis}</div>
  <section aria-label="Visualización">
    <h2>Visualización del informe</h2>
    <figure class="figure"><img src="../charts/{text(report.report_id)}.svg" width="1080" height="490"
      alt="{text(report.title)} — gráfico; valores completos en la tabla siguiente">
      <figcaption>Gráfico construido por el backend a partir de la corrida HH. Los importes exactos figuran en la tabla.</figcaption></figure>
  </section>
  <section aria-label="Detalle numérico">
    <h2>Datos que sustentan el gráfico</h2>
    <div class="tablebox"><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>
  </section>
  <section aria-label="Cómo interpretar">
    <h2>Qué se puede concluir</h2><ul>{takeaways}</ul>
  </section>
  <section class="warning" aria-label="Precaución">
    <h2>Alcance y precauciones</h2><p>{text(report.caution)}</p>
  </section>
  <footer class="footer">Fuente: {text(report.source)} · Run {text(run_id)}
  · Universo exclusivo Household (HH) · No se incluyen las otras cajas ni datos bancarios ajenos al ledger.</footer>
  <nav aria-label="Más informes Household">{nav}</nav>
</div>
</body>
</html>
"""

def build_report_contents(
    cash_year, expenses, actor_totals, actors: list[str], known: dict[str, str],
    *,
    year: str, baseline_year: str, months: list[str],
    outflow_categories: list[tuple[str, str]],
    month_labels: list[str],
) -> dict[str, ReportContent]:
    """Assemble source-grounded report messages and tables.

    No entitlement or household cost-sharing assumptions are made here.
    """
    opening = float(cash_year.opening_control.iloc[0])
    closing = float(cash_year.closing_control.iloc[-1])
    total_in = float(cash_year.total_cash_in.sum())
    total_out = float(cash_year.total_cash_out.sum())
    net = float(cash_year.net_cash_flow.sum())
    if abs((opening + net) - closing) > .02:
        raise ValueError("HH accounting bridge does not reconcile (opening + net != closing)")
    group_out=[(title,float(cash_year[column].sum())) for column,title in outflow_categories]
    if abs(sum(v for _,v in group_out)-total_out) > .02:
        raise ValueError("HH outflow categories do not exhaust cash outflows")
    expense_by_month = expenses.groupby(expenses.Date.astype(str).str[:7]).amount.sum()
    exp_ytd = float(expenses.amount.sum())
    if abs(exp_ytd - total_out) > .02:
        # The two measures can differ legitimately in other universes, but
        # this POC needs explicit reconciliation to publish a comparable view.
        raise ValueError("HH recognized spending and treasury gross outflows differ; refine accounting semantics before publishing")
    actor_annual = [
        (known.get(actor,actor),
         float(actor_totals.get((actor,baseline_year),0)),
         float(actor_totals.get((actor,year),0)))
        for actor in actors
    ]
    if abs(sum(row[2] for row in actor_annual)-total_in)>0.02:
        raise ValueError("HH recognized contributions and treasury inflows differ; do not imply all inflows are contributions")
    if not actor_annual:
        raise ValueError("Empty HH actor contribution table")
    labels=dict(zip(months,month_labels))
    closing_m=(max(0.,-closing))
    movement_rows=tuple(
        (labels[p],ars(cash_year.at[p,"opening_control"]),
         ars(cash_year.at[p,"net_cash_flow"]),ars(cash_year.at[p,"closing_control"]))
        for p in months
    )
    contribution_rows=tuple(
        (actor,ars(prior),ars(current),ars(prior+current))
        for actor,prior,current in actor_annual
    )
    out_rows=tuple(
        (title,ars(value),f"{(value/total_out*100):.1f} %")
        for title,value in sorted(group_out,key=lambda t:-t[1])
    )
    monthly_expense_rows=tuple(
        (labels[p],ars(float(expense_by_month.get(p,0))))
        for p in months
    )
    flow_rows=tuple(
        (labels[p],ars(cash_year.at[p,"total_cash_in"]),
         ars(cash_year.at[p,"total_cash_out"]),ars(cash_year.at[p,"net_cash_flow"]))
        for p in months
    )
    prior_mi=float(actor_totals.get(("MI",baseline_year),0))
    current_ca=float(actor_totals.get(("CA",year),0))
    biggest_idx=max(months,key=lambda p:float(expense_by_month.get(p,0)))
    ending_text=(f"un déficit contable de {ars(closing_m)}" if closing<0
                 else f"un saldo contable no negativo de {ars(closing)}")
    return {
        "virtual_balance":ReportContent(
            "virtual_balance","01 · Estado de la caja virtual",
            "¿Cuánto acumula la casa como posición contable y cómo llegó a ese resultado?",
            f"Al cierre de {months[-1]}, la caja virtual de Balbín muestra {ending_text}. Partió de "
            f"{ars(opening)} en enero; entre enero y octubre registró un flujo neto de {ars(net)}.",
            (("Saldo inicial 2026",ars(opening)),("Resultado del período",ars(net)),
             ("Saldo al corte",ars(closing)),("Déficit virtual (magnitud)",ars(closing_m))),
            ("Mes","Saldo inicial","Resultado del mes","Saldo final"),movement_rows,
            ("El saldo es una cuenta de control teórica: se registran gastos aun cuando Matías los pagó desde sus medios personales.",
             "El flujo mensual enlaza exactamente el saldo inicial con el saldo al corte.",
             "Para una deuda individual hacen falta pruebas y reglas adicionales; este gráfico no adjudica ese pasivo."),
            "El saldo negativo NO equivale a deuda personal exigible a favor de Matías, ni a descubierto bancario, ni a efectivo faltante. No se ajusta por inflación.",
            "monthly_cash_accountability.csv (ARS, cierre mensual)"),
        "contributions":ReportContent(
            "contributions","02 · Aportes de los integrantes",
            "¿Quién contribuyó a la administración de Balbín en 2025 y en 2026?",
            f"El ledger reconoce {ars(total_in)} de aportes a la caja HH entre enero y octubre de {year}. "
            f"Los {ars(prior_mi)} de Matías en {baseline_year} son aportes virtuales por criterio de registro, "
            "no comprobantes de depósitos bancarios independientes.",
            (("Aportes registrados 2026",ars(total_in)),("Matías virtual 2025",ars(prior_mi)),
             ("Claudia al corte",ars(current_ca))),
            ("Aportante",baseline_year,f"{year} hasta octubre","Total nominal (dos períodos)"),
            contribution_rows,
            ("Aporte reconocido no identifica quién ejecutó cada pago de servicio.",
             "Alejandro aportó mediante distribuciones de su cuenta personal: no duplicar esas transferencias como ingresos de otras cajas en Household.",
             "El cierre de octubre incluye el aporte de Claudia de agosto; el de noviembre queda fuera de este período."),
            "Comparar pesos nominales de 2025 contra 2026 no mide poder adquisitivo. Candela también pudo pagar gastos directamente fuera de la tabla de aportes; esa conciliación continúa pendiente.",
            "classification_audit.csv (solo funding_contribution, ARS)"),
        "outflows":ReportContent(
            "outflows","03 · Aplicación de los recursos",
            "¿En qué conceptos se registraron salidas de la casa hasta octubre?",
            f"Los egresos clasificados de enero a octubre suman {ars(total_out)}. "
            "Esta agrupación describe cómo fueron contabilizados, no decide automáticamente si deben repartirse entre residentes o propietarios.",
            (("Egresos registrados",ars(total_out)),("Categorías con movimientos",str(len(group_out)))),
            ("Concepto de salida","Importe","Participación"),out_rows,
            ("Los servicios incluyen pagos directos y transferencias de HH a PM cuando corresponden al uso de la vivienda.",
             "Las partidas de mantenimiento deben diferenciar bienes compartidos (perro, plantas, Macro) de reparaciones estructurales.",
             "No asignar aquí de oficio un tercio de ABL ni costos de reparaciones todavía no aprobadas."),
            "Estas categorías son semánticas, no una liquidación de expensas comunes ni distribución jurídica de cargas. Algunos meses incluyen facturas correspondientes a períodos distintos del pago.",
            "monthly_cash_accountability.csv (componentes de egresos)"),
        "monthly_costs":ReportContent(
            "monthly_costs","04 · Evolución del gasto",
            "¿Cuánto se desembolsó o reconoció como costo en cada mes?",
            f"En el período se reconocieron {ars(exp_ytd)} en {len(expenses)} movimientos de gasto; "
            f"el mes de mayor importe fue {labels[biggest_idx]} con {ars(float(expense_by_month.get(biggest_idx,0)))}.",
            (("Gasto reconocido 2026",ars(exp_ytd)),("Movimientos de gasto",str(len(expenses))),
             ("Mes de mayor gasto",labels[biggest_idx])),
            ("Mes de asiento","Gasto reconocido"),monthly_expense_rows,
            ("La evolución está agrupada por fecha del asiento, no por mes de consumo del servicio.",
             "Una factura de agua de noviembre pagada anticipadamente el 26 de octubre aparece en octubre: no representa consumo de noviembre dentro de la serie.",
             "Esta serie todavía no distingue costos corrientes compartibles de mantenimientos de propiedad con otra responsabilidad."),
            "No interpretar el total como cuota que cada integrante deba abonar. Antes de una distribución se deben revisar ABL, reparaciones edilicias y elegibilidad de los gastos.",
            "classification_audit.csv (property_opex, direction=out, ARS)"),
        "cash_flow":ReportContent(
            "cash_flow","05 · Flujo mensual de caja",
            "¿En qué meses alcanzaron los aportes y cuándo tuvo que financiarse la diferencia?",
            f"Entre enero y octubre se reconocieron {ars(total_in)} de entradas y {ars(total_out)} de salidas: "
            f"la diferencia fue {ars(net)}. Los meses con resultado positivo no compensan automáticamente obligaciones anteriores.",
            (("Entradas",ars(total_in)),("Salidas",ars(total_out)),("Flujo neto",ars(net))),
            ("Mes","Entradas","Salidas","Resultado"),flow_rows,
            ("El contraste es de la caja virtual HH, no de todas las cuentas personales de Matías.",
             "Un ingreso de Household puede ser dinero recibido por Matías como custodio, sin ingreso adicional a otra caja.",
             "Este flujo explica la variación del saldo acumulado, pero no atribuye automáticamente acreencias entre los residentes."),
            "Los movimientos reconocidos no certifican disponibilidad de efectivo, saldo de Mercado Pago ni obligación de reembolso individual. El orden usa fechas de asiento.",
            "monthly_cash_accountability.csv (total_cash_in, total_cash_out, net_cash_flow)"),
    }
