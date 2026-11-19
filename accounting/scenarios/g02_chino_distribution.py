from __future__ import annotations

"""G02: isolated, declarative Chino Juan distribution schedule and liquidity stress.

The scenario OVERLAY is not the canonical ledger, a bank statement, a valid
agreement, a debt authorization or a change to historical paid transactions.
Only explicit future/current months can create planned entries. Skipped months
never accrue an obligation.
"""

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from accounting.box_cash import box_party_match_masks
from accounting.reports.time_series import (
    TIMESERIES_CSS, TimeSeriesSpec, render_timeseries_svg, timeseries_figure,
)

SCHEMA = "accounting.g02_chino_distribution.v1"
MONTH = re.compile(r"20\d{2}-(0[1-9]|1[0-2])$")
PAID = ("2026-05", "2026-06", "2026-07", "2026-10")
UNPAID = ("2026-08", "2026-09")
PM = "Property Management"
HH = "Household"
Q = Decimal("0.01")


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not an amount")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("nonfinite or negative amount/rate")
    return result


def _money(value: Decimal) -> Decimal:
    return value.quantize(Q, rounding=ROUND_HALF_UP)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate(spec: dict, *, base: Path, monthly: Path) -> None:
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError("wrong G02 distribution YAML schema")
    if set(spec) != {
        "schema", "id", "source", "as_of_date", "through_month",
        "historical_paid_months", "historical_skipped_no_accrual",
        "no_retroactive", "agreement", "historical_settlement",
        "ars_equivalent_rates_per_usd", "variants",
    }:
        raise ValueError("unknown or missing G02 definition fields")
    if not re.fullmatch(r"\d{8}T\d{6}Z_FBPM", str(spec["source"].get("run_id", ""))):
        raise ValueError("G02 source run ID must be an exact pinned FBPM run")
    if spec["source"].get("run_id") != base.name or _sha(base / "ledger_canonical_all_status.csv") != spec["source"].get("all_status_sha256"):
        raise ValueError("G02 source ledger hash does not match pinned run")
    if _sha(monthly) != spec["source"].get("monthly_policy_inputs_sha256"):
        raise ValueError("G02 monthly forecast is not the pinned version")
    cutoff = date.fromisoformat(spec["as_of_date"])
    start = cutoff.isoformat()[:7]
    end = spec["through_month"]
    if not MONTH.fullmatch(end) or end < start or end > "2028-12":
        raise ValueError("invalid G02 forecast horizon")
    if spec.get("no_retroactive") is not True:
        raise ValueError("G02 must explicitly prohibit retrospective payouts")
    if spec.get("historical_paid_months") != list(PAID) or spec.get("historical_skipped_no_accrual") != list(UNPAID):
        raise ValueError("G02 historical paid/unpaid contract changed")
    historical_mechanism = spec["historical_settlement"]
    if historical_mechanism != {
        "ars_equivalent_physical_funder": "Matias_MercadoPago",
        "ars_equivalent_accounting_charge_box": PM,
        "usd_reference_custodian": "Matias_for_PM",
        "usd_physically_sold": False,
        "reimbursement_right_assumed": False,
    }:
        raise ValueError("G02 historical settlement must preserve MP funding, PM charge and retained USD")
    a = spec["agreement"]
    expected = {
        "rent_tag_ars", "rent_tag_usd", "monthly_rent_ars", "monthly_rent_usd",
        "primos_base_ars", "primos_usd_equivalence_reference", "alejandro_to_hh_ars",
        "alejandro_debt_offset_usd", "payment_day", "future_physical_funder",
    }
    if set(a) != expected:
        raise ValueError("G02 agreement must declare the exact audited legs")
    if (a["rent_tag_ars"], a["rent_tag_usd"]) != (
        "PM_RENT_CABA_CHINO_ARS", "PM_RENT_CABA_CHINO_USD"
    ):
        raise ValueError("G02 scenario may not select an arbitrary rental population")
    amounts = {k: _decimal(a[k]) for k in expected if k not in {
        "rent_tag_ars", "rent_tag_usd", "payment_day", "future_physical_funder",
    }}
    if amounts["primos_base_ars"] + amounts["alejandro_to_hh_ars"] != amounts["monthly_rent_ars"]:
        raise ValueError("ARS halves do not reconcile to Chino rent")
    if amounts["primos_usd_equivalence_reference"] + amounts["alejandro_debt_offset_usd"] != amounts["monthly_rent_usd"]:
        raise ValueError("USD halves do not reconcile to Chino rent")
    if any(_money(x) != x for x in amounts.values()):
        raise ValueError("G02 money amounts must have no more than 2 decimals")
    if a["future_physical_funder"] != "unassigned_until_authorized":
        raise ValueError("G02 cannot assume personal salary or a payment account for future distributions")
    if type(a["payment_day"]) is not int or not 1 <= a["payment_day"] <= 28:
        raise ValueError("invalid G02 payment day")
    variants = spec["variants"]
    if not isinstance(variants, list) or not (2 <= len(variants) <= 5):
        raise ValueError("G02 needs 2-5 explicitly named schedules")
    if len({v.get("id") for v in variants}) != len(variants):
        raise ValueError("G02 variant IDs repeat")
    if not any(v.get("months") == [] for v in variants):
        raise ValueError("G02 requires a no-new-distribution baseline")
    rates = spec["ars_equivalent_rates_per_usd"]
    if not isinstance(rates, dict):
        raise ValueError("G02 ARS equivalence references must be a month mapping")
    for m, rate in rates.items():
        if not MONTH.fullmatch(str(m)) or not start <= m <= end or _decimal(rate) <= 0:
            raise ValueError("invalid explicit G02 ARS equivalence rate")
    for variant in variants:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,65}", str(variant.get("id"))):
            raise ValueError("invalid G02 variant id")
        if set(variant) != {"id", "months", "explanation"}:
            raise ValueError("G02 variant must only provide id, months, explanation")
        months = variant["months"]
        if not isinstance(months, list) or months != sorted(set(months)):
            raise ValueError("G02 months must be sorted, unique YYYY-MM")
        if any(not isinstance(m, str) or not MONTH.fullmatch(m) or not start <= m <= end for m in months):
            raise ValueError("G02 prohibits retroactive and outside-horizon months")
        if any(m not in rates for m in months):
            raise ValueError("missing explicit ARS equivalence rate for a selected month")
        if months and months[0] == start and a["payment_day"] <= cutoff.day:
            raise ValueError("current-month scenario payment must be strictly after cutoff")


def _history_bridge(raw: pd.DataFrame) -> pd.DataFrame:
    historical = []
    for month in sorted(set(PAID) | set(UNPAID)):
        x = raw.loc[raw["Date"].str[:7].eq(month) & raw["status"].eq("pagado") & raw["Box"].eq(PM)]
        def one(payer: str, receiver: str, currency: str, kind: str) -> float:
            m = x.loc[
                x["payer"].eq(payer) & x["receiver"].eq(receiver) &
                x["Currency"].eq(currency) & x["Tipo"].eq(kind)
            ]
            if len(m) > 1:
                raise ValueError(f"duplicate historical G02 leg {month}/{payer}/{receiver}")
            return float(m["amount"].iloc[0]) if len(m) else 0.0

        primos = one("PM", "Primos", "ARS", "Dividendo")
        father_ars = one("PM", "Alejandro", "ARS", "Dividendo")
        hh_ars = one("Alejandro", "HH", "ARS", "Contribuciones")
        father_usd = one("PM", "Alejandro", "USD", "Dividendo")
        repay_usd = one("Alejandro", "PM", "USD", "Repago")
        paid_month = month in PAID
        if paid_month and not (
            primos > 200000 and father_ars == hh_ars == 200000 and
            father_usd == repay_usd == 190
        ):
            raise ValueError(f"historical G02 legs do not match known paid month {month}")
        if not paid_month and any((primos, father_ars, hh_ars, father_usd, repay_usd)):
            raise ValueError(f"cannot mark {month} unpaid: a paid G02 leg exists")
        # Earlier FX rows were excluded (X) and are not real sales: the
        # USD remained in Matias's custody, while his own Mercado Pago ARS
        # financed the USD-equivalent part paid to primos, charged ONCE to PM.
        fx = raw.loc[raw["Date"].str[:7].eq(month) & raw["Box"].eq(PM) &
                     raw["Flujo"].eq("Cambio") & raw["Tipo"].eq("FX") &
                     raw["status"].eq("X")]
        fx_received_ars = float(fx.loc[fx["receiver"].eq("PM") & fx["Currency"].eq("ARS"), "amount"].sum())
        implied_ars = primos - 200000 if paid_month else 0.0
        historical.append({
            "period": month,
            "recorded_distribution": paid_month,
            "accrues_in_scenario": False,
            "primos_received_ars": primos,
            "alejandro_assigned_ars": father_ars,
            "hh_reported_receipt_ars": hh_ars,
            "alejandro_usd_assigned": father_usd,
            "alejandro_usd_debt_repaid": repay_usd,
            "primos_usd_equivalent_paid_personally_ars": implied_ars,
            "physical_funder_of_usd_equivalent_ars": "Matias_MercadoPago" if paid_month else "none",
            "charge_to_accounting_box": PM if paid_month else "none",
            "payment_already_charged_to_pm": paid_month,
            "usd_reference_remains_in_matias_custody": 190.0 if paid_month else 0.0,
            "actual_fx_usd_sold": 0.0,
            "legacy_excluded_fx_stub_ars": fx_received_ars,
            "legacy_fx_stub_mismatch_ars": round(implied_ars - fx_received_ars, 2) if paid_month else None,
            "legacy_fx_stub_status_X": bool(len(fx)),
            "ars_equivalence_per_usd_not_a_trade": round(implied_ars / 190, 4) if paid_month else None,
        })
    return pd.DataFrame(historical)


def _rent_guard(raw: pd.DataFrame, months: list[str], agreement: dict) -> None:
    for month in months:
        for cur in ("ARS", "USD"):
            tag = agreement[f"rent_tag_{cur.lower()}"]
            expected = float(agreement[f"monthly_rent_{cur.lower()}"])
            matches = raw.loc[
                raw["Date"].str[:7].eq(month) & raw["tag"].eq(tag) &
                raw["Box"].eq(PM)
            ]
            if len(matches) != 1 or float(matches["amount"].iloc[0]) != expected:
                raise ValueError(f"Chino rent incomplete for {month}/{cur}: refuse distribution assumption")
            if matches["status"].iloc[0] not in {"pagado", "planeado"}:
                raise ValueError("invalid Chino rent source status")


def _overlay_rows(variant: str, months: list[str], a: dict, rates: dict, fields: list[str]) -> pd.DataFrame:
    """Future PM accounting charges, HH mirror, and USD debt-settlement only.

    No USD sale and no ARS inflow to PM: an ARS amount equivalent to
    USD 190 is charged to PM and paid to primos in ARS. Future physical
    disbursement/funding channel remains unassigned until authorized.
    """
    results = []
    def add(month: str, step: int, *, box: str, currency: str, value: Decimal,
            payer: str, receiver: str, flujo: str, tipo: str, notes: str) -> None:
        if value <= 0:
            raise ValueError("G02 overlay leg cannot be zero/negative")
        record = dict.fromkeys(fields, "")
        record.update({
            "tx_id": sha256(f"G02:{variant}:{month}:{step}:{box}".encode()).hexdigest()[:16],
            "Date": f"{month}-{a['payment_day']:02d}",
            "amount": str(value),
            "amount_cents": str(int(value * 100)),
            "Currency": currency, "payer": payer, "receiver": receiver,
            "Flujo": flujo, "Tipo": tipo, "Box": box,
            "Lugar": "CABA" if box == PM else "",
            "Detalle": f"G02 hypothetical PM-charged Chino distribution {month}",
            "notes": notes,
            "tag": f"G02_CHINO_{variant.upper()}_{month.replace('-', '')}_{step}",
            "status": "planeado", "source_file": "scenario://g02_chino_distribution_schedule.yaml",
            "source_row": str(step),
            "accounting_charge_box": PM if box == PM else HH,
            "physical_payment_source": "NOT_SPECIFIED_FUTURE",
            "usd_sale_executed": False,
        })
        results.append(record)
    for month in months:
        usd_equivalent = _money(_decimal(a["primos_usd_equivalence_reference"]))
        father_usd = _money(_decimal(a["alejandro_debt_offset_usd"]))
        primos_ars = _money(_decimal(a["primos_base_ars"]))
        father_ars = _money(_decimal(a["alejandro_to_hh_ars"]))
        reference_rate = _decimal(rates[month])
        ars_equivalent = _money(usd_equivalent * reference_rate)
        add(month, 1, box=PM, currency="ARS", value=_money(primos_ars + ars_equivalent),
            payer="PM", receiver="Primos", flujo="Transfer", tipo="Dividendo",
            notes=("Primos receive ARS ONLY. USD 190 determines the ARS reference "
                   "value; no USD is sold. PM balance charged; future physical payer unspecified."))
        add(month, 2, box=PM, currency="ARS", value=father_ars,
            payer="PM", receiver="Alejandro", flujo="Transfer", tipo="Dividendo",
            notes="Alejandro ARS allocation charged to PM, destined to HH")
        add(month, 3, box=PM, currency="ARS", value=father_ars,
            payer="Alejandro", receiver="HH", flujo="Transfer", tipo="Contribuciones",
            notes="Reallocation of existing PM debit to HH, not a second PM outflow")
        add(month, 4, box=HH, currency="ARS", value=father_ars,
            payer="Alejandro", receiver="HH", flujo="Transfer", tipo="Contribuciones",
            notes="HH analytic scope mirror, not another new deposit from PM")
        add(month, 5, box=PM, currency="USD", value=father_usd,
            payer="PM", receiver="Alejandro", flujo="Transfer", tipo="Dividendo",
            notes="USD 190 assigned to Alejandro for PM debt offset, NOT an FX sale")
        add(month, 6, box=PM, currency="USD", value=father_usd,
            payer="Alejandro", receiver="PM", flujo="Repago", tipo="Repago",
            notes="Assigned USD 190 netted against PM debt; no incremental PM USD flow")
    return pd.DataFrame(results, columns=[*fields, "accounting_charge_box", "physical_payment_source", "usd_sale_executed"])


def _overlay_balance_effects(overlay: pd.DataFrame) -> pd.DataFrame:
    if overlay.empty:
        return pd.DataFrame(columns=["period", "Box", "Currency", "scenario_in", "scenario_out", "scenario_net"])
    # Party matching is reused as the ledger's *economic* in/out convention.
    # It does NOT claim the physical payment was made from a PM bank account.
    inflow, outflow = box_party_match_masks(overlay)
    work = overlay.copy()
    work["period"] = work["Date"].astype(str).str[:7]
    work["scenario_in"] = work["amount"].astype(float).where(inflow, 0)
    work["scenario_out"] = work["amount"].astype(float).where(outflow, 0)
    work["scenario_net"] = work["scenario_in"] - work["scenario_out"]
    return work.groupby(["period", "Box", "Currency"], as_index=False)[
        ["scenario_in", "scenario_out", "scenario_net"]
    ].sum()


def _monthly_projection(features: pd.DataFrame, effects: pd.DataFrame, as_of: str, through: str) -> pd.DataFrame:
    """Apply only incremental *accounting* effects, never reconstruct fake FX."""
    start = as_of[:7]
    months = pd.period_range(start, through, freq="M").astype(str).tolist()
    rows = []
    for box, currency in [(PM, "ARS"), (PM, "USD"), (HH, "ARS")]:
        cumul = 0.0
        for month in months:
            d = effects.loc[(effects["period"].eq(month)) &
                            effects["Box"].eq(box) & effects["Currency"].eq(currency)]
            change = float(d["scenario_net"].sum()) if not d.empty else 0.0
            incoming = float(d["scenario_in"].sum()) if not d.empty else 0.0
            outgoing = float(d["scenario_out"].sum()) if not d.empty else 0.0
            cumul += change
            base = features.loc[
                features["period"].eq(month) & features["Box"].eq(box) &
                features["Currency"].eq(currency)
            ]
            if box != HH and len(base) != 1:
                raise ValueError(f"monthly accounting source missing {box}/{currency}/{month}")
            observed = box != HH and bool(base.iloc[0]["materialized_month"])
            base_control = float(base.iloc[0]["closing_control"]) if observed else None
            base_net = float(base.iloc[0]["net_cash_flow"]) if observed else None
            accounting_control = round(base_control + cumul, 2) if observed else None
            rows.append({
                "period": month, "Box": box, "Currency": currency,
                "basis": "HH_analytic_receiver_no_cash_opening" if box == HH else "PM_zero_origin_accounting_control_not_verified_custody",
                "base_available": observed if box != HH else False,
                "base_control": base_control,
                "base_monthly_net": base_net,
                "scenario_in": round(incoming, 2),
                "scenario_out": round(outgoing, 2),
                "scenario_net": round(change, 2),
                "scenario_cumulative_delta": round(cumul, 2),
                "scenario_control": accounting_control,
                "negative_control": bool(accounting_control < -0.005) if accounting_control is not None else None,
            })
    return pd.DataFrame(rows)


def _display_chart(results: dict[str, pd.DataFrame], months: list[str],
                   history: pd.DataFrame, equivalence_rates: dict) -> str:
    """Private decision aid: PM accounting and physical custody are distinct."""
    def fmt(v) -> str:
        return "sin plan" if pd.isna(v) else f"{float(v):,.0f}"

    history_rows = "".join(
        f'<tr><td>{r.period}</td>'
        f'<td>{"Pagado" if r.recorded_distribution else "Sin pago; no se acumula"}</td>'
        f'<td>{fmt(r.primos_received_ars)}</td>'
        f'<td>{fmt(r.primos_usd_equivalent_paid_personally_ars)}</td>'
        f'<td>{fmt(r.usd_reference_remains_in_matias_custody)}</td>'
        f'<td>{fmt(r.hh_reported_receipt_ars)}</td></tr>'
        for r in history.itertuples(index=False)
    )
    paid = history.loc[history["recorded_distribution"]]
    personally_paid = float(paid["primos_usd_equivalent_paid_personally_ars"].sum())
    custodied_usd = float(paid["usd_reference_remains_in_matias_custody"].sum())
    history_section = (
        '<section><h2>Historia: pago personal ≠ venta de USD ≠ nuevo ingreso PM</h2>'
        f'<p>Según declaración del custodio: <strong>ARS {fmt(personally_paid)}</strong> '
        'salieron de Mercado Pago de Matías y se imputaron al balance de PM '
        '<strong>una sola vez</strong> a través de las distribuciones en ARS ya pagadas. '
        f'<strong>USD {fmt(custodied_usd)}</strong> siguen bajo su custodia. '
        'No existe venta de USD ni ingreso de ARS a PM por una operación de FX.</p>'
        '<table><thead><tr><th>Mes</th><th>Estado</th><th>Primos ARS recibido</th>'
        '<th>Parte pagada desde Mercado Pago ARS</th><th>USD retenidos</th>'
        '<th>HH ARS informado</th></tr></thead><tbody>' + history_rows + '</tbody></table>'
        '<p>La fuente contable identifica el pago/imputación; el origen Mercado Pago '
        'y la custodia de USD son declaraciones del usuario pendientes de conciliación externa. '
        'Los antiguos asientos FX con estado X no son transacciones realizadas. '
        'No se genera automáticamente ninguna cuenta de reembolso personal.</p></section>'
    )
    blocks = []
    for variant, monthly in results.items():
        pm = monthly.loc[monthly["Box"].eq(PM)]
        charts, tables = [], []
        for currency in ("ARS", "USD"):
            sub = pm.loc[pm["Currency"].eq(currency)]
            points = [
                {
                    "period": row.period, "series_id": "pm_control",
                    "value": row.scenario_control, "unit": currency,
                    "scope": PM, "phase": "current" if row.period == months[0] else "projected",
                    "source_table": "monthly_policy_inputs.csv + scenario_accounting_overlay",
                }
                for row in sub.itertuples(index=False)
                if pd.notna(row.scenario_control)
            ]
            spec = TimeSeriesSpec(
                chart_id=f"g02_{variant}_{currency.lower()}",
                title=f"PM · control económico registrado en {currency} — {variant}",
                frequency="M", mark="line", unit=currency, scope=PM,
                series_ids=("pm_control",), series_labels=("PM · control contable",),
                subtitle="Pagos hipotéticos imputados a PM, sin operación FX",
                note=("Control de origen cero, no saldo bancario. No reconstruye "
                      "un flujo físico desde Mercado Pago; abril está sin plan."),
                period_domain=tuple(months), max_points=12,
            )
            svg, _ = render_timeseries_svg(spec, pd.DataFrame(points), height=230)
            charts.append(timeseries_figure(spec, svg))
            body = "".join(
                f'<tr class="{"negative" if pd.notna(row.scenario_control) and row.scenario_control < 0 else ""}">'
                f'<td>{row.period}</td><td>{fmt(row.base_control)}</td>'
                f'<td>{fmt(row.scenario_net)}</td><td>{fmt(row.scenario_cumulative_delta)}</td>'
                f'<td>{fmt(row.scenario_control)}</td></tr>'
                for row in sub.itertuples(index=False)
            )
            tables.append(
                f'<h3>PM · posición mensual y neto acumulado en {currency}</h3>'
                '<table><thead><tr><th>Mes</th><th>Control anterior</th>'
                '<th>Distribución del mes</th><th>Distribuciones acumuladas</th>'
                '<th>Control resultante</th></tr></thead><tbody>'
                + body + '</tbody></table>'
            )
        hh = monthly.loc[monthly["Box"].eq(HH) & monthly["Currency"].eq("ARS")]
        hh_rows = "".join(
            f'<tr><td>{r.period}</td><td>{fmt(r.scenario_in)}</td>'
            f'<td>{fmt(r.scenario_cumulative_delta)}</td></tr>'
            for r in hh.itertuples(index=False)
        )
        blocks.append(
            f'<section><h2>{escape(variant)}</h2>' + "".join(charts)
            + "".join(tables)
            + '<h3>HH · imputación analítica de aportes de Alejandro (ARS)</h3>'
            + '<table><thead><tr><th>Mes</th><th>Ingreso atribuido</th>'
            '<th>Total atribuido en el escenario</th></tr></thead><tbody>'
            + hh_rows + '</tbody></table>'
            '<p>La fuente física de los pagos futuros queda sin asignar. '
            'HH no tiene saldo de apertura conciliado.</p></section>'
        )
    rate_note = ", ".join(f"{m}: {float(rate):,.0f}" for m, rate in equivalence_rates.items())
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        '<title>G02 · políticas de distribución Chino</title>'
        '<style>body{font:14px system-ui;background:#f3f6fa;color:#20334a}'
        'main{max-width:1090px;margin:auto;background:white;padding:28px}'
        'section{border-top:2px solid #d4dfec;margin:24px 0;padding:13px 0}'
        'h1,h2{color:#18395d}h2{font-size:19px}table{border-collapse:collapse;width:100%;font-size:12px}'
        'td,th{border-bottom:1px solid #dbe2eb;padding:7px;text-align:right}'
        'td:first-child,th:first-child{text-align:left}.negative{background:#ffe5df;font-weight:bold}'
        'p{color:#4a607b;font-size:12px}section .ts-figure{break-inside:avoid}'
        '@media print{@page{size:A4 landscape;margin:12mm}section{break-before:page}}'
        + TIMESERIES_CSS + '</style></head><body><main>'
        '<h1>G02 · distribución de renta y sostenibilidad</h1>'
        '<p><strong>No hay FX físico.</strong> Los primos recibieron ARS; '
        'la parte expresada en USD sólo fija su equivalente en pesos. '
        'Los dólares permanecieron en custodia de Matías y el pago personal '
        'en ARS quedó cargado al balance de PM, sin acreditarse un ingreso FX ficticio. '
        'Mayo, junio, julio y octubre pagados. Agosto/septiembre sin retroactivos.</p>'
        '<p>Las filas prospectivas son hipótesis de distribución contra el saldo '
        'económico PM; no constituyen desembolsos, deuda exigible, saldos de banco '
        'ni compromiso de financiación con sueldo o Mercado Pago. Abril 2027 queda sin planes y sin saldo proyectado.</p>'
        f'<p>Referencias ARS por USD (no transacciones de cambio): {escape(rate_note)}. '
        'La compensación de deuda USD de Alejandro se refleja con dos asientos '
        'que se neutralizan en caja PM; no recalculamos la posición del motor de deuda.</p>'
        + history_section + "".join(blocks) + '</main></body></html>'
    )


def run_scenarios(*, definition: Path, repo: Path, monthly_source: Path, out: Path) -> dict:
    spec = yaml.safe_load(Path(definition).read_text())
    base = Path(repo) / "out" / "run" / "accounting" / spec["source"]["run_id"]
    monthly_path = Path(monthly_source)
    _validate(spec, base=base, monthly=monthly_path)
    out = Path(out).resolve()
    repo_path = Path(repo).resolve()
    if any(out.is_relative_to(root) for root in (
        repo_path / "out", repo_path / "public", repo_path / "scenarios",
    )):
        raise ValueError("G02 results must be created outside production and code folders")
    if out.exists():
        raise FileExistsError(f"refuse to overwrite existing G02 results: {out}")
    base_raw = pd.read_csv(base / "ledger_canonical_all_status.csv", low_memory=False)
    required = {"tx_id", "Date", "amount", "Currency", "status", "Box", "payer", "receiver", "Flujo", "Tipo", "tag"}
    if required - set(base_raw):
        raise ValueError("G02 source is missing canonical ledger fields")
    history = _history_bridge(base_raw)
    # Historical personal funding is already charged via paid PM ARS rows.
    # Do not add ARS to PM; USD 760 was held in Matias's custody, not sold.
    paid_history = history.loc[history["recorded_distribution"]]
    personal_ars = round(float(paid_history["primos_usd_equivalent_paid_personally_ars"].sum()), 2)
    retained_usd = round(float(paid_history["usd_reference_remains_in_matias_custody"].sum()), 2)
    input_rows = pd.read_csv(monthly_path, low_memory=False)
    months = pd.period_range(spec["as_of_date"][:7], spec["through_month"], freq="M").astype(str).tolist()
    by_variant = {}
    by_overlay = {}
    by_effects = {}
    summary = {"schema": SCHEMA, "base_run_id": base.name,
               "source_all_status_sha256": spec["source"]["all_status_sha256"],
               "source_monthly_sha256": spec["source"]["monthly_policy_inputs_sha256"],
               "historical_paid_months": list(PAID),
               "historical_unpaid_no_accrual": list(UNPAID),
               "no_retroactive": True, "as_of_date": spec["as_of_date"],
               "through_month": spec["through_month"],
               "historical_settlement": {
                   "personal_mercadopago_ars_paid": personal_ars,
                   "already_charged_to_PM_ARS": True,
                   "additional_pm_ars_credit": 0.0,
                   "usd_held_by_matias_for_pm": retained_usd,
                   "usd_sold": 0.0,
                   "fx_trade_executed": False,
                   "new_reimbursement_claim_created": False,
                   "provenance": "paid PM ARS ledger rows + user-confirmed physical payer and USD custody",
                   "bank_statement_verified": False,
               }, "variants": {}}
    for variant in spec["variants"]:
        ident, schedule = variant["id"], variant["months"]
        _rent_guard(base_raw, schedule, spec["agreement"])
        overlay = _overlay_rows(
            ident, schedule, spec["agreement"],
            spec["ars_equivalent_rates_per_usd"], list(base_raw.columns),
        )
        if overlay["tx_id"].duplicated().any() or set(overlay["tx_id"]) & set(base_raw["tx_id"]):
            raise ValueError("G02 synthetic ledger identities collide")
        effects = _overlay_balance_effects(overlay)
        month_values = _monthly_projection(
            input_rows, effects, spec["as_of_date"], spec["through_month"]
        )
        # The policy is a monthly proposal, never a due-backlog mechanism.
        if len(overlay) != len(schedule) * 6:
            raise ValueError("incorrect G02 journal template size")
        for month in schedule:
            delta = effects.loc[effects["period"].eq(month)]
            def net(box: str, cur: str) -> float:
                return float(delta.loc[delta["Box"].eq(box) & delta["Currency"].eq(cur), "scenario_net"].sum())
            ars_equivalent = float(_money(_decimal(spec["agreement"]["primos_usd_equivalence_reference"]) *
                                          _decimal(spec["ars_equivalent_rates_per_usd"][month])))
            expected_ars = float(spec["agreement"]["monthly_rent_ars"]) + ars_equivalent
            if (abs(net(PM, "ARS") + expected_ars) > .005 or
                abs(net(PM, "USD")) > .005 or
                abs(net(HH, "ARS") - float(spec["agreement"]["alejandro_to_hh_ars"])) > .005):
                raise ValueError("G02 overlay fails PM ARS charge / retained USD / HH reconciliation")
            if overlay.loc[overlay["Date"].str[:7].eq(month), "usd_sale_executed"].any():
                raise ValueError("FX sale not allowed by G02 settlement mechanism")
        pm_ars = month_values.loc[
            month_values["Box"].eq(PM) & month_values["Currency"].eq("ARS")
        ].copy()
        valid = pm_ars.loc[pm_ars["scenario_control"].notna()]
        negative = valid.loc[valid["scenario_control"].lt(-.005)]
        by_variant[ident], by_overlay[ident], by_effects[ident] = month_values, overlay, effects
        summary["variants"][ident] = {
            "months": schedule, "future_distribution_count": len(schedule),
            "first_negative_pm_ars_month": negative["period"].iloc[0] if len(negative) else None,
            "minimum_pm_ars_control": round(float(valid["scenario_control"].min()), 2),
            "required_buffer_to_avoid_negative_pm_ars_control": round(
                max(0.0, -float(valid["scenario_control"].min())), 2
            ),
            "hh_ars_new_receipts": round(
                float(month_values.loc[month_values["Box"].eq(HH), "scenario_in"].sum()), 2
            ),
            "alejandro_usd_debt_offset_recorded_in_scenario": round(
                len(schedule) * float(spec["agreement"]["alejandro_debt_offset_usd"]), 2
            ),
            "debt_position_recomputed_by_canonical_debt_engine": False,
            "pm_ars_accounting_charge_in_selected_months": round(
                float(month_values.loc[month_values["Box"].eq(PM) &
                                       month_values["Currency"].eq("ARS"), "scenario_out"].sum()), 2
            ),
            "usd_pm_net_delta_for_reference_and_debt_offset": 0.0,
            "physical_funding_of_future_payments": "unassigned_until_authorized",
            "funding_from_salary_assumed": False,
            "actual_bank_cash_verified": False,
        }
    out.mkdir(parents=True)
    history.to_csv(out / "historical_distribution_bridge.csv", index=False)
    # This is a custody/provenance sidecar, NOT virtual FX ledger rows.
    pd.DataFrame([
        {
            "period": r.period,
            "personal_mercadopago_ars_out": float(r.primos_usd_equivalent_paid_personally_ars),
            "pm_accounting_charge_already_booked_ars": float(r.primos_usd_equivalent_paid_personally_ars),
            "pm_new_ars_credit": 0.0,
            "usd_retained_in_matias_custody": float(r.usd_reference_remains_in_matias_custody),
            "usd_sold": 0.0,
            "payer_provenance": "user_statement",
            "accounting_provenance": "paid PM->Primos ARS dividend ledger row",
            "actual_mercadopago_statement_reconciled": False,
        }
        for r in paid_history.itertuples(index=False)
    ]).to_csv(out / "historical_personal_payment_and_usd_custody.csv", index=False)
    for ident in by_variant:
        var_dir = out / ident
        var_dir.mkdir()
        by_overlay[ident].to_csv(var_dir / "scenario_overlay_ledger.csv", index=False)
        by_effects[ident].to_csv(var_dir / "scenario_accounting_movements.csv", index=False)
        by_variant[ident].to_csv(var_dir / "monthly_controls_by_box_currency.csv", index=False)
        # Include HH past reported receipts as evidence, while future HH rows remain PLANNED.
        past_hh = history[["period", "hh_reported_receipt_ars"]].rename(
            columns={"hh_reported_receipt_ars": "amount_ars"}
        )
        past_hh["basis"] = "historical_PM_receiver_HH_reported_paid"
        future_hh = by_variant[ident].loc[by_variant[ident]["Box"].eq(HH),
            ["period", "scenario_in"]].rename(columns={"scenario_in": "amount_ars"})
        future_hh["basis"] = "scenario_HH_box_counterparty_mirror_not_paid"
        pd.concat([past_hh, future_hh], ignore_index=True).to_csv(
            var_dir / "hh_receipts_analytic.csv", index=False
        )
    (out / "comparison.html").write_text(_display_chart(by_variant, months, history, spec["ars_equivalent_rates_per_usd"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", required=True, type=Path)
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--monthly-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run_scenarios(definition=args.definition, repo=args.repo,
                           monthly_source=args.monthly_source, out=args.out)
    print(json.dumps(result["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
