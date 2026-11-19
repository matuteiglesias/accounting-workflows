from __future__ import annotations

"""Phase-A scenario inputs: stable monthly features, never policy decisions.

Consumes the same transaction detail and monthly treasury output as the official
accounting backend. It does not modify the canonical ledger or issue payouts.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


COST_CATEGORIES = ("taxes", "services", "maintenance", "legal")
SIDE_NAMES = ("primos", "alejandro")
AGREEMENT_SCHEMA = "accounting.agreement_attribution.v0"
OUTPUT_SCHEMA = "accounting.policy_inputs.v1"


def _load_csv(path: Path, fields: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(fields - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns {missing}")
    return frame


def _load_agreement(path: Path | None) -> dict:
    if path is None:
        return {
            "schema": AGREEMENT_SCHEMA, "id": "primos_alejandro_50_50",
            "box": "Family Business", "currency": "ARS",
            "excluded_rent_tags": ["FB_RENT_TIGRE32_TAIPE"],
            "rent_attribution": [],
            "allowed_expense_types": ["Impuestos", "Servicio", "Mantenimiento", "Legal"],
            "expense_boxes": ["Family Business"],
        }
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("schema") != AGREEMENT_SCHEMA:
        raise ValueError("invalid agreement schema")
    required = {"id", "box", "currency", "excluded_rent_tags", "rent_attribution",
                "allowed_expense_types", "expense_boxes"}
    if set(obj) != required | {"schema"}:
        raise ValueError("agreement requires exact, documented fields")
    if not isinstance(obj["rent_attribution"], list):
        raise ValueError("rent_attribution must be a list")
    if not isinstance(obj["excluded_rent_tags"], list) or not all(
        isinstance(tag, str) and tag for tag in obj["excluded_rent_tags"]
    ):
        raise ValueError("invalid excluded rent tags")
    if len(obj["excluded_rent_tags"]) != len(set(obj["excluded_rent_tags"])):
        raise ValueError("repeated excluded rent tag")
    if not isinstance(obj["allowed_expense_types"], list) or not obj["allowed_expense_types"]:
        raise ValueError("invalid expense types")
    if not isinstance(obj["expense_boxes"], list) or not obj["expense_boxes"]:
        raise ValueError("invalid expense boxes")
    selectors = {"tag", "Lugar", "Detalle", "transaction_id"}
    for rule in obj["rent_attribution"]:
        if set(rule) != {"shares", "equals"}:
            raise ValueError("rent attribution requires shares and literal equals")
        shares = rule["shares"]
        if not isinstance(shares, dict) or set(shares) != set(SIDE_NAMES):
            raise ValueError("attribution must specify both settlement sides")
        if any(type(v) not in (float, int) or not (0 <= v <= 1) for v in shares.values()):
            raise ValueError("invalid rent attribution shares")
        if abs(sum(shares.values()) - 1) > 1e-8:
            raise ValueError("rent attribution shares must sum to one")
        if not isinstance(rule["equals"], dict) or not rule["equals"] or not set(rule["equals"]) <= selectors:
            raise ValueError("unsafe rent attribution selector")
        if not all(isinstance(v, (str, int)) for v in rule["equals"].values()):
            raise ValueError("rent attribution selectors must be literal values")
    return obj


def _round(v: float) -> float:
    return round(float(v), 2)


def _month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def build_policy_inputs(
    *, run_root: Path, out_dir: Path, as_of_date: str, through_date: str,
    start_month: str | None = None, agreement_path: Path | None = None,
) -> dict[str, Path]:
    """Compute Phase A from already-materialized, unmodified pre-policy inputs."""
    run_root, out_dir = Path(run_root), Path(out_dir)
    as_of, through = pd.Timestamp(as_of_date), pd.Timestamp(through_date)
    if as_of > through or as_of.year < 2000 or through.year > as_of.year + 3:
        raise ValueError("invalid policy-input interval")
    month_start = str(start_month or as_of.to_period("M"))
    months = _month_range(month_start, str(through.to_period("M")))
    if not months or months[0] > str(as_of.to_period("M")):
        raise ValueError("start_month cannot follow as_of month")
    agreement = _load_agreement(agreement_path)

    treasury = _load_csv(run_root / "monthly_cash_accountability.csv", {
        "period", "Box", "Currency", "rent_in", "net_cash_flow", "funding_cash_in",
        "taxes_out", "services_out", "maintenance_out", "legal_out",
        "personal_draws_out", "dividends_out",
    })
    detail = _load_csv(run_root / "box_treasury_transaction_detail.csv", {
        "tx_id", "Date", "period", "Box", "Currency", "cash_category", "movement_basis",
        "cash_direction", "amount_out", "review_required",
    })
    ledger = _load_csv(run_root / "ledger_canonical.csv", {
        "tx_id", "Date", "Box", "Currency", "status", "payer", "receiver",
        "Flujo", "Tipo", "amount", "tag", "Lugar", "Detalle", "source_row",
    })
    ledger["Date"] = pd.to_datetime(ledger["Date"], errors="raise")
    detail["Date"] = pd.to_datetime(detail["Date"], errors="raise")
    if ledger["tx_id"].isna().any() or ledger["tx_id"].duplicated().any():
        raise ValueError("canonical ledger needs unique nonempty tx_id")
    if detail["tx_id"].duplicated().any():
        raise ValueError("transaction detail must have one row per tx_id")
    if not ledger["status"].isin(("pagado", "planeado")).all():
        raise ValueError("policy inputs only accept paid history + explicit future plans")
    if (ledger["Date"].gt(as_of) & ledger["status"].eq("pagado")).any():
        raise ValueError("future paid transaction cannot be a policy input")
    if (ledger["Date"].le(as_of) & ledger["status"].eq("planeado")).any():
        raise ValueError("past-due plans require a distinct open-obligation model")
    if (ledger["Date"].gt(through)).any():
        raise ValueError("analytic ledger contains transactions past the declared horizon")
    ledger["period"] = ledger["Date"].dt.to_period("M").astype(str)
    ledger["amount"] = pd.to_numeric(ledger["amount"], errors="raise")
    if ledger["amount"].isna().any() or (ledger["amount"] < 0).any():
        raise ValueError("invalid ledger amounts")

    source_status = ledger.set_index("tx_id")["status"]
    detail["status"] = detail["tx_id"].map(source_status)
    if detail["status"].isna().any():
        raise ValueError("treasury detail has tx_id absent from analytic ledger")
    detail["amount_out"] = pd.to_numeric(detail["amount_out"], errors="raise")
    detail["review_required"] = detail["review_required"].astype(str).str.lower().isin(("true", "1", "yes"))
    due = detail.loc[
        detail["status"].eq("planeado") & detail["Date"].gt(as_of) &
        detail["Date"].le(through) & detail["movement_basis"].eq("actual_cash") &
        detail["cash_direction"].eq("out") & detail["cash_category"].isin(COST_CATEGORIES)
    ].copy()
    if due["review_required"].any():
        raise ValueError("planned cash costs require semantic classification review")
    due["amount_out"] = due["amount_out"].round(2)
    due_sum = due.groupby(["period", "Box", "Currency"], dropna=False)["amount_out"].sum().to_dict()
    due_ids = due.groupby(["period", "Box", "Currency"], dropna=False)["tx_id"].agg(list).to_dict()

    numeric = [
        "rent_in", "net_cash_flow", "funding_cash_in", "taxes_out", "services_out",
        "maintenance_out", "legal_out", "personal_draws_out", "dividends_out",
    ]
    for col in numeric:
        treasury[col] = pd.to_numeric(treasury[col], errors="raise")
    if treasury.duplicated(["period", "Box", "Currency"]).any():
        raise ValueError("monthly treasury has duplicate Box/currency/period")
    indexed = treasury.set_index(["period", "Box", "Currency"])
    boxes = sorted({(str(r.Box), str(r.Currency)) for r in treasury.itertuples(index=False)})
    if not boxes:
        raise ValueError("empty monthly treasury")
    month_last = str(through.to_period("M"))
    as_of_month = str(as_of.to_period("M"))
    rows = []
    source_rows = []

    for period in months:
        for box, currency in boxes:
            key = (period, box, currency)
            existing = key in indexed.index
            m = indexed.loc[key] if existing else pd.Series({c: 0.0 for c in numeric})
            operating_costs = sum(float(m[f"{category}_out"]) for category in COST_CATEGORIES)
            operating_surplus = float(m["rent_in"]) - operating_costs
            pre_distribution_net = float(m["net_cash_flow"]) + float(m["personal_draws_out"]) + float(m["dividends_out"])
            planned_costs = due_sum.get(key, 0.0)
            result = {
                "period": period, "Box": box, "Currency": currency,
                "temporal_status": ("observed" if period < as_of_month
                    else "partial_month" if period == as_of_month else "projected"),
                "materialized_month": bool(existing),
                "rent_in": _round(m["rent_in"]),
                "net_cash_flow": _round(m["net_cash_flow"]),
                "pre_distribution_cash_net": _round(pre_distribution_net),
                "pre_distribution_operating_surplus": _round(operating_surplus),
                "nonrent_funding_cash_in": _round(m["funding_cash_in"]),
                "actual_or_projected_operating_costs_out": _round(operating_costs),
                "planned_cash_operating_cost_due_this_month": _round(planned_costs),
                "planned_cost_tx_count": len(due_ids.get(key, [])),
            }
            for count in (1, 2, 3):
                targets = _month_range(str(pd.Period(period) + 1), str(pd.Period(period) + count))
                complete = period >= as_of_month and targets[-1] <= month_last
                result[f"cost_due_next_{count}m_complete"] = complete
                result[f"cost_due_next_{count}m"] = (
                    _round(sum(due_sum.get((p, box, currency), 0.0) for p in targets))
                    if complete else None
                )
                if complete:
                    running, trough = 0.0, 0.0
                    for future in targets:
                        future_key = (future, box, currency)
                        running += float(indexed.loc[future_key, "net_cash_flow"]) if future_key in indexed.index else 0.0
                        trough = min(trough, running)
                    result[f"cashflow_reserve_required_next_{count}m"] = _round(-trough)
                else:
                    result[f"cashflow_reserve_required_next_{count}m"] = None
            for basis, amount in (("operating_surplus", operating_surplus),
                                  ("cash_net", pre_distribution_net)):
                for fraction, label in ((1.0, "100pct"), (0.5, "50pct")):
                    result[f"distribution_{label}_{basis}"] = (
                        _round(max(0.0, amount) * fraction) if period >= as_of_month else None
                    )
            rows.append(result)
            for tid in due_ids.get(key, []):
                source_rows.append({"period": period, "Box": box, "Currency": currency,
                                    "metric_group": "planned_cash_operating_cost",
                                    "tx_id": tid})

    # Compute a month-end liquidity stress statistic from zero *incremental*
    # initial cash, not from the control balance or an asserted bank position.
    for row in rows:
        period, box, currency = row["period"], row["Box"], row["Currency"]
        if period < as_of_month or period >= month_last:
            row["minimum_starting_liquidity_after_month_end"] = None
            continue
        running, trough = 0.0, 0.0
        for future in _month_range(str(pd.Period(period) + 1), month_last):
            key = (future, box, currency)
            amount = float(indexed.loc[key, "net_cash_flow"]) if key in indexed.index else 0.0
            running += amount
            trough = min(trough, running)
        row["minimum_starting_liquidity_after_month_end"] = _round(-trough)

    # The agreement is intentionally a separate conditional view: FB receipts
    # do NOT say who actually collected the rent.
    agreement_rows = []
    rental = ledger.loc[
        ledger["Box"].eq(agreement["box"]) & ledger["Currency"].eq(agreement["currency"]) &
        ledger["Flujo"].eq("Cobros") & ledger["Tipo"].eq("Renta")
    ].copy()
    expenses = ledger.loc[
        ledger["Box"].isin(agreement["expense_boxes"]) &
        ledger["Currency"].eq(agreement["currency"]) & ledger["Flujo"].eq("Pagos") &
        ledger["Tipo"].isin(agreement["allowed_expense_types"])
    ].copy()
    for period in months:
        rents = rental.loc[rental["period"].eq(period)]
        payments = expenses.loc[expenses["period"].eq(period)]
        totals = {"primos": 0.0, "alejandro": 0.0, "excluded": 0.0,
                  "unattributed": 0.0}
        for rent in rents.to_dict("records"):
            if str(rent.get("tag") or "") in agreement["excluded_rent_tags"]:
                side = "excluded"
            else:
                matched = [
                    rule["shares"] for rule in agreement["rent_attribution"]
                    if all(str(rent.get(k) if pd.notna(rent.get(k)) else "") == str(v)
                           for k, v in rule["equals"].items())
                ]
                if len(matched) > 1:
                    raise ValueError(f"overlapping rent attribution for tx_id {rent['tx_id']}")
                side = "attributed" if matched else "unattributed"
            if side == "attributed":
                for actor, share in matched[0].items():
                    totals[actor] += float(rent["amount"]) * float(share)
                    if share > 0:
                        source_rows.append({
                            "period": period, "Box": rent["Box"], "Currency": rent["Currency"],
                            "metric_group": "agreement_rent_" + actor, "tx_id": rent["tx_id"],
                        })
            else:
                totals[side] += float(rent["amount"])
                source_rows.append({
                    "period": period, "Box": rent["Box"], "Currency": rent["Currency"],
                    "metric_group": "agreement_rent_" + side, "tx_id": rent["tx_id"],
                })
        paid = {
            "primos": _round(payments.loc[payments["payer"].eq("Primos"), "amount"].sum()),
            "alejandro": _round(payments.loc[payments["payer"].eq("Alejandro"), "amount"].sum()),
        }
        unidentified = payments.loc[~payments["payer"].isin(("Primos", "Alejandro"))]
        unidentified_amount = _round(unidentified["amount"].sum())
        box_funded = _round(unidentified.loc[unidentified["payer"].eq("FB"), "amount"].sum())
        other_funded = _round(unidentified.loc[~unidentified["payer"].eq("FB"), "amount"].sum())
        for payment in payments.to_dict("records"):
            party = payment["payer"].lower() if payment["payer"] in ("Primos", "Alejandro") else "unallocated"
            source_rows.append({
                "period": period, "Box": payment["Box"], "Currency": payment["Currency"],
                "metric_group": "agreement_candidate_cost_" + party,
                "tx_id": payment["tx_id"],
            })
        ready = bool(len(rents) and totals["unattributed"] < 0.005 and unidentified_amount < 0.005)
        primos_net = totals["primos"] - paid["primos"]
        alejandro_net = totals["alejandro"] - paid["alejandro"]
        agreement_rows.append({
            "period": period, "Currency": agreement["currency"],
            "agreement_id": agreement["id"],
            "primos_collected": _round(totals["primos"]),
            "alejandro_collected": _round(totals["alejandro"]),
            "taipe_and_other_excluded": _round(totals["excluded"]),
            "unattributed_collections": _round(totals["unattributed"]),
            "primos_candidate_expenses_paid": paid["primos"],
            "alejandro_candidate_expenses_paid": paid["alejandro"],
            "unallocated_candidate_expenses_paid": unidentified_amount,
            "fb_reported_candidate_expenses_paid": box_funded,
            "third_party_candidate_expenses_paid": other_funded,
            "rent_attribution_complete": bool(len(rents) and totals["unattributed"] < 0.005),
            "expense_attribution_complete": unidentified_amount < 0.005,
            "attribution_complete": ready,
            "net_shared_pool": _round(primos_net + alejandro_net) if ready else None,
            "target_share_each": _round((primos_net + alejandro_net) * 0.5) if ready else None,
            "settlement_primos_to_alejandro": _round((primos_net - alejandro_net) * 0.5) if ready else None,
            "illustrative_transfer_before_unallocated_costs": _round((primos_net - alejandro_net) * 0.5),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "monthly": out_dir / "monthly_policy_inputs.csv",
        "agreement": out_dir / "monthly_agreement_settlement_inputs.csv",
        "sources": out_dir / "policy_input_sources.csv",
        "manifest": out_dir / "policy_inputs_manifest.json",
    }
    pd.DataFrame(rows).to_csv(paths["monthly"], index=False)
    pd.DataFrame(agreement_rows).to_csv(paths["agreement"], index=False)
    pd.DataFrame(source_rows, columns=["period", "Box", "Currency", "metric_group", "tx_id"]).to_csv(
        paths["sources"], index=False
    )
    paths["manifest"].write_text(json.dumps({
        "schema": OUTPUT_SCHEMA, "run_id": run_root.name,
        "as_of_date": str(as_of.date()), "through_date": str(through.date()),
        "start_month": month_start, "agreement_id": agreement["id"],
        "rent_attribution_rules": len(agreement["rent_attribution"]),
        "cash_cost_basis": "physical Box-paid planned taxes/services/maintenance/legal; not other-party payments",
        "distribution_basis": "both pre-distribution operating surplus and pre-distribution cash net; neither validated cash",
        "distribution_values": "nonnegative illustrative policy INPUT, not authorized payouts",
        "liquidity_basis": "minimum cumulative future monthly cash net (rolling horizons and full projection) from zero; not verified cash",
        "settlement_basis": "rent by documented collector mapping, excluding named Taipe offset; candidate payer expenses",
        "incomplete_agreement_months": sum(not r["attribution_complete"] for r in agreement_rows),
        "units": "native Currency; no currency conversion",
    }, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Phase-A monthly policy inputs from a pinned run.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--start-month")
    parser.add_argument("--agreement", type=Path)
    a = parser.parse_args()
    outputs = build_policy_inputs(
        run_root=a.run_root, out_dir=a.out_dir, as_of_date=a.as_of_date,
        through_date=a.through_date, start_month=a.start_month, agreement_path=a.agreement,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
