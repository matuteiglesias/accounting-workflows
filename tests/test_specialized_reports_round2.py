from pathlib import Path

import pandas as pd
import pytest

from accounting.reports.specialized.render import render_specialized
from accounting.reports.specialized.spec import REPORT_SPECS
from accounting.reports.specialized.views import build_specialized_view, source_paths_for_view


ROUND2_IDS = [
    "support_by_target_box",
    "prior_period_clearing",
    "physical_inflows_by_box",
    "physical_outflows_by_box",
    "accountability_balance",
    "accountability_cycles",
    "open_debt_positions",
    "debt_activity",
    "repayment_allocations",
]


def _write_round2_sources(run: Path, metrics: Path) -> None:
    run.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        {"period":"2026-01","Currency":"ARS","target_box":"Property Management","funding_actor":"Actor A","recognized_amount":100,"settlement_nature":"current_period_support","obligation_period":"2026-01","settlement_period":"2026-01"},
        {"period":"2026-02","Currency":"ARS","target_box":"Family Business","funding_actor":"Actor B","recognized_amount":50,"settlement_nature":"current_period_support","obligation_period":"2026-02","settlement_period":"2026-02"},
        {"period":"2026-03","Currency":"ARS","target_box":"Household","funding_actor":"Actor C","recognized_amount":999,"settlement_nature":"current_period_support","obligation_period":"2026-03","settlement_period":"2026-03"},
        {"period":"2026-04","Currency":"ARS","target_box":"Property Management","funding_actor":"Héctor","recognized_amount":60,"settlement_nature":"prior_period_clearing","obligation_period":"2024","settlement_period":"2026-04"},
    ]).to_csv(run / "monthly_stakeholder_support.csv", index=False)

    pd.DataFrame([
        {"period":"2026-01","Box":"Family Business","Currency":"ARS","total_cash_in":1000,"total_cash_out":300,"net_cash_flow":700},
        {"period":"2026-01","Box":"Property Management","Currency":"ARS","total_cash_in":100,"total_cash_out":90,"net_cash_flow":10},
        {"period":"2026-02","Box":"Family Business","Currency":"ARS","total_cash_in":500,"total_cash_out":200,"net_cash_flow":300},
        {"period":"2026-02","Box":"Property Management","Currency":"ARS","total_cash_in":0,"total_cash_out":20,"net_cash_flow":-20},
    ]).to_csv(run / "monthly_cash_accountability.csv", index=False)

    pd.DataFrame([
        {
            "cycle_id":"2025-09-01_2026-02-28","cycle_start":"2025-09-01","cycle_end":"2026-02-28",
            "view_type":"completed_cycle","as_of_date":"2026-02-28","Box":"Family Business","Currency":"ARS",
            "opening_accountability_balance":100,"accountable_receipts":1000,"documented_distributions":400,
            "supported_uses":300,"documented_transfers_out":100,"closing_accountability_balance":300,
            "validated_cash_status":"unavailable","accountability_gap_status":"unavailable_no_validated_cash","n_months":6,
        },
        {
            "cycle_id":"2026-03-01_2026-08-31","cycle_start":"2026-03-01","cycle_end":"2026-08-31",
            "view_type":"completed_cycle","as_of_date":"2026-08-31","Box":"Family Business","Currency":"ARS",
            "opening_accountability_balance":300,"accountable_receipts":1200,"documented_distributions":700,
            "supported_uses":500,"documented_transfers_out":100,"closing_accountability_balance":200,
            "validated_cash_status":"unavailable","accountability_gap_status":"unavailable_no_validated_cash","n_months":6,
        },
    ]).to_csv(run / "family_business_accountability_cycles.csv", index=False)

    pd.DataFrame([
        {"period":"2026-07","as_of_date":"2026-07-31","debtor":"PM","creditor":"MI","Currency":"USD","component":"total","position_status":"available","open_amount":200,"open_principal":180,"open_interest":20,"n_open_items":2},
        {"period":"2026-08","as_of_date":"2026-08-31","debtor":"PM","creditor":"MI","Currency":"USD","component":"total","position_status":"available","open_amount":150,"open_principal":140,"open_interest":10,"n_open_items":2},
        {"period":"2026-08","as_of_date":"2026-08-31","debtor":"PM","creditor":"Primos","Currency":"USD","component":"total","position_status":"available","open_amount":0,"open_principal":0,"open_interest":0,"n_open_items":0},
        {"period":"2026-08","as_of_date":"2026-08-31","debtor":"Hector","creditor":"MI","Currency":"USD","component":"total","position_status":"available","open_amount":50,"open_principal":50,"open_interest":0,"n_open_items":1},
    ]).to_csv(run / "monthly_debt_position.csv", index=False)

    activity_rows = []
    def add_activity(period, opening, closing, activity_type, new=0, interest=0, repayment=0, adjustment=0, n_items=1):
        activity_rows.append({
            "period":period,"Currency":"USD","debtor":"PM","creditor":"MI","activity_type":activity_type,
            "new_principal":new,"interest_accrued":interest,"repayments":repayment,"adjustments":adjustment,
            "opening_total":opening,"closing_total":closing,"n_items":n_items,
            "reconciliation_status":"reconciled",
        })
    add_activity("2026-07", 200, 150, "interest_accrual", interest=10)
    add_activity("2026-07", 200, 150, "repayment", repayment=60)
    add_activity("2026-08", 150, 190, "new_claim", new=100)
    add_activity("2026-08", 150, 190, "repayment", repayment=50)
    add_activity("2026-08", 150, 190, "adjustment", adjustment=-10)
    pd.DataFrame(activity_rows).to_csv(run / "monthly_debt_activity.csv", index=False)

    pd.DataFrame([
        {
            "period":"2026-07","repayment_tx_id":"repay-1","repayment_date":"2026-07-20","debtor":"PM","creditor":"MI","Currency":"USD",
            "repayment_amount":100,"allocated_amount":60,"leftover_amount":0,"allocation_status":"resolved",
            "target_debt_id":"secret-hash-1","target_item_type":"Interes","target_opened_at":"2026-01-01","target_detail":"Interés enero",
        },
        {
            "period":"2026-07","repayment_tx_id":"repay-1","repayment_date":"2026-07-20","debtor":"PM","creditor":"MI","Currency":"USD",
            "repayment_amount":100,"allocated_amount":40,"leftover_amount":0,"allocation_status":"resolved",
            "target_debt_id":"secret-hash-2","target_item_type":"Prestamo","target_opened_at":"2026-01-01","target_detail":"Principal enero",
        },
        {
            "period":"2026-08","repayment_tx_id":"repay-2","repayment_date":"2026-08-20","debtor":"PM","creditor":"MI","Currency":"USD",
            "repayment_amount":50,"allocated_amount":30,"leftover_amount":20,"allocation_status":"partial",
            "target_debt_id":"secret-hash-3","target_item_type":"Prestamo","target_opened_at":"2026-02-01","target_detail":"Principal febrero",
        },
    ]).to_csv(run / "monthly_debt_repayment_detail.csv", index=False)


def test_round2_recipes_are_declared_after_round1():
    ids = [spec.report_id for spec in REPORT_SPECS]
    assert ids[10:10 + len(ROUND2_IDS)] == ROUND2_IDS
    assert len(ids) >= 19


def test_support_and_prior_clearing_keep_box_and_period_semantics(tmp_path: Path):
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round2_sources(run, metrics)

    support = build_specialized_view("support_by_target_box", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert dict(zip(support["target_box"], support["value"])) == {
        "Family Business": 50,
        "Property Management": 160,
    }
    assert "Household" not in set(support["target_box"])

    clearing = build_specialized_view("prior_period_clearing", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert clearing["value"].sum() == 60
    assert clearing.iloc[0]["funding_actor"] == "Héctor"
    assert clearing.iloc[0]["obligation_period"] == "2024"
    assert clearing.iloc[0]["settlement_period"] == "2026-04"
    assert clearing["calculation_rule"].str.contains("no debt extinguishment inferred").all()


def test_physical_cash_reports_use_governed_cash_only_and_reconcile_net(tmp_path: Path):
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round2_sources(run, metrics)

    inflow = build_specialized_view("physical_inflows_by_box", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    outflow = build_specialized_view("physical_outflows_by_box", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert dict(zip(inflow["box"], inflow["value"])) == {"Family Business": 1500, "Property Management": 100}
    assert dict(zip(outflow["box"], outflow["value"])) == {"Family Business": 500, "Property Management": 110}

    bad = pd.read_csv(run / "monthly_cash_accountability.csv")
    bad.loc[0, "net_cash_flow"] = 699
    bad.to_csv(run / "monthly_cash_accountability.csv", index=False)
    with pytest.raises(ValueError, match="does not reconcile"):
        build_specialized_view("physical_inflows_by_box", run_root=run, metrics_dir=metrics, scope="FBPM")


def test_accountability_balance_and_cycles_use_governed_cycle_equation(tmp_path: Path):
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round2_sources(run, metrics)

    balance = build_specialized_view("accountability_balance", run_root=run, metrics_dir=metrics, scope="Family Business").frame
    assert set(balance["concept"]) == {
        "Saldo inicial a rendir",
        "Ingresos sujetos a rendición",
        "Distribuciones documentadas",
        "Usos respaldados",
        "Transferencias documentadas",
        "Saldo final a rendir",
    }
    closing = balance.loc[balance["concept"].eq("Saldo final a rendir"), "value"]
    assert closing.tolist() == [200]
    assert set(balance["validated_cash_status"]) == {"unavailable"}

    cycles = build_specialized_view("accountability_cycles", run_root=run, metrics_dir=metrics, scope="Family Business").frame
    assert len(cycles) == 2
    assert set(cycles["value"]) == {200, 300}
    assert cycles["cycle"].str.contains("2025-09 → 2026-02").any()
    assert cycles["cycle"].str.contains("2026-03 → 2026-08").any()

    bad = pd.read_csv(run / "family_business_accountability_cycles.csv")
    bad.loc[1, "closing_accountability_balance"] = 201
    bad.to_csv(run / "family_business_accountability_cycles.csv", index=False)
    with pytest.raises(ValueError, match="cycle equation does not reconcile"):
        build_specialized_view("accountability_balance", run_root=run, metrics_dir=metrics, scope="Family Business")


def test_open_debt_position_uses_latest_stock_and_never_sums_monthly_stock(tmp_path: Path):
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round2_sources(run, metrics)

    position = build_specialized_view("open_debt_positions", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert position["value"].sum() == 200
    assert dict(zip(position["relation"], position["value"])) == {"Hector → MI": 50, "PM → MI": 150}
    assert "PM → Primos" not in set(position["relation"])
    assert set(position["period"]) == {"2026"}
    assert position["calculation_rule"].str.contains("no monthly stock summation").all()

    bad = pd.read_csv(run / "monthly_debt_position.csv")
    bad.loc[(bad["period"].eq("2026-08")) & (bad["debtor"].eq("PM")) & (bad["creditor"].eq("MI")), "position_status"] = "unavailable"
    bad.to_csv(run / "monthly_debt_position.csv", index=False)
    with pytest.raises(ValueError, match="will not backfill"):
        build_specialized_view("open_debt_positions", run_root=run, metrics_dir=metrics, scope="FBPM")


def test_debt_activity_aggregates_flows_only_and_keeps_adjustment_direction(tmp_path: Path):
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round2_sources(run, metrics)

    activity = build_specialized_view("debt_activity", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    values = dict(zip(activity["activity_label"], activity["value"]))
    assert values == {
        "Ajuste que reduce deuda": 10,
        "Interés devengado": 10,
        "Nueva obligación / principal": 100,
        "Repago aplicado": 110,
    }
    assert activity["calculation_rule"].str.contains("opening/closing stocks are excluded").all()

    bad = pd.read_csv(run / "monthly_debt_activity.csv")
    bad.loc[bad["period"].eq("2026-08"), "closing_total"] = 191
    bad.to_csv(run / "monthly_debt_activity.csv", index=False)
    with pytest.raises(ValueError, match="does not reconcile to position"):
        build_specialized_view("debt_activity", run_root=run, metrics_dir=metrics, scope="FBPM")


def test_repayment_allocations_reconcile_each_event_without_repeating_repayment_total(tmp_path: Path):
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round2_sources(run, metrics)

    allocations = build_specialized_view("repayment_allocations", run_root=run, metrics_dir=metrics, scope="FBPM").frame
    assert allocations["value"].sum() == 130
    assert set(allocations["allocation_status"]) == {"resolved", "partial"}
    assert allocations["calculation_rule"].str.contains("repayment_amount is not repeated").all()

    bad = pd.read_csv(run / "monthly_debt_repayment_detail.csv")
    bad.loc[bad["repayment_tx_id"].eq("repay-2"), "leftover_amount"] = 19
    bad.to_csv(run / "monthly_debt_repayment_detail.csv", index=False)
    with pytest.raises(ValueError, match="do not reconcile"):
        build_specialized_view("repayment_allocations", run_root=run, metrics_dir=metrics, scope="FBPM")


def test_round2_reports_render_without_exposing_internal_debt_ids(tmp_path: Path):
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round2_sources(run, metrics)

    for report_id in ROUND2_IDS:
        out = tmp_path / "reports" / report_id
        outputs = render_specialized(
            report_id=report_id,
            run_root=run,
            metrics_dir=metrics,
            out_dir=out,
            as_of_date="2026-08-31",
            require_pdf=False,
        )
        validation = pd.read_csv(outputs["validation"])
        assert set(validation["status"]) == {"pass"}
        html = outputs["html"].read_text(encoding="utf-8")
        assert "Qué establece:" in html and "Qué no establece:" in html
        assert "secret-hash" not in html

    cycle_html = (tmp_path / "reports" / "accountability_cycles" / "report.html").read_text(encoding="utf-8")
    assert "2025-09 → 2026-02" in cycle_html
    assert "2026-03 → 2026-08" in cycle_html


def test_round2_source_paths_remain_backend_artifacts(tmp_path: Path):
    run, metrics = tmp_path / "run", tmp_path / "metrics"
    _write_round2_sources(run, metrics)
    paths = source_paths_for_view("repayment_allocations", run, metrics)
    assert paths == ((run / "monthly_debt_repayment_detail.csv", "run/monthly_debt_repayment_detail.csv"),)
