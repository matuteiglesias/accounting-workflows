"""Phase-A policy inputs: cash semantics and no fabricated settlements."""
import json

import pandas as pd
import pytest
import yaml

from accounting.scenarios.policy_inputs import build_policy_inputs


def _fixture(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    out = tmp_path / "features"
    agreement = tmp_path / "attribution.yaml"
    agreement.write_text(yaml.safe_dump({
        "schema": "accounting.agreement_attribution.v0",
        "id": "test_agreement", "box": "Family Business", "currency": "ARS",
        "excluded_rent_tags": ["TAIPE"],
        "rent_attribution": [
            {"equals": {"tag": "KIOSK"}, "shares": {"primos": 1.0, "alejandro": 0.0}},
            {"equals": {"tag": "CORREA"}, "shares": {"primos": 0.0, "alejandro": 1.0}},
            {"equals": {"tag": "TECH"}, "shares": {"primos": 0.5, "alejandro": 0.5}},
        ],
        "allowed_expense_types": ["Impuestos"], "expense_boxes": ["Family Business"],
    }))
    ledger = [
        ("a0", "2026-11-12", "Family Business", "ARS", "pagado", "Inq", "FB", "Cobros", "Renta", 100, "KIOSK"),
        ("d1", "2026-12-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 1000, "KIOSK"),
        ("d2", "2026-12-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 400, "CORREA"),
        ("d3", "2026-12-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 200, "TECH"),
        ("d4", "2026-12-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 100, "TAIPE"),
        ("d5", "2026-12-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 300, "FERNANDO"),
        ("d6", "2026-12-15", "Family Business", "ARS", "planeado", "FB", "Impuestos", "Pagos", "Impuestos", 250, "FB_TAX"),
        ("p1", "2026-12-30", "Property Management", "ARS", "planeado", "PM", "Impuestos", "Pagos", "Impuestos", 600, "PM_DEC"),
        ("j1", "2027-01-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 1000, "KIOSK"),
        ("j2", "2027-01-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 400, "CORREA"),
        ("j3", "2027-01-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 200, "TECH"),
        ("j4", "2027-01-10", "Family Business", "ARS", "planeado", "Inq", "FB", "Cobros", "Renta", 100, "TAIPE"),
        ("j5", "2027-01-15", "Family Business", "ARS", "planeado", "Primos", "Impuestos", "Pagos", "Impuestos", 10, "TAX1"),
        ("j6", "2027-01-15", "Family Business", "ARS", "planeado", "Alejandro", "Impuestos", "Pagos", "Impuestos", 20, "TAX2"),
        ("p2", "2027-01-28", "Property Management", "ARS", "planeado", "PM", "Impuestos", "Pagos", "Impuestos", 50, "PM_JAN"),
        ("p3", "2027-02-15", "Property Management", "ARS", "planeado", "PM", "Impuestos", "Pagos", "Impuestos", 300, "PM_FEB"),
    ]
    columns = ["tx_id", "Date", "Box", "Currency", "status", "payer", "receiver", "Flujo", "Tipo", "amount", "tag"]
    frame = pd.DataFrame(ledger, columns=columns)
    frame["Lugar"] = "Tigre"
    frame["Detalle"] = frame["tag"]
    frame["source_row"] = range(2, len(frame) + 2)
    frame.to_csv(root / "ledger_canonical.csv", index=False)

    data = [
        {"period": "2026-11", "Box": "Family Business", "Currency": "ARS",
         "rent_in": 100, "net_cash_flow": 100, "funding_cash_in": 0,
         "taxes_out": 0, "services_out": 0, "maintenance_out": 0, "legal_out": 0,
         "personal_draws_out": 0, "dividends_out": 0,
         "total_cash_in": 100, "total_cash_out": 0, "closing_control": 100},
        {"period": "2026-12", "Box": "Family Business", "Currency": "ARS",
         "rent_in": 2000, "net_cash_flow": 1750, "funding_cash_in": 0,
         "taxes_out": 250, "services_out": 0, "maintenance_out": 0, "legal_out": 0,
         "personal_draws_out": 0, "dividends_out": 0,
         "total_cash_in": 2000, "total_cash_out": 250, "closing_control": 1850},
        {"period": "2027-01", "Box": "Family Business", "Currency": "ARS",
         "rent_in": 1700, "net_cash_flow": 1670, "funding_cash_in": 0,
         "taxes_out": 30, "services_out": 0, "maintenance_out": 0, "legal_out": 0,
         "personal_draws_out": 0, "dividends_out": 0,
         "total_cash_in": 1700, "total_cash_out": 30, "closing_control": 3520},
    ]
    for period, cost in [("2026-12", 600), ("2027-01", 50), ("2027-02", 300)]:
        data.append({
            "period": period, "Box": "Property Management", "Currency": "ARS",
            "rent_in": 0, "net_cash_flow": -cost, "funding_cash_in": 0,
            "taxes_out": cost, "services_out": 0, "maintenance_out": 0,
            "legal_out": 0, "personal_draws_out": 0, "dividends_out": 0,
        })
    pd.DataFrame(data).to_csv(root / "monthly_cash_accountability.csv", index=False)
    details = []
    for row in frame.itertuples(index=False):
        category = "rent" if row.Flujo == "Cobros" else "taxes"
        direction = "in" if row.Flujo == "Cobros" else "out"
        details.append({
            "tx_id": row.tx_id, "Date": row.Date, "period": row.Date[:7],
            "Box": row.Box, "Currency": row.Currency,
            "cash_category": category, "movement_basis": "actual_cash",
            "cash_direction": direction, "amount_out": row.amount if direction == "out" else 0,
            "review_required": False,
        })
    pd.DataFrame(details).to_csv(root / "box_treasury_transaction_detail.csv", index=False)
    return root, out, agreement


def test_monthly_helpers_and_guarded_agreement(tmp_path):
    root, out, agreement = _fixture(tmp_path)
    build_policy_inputs(
        run_root=root, out_dir=out,
        as_of_date="2026-11-19", through_date="2027-03-31",
        agreement_path=agreement,
    )
    monthly = pd.read_csv(out / "monthly_policy_inputs.csv")
    dec = monthly.query("period == '2026-12' and Box == 'Family Business'").iloc[0]
    assert dec.pre_distribution_operating_surplus == 1750
    assert dec.distribution_100pct_operating_surplus == 1750
    assert dec.distribution_50pct_operating_surplus == 875
    assert dec.distribution_100pct_cash_net == 1750
    assert dec.distribution_50pct_cash_net == 875
    nov_pm = monthly.query("period == '2026-11' and Box == 'Property Management'").iloc[0]
    assert [nov_pm.cost_due_next_1m, nov_pm.cost_due_next_2m, nov_pm.cost_due_next_3m] == [600, 650, 950]
    assert bool(nov_pm.cost_due_next_3m_complete)
    assert [nov_pm.cashflow_reserve_required_next_1m,
            nov_pm.cashflow_reserve_required_next_2m,
            nov_pm.cashflow_reserve_required_next_3m] == [600, 650, 950]
    feb_pm = monthly.query("period == '2027-02' and Box == 'Property Management'").iloc[0]
    assert pd.isna(feb_pm.cost_due_next_2m)
    assert not bool(feb_pm.cost_due_next_2m_complete)

    settlement = pd.read_csv(out / "monthly_agreement_settlement_inputs.csv")
    dec = settlement.query("period == '2026-12'").iloc[0]
    assert dec.primos_collected == 1100
    assert dec.alejandro_collected == 500
    assert dec.taipe_and_other_excluded == 100
    assert dec.unattributed_collections == 300
    assert dec.unallocated_candidate_expenses_paid == 250
    assert dec.fb_reported_candidate_expenses_paid == 250
    assert dec.third_party_candidate_expenses_paid == 0
    assert pd.isna(dec.settlement_primos_to_alejandro)
    jan = settlement.query("period == '2027-01'").iloc[0]
    assert jan.primos_candidate_expenses_paid == 10
    assert jan.alejandro_candidate_expenses_paid == 20
    assert jan.unattributed_collections == 0
    assert jan.net_shared_pool == 1570
    assert jan.target_share_each == 785
    assert jan.settlement_primos_to_alejandro == 305
    sources = pd.read_csv(out / "policy_input_sources.csv")
    assert len(sources.query("tx_id == 'j3'")) == 2  # 50/50 Técnico source once per side
    assert json.loads((out / "policy_inputs_manifest.json").read_text())["schema"] == "accounting.policy_inputs.v1"


def test_reject_future_paid_as_observed(tmp_path):
    root, out, agreement = _fixture(tmp_path)
    path = root / "ledger_canonical.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.tx_id.eq("p1"), "status"] = "pagado"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="future paid"):
        build_policy_inputs(run_root=root, out_dir=out,
                            as_of_date="2026-11-19", through_date="2027-03-31",
                            agreement_path=agreement)


def test_reject_overlapping_or_invalid_rent_attribution(tmp_path):
    root, out, agreement = _fixture(tmp_path)
    data = yaml.safe_load(agreement.read_text())
    data["rent_attribution"].append(data["rent_attribution"][0].copy())
    agreement.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="overlapping"):
        build_policy_inputs(run_root=root, out_dir=out,
                            as_of_date="2026-11-19", through_date="2027-03-31",
                            agreement_path=agreement)
    data["rent_attribution"][-1]["shares"] = {"primos": 0.8, "alejandro": 0.1}
    agreement.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="sum to one"):
        build_policy_inputs(run_root=root, out_dir=out,
                            as_of_date="2026-11-19", through_date="2027-03-31",
                            agreement_path=agreement)
