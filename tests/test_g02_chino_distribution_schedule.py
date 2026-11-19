"""G02: repeatable future-only schedule, ledger overlays, FX and HH counterparty."""
import json
from pathlib import Path
from hashlib import sha256

import pandas as pd
import pytest
import yaml

from accounting.scenarios.g02_chino_distribution import run_scenarios, _validate


def fixture(tmp_path):
    repo = tmp_path / "repo"
    base = repo / "out/run/accounting/20261119T061512Z_FBPM"
    base.mkdir(parents=True)
    rows = []
    def add(month, day, box, cur, amount, payer, receiver, flow, kind,
            status="pagado", tag=""):
        rows.append({
            "tx_id": f"raw-{len(rows)}", "Date": f"{month}-{day:02d}",
            "Box": box, "Currency": cur, "amount": amount,
            "payer": payer, "receiver": receiver,
            "Flujo": flow, "Tipo": kind,
            "status": status, "tag": tag,
            "notes": "", "Detalle": "G02 testing", "Lugar": "CABA",
            "source_file": "synthetic_source.csv", "source_row": len(rows) + 2,
        })
    for month in ["2026-05", "2026-06", "2026-07", "2026-08",
                  "2026-09", "2026-10", "2026-11", "2026-12",
                  "2027-01", "2027-02", "2027-03"]:
        status = "pagado" if month <= "2026-11" else "planeado"
        add(month, 10, "Property Management", "ARS", 400000, "Inq", "PM",
            "Cobros", "Renta", status, "PM_RENT_CABA_CHINO_ARS")
        add(month, 10, "Property Management", "USD", 380, "Inq", "PM",
            "Cobros", "Renta", status, "PM_RENT_CABA_CHINO_USD")
    for m, paid_ars in [("2026-05", 482384), ("2026-06", 485000),
                        ("2026-07", 514000), ("2026-10", 506900)]:
        add(m, 12, "Property Management", "ARS", paid_ars,
            "PM", "Primos", "Transfer", "Dividendo")
        add(m, 12, "Property Management", "ARS", 200000,
            "PM", "Alejandro", "Transfer", "Dividendo")
        add(m, 12, "Property Management", "ARS", 200000,
            "Alejandro", "HH", "Transfer", "Contribuciones")
        add(m, 12, "Property Management", "USD", 190,
            "PM", "Alejandro", "Transfer", "Dividendo")
        add(m, 12, "Property Management", "USD", 190,
            "Alejandro", "PM", "Repago", "Repago")
        if m in ("2026-05", "2026-06", "2026-07"):
            fx_amount = {"2026-05": 282384, "2026-06": 285000, "2026-07": 285000}[m]
            add(m, 11, "Property Management", "USD", 190,
                "PM", "FX", "Cambio", "FX", "X")
            add(m, 11, "Property Management", "ARS", fx_amount,
                "FX", "PM", "Cambio", "FX", "X")
    original = base / "ledger_canonical_all_status.csv"
    pd.DataFrame(rows).to_csv(original, index=False)
    monthfile = tmp_path / "monthly.csv"
    monthly = []
    for month in ["2026-11", "2026-12", "2027-01", "2027-02", "2027-03", "2027-04"]:
        for cur in ("ARS", "USD"):
            controls = {
                "2026-11": 3871601.57, "2026-12": 3507450.57,
                "2027-01": 245150.57, "2027-02": -254849.43,
                "2027-03": 482850.57,
            }
            monthly.append({
                "period": month, "Box": "Property Management",
                "Currency": cur, "materialized_month": month != "2027-04",
                "closing_control": controls.get(month) if cur == "ARS" else
                    (1660 + list(controls).index(month) * 380 if month in controls else None),
                "net_cash_flow": 0 if month != "2027-04" else None,
            })
    pd.DataFrame(monthly).to_csv(monthfile, index=False)
    config = {
        "schema": "accounting.g02_chino_distribution.v1",
        "id": "g02_test", "source": {
            "run_id": base.name,
            "all_status_sha256": sha256(original.read_bytes()).hexdigest(),
            "monthly_policy_inputs_sha256": sha256(monthfile.read_bytes()).hexdigest(),
        },
        "as_of_date": "2026-11-19", "through_month": "2027-04",
        "historical_paid_months": ["2026-05", "2026-06", "2026-07", "2026-10"],
        "historical_skipped_no_accrual": ["2026-08", "2026-09"],
        "no_retroactive": True,
        "historical_settlement": {
            "ars_equivalent_physical_funder": "Matias_MercadoPago",
            "ars_equivalent_accounting_charge_box": "Property Management",
            "usd_reference_custodian": "Matias_for_PM",
            "usd_physically_sold": False,
            "reimbursement_right_assumed": False,
        },
        "agreement": {
            "rent_tag_ars": "PM_RENT_CABA_CHINO_ARS",
            "rent_tag_usd": "PM_RENT_CABA_CHINO_USD",
            "monthly_rent_ars": 400000, "monthly_rent_usd": 380,
            "primos_base_ars": 200000, "primos_usd_equivalence_reference": 190,
            "alejandro_to_hh_ars": 200000, "alejandro_debt_offset_usd": 190,
            "payment_day": 20,
            "future_physical_funder": "unassigned_until_authorized",
        },
        "ars_equivalent_rates_per_usd": {
            "2026-11": 1500, "2026-12": 1500,
            "2027-01": 1500, "2027-02": 1500,
        },
        "variants": [
            {"id": "baseline", "months": [], "explanation": "No payouts"},
            {"id": "all_four", "months": ["2026-11", "2026-12", "2027-01", "2027-02"],
             "explanation": "stress test"},
        ],
    }
    definition = tmp_path / "g02.yaml"
    definition.write_text(yaml.safe_dump(config, sort_keys=False))
    return repo, original, monthfile, definition, config


def test_overlay_preserves_history_fx_and_hh(tmp_path):
    repo, original, monthly, definition, config = fixture(tmp_path)
    before = original.read_bytes()
    out = tmp_path / "result"
    summary = run_scenarios(
        definition=definition, repo=repo, monthly_source=monthly, out=out
    )
    assert original.read_bytes() == before
    assert len(pd.read_csv(out / "baseline/scenario_overlay_ledger.csv")) == 0
    overlay = pd.read_csv(out / "all_four/scenario_overlay_ledger.csv")
    assert len(overlay) == 24
    assert overlay.tx_id.is_unique
    assert set(overlay.status) == {"planeado"}
    assert not overlay.Date.str[:7].isin(["2026-05", "2026-06", "2026-07",
                                         "2026-08", "2026-09", "2026-10"]).any()
    assert len(overlay.query("Box == 'Household'")) == 4
    assert len(overlay.query("Box == 'Property Management'")) == 20
    assert overlay.query("payer == 'PM' and receiver == 'Primos'").Currency.eq("ARS").all()
    assert overlay.query("payer == 'PM' and receiver == 'Primos'").amount.eq(485000).all()
    assert not overlay.Flujo.eq("Cambio").any()
    assert not overlay.Tipo.eq("FX").any()
    assert not overlay.usd_sale_executed.any()
    assert overlay.physical_payment_source.eq("NOT_SPECIFIED_FUTURE").all()
    assert overlay.query("payer == 'Alejandro' and receiver == 'HH'").shape[0] == 8
    effects = pd.read_csv(out / "all_four/scenario_accounting_movements.csv")
    for period in ["2026-11", "2026-12", "2027-01", "2027-02"]:
        current = effects.query("period == @period")
        assert current.query("Box == 'Property Management' and Currency == 'ARS'").scenario_net.iloc[0] == -685000
        assert current.query("Box == 'Property Management' and Currency == 'USD'").scenario_net.iloc[0] == 0
        assert current.query("Box == 'Household' and Currency == 'ARS'").scenario_net.iloc[0] == 200000
    cash = pd.read_csv(out / "all_four/monthly_controls_by_box_currency.csv")
    assert cash.query("period == '2027-01' and Box == 'Property Management' and Currency == 'ARS'").scenario_control.iloc[0] == -1809849.43
    assert pd.isna(cash.query("period == '2027-04' and Box == 'Property Management' and Currency == 'ARS'").scenario_control.iloc[0])
    assert summary["variants"]["baseline"]["first_negative_pm_ars_month"] == "2027-02"
    assert summary["variants"]["all_four"]["first_negative_pm_ars_month"] == "2027-01"
    assert summary["variants"]["all_four"]["alejandro_usd_debt_offset_recorded_in_scenario"] == 760
    assert summary["variants"]["all_four"]["hh_ars_new_receipts"] == 800000
    assert not summary["variants"]["all_four"]["debt_position_recomputed_by_canonical_debt_engine"]
    assert summary["historical_settlement"]["personal_mercadopago_ars_paid"] == 1188284
    assert summary["historical_settlement"]["usd_held_by_matias_for_pm"] == 760
    assert summary["historical_settlement"]["additional_pm_ars_credit"] == 0
    assert summary["historical_settlement"]["usd_sold"] == 0
    assert not summary["variants"]["all_four"]["funding_from_salary_assumed"]
    assert not any("fx_proxy" in key for key in summary["variants"]["all_four"])
    provenance = pd.read_csv(out / "historical_personal_payment_and_usd_custody.csv")
    assert provenance.personal_mercadopago_ars_out.sum() == 1188284
    assert provenance.usd_retained_in_matias_custody.sum() == 760
    assert provenance.pm_new_ars_credit.sum() == 0
    hh = pd.read_csv(out / "all_four/hh_receipts_analytic.csv")
    assert hh.query("period in ['2026-08','2026-09']").amount_ars.eq(0).all()
    assert hh.query("period == '2026-05'").amount_ars.iloc[0] == 200000
    historical = pd.read_csv(out / "historical_distribution_bridge.csv")
    assert historical.query("period == '2026-07'").legacy_fx_stub_mismatch_ars.iloc[0] == 29000
    assert historical.query("period == '2026-10'").legacy_fx_stub_status_X.iloc[0] == False
    assert "Abril" in (out / "comparison.html").read_text()


@pytest.mark.parametrize("months", [
    ["2026-07"], ["2026-08"], ["2026-09", "2026-11"],
    ["2026-11", "2026-11"], ["2026-12", "2026-11"],
    ["2027-04"], ["2027-05"],
])
def test_no_retroactive_and_no_unplanned_months(tmp_path, months):
    repo, original, monthly, definition, spec = fixture(tmp_path)
    spec["variants"][1]["months"] = months
    definition.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="retroactive|sorted|ARS equivalence rate|Chino rent incomplete"):
        run_scenarios(definition=definition, repo=repo, monthly_source=monthly, out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_ars_equivalence_change_raises_pm_ars_charge_without_dollar_sale(tmp_path):
    repo, _, monthly, definition, spec = fixture(tmp_path)
    spec["ars_equivalent_rates_per_usd"]["2026-11"] = 1600
    definition.write_text(yaml.safe_dump(spec))
    out = tmp_path / "fx"
    run_scenarios(definition=definition, repo=repo, monthly_source=monthly, out=out)
    overlay = pd.read_csv(out / "all_four/scenario_overlay_ledger.csv")
    primos = overlay.query("Date == '2026-11-20' and payer == 'PM' and receiver == 'Primos'")
    assert primos.amount.iloc[0] == 504000
    effects = pd.read_csv(out / "all_four/scenario_accounting_movements.csv")
    assert effects.query("period == '2026-11' and Currency == 'ARS' and Box == 'Property Management'").scenario_net.iloc[0] == -704000
    assert effects.query("period == '2026-11' and Currency == 'USD' and Box == 'Property Management'").scenario_net.iloc[0] == 0


def test_source_pin_and_no_retroactive_flag_fail_closed(tmp_path):
    repo, source, monthly, definition, spec = fixture(tmp_path)
    spec["no_retroactive"] = False
    definition.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="prohibit retrospective"):
        run_scenarios(definition=definition, repo=repo, monthly_source=monthly, out=tmp_path / "out")
    spec["no_retroactive"] = True
    spec["source"]["all_status_sha256"] = "0" * 64
    definition.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="source ledger hash"):
        run_scenarios(definition=definition, repo=repo, monthly_source=monthly, out=tmp_path / "out")
    assert not (tmp_path / "out").exists()
