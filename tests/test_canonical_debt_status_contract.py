import json
import subprocess
import sys

import pandas as pd


FIXTURE = "fixtures/ledger_debt_status_fixture.csv"


def _run_debt_pipeline(ledger, root):
    debt_dir = root / "debt"
    mart_dir = root / "mart"
    subprocess.run([
        sys.executable, "-m", "accounting.debt.resolve",
        "--ledger-csv", str(ledger), "--write-dir", str(debt_dir),
        "--currencies", "USD", "--repayment-statuses", "pagado",
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "accounting.debt.balance_views",
        "--open-items", str(debt_dir / "debt_open_items.csv"),
        "--write-dir", str(debt_dir),
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "accounting.marts.debt",
        "--debt-dir", str(debt_dir), "--write-dir", str(mart_dir),
    ], check=True)
    return debt_dir, mart_dir


def test_one_fetch_writes_scoped_all_status_evidence_and_paid_ledger(tmp_path):
    run_root = tmp_path / "fixture_FBPM"
    subprocess.run([
        sys.executable, "-m", "accounting.ledger.ingest",
        "--fixture", FIXTURE, "--out-dir", str(run_root),
        "--mode", "smoke", "--run-id", "fixture_FBPM",
        "--boxes", "Family Business,Property Management",
    ], check=True)

    all_status = pd.read_csv(run_root / "ledger_canonical_all_status.csv")
    recognized = pd.read_csv(run_root / "ledger_canonical.csv")
    assert all_status["tx_id"].tolist() == [
        "pm-loan-open", "pm-interest-open", "pm-repayment",
        "pm-operating-paid", "pm-operating-pending",
    ]
    assert recognized["tx_id"].tolist() == ["pm-repayment", "pm-operating-paid"]
    assert set(all_status["Box"]) == {"Property Management"}
    assert "hh-loan-open" not in set(all_status["tx_id"])

    manifest = json.loads((run_root / "meta" / "stage_A_ingest.json").read_text())
    assert manifest["params"]["only_status"] == ["pagado"]
    assert {item["name"] for item in manifest["outputs"]} >= {
        "ledger_canonical", "ledger_canonical_all_status"
    }

    hh_root = tmp_path / "fixture_HH"
    subprocess.run([
        sys.executable, "-m", "accounting.ledger.ingest",
        "--fixture", FIXTURE, "--out-dir", str(hh_root),
        "--mode", "smoke", "--run-id", "fixture_HH", "--boxes", "Household",
    ], check=True)
    hh_all_status = pd.read_csv(hh_root / "ledger_canonical_all_status.csv")
    assert hh_all_status["tx_id"].tolist() == ["hh-loan-open"]
    assert set(hh_all_status["Box"]) == {"Household"}


def test_debt_uses_all_status_evidence_while_materialization_uses_paid_ledger(tmp_path):
    run_root = tmp_path / "fixture_FBPM"
    subprocess.run([
        sys.executable, "-m", "accounting.ledger.ingest",
        "--fixture", FIXTURE, "--out-dir", str(run_root),
        "--mode", "smoke", "--run-id", "fixture_FBPM",
        "--boxes", "Family Business,Property Management",
    ], check=True)
    debt_dir = tmp_path / "debt"
    subprocess.run([
        sys.executable, "-m", "accounting.debt.resolve",
        "--ledger-csv", str(run_root / "ledger_canonical_all_status.csv"),
        "--write-dir", str(debt_dir), "--currencies", "USD",
        "--repayment-statuses", "pagado",
    ], check=True)
    open_items = pd.read_csv(debt_dir / "debt_open_items.csv")
    assert set(open_items["source_tx_id"]) == {"pm-loan-open", "pm-interest-open"}
    assert open_items.groupby("item_type")["original_amount"].sum().to_dict() == {
        "Interes": 10, "Prestamo": 100
    }
    allocations = pd.read_csv(debt_dir / "debt_allocations.csv")
    assert allocations["allocated_amount"].sum() == 60
    assert open_items["open_amount"].sum() == 50

    subprocess.run([
        sys.executable, "-m", "accounting.stage_d.materialize",
        "--out-dir", str(run_root), "--mode", "smoke",
        "--run-id", "fixture_FBPM", "--freq", "M", "--force", "0",
    ], check=True)
    audit = pd.read_csv(run_root / "classification_audit.csv")
    assert set(audit["tx_id"]) == {"pm-repayment", "pm-operating-paid"}


def test_debt_item_without_repayments_keeps_open_balance_and_empty_contracts(tmp_path):
    ledger = tmp_path / "hh_all_status.csv"
    pd.DataFrame([{
        "tx_id": "hh-loan", "Date": "2024-09-13", "amount": 100,
        "Currency": "USD", "payer": "MI", "receiver": "Cande",
        "Flujo": "Debt", "Tipo": "Prestamo", "status": "pagado",
        "Box": "Household",
    }]).to_csv(ledger, index=False)
    debt_dir, mart_dir = _run_debt_pipeline(ledger, tmp_path)

    items = pd.read_csv(debt_dir / "debt_open_items.csv")
    assert items.loc[0, "open_amount"] == 100
    assert items.loc[0, "engine_status"] == "open"
    for filename in ["debt_allocations.csv", "debt_repayment_events.csv"]:
        frame = pd.read_csv(debt_dir / filename)
        assert frame.empty and len(frame.columns) > 0
    assert (mart_dir / "monthly_debt_position.csv").is_file()
    assert (mart_dir / "monthly_debt_activity.csv").is_file()


def test_repayment_without_debt_items_keeps_schemas_and_pipeline_completes(tmp_path):
    ledger = tmp_path / "fbpm_all_status.csv"
    pd.DataFrame([{
        "tx_id": "orphan-repayment", "Date": "2026-01-01", "amount": 25,
        "Currency": "USD", "payer": "PM", "receiver": "Owner",
        "Flujo": "Debt", "Tipo": "Repago", "status": "pagado",
        "Box": "Property Management",
    }]).to_csv(ledger, index=False)
    debt_dir, mart_dir = _run_debt_pipeline(ledger, tmp_path)

    assert pd.read_csv(debt_dir / "debt_open_items.csv").empty
    assert pd.read_csv(debt_dir / "debt_allocations.csv").empty
    events = pd.read_csv(debt_dir / "debt_repayment_events.csv")
    assert events["repayment_tx_id"].tolist() == ["orphan-repayment"]
    assert events["leftover_amount"].tolist() == [25]
    assert (mart_dir / "monthly_debt_position.csv").is_file()
    assert (mart_dir / "monthly_debt_activity.csv").is_file()


def test_repayment_never_allocates_to_an_obligation_opened_later(tmp_path):
    ledger = tmp_path / "future_debt.csv"
    pd.DataFrame([
        {
            "tx_id": "eligible-principal", "Date": "2025-01-01", "amount": 40,
            "Currency": "USD", "payer": "PM", "receiver": "Creditor",
            "Flujo": "Debt", "Tipo": "Prestamo", "status": "abierto",
            "Box": "Property Management", "Detalle": "Eligible obligation",
        },
        {
            "tx_id": "repayment", "Date": "2025-02-01", "amount": 50,
            "Currency": "USD", "payer": "PM", "receiver": "Creditor",
            "Flujo": "Debt", "Tipo": "Repago", "status": "pagado",
            "Box": "Property Management", "Detalle": "February repayment",
        },
        {
            "tx_id": "future-interest", "Date": "2025-03-01", "amount": 10,
            "Currency": "USD", "payer": "PM", "receiver": "Creditor",
            "Flujo": "Debt", "Tipo": "Interes", "status": "abierto",
            "Box": "Property Management", "Detalle": "Future interest",
        },
    ]).to_csv(ledger, index=False)

    debt_dir, mart_dir = _run_debt_pipeline(ledger, tmp_path)
    allocations = pd.read_csv(debt_dir / "debt_allocations.csv")
    assert allocations["target_source_tx_id"].tolist() == ["eligible-principal"]
    assert allocations.loc[0, "balance_before"] == 40
    assert allocations.loc[0, "allocated_amount"] == 40
    assert allocations.loc[0, "balance_after"] == 0
    events = pd.read_csv(debt_dir / "debt_repayment_events.csv")
    assert events.loc[0, "leftover_amount"] == 10

    detail = pd.read_csv(mart_dir / "monthly_debt_repayment_detail.csv")
    assert detail["target_debt_id"].tolist() == ["prestamo::eligible-principal"]
    assert detail.loc[0, "target_detail"] == "Eligible obligation"
    assert detail.loc[0, "repayment_detail"] == "February repayment"
    assert detail.loc[0, "allocation_status"] == "partial"


def test_zero_row_debt_outputs_keep_schema_and_balance_views_can_read_them(tmp_path):
    ledger = tmp_path / "empty_debt_ledger.csv"
    pd.DataFrame([{
        "tx_id": "ordinary", "Date": "2026-01-01", "amount": 1,
        "Currency": "USD", "payer": "A", "receiver": "B",
        "Flujo": "Other", "Tipo": "Ingreso", "status": "pagado",
        "Box": "Property Management",
    }]).to_csv(ledger, index=False)
    debt_dir = tmp_path / "debt"
    subprocess.run([
        sys.executable, "-m", "accounting.debt.resolve",
        "--ledger-csv", str(ledger), "--write-dir", str(debt_dir),
    ], check=True)

    required = {
        "debt_open_items.csv": {"debt_id", "source_tx_id", "open_amount"},
        "debt_allocations.csv": {"allocation_id", "repayment_tx_id", "allocated_amount"},
        "debt_repayment_events.csv": {"repayment_tx_id", "repayment_amount"},
        "debt_resolution_timeline.csv": {"event_date", "event_kind", "tx_id"},
        "debt_status_reconciliation.csv": {"debt_id", "reconciliation_note"},
    }
    for filename, columns in required.items():
        frame = pd.read_csv(debt_dir / filename)
        assert frame.empty
        assert columns <= set(frame.columns)

    subprocess.run([
        sys.executable, "-m", "accounting.debt.balance_views",
        "--open-items", str(debt_dir / "debt_open_items.csv"),
        "--write-dir", str(debt_dir),
    ], check=True)
