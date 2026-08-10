import json
import subprocess
import sys

import pandas as pd


FIXTURE = "fixtures/ledger_debt_status_fixture.csv"


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

    subprocess.run([
        sys.executable, "-m", "accounting.stage_d.materialize",
        "--out-dir", str(run_root), "--mode", "smoke",
        "--run-id", "fixture_FBPM", "--freq", "M", "--force", "0",
    ], check=True)
    audit = pd.read_csv(run_root / "classification_audit.csv")
    assert set(audit["tx_id"]) == {"pm-repayment", "pm-operating-paid"}


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
