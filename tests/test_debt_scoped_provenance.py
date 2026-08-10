from argparse import Namespace
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from accounting.debt.resolve import load_debt_ledger


def _args(ledger_csv):
    return Namespace(
        ledger_csv=str(ledger_csv), fixture=None, sheet_url=None,
        service_account=None, sheet_name="unused", exclude_household=False,
        currencies="USD",
    )


def test_debt_reads_only_scoped_canonical_ledger_and_ignores_environment(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger_canonical.csv"
    pd.DataFrame([
        {"tx_id": "pm-debt", "Date": "2026-01-01", "payer": "PM", "receiver": "Owner", "amount": 100, "Currency": "USD", "Tipo": "Prestamo", "Flujo": "Debt", "status": "pagado", "Box": "Property Management"},
        {"tx_id": "pm-repay", "Date": "2026-02-01", "payer": "PM", "receiver": "Owner", "amount": 100, "Currency": "USD", "Tipo": "Repago", "Flujo": "Debt", "status": "pagado", "Box": "Property Management"},
    ]).to_csv(ledger, index=False)

    monkeypatch.setenv("BOXES", "Household")
    under_hh_env = load_debt_ledger(_args(ledger))
    monkeypatch.setenv("BOXES", "Family Business,Property Management")
    under_fbpm_env = load_debt_ledger(_args(ledger))

    pd.testing.assert_frame_equal(under_hh_env, under_fbpm_env)
    assert set(under_hh_env["tx_id"]) == {"pm-debt", "pm-repay"}
    assert set(under_hh_env["Box"]) == {"Property Management"}

    outputs = []
    for box_env in ["Household", "Family Business,Property Management"]:
        write_dir = tmp_path / f"debt_{len(outputs)}"
        subprocess.run([
            sys.executable, "-m", "accounting.debt.resolve",
            "--ledger-csv", str(ledger), "--write-dir", str(write_dir),
            "--currencies", "USD", "--repayment-statuses", "pagado",
        ], check=True, env=dict(os.environ, BOXES=box_env), capture_output=True, text=True)
        outputs.append(write_dir)
    for filename in ["debt_open_items.csv", "debt_allocations.csv", "debt_repayment_events.csv", "debt_resolution_timeline.csv", "debt_status_reconciliation.csv"]:
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
    open_items = pd.read_csv(outputs[0] / "debt_open_items.csv")
    assert set(open_items["source_tx_id"]) <= {"pm-debt", "pm-repay"}


def test_canonical_make_debt_path_has_one_source_and_no_household_policy():
    text = Path("Makefile").read_text()
    action = text.split("_run_debt_action:", 1)[1].split(".PHONY: run-debt-views", 1)[0]
    assert '--ledger-csv "$(RUN_OUT)/ledger_canonical_all_status.csv"' in action
    assert "--sheet-url" not in action
    assert "--service-account" not in action
    assert "exclude-household" not in action
    assert "DEBT_EXCLUDE_HOUSEHOLD" not in text
