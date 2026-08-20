from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from accounting.cutoff import CUTOFF_RULE, CUTOFF_VERSION


def _fixture(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "tx_id": "jul-paid",
                "Date": "2026-07-31",
                "amount": 100,
                "Currency": "ARS",
                "payer": "Tenant",
                "receiver": "PM",
                "Flujo": "Cobros",
                "Tipo": "Renta",
                "status": "pagado",
                "Box": "Property Management",
            },
            {
                "tx_id": "aug-anomaly",
                "Date": "2026-08-01",
                "amount": 50,
                "Currency": "ARS",
                "payer": "PM",
                "receiver": "Vendor",
                "Flujo": "Pagos",
                "Tipo": "",
                "status": "pagado",
                "Box": "Property Management",
            },
            {
                "tx_id": "missing-date",
                "Date": "",
                "amount": 25,
                "Currency": "ARS",
                "payer": "PM",
                "receiver": "Vendor",
                "Flujo": "Pagos",
                "Tipo": "Mantenimiento",
                "status": "pagado",
                "Box": "Property Management",
            },
            {
                "tx_id": "jul-pending-debt",
                "Date": "2026-07-15",
                "amount": 40,
                "Currency": "USD",
                "payer": "MI",
                "receiver": "PM",
                "Flujo": "Debt",
                "Tipo": "Prestamo",
                "status": "pendiente",
                "Box": "Property Management",
            },
            {
                "tx_id": "aug-pending-debt",
                "Date": "2026-08-02",
                "amount": 60,
                "Currency": "USD",
                "payer": "MI",
                "receiver": "PM",
                "Flujo": "Debt",
                "Tipo": "Prestamo",
                "status": "pendiente",
                "Box": "Property Management",
            },
        ]
    ).to_csv(path, index=False)


def _run_ingest(tmp_path: Path, *, env_cutoff: bool = False) -> Path:
    fixture = tmp_path / "ledger.csv"
    _fixture(fixture)
    out_dir = tmp_path / ("env_run" if env_cutoff else "arg_run")
    cmd = [
        sys.executable,
        "-m",
        "accounting.ledger.ingest",
        "--fixture",
        str(fixture),
        "--out-dir",
        str(out_dir),
        "--mode",
        "smoke",
        "--run-id",
        out_dir.name,
        "--boxes",
        "Property Management",
    ]
    env = dict(os.environ)
    if env_cutoff:
        env["CUTOFF_DATE"] = "2026-07-31"
    else:
        cmd.extend(["--cutoff-date", "2026-07-31"])
    subprocess.run(cmd, check=True, env=env)
    return out_dir


def _assert_cutoff_outputs(out_dir: Path) -> None:
    all_status = pd.read_csv(out_dir / "ledger_canonical_all_status.csv")
    recognized = pd.read_csv(out_dir / "ledger_canonical.csv")

    assert all_status["tx_id"].tolist() == [
        "jul-paid",
        "missing-date",
        "jul-pending-debt",
    ]
    assert recognized["tx_id"].tolist() == ["jul-paid", "missing-date"]

    for frame in [all_status, recognized]:
        dates = pd.to_datetime(frame["Date"], errors="coerce")
        assert dates.dropna().max() <= pd.Timestamp("2026-07-31")

    anomalies = pd.read_csv(out_dir / "anomalies.csv")
    assert "missing-date" in set(anomalies["tx_id"])
    assert "aug-anomaly" not in set(anomalies["tx_id"])
    assert (
        anomalies.loc[anomalies["tx_id"].eq("missing-date"), "issue"]
        .astype(str)
        .eq("missing_date")
        .any()
    )

    manifest = json.loads((out_dir / "meta" / "stage_A_ingest.json").read_text())
    params = manifest["params"]
    assert params["cutoff_date"] == "2026-07-31"
    assert params["cutoff_rule"] == CUTOFF_RULE
    assert params["cutoff_version"] == CUTOFF_VERSION

    check_env = dict(os.environ, OUT_DIR=str(out_dir), MODE="smoke")
    subprocess.run([sys.executable, "scripts/check_ingest.py"], check=True, env=check_env)
    check = json.loads((out_dir / "meta" / "check_A_ingest.json").read_text())
    assert check["ok"] is True
    names = {row["name"] for row in check["checks"]}
    assert "ledger_canonical_within_cutoff" in names
    assert "ledger_canonical_all_status_within_cutoff" in names


def test_cutoff_filters_all_status_before_status_recognition_and_records_contract(tmp_path):
    _assert_cutoff_outputs(_run_ingest(tmp_path))


def test_cutoff_can_arrive_from_environment_for_make_style_execution(tmp_path):
    _assert_cutoff_outputs(_run_ingest(tmp_path, env_cutoff=True))
