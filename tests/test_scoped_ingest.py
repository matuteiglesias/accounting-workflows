import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from accounting.metrics.annual import build_annual_balance_dashboard
from accounting.scope import canonical_scope_tag, parse_box_scope


FIXTURE = Path(__file__).parents[1] / "fixtures" / "ledger_scope_fixture.csv"


def _run_scoped_ingest(tmp_path, tag, boxes):
    out_dir = tmp_path / tag
    subprocess.run(
        [
            sys.executable,
            "-m",
            "accounting.ledger.ingest",
            "--fixture",
            str(FIXTURE),
            "--out-dir",
            str(out_dir),
            "--mode",
            "smoke",
            "--run-id",
            tag,
            "--boxes",
            boxes,
        ],
        check=True,
    )
    ledger = pd.read_csv(out_dir / "ledger_canonical.csv")
    manifest = json.loads((out_dir / "meta" / "stage_A_ingest.json").read_text())
    return out_dir, ledger, manifest


def test_scope_tags_are_canonical_and_order_independent():
    assert canonical_scope_tag(parse_box_scope("Family Business,Property Management")) == "FBPM"
    assert canonical_scope_tag(parse_box_scope("Property Management,Family Business")) == "FBPM"
    assert canonical_scope_tag(parse_box_scope("Household")) == "HH"


def test_canonical_ingest_writes_scoped_ledgers_and_metadata(tmp_path):
    source = pd.read_csv(FIXTURE)
    assert source.groupby(["Box", "Currency"])["amount"].agg(["count", "sum"]).to_dict("index") == {
        ("Family Business", "ARS"): {"count": 2, "sum": 50},
        ("Household", "USD"): {"count": 2, "sum": 20},
        ("Property Management", "ARS"): {"count": 2, "sum": 250},
        ("Property Management", "USD"): {"count": 1, "sum": 10},
    }

    fbpm_dir, fbpm, fbpm_manifest = _run_scoped_ingest(
        tmp_path, "fixture_FBPM", "Property Management,Family Business"
    )
    hh_dir, hh, hh_manifest = _run_scoped_ingest(tmp_path, "fixture_HH", "Household")

    assert set(fbpm["Box"]) == {"Family Business", "Property Management"}
    assert fbpm["tx_id"].tolist() == ["fb-native", "pm-native", "fb-pm-fb", "fb-pm-pm", "hh-pm-pm"]
    assert set(hh["Box"]) == {"Household"}
    assert hh["tx_id"].tolist() == ["hh-native", "hh-pm-hh"]

    # Counterparties do not override ownership: PM stays FBPM despite payer HH,
    # while the mirrored HH record remains HH despite receiver PM.
    assert "hh-pm-pm" in set(fbpm["tx_id"])
    assert "hh-pm-hh" not in set(fbpm["tx_id"])
    assert "hh-pm-hh" in set(hh["tx_id"])

    # Both records of the mirrored FB/PM event survive combined selection.
    assert set(fbpm["tx_id"]) >= {"fb-pm-fb", "fb-pm-pm"}

    assert fbpm.groupby(["Box", "Currency"])["amount"].agg(["count", "sum"]).to_dict("index") == {
        ("Family Business", "ARS"): {"count": 2, "sum": 50},
        ("Property Management", "ARS"): {"count": 2, "sum": 250},
        ("Property Management", "USD"): {"count": 1, "sum": 10},
    }
    assert hh.groupby(["Box", "Currency"])["amount"].agg(["count", "sum"]).to_dict("index") == {
        ("Household", "USD"): {"count": 2, "sum": 20},
    }

    expected_scope = {
        "scope_boxes": ["Family Business", "Property Management"],
        "scope_codes": ["FB", "PM"],
        "scope_tag": "FBPM",
        "scope_rule": "box_exact_membership",
        "scope_version": 1,
    }
    assert {key: fbpm_manifest["params"][key] for key in expected_scope} == expected_scope
    assert hh_manifest["params"]["scope_tag"] == "HH"

    # Stage D consumes the immutable Stage A universe even when the caller's
    # environment requests a different scope.
    env = dict(os.environ, BOXES="Household")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "accounting.stage_d.materialize",
            "--out-dir",
            str(fbpm_dir),
            "--mode",
            "smoke",
            "--run-id",
            "fixture_FBPM",
            "--freq",
            "M",
            "--force",
            "0",
        ],
        check=True,
        env=env,
    )
    for filename in ["classification_audit.csv", "monthly_flow_semantic_split.csv"]:
        frame = pd.read_csv(fbpm_dir / filename)
        assert set(frame["Box"].dropna()) <= {"Family Business", "Property Management"}

    subprocess.run(
        [
            sys.executable, "-m", "accounting.stage_d.materialize",
            "--out-dir", str(hh_dir), "--mode", "smoke", "--run-id", "fixture_HH",
            "--freq", "M", "--force", "0",
        ],
        check=True,
        env=dict(os.environ, BOXES="Family Business,Property Management"),
    )
    for filename in ["classification_audit.csv", "monthly_flow_semantic_split.csv"]:
        frame = pd.read_csv(hh_dir / filename)
        assert set(frame["Box"].dropna()) <= {"Household"}

    metrics_hh_env = tmp_path / "metrics_hh_env"
    metrics_fb_env = tmp_path / "metrics_fb_env"
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("BOXES", "Household")
        first = build_annual_balance_dashboard(
            fbpm_dir, metrics_hh_env, run_id="fixture_FBPM", as_of_date="2026-01-31"
        )
        monkeypatch.setenv("BOXES", "Family Business,Property Management")
        second = build_annual_balance_dashboard(
            fbpm_dir, metrics_fb_env, run_id="fixture_FBPM", as_of_date="2026-01-31"
        )
    finally:
        monkeypatch.undo()
    assert (first["annual_balance_dashboard_metrics"].read_bytes()
            == second["annual_balance_dashboard_metrics"].read_bytes())
