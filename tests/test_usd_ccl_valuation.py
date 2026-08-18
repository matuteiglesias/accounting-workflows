from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from accounting.artifacts.manifest import artifact_contract_for_name
from accounting.valuation.usd_ccl import ValuationContractError, build_usd_ccl_valuation


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "fixtures" / "ledger_valuation_fixture.csv"
RATES = ROOT / "fixtures" / "synthetic_ccl_rates.csv"
POLICY = ROOT / "fixtures" / "valuation_policy_v1.json"
PREVIOUS_POLICY = ROOT / "reference" / "fx" / "ccl_txn_prev_available_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(tmp_path: Path, *, ledger: Path = LEDGER, rates: Path = RATES, name: str = "valuation"):
    return build_usd_ccl_valuation(
        ledger_path=ledger,
        rates_path=rates,
        policy_path=POLICY,
        output_dir=tmp_path / name,
        run_id="test-run",
        source_scope_tag="FBPM",
    )


def _sidecar_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_exact_date_sidecar_preserves_ledger_and_reconciles_statuses(tmp_path: Path) -> None:
    before = LEDGER.read_bytes()
    outputs = _build(tmp_path)
    after = LEDGER.read_bytes()
    assert before == after

    rows = _sidecar_rows(outputs["sidecar"])
    by_id = {row["tx_id"]: row for row in rows}
    assert len(rows) == 5
    assert len(by_id) == 5
    assert by_id["tx-usd"]["amount_usd_ccl"] == "75.250000"
    assert by_id["tx-usd"]["fx_rate_to_usd_ccl"] == "1.000000000000000000"
    assert by_id["tx-usd"]["fx_conversion_status"] == "identity_native_usd"
    assert by_id["tx-ars-exact"]["amount_usd_ccl"] == "100.000000"
    assert by_id["tx-ars-exact"]["fx_conversion_status"] == "converted_exact_date"
    assert by_id["tx-ars-missing"]["amount_usd_ccl"] == ""
    assert by_id["tx-ars-missing"]["fx_conversion_status"] == "unavailable_missing_rate"
    assert by_id["tx-unsupported"]["amount_usd_ccl"] == ""
    assert by_id["tx-unsupported"]["fx_conversion_status"] == "unsupported_currency"
    assert by_id["tx-invalid-native"]["amount_usd_ccl"] == ""
    assert by_id["tx-invalid-native"]["fx_conversion_status"] == "invalid_native_amount"

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["source_ledger_sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["valuation_rows"] == 5
    assert manifest["native_usd_identity_rows"] == 1
    assert manifest["exact_matches"] == 1
    assert manifest["missing_rates"] == 1
    assert manifest["unsupported_currency_rows"] == 1
    assert manifest["invalid_native_rows"] == 1
    assert manifest["generated_by_network_access"] is False
    assert manifest["valuation_artifact_sha256"] == _sha(outputs["sidecar"])


def test_sidecar_is_byte_deterministic_and_input_order_independent(tmp_path: Path) -> None:
    first = _build(tmp_path, name="first")["sidecar"]
    ledger_lines = LEDGER.read_text(encoding="utf-8").splitlines()
    reordered = tmp_path / "ledger_reordered.csv"
    reordered.write_text("\n".join([ledger_lines[0], *reversed(ledger_lines[1:])]) + "\n", encoding="utf-8")
    second = _build(tmp_path, ledger=reordered, name="second")["sidecar"]
    assert first.read_bytes() == second.read_bytes()


def test_rate_revision_changes_only_valuation_identity(tmp_path: Path) -> None:
    ledger_sha = _sha(LEDGER)
    first = _build(tmp_path, name="first")
    revised_rates = tmp_path / "rates_v2.csv"
    revised_rates.write_text(
        RATES.read_text(encoding="utf-8").replace(",1200,", ",1500,"),
        encoding="utf-8",
    )
    second = _build(tmp_path, rates=revised_rates, name="second")
    assert _sha(LEDGER) == ledger_sha
    assert first["sidecar"].read_bytes() != second["sidecar"].read_bytes()
    first_manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    second_manifest = json.loads(second["manifest"].read_text(encoding="utf-8"))
    assert first_manifest["source_ledger_sha256"] == second_manifest["source_ledger_sha256"]
    assert first_manifest["rate_artifact_sha256"] != second_manifest["rate_artifact_sha256"]
    assert first_manifest["valuation_id"] != second_manifest["valuation_id"]
    assert first_manifest["valuation_artifact_sha256"] != second_manifest["valuation_artifact_sha256"]


def test_rerun_accepts_identical_sidecar_and_rejects_conflicting_overwrite(tmp_path: Path) -> None:
    first = _build(tmp_path, name="stable")
    original = first["sidecar"].read_bytes()
    second = _build(tmp_path, name="stable")
    assert second["sidecar"].read_bytes() == original

    revised_rates = tmp_path / "rates_conflict.csv"
    revised_rates.write_text(
        RATES.read_text(encoding="utf-8").replace(",1200,", ",1300,"),
        encoding="utf-8",
    )
    with pytest.raises(ValuationContractError, match="refusing to overwrite"):
        _build(tmp_path, rates=revised_rates, name="stable")


def test_duplicate_ledger_tx_id_fails_closed(tmp_path: Path) -> None:
    lines = LEDGER.read_text(encoding="utf-8").splitlines()
    duplicate = tmp_path / "duplicate_ledger.csv"
    duplicate.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValuationContractError, match="duplicate tx_id"):
        _build(tmp_path, ledger=duplicate)


def test_duplicate_rate_observation_fails_closed(tmp_path: Path) -> None:
    lines = RATES.read_text(encoding="utf-8").splitlines()
    duplicate = tmp_path / "duplicate_rates.csv"
    duplicate.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValuationContractError, match="duplicate canonical rate observation"):
        _build(tmp_path, rates=duplicate)


def test_rate_sha_mismatch_and_url_inputs_fail_before_valuation(tmp_path: Path) -> None:
    with pytest.raises(ValuationContractError, match="SHA mismatch"):
        build_usd_ccl_valuation(
            ledger_path=LEDGER,
            rates_path=RATES,
            policy_path=POLICY,
            output_dir=tmp_path / "sha",
            run_id="test-run",
            source_scope_tag="FBPM",
            expected_rate_sha256="0" * 64,
        )
    with pytest.raises(ValuationContractError, match="local filesystem path"):
        build_usd_ccl_valuation(
            ledger_path=LEDGER,
            rates_path="https://example.invalid/ccl.csv",
            policy_path=POLICY,
            output_dir=tmp_path / "network",
            run_id="test-run",
            source_scope_tag="FBPM",
        )


def test_artifact_contract_keeps_sidecar_outside_canonical_authority() -> None:
    contract = artifact_contract_for_name("ledger_valuation_usd_ccl.csv")
    assert contract["artifact_role"] == "derived_valuation"
    assert contract["source_authority"] == "derived_valuation_evidence"
    assert contract["currency_policy"] == "converted"
    assert contract["frontend_suitability"] == "internal_only"
    coverage = artifact_contract_for_name("valuation_coverage_by_year.csv")
    assert coverage["artifact_role"] == "qa"
    assert coverage["grain"] == "annual"


def test_previous_available_policy_weekends_staleness_and_history(tmp_path: Path) -> None:
    ledger = tmp_path / "dated_ledger.csv"
    ledger.write_text(
        "tx_id,Date,amount,Currency,payer,receiver,Flujo,Tipo,status,Box\n"
        "exact,2026-01-02,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n"
        "saturday,2026-01-03,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n"
        "sunday,2026-01-04,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n"
        "holiday-gap,2026-01-05,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n"
        "five-days,2026-01-07,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n"
        "six-days,2026-01-08,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n"
        "tomorrow-only,2026-01-01,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n"
        "before-history,2025-12-31,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n"
        "native-usd,2026-01-08,10,USD,INQ,PM,Cobros,Renta,recognized,Property Management\n",
        encoding="utf-8",
    )
    rates = tmp_path / "rates.csv"
    rates.write_text(
        "rate_date,ars_per_usd_ccl,rate_source,rate_series,source_reference\n"
        "2026-01-02,1200,SYNTHETIC,CCL_TEST,fixture:2026-01-02\n",
        encoding="utf-8",
    )
    outputs = build_usd_ccl_valuation(
        ledger_path=ledger,
        rates_path=rates,
        policy_path=PREVIOUS_POLICY,
        output_dir=tmp_path / "valuations",
        run_id="dated-policy",
        source_scope_tag="synthetic",
    )
    rows = {row["tx_id"]: row for row in _sidecar_rows(outputs["sidecar"])}

    assert rows["exact"]["fx_conversion_status"] == "converted_exact_date"
    for tx_id, age in [("saturday", "1"), ("sunday", "2"), ("holiday-gap", "3"), ("five-days", "5")]:
        assert rows[tx_id]["fx_conversion_status"] == "converted_previous_available"
        assert rows[tx_id]["fx_rate_date"] == "2026-01-02"
        assert rows[tx_id]["fx_rate_age_days"] == age
        assert rows[tx_id]["amount_usd_ccl"] == "1.000000"
    assert rows["six-days"]["fx_conversion_status"] == "unavailable_stale_rate"
    assert rows["six-days"]["amount_usd_ccl"] == ""
    assert rows["six-days"]["fx_rate_date"] == "2026-01-02"
    assert rows["six-days"]["fx_rate_age_days"] == "6"
    assert rows["tomorrow-only"]["fx_conversion_status"] == "unavailable_missing_history"
    assert rows["before-history"]["fx_conversion_status"] == "unavailable_missing_history"
    assert rows["native-usd"]["fx_conversion_status"] == "identity_native_usd"

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["exact_matches"] == 1
    assert manifest["previous_available_matches"] == 4
    assert manifest["stale_rejections"] == 1
    assert manifest["missing_history_rows"] == 2
    assert manifest["native_usd_identity_rows"] == 1
    assert manifest["max_applied_rate_age_days"] == 5
    assert manifest["ledger_min_date"] == "2025-12-31"
    assert manifest["ledger_max_date"] == "2026-01-08"
    assert sum(manifest[field] for field in [
        "native_usd_identity_rows", "exact_matches", "previous_available_matches",
        "stale_rejections", "missing_history_rows", "missing_exact_date_rows",
        "unsupported_currency_rows", "invalid_native_rows",
    ]) == manifest["valuation_rows"]

    coverage = list(csv.DictReader(outputs["coverage"].open(encoding="utf-8")))
    by_year = {row["year"]: row for row in coverage}
    assert by_year["2026"]["rows"] == "8"
    assert by_year["2026"]["valued"] == "6"
    assert by_year["2026"]["previous_available"] == "4"
    assert by_year["2026"]["stale"] == "1"
    assert by_year["2026"]["missing"] == "1"


def test_content_addressed_identity_and_rate_order_behavior(tmp_path: Path) -> None:
    rates_a = tmp_path / "rates_a.csv"
    rates_a.write_text(
        "rate_date,ars_per_usd_ccl,rate_source,rate_series,source_reference\n"
        "2026-01-04,1210,SYNTHETIC,CCL_TEST,fixture:2026-01-04\n"
        "2026-01-02,1200,SYNTHETIC,CCL_TEST,fixture:2026-01-02\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "tx_id,Date,amount,Currency,payer,receiver,Flujo,Tipo,status,Box\n"
        "weekend,2026-01-03,1200,ARS,INQ,PM,Cobros,Renta,recognized,Property Management\n",
        encoding="utf-8",
    )

    def build(rates: Path, name: str):
        return build_usd_ccl_valuation(
            ledger_path=ledger,
            rates_path=rates,
            policy_path=PREVIOUS_POLICY,
            output_dir=tmp_path / name,
            run_id="content-addressed",
            source_scope_tag="synthetic",
            content_addressed=True,
        )

    first = build(rates_a, "first")
    first_manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    assert first["sidecar"].parent.name == first_manifest["valuation_id"]

    shuffled = tmp_path / "rates_shuffled.csv"
    lines = rates_a.read_text(encoding="utf-8").splitlines()
    shuffled.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n", encoding="utf-8")
    second = build(shuffled, "second")
    assert first["sidecar"].read_bytes() == second["sidecar"].read_bytes()
    second_manifest = json.loads(second["manifest"].read_text(encoding="utf-8"))
    assert second_manifest["valuation_id"] != first_manifest["valuation_id"]

    corrected = tmp_path / "rates_corrected.csv"
    corrected.write_text(rates_a.read_text(encoding="utf-8").replace("1200,", "1000,"), encoding="utf-8")
    third = build(corrected, "third")
    third_manifest = json.loads(third["manifest"].read_text(encoding="utf-8"))
    assert third_manifest["source_ledger_sha256"] == first_manifest["source_ledger_sha256"]
    assert third_manifest["valuation_id"] != first_manifest["valuation_id"]
    assert third["sidecar"].parent != first["sidecar"].parent
