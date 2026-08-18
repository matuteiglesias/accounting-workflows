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
