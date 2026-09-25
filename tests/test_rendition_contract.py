from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from accounting.cutoff import cutoff_metadata
from accounting.rendition.scope import PropertyRegistry
from accounting.rendition.spec import load_rendition_spec
from accounting.rendition.validation import validate_rendition_context
from accounting.scope import scope_metadata


def _run(tmp_path: Path, run_id: str = "20260930T000000Z_FBPM", cutoff: str = "2026-09-30") -> Path:
    root = tmp_path / run_id
    (root / "meta").mkdir(parents=True)
    params = {}
    params.update(scope_metadata({"Family Business", "Property Management"}))
    params.update(cutoff_metadata(cutoff))
    (root / "meta" / "stage_A_ingest.json").write_text(
        json.dumps({"params": params}), encoding="utf-8"
    )
    return root


def _spec_payload(run_id: str = "20260930T000000Z_FBPM") -> dict:
    return {
        "schema": "accounting.rendition_spec.v1",
        "rendition_id": "RDC-MI-2026-09-30-001",
        "source_run_id": run_id,
        "rendidor": {"display_name": "Matias Iglesias"},
        "period": {"from": "2026-01-01", "through": "2026-09-30"},
        "properties": ["BALBIN_4148", "TIGRE_01"],
        "boxes": ["Property Management", "Family Business"],
        "recipients": ["Eduardo Iglesias", "Alejandro Iglesias"],
        "purpose": "private_account_rendition",
        "language": "es-AR",
    }


def _write_spec(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "rendition.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _write_registry(tmp_path: Path, rows: str | None = None) -> Path:
    path = tmp_path / "properties.csv"
    path.write_text(
        rows
        or (
            "property_id,display_name,lugar_selector,box_selector,optional_notes\n"
            "BALBIN_4148,Balbin 4148,Balbin 4148,Property Management,reporting only\n"
            "TIGRE_01,Tigre 01,Tigre 01,Family Business,reporting only\n"
        ),
        encoding="utf-8",
    )
    return path


def test_valid_rendition_spec_binds_one_exact_run_and_reporting_registry(tmp_path: Path) -> None:
    run_root = _run(tmp_path)
    spec = load_rendition_spec(_write_spec(tmp_path, _spec_payload()))
    registry = PropertyRegistry.from_csv(_write_registry(tmp_path))

    result = validate_rendition_context(spec, registry, run_root)

    assert result.status == "pass"
    assert result.source_run_id == run_root.name
    assert result.properties == ("BALBIN_4148", "TIGRE_01")
    assert set(result.boxes) == {"Property Management", "Family Business"}


def test_rendition_rejects_mixed_run(tmp_path: Path) -> None:
    run_root = _run(tmp_path, run_id="run_A")
    spec = load_rendition_spec(_write_spec(tmp_path, _spec_payload(run_id="run_B")))
    registry = PropertyRegistry.from_csv(_write_registry(tmp_path))

    with pytest.raises(ValueError, match="mix accounting runs"):
        validate_rendition_context(spec, registry, run_root)


def test_rendition_rejects_period_after_immutable_run_cutoff(tmp_path: Path) -> None:
    run_root = _run(tmp_path, cutoff="2026-09-15")
    spec = load_rendition_spec(_write_spec(tmp_path, _spec_payload()))
    registry = PropertyRegistry.from_csv(_write_registry(tmp_path))

    with pytest.raises(ValueError, match="exceeds immutable run cutoff"):
        validate_rendition_context(spec, registry, run_root)


def test_rendition_rejects_unknown_property_id(tmp_path: Path) -> None:
    run_root = _run(tmp_path)
    payload = _spec_payload()
    payload["properties"] = ["UNKNOWN_PROPERTY"]
    spec = load_rendition_spec(_write_spec(tmp_path, payload))
    registry = PropertyRegistry.from_csv(_write_registry(tmp_path))

    with pytest.raises(ValueError, match="unknown property_id"):
        validate_rendition_context(spec, registry, run_root)


def test_property_registry_rejects_ambiguous_reporting_selector(tmp_path: Path) -> None:
    rows = (
        "property_id,display_name,lugar_selector,box_selector,optional_notes\n"
        "SITE_A,Site A,Same Place,Property Management,\n"
        "SITE_B,Site B,Same Place,Property Management,\n"
    )
    with pytest.raises(ValueError, match="ambiguous property selector"):
        PropertyRegistry.from_csv(_write_registry(tmp_path, rows))


def test_spec_rejects_ownership_percentage_semantics(tmp_path: Path) -> None:
    payload = _spec_payload()
    payload["ownership_percentage"] = 50
    with pytest.raises(ValueError, match="title percentages"):
        load_rendition_spec(_write_spec(tmp_path, payload))


def test_spec_rejects_synthetic_currency_authority(tmp_path: Path) -> None:
    payload = _spec_payload()
    payload["currency"] = "ARS+USD"
    with pytest.raises(ValueError, match="ARS/USD"):
        load_rendition_spec(_write_spec(tmp_path, payload))


def test_property_registry_rejects_legal_semantic_columns(tmp_path: Path) -> None:
    rows = (
        "property_id,display_name,lugar_selector,box_selector,ownership_percentage\n"
        "SITE_A,Site A,Site A,Property Management,50\n"
    )
    with pytest.raises(ValueError, match="legal/title semantics"):
        PropertyRegistry.from_csv(_write_registry(tmp_path, rows))


def test_selected_property_box_must_be_declared_by_spec(tmp_path: Path) -> None:
    run_root = _run(tmp_path)
    payload = _spec_payload()
    payload["boxes"] = ["Property Management"]
    spec = load_rendition_spec(_write_spec(tmp_path, payload))
    registry = PropertyRegistry.from_csv(_write_registry(tmp_path))

    with pytest.raises(ValueError, match="outside rendition spec boxes"):
        validate_rendition_context(spec, registry, run_root)
