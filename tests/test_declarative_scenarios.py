from __future__ import annotations

import csv
from pathlib import Path

import pytest

from accounting.scenarios.runner import read_definition, transform_all_status


FIELDS = ["tx_id", "Date", "amount", "Currency", "status", "Box", "payer",
          "receiver", "Flujo", "Tipo", "Lugar", "Detalle", "tag"]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _sample() -> list[dict]:
    base = {"Box": "Property Management", "payer": "Inq", "receiver": "PM",
            "Flujo": "Cobros", "Tipo": "Renta", "Lugar": "CABA", "Detalle": "Chino Juan"}
    rows = []
    for i, month in enumerate(["2026-12", "2027-01", "2027-02", "2027-03"]):
        for currency, amount in [("ARS", "400000"), ("USD", "380")]:
            rows.append(dict(base, tx_id=f"id{i}{currency}", Date=month + "-10",
                             amount=amount, Currency=currency, status="planeado",
                             tag=f"PM_RENT_CABA_CHINO_{currency}"))
    rows.append(dict(base, tx_id="real_payment", Date="2026-10-10", amount="380",
                     Currency="USD", status="pagado", tag="PM_RENT_CABA_CHINO_USD"))
    return rows


def _op() -> dict:
    return {"op": "suppress_planned",
            "equals": {"Box": "Property Management", "status": "planeado", "Tipo": "Renta"},
            "tags": ["PM_RENT_CABA_CHINO_ARS", "PM_RENT_CABA_CHINO_USD"],
            "from_month": "2026-12", "through_month": "2027-03",
            "expect": {"rows": 8, "amount_by_currency": {"ARS": 1600000, "USD": 1520}}}


def test_real_scenario_yaml_is_valid() -> None:
    repo = Path(__file__).resolve().parents[1]
    obj = read_definition(repo / "scenarios/chino_juan_no_paga_desde_2026_12.yaml")
    assert [v["id"] for v in obj["variants"]] == ["base_planeada", "chino_no_paga"]


def test_planned_suppression_keeps_realized_and_reconciles_by_currency(tmp_path: Path) -> None:
    source, out, baseline = tmp_path / "source.csv", tmp_path / "scenario.csv", tmp_path / "base.csv"
    _write(source, _sample())
    assert transform_all_status(source, baseline, []) == []
    assert baseline.read_bytes() == source.read_bytes()
    audit = transform_all_status(source, out, [_op()])
    assert audit == [{"operator": "suppress_planned", "rows": 8,
                      "totals": {"ARS": 1600000.0, "USD": 1520.0}}]
    with out.open(encoding="utf-8", newline="") as fp:
        remaining = list(csv.DictReader(fp))
    assert len(remaining) == 1 and remaining[0]["tx_id"] == "real_payment"


def test_mismatched_expected_amounts_fail_before_output(tmp_path: Path) -> None:
    source, out = tmp_path / "source.csv", tmp_path / "out.csv"
    _write(source, _sample())
    op = _op()
    op["expect"]["amount_by_currency"]["USD"] = 0
    with pytest.raises(ValueError, match="unexpected currency amounts"):
        transform_all_status(source, out, [op])
    assert not out.exists()


def test_mismatched_expected_rows_fail_before_output(tmp_path: Path) -> None:
    source, out = tmp_path / "source.csv", tmp_path / "out.csv"
    _write(source, _sample())
    op = _op()
    op["expect"]["rows"] = 9
    with pytest.raises(ValueError, match="unsafe scenario selector"):
        transform_all_status(source, out, [op])
    assert not out.exists()


def test_operator_does_not_match_paid_even_with_same_tag(tmp_path: Path) -> None:
    source, out = tmp_path / "source.csv", tmp_path / "out.csv"
    _write(source, _sample())
    op = _op()
    op["equals"]["status"] = "pagado"
    op["from_month"] = "2026-10"
    op["through_month"] = "2026-10"
    op["expect"] = {"rows": 1, "amount_by_currency": {"USD": 380}}
    with pytest.raises(ValueError, match="selected realized history"):
        transform_all_status(source, out, [op])
    assert not out.exists()
