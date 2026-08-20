from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_derived_executor_cannot_rediscover_ledger_or_semantic_membership() -> None:
    executor = (
        ROOT / "accounting" / "professional" / "derived_metric_executor.py"
    ).read_text(encoding="utf-8")

    forbidden = [
        "semantic_bucket",
        "semantic_subbucket",
        "monthly_flow_semantic_split",
        "classification_audit",
        "_annual_formula_spec",
        "_safe_div",
        "Margen operativo",
        "OPEX / renta",
        "Retiros / resultado operativo",
        "Cobertura después de funding y retiros",
    ]
    for token in forbidden:
        assert token not in executor, f"derived executor leaked forbidden authority: {token}"


def test_presentation_labels_are_confined_to_metadata_adapter() -> None:
    contract = (
        ROOT / "accounting" / "contracts" / "derived_metrics.py"
    ).read_text(encoding="utf-8")
    executor = (
        ROOT / "accounting" / "professional" / "derived_metric_executor.py"
    ).read_text(encoding="utf-8")
    metadata = (
        ROOT / "accounting" / "professional" / "derived_metric_metadata.py"
    ).read_text(encoding="utf-8")

    for label in [
        "Margen operativo",
        "OPEX / renta",
        "Retiros / resultado operativo",
        "Cobertura después de funding y retiros",
    ]:
        assert label not in contract
        assert label not in executor
        assert label.casefold() in metadata.casefold()


def test_specialized_deferred_formulas_stay_out_of_generic_executor() -> None:
    executor = (
        ROOT / "accounting" / "professional" / "derived_metric_executor.py"
    ).read_text(encoding="utf-8")
    for token in ["TR.FX.NET", "ID.DEBT.NET_PM_POSITION", "net_flow"]:
        assert token not in executor
