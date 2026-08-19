from __future__ import annotations

from pathlib import Path

import pandas as pd

from accounting.marts.semantic import build_monthly_operating_statement_from_split


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "fixtures" / "semantic_measure_statement_input.csv"
EXPECTED = ROOT / "fixtures" / "semantic_measure_statement_expected.csv"


def test_monthly_operating_statement_is_byte_identical_to_pre_migration_fixture(
    tmp_path: Path,
) -> None:
    statement, _ = build_monthly_operating_statement_from_split(pd.read_csv(INPUT))
    actual = tmp_path / "monthly_operating_statement.csv"
    statement.to_csv(actual, index=False, lineterminator="\n")

    assert actual.read_bytes() == EXPECTED.read_bytes()
