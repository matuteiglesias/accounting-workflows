from __future__ import annotations

import pandas as pd

from accounting.marts.debt import build_monthly_debt_position


def test_build_monthly_debt_position_selects_latest_monthly_snapshot(tmp_path):
    debt_dir = tmp_path / "debt"
    out_dir = tmp_path / "out"
    debt_dir.mkdir()
    pd.DataFrame(
        [
            {
                "as_of_date": "2025-03-19",
                "period": "2025-03",
                "debtor": "PM",
                "creditor": "MI",
                "currency": "usd",
                "open_principal": 8804.2,
                "open_interest": 104.0,
                "open_total": 8908.2,
            },
            {
                "as_of_date": "2025-03-31",
                "period": "2025-03",
                "debtor": "PM",
                "creditor": "MI",
                "currency": "usd",
                "open_principal": 8726.2,
                "open_interest": 0.0,
                "open_total": 8726.2,
            },
        ]
    ).to_csv(debt_dir / "debt_balance_monthly.csv", index=False)

    paths = build_monthly_debt_position(debt_dir, out_dir)
    out = pd.read_csv(paths["monthly_debt_position"])

    assert set(out["as_of_date"]) == {"2025-03-31"}
    principal = out[out["component"].eq("principal")].iloc[0]
    total = out[out["component"].eq("total")].iloc[0]
    assert principal["open_amount"] == 8726.2
    assert total["open_amount"] == 8726.2


def test_monthly_dynamics_notebook_debt_position_matrix_uses_latest_snapshot():
    import json
    from pathlib import Path

    notebook = Path("accounting/notebooks/accounting_reports/07_monthly_dynamics_tables.ipynb")
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if "def matrix_debt_position" in "".join(cell.get("source", []))
    )
    namespace = {"pd": pd, "re": __import__("re")}
    exec(source, namespace)

    debt_pos = pd.DataFrame(
        [
            {"period": "2025-03", "as_of_date": "2025-03-19", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 8804.2},
            {"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "principal", "open_amount": 8726.2},
            {"period": "2025-03", "as_of_date": "2025-03-19", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "total", "open_amount": 8908.2},
            {"period": "2025-03", "as_of_date": "2025-03-31", "Currency": "USD", "debtor": "PM", "creditor": "MI", "component": "total", "open_amount": 8726.2},
        ]
    )

    matrix = namespace["matrix_debt_position"](debt_pos)

    principal = matrix[matrix["measure"].eq("open_principal")].iloc[0]
    total = matrix[matrix["measure"].eq("open_total")].iloc[0]
    assert principal["2025-03"] == 8726.2
    assert total["2025-03"] == 8726.2
