from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from accounting.stage_d import materialize as module


@pytest.mark.parametrize("failed_producer", [
    "materialize_per_flow",
    "materialize_per_party",
    "materialize_loans",
    "materialize_daily_cash",
])
def test_required_materialization_failure_is_not_masked_by_existing_file(
    tmp_path: Path, monkeypatch, failed_producer: str,
) -> None:
    ledger = pd.DataFrame([
        {"Date": "2026-01-02", "Box": "Family Business", "Currency": "ARS",
         "Flujo": "Cobros", "Tipo": "Renta", "payer": "Tenant",
         "receiver": "FB", "amount": 100.0, "tag": ""},
    ])
    output_names = {
        "materialize_per_flow": "per_flow_time_long.freq=M.csv",
        "materialize_per_party": "per_party_time_long.freq=M.csv",
        "materialize_box_balance_time_long": "box_balance_time_long.freq=M.csv",
        "materialize_box_flow_balance_time_long": "box_flow_balance_time_long.freq=M.csv",
        "materialize_loans": "loans_time.freq=M.csv",
        "materialize_daily_cash": "daily_cash_position.csv",
    }
    def successful(name: str):
        def producer(_ledger, *args, **kwargs):
            # loans receives loan_register_df before out_dir; other motors don't.
            out_dir = kwargs.get("out_dir") or next(
                value for value in args if isinstance(value, (str, Path))
            )
            path = Path(out_dir) / output_names[name]
            path.write_text("TimePeriod_ts_end\n2026-01-31\n", encoding="utf-8")
            return pd.DataFrame({"TimePeriod_ts_end": ["2026-01-31"]}), path
        return producer
    for name in output_names:
        monkeypatch.setattr(module, name, successful(name))

    expected_path = tmp_path / output_names[failed_producer]
    expected_path.write_text("stale data", encoding="utf-8")
    def fail(*args, **kwargs):
        raise RuntimeError(f"synthetic {failed_producer} failure")
    monkeypatch.setattr(module, failed_producer, fail)
    with pytest.raises(RuntimeError, match=f"synthetic {failed_producer} failure"):
        module.materialize_all(ledger, tmp_path, freq="M")
    # Prior files cannot turn a failed required producer into a successful stage.
    assert expected_path.read_text(encoding="utf-8") == "stale data"
    assert not (tmp_path / "partitions.json").exists()
