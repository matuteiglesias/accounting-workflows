from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from accounting.marts.semantic import _infer_box_party


REQUIRED_COLUMNS = {"tx_id", "amount", "Currency", "Box", "payer", "receiver"}


def _local_path(value: str, label: str) -> Path:
    if "://" in value:
        raise ValueError(f"{label} must be a local filesystem path")
    return Path(value).expanduser().resolve()


def build_amount_direction_diagnostic(
    ledger_path: Path, output_dir: Path, examples_per_group: int = 3
) -> dict[str, Path]:
    """Characterize amount signs and party direction without changing the ledger."""
    ledger_path = Path(ledger_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not ledger_path.is_file():
        raise FileNotFoundError(f"ledger does not exist: {ledger_path}")
    if examples_per_group < 0:
        raise ValueError("examples_per_group must be non-negative")

    ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS - set(ledger.columns))
    if missing:
        raise ValueError(f"ledger missing required columns: {missing}")

    work = ledger.copy()
    work["amount_numeric"] = pd.to_numeric(work["amount"], errors="coerce")
    work["amount_sign"] = "invalid"
    work.loc[work["amount_numeric"].lt(0), "amount_sign"] = "negative"
    work.loc[work["amount_numeric"].eq(0), "amount_sign"] = "zero"
    work.loc[work["amount_numeric"].gt(0), "amount_sign"] = "positive"

    box_party = work["Box"].map(_infer_box_party).str.upper()
    payer_is_box = work["payer"].str.strip().str.upper().eq(box_party) & box_party.ne("")
    receiver_is_box = work["receiver"].str.strip().str.upper().eq(box_party) & box_party.ne("")
    work["party_direction"] = "neither"
    work.loc[payer_is_box & ~receiver_is_box, "party_direction"] = "payer_is_box"
    work.loc[receiver_is_box & ~payer_is_box, "party_direction"] = "receiver_is_box"
    work.loc[payer_is_box & receiver_is_box, "party_direction"] = "internal"

    def aggregate(keys: list[str]) -> pd.DataFrame:
        grouped = work.groupby(keys, dropna=False, sort=True)
        result = grouped.agg(
            row_count=("tx_id", "size"),
            amount_sum_native=("amount_numeric", "sum"),
            amount_abs_sum_native=("amount_numeric", lambda values: values.abs().sum()),
        ).reset_index()
        if keys == ["Box", "Currency"]:
            counts = pd.crosstab(
                [work["Box"], work["Currency"]], work["amount_sign"]
            ).reindex(columns=["negative", "zero", "positive", "invalid"], fill_value=0)
            counts.columns = [f"{name}_rows" for name in counts.columns]
            result = result.merge(counts.reset_index(), on=keys, how="left")
        return result

    by_box_currency = aggregate(["Box", "Currency"])
    direction_sign = aggregate(["party_direction", "amount_sign"])

    example_columns = [
        column
        for column in [
            "tx_id", "Date", "amount", "Currency", "Box", "payer", "receiver",
            "Flujo", "Tipo", "Detalle", "party_direction", "amount_sign",
        ]
        if column in work.columns
    ]
    examples = (
        work.sort_values(["party_direction", "amount_sign", "tx_id"], kind="stable")
        .groupby(["party_direction", "amount_sign"], sort=True, group_keys=False)
        .head(examples_per_group)[example_columns]
    )

    summary = {
        "source_ledger": str(ledger_path),
        "row_count": int(len(work)),
        "amount_sign_counts": {
            sign: int(work["amount_sign"].eq(sign).sum())
            for sign in ["negative", "zero", "positive", "invalid"]
        },
        "party_direction_counts": {
            direction: int(work["party_direction"].eq(direction).sum())
            for direction in ["payer_is_box", "receiver_is_box", "internal", "neither"]
        },
        "examples_per_group": examples_per_group,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "amount_direction_summary.json",
        "by_box_currency": output_dir / "amount_by_box_currency.csv",
        "direction_sign_matrix": output_dir / "direction_sign_matrix.csv",
        "examples": output_dir / "amount_direction_examples.csv",
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    by_box_currency.to_csv(paths["by_box_currency"], index=False)
    direction_sign.to_csv(paths["direction_sign_matrix"], index=False)
    examples.to_csv(paths["examples"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only amount-sign and Box-party direction diagnostic."
    )
    parser.add_argument("--ledger", required=True, help="Explicit local canonical ledger path")
    parser.add_argument("--output-dir", required=True, help="Explicit output directory")
    parser.add_argument("--examples-per-group", type=int, default=3)
    args = parser.parse_args()
    paths = build_amount_direction_diagnostic(
        _local_path(args.ledger, "--ledger"),
        _local_path(args.output_dir, "--output-dir"),
        args.examples_per_group,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
