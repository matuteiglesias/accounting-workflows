# Monthly treasury accountability — frozen implementation contract

Status: implementation contract for the initial backend PR, 2026-08-20.

## Invariant

Economic attribution does not imply cash movement. Actual Box cash requires
physical counterparty evidence:

```text
direction_source == box_party_match
and direction in {in, out}
```

Semantic fallback, rule defaults, direct third-party payments and non-cash
support may explain economics/accountability but must never manufacture cash.

## Products

```text
classified transactions in memory
    -> monthly_box_treasury_flow.csv
    -> physical Box motor reconciliation
    -> debt resolution
    -> monthly_cash_accountability.csv
```

`monthly_box_treasury_flow.csv` is long and traceable. It preserves semantic
bucket/subbucket, cash category, movement basis, direction source, funding
dimensions, review state, transaction counts and sample transaction IDs.

`monthly_cash_accountability.csv` is one row per month / Box / native currency.
It composes effective cash movements, non-cash support, inferred zero-origin
Box control, governed validated cash snapshots and debt repayment cross-checks.

## Hard failures

- treasury cash net != Box motor net;
- Box flow motor net != Box balance motor net;
- cash components != total in/out/net;
- opening control + net cash flow != closing control;
- semantic/non-cash evidence alters cash arithmetic;
- currency/Box keys cannot reconcile.

## Visible residuals, not hard failures

- unknown or review-required actual cash;
- cash repayment vs debt-engine allocation gap;
- validated cash anchor offsets that disagree;
- validated account snapshots with incompatible as-of dates.

## Cash levels

`opening_control` and `closing_control` are zero-origin inferred movement
controls, not validated liquidity. `validated_cash_close` remains governed by
the existing cash authority. A temporally aligned validated anchor can reveal
an opening offset:

```text
validated_anchor_offset = validated_cash_close - closing_control
```

Two aligned anchors should imply the same offset; differences are explicit
reconciliation residuals.

## Currency

ARS and USD are parallel native-currency books. This product performs no FX
conversion and never sums across currencies. USD/CCL projection remains a
separate derived valuation surface.

## Non-goals

No HTML/report logic, no annual formula changes, no debt allocation rule
changes, no cash authority changes, no semantic classification changes, and no
ownership/governance inference.
