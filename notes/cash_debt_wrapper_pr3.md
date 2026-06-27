# Cash and debt consumption wrappers PR 3

## What this PR adds

This PR adds two backend-owned monthly consumption tables:

- `monthly_cash_close.csv`
- `monthly_debt_position.csv`

It also adds QA files:

- `monthly_cash_close_qa.csv`
- `monthly_debt_position_qa.csv`

The change is additive. Existing cash, box motor, party balance, debt engine, metrics, and legacy outputs are not removed or renamed.

## Sources used

`monthly_cash_close.csv` is built from `daily_cash_position.csv` using the last observed balance in each month by `Box`, `party`, and `Currency`. If available, `box_balance_time_long.freq=<FREQ>.csv` is included as an inferred/reconciliation box motor row.

`monthly_debt_position.csv` is built from `debt_balance_monthly.csv`, which is produced by the existing debt balance view stage. `debt_open_items.csv` is used only to populate source row counts when available.

## Cash suitability

Cash rows always carry explicit suitability metadata:

- `position_type`
- `cash_suitability`
- `is_frontend_safe`
- `caveat`

Party-level `daily_cash_position` rows are marked as `internal_balance`, `internal_only`, and `is_frontend_safe=false`. Box motor rows are marked as `inferred_box_motor`, `safe_with_caveat`, and `is_frontend_safe=false`.

## Why box/party balances are not automatically cash

Party balances and box motors can represent internal claims, inferred movements, or reconciliation views. They are useful for audit and operations, but they are not automatically bank/account cash. This PR deliberately emits no frontend-safe cash total unless a future source can justify real cash-close semantics.

## Debt wrapper

`monthly_debt_position.csv` wraps the debt engine output into debtor/creditor/currency/month/component rows for `principal`, `interest`, and `total`. It does not alter debt allocation or resolution logic.

## Known caveats

If debt balance inputs are missing, the wrapper can emit an empty output plus QA warnings. Cash close requires `daily_cash_position.csv` and fails clearly if that Stage D source is missing.

## Follow-up for metrics frontier

Future PRs should decide which cash/debt rows are safe to expose in `metric_values.csv`, human reports, and frontend snapshots. No frontend-safe cash metric should be published unless `position_type`, `cash_suitability`, and `is_frontend_safe` are explicit.
