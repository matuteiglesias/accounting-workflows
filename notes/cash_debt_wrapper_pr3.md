# Cash and debt consumption wrappers PR 3

## What this PR adds

This PR adds two backend-owned monthly consumption tables:

- `monthly_cash_close.csv`
- `monthly_debt_position.csv`
- `monthly_debt_activity.csv`

It also adds QA files:

- `monthly_cash_close_qa.csv`
- `monthly_debt_position_qa.csv`
- `monthly_debt_activity_qa.csv`

The change is additive. Existing cash, box motor, party balance, debt engine, metrics, and legacy outputs are not removed or renamed.

## Sources used

`monthly_cash_close.csv` is built from explicitly separated inputs. `daily_cash_position.csv` contributes last-observed internal party balances by `Box`, `party`, and `Currency`, and `box_balance_time_long.freq=<FREQ>.csv` contributes inferred/reconciliation box motor rows when available. Both are always non-frontend-safe. An optional `validated_cash_close.csv` can contribute real cash close rows only when each row has an allowed `source_type`, a non-empty `validated_by`, and an explicit validation status (`validated`, `approved`, or `reconciled`).

`monthly_debt_position.csv` is built from `debt_balance_monthly.csv`, which is produced by the existing debt balance view stage. `debt_open_items.csv` is used only to populate source row counts when available.

`monthly_debt_activity.csv` is built as a movement wrapper over `monthly_debt_position.csv` plus available debt engine event outputs (`debt_open_items.csv` and `debt_repayment_events.csv`). It exposes opening balance, closing balance, net change, new principal claims, interest accrual, repayments, and visible residual adjustments by period/debtor/creditor/currency.

## Cash suitability

Cash rows always carry explicit suitability metadata:

- `position_type`
- `cash_suitability`
- `is_frontend_safe`
- `caveat`

Party-level `daily_cash_position` rows are marked as `internal_balance`, `internal_only`, and `is_frontend_safe=false`. Box motor rows are marked as `inferred_box_motor`, `safe_with_caveat`, and `is_frontend_safe=false`. Optional validated cash rows are marked as `cash_close`, `frontend_safe`, and `is_frontend_safe=true` only after explicit row-level validation.

## Why box/party balances are not automatically cash

Party balances and box motors can represent internal claims, inferred movements, or reconciliation views. They are useful for audit and operations, but they are not automatically bank/account cash. This PR deliberately emits no frontend-safe cash total unless `monthly_cash_close.csv` contains validated rows with `is_frontend_safe=true`.

## Debt wrapper

`monthly_debt_position.csv` wraps the debt engine output into debtor/creditor/currency/month/component rows for `principal`, `interest`, and `total`. It does not alter debt allocation or resolution logic.

`monthly_debt_activity.csv` is the debt movement source. It is separate from operating statements and from debt stock. Residual differences between opening, known activity, and closing are emitted as `activity_type=adjustment` with `reconciliation_status`, not hidden or moved into OPEX.

## Known caveats

If debt balance inputs are missing, the position and activity wrappers can emit empty outputs plus QA warnings. If no validated cash input exists, cash close still builds from available diagnostic/internal sources, but dashboard cash remains unavailable because no rows are `is_frontend_safe=true`.

## Follow-up for metrics frontier

Metrics frontier and human reports should expose cash only from `monthly_cash_close.csv` rows where `is_frontend_safe=true`. No frontend-safe cash metric should be published unless `position_type`, `cash_suitability`, and `is_frontend_safe` are explicit. Debt stock metrics should source from `monthly_debt_position.csv`; debt movement metrics should source from `monthly_debt_activity.csv`; neither should source from operating flows.
