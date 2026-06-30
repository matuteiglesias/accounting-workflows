# Backend pruning decisions

This PR does not delete legacy outputs or change formulas. It makes misuse visible.

## Keep for compatibility, but demote

- Stage D aggregate tables remain available as diagnostic/materialized evidence.
- Metric wide/statement views remain available as legacy or presentation-only outputs.
- Raw debt engine outputs remain internal diagnostic evidence.

## Canonical consumption preference

Dashboard and public consumers should prefer:

1. `monthly_operating_statement.csv` for operating result sections.
2. `monthly_cash_close.csv` only where rows are `is_frontend_safe=true` for cash.
3. `monthly_debt_position.csv` for debt stock.
4. `metric_contract_frontier.csv` and `frontend_metric_series.csv` for frontend metrics.

## Future pruning candidates

- Direct metrics dependencies on `ledger_canonical.csv`, `per_flow_time_long`, `daily_cash_position`, and legacy views.
- Public publication of legacy annual/quarterly metric tables except under clearly labeled legacy namespaces.
