# Backend pruning decisions

## Canonical monthly path

The authoritative accounting path is:

```text
ledger_canonical.csv
  -> semantic classifier / classification_audit.csv
  -> monthly_flow_semantic_split.csv
  -> monthly_operating_statement.csv
  -> frontend_metric_series.csv / compact reports / notebooks
```

`monthly_operating_statement.csv` is the only source for operating statement presentation views. Legacy monthly income views are compatibility presentations and are not allowed to reclassify ledger rows.

## Cash

Cash metrics are unavailable unless `monthly_cash_close.csv` contains rows with `is_frontend_safe=true`. If there are no safe rows, `BS.CASH.TOTAL` has unavailable contract status, no cash series rows are emitted, and reports must show `s/d`/blocked rather than reconstructing cash from party balances, box motor, or daily cash rows.

## Debt

Debt is a stock, remains currency-aware, and is not summed into rent, OPEX, withdrawals, or ARS operating flows. Debt frontend rows must carry `Currency` and use debt metric identifiers/sources from `monthly_debt_position.csv`.

## Demoted unsafe paths

The following are retained only for compatibility, reconciliation, or diagnostics:

- `per_flow_time_long.freq=M.csv`: raw flow evidence, not semantic reporting.
- `box_balance_time_long.freq=M.csv`: inferred box motor/reconciliation, not real cash.
- `per_party_time_long.freq=M.csv`: actor/internal balance evidence, not real cash.
- `income_statement_monthly_last6.csv`: presentation only, now derived from canonical monthly statement.
- wide/pivot metric tables: presentation only, not automation sources.
- `IS.NET.AFTER_COSTS`: legacy/coverage-like metric when it mixes funding and operating income.

## QA enforcement

Automated or warning-row checks now cover:

- semester month count <= 6 and compact totals reconciliation (compact builder QA);
- no silent cross-currency frontend aggregation;
- Currency column presence for frontend money outputs;
- no debt stock mixed with ARS flow rows without Currency;
- cash metrics unavailable without frontend-safe cash rows;
- property OPEX leakage patterns for personal, dividend, transfer/gasto, distribution, and draw text;
- frontier source whitelist and no wide/pivot canonical sources;
- notebook flow-classification prohibition documented as a high-severity contract warning until notebooks are fully converted to backend-only sourcing.
