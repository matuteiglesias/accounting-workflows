# Accounting source-of-truth contracts

The dashboard contract is layered and explicit:

```text
ledger_canonical
→ semantic mart
→ monthly operating statement
→ cash wrapper
→ debt wrapper
→ metrics frontier
→ dashboard / human reports / publish
```

## Canonical/report-safe chain

- `ledger_canonical.csv` is the canonical transaction source. It is not the preferred dashboard input once semantic/cash/debt marts exist.
- `monthly_flow_semantic_split.csv` is the canonical monthly semantic flow split.
- `monthly_operating_statement.csv` is the canonical monthly operating statement.
- `monthly_box_treasury_flow.csv` is the canonical monthly effective Box cash-flow mart. Economic attribution alone never establishes cash: actual cash requires physical Box-counterparty evidence.
- `monthly_cash_accountability.csv` is the canonical monthly treasury accountability composition. It reconciles treasury flow to inferred Box control, keeps validated cash separate, and cross-checks debt repayments without using debt activity to manufacture cash.
- `monthly_cash_close.csv` is the cash wrapper. It is source-of-truth for frontend cash only at row level when `is_frontend_safe=true`.
- `monthly_debt_position.csv` is the canonical debt stock wrapper.
- `metric_contract_frontier.csv` and `frontend_metric_series.csv` are the frontend metric contract.

## Diagnostic/internal evidence

- `per_flow_time_long.*.csv` is diagnostic flow/type evidence, not semantic dashboard truth.
- `per_party_time_long.*.csv` is actor movement evidence/internal balance, not real cash.
- `daily_cash_position.csv` is a party-level internal balance/claim view, not account-level cash.
- `box_balance_time_long.*.csv` and `box_flow_balance_time_long.*.csv` are inferred reconciliation/box motor artifacts, not validated liquidity. `monthly_cash_accountability.csv` may expose them only as `opening_control`/`closing_control` reconciliation controls, never as real cash.
- Debt engine raw files (`debt_open_items.csv`, `debt_repayment_events.csv`, `debt_status_reconciliation.csv`) are diagnostic evidence. Use `monthly_debt_position.csv` for report-safe debt stock.

## Presentation and legacy outputs

- `metric_views/*.csv` are presentation-only convenience views.
- `income_statement_y.csv`, `balance_cash_y.csv`, `balance_debt_y.csv`, and quarterly equivalents are legacy compatibility outputs.

Every important artifact should carry `artifact_role`, `accounting_nature`, `grain`, `currency_policy`, `frontend_suitability`, and `source_authority` in manifests or artifact contract CSVs.
