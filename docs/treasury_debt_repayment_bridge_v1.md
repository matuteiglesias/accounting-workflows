# Treasury–debt repayment bridge v1

## Scope
Compare actual Box repayment movements and resolved debt repayment events from the same exact run by canonical transaction identity. The former Box-name/debtor-name aggregate join did not establish event coverage. The bridge does not change cash, debt, interest or classification authority.

## Inputs and output
Inputs: box_treasury_transaction_detail.csv, monthly_debt_repayment_detail.csv, ledger_canonical.csv (explicit FX marker and CCL). Output: treasury_debt_repayment_bridge.csv, one row per union of repayment identities. Repeated event gross amounts are not summed over allocation rows. Principal, interest and unallocated balance reconcile to gross debt event amount. Duplicate cash keys, inconsistent repeated metadata and duplicate allocation targets fail closed.

Cash currency and amount remain native. Debt currency, event amount, principal and interest allocations and surplus remain in obligation denomination. An explicitly tagged ARS settlement uses recorded CCL and cent rounding; date, payer, receiver and amount must match. No currency totals across lanes.

Statuses: matched (fully allocated), partial (unapplied payment), cash_only, debt_only (no actual Box cash), mismatch. Missing required input lineage yields unavailable monthly reconciliation rather than a fabricated zero. Debt-only is review evidence, not proof of missing cash.

## Compatibility
The existing monthly debt_engine_repayments field is now matched gross outgoing native cash, not allocated debt principal. debt_repayment_gap is native outgoing cash minus matched gross native cash. Allocation surpluses and incoming event issues remain distinct exceptions, with details in the private bridge and QA. Existing report schemas and cash motor arithmetic are preserved.

## Verification and limitations
Synthetic tests cover split allocations, incoming cash, explicit FX, unallocated balance, missing event, debt-only settlement, duplicate target and missing lineage. Private historical replay with verified source corrections applied passed event coverage and native-currency reconciliation. Detailed results are retained privately, not in this repository. This is not a fresh live ingestion or publication. Historical opening balances and interest policy remain subject to separate validation. The diagnostic is internal only and must not be published as source data.
