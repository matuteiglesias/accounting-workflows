# Debt position and debt activity contracts v1

Wave 4 PR12 introduces typed contracts for resolved-debt **position** and **activity** only. It does not migrate professional consumers and it does not define cash authority.

## Boundary

The contracts deliberately preserve three different semantic classes:

- atomic operating/management flows remain governed by `FlowCellSpec` from Wave 3;
- debt position is a **stock snapshot**;
- debt activity is a **period flow over resolved debt**.

No generic stock/flow executor is introduced in this PR.

## DebtPositionSpec

Registry version: `debt_position_specs_v1`.

Three specs preserve the current characterized physical value columns:

| spec_id | component | value_ref |
|---|---|---|
| `debt.position.principal` | principal | `open_principal` |
| `debt.position.interest` | interest | `open_interest` |
| `debt.position.total` | total | `open_total` |

All position specs declare:

```text
source_contract = monthly_debt_position
grain = period, Currency, debtor, creditor, component
aggregation = snapshot
selection = latest_valid_as_of_date
as_of_field = as_of_date
invalid_as_of_policy = unavailable
annualization = latest_period_then_latest_valid_as_of_date
```

The v1 contract intentionally does **not** normalize all components to `open_amount`, even though that column is component-specific in `monthly_debt_position.csv`. PR11 characterized the existing professional consumption as using `open_principal`, `open_interest`, and `open_total`; changing that physical authority would require separate parity evidence.

### Invalid/missing as-of policy

PR11 froze an undesirable current edge case: if every candidate `as_of_date` is invalid, the current mart can select a row through secondary string ordering. The v1 contract does not canonize that fallback.

The governed target is:

> A debt stock must have a valid as-of observation. If no valid `as_of_date` exists for the candidate position set, the future consumer returns unavailable/review rather than choosing a lexical fallback.

This contract declaration does not change production behavior yet. PR13 must make any before/after effect on invalid-as-of fixtures explicit while preserving parity for valid snapshots.

## DebtActivitySpec

Registry version: `debt_activity_specs_v1`.

Five activity specs make the sparse-row mapping explicit:

| spec_id | activity_type | measure_ref |
|---|---|---|
| `debt.activity.new_claim` | new_claim | `new_principal` |
| `debt.activity.interest_accrual` | interest_accrual | `interest_accrued` |
| `debt.activity.repayment` | repayment | `repayments` |
| `debt.activity.adjustment` | adjustment | `adjustments` |
| `debt.activity.net_change` | net_change | `net_change` |

All activity specs declare:

```text
source_contract = monthly_debt_activity
grain = period, Currency, debtor, creditor, activity_type
aggregation = sum_flow
annualization = sum_periods
```

`opening_balance` and `closing_balance` rows from the debt activity wrapper are deliberately not governed as activity-flow specs. They are stock/control context and must not be smuggled into a SUM-flow executor merely because they live in the same CSV.

## Architectural invariants

1. `DebtPositionSpec.aggregation` is always `snapshot`.
2. `DebtActivitySpec.aggregation` is always `sum_flow`.
3. Position and activity registries have disjoint IDs and separate resolvers.
4. Debt position requires valid `as_of_date`; lexical fallback is not an approved selection policy.
5. Annual debt position selects a closing snapshot; it never sums monthly stock values.
6. Annual debt activity sums monthly activity; it never selects the latest activity row as a stock.
7. Contracts contain no arbitrary callables or lambdas.
8. The contracts preserve native currency grain; they do not introduce ARS/USD aggregation.
9. Debt resolver behavior is unchanged.
10. `CashPositionSpec` remains intentionally absent because PR11 found unresolved headline-authority ambiguity.

## Migration boundary

This PR is contract-only. In particular, these modules do not import the new contract yet:

- `accounting/professional/drilldown.py`
- `accounting/professional/drilldown_legacy.py`
- `accounting/marts/debt.py`
- `accounting/metrics/annual.py`

Recommended next sequence:

1. PR13: migrate professional debt-position drilldowns to `DebtPositionSpec`, preserving valid-snapshot parity and making invalid-as-of unavailability explicit.
2. PR14: migrate debt-activity drilldowns separately to `DebtActivitySpec`.
3. Only after debt is closed, return to the blocked cash-authority decision from PR11.

No live accounting inputs, generated reports, debt balances, classifications, or publication artifacts are modified by this contract PR.
