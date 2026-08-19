# Governed professional debt-position migration v1

## Scope

Wave 4 PR13 migrates only professional **debt-position** drilldowns to `DebtPositionSpec`.

It does not change the debt resolver, `monthly_debt_position.csv` construction, debt activity, cash position, annual accounting formulas, or publication policy.

The two governed professional surfaces are:

- `monthly_tables_debt_position_matrix`
- `annual_debt_stock_by_pair_wide`

`overview_balance_dashboard` / `income_operating_statement` debt rows retain their established annual-metric lineage; this PR does not recompute those surfaces from the monthly debt wrapper.

## Accounting/reporting invariant

Debt position is a stock snapshot, never a period flow.

```text
period / Currency / debtor / creditor / component
        ↓
DebtPositionSpec
        ↓
latest valid as_of_date
        ↓
component-specific governed value
```

For annual stock:

```text
year / Currency / debtor / creditor / component
        ↓
latest available period in year
        ↓
latest valid as_of_date in that period
        ↓
governed value
```

Monthly stock values are never summed across periods.

## Governed mappings

The migration preserves the PR12 physical mappings exactly:

| spec | component | value |
|---|---|---|
| `debt.position.principal` | principal | `open_principal` |
| `debt.position.interest` | interest | `open_interest` |
| `debt.position.total` | total | `open_total` |

No normalization to `open_amount` is introduced.

## Source record tracing

The governed executor reads only modern, component-grained `monthly_debt_position.csv` and filters strictly by:

- period/year;
- native `Currency`;
- debtor;
- creditor;
- component.

For monthly cells, all matching component snapshots are retained as candidate evidence and one latest **valid** `as_of_date` row is selected.

For annual cells, all matching rows in the year are retained as candidate evidence; the latest period is selected first, and then the same latest-valid-as-of primitive is applied inside that closing period.

The drilldown exposes candidate rows, selected rows, `spec_id`, contract version, selected period/as-of, and valid-as-of counts.

## Intentional before/after change: invalid `as_of_date`

PR11 characterized an undesirable compatibility behavior: the debt mart can still emit a row whose source `as_of_date` is unparseable, and the historical professional helper can consume it as a stock value.

PR12 explicitly declined to canonize that behavior:

```text
invalid_as_of_policy = unavailable
```

PR13 therefore changes governed professional consumption as follows:

### Before

An invalid/undated candidate could still be selected and displayed as debt stock.

### After

If the governed closing candidate set contains **no valid `as_of_date`**:

- status = `unavailable`;
- matched value = 0 only as the technical reconciliation placeholder, not as an asserted zero balance;
- the displayed value remains visible through the residual;
- candidate evidence is preserved in the drilldown;
- prior periods are **not** substituted for an invalid latest annual period;
- filters record `availability_status=unavailable` and the reason.

This is a reporting-contract hardening change, not a change to debt resolution or source records.

## Parity requirements

For valid governed snapshots, PR13 preserves:

- selected period;
- selected `as_of_date`;
- Currency;
- debtor / creditor;
- component;
- value;
- residual;
- candidate evidence;
- historical section headings where they are part of existing HTML regression expectations.

The migration adds tests with multiple snapshots inside one month and multiple months inside one year to prove latest-snapshot rather than summation behavior.

## Compatibility boundary

Two compatibility cases deliberately remain outside the governed v1 path:

1. unknown/non-v1 debt-position measures;
2. historical `monthly_debt_position.csv` artifacts that do **not** contain the `component` column required by the v1 grain.

The second case is important: PR13 does **not** synthesize `component` from the requested measure merely to make a legacy artifact satisfy the contract. Component-less artifacts continue through the historical helper and preserve their established output. Modern component-grained mart artifacts use `DebtPositionSpec`.

`drilldown_legacy.py` remains unchanged and continues to represent the characterized before-state. The public facade intercepts only the two governed debt-position surfaces when the source satisfies the v1 grain.

Debt activity remains untouched and is owned by PR14.

## Completion gate

PR13 is complete when:

1. valid monthly and annual component-grained debt-position examples reconcile exactly;
2. invalid-as-of governed examples return explicit unavailable rather than a lexical/undated stock;
3. annual latest-period selection never backfills from a prior period;
4. component-less historical sources preserve legacy compatibility instead of being silently upgraded;
5. `DebtActivitySpec` is not consumed;
6. `drilldown_legacy.py` remains free of the new contract import;
7. fixture-safe repository validation passes.
