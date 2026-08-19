# Governed professional debt-activity migration v1

## Scope

Wave 4 PR14 migrates only professional **debt-activity** drilldowns to `DebtActivitySpec`.

It does not change the debt resolver, `monthly_debt_activity.csv` construction, debt position, cash position, annual accounting formulas, or publication policy.

The governed professional surfaces are:

- `monthly_tables_debt_activity_matrix`
- `annual_debt_activity_by_pair_wide`

`drilldown_legacy.py` remains unchanged as the characterized compatibility baseline.

## Accounting/reporting invariant

Debt activity is a period flow, never a stock snapshot.

Monthly:

```text
period / Currency / debtor / creditor / activity_type
        ↓
DebtActivitySpec
        ↓
governed activity measure
        ↓
SUM owning activity rows
        ↓
reconciliation
```

Annual:

```text
year / Currency / debtor / creditor / activity_type
        ↓
all eligible monthly activity rows
        ↓
governed activity measure
        ↓
SUM periods
        ↓
reconciliation
```

No `as_of_date`, latest-period, or stock-selection rule is permitted in the activity executor.

## Governed mappings

PR14 preserves the PR12 contract mappings exactly:

| activity type | measure |
|---|---|
| `new_claim` | `new_principal` |
| `interest_accrual` | `interest_accrued` |
| `repayment` | `repayments` |
| `adjustment` | `adjustments` |
| `net_change` | `net_change` |

Professional table view labels such as `repayments`, `new_principal`, `interest_accrued`, and `adjustments` are aliases that resolve to the corresponding contract identity. They do not create new accounting semantics.

`settlements` remains outside `debt_activity_specs_v1` and therefore stays on the legacy compatibility path. This PR does not expand the contract.

## Source record tracing

The governed executor reads only `monthly_debt_activity.csv` and filters strictly by:

- period or year;
- native `Currency`;
- debtor;
- creditor;
- governed `activity_type`.

The value is the sum of the contract-selected physical measure across those rows.

`pair` is retained as report context/evidence but is not a separate accounting identity beyond debtor/creditor.

Historical activity artifacts without an `activity_type` column remain on the legacy helper rather than having activity identity inferred from the requested measure.

## Reconciliation invariant

The professional layer must not repair or hide debt-mart inconsistencies.

A clean sparse mart has nonzero activity measures only on the owning `activity_type` row. For example, `repayments` belongs on `activity_type=repayment`.

PR14 therefore deliberately changes annual reconciliation behavior in an adversarial case:

### Before

The legacy annual helper sums a requested measure across all rows for the pair/year. A stray `repayments` value on an `adjustment` row can be absorbed into the displayed total.

### After

The governed helper sums `repayments` only on `activity_type=repayment`. If the displayed annual table included a stray non-owning value, the professional drilldown returns `residual_warning` rather than broadening membership to force reconciliation.

This is evidence preservation, not correction. The residual remains visible until the mart/source inconsistency is resolved at the appropriate layer.

## Position/activity separation

PR13 and PR14 intentionally use separate executors.

The architectural regression requires:

```text
DebtActivitySpec  -> activity sum executor only
DebtPositionSpec  -> position snapshot executor only
```

The activity executor cannot import/use `DebtPositionSpec`, the position executor, `as_of_date`, or datetime snapshot selection.

The position executor cannot import/use `DebtActivitySpec`, the activity executor, or `sum_flow`.

A shared generic position/activity executor is explicitly out of scope.

## Before/after effect

For clean characterized debt activity:

- monthly values: unchanged;
- annual values: unchanged;
- Currency/debtor/creditor identity: unchanged;
- annualization: remains SUM over periods;
- source artifact: unchanged;
- display/detail section headings: preserved where applicable.

For semantic leakage in the debt mart:

- before: broad annual measure sum could reconcile accidentally;
- after: governed activity membership exposes the discrepancy as a residual.

## Completion gate

PR14 is complete when:

1. clean monthly activity values reconcile exactly;
2. clean annual activity values reconcile exactly as sums of periods;
3. all five v1 activity mappings use the shared contract;
4. `settlements` remains explicitly legacy/uncontracted;
5. historical sources without `activity_type` retain compatibility behavior;
6. non-owning activity-measure leakage produces a reconciliation residual instead of being hidden;
7. position/activity executors are structurally non-interchangeable;
8. fixture-safe repository validation passes;
9. no live accounting inputs or publication are used.
