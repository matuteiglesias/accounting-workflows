# Wave 4 PR11 — position and activity semantics characterization

Date: 2026-08-19  
Base: `c601698b7abb91e2b4863b66bdc42d2ab76f8793` (post-Wave-3 closeout)

## Mandate

This PR is investigation and characterization only. It does not change accounting policy, debt resolution, cash classification, professional routing, generated accounting outputs, publication behavior, or any live run.

The purpose is to freeze the current executable semantics of three distinct domains before introducing typed contracts:

```text
DebtPosition  = stock / snapshot / as-of
DebtActivity  = period flow over resolved debt
CashPosition  = stock/control state with unresolved authority
```

`diagnostic_box_level` is deliberately treated as a derived formula over cash positions, not as a cash-position snapshot.

The machine-readable companion is:

`diagnostics/position_activity_semantics_inventory_20260819.csv`

The executable characterization fixture is in:

`tests/test_position_activity_semantics_characterization.py`

No live accounting input is read.

---

## 1. Debt position

### Current mart authority

`accounting/marts/debt.py::build_monthly_debt_position` consumes `debt_balance_monthly.csv` and requires:

```text
as_of_date
period
debtor
creditor
currency
open_principal
open_interest
open_total
```

The mart first removes duplicate rows at:

```text
period
+ debtor
+ creditor
+ Currency
+ as_of_date
```

and then selects one monthly source snapshot at:

```text
period
+ debtor
+ creditor
+ Currency
```

using the latest parsed `as_of_date`.

For each selected source snapshot it emits three rows:

```text
component=principal
component=interest
component=total
```

with:

```text
open_amount = component-specific amount
```

while also repeating all three denormalized fields on every component row:

```text
open_principal
open_interest
open_total
```

The adversarial fixture freezes this shape. For a selected March close of:

```text
principal 850
interest   20
total      870
```

the mart emits:

| component | open_amount | open_principal | open_interest | open_total |
| --- | ---: | ---: | ---: | ---: |
| principal | 850 | 850 | 20 | 870 |
| interest | 20 | 850 | 20 | 870 |
| total | 870 | 850 | 20 | 870 |

### Interpretation

`open_amount` is the normalized component amount. The three `open_*` fields are denormalized copies of the selected snapshot and are currently consumed by professional/annual code.

This PR does **not** choose one representation as the future canonical physical value. PR12 should preserve parity first. A later cleanup may decide whether a typed `DebtPositionSpec` can safely resolve component -> `open_amount` and remove the redundant dependency.

### Monthly professional behavior

The monthly debt-position drilldown filters:

```text
period
+ Currency
+ debtor
+ creditor
+ component implied by measure
```

and then calls the stock-snapshot selector. It does **not** sum candidate snapshots.

For example:

```text
measure=open_principal -> component=principal
```

and the selected March value is 850, not `1000 + 850` or any other monthly sum.

This boundary is mature enough for a typed `DebtPositionSpec`.

### Annual professional behavior

Annual debt stock is also a snapshot operation:

```text
year candidates
    -> latest period in year
    -> monthly snapshot selection
    -> one stock value
```

It is explicitly **not**:

```text
SUM(monthly debt positions)
```

The current annual helper relies on the repeated `open_principal/open_interest/open_total` fields rather than `open_amount`. That dependency must be characterized in PR12/PR13 rather than silently normalized away.

### Invalid or missing `as_of_date`

This is the important remaining debt-position gap.

When valid and invalid `as_of_date` values coexist, valid parsed dates sort after invalid/NaT values and the latest valid date wins.

However, when **all** candidate `as_of_date` values are invalid, the mart does not fail closed. The secondary raw-string ordering still selects a source row. The adversarial fixture pins:

```text
not-a-date-a -> 800
not-a-date-z -> 700
```

and current mart behavior selects `not-a-date-z` / 700.

The downstream snapshot helper also returns candidate rows when none has a parseable `as_of_date`, so annual stock can continue using that latest period.

This is not an accounting-policy decision, but it is an unresolved data-quality contract. Before PR13, `DebtPositionSpec` needs an explicit rule for:

```text
no valid as_of_date in candidate set
```

Recommended contract direction: fail closed / review-required rather than lexical date fallback. That recommendation is **not implemented in this PR** because it would change current output behavior.

### Debt-position readiness

Status:

```text
READY_WITH_AS_OF_POLICY_GAP
```

The core invariant is already clear:

> Debt position is a stock snapshot. Select one governed close; never sum snapshots across time.

---

## 2. Debt activity

### Current mart shape

`monthly_debt_activity.csv` is built from resolved position plus debt-engine event evidence.

The mart emits one sparse row per activity type:

```text
opening_balance
new_claim
interest_accrual
repayment
adjustment
closing_balance
net_change
```

The governed event-measure correspondence is:

```text
new_claim        -> new_principal
interest_accrual -> interest_accrued
repayment        -> repayments
adjustment       -> adjustments
net_change       -> net_change
```

The characterization proves that each event measure is non-zero only on its owning activity row.

`opening_total` and `closing_total` are repeated control values and are not event-flow measures. `opening_balance` and `closing_balance` are explanatory/control rows; they should not be turned into annual additive activity measures.

### Reconciliation

Current mart logic is:

```text
net_change
  - new_principal
  - interest_accrued
  + repayments
  - adjustments
= 0
```

with residual adjustments surfaced explicitly rather than hidden.

That is a useful invariant to preserve in PR12/PR14.

### Monthly professional behavior

For a requested measure, professional selects the corresponding `activity_type` and sums only that measure within:

```text
period
+ Currency
+ debtor
+ creditor
```

Example fixture:

```text
2025-03 repayment = 180
```

The professional monthly drilldown returns 180 from the `repayment` activity row.

### Annual behavior

Annual activity is additive:

```text
SUM(monthly activity over year)
```

The fixture has:

```text
2025-03 repayments = 180
2025-04 repayments = 170
```

and annual repayment activity is 350.

The current annual helper does not explicitly filter `activity_type`; instead it relies on the sparse-row invariant that non-owning rows carry zero for the requested measure. That currently reconciles, but the future `DebtActivitySpec` should make activity type explicit rather than depending on sparsity as an implicit interface.

### Debt-activity readiness

Status:

```text
READY
```

The architectural invariant is explicit:

> Debt activity is a period flow. It must use `sum_flow` and must never enter a stock/snapshot executor.

PR12 should add a type-level guard so a `DebtActivitySpec` cannot be executed by the debt-position snapshot path and vice versa.

---

## 3. Cash position

### Mart is already more rigorous than professional consumption

`accounting/marts/cash.py::build_monthly_cash_close` deliberately emits three different populations:

#### Internal party balance

```text
position_type      = internal_balance
source_type        = internal_party_balance
cash_suitability   = internal_only
is_frontend_safe   = false
party              = named party
```

This is explicitly **not frontend cash**.

#### Inferred box motor

```text
position_type      = inferred_box_motor
source_type        = inferred_box_motor
cash_suitability   = safe_with_caveat
is_frontend_safe   = false
party              = blank
```

The caveat says this is inferred/reconciliation movement and not a real cash close.

#### Explicitly validated cash

```text
position_type      = cash_close
cash_suitability   = frontend_safe
is_frontend_safe   = true
party              = blank
account_id         = populated
validation_status  = validated/approved/reconciled
validated_by       = nonblank
source_type        = approved cash source type
```

The mart itself emits no cross-population headline total. That boundary is good.

### Adversarial fixture

The fixture deliberately creates the same `period / Currency / Box` with:

```text
2 internal_balance party rows
1 inferred_box_motor row
2 validated cash account rows
```

For 2026-01:

```text
internal Alice             40
internal Bob               10
inferred_box_motor        100
validated Bank A           70
validated Bank B           30
```

Therefore:

```text
internal total   = 50
inferred total   = 100
validated total  = 100
```

A safe headline selection must not blindly return 250 or 200 simply because these rows share `Box`.

### Current monthly professional behavior: ambiguous mix

`_is_box_level_cash_row` currently treats a row as box-level when any of these are true:

```text
source_table == box_balance_time_long.freq=M.csv
OR source_type == inferred_box_motor
OR position_type == inferred_box_motor
OR party is blank
```

Validated cash rows have blank `party`, so the monthly cash-close helper selects:

```text
inferred_box_motor 100
+ validated Bank A 70
+ validated Bank B 30
= 200
```

It correctly excludes the named internal balances, but it still mixes two mutually distinct cash/control populations.

The characterization test intentionally expects **200**. This is evidence of current behavior, not a proposed target.

### Current annual professional behavior: broader mix

`annual_cash_close_by_box_wide` currently:

```text
filter year + Currency + Box
-> choose latest period in year
-> sum close_amount across every row in that period
```

It does not filter `position_type`, `cash_suitability`, `is_frontend_safe`, `source_type`, `account_id`, or validation state.

On the 2026-01 fixture it therefore returns:

```text
internal 50
+ inferred 100
+ validated 100
= 250
```

So monthly and annual professional surfaces do not even use the same cash population.

This is a real reporting-invariant problem. PR11 records it but does not fix it.

### Current diagnostic behavior

`monthly_tables_diagnostic_box_level_matrix` is not a stock. It computes:

```text
current selected cash close
- previous selected cash close
```

using the same broad box-level selector.

Fixture:

```text
2025-12 inferred + validated = 80 + 80 = 160
2026-01 inferred + validated = 100 + 100 = 200
```

Current diagnostic:

```text
200 - 160 = 40
```

while either validated-only or inferred-only movement would be 20.

This demonstrates why the diagnostic must stay outside `CashPositionSpec`: it is a derived formula over whatever cash-position authority is eventually approved.

### Missing previous period

Current diagnostic behavior treats a completely missing prior month as zero.

With no 2025-11 fixture row:

```text
diagnostic(2025-12) = current 160 - previous 0 = 160
```

That is another behavior that must become explicit in the later derived-formula contract. Missing previous position and a genuine zero cash position are not necessarily equivalent states.

### Cash contract readiness

Status:

```text
BLOCKED_CASH_AUTHORITY
```

The audit does **not** support creating `CashPositionSpec` yet.

The following constraints are already clear:

1. `internal_balance` must not become headline cash merely because it shares `Box`.
2. `inferred_box_motor` and validated `cash_close` must not be summed together merely because they share period/Currency/Box.
3. Blank `party` is not a sufficient authority signal.
4. Multiple validated account rows may legitimately need account-level aggregation, but repeated snapshots of the same account need an explicit as-of rule.
5. Annual cash must reuse the same governed cash-position primitive as monthly cash; it must not choose a broader population.
6. Missing cash position must remain distinguishable from genuine zero.
7. `diagnostic_box_level` belongs to a later derived-formula layer.

### Decision required before CashPositionSpec

PR15A will require an explicit reporting decision among at least these strategies:

#### Option A — validated-only headline

```text
headline cash = explicitly validated cash_close rows
inferred_box_motor = reconciliation diagnostic only
internal_balance = internal evidence only
```

Strongest evidentiary meaning, but can be unavailable for periods before validated cash evidence exists.

#### Option B — precedence/fallback

```text
if validated cash exists:
    use validated cash
else if approved inferred control is allowed:
    use inferred_box_motor with caveat
else:
    unavailable
```

This provides historical continuity but requires an explicit rule that inferred is a fallback, never additive to validated cash.

#### Option C — separate metrics

```text
validated_cash_close
inferred_box_control
internal_party_balance
```

and no single headline until a higher-level presentation policy chooses one.

PR11 does not choose among A/B/C.

---

## 4. Current/target matrix

| domain | current behavior | minimum target invariant | ready for contract? |
| --- | --- | --- | --- |
| DebtPosition monthly | latest monthly snapshot; no sum | typed snapshot selector | yes, with invalid-as-of gap |
| DebtPosition annual | latest period then snapshot; no annual sum | reuse monthly snapshot primitive | yes, with invalid-as-of gap |
| DebtActivity monthly | activity-specific sparse flow | explicit activity type + sum-flow measure | yes |
| DebtActivity annual | sum monthly measure relying on sparse rows | explicit annual sum-flow | yes |
| Cash monthly | mixes inferred + validated | one explicit population/precedence | **no** |
| Cash annual | mixes internal + inferred + validated | same governed primitive as monthly | **no** |
| Cash diagnostic | delta over mixed cash selection; missing prior -> zero | later derived formula over governed positions | **no / outside CashPositionSpec** |

---

## 5. Recommended Wave 4 continuation

The audit changes the implementation order slightly but confirms the overall Wave 4 plan.

### PR12

Add contracts only for:

```text
DebtPositionSpec
DebtActivitySpec
```

Do **not** add `CashPositionSpec` yet.

Required PR12 decisions:

- explicit invalid/missing `as_of_date` behavior for DebtPosition;
- explicit stock vs flow type guard;
- component/value representation preserving current outputs;
- explicit `activity_type -> measure` mapping rather than relying only on sparse zero columns.

### PR13

Migrate monthly + annual debt position to the governed snapshot primitive. Do not touch debt resolution.

### PR14

Migrate monthly + annual debt activity to the governed flow primitive. Preserve residual adjustments visibly.

### Then cash decision packet

Only after debt migration should we approve one of the cash authority options and create `CashPositionSpec`.

---

## 6. Completion record

```text
Changed: audit CSV, documentation, characterization tests only
Accounting rule changed: no
Debt resolver changed: no
Cash authority changed: no
Live inputs accessed: no
Publication performed: no
Generated accounting reports committed: no
Fixture evidence: synthetic adversarial cash + debt sources in pytest temp directories
Current cash ambiguity fixed: no; intentionally characterized and blocked
Next bounded action: PR12 DebtPositionSpec + DebtActivitySpec contract-only
```
