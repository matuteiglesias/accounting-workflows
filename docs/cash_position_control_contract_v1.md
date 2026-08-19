# Wave 4 PR15A — governed validated cash and inferred box control contracts

Date: 2026-08-19

## Purpose

PR15A converts the cash-authority decision from PR11 into typed declarative
contracts. It does **not** migrate professional consumers and does not change
any accounting output.

The reporting decision is explicit:

```text
validated cash_close
    -> headline cash authority

inferred_box_motor
    -> reconciliation/control position only

internal_balance
    -> internal evidence only
```

There is no automatic fallback from validated cash to inferred control.

## Reporting invariant

A cash headline must not change epistemic meaning merely because validated
account evidence is absent in one period.

Therefore:

- validated cash and inferred box control are distinct contracts;
- they are never additive;
- inferred control is never headline eligible;
- internal party balances are not cash-headline rows;
- missing validated cash is `unavailable`, not zero and not inferred fallback.

## `ValidatedCashPositionSpec`

Stable ID:

```text
cash.position.validated
```

Source:

```text
monthly_cash_close.csv
```

Required row authority:

```text
position_type = cash_close
cash_suitability = frontend_safe
is_frontend_safe = true
validation_status in {validated, approved, reconciled}
validated_by != ""
source_type in {
  bank_statement,
  manual_cash_count,
  account_snapshot,
  reconciled_opening_plus_movements,
}
account_id != ""
valid as_of_date
```

The approved validation/status vocabularies intentionally match the current
cash mart. A regression test pins that parity while PR15A leaves the mart
unchanged.

### Grain and selection

```text
period
+ Currency
+ Box
+ account_id
```

For each candidate account in a period:

```text
select latest valid as_of_date
```

Then:

```text
SUM(selected account close_amount)
```

within the requested `period / Currency / Box`.

### Fail-closed completeness

Two silent-understatement paths are prohibited.

#### Duplicate same-account / same-as-of snapshots

If the same governed account has more than one candidate snapshot at the same
selected `as_of_date`, the position is:

```text
unavailable
```

PR15A does not approve arbitrary summing or arbitrary row choice.

#### Candidate account with no valid as-of snapshot

If an otherwise eligible account is present but has no valid `as_of_date`, the
whole requested cash position is:

```text
unavailable
```

The consumer may not simply drop the account and publish a smaller total.

### Annualization

Annual cash must reuse the same monthly account-snapshot primitive:

```text
year
  -> latest governed period with validated cash authority
  -> latest valid snapshot per candidate account
  -> SUM selected account closes
```

It is never a sum of monthly cash positions.

### Missing policy

```text
validated cash absent -> unavailable
fallback to inferred   -> never
```

## `InferredBoxControlSpec`

Stable ID:

```text
cash.control.inferred_box_motor
```

Authority:

```text
position_type = inferred_box_motor
source_type = inferred_box_motor
cash_suitability = safe_with_caveat
is_frontend_safe = false
```

Grain:

```text
period
+ Currency
+ Box
```

Selection:

```text
latest valid as_of_date
```

Aggregation:

```text
snapshot
```

A duplicate selected snapshot fails closed as `unavailable`.

The contract explicitly states:

```text
headline_eligible = false
fallback_role = never_cash_headline
```

This primitive exists so later derived/control reporting can use a governed box
motor position without mislabelling it as validated cash.

## Internal balances

PR15A deliberately defines no `InternalBalanceSpec` and no generic
`CashPositionSpec`.

`internal_balance` remains internal party-level evidence. Sharing a Box,
Currency, period, or blank/nonblank party field is not a reporting-authority
rule.

## Consumer boundary

PR15A is contract-only. The following continue unchanged:

- `accounting/marts/cash.py`
- professional monthly cash drilldown
- annual cash companion table/drilldown
- `diagnostic_box_level`
- annual metrics
- published packs

A regression test verifies that none of those production consumers imports the
new contract yet.

## Expected PR15B behavior change

PR15B will be the consumer migration and must measure the before/after effect on
the adversarial PR11 fixture.

Characterized current fixture:

```text
internal party balances = 50
inferred box motor      = 100
validated accounts      = 100
```

Current professional behavior:

```text
monthly cash = 200  # inferred + validated
annual cash  = 250  # internal + inferred + validated
```

Target governed headline:

```text
monthly cash = 100  # validated accounts only
annual cash  = 100  # same validated primitive
```

The inferred 100 remains separately available as control evidence; the internal
50 remains separately available as internal evidence.

## Completion record

```text
Accounting rule changed: reporting authority contract only; no consumer yet
Cash mart changed: no
Professional routing changed: no
Annual table changed: no
Live inputs accessed: no
Publication performed: no
Generated accounting artifacts committed: no
Next bounded action: PR15B professional + annual consumer migration
```
