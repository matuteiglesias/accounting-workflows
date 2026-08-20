# Wave 5 PR16 — derived metric and formula authority audit

Date: 2026-08-19

## Purpose

PR16 is characterization only. It does not add `DerivedMetricSpec`, does not change any displayed value, and does not migrate a professional route.

The invariant for the final migration wave is:

> a derived metric must combine already-governed quantities through one explicit formula authority; professional presentation must not independently rediscover semantic membership, silently substitute zero for missing evidence, or select a formula from a human-facing label.

The machine-readable companion is `diagnostics/derived_metric_authority_inventory_20260819.csv`.

## Scope traced

The audit follows derived values through the current post-Wave-4 pipeline:

```text
monthly_flow_semantic_split
    -> monthly_operating_statement
    -> annual_balance_dashboard_metrics
    -> professional tables
    -> professional drilldowns
```

It also inspects the governed inferred-box-control primitive introduced in Wave 4 because `diagnostic_box_level` is a period delta over a control position rather than a cash headline.

No live ledger or family record is read by this PR.

## What is already authoritative upstream

### `IS.NET.OPERATING`

`accounting/marts/semantic.py` already produces:

```text
net_operating = operating_revenue - property_opex_true
```

and materializes it as `statement_line=net_operating`. Annual metrics sum that monthly statement line by year and currency.

Therefore Wave 5 must **not** make professional reclassification of rent/OPEX the new authority. A derived contract may describe and independently reconcile the formula, but the statement value remains the production source value unless a later accounting decision deliberately changes the statement contract.

### `COV.NET.AFTER_DRAWS`

The semantic mart already produces:

```text
coverage_after_draws
    = net_operating
    + funding_contributions
    - family_draws_or_distributions
```

Professional currently has a second authority: use `COV.NET.AFTER_DRAWS` if present, otherwise recompute from annual components whose missing values default to zero. That `source OR recompute` behavior is precisely what Wave 5 should remove.

Target rule:

```text
source value = authoritative displayed value
formula composition = independent reconciliation/explanation
missing component = visible reconciliation gap, not an alternate production value
```

### Treasury FX net

`treasury_fx_net` is also produced upstream, but its formula includes `other_fx` in addition to the three known governed FX subbuckets. It is therefore classified `SPECIALIZED_UPSTREAM_DERIVED`, not a generic `DerivedMetricSpec` v1 candidate.

## Professional formula authority today

`drilldown_legacy.py` contains a local `AnnualFormulaSpec` selected from four Spanish display labels:

| display label | local formula id | current inputs |
| --- | --- | --- |
| Margen operativo | `operating_margin` | `IS.NET.OPERATING / IS.REVENUE.OPERATING` |
| OPEX / renta | `opex_to_rent` | `IS.OPEX.PROPERTY / IS.REVENUE.OPERATING` |
| Retiros / resultado operativo | `draws_to_operating_result` | `DIST.DRAWS.PERSONAL / IS.NET.OPERATING` |
| Cobertura después de funding y retiros | `coverage_after_funding_and_draws` | source `COV.NET.AFTER_DRAWS` or recomputation |

This has three authority problems.

### 1. Human labels select formulas

A translated or edited label can change whether a cell receives derived lineage. PR17 should introduce a stable `derived_metric_id`; the renderer label must not be semantic identity.

### 2. Zero denominator policies disagree

Professional `_safe_div()` currently returns `0.0` when the denominator is within tolerance of zero.

Annual `COV.SAVINGS_RATE`, by contrast, emits `value_status=not_applicable` when annual `IS.NET.OPERATING == 0`.

PR16 freezes both behaviors. The recommended Wave-5 policy is to align ratios with the safer annual behavior:

```text
denominator == 0 -> not_applicable
```

rather than asserting that the economic ratio equals zero.

### 3. Missing components silently become zero

Professional builds a `values` dictionary and uses `.get(metric_id, 0.0)`. A partially missing formula can therefore still reconcile numerically.

Recommended Wave-5 policy:

```text
required component absent -> unavailable
```

For a source-authoritative derived line such as `COV.NET.AFTER_DRAWS`, the source value may remain available while formula reconciliation is marked incomplete; the missing component must not trigger a replacement production calculation.

## Two decision gates before PR17

### `OPEX / renta`

The current label says **rent**, but the denominator is `IS.REVENUE.OPERATING`.

Today those concepts may coincide because operating revenue is rent-dominated, but the metric would change meaning if a future non-rent operating revenue category were introduced. `IS.RENT.TOTAL` already exists.

PR17 must make an explicit decision between:

```text
A. preserve current behavior: IS.OPEX.PROPERTY / IS.REVENUE.OPERATING
B. honor the label:       IS.OPEX.PROPERTY / IS.RENT.TOTAL
```

The audit recommends **B** for semantic clarity, but PR16 does not change behavior.

### `diagnostic_box_level`

Wave 4 deliberately left this formula untouched. Current professional behavior is:

```text
current legacy box-level close
- previous legacy box-level close
```

where the box-level selector accepts:

```text
inferred_box_motor
OR blank party fallback
```

and an empty current/previous subset sums to `0.0`.

Two consequences are now explicit:

1. a validated cash row with blank `party` can be admitted alongside inferred control, even though Wave 4 separated those populations for cash headlines;
2. a missing previous month becomes a zero baseline and can create a false first-period delta.

Recommended target:

```text
derived.diagnostic_box_level
operation        = period_delta
component        = cash.control.inferred_box_motor
missing_current  = unavailable
missing_previous = unavailable
```

This is a deliberate PR17/PR18 behavior decision, not a PR16 fix.

## Duplicate authority to remove

The annual professional line path for `resultado operativo neto` currently rebuilds net operating directly from `monthly_flow_semantic_split.csv`:

```text
operating_revenue/rent amount_in
- property_opex amount_out
```

That is more dangerous than simple duplicate arithmetic: it duplicates semantic membership downstream and creates a route where scope filters could diverge from the canonical statement.

Wave-5 target:

```text
IS.NET.OPERATING source row
+ governed component references for explanation/reconciliation
```

Professional should not decide again which raw semantic rows constitute operating revenue or OPEX.

## Derived surfaces that should stay outside generic v1

Not every net or difference belongs in `DerivedMetricSpec`.

### `TR.FX.NET`

Specialized treasury semantics include residual `other_fx`; keep upstream/specialized.

### `ID.DEBT.NET_PM_POSITION`

This is a debt-position specialization over debtor/creditor identity, currently including text matching for `Property Management`. It should remain debt-specific until that identity is explicitly governed; it is not merely subtraction of two stable scalar metric IDs.

### `cash_annual_box_flow_bridge_wide` net flow

`flujo neto observado` aggregates the already-defined row measure `net_amount` over a flow population. That is an aggregate drilldown, not scalar metric algebra.

Keeping these out prevents the final wave from becoming another catch-all abstraction.

## PR17 contract boundary proposed by this audit

A generic v1 contract should be declarative and closed:

```text
DerivedMetricSpec
    derived_metric_id
    authority_mode
    operation
    component_refs
    grain
    missing_component_policy
    zero_denominator_policy
    tolerance
```

Candidate closed operations:

```text
subtract
add_subtract
ratio
period_delta
```

Candidate authority modes:

```text
COMPUTED_DERIVED
SOURCE_VALUE_WITH_FORMULA_RECONCILIATION
```

No lambdas, `eval`, human-label dispatch, semantic-bucket predicates, or arbitrary Python callables belong in the contract.

### READY for PR17

- `IS.NET.OPERATING` — source value + reconciliation formula.
- `COV.NET.AFTER_DRAWS` — source value + reconciliation formula.
- `COV.SAVINGS_RATE` — ratio of annual aggregates.
- operating margin — ratio.
- draws / operating result — ratio.
- coverage professional presentation — stable reference to `COV.NET.AFTER_DRAWS` plus reconciliation.
- removal target for annual professional net-operating rebuild.

### BLOCKED pending explicit decision

- OPEX / rent denominator (`IS.REVENUE.OPERATING` vs `IS.RENT.TOTAL`).
- diagnostic box-level period delta policy and use of governed inferred control.

### DEFER from generic v1

- treasury FX net.
- net PM debt position.
- bridge net-flow aggregates.

## Acceptance gate for PR17

PR17 should remain contract/metadata-first. Before a formula executor is allowed to change production behavior, tests should prove:

1. formula identity is stable and label-independent;
2. every component ref names an already-governed metric/control authority;
3. currency is preserved and cross-currency formulas fail closed;
4. missing required components cannot silently become zero;
5. ratio zero-denominator policy is explicit;
6. source-authoritative values cannot be overridden by a fallback recomputation;
7. `period_delta` cannot use validated cash or internal balances when its component is inferred control;
8. specialized FX/debt/aggregate routes cannot accidentally resolve as generic derived specs.

## PR16 change effect

Before and after accounting/reporting values: **no change**.

PR16 adds only:

- this audit;
- a machine-readable authority inventory;
- characterization regressions that freeze the current behavior, including the known undesirable behaviors, so later changes must be explicit.

The full repository test suite remains the merge gate. A green pipeline is necessary but not sufficient: the inventory and regressions are the evidence needed to design PR17 without creating another shadow semantic authority.
