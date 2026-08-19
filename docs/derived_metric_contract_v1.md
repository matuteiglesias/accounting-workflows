# Derived metric contract v1

Date: 2026-08-19

This document is the PR17 contract boundary that follows the PR16 derived-authority characterization. It defines formula identity and policy only. No production consumer is migrated in PR17.

## Invariant

A derived metric may combine only already-governed component authorities. The contract must not rediscover ledger membership, branch on presentation labels, infer missing components as zero, or execute arbitrary expressions.

The v1 dependency shape is:

```text
governed source metric / governed control position
                  ↓
          DerivedMetricSpec
                  ↓
     closed operation + policy
                  ↓
        future PR18 executor
```

## Authority modes

### source_value_with_formula_reconciliation

The upstream metric remains the production authority. The formula exists only to explain and independently reconcile it.

This mode governs:

- `derived.net_operating` → authoritative value `metric:IS.NET.OPERATING`;
- `derived.coverage_after_draws` → authoritative value `metric:COV.NET.AFTER_DRAWS`.

The future executor must never overwrite these values with its recomputation. A mismatch is evidence and must surface as a residual/status.

### computed_derived

The value is computed from the declared governed component refs through a closed operation.

This mode governs annual ratios and the monthly inferred-control period delta.

## Closed operations

V1 permits only:

- `subtract`;
- `add_subtract`;
- `ratio`;
- `period_delta`.

No lambdas, `eval`, free-form formula strings, semantic-bucket predicates, renderer labels, or arbitrary callables are allowed.

## Registry

`derived_metric_specs_v1` contains exactly seven specs.

| spec_id | mode | operation | components | period grains |
| --- | --- | --- | --- | --- |
| `derived.net_operating` | source reconciliation | subtract | `IS.REVENUE.OPERATING`, `IS.OPEX.PROPERTY` | M, Y |
| `derived.coverage_after_draws` | source reconciliation | add/subtract | `IS.NET.OPERATING`, `FUND.CONTRIB.TOTAL`, `DIST.DRAWS.PERSONAL` | M, Y |
| `derived.savings_rate` | computed | ratio | `COV.NET.AFTER_DRAWS`, `IS.NET.OPERATING` | Y |
| `derived.operating_margin` | computed | ratio | `IS.NET.OPERATING`, `IS.REVENUE.OPERATING` | Y |
| `derived.opex_to_rent` | computed | ratio | `IS.OPEX.PROPERTY`, `IS.RENT.TOTAL` | Y |
| `derived.draws_to_operating_result` | computed | ratio | `DIST.DRAWS.PERSONAL`, `IS.NET.OPERATING` | Y |
| `derived.diagnostic_box_level` | computed | period delta | `cash.control.inferred_box_motor` | M |

## Resolved PR16 decisions

### OPEX / renta denominator

PR16 found that the current professional formula uses `IS.REVENUE.OPERATING` even though the presentation label says rent and `IS.RENT.TOTAL` already exists.

V1 resolves this in favor of:

```text
metric:IS.OPEX.PROPERTY / metric:IS.RENT.TOTAL
```

Reason: the denominator must match the named economic concept. If non-rent operating revenue appears later, it must not silently change an OPEX/rent ratio.

This is a contract decision only in PR17. The current report value does not change until PR18 migrates the consumer and measures the before/after effect.

### diagnostic_box_level authority

V1 defines:

```text
derived.diagnostic_box_level
= period_delta(cash.control.inferred_box_motor)
```

at grain `period / Currency / Box`.

The diagnostic therefore cannot use validated cash merely because `party` is blank. Missing current or previous inferred-control position means `unavailable`; it is not a zero baseline.

Again, PR17 only establishes the target contract. PR18 must show the deliberate before/after against the PR16 characterization fixture.

## Missing and zero policies

Every v1 spec uses:

```text
missing_component_policy = unavailable
```

Ratios additionally use:

```text
zero_denominator_policy = not_applicable
```

This intentionally rejects professional's characterized `_safe_div(...)=0.0` behavior for denominator≈0. A zero ratio and a mathematically undefined ratio are different reporting facts.

## Annual semantics

For source-authority formulas, annual values remain the authoritative annual metric values. Formula composition is reconciliation only.

Annual ratio specs are recomputed from annual component aggregates. They are never averages of monthly ratios.

`derived.diagnostic_box_level` has no annualization in v1.

## Explicit deferrals

The following PR16 authorities remain outside generic `DerivedMetricSpec` v1:

- `TR.FX.NET`: specialized treasury formula with residual `other_fx` semantics;
- `ID.DEBT.NET_PM_POSITION`: specialized net position requiring governed debtor/creditor identity;
- annual bridge `net_flow`: aggregation of governed row-level `net_amount`, not scalar metric algebra.

They are not unresolved generic-formula debt. They remain named specialized authorities and must not be pulled into the future executor accidentally.

## PR18 migration gate

PR18 may consume this contract only if it preserves these rules:

1. stable `derived_metric_id` metadata, never label dispatch;
2. source-authority specs reconcile rather than replace upstream values;
3. missing components produce unavailable, never invented zero;
4. ratio zero denominators produce not-applicable;
5. `OPEX / renta` uses `IS.RENT.TOTAL`;
6. diagnostic period delta uses only governed inferred box control;
7. formula lineage shows component values and their own governed drilldown identities;
8. no formula can reintroduce Household/out-of-scope rows or mix currencies by rebuilding semantic membership.

## PR17 non-goals

PR17 does not:

- import the contract from professional or metrics production code;
- change any report value;
- change `monthly_operating_statement.csv`;
- change annual metric production;
- change cash or debt selection;
- read live family data;
- commit generated reports.
