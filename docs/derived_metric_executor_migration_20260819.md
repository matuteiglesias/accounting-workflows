# Wave 5 PR18 — governed derived-metric execution

Date: 2026-08-19

## Scope

PR18 is the production-consumer migration for the Wave 5 derived-metric
contract introduced in PR17. It changes professional drilldown authority only;
it does not reclassify ledger rows, alter the semantic mart, change debt/cash
source contracts, or commit live family accounting data or generated report
bundles.

## Accounting/reporting invariant

A derived metric must combine already-governed quantities through exactly one
explicit formula contract. Professional presentation may not rediscover ledger
membership, infer missing components as zero, select a formula from a human
label, mix validated cash with inferred control, or replace an authoritative
upstream value with a locally recomputed alternative.

The migration preserves the authority hierarchy:

- atomic flows remain governed by FlowCellSpec + semantic measure authority;
- debt stock/activity remain governed by their distinct Wave 4 contracts;
- validated cash remains headline cash authority;
- inferred box motor remains a separate control position;
- DerivedMetricSpec governs only composition of those already-governed values.

## Source records and artifacts traced

PR18 consumes only these already-materialized authorities:

- `annual_balance_dashboard_metrics.csv` for annual scalar metric references;
- `monthly_cash_close.csv` through `select_inferred_box_control_period` for the
  diagnostic control delta;
- `DerivedMetricSpec` v1 for formula identity/policy;
- professional table metadata via `derived_metric_id`.

The generic executor does **not** read `monthly_flow_semantic_split.csv`,
classification audit rows, semantic buckets/subbuckets, or raw ledger records.

## Semantic classification / composition rules

### Source-value authorities

`derived.net_operating` and `derived.coverage_after_draws` preserve their
upstream metric values as authoritative. Their formulas are independent
reconciliation/explanation only.

If formula composition disagrees with the source metric, the source value is
retained and the drilldown reports a residual warning.

### Annual computed ratios

Ratios resolve exact governed annual metric rows by stable metric ID,
period, and currency. A component must be a single available scalar value.
Missing/duplicate/unavailable components fail closed.

A zero denominator is `not_applicable`; it is never converted into a numeric
zero ratio.

### OPEX / rent

The governed formula is:

`IS.OPEX.PROPERTY / IS.RENT.TOTAL`

not OPEX divided by total operating revenue.

### Diagnostic box level

The governed formula is:

`inferred_box_control[t] - inferred_box_control[t-1]`

using only the Wave 4 `cash.control.inferred_box_motor` authority.
Validated cash and internal balances cannot enter this formula. Missing current
or previous control position makes the diagnostic unavailable.

## Before / after evidence

The tests use deliberately adversarial synthetic fixtures so the policy changes
are observable without reading live accounting data.

| Surface | Before | After | Reason |
|---|---:|---:|---|
| OPEX / rent with revenue=1000, rent=800, OPEX=200 | 0.20 | 0.25 | denominator now matches rent authority |
| Ratio with zero denominator | 0.0 | not applicable / unsupported drilldown | false numeric zero removed |
| Coverage with missing funding component | local recompute using funding=0 | unavailable | missing evidence fails closed |
| Diagnostic with Feb inferred=100, validated=1000; Jan inferred=80, validated=500 | 520 | 20 | validated cash excluded from inferred-control delta |
| Diagnostic with no prior control month | current value minus implicit zero | unavailable | missing previous position is not a zero balance |
| Coverage source=560 while formula components imply 550 | local source-or-recompute ambiguity | source remains 560 + formula residual -10 warning | upstream authority preserved |

These are intentional semantic/reporting deltas. They are not changes to ledger
classification or source accounting amounts.

## Reconciliation by layer

### Ledger canonicalization / materialization / semantic marts

No changes. Existing Wave 3 and semantic-measure regressions remain the gate.

### Debt

No changes. DebtPositionSpec and DebtActivitySpec executors remain physically
separate and retain their snapshot-versus-flow invariants.

### Metrics

PR18 consumes existing annual metric rows; it does not replace the annual
metric producer. `IS.NET.OPERATING` and `COV.NET.AFTER_DRAWS` therefore retain
upstream value authority.

### Professional pack / drilldowns

Modern professional tables receive stable `derived_metric_id` metadata before
dispatch. The executor then uses only that ID and the v1 contract.

Historical/minimal artifacts that lack the modern governed source schema remain
on compatibility paths rather than being silently interpreted as modern data.

## Drilldown semantic-leakage checks

Architecture regressions enforce that `derived_metric_executor.py` contains no:

- semantic bucket/subbucket membership logic;
- monthly semantic split or classification-audit dependency;
- legacy `_annual_formula_spec` / `_safe_div` authority;
- human-facing formula labels;
- specialized `TR.FX.NET`, `ID.DEBT.NET_PM_POSITION`, or bridge `net_flow`
  formula logic.

Presentation labels exist only in the compatibility metadata adapter that maps
historical table rows to stable derived IDs. The executor itself is label-free.

## Compatibility boundary

PR18 deliberately retains the pre-migration path for historical/minimal
artifacts that do not expose the modern contractual source schema. This is a
compatibility adapter, not a competing semantic authority.

PR19 must measure the reachability of those paths before any physical deletion.

## Explicit deferrals

The generic DerivedMetricSpec executor still excludes:

- `TR.FX.NET` — specialized treasury semantics;
- `ID.DEBT.NET_PM_POSITION` — specialized debt/counterparty identity;
- annual bridge `net_flow` — governed row-measure aggregation, not scalar DAG
  composition.

These exclusions are deliberate and must not be treated as silent generic
formula fallbacks.

## Change summary

PR18 turns the derived professional layer from label-selected procedural
formula logic into stable-ID governed composition. It removes four classes of
semantic leakage from the modern path: zero-default missing evidence,
zero-denominator false ratios, OPEX/rent denominator drift, and validated-cash
contamination of the box-control diagnostic.

No live family records were read or committed. No generated reports or caches
were added. Final readiness requires the repository-wide CI gate plus PR19's
reachability/closure audit after merge.
