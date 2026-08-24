# Test governance

The accounting test suite is organized by **accounting invariant and test role**, not by implementation wave, PR number, or chronology.

This repository's canonical pipeline is:

`ledger canonicalization -> materialization -> semantic marts -> debt -> metrics -> human reports -> professional pack -> drilldowns`

A green suite is necessary but not sufficient. Tests must make it possible to tell which layer owns a meaning, whether that meaning survives downstream transformations, and whether compatibility code can still override a governed identity.

## Permanent test roles

Use these roles when adding or restructuring tests.

### Contract

Defines the accounting or reporting meaning: measure, grain, stock-vs-flow rule, selection rule, fail-closed policy, or stable identity.

Examples:

- validated cash is account-level evidence and never falls back to inferred control;
- debt position is a closing stock selected by valid `as_of_date`;
- approved semantic bucket/subbucket pairs resolve exactly one atomic amount measure.

A contract test should not depend on report labels or migration history unless the label itself is the supported compatibility interface.

### Stage

Proves that one producer implements the contract at its pipeline layer.

Examples:

- Stage A enforces immutable Box scope and cutoff;
- semantic materialization preserves conservative economic classification;
- debt materialization emits stock/activity at their governed grains.

### Reconciliation

Proves that meaning survives between layers.

High-value reconciliation boundaries include:

- scoped ledger -> semantic mart;
- semantic mart -> monthly statement -> annual metrics;
- validated cash mart -> monthly/annual metrics;
- debt position/activity -> annual metrics;
- displayed professional cell -> drilldown matched value;
- native-currency source -> native-currency report/drilldown.

### Architecture

Prevents a second production authority from reappearing.

Use architecture tests sparingly for durable boundaries such as:

- a modern stable identity executes the governed resolver/executor before compatibility;
- validated cash and inferred control remain type-separated;
- debt position and debt activity remain separate contracts;
- unknown governed identities fail closed instead of label-guessing.

Prefer behavioral assertions over exact source-text inventories when both can protect the same invariant.

### Compatibility

Preserves an intentionally supported historical/minimal-schema path that is **not** the current governed meaning.

Every compatibility test must state a removal condition. Examples:

- legacy derived-label formulas: remove when no supported artifact lacking stable `derived_metric_id` can reach them;
- componentless debt sources: remove when supported producers/artifacts all emit component-grained debt position;
- metadata label inference: remove when supported table producers emit stable semantic identities;
- debt aliases: remove after a repository/notebook/caller usage census reaches zero.

Compatibility tests must never be used as evidence that two different modern accounting meanings are both acceptable.

### Ops / presentation

README commands, latest-pointer behavior, HTML links, rendering limits, and notebook-loading behavior are useful tests but are not accounting semantic authorities. Keep them separate from accounting reconciliation when possible.

## Core invariants

The current suite should preserve at least these invariant families:

- **Scope:** a run has one immutable owning-Box universe; downstream layers do not redefine it.
- **Cutoff:** future evidence cannot re-enter a bounded run downstream.
- **Recognition status:** recognized operating evidence and scoped all-status debt evidence remain distinct source contracts.
- **Classification:** ambiguous rows stay visible/review-required instead of being forced into revenue/OPEX.
- **Atomic measure:** approved semantic identities have one governed physical amount measure and declared grain.
- **Funding/support:** broader support evidence may cross economic classes without reclassifying OPEX or debt as `funding_contribution`.
- **Cash direction:** economic semantics cannot manufacture physical Box cash without payer/receiver evidence.
- **Cash position:** validated cash, inferred Box control, and internal balances remain epistemically distinct.
- **Debt position:** stock is latest governed close, never annual-summed; invalid closing evidence must not become an available stock.
- **Debt activity:** period flows remain distinct from debt stock and annualize by summing periods.
- **Derived metrics:** stable modern formulas fail closed on missing components and zero denominators according to contract.
- **Currency:** native currencies are never arithmetically co-aggregated without a declared valuation layer.
- **Drilldown:** supported drilldowns reconcile displayed values and cannot reintroduce excluded semantic membership.
- **Authority:** a governed identity cannot silently fall back to legacy semantic/value execution.

## Migration-era tests

Do not add new permanent modules named after `wave`, `PR`, or a dated audit.

A migration/closeout test may exist temporarily while work is active, but before closing the migration:

1. move durable contract assertions into contract tests;
2. move cross-layer checks into reconciliation tests;
3. move no-second-authority checks into architecture tests;
4. isolate still-supported old behavior under compatibility with a removal condition;
5. remove exact dated inventory counts/classifications once runtime invariants cover the boundary.

Historical diagnostic CSV/Markdown files may remain as evidence. They must not become current semantic authorities merely because tests assert their exact contents.

## Review checklist for test changes

For any accounting-semantic test change, record:

1. invariant protected;
2. authoritative source/contract;
3. fixture/source records involved;
4. old vs new tested behavior, if behavior changes;
5. affected pipeline layers;
6. reconciliation preserved or newly added;
7. compatibility path, if any, and its removal condition;
8. confirmation that native currency and Box scope did not leak;
9. confirmation that a deleted test's unique durable assertion exists elsewhere or is intentionally retired.

Do not use test-count reduction as a success metric. The goal is one intelligible job per test and no competing accounting authority hidden inside the regression suite.
