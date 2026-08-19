# Wave 4 closure audit — governed position and activity lineage

Date: 2026-08-19

This audit is the STOP gate after PR15B. It is intentionally smaller than the Wave 4 characterization audit: the question is whether the position/activity authorities introduced in PR12–PR15B are now real production boundaries and whether cash headline populations reconcile across layers.

## Gate matrix

| gate | target | evidence | status before final CI |
| --- | --- | --- | --- |
| FlowCellSpec production consumer | governed atomic-flow executor active | `accounting/professional/drilldown_wave4_base.py` | PASS_BY_STRUCTURE |
| DebtPositionSpec production consumer | snapshot executor active | `accounting/professional/debt_position_executor.py` | PASS_BY_STRUCTURE |
| DebtActivitySpec production consumer | sum-flow executor active | `accounting/professional/debt_activity_executor.py` | PASS_BY_STRUCTURE |
| ValidatedCashPositionSpec production consumer | shared runtime selector used by metrics + professional | `accounting/cash_authority.py` and PR15B facades | PASS_BY_STRUCTURE |
| InferredBoxControlSpec governed primitive | independent selector exists and is not headline eligible | `select_inferred_box_control_period` | PASS_BY_STRUCTURE |
| monthly cash population == annual cash population | same account-snapshot primitive | adversarial fixture expects 100 monthly and 100 annual | TEST_GATE |
| validated + inferred never additive | inferred excluded from headline | adversarial fixture | TEST_GATE |
| internal balance never headline cash | internal rows excluded but retained as evidence | adversarial fixture | TEST_GATE |
| stocks never annual-summed | annual cash/debt use closing snapshot rules | cash/debt contract regressions | TEST_GATE |
| debt activity never snapshot-selected | activity executor has no as-of selection | PR14 architecture regression | TEST_GATE |
| diagnostic box level unchanged in PR15B | remains derived-formula legacy path | fixture stays 40 | TEST_GATE |

Final status is **not declared here until GitHub CI passes on the complete PR15B head**.

## Accounting/reporting invariant

A successful pipeline run is not sufficient. For modern cash artifacts, all affected layers must identify the same source population:

```text
validated account snapshots
!= inferred box control
!= internal party balances
```

`BS.CASH.TOTAL`, `BS.CASH.CLOSE.BOX`, the annual cash companion, and professional cash drilldowns must therefore reconcile to the same selected account snapshots.

## Source trace

The canonical cash mart remains `monthly_cash_close.csv`. PR15B does not alter its three populations. Instead it centralizes consumption in `accounting.cash_authority`:

- headline cash: `cash.position.validated`;
- inferred control: `cash.control.inferred_box_motor`;
- internal balance: excluded evidence only.

No upstream source row is reclassified or deleted.

## Before/after effect

The versioned synthetic adversarial case is recorded in `diagnostics/wave4_cash_authority_change_20260819.csv`.

Expected deliberate reporting change:

```text
monthly cash: 200 -> 100
annual cash:  250 -> 100
```

The difference is entirely population selection:

- inferred 100 no longer added to monthly cash;
- inferred 100 and internal 50 no longer added to annual cash;
- validated Bank A 70 + Bank B 30 remain.

## Layer reconciliation target

For the modern fixture:

```text
cash_authority selector
= metrics frontier monthly BS.CASH
= annual dashboard BS.CASH
= annual companion cash close
= professional monthly drilldown
= professional annual drilldown
= 100 ARS
```

Any disagreement fails Wave 4 closure.

## Semantic leakage checks

The cash drilldown must retain excluded evidence sections for inferred control and internal balances rather than silently dropping those populations. Missing validated evidence remains unavailable; it is never replaced by zero or inferred control.

`monthly_tables_diagnostic_box_level_matrix` is not a cash-position headline and is deliberately deferred to Wave 5, where its current missing-prior-as-zero behavior and control-period delta formula will be governed as a derived metric.

## Migration scaffolding

PR15B preserves pre-change implementations as compatibility modules. This audit does **not** count facade LOC reduction as architectural deletion. Legacy code is still reachable for historical schemas and non-cash behavior. Physical pruning belongs to the final migration audit after Wave 5, once modern reachability is proven.

## Decision gate

Wave 4 may be declared DONE only when:

1. the full repository validation suite is green on PR15B;
2. the adversarial cross-layer reconciliation tests pass;
3. modern cash uses no inferred/internal fallback;
4. debt position/activity separation regressions remain green;
5. no change was made to diagnostic formula semantics.
