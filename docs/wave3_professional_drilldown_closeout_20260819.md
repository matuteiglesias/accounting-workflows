# Wave 3 closeout — governed professional flow drilldowns

Date: 2026-08-19

Wave 3 is closed on **semantic ownership and executable routing**. The system is now ready to characterize stock/debt drilldowns in Wave 4 without mixing those semantics into the atomic-flow contract.

## Closeout invariant

For every simple atomic monthly professional flow in the audited surface:

`stable producer identity → drilldown_cell_id → FlowCellSpec → semantic_measure_registry_v1 → governed executor`

No migrated producer is allowed to fall through to a legacy semantic-membership branch. Missing or broader semantics remain explicitly deferred rather than being widened to satisfy the generic executor.

## Evidence

The versioned reachability audit contains 20 representative routes. After PR60:

- 12 / 12 audited simple monthly atomic routes are `GOVERNED_ATOMIC`;
- 0 audited simple monthly routes remain `LEGACY_ATOMIC`;
- semantic measure edit distance remains 1;
- the FlowCellSpec registry contains 23 declarative specs;
- professional drilldown is the production FlowCellSpec consumer;
- annual professional rows preserve `annual_balance_dashboard_metrics.csv` lineage;
- compatibility inference cannot opt rows into governed execution;
- direct OPEX category membership preserves `Box` and `semantic_subbucket` grain.

The closeout regression additionally replaces a source-layout assumption with an executable boundary: for the three historical direct table branches below, the test makes the legacy router raise if called and proves production routing still succeeds through the governed executor:

- `monthly_tables_draws_by_box_amount_out`
- `monthly_tables_draws_by_type_amount_out`
- `monthly_tables_opex_by_type_amount_out`

This is stronger evidence of semantic reachability than counting textual branches in the compatibility module.

## Explicitly deferred — not migration failures

Seven current FlowCellSpec IDs remain intentionally outside generic execution:

### Multi-semantic funding/support

- `flow.funding_contribution.by_actor`
- `flow.funding_contribution.by_channel`
- `flow.funding_contribution.by_cash_effect`
- `flow.funding_contribution.by_target_box`

Current professional funding/support semantics can include direct obligations or debt-linked support in addition to `semantic_bucket=funding_contribution`. They need a dedicated support-membership contract rather than a false atomic simplification.

### FX total vs by-Box grain

- `flow.fx.conversion_proceeds`
- `flow.fx.conversion_outflow`
- `flow.fx.cost_or_spread`

The v1 FX specs require `Box`, while statement rows can represent period/Currency totals. Required grain is not silently discarded.

## Remaining legacy responsibilities

The compatibility module is still physically large. That is **not** reported as a 3.9k → 0 migration and the facade split is not counted as a LOC reduction. Remaining legacy code has explicit responsibilities including:

- all-measures diagnostics;
- net-flow and unknown/review visibility;
- FB / PM / Household bridge compatibility semantics;
- deferred funding/support;
- deferred FX and FX-net diagnostics;
- annual metric lineage and annual compatibility fallback;
- cash/control stock drilldowns;
- debt position and debt activity drilldowns;
- formulas and quality ratios;
- unsupported/error guards and rendering/orchestration.

Three old direct membership definitions are textually still present inside the preserved compatibility module. They are now unreachable for their migrated producers and guarded against runtime fallback. Removing those few lines would require replacing the entire ~3.9k-line compatibility file through the current repository editing surface; that rewrite has no accounting benefit and carries disproportionate merge risk. Their physical deletion is therefore classified as **non-blocking code hygiene**, not unfinished semantic migration.

## Accounting impact

Wave 3 changed drilldown authority and routing, not accounting facts:

- no ledger classifications changed;
- no semantic measures changed;
- no amounts changed by the closeout work;
- no debt interpretation changed;
- no cash-stock interpretation changed;
- no management eligibility changed;
- no live accounting input was used for these closeout tests;
- no generated professional report or confidential dataset was committed.

## Wave 3 gate

| Gate | Status |
|---|---|
| semantic measure authority has one edit point | PASS |
| production FlowCellSpec consumer exists | PASS |
| all audited simple monthly atomic rows governed | PASS — 12/12 |
| audited simple monthly legacy atomics | PASS — 0 |
| migrated direct producers can fall back to legacy membership | PASS — prohibited by regression |
| annual lineage preserved | PASS |
| deferred semantics explicit | PASS — 7 IDs |
| stock/debt/formula semantics kept outside atomic-flow contract | PASS |
| fixture-safe validation | required green before merge |

## Wave 4 entry boundary

Wave 4 should begin with characterization, not implementation. Keep these three semantic objects separate:

1. `DebtPosition` — stock / snapshot, latest applicable close or as-of state;
2. `DebtActivity` — flows over a period;
3. `CashPosition` — stock/control state whose current source surfaces may mix validated cash, inferred box motor, and internal balances.

Recommended order: characterize debt position/activity first, then cash. Debt already has clearer stock-vs-flow semantics; cash has the greater risk of silently choosing the wrong authority.
