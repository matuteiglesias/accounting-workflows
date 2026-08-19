# Governed atomic-flow executor v1

## Scope

This migration makes `FlowCellSpec` a real production input to professional
drilldowns without widening the contract to stocks, debt activity, formulas,
ratios, or compatibility fallbacks.

The public module `accounting.professional.drilldown` is now a governed facade.
The historical implementation is preserved byte-for-byte as
`accounting.professional.drilldown_legacy` so unsupported and legacy routes keep
their current behavior while atomic-flow membership is migrated incrementally.

This is a compatibility isolation, not a claim that total professional LOC has
fallen yet. Architecture audits must count the facade and legacy module together
until dead procedural branches are deleted after parity evidence.

## Execution path

For a row with a non-empty `drilldown_cell_id`:

1. Resolve the ID through `atomic_flow_drilldown_specs_v1`.
2. Resolve `measure_ref` through `semantic_measure_registry_v1`.
3. Resolve every declared grain dimension from an explicit row column or the
   structured `dimension_name` / `dimension_value` pair.
4. Fail closed if any required grain dimension is absent.
5. Select the requested month or year, explicit Currency, declared semantic
   member(s), and declared grain dimensions.
6. Aggregate only the governed measure.
7. Expand classification evidence using the same membership and source tx IDs.
8. Reconcile to the displayed value using the existing tolerance/status rules.

The executor does not branch on concepts such as rent, OPEX, funding, or draws.
Those meanings live in the contracts.

## Deliberately deferred IDs

The following metadata IDs remain on the legacy path:

- `flow.funding_contribution.by_actor`
- `flow.funding_contribution.by_channel`
- `flow.funding_contribution.by_cash_effect`
- `flow.funding_contribution.by_target_box`

The current annual/professional surfaces behind those IDs can include direct
obligation and debt-linked support rows in addition to
`semantic_bucket=funding_contribution`. Treating them as a one-member atomic
flow would therefore be a false simplification. They require a separate
multi-semantic support-membership contract.

Likewise, an FX spec that requires `Box` is not broadened when a producer row
omits Box. Missing contract grain fails closed or remains on its existing
compatibility path; the executor never silently drops a required dimension.

## Compatibility boundary

The following remain explicitly outside the governed atomic-flow executor:

- all-measures diagnostics;
- unknown/review visibility;
- signed net-flow views;
- FX net views;
- debt net views;
- multi-semantic funding/support;
- cash/debt stock selection;
- debt activity;
- derived formulas and ratios;
- label-driven compatibility fallbacks.

## Accounting invariant

No accounting classification or semantic-measure rule changes in this PR.
`semantic_measure_registry_v1` remains the sole authority for atomic
`amount_in`, `amount_out`, and `amount_abs` selection. `FlowCellSpec` governs
membership and grain. The professional layer only executes and reconciles those
contracts.

No live accounting inputs are read by the implementation or tests, and no
publication path is added.
