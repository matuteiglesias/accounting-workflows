# Atomic-flow drilldown contract v1

## Scope

`accounting.contracts.atomic_flow_drilldowns` defines immutable, typed
`FlowCellSpec` records for atomic semantic flows. It is a contract-only change:
no consumer imports the registry yet.

The v1 contract deliberately excludes:

- cash and debt snapshots;
- debt activity routing;
- derived formulas and quality ratios;
- compatibility fallbacks;
- unsupported-route policy;
- arbitrary Python predicates or callables.

Those families have different selection and fallback behavior and must not be
forced into a premature common `CellSpec`.

## Measure authority

A spec contains `measure_ref=(semantic_bucket, semantic_subbucket)`. It does
not contain `measure`, `amount_in`, `amount_out`, `amount_abs`, or `net_amount`
fields. Construction fails unless the reference resolves through
`semantic_measure_registry_v1`.

The registry therefore governs membership and grain without becoming another
atomic-measure authority. A future executor must resolve `measure_ref` at use
time rather than materialize the physical amount column into the spec.

## Declarative boundary

Membership is limited to normalized semantic bucket/subbucket values and a
typed grain. Every grain begins with `period, Currency`; optional dimensions
are drawn from the explicit `FlowGrainDimension` vocabulary. `FlowCellSpec` is
frozen and slotted, the v1 registry is read-only, and callable fields are
rejected.

The funding entries are intentionally named `funding_contribution`, not generic
funding/support. Direct obligation and debt-linked support currently require
multi-semantic membership behavior and are not misrepresented as one atomic
flow spec.

## Version and migration rule

The registry version is `atomic_flow_drilldown_specs_v1`. Adding a consumer is
a separate migration requiring fixture parity for membership, matched values,
fallback behavior, and reconciliation tolerance. This PR does not change any
production route or generated output.
