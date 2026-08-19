# Atomic-flow drilldown metadata propagation v1

## Purpose

This change carries a stable `drilldown_cell_id` on professional table rows that
already expose sufficient producer metadata to identify one governed
`FlowCellSpec`.

It is metadata propagation only. The professional drilldown executor does not
consume `FlowCellSpec` in this PR, so accounting values, membership, fallback
behavior, reconciliation, and publication behavior remain unchanged.

## Authority boundary

`drilldown_cell_id` is never inferred from human-facing labels.

A non-empty ID may come only from one of these stable producer surfaces:

1. an explicit existing `drilldown_cell_id`;
2. an explicit `metric_id` that survives table-contract enrichment unchanged;
3. an explicit `statement_line`;
4. a table family whose identity is itself unambiguous for one governed atomic
   flow contract.

Compatibility logic may still infer or repair `metric_id` from presentation
labels for older packs. Those inferred values do **not** opt the row into the
governed atomic-flow path.

If structured metadata is absent, unsupported, or contradictory,
`drilldown_cell_id` remains blank and the row stays on the existing
compatibility behavior.

## Fail-closed invariant

Every non-empty `drilldown_cell_id` must satisfy:

```python
resolve_flow_cell_spec(drilldown_cell_id) is not None
```

Producer-provided IDs that are unknown fail validation. A producer-provided ID
that conflicts with another stable structured identity also fails validation.

When two structured metadata fields disagree and no explicit producer cell ID
is present, enrichment leaves the ID blank rather than guessing.

## Initial exact mappings

The initial mapping is deliberately narrower than the legacy drilldown routing.
It covers approved atomic flows whose current producer metadata already matches
a `FlowCellSpec`, including:

- operating revenue and rent;
- property OPEX total/category;
- funding total and governed funding dimensions;
- family draws/distributions total and draws by type;
- approved FX proceeds/outflow/cost;
- the unambiguous monthly draws-by-box and draws-by-type table families.

The following remain intentionally blank where no exact atomic-flow spec exists:

- direct-obligation and debt-linked funding;
- generic actor funding where the grain does not match the governed
  `funding_actor` spec;
- dividends as a narrower subset without its own flow spec;
- FX net and raw all-measures surfaces;
- OPEX routes whose table grain is not yet represented by a `FlowCellSpec`;
- derived formulas, cash/debt positions, debt activity, ratios, and legacy
  fallbacks.

## Migration boundary

This is PR10B. It does not import `atomic_flow_drilldowns` from
`professional/drilldown.py` and does not replace `_spec_for_cell`.

PR10C may consume non-empty `drilldown_cell_id` values through a governed
atomic-flow executor and must prove membership/value/reconciliation parity before
removing procedural routes.
