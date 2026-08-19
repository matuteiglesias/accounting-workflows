# Atomic-flow reachability audit — Wave 3 closeout

This audit characterizes the professional drilldown boundary after the atomic-flow executor migration and its identity-coverage completion, before any stock migration.

It is deliberately fixture-safe: it does not read live accounting runs, does not publish artifacts, and does not alter accounting semantics. The audit records representative professional row families and freezes which executor is supposed to own them.

## Invariant

A simple atomic flow may be executed by the generic governed executor only when the professional row carries one explicit, validated `drilldown_cell_id` whose `FlowCellSpec` fully declares semantic membership and grain. Missing or broader semantics remain on their existing route; the migration must never widen membership merely to reduce code.

Annual professional rows remain evidence-linked to `annual_balance_dashboard_metrics.csv`. Their upstream atomic measures are governed, but the professional drilldown keeps the established annual artifact as the lineage authority rather than recomputing annual evidence directly from monthly semantic rows.

## Representative reachability result after identity coverage

The committed matrix contains 20 representative route cases:

- 12 `GOVERNED_ATOMIC` monthly routes;
- 0 simple `LEGACY_ATOMIC` routes in the audited surface;
- 2 `ANNUAL_METRIC` routes whose annual lineage is intentionally preserved;
- 1 `DEFERRED_FUNDING_SUPPORT` route;
- 1 `DEFERRED_FX` route;
- 3 `NET_DIAGNOSTIC` / review routes;
- 1 `COMPATIBILITY` route.

The six simple atomic gaps found by PR59 are now covered explicitly:

1. `rent_revenue` → `flow.rent.total`
2. `taxes` → `flow.property_opex.taxes`
3. `services` → `flow.property_opex.services`
4. `maintenance` → `flow.property_opex.maintenance`
5. `legal` → `flow.property_opex.legal`
6. `monthly_tables_opex_by_type_amount_out` → `flow.property_opex.by_box_category`

The new OPEX contract declares grain `period, Currency, Box, semantic_subbucket`. The existing `flow.property_opex.by_category` contract is intentionally not reused because it does not declare `Box`; dropping that dimension would broaden membership.

## Legacy deletion candidates

Three direct legacy table branches are now demonstrably bypassed by stable production identities:

- `monthly_tables_draws_by_box_amount_out`
- `monthly_tables_draws_by_type_amount_out`
- `monthly_tables_opex_by_type_amount_out`

Those branches can be deleted in the final Wave 3 cleanup with a regression guard preventing their reintroduction. Monthly statement atomics are also governed, but their historical helper machinery is shared with annual/compatibility lineage; it must not be deleted wholesale merely because monthly dispatch no longer uses it.

The 770 LOC previously attributed to atomic-flow routing therefore still cannot be treated as one deletion unit. Remaining code may serve annual lineage, multi-semantic support, diagnostics, FX compatibility, and other explicitly deferred behavior.

## Wave 3 close gate

Wave 3 is considered closed when:

- every simple atomic professional row in this audit has one governed membership authority — **met**;
- migrated direct branches are removed from the legacy router and guarded against reintroduction — **final cleanup pending**;
- remaining legacy routes are explicitly diagnostic, multi-semantic, FX-deferred, annual-lineage, compatibility, stock/debt, formula, ratio, or unsupported behavior;
- fixture-safe validation remains green;
- the final audit reports total `drilldown.py + drilldown_legacy.py` LOC rather than treating the facade split as a reduction by itself.

The next wave begins only after those conditions are measured, not merely after a successful pipeline run.
