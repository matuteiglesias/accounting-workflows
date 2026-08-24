# Accounting simplification Phase 5 — shared governed cash projection

Date: 2026-08-24

## Reassessment after Phases 1–4

Phase 4 made the migration facades explicit, so Phase 5 no longer needs to solve a hidden import-surface problem. The remaining duplication is narrower and clearer:

- `accounting.metrics.annual` independently discovered years/currencies/Boxes and called the annual selector;
- `accounting.metrics.frontier` independently discovered periods/currencies/Boxes and called the monthly selector;
- `accounting.professional.annual_dashboard_tables` independently discovered years/currencies/Boxes and called the annual selector.

`accounting.cash_authority` is already the trusted semantic/runtime authority. Phase 5 therefore does **not** add a new cash rule or selector.

## Accounting/reporting invariant

> Annual metrics, frontend cash series, and professional annual cash tables must be projections of the same governed selected population for the same source-backed scope.

More specifically:

1. `cash_authority` alone decides validated eligibility, latest valid `as_of_date`, account identity, value aggregation, annual latest-period selection, and fail-closed status.
2. Reporting code may not independently synthesize period/currency/Box populations.
3. Reporting code may not use inferred Box control or internal party balances as validated cash or fallback cash.
4. Annual cash remains a stock: latest governed annual closing position, never a sum of monthly positions.
5. Currency remains explicit; no native-currency aggregation is introduced.

## Implementation boundary

New `accounting.cash_projection` is a mechanical projection adapter only. It:

- normalizes source-backed `period/Currency[/Box]` scopes;
- derives source-backed `year/Currency[/Box]` scopes from those monthly rows;
- delegates every selected value/status to the existing `cash_authority` selectors;
- returns `ValidatedCashProjection`, which carries reporting scope plus the existing `CashSelection` object.

It does **not** inspect validation semantics, suitability, source type, account identity, or amount logic when discovering scopes.

The professional per-cell cash drilldown executor intentionally continues to call `cash_authority` directly because it receives an already identified display cell; it is not a bulk report-population builder.

## Before → after

Before:

```text
annual facade        -> discover scope -> cash selector -> annual row
frontier facade      -> discover scope -> cash selector -> series row
professional facade  -> discover scope -> cash selector -> companion row
```

After:

```text
monthly_cash_close
      ↓
source-backed cash projection
      ↓
existing cash_authority selector
      ↓
shared ValidatedCashProjection population
      ├─ annual schema adapter
      ├─ frontend schema adapter
      └─ professional companion schema adapter
```

## Deliberate scope tightening

The old annual facade used global `years × currencies × boxes` loops and then partially filtered them, so it could emit unavailable cash rows for year/currency/Box combinations that never existed in `monthly_cash_close`.

Phase 5 enumerates only actual source-backed scopes. This can remove **phantom unavailable rows**. It does not change an available governed cash value, selected account population, annual stock rule, or currency treatment.

Tests characterize this explicitly with disjoint ARS/2025 and USD/2026 source scopes: the shared projection produces only the four real currency/Box scopes and never fabricates 2025/USD or 2026/ARS rows.

## Validation and reconciliation evidence

The PR-specific Phase 5 gate passed on the corrected branch head:

- `make validate`: **PASS**;
- pytest: **298 passed, 1 warning**;
- the warning is the unchanged intentional legacy invalid-`as_of_date` debt compatibility case;
- compile, artifact-contract, source-contract, annual-schema, and publish-contract checks: **PASS**;
- `make smoke-full`: **PASS** for the repository's documented fixture-safe surface;
- semantic leakage QA: **0 rows**;
- exact Phase-0 semantic classification census: **UNCHANGED**;
- exact Phase-0 ARS/USD annual governed non-cash values/statuses: **UNCHANGED**;
- native-currency separation: **UNCHANGED**.

Representative annual anchors still reconcile exactly:

- ARS operating revenue: `33,075,422`;
- ARS property OPEX: `13,615,938.94`;
- ARS net operating: `19,459,483.06`;
- ARS funding: `1,427,956`;
- ARS personal draws: `29,412,662`;
- ARS net after draws: `-8,525,222.94`;
- USD operating revenue/net operating: `4,180`;
- USD savings rate: `1.0`.

The new disjoint-scope regression also proves the only intended reporting-shape change: impossible 2025/USD and 2026/ARS cash scopes are no longer fabricated merely because both currencies exist somewhere in the source history.

The smoke fixture still lacks governed debt-position/activity inputs and a real professional pack. Phase 5 does not alter those domains and does not treat the fixture run as evidence about real debt/professional-pack values.

No generated reports, smoke outputs, live ledgers, professional packs, large datasets, caches, or confidential accounting records are versioned by this phase.
