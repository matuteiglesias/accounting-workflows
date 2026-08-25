# Legacy metrics engine deletion — 2026-08-25

Stacked base: `audit/metric-capability-census-frontier-cut` at `736b06e7fe741b0f2106280b4c09d0d75e9e3a43`.

## Invariant

Delete the generic legacy metric universe only after the governed frontier is independent from it and every useful accounting fact is already owned by canonical semantic/cash/debt/annual authorities or trivially recoverable from them.

This change does not change ledger classification, semantic membership, cash eligibility, debt position authority, annual aggregation rules, FX classification, or professional drilldown membership.

## Deleted runtime engine

The following modules formed one closed legacy runtime cluster whose only production orchestrator was `accounting.metrics.build`:

- `accounting/metrics/builders.py` — leaf builders over per-flow, legacy contribution/OPEX views, raw-regex draws, daily internal cash and debt balance exports;
- `accounting/metrics/derive.py` — generic `metric_values` formula derivation;
- `accounting/metrics/io.py` — generic metric-value schema and `MetricsContext`;
- `accounting/metrics/registry.py` — old `MetricSpec` registry and namespace bridge;
- `accounting/metrics/validate.py` — validation of the retired metric-value universe;
- `accounting/metrics/views.py` — six/twelve-month presentation conveniences over ledger/legacy inputs;
- `accounting/metrics/drilldown.py` — old metric-view drilldown artifacts.

Repository code search found no current production caller for these APIs outside the old `accounting.metrics.build` engine. Documentation/history references are not runtime consumers.

The modern derived-metric authority is **not** this deleted registry. `accounting/contracts/derived_metrics.py` and the current professional derived-metric executor remain untouched.

## New metrics build boundary

`accounting.metrics.build` now only:

1. resolves run identity and reporting cutoff;
2. builds `metric_contract_frontier.csv` / `frontend_metric_series.csv` and frontier QA from governed monthly sources;
3. builds the annual dashboard metrics/contract/QA;
4. writes source/artifact contracts for those governed handoffs;
5. writes a small build manifest.

It no longer computes or writes:

- `metric_registry.csv`;
- `metric_values.csv` / parquet;
- generic Q/Y wide tables;
- legacy income/cash/debt statement exports;
- `metric_views/*`;
- raw-ledger or legacy-view metric drilldowns.

## Accounting facts preserved by authority

| Family | Current authority after deletion |
| --- | --- |
| operating revenue / rent / OPEX / net operating | monthly operating statement + semantic split |
| core funding / broader support | monthly statement + governed funding-support contract |
| personal draws / dividends / coverage | monthly statement + governed semantic membership |
| validated cash | monthly cash close + `cash.position.validated` projection |
| debt stock | monthly debt position + debt position authority |
| debt activity | monthly debt activity |
| annual facts | annual dashboard projection over governed monthly facts |
| FX | treasury semantic rows / statement and explicit professional FX authority |

## Deliberate non-parity

Differences caused by retired semantics are expected and must not be "fixed" back toward the legacy engine:

- funding is not operating income;
- Household is not property OPEX;
- raw text matching is not draw authority;
- inferred/internal balances are not validated cash;
- counterparty identity is generally a debt dimension rather than a permanent metric ID;
- quarterly convenience output is not a separate accounting authority.

## Remaining stacked dependency

The repository Makefile still asserts the retired metric outputs and still routes the canonical path through the legacy views stage. That control-plane dependency is intentionally removed in the following stacked views-stage PR. Until that PR lands, this Phase-B draft is code-complete at the metrics layer but should be reviewed as part of the ordered stack.
