# Metrics frontier PR 4

## What frontier means

The metrics frontier is a curated, long-format consumption contract for notebooks, professional reports, and frontends. Frontier means curated, not complete: a metric is included only when it can point to a canonical or explicitly caveated source.

## Included metrics

The initial frontier includes clean operating metrics, funding/distribution/coverage metrics, data-quality metrics, explicit cash metrics, internal debt metrics, and selected legacy compatibility metrics. The required initial set is represented in `metric_contract_frontier.csv`:

- `IS.RENT.TOTAL`
- `IS.REVENUE.OPERATING`
- `IS.OPEX.PROPERTY`
- `IS.NET.OPERATING`
- `FUND.CONTRIB.TOTAL`
- `DIST.DRAWS.PERSONAL`
- `COV.NET.AFTER_DRAWS`
- `BS.CASH.TOTAL`
- `BS.CASH.CLOSE.BOX`
- `ID.DEBT.OPEN.BY_COUNTERPARTY`
- `DQ.CLASSIFICATION.COVERAGE`
- `DQ.UNKNOWN.AMOUNT`

## Sources

The frontier reads from these backend outputs when available:

- `monthly_flow_semantic_split.csv`
- `monthly_operating_statement.csv`
- `monthly_cash_close.csv`
- `monthly_debt_position.csv`
- `metric_registry.csv`
- `metric_values.csv`

It does not silently fall back to unsafe legacy tables for clean metrics.

## Frontend-safe and caveated metrics

Operating revenue and rent are frontend-safe when semantic classification sources exist. Property OPEX, net operating, funding, draws, coverage, data-quality, and internal debt metrics are marked `safe_with_caveat` because they depend on classification quality or represent internal/cash-coverage concepts.

Cash metrics are `unavailable` unless `monthly_cash_close.csv` contains rows with `is_frontend_safe=true`. If no such rows exist, the frontier emits no cash series and does not derive cash from party balances or box motors.

## Legacy outputs remain

This PR does not delete or rename `metric_values.csv`, `metric_registry.csv`, `metric_views/`, income statement views, human reports, or public compatibility files. Selected legacy metric IDs are represented in the frontier with `legacy_flag=true` and `frontend_suitability=legacy_only`.

## Frontend consumption

Frontends should read `frontend_metric_series.csv` as the chart-ready long table and join to `metric_contract_frontier.csv` for labels, caveats, public/internal flags, suitability, and status. Frontends should not chart metrics with `frontend_suitability=unavailable`, and should display caveats for `safe_with_caveat` or `legacy_only` metrics.
