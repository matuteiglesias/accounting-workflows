# Accounting Backend Module Inventory

Status: authority draft
Scope: `accounting/*`
Last reviewed: 2026-05-10

## Purpose

This inventory classifies the flat accounting modules by responsibility and current status. It is a navigation aid for the current codebase and a guide for later compatibility-package migrations.

## Status vocabulary

| Status | Meaning |
|---|---|
| canonical | Preferred current implementation or contract |
| current | Actively used current implementation |
| support | Helper/downstream module that supports the pipeline |
| experimental | Future/stub/incomplete surface |
| legacy | Historical compatibility only |
| unknown | Needs more runtime evidence |

## Inventory

| Module | Layer | Status | Primary responsibility / outputs |
|---|---|---|---|
| `__init__.py` | package | support | Package marker. |
| `config.py` | config | support | Loads config/default path options. |
| `logging_utils.py` | support | support | Shared logging setup. |
| `utils.py` | support | support | Shared helpers such as run-id/path helpers. |
| `models.py` | support/domain | unknown | Shared/domain models; keep under review before promoting. |
| `manifest.py` | support/materialize | support | Manifest helper utilities. |
| `core/timeseries.py` | core | support | Pure time-series aggregation and expansion helpers. |
| `core_timeseries.py` | core | compatibility wrapper | Old time-series primitive import path. |
| `ledger/ingest.py` | Level 1 canonical ledger | canonical | Reads source inputs and builds the canonical ledger/anomaly contract. |
| `ingest.py` | Level 1 canonical ledger | compatibility wrapper | Old ledger ingest import/command path. |
| `stage_d/materialize.py` | Level 2 materialized artifacts | canonical | Writes per-flow, per-party, daily cash, loan/time, and materialization metadata artifacts. |
| `materialize.py` | Level 2 materialized artifacts | compatibility wrapper | Old Stage D materializer import/command path. |
| `views.py` | Level 2/3 view bridge | current support | Builds/loads view tables from Stage D materialized artifacts; legacy report inputs are best-effort only. |
| `metrics/io.py` | Level 3 metric contract | canonical | Defines metric-values schema and metric context. |
| `metrics/registry.py` | Level 3 metric contract | canonical | Defines metric registry specs and normalization. |
| `metrics/builders.py` | Level 3 metric builders | canonical support | Builds leaf metric value frames from context. |
| `metrics/derive.py` | Level 3 derived metrics | support | Derivation helpers for parent/formula metrics. |
| `metrics/validate.py` | Level 3 validation | support | Registry/value validation and validation-report rows. |
| `metrics/views.py` | Level 3 metric views | support | Builds income/rent/flow/draws/debt-facing metric view CSVs. |
| `metrics/drilldown.py` | Level 3/4 evidence | support | Builds drilldown detail/index/manifest artifacts for report evidence. |
| `metrics/build.py` | Level 3 metrics orchestration | canonical | Main metrics build entrypoint; writes registry, values, validation, views, drilldowns, and manifest. |
| `metrics_io.py`, `metrics_registry.py`, `metrics_builders.py`, `metrics_derive.py`, `metrics_validate.py`, `metrics_views.py`, `metric_drilldown.py`, `build_metric_values.py` | Level 3 metrics compatibility | support | Thin wrappers that preserve old imports and `python -m accounting.build_metric_values`. |
| `debt/resolve.py` | Level 3 debt resolution | canonical | Current debt engine; writes open items, allocations, repayment events, timeline, and reconciliation. |
| `resolve_internal_debt_v2.py` | Level 3 debt resolution | compatibility wrapper | Old debt resolver import/command path. |
| `debt/balance_views.py` | Level 3 debt views | canonical | Builds daily/monthly/quarterly/yearly debt balance views from debt open items. |
| `build_debt_balance_views.py` | Level 3 debt views | compatibility wrapper | Old debt balance view import/command path. |
| `human/tables.py` | Level 4 human tables | canonical support | Defines reusable human-facing table specs and table builders. |
| `human_balance_tables.py` | Level 4 human tables | compatibility wrapper | Old human table import path. |
| `human/document.py` | Level 4 human report | current canonical | Current `balance_human_v2` human report/document factory. |
| `human_balance_document_factory.py` | Level 4 human report | compatibility wrapper | Old human report import/command path. |
| `human/front.py` | Level 4/5 front report | experimental | Future/front-oriented report builder; not production canonical yet. |
| `human_balance_front_factory.py` | Level 4/5 front report | compatibility wrapper | Old front report import/command path. |
| `publish/latest.py` | Level 5 frontend snapshot | current canonical | Packages selected latest artifacts into `public/accounting/latest/*`. |
| `publish/manifest.py` | Level 5 frontend snapshot | support | Defines the frontend snapshot manifest schema helper. |
| `publish/snapshot.py` | Level 5 frontend snapshot | support seam | Reserved seam for snapshot copy/filter helpers. |
| `publish_latest.py` | Level 5 frontend snapshot | compatibility wrapper | Old publish import/command path. |
| `reports.py` | legacy/reporting | legacy | Older report entrypoint; do not use for new canonical flow without revalidation. |
| `plots.py` | support/visualization | support | Plot generation utility. |
| `hashlib` | compatibility/support | unknown | Non-Python helper/file; inspect before changing. |

## Package map

When layer boundaries are proven, modules can migrate toward these packages while preserving old wrappers:

```text
accounting.ledger      source inputs → canonical ledger
accounting.core        pure money/time/normalization helpers
accounting.stage_d     canonical ledger → analytical tables
accounting.metrics     analytical tables → metric contracts
accounting.debt        ledger/debt rows → debt contracts
accounting.human       metric/debt contracts → human report surfaces
accounting.publish     human/report artifacts → frontend snapshot
accounting.cli         thin operational wrappers
accounting.support     logging/utils/support code
```

## Metrics package migration

Metrics are now the first package migration. Canonical imports live under:

- `accounting.metrics.io`
- `accounting.metrics.registry`
- `accounting.metrics.builders`
- `accounting.metrics.derive`
- `accounting.metrics.validate`
- `accounting.metrics.views`
- `accounting.metrics.drilldown`
- `accounting.metrics.build`

The old flat modules remain as compatibility wrappers during the transition.
