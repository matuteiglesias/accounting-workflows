---
id: notes/module_inventory
title: "Accounting Backend Module Inventory"
sidebar_label: "Accounting Backend Module Inventory"
---

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
| `contracts/models.py` | contracts/domain | support | Shared Pydantic domain models; not part of the active pipeline path yet. |
| `artifacts/manifest.py` | artifacts | support | Run/stage manifest, artifact row, structure hint, hash, and metadata helpers. |
| `core/timeseries.py` | core | support | Pure time-series aggregation and expansion helpers. |
| `ledger/ingest.py` | Level 1 canonical ledger | canonical | Reads source inputs and builds the canonical ledger/anomaly contract. |
| `stage_d/materialize.py` | Level 2 materialized artifacts | canonical | Writes per-flow, per-party, daily cash, loan/time, and materialization metadata artifacts. |
| `views.py` | Level 2/3 view bridge | current support | Builds/loads view tables from Stage D materialized artifacts; legacy report inputs are best-effort only. |
| `metrics/io.py` | Level 3 metric contract | canonical | Defines metric-values schema and metric context. |
| `metrics/registry.py` | Level 3 metric contract | canonical | Defines metric registry specs and normalization. |
| `metrics/builders.py` | Level 3 metric builders | canonical support | Builds leaf metric value frames from context. |
| `metrics/derive.py` | Level 3 derived metrics | support | Derivation helpers for parent/formula metrics. |
| `metrics/validate.py` | Level 3 validation | support | Registry/value validation and validation-report rows. |
| `metrics/views.py` | Level 3 metric views | support | Builds income/rent/flow/draws/debt-facing metric view CSVs. |
| `metrics/drilldown.py` | Level 3/4 evidence | support | Builds drilldown detail/index/manifest artifacts for report evidence. |
| `metrics/build.py` | Level 3 metrics orchestration | canonical | Main metrics build entrypoint; writes registry, values, validation, views, drilldowns, and manifest. |
| `debt/resolve.py` | Level 3 debt resolution | canonical | Current debt engine; writes open items, allocations, repayment events, timeline, and reconciliation. |
| `debt/balance_views.py` | Level 3 debt views | canonical | Builds daily/monthly/quarterly/yearly debt balance views from debt open items. |
| `human/tables.py` | Level 4 human tables | canonical support | Defines reusable human-facing table specs and table builders. |
| `human/document.py` | Level 4 human report | current canonical | Current `balance_human_v2` human report/document factory. |
| `human/front.py` | Level 4/5 front report | experimental | Future/front-oriented report builder; not production canonical yet. |
| `publish/latest.py` | Level 5 frontend snapshot | current canonical | Packages selected latest artifacts into `public/accounting/latest/*`. |
| `publish/manifest.py` | Level 5 frontend snapshot | support | Defines the frontend snapshot manifest schema helper. |
| `publish/snapshot.py` | Level 5 frontend snapshot | support seam | Reserved seam for snapshot copy/filter helpers. |
| `human/reports.py` | legacy/reporting | legacy | Deprecated bridge to `views.export_views`; do not use for new canonical flow. |
| `viz/plots.py` | support/visualization | support | Optional matplotlib plot generation utility. |
| `support/run_id.py` | support | support | Run-id inference and resolution helpers. |
| `support/io.py` | support | support | CSV/path helpers, atomic DataFrame writes, and manifest JSON writes. |
| `support/currency.py` | support | support | Currency normalization, amount normalization, and currency conversion helpers. |
| `support/env.py` | support | support | Environment variable helpers and project-wide env constants. |
| `support/hashing.py` | support | support | File and source dataframe hashing helpers. |
| `support/partitions.py` | support | support | Partition metadata and parquet write helpers. |

## Removed compatibility shims

The following flat modules were removed after the canonical packages took ownership and repository checks found no active internal imports or Makefile dependencies:

- `core_timeseries.py` → `core/timeseries.py`
- `ingest.py` → `ledger/ingest.py`
- `materialize.py` → `stage_d/materialize.py`
- `metrics_io.py`, `metrics_registry.py`, `metrics_builders.py`, `metrics_derive.py`, `metrics_validate.py`, `metrics_views.py`, `metric_drilldown.py`, `build_metric_values.py` → `metrics/*`
- `resolve_internal_debt_v2.py` → `debt/resolve.py`
- `build_debt_balance_views.py` → `debt/balance_views.py`
- `human_balance_tables.py`, `human_balance_document_factory.py`, `human_balance_front_factory.py` → `human/*`
- `publish_latest.py` → `publish/latest.py`

## Migrated owner modules

The following formerly flat owner modules now live under their owning packages after imports were updated and flat paths were removed:

- `manifest.py` → `artifacts/manifest.py`
- `utils.py` → `support/run_id.py`, `support/io.py`, `support/currency.py`, `support/env.py`, `support/hashing.py`, and `support/partitions.py`
- `models.py` → `contracts/models.py`
- `reports.py` → `human/reports.py`
- `plots.py` → `viz/plots.py`
- empty `hashlib` placeholder removed

## Package map

When layer boundaries are proven, modules can migrate toward these packages while preserving old wrappers:

```text
accounting.ledger      source inputs → canonical ledger
accounting.core        pure money/time/normalization helpers
accounting.stage_d     canonical ledger → analytical tables
accounting.metrics     analytical tables → metric contracts
accounting.debt        ledger/debt rows → debt contracts
accounting.contracts   shared domain model contracts
accounting.human       metric/debt contracts → human report surfaces
accounting.viz         optional visualization helpers
accounting.publish     human/report artifacts → frontend snapshot
accounting.cli         thin operational wrappers
accounting.artifacts   run/stage manifests and artifact metadata
accounting.support     project-wide helper modules
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

The old flat metrics compatibility modules have been removed; use the `accounting.metrics.*` package paths above.
