
---

## `src/notes/module_inventory.md`

```markdown
# Accounting Backend Module Inventory

Status: draft  
Scope: `src/accounting/*`  
Last reviewed: 2026-05-10

## Purpose

This document classifies accounting backend modules by architectural layer, current status, canonical outputs, and known caveats.

The goal is to make the codebase easier to navigate without immediately refactoring it.

## Status vocabulary

| Status | Meaning |
|---|---|
| canonical | Current preferred implementation |
| support | Required helper or downstream support module |
| current | Actively used current implementation, even if it may later be replaced |
| legacy candidate | Likely superseded, retained only if still referenced |
| experimental | Future/stub/incomplete surface |
| unknown | Requires Makefile or runtime evidence |

## Module inventory

| Module | Layer | Status | Canonical outputs | Notes |
|---|---|---|---|---|
| `config.py` | Configuration | support | `Config` object, resolved path/options map | Loads YAML/JSON config and defaults for paths, fixtures, Google Sheets, output dirs, frequency, dry-run, force, base currency, FX table, manifests, and validation headers. |
| `ingest.py` | Level 1 - canonical ledger | canonical | canonical ledger DataFrame, ingest anomalies | Primary source canonicalization layer. The internal contract includes `tx_id`, `Date`, `amount`, `Currency`, `payer`, `receiver`, `Flujo`, `Tipo`, `status`, `Box`, source fields, and anomalies. |
| `core_timeseries.py` | Pure computation | support | in-memory aggregate DataFrames | Pure deterministic time-series primitives. No I/O. Used by materialization and possibly views. |
| `materialize.py` | Level 2 - materialized views | canonical | `per_flow_time_long.freq=*.csv`, `per_party_time_long.freq=*.csv`, `daily_cash_position.csv`, loan/time outputs, materialization manifest | Main CSV materialization layer from canonical ledger. |
| `views.py` | Level 2/3 - view composition | support/current | report/view tables, loaded Stage D artifacts | Bridges materialized outputs into higher-level views. Important doctrine: Stage D materialized artifacts are source of truth; legacy report artifacts are best-effort only. |
| `metrics_views.py` | Level 3 - metric views | support | `income_statement_monthly_last6.csv`, rent rollups, flow rollups, draws discipline views | Builds human/metric-oriented views from ledger and materialized artifacts. |
| `metrics_io.py` | Level 3 - metric contract | canonical | normalized `metric_values` schema | Defines required metric values columns and `MetricsContext`. |
| `metrics_registry.py` | Level 3 - metric contract | canonical | `metric_registry.csv` schema / registry DataFrame | Defines metric registry structure through `MetricSpec` and registry normalization. |
| `metrics_builders.py` | Level 3 - metric builders | canonical support | leaf metric DataFrames | Builds leaf metrics from `MetricsContext`. |
| `metrics_derive.py` | Level 3 - derived metrics | support | derived metric value frames | Provides formula helpers such as sum components and subtract formulas. |
| `metrics_validate.py` | Level 3 - validation | support | validation issue DataFrames, `validation_report.csv` via orchestrator | Checks uniqueness, registry integrity, leaf builder keys, and known metric IDs. |
| `build_metric_values.py` | Level 3 - metric orchestration | canonical | `metric_values.csv`, `metric_registry.csv`, `validation_report.csv`, wide views, statement views, metric views, drilldowns, build manifest | Main metric artifact builder. Reads latest or explicit accounting run root and writes metric artifacts to `out/metrics/latest` by default. |
| `metric_drilldown.py` | Level 3/4 - evidence drilldowns | support | `metric_drilldown_index.csv`, detail CSVs, drilldown manifest | Produces traceable drilldowns for selected metrics such as rent, opex, and personal draws. |
| `resolve_internal_debt_v2.py` | Level 3 - debt resolution | canonical candidate | `debt_open_items.csv`, `debt_allocations.csv`, `debt_repayment_events.csv`, timeline, reconciliation | Current debt engine candidate. Handles ledger normalization, open items, repayment allocation, timeline, and status reconciliation. |
| `build_debt_balance_views.py` | Level 3 - debt analytical views | support | `debt_balance_monthly.csv`, `debt_balance_quarterly.csv`, `debt_balance_yearly.csv` or equivalent period views | Builds debt balances over time from open debt items. |
| `human_balance_tables.py` | Level 4 - human tables | canonical support | human-facing tables keyed by table specs | Defines `HumanTableSpec` and default human table registry. This is the preferred decomposition seam for human-facing reporting. |
| `human_balance_document_factory.py` | Level 4 - human report | current/canonical | current human balance HTML/report outputs, drilldown HTML pages | Current report factory around `balance_human_v2`. Should compose tables and evidence, not absorb new accounting semantics. |
| `human_balance_front_factory.py` | Level 4/5 - front report | experimental/stub/future | front report pages, `front_manifest.json`, assets, standalone tables | Explicitly stubbed/front-oriented architecture. Intended to migrate away from the legacy balance report monolith. Do not treat as canonical yet. |
| `publish_latest.py` | Level 5 - frontend handoff | unknown/support | `public/accounting/latest/*` or viewer-ready snapshot | Not reviewed in this batch. Likely important for accounting-viewer handoff. Needs inspection. |
| `logging_utils.py` | Infrastructure | support | configured logging | Shared logging support. |
| `utils.py` | Infrastructure | support/unknown | atomic writes, run ID resolution, CSV helpers | Needs inspection to classify stable helper surface. |
| `models.py` | Data models | unknown | model classes or schema helpers | Needs inspection. |
| `manifest.py` | Artifact metadata | unknown/support | run/artifact manifest | Needs inspection. May become important for artifact ladder. |
| `reports.py` | Reporting | unknown/possibly legacy | report artifacts | Needs inspection. |
| `plots.py` | Visualization | support/possibly optional | chart files | Needs inspection. |
| `build_debt_balance_views.py` | Debt views | support | debt balance time views | Keep as downstream of `resolve_internal_debt_v2.py`. |

## High-confidence architecture seams

```text
ingest.py
  → canonical ledger

canonical ledger
  → materialize.py
  → Stage D CSV artifacts

Stage D artifacts + debt artifacts
  → build_metric_values.py
  → metric_values.csv + metric_registry.csv + validation_report.csv + metric_views

metric artifacts + debt artifacts
  → human_balance_tables.py
  → human_balance_document_factory.py

front/report artifacts
  → publish_latest.py / accounting-viewer
```


Current module decisions
Debt

resolve_internal_debt_v2.py is the current canonical candidate. The removed resolve_internal_debt.py should remain absent from docs and Make targets.

Human balance

human_balance_document_factory.py is current.
human_balance_front_factory.py is future/stub.
human_balance_tables.py is the reusable seam and should be protected from report-factory bloat.

Metrics

The metrics subsystem is the cleanest part of the codebase. Its schema modules should be treated as examples for future contracts:

metrics_io.py
metrics_registry.py
metrics_validate.py
metrics_derive.py
metrics_builders.py
build_metric_values.py
Open questions
Which Makefile target is the true full pipeline?
Does publish_latest.py define the canonical frontend handoff?
Are manifest.py, models.py, reports.py, and plots.py still active?
What is the current canonical output root: out/run/accounting/<run_id>, out/metrics/latest, out/front, or public/accounting/latest?
Should debt artifacts be promoted to a more explicit contract layer?
