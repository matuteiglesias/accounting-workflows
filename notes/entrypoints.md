---
id: notes/entrypoints
title: "Accounting Backend Entrypoints"
sidebar_label: "Accounting Backend Entrypoints"
---

# Accounting Backend Entrypoints

Status: authority draft
Last reviewed: 2026-05-10

## Purpose

This document identifies the current operational entrypoints for the accounting backend and distinguishes canonical, support, legacy, and experimental commands.

The Makefile is the command authority. Module CLIs remain useful implementation entrypoints, but users should start with `make help`.

## Status vocabulary

| Status | Meaning |
|---|---|
| canonical | Preferred current command/module for the pipeline |
| support | Useful diagnostic, validation, partial, or packaging command |
| legacy | Historical or compatibility surface; avoid new dependencies |
| experimental | Future/incomplete surface, not yet a reliable dependency |

## Canonical Make targets

| Target | Layer | Expands to / purpose |
|---|---|---|
| `make ledger` | Level 1 | Build the live canonical ledger via `run-ingest`. |
| `make materialize` | Level 2 | Build live materialized Stage D artifacts via `run-materialize`. |
| `make debt` | Level 3 | Resolve live internal debt artifacts via `run-debt`. |
| `make debt-views` | Level 3 | Build debt balance views via `run-debt-views`. |
| `make metrics` | Level 3 | Build metric values, registry, validation, views, and drilldowns via `run-metrics`. |
| `make human-report` | Level 4 | Build the current human balance report via `run-human-report`. |
| `make publish-latest` | Level 5 | Publish latest producer artifacts to `public/accounting/latest/*`. |
| `make publish` | Level 5 | Compatibility alias for `publish-latest`. |
| `make build-all` | composite | Run the full canonical build through publish. |
| `make build-report` | composite | Run through the current human report without publishing. |
| `make build-front` | composite | Publish the latest report/metrics/debt snapshot for frontend consumption. |

## Support Make targets

| Target | Purpose |
|---|---|
| `make doctor` | Compile-check key command modules and print Python version. |
| `make smoke` / `make smoke-accounting` | Run the fixture/offline smoke path through views. |
| `make validate` | Run lightweight command-surface checks. |
| `make clean-derived` | Remove derived accounting outputs under `out/` and `public/accounting/latest`. |
| `make run-downstream-from-ledger` | Reuse an existing canonical ledger and rebuild downstream artifacts. |
| `make run-metrics-and-human` | Reuse existing views and rebuild debt, metrics, and human report. |
| `make run-human-balance-only` | Reuse existing metrics and rebuild only the current human report. |

## Experimental Make targets

| Target | Purpose |
|---|---|
| `make front-report` | Build the future/front-oriented human report using `accounting.human.front`. This remains experimental. |

## Legacy aliases and compatibility targets

The older `run-*` targets remain available as compatibility aliases and implementation targets. They should be considered lower-level than the canonical target names above.

| Existing target | Current classification |
|---|---|
| `run-ingest` | compatibility implementation for `ledger` |
| `run-materialize` | compatibility implementation for `materialize` |
| `run-views` | support bridge for Stage D view composition |
| `run-debt` | compatibility implementation for `debt` |
| `run-debt-views` | compatibility implementation for `debt-views` |
| `run-debt-balance` | legacy compatibility alias for `run-debt-views` |
| `run-metrics` | compatibility implementation for `metrics` |
| `run-human-report` | compatibility implementation for `human-report` |
| `run-human-balance` | legacy compatibility alias for `run-human-report` |
| `run-accounting` / `run-accounting-full` | compatibility implementation for `build-report` |
| `run` / `run-all` | legacy convenience aliases for `run-accounting` |

## Canonical module CLIs

| Module command | Layer | Status | Notes |
|---|---|---|---|
| `python -m accounting.ledger.ingest ...` | Level 1 | canonical implementation | Builds canonical ledger from fixture or Google Sheet. |
| `python -m accounting.stage_d.materialize ...` | Level 2 | canonical implementation | Materializes the canonical ledger into Stage D CSV artifacts. |
| `python -m accounting.views ...` | Level 2/3 | support | Builds view tables from Stage D artifacts. |
| `python -m accounting.debt.resolve ...` | Level 3 | canonical implementation | Current debt resolver. |
| `python -m accounting.debt.balance_views ...` | Level 3 | canonical implementation | Builds debt balance view CSVs. |
| `python -m accounting.metrics.build ...` | Level 3 | canonical implementation | Main metric artifact builder. |
| `python -m accounting.human.document ...` | Level 4 | current canonical implementation | Current human report factory. |
| `python -m accounting.publish.latest ...` | Level 5 | current canonical publish implementation | Builds frontend-safe latest snapshot. |
| `python -m accounting.human.front ...` | Level 4/5 | experimental | Future/front-oriented report factory. |

## Removed compatibility module paths

The flat compatibility shim modules were removed after import, Makefile, and docs checks showed no active internal dependency. Use the canonical package paths in the table above instead.

| Removed path | Replacement |
|---|---|
| `python -m accounting.ingest ...` | `python -m accounting.ledger.ingest ...` |
| `python -m accounting.materialize ...` | `python -m accounting.stage_d.materialize ...` |
| `python -m accounting.resolve_internal_debt_v2 ...` | `python -m accounting.debt.resolve ...` |
| `python -m accounting.build_debt_balance_views ...` | `python -m accounting.debt.balance_views ...` |
| `python -m accounting.build_metric_values ...` | `python -m accounting.metrics.build ...` |
| `python -m accounting.human_balance_document_factory ...` | `python -m accounting.human.document ...` |
| `python -m accounting.human_balance_front_factory ...` | `python -m accounting.human.front ...` |
| `python -m accounting.publish_latest ...` | `python -m accounting.publish.latest ...` |


## Operational rule

A future user should normally run:

```text
make build-all
```

For partial rebuilds, use the named layer targets in order:

```text
make ledger
make materialize
make debt
make debt-views
make metrics
make human-report
make publish-latest
```
