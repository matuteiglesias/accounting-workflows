# Accounting Backend Entrypoints

Status: draft  
Scope: accounting-backend current command surface  
Last reviewed: 2026-05-10

## Purpose

This document identifies the current operational entrypoints for the accounting backend.

The goal is to distinguish:

- canonical commands: expected current production or main workflow commands
- support commands: useful downstream, diagnostic, or auxiliary commands
- legacy commands: retained for historical or compatibility reasons
- experimental commands: promising but not yet canonical

This file should be reconciled against the Makefile. When Makefile targets and module CLIs disagree, the Makefile may describe operational habit, while the module CLIs describe available implementation surfaces.

## Status vocabulary

| Status | Meaning |
|---|---|
| canonical | Preferred command or module entrypoint for the current pipeline |
| support | Useful supporting command, but not the primary spine |
| legacy | Historical or superseded entrypoint |
| experimental | Future or incomplete surface, not yet a reliable dependency |
| unknown | Needs Makefile/run evidence |

## Entrypoints

| Command | Layer | Purpose | Status |
|---|---|---|---|
| `python -m accounting.ingest ...` | Level 1 - canonical ledger | Build canonical ledger from fixture, Google Sheet, or source ledger rows | canonical candidate |
| `python -m accounting.materialize ...` | Level 2 - materialized views | Materialize canonical ledger into per-flow, per-party, daily cash, loan, and manifest artifacts | canonical candidate |
| `python -m accounting.views ...` | Level 2/3 - report views | Build or load report/view tables from materialized Stage D artifacts | support |
| `python -m accounting.resolve_internal_debt_v2 --write-dir ...` | Level 3 - debt resolution | Resolve internal debts, repayments, allocations, timeline, and reconciliation from canonical ledger or ledger CSV | canonical candidate |
| `python -m accounting.build_debt_balance_views ...` | Level 3 - debt analytical views | Build debt balance views over time from resolved debt open items | support |
| `python -m accounting.build_metric_values --run-root ... --out-dir ...` | Level 3 - metrics | Build `metric_values.csv`, `metric_registry.csv`, validation report, wide views, statement views, metric views, and drilldown artifacts | canonical |
| `python -m accounting.human_balance_document_factory ...` | Level 4 - human report | Build the current human balance report/document from metrics, human tables, and drilldowns | canonical current |
| `python -m accounting.human_balance_front_factory --run-root ... --metrics-dir ... --write-dir ...` | Level 4/5 - front report | Build front-oriented human balance report pages and manifest | experimental |
| `python -m accounting.publish_latest ...` | Level 5 - frontend handoff | Publish or sync latest accounting artifacts for frontend/viewer consumption | unknown, likely support/canonical once verified |
| `make ...` | orchestration | Makefile target surface for the full or partial pipeline | unknown until Makefile reconciliation |

## Probable canonical spine

The current canonical spine is probably:

```text
ingest
  → materialize
  → resolve_internal_debt_v2
  → build_debt_balance_views
  → build_metric_values
  → human_balance_document_factory
  → publish_latest / accounting-viewer
```


This should be confirmed against the Makefile and latest successful run artifacts.

Notes on known command risks
Makefile drift

The Makefile may contain shortcuts or historical targets that no longer reflect the clean architecture. Treat Make targets as operational evidence, not necessarily architectural truth.

Debt resolver

resolve_internal_debt_v2.py is the current debt resolver candidate. The old resolve_internal_debt.py has been removed and should not appear as an active entrypoint.

Human balance front

human_balance_front_factory.py is not yet the current report authority. It is a future/front-oriented architecture. Use human_balance_document_factory.py as current unless a runbook or Makefile target says otherwise.

Reconciliation checklist
 Inspect Makefile targets.
 Mark each target as canonical/support/legacy/experimental.
 Confirm the latest full pipeline run path.
 Confirm whether publish_latest.py is currently used.
 Confirm whether human_balance_front_factory.py is called anywhere.
 Confirm the exact output root for current production artifacts.
 
 
 
