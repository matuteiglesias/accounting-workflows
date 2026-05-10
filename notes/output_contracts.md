
---

## `src/notes/output_contracts.md`

```markdown
# Accounting Backend Output Contracts

Status: draft  
Scope: stable and emerging accounting artifacts  
Last reviewed: 2026-05-10

## Purpose

This document defines the main output contracts in the accounting backend.

The goal is to prevent downstream consumers from reading arbitrary intermediate files and to make the stable seams explicit.

## Stability vocabulary

| Stability | Meaning |
|---|---|
| stable | Safe for downstream modules to depend on |
| current | Used today, but may still evolve |
| experimental | Future-facing or incomplete |
| legacy | Historical, avoid new dependencies |
| unknown | Requires runtime/Makefile verification |

---

# 1. Canonical ledger

## `ledger_canonical.csv`

Producer:  
`accounting.ingest` and/or the pipeline wrapper that persists the DataFrame returned by `build_ledger_base(...)`.

Consumer:  
`materialize.py`, `metrics_views.py`, `resolve_internal_debt_v2.py`, `metric_drilldown.py`, report builders.

Required columns:

```text
tx_id
Date
amount
Currency
payer
receiver
Flujo
Tipo
status
Box
source_file
source_row
ingest_ts
notes
```


Common optional columns:

amount_cents
base_amount
Detalle
Lugar
medio
tag

Validation:

Required accounting fields must exist before downstream materialization.
Date must parse as date/datetime.
amount must parse as numeric.
Currency must be non-empty.
tx_id should be stable and non-empty when require_tx_id=True.
Ingest anomalies should be captured separately and not silently ignored.

Example path:

out/run/accounting/<run_id>/ledger_canonical.csv

Stability:
stable candidate.

Notes:

The canonical ledger is the most important accounting contract. Everything else should be traceable back to this artifact.

2. Metric values
metric_values.csv

Producer:
accounting.build_metric_values.

Consumer:
human_balance_tables.py, human_balance_document_factory.py, human_balance_front_factory.py, frontend/viewer surfaces, validation, statement views.

Required columns:

metric_id
period_grain
period
currency
value
run_id
as_of_date
source_layer
build_status
build_detail

Validation:

Enforced by ensure_metric_values_schema(...).
Duplicate keys should be checked across:
metric_id
period_grain
period
currency
run_id
as_of_date
metric_id should exist in metric_registry.csv.
value should be numeric.
build_status should default to ok when not set.

Example path:

out/metrics/latest/metric_values.csv

or:

out/metrics/<run_id>/metric_values.csv

Stability:
stable.

Notes:

This is the central numeric read model for accounting metrics. Downstream reports should prefer this over recomputing metrics from raw ledger rows.

3. Metric registry
metric_registry.csv

Producer:
accounting.metrics_registry through accounting.build_metric_values.

Consumer:
metric validation, report labels, statement views, frontend display logic.

Required columns:

metric_id
statement
section
label
agg_rule
is_leaf
source_layer
builder_key
parent_metric_id
display_code
sort_key
currency_mode
status
notes

Validation:

metric_id must be unique.
active leaf metrics should have builder_key.
derived metrics should have enough metadata to explain construction.
status should default to active.
currency_mode should default to by_currency.

Example path:

out/metrics/latest/metric_registry.csv

Stability:
stable.

Notes:

The registry is the semantic layer for metrics. It should become the source for labels, ordering, grouping, and report interpretation.

4. Validation report
validation_report.csv

Producer:
accounting.metrics_validate through accounting.build_metric_values.

Consumer:
human balance reports, QA checks, pipeline health checks, release gates.

Required columns:

level
check_name
message
n_rows

Validation:

The file may be empty or contain zero rows if no issues are found.
level should distinguish at least error and warning.
Any error level issue should be treated as a failed or degraded build unless explicitly waived.

Example path:

out/metrics/latest/validation_report.csv

Stability:
stable candidate.

Notes:

This should become part of the pipeline gate. Reports should show or link validation status instead of hiding it.

5. Debt open items
debt_open_items.csv

Producer:
accounting.resolve_internal_debt_v2.

Consumer:
build_debt_balance_views.py, metrics, human balance tables, debt reports.

Required columns:

debt_id
source_tx_id
opened_at
debtor
creditor
currency
item_type
original_amount
open_amount
detalle
lugar
issuer
ledger_status
engine_status
closed_at

Validation:

opened_at must parse as date.
debtor, creditor, currency, and item_type must be non-empty.
original_amount and open_amount must be numeric.
engine_status should be one of open or closed.
item_type should be one of the valid debt types, currently Prestamo or Interes.
Closed items should have closed_at when available.
Reconciliation with ledger status should be inspected through the resolver's reconciliation output.

Example path:

out/debt_resolution/<run_id>/debt_open_items.csv

or latest pointer:

out/debt_resolution/latest/debt_open_items.csv

Stability:
current/canonical candidate.

Notes:

This is the main debt-state contract. Debt balance views should consume this or its derived balance artifacts, not re-resolve debts independently.

6. Human tables
Human table artifacts

Producer:
accounting.human_balance_tables.

Consumer:
human_balance_document_factory.py, human_balance_front_factory.py, report pages, frontend surfaces.

Required structure:

Each human table should have:

slug
title
builder_key
group
notes
enabled_by_default

Expected output shape:

tables keyed by slug
table specs keyed by slug
optional CSV/HTML exports per table

Important table groups include:

liquidity
income
flows
debt
validation
methodology

Validation:

Every generated table should have a corresponding HumanTableSpec.
Empty tables should be either hidden, marked partial, or explicitly included with a note.
Report factories should not invent accounting semantics outside table builders.
Table slugs should be stable because front/report blocks depend on them.

Example paths:

out/human_reports/<run_id>/tables/*.csv
out/front/<run_id>/tables/*.csv
out/front/<run_id>/html/*.html

Stability:
current/stable seam.

Notes:

This is the best decomposition seam for human-facing reporting. Report factories should compose human tables into narratives, not become the main business-logic layer.

7. Frontend snapshot
Frontend/public accounting snapshot

Producer:
publish_latest.py or equivalent publish/sync step.

Consumer:
accounting-viewer, static/public surfaces, human review UI.

Expected contents:

manifest.json
story_manifest.json
metrics/*
debt/*
human reports or front pages
latest pointers

Required fields for manifest:

built_at
source_run_id
source_paths
metrics_dir
debt_dir
report_dir
status
files

Validation:

Snapshot should only expose frontend-safe artifacts.
It should not expose raw private input credentials, service account paths, or local-only source paths.
It should contain enough provenance to trace back to the source run.
It should be atomically replaceable or versioned with a latest pointer.

Example paths:

public/accounting/latest/*
accounting-viewer/public/accounting/latest/*
accounting-viewer/accounting_surface/data/*

Stability:
unknown/current, pending inspection of publish_latest.py.

Notes:

This is the accounting equivalent of a public snapshot layer. It should be read-only for the frontend. The frontend should not become the source of truth.

Contract ladder summary
Level 1
  ledger_canonical.csv

Level 2
  per_flow_time_long.freq=*.csv
  per_party_time_long.freq=*.csv
  daily_cash_position.csv
  views/*

Level 3
  metric_values.csv
  metric_registry.csv
  validation_report.csv
  debt_open_items.csv
  debt_balance_*.csv
  metric_drilldown/*

Level 4
  human tables
  human balance reports
  front report pages

Level 5
  public/accounting/latest/*
  accounting-viewer data
Next contract work
 Inspect publish_latest.py.
 Confirm exact output paths from Makefile.
 Add a build_manifest.json contract.
 Add explicit debt_reconciliation.csv contract.
 Decide whether debt_balance_monthly.csv, debt_balance_quarterly.csv, and debt_balance_yearly.csv are stable contracts.
 Add schema checks for frontend snapshot manifest.
