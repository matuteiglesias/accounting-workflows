---
id: notes/artifact_ladder
title: "Accounting Artifact Ladder"
sidebar_label: "Accounting Artifact Ladder"
---

# Accounting Artifact Ladder

Status: authority draft
Last reviewed: 2026-05-10

## Purpose

The accounting backend should be read as a layered artifact pipeline, not as a flat collection of scripts.

```text
source inputs
  → canonical ledger
  → materialized analytical artifacts
  → metric/debt contracts
  → human report surfaces
  → frontend snapshot
```

Each layer should read from the previous layer's contracts whenever possible. Downstream report builders should not read raw source inputs, metrics should read canonical/materialized artifacts, and frontend code should read frontend snapshots rather than internal work folders.

## Level 0 — Source inputs

Examples:

- Google Sheet tabs.
- Fixture CSVs.
- Raw ledger rows or local exports.

Producer: external systems, humans, fixtures.
Primary consumer: `accounting.ledger.ingest`.

Governance rule: source inputs are not stable downstream contracts. They can change shape, naming, and availability.

## Level 1 — Canonical ledger

Primary artifacts:

- `ledger_canonical.csv`.
- Ingest anomalies, currently attached to the canonical DataFrame and expected to become an explicit artifact.

Producer: `accounting.ledger.ingest`.
Primary consumers: `accounting.stage_d.materialize`, debt resolution, metric/drilldown builders, and report evidence loaders.

Governance rule: this is the first stable accounting fact layer. All downstream analytical artifacts should be traceable to canonical ledger rows.

## Level 2 — Materialized analytical artifacts

Primary artifacts:

- `per_flow_time_long.freq=*.csv`.
- `per_party_time_long.freq=*.csv`.
- `daily_cash_position.csv`.
- `box_balance_time_long.freq=*.csv` and `box_flow_balance_time_long.freq=*.csv` where present.
- Stage D/materialization metadata and manifest files.
- `views/*` outputs built from materialized Stage D artifacts.

Producers: `accounting.stage_d.materialize`, with `accounting.views` as the current view-composition bridge.
Primary consumers: metrics, human tables, and report factories.

Governance rule: materialized Stage D outputs are the source of truth for views. Legacy report artifacts are optional compatibility inputs only and must not become required upstream dependencies.

## Level 3 — Metric and debt contracts

Primary metric artifacts:

- `metric_values.csv`.
- `metric_registry.csv`.
- `validation_report.csv`.
- `metric_views/*`.
- `metric_drilldown/*`.
- `build_manifest.json`.

Primary debt artifacts:

- `debt_open_items.csv`.
- `debt_allocations.csv`.
- `debt_repayment_events.csv`.
- `debt_resolution_timeline.csv`.
- `debt_status_reconciliation.csv`.
- `debt_balance_daily.csv`.
- `debt_balance_monthly.csv`.
- `debt_balance_quarterly.csv`.
- `debt_balance_yearly.csv`.

Producers: `accounting.metrics.build`, `accounting.debt.resolve`, and `accounting.debt.balance_views`.
Primary consumers: human tables, human reports, validation gates, and publish/frontend packaging.

Governance rule: this is the most important contract layer for downstream interpretation. Reports should read metric/debt contracts rather than recomputing core metric formulas or debt allocation rules.

## Level 4 — Human/report surfaces

Primary artifacts:

- Human table specs and generated human-facing tables.
- `balance_humano_v2.html`.
- `story_manifest.json`.
- Drilldown-linked HTML/report pages.
- Experimental front report pages and blocks.

Producers: `accounting.human.tables`, `accounting.human.document`, and experimental `accounting.human.front`.
Primary consumers: humans and publish/frontend packaging.

Governance rule: report factories compose metric, debt, and table contracts. They should not become the place where new ledger canonicalization, core metric formulas, or debt allocation semantics are defined.

## Level 5 — Frontend/public snapshot

Primary artifacts:

- `public/accounting/latest/*`.
- `public/accounting/latest/manifest.json` using schema `accounting_frontend_snapshot.v1`.
- Frontend-safe `report/`, `metrics/`, and `debt/` subsets.

Producer: `accounting.publish.latest`.
Primary consumer: accounting viewer/static frontend surfaces.

Governance rule: the frontend reads the published snapshot. It should not read arbitrary `out/run`, `out/metrics`, `out/debt_resolution`, or `out/human_reports` internals directly.

## Current architectural decisions

- `accounting.debt.resolve` is the current debt resolver; the old `resolve_internal_debt_v2.py` compatibility shim has been removed.
- `accounting.human.document` is the current human report factory; the old `human_balance_document_factory.py` compatibility shim has been removed.
- `accounting.human.front` is experimental/future until it has a clear consumer and output contract; the old `human_balance_front_factory.py` compatibility shim has been removed.
- The metrics subsystem is the most mature contract layer and is the best first candidate for a later compatibility-package refactor.
- Stage D materialized outputs are authoritative for view construction.
- Legacy report artifacts are optional compatibility aids, not required pipeline inputs.
