---
id: notes/current_state_map
title: "Accounting Backend Current State Map"
sidebar_label: "Accounting Backend Current State Map"
---

# Accounting Backend Current State Map

Status: authority draft
Last reviewed: 2026-05-10

## Purpose

The accounting backend converts messy household/property accounting records into validated financial views, debt-resolution artifacts, human-readable reports, and frontend-ready snapshots.

The current system should be understood as this artifact ladder:

```text
source inputs
  → canonical ledger
  → materialized analytical artifacts
  → metric/debt contracts
  → human report surfaces
  → frontend snapshot
```

This document names the current layers and decisions before any package movement. The goal is legibility and reduced accidental coupling, not aesthetic folder tidiness.

## Current pipeline spine

```text
Google Sheet / fixture CSV / local export
  → accounting.ledger.ingest
      ledger_canonical.csv
      ingest anomalies
  → accounting.stage_d.materialize
      per_flow_time_long.freq=*.csv
      per_party_time_long.freq=*.csv
      daily_cash_position.csv
      Stage D/materialization metadata
  → accounting.views
      current view tables built from Stage D outputs
  → accounting.debt.resolve
      debt_open_items.csv
      debt_allocations.csv
      debt_repayment_events.csv
      debt_resolution_timeline.csv
      debt_status_reconciliation.csv
  → accounting.debt.balance_views
      debt_balance_daily.csv
      debt_balance_monthly.csv
      debt_balance_quarterly.csv
      debt_balance_yearly.csv
  → accounting.metrics.build
      metric_values.csv
      metric_registry.csv
      validation_report.csv
      metric_views/*
      metric_drilldown/*
      build_manifest.json
  → accounting.human.document
      balance_humano_v2.html
      story_manifest.json
      human-readable report assets
  → accounting.publish.latest
      public/accounting/latest/*
      frontend-safe manifest and selected artifacts
```

The Makefile currently orchestrates this as live `run-*` targets and now exposes shorter canonical aliases such as `ledger`, `materialize`, `debt`, `metrics`, `human-report`, and `publish`.

## Layer responsibilities

### Source inputs

Responsible for external records only: Google Sheets, fixtures, raw CSVs, and manual exports. These inputs are intentionally not stable contracts.

### Canonical ledger

`accounting.ledger.ingest` is responsible for turning source records into canonical ledger rows; the old `accounting.ingest` compatibility wrapper has been removed. Its stable internal contract includes columns such as `tx_id`, `Date`, `amount`, `Currency`, `payer`, `receiver`, `Flujo`, `Tipo`, `status`, `Box`, `source_file`, `source_row`, `ingest_ts`, and notes/anomaly metadata.

### Materialized analytical artifacts

`accounting.stage_d.materialize` is responsible for CSV-first analytical tables derived from the canonical ledger; the old `accounting.materialize` compatibility wrapper has been removed. `accounting.views` remains a support bridge for view tables; the doctrine is that Stage D materialized artifacts are source-of-truth for views.

### Metric/debt contracts

The metrics subsystem is the most mature contract layer today. Its canonical implementation now lives under `accounting.metrics` (`io`, `registry`, `builders`, `derive`, `validate`, `views`, `drilldown`, and `build`), while the old flat module compatibility wrappers have been removed.

Debt resolution is also a named contract layer. `accounting.debt.resolve` is the current debt resolver, and `accounting.debt.balance_views` derives time-series debt balance artifacts from debt open items. The old flat human compatibility module paths have been removed; use the `accounting.human.*` package paths directly.

### Human/report surfaces

`accounting.human.tables` defines reusable human-facing table specs and builders. `accounting.human.document` is the current human report factory. `accounting.human.front` is future/experimental and should not be promoted to production until its output and consumer are clear. The old flat human compatibility module paths have been removed; use the `accounting.human.*` package paths directly.

### Frontend snapshot

`accounting.publish.latest` packages selected latest metrics, debt, and human report artifacts into `public/accounting/latest/*`; the old `publish_latest.py` compatibility shim has been removed. The frontend should consume this snapshot instead of internal producer directories.

## Current decisions

| Decision | Current authority |
|---|---|
| Debt resolver | `accounting.debt.resolve` |
| Human report factory | `accounting.human.document` |
| Front report factory | `accounting.human.front` is experimental/future |
| Metrics layer | Most mature contract layer; safest first code refactor candidate later |
| View source of truth | Stage D materialized outputs |
| Legacy report artifacts | Optional compatibility only |
| Frontend handoff | `accounting.publish.latest` writes `public/accounting/latest/*` with `accounting_frontend_snapshot.v1` manifest |

## Known coupling risks

- Report factories can become business-logic dumps if new metric formulas or debt allocation logic are added there.
- Some modules infer candidate paths dynamically for convenience; docs and Make targets should identify the preferred paths.
- `views.py` is a transitional support bridge between materialized artifacts and higher-level report/metric consumers.
- `accounting.human.front` should remain experimental until it has a stable consumer and contract.

## Migration posture

Do not begin by moving all modules. First keep these documents and Makefile commands authoritative. Later migrations should introduce package folders only where the boundaries are already proven, starting with metrics.
