---
id: notes/library/10-operations/14-stage-output-checklist
title: "14 stage output checklist"
sidebar_label: "14 stage output checklist"
sidebar_position: 14
---

# 14 stage output checklist

Status: draft (code-anchored)
Last reviewed: 2026-05-22

## Purpose

Provide a fast operator checklist to confirm each stage produced required artifacts.

## Stage checklist

### Ingest (`make run-ingest`)

Must exist:
- `out/run/accounting/<RUN_ID>/ledger_canonical.csv`

### Materialize (`make run-materialize`)

Must exist:
- `out/run/accounting/<RUN_ID>/per_flow_time_long.freq=<FREQ>.csv`
- `out/run/accounting/<RUN_ID>/per_party_time_long.freq=<FREQ>.csv`
- `out/run/accounting/<RUN_ID>/daily_cash_position.csv`

### Views (`make run-views`)

Must exist:
- `out/run/accounting/<RUN_ID>/views/views_sanity.json`
- `out/run/accounting/<RUN_ID>/views/v_contributions_monthly.csv`
- `out/run/accounting/<RUN_ID>/views/v_opex_category_monthly.csv`

### Metrics (`make run-metrics`)

Must exist:
- `out/metrics/<RUN_ID>/metric_registry.csv`
- `out/metrics/<RUN_ID>/metric_values.csv`
- `out/metrics/<RUN_ID>/validation_report.csv`
- required `metric_views/*.csv` listed in runbook

### Human report (`make run-human-report`)

Must exist:
- `out/human_reports/<RUN_ID>/balance_human_v2/balance_humano_v2.html`
- `out/human_reports/<RUN_ID>/balance_human_v2/story_manifest.json`

### Publish (`make publish-latest`)

Must exist:
- `public/accounting/latest/manifest.json`
- published report/metrics/debt files listed in snapshot manifest
