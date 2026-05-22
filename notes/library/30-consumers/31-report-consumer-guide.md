---
id: notes/library/30-consumers/31-report-consumer-guide
title: "31 report consumer guide"
sidebar_label: "31 report consumer guide"
sidebar_position: 31
---

# 31 report consumer guide

Status: draft (code-anchored)
Last reviewed: 2026-05-22

## Purpose

Help humans and agents consume accounting outputs without digging through internal pipeline folders.

## Which artifact each audience should read

### 1) Stakeholder / family reader

Primary artifact:
- `public/accounting/latest/report/balance_humano_v2.html`

Why:
- this is the human-oriented narrative/report surface produced from metrics + debt artifacts.

### 2) Operator / support responder

Primary artifacts:
- `public/accounting/latest/manifest.json`
- `public/accounting/latest/report/story_manifest.json`
- selected metrics and debt CSVs under `public/accounting/latest/metrics/*` and `public/accounting/latest/debt/*`

Why:
- these show what was actually published for consumers.

### 3) Analyst / power user

Primary artifacts:
- published metric views in `public/accounting/latest/metrics/metric_views/*`
- published debt tables in `public/accounting/latest/debt/*`

Optional deep-dive (debug only):
- `out/metrics/<RUN_ID>/metric_views/*`
- `out/debt_resolution/<RUN_ID>/*`

## Internal-only vs consumer-safe directories

Consumer-safe (official handoff):
- `public/accounting/latest/*`

Internal producer directories (debug/backstage):
- `out/run/accounting/<RUN_ID>/...`
- `out/metrics/<RUN_ID>/...`
- `out/debt_resolution/<RUN_ID>/...`
- `out/human_reports/<RUN_ID>/...`

Rule:
- consumer apps and downstream automation should read published snapshot paths, not raw producer internals.

## What gets published today (high-value files)

Metrics subset published by `accounting.publish.latest` includes:
- `metrics/build_manifest.json`
- `metrics/metric_views/income_statement_monthly_last6.csv`
- `metrics/metric_views/rent_rollup_by_place_m_last6.csv`
- `metrics/metric_views/flow_type_rollup_m_last6.csv`
- `metrics/metric_views/draws_discipline_monthly_last6.csv`

Debt subset includes:
- `debt/debt_open_items.csv`
- `debt/debt_repayment_events.csv`
- `debt/debt_status_reconciliation.csv`

Report surface includes:
- `report/balance_humano_v2.html`
- `report/story_manifest.json`

## Source anchors

- `accounting.human.document` (human report outputs and story manifest)
- `accounting.publish.latest` (published file list and snapshot manifest)
- `notes/frontend_snapshot_contract.md` (consumer boundary)
- `notes/accounting_spine_runbook.md` (required metric view artifacts in run output)
