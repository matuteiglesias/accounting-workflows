---
id: notes/library/30-consumers/32-where-to-find-latest-outputs
title: "32 where to find latest outputs"
sidebar_label: "32 where to find latest outputs"
sidebar_position: 32
---

# 32 where to find latest outputs

Status: draft (code-anchored)
Last reviewed: 2026-05-22

## Quick path map

### Consumer-safe latest snapshot

```text
public/accounting/latest/
  manifest.json
  report/balance_humano_v2.html
  report/story_manifest.json
  metrics/*
  debt/*
```

Use this first for dashboards, viewers, and shared links.

### Producer run outputs (debug / traceability)

```text
out/run/accounting/<RUN_ID>/
out/metrics/<RUN_ID>/
out/debt_resolution/<RUN_ID>/
out/human_reports/<RUN_ID>/balance_human_v2/
```

Use only when diagnosing failed builds or validating upstream contracts.

## Typical lookup tasks

- Latest human-readable report: `public/accounting/latest/report/balance_humano_v2.html`
- Latest snapshot manifest: `public/accounting/latest/manifest.json`
- Latest rent rollup view: `public/accounting/latest/metrics/metric_views/rent_rollup_by_place_m_last6.csv`
- Latest debt open items: `public/accounting/latest/debt/debt_open_items.csv`

## Why two layers exist

- Producer outputs are rich and volatile across stages.
- Published snapshot is intentionally curated and stable for consumers.
