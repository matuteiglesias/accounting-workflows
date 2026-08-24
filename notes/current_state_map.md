---
id: notes/current_state_map
title: "Accounting Backend Current State Map"
sidebar_label: "Accounting Backend Current State Map"
---

# Accounting Backend Current State Map

Status: current authority
Last reviewed: 2026-08-24

## Artifact ladder

```text
source inputs
  -> canonical ledger
  -> materialization + semantic marts
  -> debt stock/activity contracts
  -> governed metrics + annual dashboard
  -> published accounting bundle
  -> professional pack / drilldowns / linked digest (presentation)
```

## Ownership

- `accounting.ledger` owns canonical ingest evidence.
- `accounting.stage_d` / `accounting.marts` own materialized and semantic tables.
- `accounting.debt.resolve` and `accounting.debt.balance_views` own debt resolution/balance evidence; empty re-export `models`/`rules` seams are gone.
- `accounting.metrics` owns governed metric and annual-dashboard contracts.
- `accounting.professional` owns professional table/drilldown/presentation machinery; it must consume governed values rather than become a parallel accounting engine.
- `accounting.publish.latest` owns scope-safe packaging of metrics/debt into `public/accounting/latest_<SCOPE_TAG>`.

## Phase-1 removals

`accounting.human`, `accounting.viz`, `accounting.config`, `accounting.contracts.models`, `accounting.debt.models`, `accounting.debt.rules`, and `accounting.publish.snapshot` were removed after exact reachability census showed no supported production caller. The old front factory was static HTML scaffolding; no Flask import/runtime was present.

The former `human` capabilities were either pass-through projections of governed metric views or presentation duplication. Reusable current presentation belongs to professional table contracts, drilldowns, linked digest, and notebook/report consumers. No accounting formula was migrated into presentation code.
