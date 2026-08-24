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
- `accounting.stage_d` owns Stage-D orchestration and mechanical builders but delegates generic CSV/hash/partition/manifest infrastructure to `accounting.support` / `accounting.artifacts`; semantic/cash sequencing remains unchanged.
- `accounting.debt.resolve` and `accounting.debt.balance_views` own debt resolution/balance evidence; empty re-export `models`/`rules` seams are gone.
- `accounting.metrics` owns governed metric and annual-dashboard contracts.
- `accounting.professional` owns professional table producers, contracts/adapters, drilldown execution, and professional-pack rendering. It consumes governed values and must not become either a parallel accounting engine or a home for forensic characterization utilities.
- `accounting.diagnostics` owns read-only forensic audits, issue digests, migration characterization, and other diagnostics over supplied artifacts. Diagnostics may inspect professional outputs but do not own the displayed values or accounting semantics they audit.
- `accounting.publish.latest` owns scope-safe packaging of metrics/debt into `public/accounting/latest_<SCOPE_TAG>`.

## Phase-1 removals

`accounting.human`, `accounting.viz`, `accounting.config`, `accounting.contracts.models`, `accounting.debt.models`, `accounting.debt.rules`, and `accounting.publish.snapshot` were removed after exact reachability census showed no supported production caller. The old front factory was static HTML scaffolding; no Flask import/runtime was present.

The former `human` capabilities were either pass-through projections of governed metric views or presentation duplication. Reusable current presentation belongs to professional table contracts, drilldowns, linked digest, and notebook/report consumers. No accounting formula was migrated into presentation code.

## Phase-2 ownership cleanup

`funding_lineage_audit.py` and `issue_digest.py` were moved out of `accounting.professional` into `accounting.diagnostics` without changing their diagnostic algorithms. The professional package is therefore limited more clearly to production/reporting responsibilities, while diagnostic tooling remains available under explicit diagnostic module paths.


## Phase 4 facade ownership (2026-08-24)

- Modern migration facades expose only an explicit repository-caller compatibility surface; broad `dir(delegate) -> globals()` re-exports are forbidden.
- `accounting.professional.drilldown_legacy` remains a compatibility implementation, not current semantic authority. Its remaining route families and removal blockers are tracked in `notes/accounting_simplification_phase4_drilldown_deletion_map_20260824.csv`.
- New governed consumers must import modern contracts/executors directly rather than creating new dependencies on `*_legacy` symbols.
