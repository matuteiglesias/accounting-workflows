---
id: notes/library/30-consumers/31-report-consumer-guide
title: "31 report consumer guide"
sidebar_label: "31 report consumer guide"
sidebar_position: 31
---

# 31 report consumer guide

Status: current (code-anchored)
Last reviewed: 2026-08-24

## Choose the surface by job

**Family/stakeholder review:** use the professional pack and `professional-linked-digest` when a human-facing pack is available. Its linked drilldowns must reconcile to the displayed cells.

**Programmatic consumer:** use the scope-qualified `public/accounting/latest_<SCOPE_TAG>/manifest.json` and its listed governed artifacts.

**Analyst/developer:** use canonical run, debt, and metrics roots directly for trace/reconciliation work; do not infer authority from presentation HTML.

## Retired surface

`public/accounting/.../report/balance_humano_v2.html`, `out/human_reports/*`, and `accounting.human.*` are no longer produced or supported. No standalone Flask/frontend application is part of this repository.

See `notes/public_bundle_contract.md` and `notes/accounting_spine_runbook.md`.
