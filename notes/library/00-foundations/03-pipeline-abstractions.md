---
id: notes/library/00-foundations/03-pipeline-abstractions
title: "03 pipeline abstractions"
sidebar_label: "03 pipeline abstractions"
sidebar_position: 3
---

# 03 pipeline abstractions

Status: current pointer (code-anchored)
Last reviewed: 2026-08-25

## Layer model

1. Source inputs
2. Canonical ledger
3. Governed materialization / semantic + cash facts
4. Debt position/activity + treasury
5. Governed frontier + annual metrics
6. Finished human reports
7. Published machine/document handoffs
8. Professional evidence and drilldowns

## Command abstraction

```text
run-canonical [live]
  -> run-debt [exact run]
  -> run-metrics [exact run]
  -> run-reports [exact run]
```

`run-full` is the ordered live composite through latest alignment, publication, and release checks. Focused downstream stages operate on `RUN_ID` and do not silently invoke live ingest.

## Consumer abstraction

- machine consumers: `public/accounting/latest_<SCOPE>/`;
- report consumers: `public/reports/latest_<SCOPE>/`;
- debugging/replay: exact-run artifacts under `out/*`.

## Governance abstraction

- every layer publishes named governed artifacts;
- downstream layers consume contract artifacts, not ad-hoc intermediate files;
- presentation cannot redefine accounting semantics;
- compatibility aliases do not substitute for stage boundaries;
- historical diagnostics/notebooks are evidence, not runtime layers.
