---
id: notes/library/40-development/41-refactor-definition-of-done
title: "41 refactor definition of done"
sidebar_label: "41 refactor definition of done"
sidebar_position: 41
---

# 41 refactor definition of done

Status: draft (code-anchored)
Last reviewed: 2026-05-22

## Purpose

Convert refactor risk into a repeatable release gate that humans and agents can run before merge.

## Anchor sources

- `Makefile` command graph and canonical/supported targets
- `notes/accounting_spine_runbook.md` required stage outputs
- `notes/output_contracts.md` artifact contracts and expected producers

## Mandatory checks before merge

Run from repo root.

```bash
python -m compileall accounting
make help
make validate
make doctor
git diff --check
```

If environment has dependencies installed, run pipeline confidence checks:

```bash
make smoke
make build-report
make publish-latest
```

Gate rule:
- Merge only when mandatory checks pass, or when failures are explicitly classified as environment limitations with documented remediation.

## Required artifact existence checks

For active `<RUN_ID>`, verify required pipeline outputs:

```text
out/run/accounting/<RUN_ID>/ledger_canonical.csv
out/run/accounting/<RUN_ID>/views/views_sanity.json
out/metrics/<RUN_ID>/metric_registry.csv
out/metrics/<RUN_ID>/metric_values.csv
out/metrics/<RUN_ID>/validation_report.csv
out/metrics/<RUN_ID>/metric_views/income_statement_monthly_last6.csv
out/metrics/<RUN_ID>/metric_views/rent_rollup_by_place_m_last6.csv
out/metrics/<RUN_ID>/metric_views/rent_rollup_by_detail_m_last6.csv
out/metrics/<RUN_ID>/metric_views/flow_type_rollup_m_last6.csv
out/metrics/<RUN_ID>/metric_views/draws_discipline_monthly_last6.csv
out/metrics/<RUN_ID>/metric_views/metric_views_manifest.csv
out/human_reports/<RUN_ID>/balance_human_v2/balance_humano_v2.html
out/human_reports/<RUN_ID>/balance_human_v2/story_manifest.json
```

## Publish snapshot verification checklist

After `make publish-latest`, verify:

```text
public/accounting/latest/manifest.json
public/accounting/latest/report/balance_humano_v2.html
public/accounting/latest/report/story_manifest.json
public/accounting/latest/metrics/build_manifest.json
public/accounting/latest/metrics/metric_views/income_statement_monthly_last6.csv
public/accounting/latest/metrics/metric_views/rent_rollup_by_place_m_last6.csv
public/accounting/latest/metrics/metric_views/flow_type_rollup_m_last6.csv
public/accounting/latest/metrics/metric_views/draws_discipline_monthly_last6.csv
public/accounting/latest/debt/debt_open_items.csv
public/accounting/latest/debt/debt_repayment_events.csv
public/accounting/latest/debt/debt_status_reconciliation.csv
```

Verification intent:
- producer outputs exist and are consumable,
- publish bundle is present for downstream viewer/consumers,
- no accidental reliance on internal-only `out/*` paths.

## Failure classification for release decision

- **Scheduler/wiring issue**: automation invocation not calling canonical targets.
- **Bootstrap/runtime issue**: missing dependencies/env vars/python mismatch.
- **Pipeline regression**: stage crashes, invariant errors, contract outputs missing.

Release decision:
- Ship only if no unresolved pipeline regression remains.
- Bootstrap/scheduler issues may be acceptable for code merge only when explicitly out-of-scope and tracked.
