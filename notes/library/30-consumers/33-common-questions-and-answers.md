---
id: notes/library/30-consumers/33-common-questions-and-answers
title: "33 common questions and answers"
sidebar_label: "33 common questions and answers"
sidebar_position: 33
---

# 33 common questions and answers

Status: draft (code-anchored)
Last reviewed: 2026-05-22

## Q1) Where do I see the latest balance report?

Open:
- `public/accounting/latest/report/balance_humano_v2.html`

If missing, verify publish step:
- `make publish-latest`

## Q2) Where do I check rent rollup quickly?

Open:
- `public/accounting/latest/metrics/metric_views/rent_rollup_by_place_m_last6.csv`

Upstream source contract is produced by metrics build in run outputs before publish.

## Q3) Where do I inspect debt snapshot tables?

Open:
- `public/accounting/latest/debt/debt_open_items.csv`
- `public/accounting/latest/debt/debt_repayment_events.csv`
- `public/accounting/latest/debt/debt_status_reconciliation.csv`

## Q4) How do I know what exactly was published?

Open:
- `public/accounting/latest/manifest.json`

It lists source paths and published files.

## Q5) Should consumers read out/* directly?

No for normal usage.

Use published snapshot only:
- `public/accounting/latest/*`

Use `out/*` only for debugging or pipeline development.

## Q6) What if a file is missing in latest snapshot?

First checks:

```bash
make help
make doctor
make build-all
```

Then inspect logs:

```bash
journalctl --user -u accounting-spine-live.service -n 200 --no-pager
```

Classify issue as scheduler, bootstrap/runtime, or pipeline regression.
