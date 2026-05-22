---
id: notes/library/30-consumers/30-consumer-start-here
title: "30 consumer start here"
sidebar_label: "30 consumer start here"
sidebar_position: 30
---

# 30 consumer start here

Status: draft
Last reviewed: 2026-05-22

## Audience

- family/business stakeholders reading balance outputs
- operators validating latest published reports
- coding agents answering "where is X metric/report?"

## Read order

1. `31-report-consumer-guide.md`
2. `32-where-to-find-latest-outputs.md`
3. `33-common-questions-and-answers.md`

## Consumer rule (critical)

Read from published snapshot first:
- `public/accounting/latest/*`

Avoid direct dependency on producer internals unless debugging:
- `out/run/...`
- `out/metrics/...`
- `out/debt_resolution/...`
- `out/human_reports/...`
