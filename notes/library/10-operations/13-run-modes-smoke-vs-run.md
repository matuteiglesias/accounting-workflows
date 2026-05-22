---
id: notes/library/10-operations/13-run-modes-smoke-vs-run
title: "13 run modes smoke vs run"
sidebar_label: "13 run modes smoke vs run"
sidebar_position: 13
---

# 13 run modes smoke vs run

Status: draft
Last reviewed: 2026-05-22

## Smoke mode

Purpose:
- Fixture/offline confidence without live sheet dependencies.

Entry command:
- `make smoke`

Key characteristics:
- Uses fixture CSV (`FIXTURE` variable defaulting to `fixtures/ledger_fixture.csv`).
- Writes under `out/smoke/accounting`.

## Run mode

Purpose:
- Live bounded run using configured sheet/env vars.

Entry command:
- `make build-report` (or layer-by-layer canonical targets)

Key characteristics:
- Uses `ACCOUNT_SHEET_URL`, `ACCOUNT_SA`, `ACCOUNT_SHEET_NAME` for live ingest.
- Writes timestamped artifacts under `out/run/accounting/<RUN_ID>`, `out/metrics/<RUN_ID>`, and `out/human_reports/<RUN_ID>/...`.

## Why this distinction matters

- Smoke failures often isolate environment/setup issues quickly.
- Run failures can indicate data-source issues, env wiring, or pipeline regressions.
