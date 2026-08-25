---
id: notes/automation_wiring_spec
title: "Automation Wiring Spec"
sidebar_label: "Automation Wiring Spec"
---

# Automation Wiring Spec

Status: current repository wiring contract
Last reviewed: 2026-08-25
Audience: operators, automation stewards, coding agents

## Scope

This page defines the command contract a scheduler may invoke. It does **not** assert that cron, systemd, or any other scheduler is currently deployed on a host; deployment state must be established from host evidence separately.

## Canonical automation command

For a complete live run using an environment file:

```bash
make run-env
```

`run-env` loads `ENV_FILE` (default `private/accounting.env`) and delegates exactly once to `run-full`. `run-full` already performs canonical ingest/materialization, debt/treasury, metrics, reports, latest alignment, machine/report publication, and `release-check`. Automation must not append a second `publish-latest` or recreate the historical `build-all` / `run-accounting` aliases.

For an interactive live run where the environment is already loaded:

```bash
make run-full
```

## Partial replay contract

Downstream recovery should use an existing exact run identity and the smallest required stage:

```bash
make run-materialize RUN_ID=<exact-run-id>
make run-debt        RUN_ID=<exact-run-id>
make run-metrics     RUN_ID=<exact-run-id>
make run-reports     RUN_ID=<exact-run-id>
```

These targets do not pull live inputs or move latest pointers. After a repaired run has all required stage products, publication/latest movement is a separate deliberate operation; do not silently promote a partial replay.

## Working directory and environment

Scheduler jobs must execute from the repository root. The default environment file is:

```text
private/accounting.env
```

Live ingest requires `ACCOUNT_SHEET_URL`; `ACCOUNT_SA` is passed to the ingest entrypoint and `ACCOUNT_SHEET_NAME` defaults to `C. Long Ledger` when omitted. Operational overrides such as `OUT`, `FREQ`, `BOXES`, and `REPORT_BROWSER_BIN` remain explicit Make variables.

## Concurrency

Overlapping same-scope live runs are unsupported until concurrency issue #44 is resolved. A scheduler must therefore serialize same-scope invocations rather than assuming `run-full` is safe under overlap. This repository does not claim a particular host-level locking implementation.

## Failure routing

Start with fixture/static evidence before touching live state:

```bash
make help
make doctor
make smoke-full
```

Then identify the failed exact `RUN_ID` and replay only the first incomplete downstream stage. Preserve the run identity across `out/run/accounting`, `out/debt_resolution`, `out/metrics`, and `out/reports`.

Typical failure classes:

- missing module or environment variable: bootstrap/wiring issue;
- stage crash or contract output missing: pipeline/data-shape issue;
- browser/PDF failure: report-rendering environment issue;
- scheduler did not fire: host scheduler issue, outside repository evidence unless inspected directly;
- overlapping same-scope invocations: unsupported concurrency, not a retry-safe condition.

## Publication boundary

Automated live execution may publish only through the canonical `run-full` sequence. Consumer applications read the publication handoffs, not producer internals:

```text
public/accounting/latest_<SCOPE>/
public/reports/latest_<SCOPE>/
```

They must not depend on `out/run`, `out/debt_resolution`, `out/metrics`, or `out/reports` as runtime APIs.

## Verification anchors

- `Makefile` — executable command graph;
- `notes/canonical_commands.md` — operator command contract;
- `notes/pipeline_dag_contract.md` — stage ordering and exact-run semantics;
- `notes/report_bundle_contract.md` and `notes/public_bundle_contract.md` — publication boundaries;
- issue #44 — same-scope concurrency limitation.
