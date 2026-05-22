---
id: notes/human_agent_playbook
title: "Human + Agent Operations Playbook (Stability-First)"
sidebar_label: "Human + Agent Operations Playbook (Stability-First)"
---

# Human + Agent Operations Playbook (Stability-First)

Status: draft
Last reviewed: 2026-05-21

## Why this doc exists

This repository already has strong technical runbooks, but they are split across multiple files and optimized for someone who already knows the system. During refactors and incident recovery, humans and coding agents need a single "first 15 minutes" guide that answers:

1. What command path is canonical right now?
2. How do we detect if automation is miswired vs dependencies missing?
3. What files prove each stage is healthy?
4. Who should do what (human operator vs agent) to reduce recovery time?

## Current evidence from this repo

- Canonical command surface is Makefile-first (`make build-all`, layered targets). See `notes/entrypoints.md`.
- Official stage contracts and required outputs are defined. See `notes/accounting_spine_runbook.md` and `notes/output_contracts.md`.
- There are both canonical and compatibility aliases, which is useful but increases confusion during incidents if not explicitly pinned per environment. See `notes/entrypoints.md`.
- Smoke path currently fails in this environment because Python dependencies are not installed (`ModuleNotFoundError: pandas`). This is not a data-pipeline logic failure; it is a runtime environment/dependency failure.

## Stability-first triage sequence

Use this exact order after a refactor.

### 0) Fast command-surface sanity (no data needed)

```bash
make help
make doctor
```

Goal: verify command graph and module imports compile.

### 1) Smoke ingest path (fixture/offline)

```bash
make smoke
```

Interpretation:

- If it fails with missing modules (for example pandas), classify as **environment/bootstrap issue**.
- If it fails with invariant errors or missing contract outputs, classify as **pipeline regression**.

### 2) Contract-driven artifact checks (for run mode)

For the active `RUN_ID`, verify the canonical outputs defined in the runbook:

- `out/run/accounting/<RUN_ID>/ledger_canonical.csv`
- `out/run/accounting/<RUN_ID>/views/views_sanity.json`
- `out/metrics/<RUN_ID>/metric_values.csv`
- `out/human_reports/<RUN_ID>/balance_human_v2/balance_humano_v2.html`

### 3) Automation health checks

If live automation is expected via systemd:

```bash
journalctl --user -u accounting-spine-live.service -n 200 --no-pager
journalctl --user -u accounting-spine-live.service --since "1 hour ago"
```

Classify failures into one of three buckets:

- **Scheduler miswire** (service/timer not invoking canonical targets)
- **Runtime env mismatch** (missing deps, wrong Python, missing env vars)
- **Pipeline logic regression** (stage crashes, contract violations)

## Human/agent split of responsibilities

### Human operator (owner)

- Decides source of truth (fixture vs live sheet) for each recovery attempt.
- Owns secrets/env (`ACCOUNT_SA`, sheet URLs, production-only paths).
- Confirms acceptable fallback behavior for publish/report stages.

### Coding agent

- Runs reproducible local checks (`make help`, `make doctor`, `make smoke`).
- Maps failures to one of the three buckets above.
- Proposes smallest repair set and updates docs/runbooks when command surfaces or contracts change.
- Avoids introducing new aliases unless necessary; prefers canonical names.

## Minimal docs that would materially reduce incident time

If you want this "beast" to be safer for both humans and agents, these additions are high ROI:

1. **Single-page Incident Runbook** (this file): first-response flow, failure taxonomy, escalation rules.
2. **Environment Bootstrap Contract**: pinned Python + dependency installation command(s) + required env vars with validation command.
3. **Automation Wiring Spec**: exact systemd unit/timer command, working directory, and env file path (copy-paste authoritative).
4. **Definition of Done for Refactors**: checklist requiring smoke + run-contract outputs + publish snapshot verification.

## Recommendation

Yes: this project would clearly benefit from documentation explicitly guiding both humans and agents.

You already have most of the deep technical pieces; the gap is orchestration and recovery ergonomics. A concise operations playbook plus explicit environment/automation wiring docs will reduce time-to-diagnosis and prevent command alias confusion during stressful recovery windows.
