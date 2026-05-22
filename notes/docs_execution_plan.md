---
id: notes/docs_execution_plan
title: "Documentation Production Team + Execution Plan"
sidebar_label: "Documentation Production Team + Execution Plan"
---

# Documentation Production Team + Execution Plan

Status: planning draft
Last reviewed: 2026-05-21

## Objective

Create a practical documentation production plan that:
1. Builds a small "docs team" (human + agent roles).
2. Starts with documentation that can be anchored directly in the current codebase and Makefile.
3. Fixes factual drift by requiring every statement to map to source files or runnable commands.

This plan intentionally precedes code changes.

## Team ensemble (who produces what)

### 1) Documentation Lead (human owner)

Mission:
- Own final voice, priorities, and publication order.
- Resolve business-language choices (family/operator wording, audience tone).

Deliverables:
- Approval on doc scope for each sprint.
- Sign-off checklist per document.

### 2) Code-Truth Curator (agent)

Mission:
- Extract facts from code and command surfaces only.
- Maintain a "fact ledger" linking claims -> file/line or command output.

Deliverables:
- Evidence-backed doc fragments for commands, outputs, and contracts.
- Drift alerts when docs conflict with `Makefile` or module CLIs.

### 3) Operations Writer (human or agent)

Mission:
- Convert technical facts into runbooks and incident procedures.
- Optimize for "first 15 minutes" usability.

Deliverables:
- Step-by-step operator playbooks.
- Failure taxonomy and escalation paths.

### 4) Consumer Experience Writer (human)

Mission:
- Translate backend outputs into user journeys (operator, analyst, stakeholder).
- Clarify which artifact each audience should consume.

Deliverables:
- Report consumer guide.
- "What to look at first" summaries.

### 5) Automation Steward (human + agent pair)

Mission:
- Document periodic automation wiring and runtime assumptions.
- Ensure scheduled runs call canonical surfaces.

Deliverables:
- Automation wiring spec (systemd/cron template + env contract).
- Recovery and rollback guidance for scheduled workflows.

## Working model (fact-first docs)

For each document:
1. **Scope statement** (who it serves, what decisions it supports).
2. **Code-anchored fact extraction** from:
   - `Makefile`
   - canonical module entrypoints under `accounting/*`
   - authority docs (`notes/entrypoints.md`, `notes/accounting_spine_runbook.md`, `notes/output_contracts.md`)
3. **Executable validation** (run listed commands where possible).
4. **Publish with evidence map** (claim -> source path/command).

## First documentation wave (anchored on existing code)

Priority is ordered by operational impact and factual confidence.

### Wave 1 — `notes/environment_bootstrap.md` (new)

Why first:
- Current smoke failure risk is often environment-level (Python/dependencies).
- Prevents false alarms that look like pipeline regressions.

Anchor sources:
- `README.md` logging + run guidance
- `Makefile` variable defaults and command targets
- import requirements visible from stage entrypoints

Minimum sections:
- Supported Python version policy
- Virtualenv setup and install commands
- Required env vars (`ACCOUNT_SA`, `ACCOUNT_SHEET_URL`, etc.)
- Sanity commands: `make doctor`, `make help`, `make smoke`

### Wave 2 — `notes/automation_wiring_spec.md` (new)

Why second:
- Your vision requires periodic autonomous reporting.
- Miswired scheduler is a top reliability risk.

Anchor sources:
- `Makefile` canonical targets (`build-all`, `publish-latest`, layered targets)
- `README.md` journalctl operational pattern
- publish contract docs

Minimum sections:
- Canonical scheduled command(s)
- Working directory and env file contract
- systemd unit/timer template + cron fallback
- failure routing and first-response commands

### Wave 3 — `notes/report_consumer_guide.md` (new)

Why third:
- Helps humans/agents get value from outputs without reading internals.

Anchor sources:
- `accounting.human.document`
- `accounting.publish.latest`
- `notes/frontend_snapshot_contract.md`
- metric view file list in runbook/contracts

Minimum sections:
- Which artifact each audience should read
- Which directories are internal-only vs consumer-safe
- Common questions: latest balance, rent rollup, debt snapshots, anomalies

### Wave 4 — `notes/refactor_definition_of_done.md` (new)

Why fourth:
- Converts "large refactor anxiety" into a repeatable release gate.

Anchor sources:
- `Makefile` command graph
- `notes/accounting_spine_runbook.md` required outputs
- `notes/output_contracts.md` required artifacts

Minimum sections:
- Mandatory checks before merge
- Required artifact existence checks
- publish snapshot verification checklist

## Evidence map template (must be used in each new doc)

At the end of each documentation file, include:

- **Commands validated** (date + command + pass/fail)
- **Code anchors** (path references for each factual section)
- **Known assumptions** (items not yet machine-validated)

This reduces drift and makes agent updates safer.

## 10-day execution plan

### Day 1–2
- Build fact ledger from `Makefile`, README, runbook, and contracts.
- Draft `environment_bootstrap.md`.

### Day 3–4
- Draft `automation_wiring_spec.md` with canonical schedule paths.
- Review against real operator workflow.

### Day 5–6
- Draft `report_consumer_guide.md` using current publish and report surfaces.

### Day 7
- Draft `refactor_definition_of_done.md` with concrete pass/fail gates.

### Day 8
- Cross-link all new docs from `README.md` and `documentation_compass.md`.

### Day 9
- Run docs drift pass: verify command names and artifact filenames still match code.

### Day 10
- Publish v1 docs set and open a "docs debt" backlog for remaining gaps.

## Acceptance criteria for this planning phase

1. Team roles are clear and assigned.
2. First four documentation artifacts are prioritized and scoped.
3. Every artifact has explicit code anchors and validation commands.
4. Plan is executable without changing pipeline code.

## Recommendation

Proceed with Wave 1 immediately (`environment_bootstrap.md`) because it gives the fastest reliability gains and best supports both humans and agents during incident triage.
