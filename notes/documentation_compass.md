---
id: notes/documentation_compass
title: "Documentation Compass (Humans + Agents)"
sidebar_label: "Documentation Compass (Humans + Agents)"
---

# Documentation Compass (Humans + Agents)

Status: current guidance
Last reviewed: 2026-05-21

## Why this exists

The repository has many strong documents, but they are distributed and optimized for different audiences. This compass is the entrypoint that tells each user where to go first, depending on the task.

## Start here by role

### 1) Operator ("I need the system to run reliably")

Read in this order:
1. `notes/human_agent_playbook.md` (incident-first triage and failure taxonomy)
2. `notes/accounting_spine_runbook.md` (official stage order + required outputs)
3. `notes/canonical_commands.md` (current command surface)
4. `notes/frontend_snapshot_contract.md` (what publish must deliver)

Primary commands:
- `make help`
- `make doctor`
- `make build-report`
- `make publish-latest`

### 2) Developer ("I need to change code safely")

Read in this order:
1. `notes/current_state_map.md` (architecture and boundaries)
2. `notes/module_inventory.md` (module ownership)
3. `notes/output_contracts.md` (artifact-level contracts)
4. `notes/entrypoints.md` (canonical vs support vs legacy surfaces)

Primary goal: preserve artifact contracts and avoid accidental coupling across layers.

### 3) Analyst / stakeholder ("I need useful outputs")

Read in this order:
1. `notes/narrative.md` (business stories and reporting intent)
2. `notes/contracts.md` (domain rules/contracts)
3. `notes/frontend_snapshot_contract.md` (stable consumer surface)

Primary outputs:
- human report HTML under `out/human_reports/<RUN_ID>/...`
- published snapshot under `public/accounting/latest/*`

### 4) Coding agent ("I need to diagnose and act fast")

Read in this order:
1. `notes/human_agent_playbook.md`
2. `notes/entrypoints.md`
3. `notes/accounting_spine_runbook.md`
4. `notes/output_contracts.md`

Agent rule of thumb:
- Prefer canonical `make` targets over legacy aliases unless compatibility behavior is explicitly required.

## High-level abstractions (what users should think in)

The system should be operated using these abstractions, not low-level scripts:

1. **Pipeline layers**: ledger -> materialize -> views/debt -> metrics -> human report -> publish
2. **Contracts**: each layer must emit required artifacts (CSV/JSON/HTML) with known names
3. **Surfaces**:
   - producer surfaces in `out/...`
   - consumer-safe surface in `public/accounting/latest/*`
4. **Modes**:
   - smoke mode for fixture/offline confidence
   - run mode for live/real inputs

## Periodic automation vision (documented expectations)

For autonomous periodic reporting to be dependable, docs must explicitly define:

1. **Scheduler wiring**: exact systemd unit/timer or cron command and working directory.
2. **Environment contract**: Python version, dependency install command, required env vars.
3. **Run cadence and SLOs**: how often to run, acceptable latency/failure windows.
4. **Failure routing**: where failures appear (journal/log path) and who is paged.
5. **Recovery sequence**: exact commands for first response (`make doctor`, `make smoke`, targeted rebuild).

This repo already defines much of (4)-(5); (1)-(3) need a single authoritative document.

## Next documentation upgrades (highest ROI)

1. Add `notes/automation_wiring_spec.md` with copy-paste unit/timer/cron examples and required env.
2. Add `notes/environment_bootstrap.md` with deterministic setup (venv, install, sanity command).
3. Add `notes/report_consumer_guide.md` explaining which outputs each audience should use and ignore.
4. Add a "docs freshness" check in CI (or local validation target) to catch stale command names.

## Note on legacy docs

`notes/runbook.md` contains historical material that overlaps with newer authority docs.
Prefer `notes/accounting_spine_runbook.md`, `notes/output_contracts.md`, and this compass for current operations decisions.

## Documentation library (numbered)

The hierarchical library lives under `notes/library/` with numbered sections (`00`-`99`) for predictable navigation and future growth. Start at `notes/library/00-foundations/00-index.md`.
