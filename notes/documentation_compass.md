---
id: notes/documentation_compass
title: "Documentation Compass (Humans + Agents)"
sidebar_label: "Documentation Compass (Humans + Agents)"
---

# Documentation Compass (Humans + Agents)

Status: current guidance
Last reviewed: 2026-08-24

## Operator
Read `notes/accounting_spine_runbook.md`, `notes/canonical_commands.md`, then `notes/public_bundle_contract.md`. Start with `make help`, `make doctor`, `make smoke-full`, and the smallest bounded live target needed.

## Developer
Read `notes/current_state_map.md`, `notes/output_contracts.md`, `notes/entrypoints.md`, `tests/TESTING.md`, and the current Phase-0/Phase-1 simplification evidence notes. Preserve accounting authority and validate affected downstream layers.

## Analyst / stakeholder
Use governed metrics/dashboard artifacts, or the professional pack plus linked drilldowns for human-facing review. The removed `out/human_reports` / `balance_humano_v2` path is not a supported surface.

## Coding agent
Prefer canonical Make/module entrypoints. Do not recreate empty compatibility modules or alternate reporting engines. Historical dated audits are useful evidence but not current command authority.

## Current pipeline abstraction

```text
ledger -> materialization/semantic marts -> debt -> metrics/dashboard -> publication
                                           \
                                            -> professional pack -> drilldowns/linked digest
```

Publication is an artifact handoff; professional presentation is downstream. Neither is allowed to redefine accounting semantics.
