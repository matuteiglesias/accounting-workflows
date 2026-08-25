---
id: notes/documentation_compass
title: "Documentation Compass (Humans + Agents)"
sidebar_label: "Documentation Compass (Humans + Agents)"
---

# Documentation Compass (Humans + Agents)

Status: current guidance
Last reviewed: 2026-08-25

## Operator
Read `notes/accounting_spine_runbook.md`, `notes/canonical_commands.md`, `notes/pipeline_dag_contract.md`, then the relevant publication contract. Start with `make help`, `make doctor`, `make smoke-full`, and the smallest bounded stage needed. Only `run-canonical`, `run-full`, and the focused `run-ingest` operation pull live source inputs.

## Developer
Read `notes/current_state_map.md`, `notes/output_contracts.md`, `notes/entrypoints.md`, `notes/repository_tree_contract.md`, and `tests/TESTING.md`. Preserve accounting authority and validate affected downstream layers. The top-level `accounting/` package is the current import root; do not create a parallel `src/accounting` layout as incidental cleanup.

## Analyst / stakeholder
Use governed metrics/dashboard artifacts, finished report bundles, or the professional pack plus linked drilldowns for human-facing review. Removed notebook/human-report compatibility paths are not supported surfaces.

## Coding agent
Prefer canonical Make/module entrypoints. Do not recreate empty compatibility modules, command aliases, notebook runtime trees, or alternate reporting engines. Historical dated audits are useful evidence but not current command authority. A compatibility seam requires a concrete current caller and a stated removal condition.

## Current pipeline abstraction

```text
live source -> canonical/materialization -> debt/treasury -> metrics -> reports -> publication
                                                        \
                                                         -> professional evidence/drilldowns
```

Publication and professional presentation are downstream artifact consumers. Neither is allowed to redefine accounting semantics.
