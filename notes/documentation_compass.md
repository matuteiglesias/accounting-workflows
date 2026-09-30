---
id: notes/documentation_compass
title: "Documentation Compass (Humans + Agents)"
sidebar_label: "Documentation Compass (Humans + Agents)"
---

# Documentation Compass (Humans + Agents)

Status: current guidance
Last reviewed: 2026-09-17

## Operator
Read `notes/accounting_spine_runbook.md`, `notes/canonical_commands.md`, `notes/pipeline_dag_contract.md`, then the relevant publication contract. Start with `make help`, `make doctor`, `make smoke-full`, and the smallest bounded stage needed. Only `run-canonical`, `run-full`, and the focused `run-ingest` operation pull live source inputs.

The docs-input bridge is an exact-run downstream projection, not a live operation. Build it only when an internal documentation/professional consumer needs factual inputs:

```bash
python -m accounting.docs_input.build \
  --run-root out/run/accounting/<exact-run-id> \
  --out-dir out/docs_input/<exact-run-id>
```

Optional professional-pack and prospective-commitment inputs must be supplied explicitly. The bridge does not publish or move latest pointers.

## Developer
Read `notes/current_state_map.md`, `notes/output_contracts.md`, `notes/entrypoints.md`, `notes/repository_tree_contract.md`, and `tests/TESTING.md`. Preserve accounting authority and validate affected downstream layers. The top-level `accounting/` package is the current import root; do not create a parallel `src/accounting` layout as incidental cleanup.

For work on downstream factual documentation inputs, also read `notes/docs_input_bridge_program_20260917.md`. It records the allowed source authorities, explicit-ID join rule, prospective-reserve boundary, stop conditions, and required zero-delta reconciliation against existing accounting outputs.

## Analyst / stakeholder
Use governed metrics/dashboard artifacts, finished report bundles, or the professional pack plus linked drilldowns for human-facing review. Internal documentation systems may consume `acct.docs-input@1` when they need evidence coverage, actor/property/cost-support facts, unresolved allocation detail, or a prospective required-reserve schedule. Docs-input facts do not establish legal liability, ownership, entitlement, fairness, or a distributable balance.

Removed notebook/human-report compatibility paths are not supported surfaces.

## Coding agent
Prefer canonical Make/module entrypoints. Do not recreate empty compatibility modules, command aliases, notebook runtime trees, or alternate reporting engines. Historical dated audits are useful evidence but not current command authority. A compatibility seam requires a concrete current caller and a stated removal condition.

When touching docs-input, preserve the firewall:

```text
existing governed accounting/evidence authority
  -> bounded factual projection
  -> downstream legal/strategy/governance/psychology interpretation
```

Never reverse that arrow by feeding downstream interpretations back into accounting classification.

## Current pipeline abstraction

```text
live source -> canonical/materialization -> debt/treasury -> metrics -> reports -> publication
                                                        \
                                                         -> professional evidence/drilldowns -> docs-input
```

Publication, professional presentation, and docs-input are downstream artifact consumers. None is allowed to redefine accounting semantics.
