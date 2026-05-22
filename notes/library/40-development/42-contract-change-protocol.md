---
id: notes/library/40-development/42-contract-change-protocol
title: "42 contract change protocol"
sidebar_label: "42 contract change protocol"
sidebar_position: 42
---

# 42 contract change protocol

Status: draft (code-anchored)
Last reviewed: 2026-05-22

## Purpose

Ensure output contract changes are explicit, reviewed, and discoverable.

## When this protocol is required

Run this protocol whenever you change:
- artifact filenames or required columns,
- required stage outputs,
- publish bundle file list,
- canonical command surfaces.

## Required update set

1. Update producer code.
2. Update contract docs:
   - `notes/output_contracts.md`
   - `notes/accounting_spine_runbook.md`
   - `notes/frontend_snapshot_contract.md` (if publish surface changes)
3. Update user navigation docs:
   - relevant pages in `notes/library/*`
4. Run verification commands from Definition of Done.

## PR checklist

- [ ] Changed contracts are documented with old/new behavior.
- [ ] Required outputs are validated for at least one run.
- [ ] Publish snapshot paths still resolve.
- [ ] Consumer docs updated if user-facing paths changed.
- [ ] Compatibility note added for removed/deprecated paths.
