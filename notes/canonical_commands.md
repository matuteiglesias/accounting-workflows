---
id: notes/canonical_commands
title: "Accounting Canonical Commands"
sidebar_label: "Accounting Canonical Commands"
---

# Accounting Canonical Commands

Status: authority draft
Last reviewed: 2026-05-10

## Purpose

This file is the command authority companion to `notes/entrypoints.md`. It maps the desired command surface to current Makefile targets.

Run `make help` for the live command list. The canonical debt module CLIs are `python -m accounting.debt.resolve` and `python -m accounting.debt.balance_views`; the old flat debt module compatibility paths have been removed.

## Core pipeline

```text
make ledger        # source inputs → canonical ledger
make materialize   # canonical ledger → Stage D analytical artifacts
make debt          # canonical ledger/views → resolved debt contracts
make debt-views    # debt contracts → debt balance views
make metrics       # analytical artifacts → metric contracts
make human-report  # metric/debt contracts → current human report
make publish-latest # latest artifacts → frontend snapshot
```

## Composite commands

```text
make build-all     # full canonical path through publish
make build-report  # full current report path, no publish
make build-front   # publish latest frontend snapshot
```

## Support commands

```text
make doctor
make smoke
make validate
make clean-derived
```

## Experimental command

```text
make front-report
```

`front-report` calls `accounting.human.front`. It remains experimental until the front report has a stable consumer and output contract.

## Legacy/compatibility command surface

The existing `run-*` targets remain available during transition. They are implementation/compatibility targets behind the clearer canonical aliases.

New docs and scripts should prefer the canonical names unless they specifically need lower-level `run-*` behavior.
