---
id: notes/canonical_commands
title: "Accounting Canonical Commands"
sidebar_label: "Accounting Canonical Commands"
---

# Accounting Canonical Commands

Status: current authority
Last reviewed: 2026-08-24

The Makefile is the command authority. `make help` is the live command list.

## Core pipeline

```text
make ledger          # source inputs -> canonical ledger
make materialize     # canonical ledger -> materialized analytical artifacts
make debt            # canonical evidence -> resolved debt contracts
make debt-views      # debt contracts -> stock/activity views
make metrics         # semantic/debt artifacts -> governed metrics
make run-dashboard   # assert governed annual dashboard outputs
make publish-latest  # governed metrics/debt -> published artifact bundle
```

## Composite command

```text
make build-all       # full canonical path through publication + release check
```

## Professional presentation

```text
make professional-drilldowns
make professional-linked-digest
```

These are downstream presentation/reconciliation operations over an existing professional pack. They are not an alternate semantic pipeline.

## Removed Phase-1 surfaces

The `human-report`, `run-human*`, `front-report`, `build-report`, and `build-front` command families were removed on 2026-08-24. Their former producer package (`accounting.human`) was an alternate presentation stack with no production Python caller outside its own package. The old front factory was static HTML scaffolding; the repository contains no Flask runtime.

`accounting.publish.latest` remains supported, but it publishes a metrics/debt artifact bundle rather than requiring `human_reports` or a viewer application.
