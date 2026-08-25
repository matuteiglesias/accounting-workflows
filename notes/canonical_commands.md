---
id: notes/canonical_commands
title: "Accounting Canonical Commands"
sidebar_label: "Accounting Canonical Commands"
---

# Accounting Canonical Commands

Status: current authority
Last reviewed: 2026-08-25

The Makefile is the executable command authority. `make help` is the human-readable live surface. Compatibility aliases are intentionally absent.

## Command model

There are two kinds of operations:

1. **Live source composites** explicitly allowed to pull current source inputs.
2. **Exact-run stages** that operate only on the selected `RUN_ID` and must not silently re-ingest.

### Live source composites

```text
make run-canonical   # live ingest -> governed materialization
make run-full        # ordered live pipeline through reports/publication/release check
make run-env         # load ENV_FILE, then run-full
```

`run-canonical` is the only canonical live source stage. `run-full` wires the downstream stages in order using the same generated run identity.

### Exact-run stages

```text
make run-materialize RUN_ID=<exact-run-id>
make run-debt        RUN_ID=<exact-run-id>
make run-metrics     RUN_ID=<exact-run-id>
make run-reports     RUN_ID=<exact-run-id>
```

Stage ownership is explicit:

- `run-materialize`: canonical ledger -> semantic/cash governed artifacts;
- `run-debt`: debt resolution -> balance views -> monthly debt position/activity -> treasury accountability;
- `run-metrics`: governed frontier + annual dashboard artifacts;
- `run-reports`: annual-management + treasury-accountability HTML/PDF bundle.

The stage targets do not trigger upstream live work. This replaces historical `*-from-run`, `run-debt-views`, `run-dashboard`, and light/downstream shortcut families.

## Fixture / validation

```text
make smoke-core
make smoke-full
make validate
```

## Publication

```text
make publish-latest
make publish-reports
make release-check
```

Publication is separate from stage replay. `run-full` performs atomic latest alignment before invoking publication.

## Professional presentation

```text
make professional-drilldowns
make professional-linked-digest
```

These are downstream presentation/reconciliation operations over an existing professional pack. They are not an alternate semantic pipeline.

## Retired command families

The following names are deliberately unsupported and should not be reintroduced as aliases: bare noun aliases (`ledger`, `materialize`, `debt`, `debt-views`, `metrics`, `publish`), `build-all`, `run-accounting*`, `run-debt-views`, `run-debt-balance`, `run-dashboard`, `metrics-from-run`, `reports-from-run`, `run-live-light`, `run-downstream-from-ledger`, `run`, `run-all`, `smoke`, and `smoke-accounting`.

A new alias requires a concrete external consumer that cannot migrate immediately plus an explicit removal condition. Internal convenience alone is not sufficient.
