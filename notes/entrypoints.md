---
id: notes/entrypoints
title: "Accounting Backend Entrypoints"
sidebar_label: "Accounting Backend Entrypoints"
---

# Accounting Backend Entrypoints

Status: current authority
Last reviewed: 2026-08-24

The Makefile is the command authority. Module CLIs are implementation entrypoints; start with `make help`.

## Canonical Make targets

| Target | Responsibility |
|---|---|
| `make ledger` | Canonical ledger ingest. |
| `make materialize` | Materialized Stage-D/semantic artifacts. |
| `make debt` | Internal-debt resolution. |
| `make debt-views` | Debt balance/activity views. |
| `make metrics` | Metric values, registry, validation, views, drilldowns, annual dashboard artifacts. |
| `make run-dashboard` | Assert governed annual dashboard contract outputs. |
| `make publish-latest` | Package scope-qualified governed metrics/debt for downstream consumers. |
| `make build-all` / `make run-full` | Full canonical path through publication and release check. |

`make publish` is a compatibility alias for `publish-latest`. `run-accounting` and `run-accounting-full` remain compatibility aliases for `run-full`.

## Professional presentation targets

| Target | Responsibility |
|---|---|
| `make professional-drilldowns` | Build/reconcile drilldowns for an existing professional pack. |
| `make professional-linked-digest` | Render the professional pack plus drilldown links; presentation only. |

These targets do not recompute accounting semantics and are not part of fixture CI because a real professional pack is local/external evidence.

## Canonical module CLIs

- `python -m accounting.ledger.ingest`
- `python -m accounting.stage_d.materialize`
- `python -m accounting.marts.build`
- `python -m accounting.debt.resolve`
- `python -m accounting.debt.balance_views`
- `python -m accounting.metrics.build`
- `python -m accounting.publish.latest`
- `python -m accounting.professional.drilldown`
- `python -m accounting.professional.render_linked_digest`

## Retired Phase-1 entrypoints

`accounting.human.*`, `accounting.viz.*`, `run-human*`, `human-report`, `front-report`, `build-report`, and `build-front` are removed. No supported code should recreate them as compatibility aliases. Historical documents may still mention them when describing prior architecture.
