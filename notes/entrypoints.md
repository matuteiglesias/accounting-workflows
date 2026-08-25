---
id: notes/entrypoints
title: "Accounting Backend Entrypoints"
sidebar_label: "Accounting Backend Entrypoints"
---

# Accounting Backend Entrypoints

Status: current authority
Last reviewed: 2026-08-25

The Makefile is the command authority. Module CLIs are implementation entrypoints; start with `make help`.

## Canonical Make targets

| Target | Input behavior | Responsibility |
|---|---|---|
| `make run-ingest` | live source | Focused canonical-ledger ingest for one `RUN_ID`. |
| `make run-materialize RUN_ID=...` | exact run | Existing canonical ledger -> governed semantic/cash artifacts. |
| `make run-canonical` | live source | Ordered `run-ingest` + `run-materialize`. |
| `make run-debt RUN_ID=...` | exact run | Debt resolution, balances, monthly position/activity, treasury accountability. |
| `make run-metrics RUN_ID=...` | exact run | Governed frontier and annual-dashboard artifacts. |
| `make run-reports RUN_ID=...` | exact run | Finished annual-management and treasury-accountability documents. |
| `make run-full` | live composite | Ordered canonical -> debt -> metrics -> reports -> latest -> publication -> release check. |
| `make publish-latest` | latest pointers | Publish the governed machine-artifact handoff. |
| `make publish-reports` | latest report pointer | Publish finished HTML/PDF reports only. |
| `make release-check` | published bundle | Verify release-readiness contract. |

`make run-env` loads `ENV_FILE` and delegates to `run-full`. It is an environment wrapper, not a second pipeline.

There are no compatibility aliases for these targets. In particular, bare noun aliases, `build-all`, `run-accounting*`, `run-debt-views`, `run-dashboard`, and `*-from-run` names are retired.

## Professional presentation targets

| Target | Responsibility |
|---|---|
| `make professional-drilldowns` | Build/reconcile drilldowns for an existing professional pack. |
| `make professional-linked-digest` | Render the professional pack plus drilldown links; presentation only. |

These targets do not recompute accounting semantics and are not part of fixture CI because a real professional pack is local/external evidence.

## Canonical module CLIs

- `python -m accounting.ledger.ingest`
- `python -m accounting.stage_d.materialize`
- `python -m accounting.debt.resolve`
- `python -m accounting.debt.balance_views`
- `python -m accounting.marts.debt`
- `python -m accounting.marts.treasury`
- `python -m accounting.metrics.build`
- `python -m accounting.reports.build`
- `python -m accounting.publish.latest`
- `python -m accounting.reports.publish`
- `python -m accounting.professional.drilldown`
- `python -m accounting.professional.render_linked_digest`

`accounting.marts.build` is retired; there is no generic views builder between materialization and the governed debt/metrics stages.

## Retired entrypoints

`accounting.human.*`, `accounting.viz.*`, `accounting.notebooks/*`, `run-human*`, `human-report`, `front-report`, `build-report`, `build-front`, and the historical Make compatibility aliases are removed. Historical documents may still mention them when describing prior architecture, but current docs and automation must not invoke them.
