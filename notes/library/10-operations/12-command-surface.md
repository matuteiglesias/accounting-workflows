---
id: notes/library/10-operations/12-command-surface
title: "12 command surface"
sidebar_label: "12 command surface"
sidebar_position: 12
---

# 12 command surface

Status: current pointer
Last reviewed: 2026-08-25

The executable authority is `Makefile` / `make help`. The canonical operator contract is `notes/canonical_commands.md`.

## Supported spine

```bash
make run-canonical
make run-debt RUN_ID=<exact-run-id>
make run-metrics RUN_ID=<exact-run-id>
make run-reports RUN_ID=<exact-run-id>
make run-full
```

`run-canonical` and `run-full` are the live paths. Downstream stage targets replay an exact `RUN_ID` and do not pull live inputs.

There is no compatibility-alias layer. Historical bare noun targets, `build-all`, `run-accounting*`, `run-debt-views`, `run-dashboard`, `*-from-run`, and light/downstream shortcut targets are retired.

## Source anchors

- `Makefile`
- `notes/canonical_commands.md`
- `notes/entrypoints.md`
- `notes/pipeline_dag_contract.md`
- `notes/makefile_target_inventory.csv`
