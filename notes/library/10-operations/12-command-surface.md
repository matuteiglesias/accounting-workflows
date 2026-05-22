---
id: notes/library/10-operations/12-command-surface
title: "12 command surface"
sidebar_label: "12 command surface"
sidebar_position: 12
---

# 12 command surface

Status: draft
Last reviewed: 2026-05-22

## Canonical commands (preferred)

```bash
make ledger
make materialize
make debt
make debt-views
make metrics
make human-report
make publish-latest
```

Composite shortcuts:

```bash
make build-report
make build-all
```

## Compatibility commands (still available)

`run-*` targets still exist as lower-level compatibility targets. Prefer canonical names for new docs/automation.

## Source anchors

- `Makefile` (`make help`, `.PHONY` command groups)
- `notes/entrypoints.md`
- `notes/canonical_commands.md`
