---
id: notes/library/20-automation/24-recovery-and-rollback
title: "24 recovery and rollback"
sidebar_label: "24 recovery and rollback"
sidebar_position: 24
---

# 24 recovery and rollback

Status: draft
Last reviewed: 2026-05-22

## First response

```bash
journalctl --user -u accounting-spine-live.service -n 200 --no-pager
make help
make doctor
make smoke
```

## Classification

- Scheduler issue: timer/service not firing or wrong working directory.
- Runtime issue: missing dependency/env var.
- Pipeline issue: stage crash or contract artifact missing.

## Roll-forward preference

Prefer fixing env/wiring and rerunning canonical commands over ad-hoc manual artifact edits.
