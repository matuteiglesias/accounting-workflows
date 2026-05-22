---
id: notes/library/10-operations/15-incidents-first-15-minutes
title: "15 incidents first 15 minutes"
sidebar_label: "15 incidents first 15 minutes"
sidebar_position: 15
---

# 15 incidents first 15 minutes

Status: draft
Last reviewed: 2026-05-22

## First 15 minutes sequence

```bash
make help
make doctor
make smoke
journalctl --user -u accounting-spine-live.service -n 200 --no-pager
```

## Decision tree

1. If `make doctor` fails:
   - classify as runtime/bootstrap issue.
2. If `make smoke` fails with import/env errors:
   - classify as bootstrap issue.
3. If `make smoke` fails with invariant/output errors:
   - classify as likely pipeline regression.
4. If service did not run:
   - classify as scheduler wiring issue.

## Immediate communication template

- What failed:
- Classification (scheduler/bootstrap/pipeline):
- Impacted outputs:
- Temporary workaround:
- Next action and owner:
