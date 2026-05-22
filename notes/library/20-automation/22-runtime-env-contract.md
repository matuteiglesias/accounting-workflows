---
id: notes/library/20-automation/22-runtime-env-contract
title: "22 runtime env contract"
sidebar_label: "22 runtime env contract"
sidebar_position: 22
---

# 22 runtime env contract

Status: draft
Last reviewed: 2026-05-22

## Required keys

- `ACCOUNT_SA`
- `ACCOUNT_SHEET_URL`
- `ACCOUNT_SHEET_NAME` (default `C. Long Ledger`)

## Automation wrapper

Preferred scheduled entrypoint:

```bash
make run-env
```

Default env file contract:

```text
ENV_FILE=private/accounting.env
```

## Related authority pages

- `notes/environment_bootstrap.md`
- `notes/automation_wiring_spec.md`
