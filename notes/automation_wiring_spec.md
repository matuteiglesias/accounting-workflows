---
id: notes/automation_wiring_spec
title: "Automation Wiring Spec (Wave 2)"
sidebar_label: "Automation Wiring Spec (Wave 2)"
---

# Automation Wiring Spec (Wave 2)

Status: authority draft
Last reviewed: 2026-05-22
Audience: operators, automation stewards, coding agents

## Purpose

Define the authoritative wiring for periodic automated reporting so scheduler failures are easy to diagnose and do not silently diverge from the canonical command surface.

## Source-of-truth anchors

- `Makefile` canonical targets and env wrappers
- `README.md` operations + `journalctl` guidance
- `notes/frontend_snapshot_contract.md` publish contract and consumer boundary
- `notes/entrypoints.md` command taxonomy and canonical layering

## Canonical scheduled commands

Primary scheduled command (full pipeline + publish):

```bash
make build-all
```

Equivalent expanded path (for debugging/partial replay):

```bash
make ledger
make materialize
make debt
make debt-views
make metrics
make human-report
make publish-latest
```

Environment-file wrapper path (recommended for automation):

```bash
make run-env
```

`run-env` loads `ENV_FILE` (default `private/accounting.env`) and delegates to `run-accounting`.

## Working directory and environment contract

### Working directory

Scheduler jobs must execute from repository root:

```text
/workspace/accounting-workflows
```

Reason:
- Makefile assumes relative paths for `fixtures/`, `private/`, and output directories under `out/`.

### Environment file contract

Default env file path:

```text
private/accounting.env
```

Required keys for live ingest/reporting:
- `ACCOUNT_SA`
- `ACCOUNT_SHEET_URL`
- `ACCOUNT_SHEET_NAME` (defaults to `C. Long Ledger` if omitted)

Optional/operational keys:
- `OUT`
- `FREQ`
- `METRIC_MONTHS`
- `FIXTURE` (mainly for smoke mode)

## systemd template (recommended)

### User service unit example

`~/.config/systemd/user/accounting-spine-live.service`

```ini
[Unit]
Description=Accounting spine periodic build and publish
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/workspace/accounting-workflows
Environment=ENV_FILE=private/accounting.env
ExecStart=/usr/bin/bash -lc 'set -euo pipefail; make run-env && make publish-latest'
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

### User timer example

`~/.config/systemd/user/accounting-spine-live.timer`

```ini
[Unit]
Description=Run accounting spine periodically

[Timer]
OnCalendar=hourly
Persistent=true
Unit=accounting-spine-live.service

[Install]
WantedBy=timers.target
```

### Activation commands

```bash
systemctl --user daemon-reload
systemctl --user enable --now accounting-spine-live.timer
systemctl --user list-timers | rg accounting-spine-live
```

## Cron fallback (if systemd user timers are unavailable)

```cron
# Run at minute 5 every hour
5 * * * * cd /workspace/accounting-workflows && ENV_FILE=private/accounting.env /usr/bin/bash -lc 'make run-env && make publish-latest' >> /tmp/accounting-spine-live.log 2>&1
```

Note:
- Cron fallback is acceptable but weaker than systemd for structured logs and retry semantics.

## Failure routing and first response

Primary operational log source:

```bash
journalctl --user -u accounting-spine-live.service -n 200 --no-pager
journalctl --user -u accounting-spine-live.service --since "1 hour ago"
```

First-response sequence:

```bash
make help
make doctor
make smoke
```

Failure taxonomy:
- Missing module/env var -> runtime/bootstrap wiring issue
- Stage crash or contract output missing -> pipeline regression or data-shape issue
- Service/timer not firing -> scheduler wiring issue

## Publish boundary rule

Automated workflows should treat `public/accounting/latest/*` as the consumer handoff surface.

Consumer applications should not read producer internals under:
- `out/run/...`
- `out/metrics/...`
- `out/debt_resolution/...`
- `out/human_reports/...`

## Evidence map

### Commands validated (2026-05-22)

- `make help` -> pass
- `make doctor` -> pass
- `make smoke` -> fails in current environment when `pandas` is missing

### Code anchors

- `Makefile` (`build-all`, `publish-latest`, layered targets, `run-env`, `ENV_FILE`)
- `README.md` (`journalctl` usage and logging convention)
- `notes/frontend_snapshot_contract.md` (publish contract)
- `notes/entrypoints.md` (canonical command layering)

### Known assumptions

- Example service/timer paths assume Linux user-level systemd.
- Cron fallback uses `/tmp` log path by convention; adapt to host policy.
