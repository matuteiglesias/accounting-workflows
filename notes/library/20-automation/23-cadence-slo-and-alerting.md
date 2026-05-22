---
id: notes/library/20-automation/23-cadence-slo-and-alerting
title: "23 cadence slo and alerting"
sidebar_label: "23 cadence slo and alerting"
sidebar_position: 23
---

# 23 cadence slo and alerting

Status: draft
Last reviewed: 2026-05-22

## Recommended cadence (initial)

- Frequency: hourly timer (`OnCalendar=hourly`) or business-specific schedule.
- Command: `make run-env && make publish-latest` (or `make build-all`).

## Suggested SLOs

- Freshness SLO: `public/accounting/latest/manifest.json` updated within expected cadence window.
- Availability SLO: scheduled job success rate >= target threshold (team-defined).
- Correctness SLO: required published files present after each successful run.

## Alert triggers

Trigger alert when:
- scheduler missed expected run window,
- publish manifest missing,
- required report entry HTML missing,
- repeated stage failures in logs.

## Alert evidence

Capture in ticket:
- latest `journalctl` excerpt,
- failing command,
- missing output paths,
- classification bucket (scheduler/bootstrap/pipeline).
