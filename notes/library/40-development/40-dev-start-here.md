---
id: notes/library/40-development/40-dev-start-here
title: "40 dev start here"
sidebar_label: "40 dev start here"
sidebar_position: 40
---

# 40 dev start here

Status: draft
Last reviewed: 2026-05-22

## Goal

Help contributors change code without breaking pipeline contracts or published consumer surfaces.

## Read order

1. `41-refactor-definition-of-done.md`
2. `42-contract-change-protocol.md`
3. `../10-operations/12-command-surface.md`
4. `../30-consumers/31-report-consumer-guide.md`

## Golden rules

- Prefer canonical `make` targets in docs/automation.
- Treat `out/*` as producer internals; treat `public/accounting/latest/*` as consumer handoff.
- Keep metrics/debt/report logic in their owning subsystems.
