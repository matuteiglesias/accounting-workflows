# 03 pipeline abstractions

Status: draft (code-anchored)
Last reviewed: 2026-05-22

## Layer model

1. Source inputs
2. Canonical ledger
3. Stage D materialized artifacts
4. Views/debt/metrics contracts
5. Human report surfaces
6. Published frontend snapshot

## Command abstraction

- canonical layer commands: `make ledger`, `materialize`, `debt`, `debt-views`, `metrics`, `human-report`, `publish-latest`
- composite command: `make build-all`

## Consumer abstraction

- read for consumption: `public/accounting/latest/*`
- read for debugging: `out/*`

## Governance abstraction

- every layer publishes named artifacts,
- downstream layers consume contract artifacts, not ad-hoc intermediate files.
