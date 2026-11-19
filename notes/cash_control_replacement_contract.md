# Recorded Box cash-control replacement contract

Status: PR0 characterization contract, 2026-11-18.

## Authority and scope

`monthly_cash_accountability.csv`, produced by
`accounting.marts.treasury.build_monthly_cash_accountability`, is the governed
source for these controls. It is one row per `period` / `period_end` / `Box` /
`Currency`. The producer remains the only place that calculates the movement
components and reconciliation; the new metric IDs are references to its
existing columns, not a second financial formula.

The characterized run has 137 rows. Its private contents are intentionally not
copied into this repository. A skip-safe test records only the expected column
set, row count, and SHA-256 fingerprint for the exact run
`20261118T210059Z_FBPM` when that local reference exists.

## Metrics

| Metric | Source column | Meaning | Annual rule |
| --- | --- | --- | --- |
| `CTRL.BOX.NET_MOVEMENT` | `net_cash_flow` | recorded inflows minus recorded outflows during the month, by Box and native currency | sum monthly flows |
| `CTRL.BOX.CUM_NET` | `closing_control` | accumulated recorded movement from the defined zero origin through month-end, by Box and native currency | select the latest month-end; never sum stocks |

The arithmetic control is `opening_control + net_cash_flow = closing_control`.
The origin is a reporting convention: `closing_control` starts at zero at the
defined source horizon and accumulates recorded movement. It is not an actual
cash balance, verified liquidity, or a claim about a third-party account.

An unknown opening balance remains unknown. A zero-origin cumulative sum may be
reported alongside that limitation, but must not be described as an estimated
actual position or as evidence that the opening balance was zero.

Transfers are retained as recorded per-Box legs in the native currency. They
must not be silently dropped, netted across Boxes, or used to introduce a
consolidated Box-total metric in PR0. ARS and USD remain separate books; there
is no FX conversion or cross-currency aggregation.

## Boundaries

PR0 adds only the executable contract and characterization. Existing
`BS.CASH.*`, validated selectors, `monthly_cash_close`, annual report elements,
and professional interfaces remain unchanged. PR1 may add the new series to
consumers after deciding the publication/report contract. Later PRs can retire
validated-cash execution paths only after those consumers migrate.
