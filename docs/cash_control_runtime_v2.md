# Cash-control runtime v2 target

Status: documentation target for cash retirement 2026-11; PR3c/PR3d runtime
cleanup pending integration.

This is a small entrypoint for the post-retirement contract. It does not
replace the detailed characterization or historical migration notes.

## Source graph

```text
ledger transactions
  -> box_balance_time_long.freq=M.csv
       -> direct Box-motor diagnostic month delta
  -> monthly_cash_accountability.csv
       -> CTRL.BOX.NET_MOVEMENT  (annual flow: sum months)
       -> CTRL.BOX.CUM_NET      (annual stock: latest month-end)
       -> treasury accountability report and governed metric consumers
```

The accountability producer is the source authority for the two `CTRL.BOX`
metrics. The native Box motor is the direct source for
`derived.diagnostic_box_level`. Provenance must name the source artifact,
period, Box, and native currency; ARS and USD are not combined.

## Retired target surface

After PR3c/PR3d integration, the runtime no longer depends on or produces:

- `accounting/cash_authority.py`;
- `accounting/cash_projection.py`;
- `accounting/cash_position_control.py`;
- `monthly_cash_close.csv` and its cash-mart producer.

Historical documents and completed runs may still mention these paths. That is
recorded evidence, not an active authority.

## Interpretation and consumers

`CTRL.BOX.NET_MOVEMENT` is a recorded monthly flow. `CTRL.BOX.CUM_NET` is a
zero-origin cumulative control and an annual stock selected at the latest
month-end. Neither is a validated bank balance or a guarantee of money held by
an administrator. External balance observations are optional future evidence,
not an implemented subsystem or a prerequisite for reporting.

The intended consumers are governed metrics, annual management presentation,
treasury accountability, and the derived Box-level diagnostic. Scenarios keep
their isolated, offline, non-publishing safety boundary. QA retains ledger,
movement, reconciliation, currency, Box-boundary, debt-consistency, annual
selection, report-contract, and historical-preservation checks.
