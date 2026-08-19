# Wave 4 PR15B — governed validated cash consumption

## Invariant

Cash headline values are positions backed by explicitly validated account snapshots. They are not sums of every cash/control row sharing period, Currency, and Box.

The runtime authority is `accounting.cash_authority` consuming `cash.position.validated` from `cash_position_control_specs_v1`.

Monthly selection:

```text
period / Currency / Box
  -> validated cash candidates only
  -> latest valid as_of_date per Box/account_id
  -> exactly one selected snapshot per account
  -> sum selected account closes
```

Annual selection:

```text
year / Currency / Box
  -> last period in the year containing validated cash candidates
  -> exact same monthly account-snapshot primitive
```

Annual cash never sums monthly positions.

## Exclusions

- `inferred_box_motor` is governed separately as `cash.control.inferred_box_motor`; it is reconciliation/control evidence and never a cash-headline fallback.
- `internal_balance` is internal party evidence and never headline cash.
- blank `party` is not an authority signal.
- missing or incomplete validated cash is `unavailable`, not zero.
- a candidate account with no valid as-of snapshot makes the position unavailable.
- duplicate latest account/as-of snapshots make the position unavailable.

## Affected layers

The same selector now governs modern cash values in:

- monthly metrics frontier (`BS.CASH.TOTAL`, `BS.CASH.CLOSE.BOX`);
- annual dashboard metrics (`BS.CASH.TOTAL`, `BS.CASH.CLOSE.BOX`);
- annual professional companion `annual_cash_close_by_box_wide`;
- professional drilldowns for `monthly_tables_cash_close_matrix` and `annual_cash_close_by_box_wide`.

Historical/non-modern schemas remain on the preserved compatibility implementations rather than having account identity synthesized.

`monthly_tables_diagnostic_box_level_matrix` is intentionally not migrated. It is a derived period-delta over inferred box control and belongs to Wave 5.

## Characterized before/after fixture

For 2026-01 / ARS / Property Management:

```text
internal balances          50
inferred box control      100
validated Bank A           70
validated Bank B           30
```

Before PR15B:

```text
monthly professional cash 200  # inferred + validated
annual professional cash  250  # internal + inferred + validated
```

After PR15B:

```text
monthly governed cash     100
annual governed cash      100
```

The 100 inferred control remains independently selectable as control evidence. The 50 internal balance remains visible as excluded evidence.

## Compatibility-preservation method

Pre-PR15B implementations are copied byte-for-byte to compatibility modules and public modules act as narrow facades. This avoids incidental changes to non-cash flows/debt/formulas while the cash authority is replaced.

This is migration scaffolding, not claimed LOC reduction. The Wave 4 closure audit counts both public facades and preserved compatibility code.
