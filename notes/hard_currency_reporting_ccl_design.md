# Hard-currency / CCL reporting design groundwork

## Native truth

Native accounting truth remains unchanged and by currency:

- `amount`
- `Currency`

Native reports must not sum ARS and USD.

## Future projection columns

Future hard-currency views may add projection columns without replacing native truth:

- `amount_native`
- `currency_native`
- `fx_rate_to_usd_ccl`
- `fx_rate_to_ars_ccl`
- `amount_usd_ccl`
- `amount_ars_ccl`
- `fx_rate_source`
- `fx_rate_date`
- `fx_rate_policy`
- `fx_conversion_status`

## Policy

1. Native reports remain by-currency and never sum ARS + USD.
2. Hard-currency reports are projections.
3. FX conversion rows must be classified before conversion mode is trusted.
4. One-sided FX proceeds can explain ARS liquidity but cannot be treated as economic income in USD CCL.
5. If both legs of an FX trade are present, hard-currency net should be near zero except spread/cost.
6. If only one leg is present, hard-currency view must show caveat `one_sided_fx_conversion`.
7. Missing CCL rate should produce unavailable, not zero.
8. Downstream report code should eventually select value column by mode: native uses `value`, `usd_ccl` uses `amount_usd_ccl`, and `ars_ccl` uses `amount_ars_ccl`.

## Potential future architecture

- Ingest: attach FX rates and converted projection columns where possible.
- Semantic: classify economic meaning independent of projection currency.
- Metrics: aggregate either native value or selected projection value.
- Reports: expose mode `native`, `usd_ccl`, or `ars_ccl`.

This PR documents and prepares contracts only; it does not implement full hard-currency mode.
