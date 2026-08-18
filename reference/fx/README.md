# CCL reference data

`ccl_ars_usd.csv` is the reviewed local reference-data contract for explicit
offline USD/CCL flow valuation. It is intentionally committed with only its
header: no authoritative CCL observations or source were supplied with the
implementation mandate, and agents must not invent or acquire accounting input.

Before an authoritative valuation, Matías must populate and commit observations
with one market-date row per observation:

```csv
rate_date,ars_per_usd_ccl,rate_source,rate_series,source_reference
```

Do not manufacture weekend or holiday rows. `ars_per_usd_ccl` is ARS per one USD
at CCL. Every provenance field must be nonblank. Duplicate dates, nonpositive or
nonfinite rates, and empty rate files fail closed. Accounting runs never perform
network lookup; Git history plus the manifest SHA identifies the exact snapshot.
