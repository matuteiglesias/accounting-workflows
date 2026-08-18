# USD/CCL Engineer 4 review — metrics, reporting and drilldown

**Status:** investigation only; no implementation or accounting-rule change.
**Mandate:** find the shortest safe path from reliable row valuation to the two desired management figures without converting the reporting stack.

## Decision

Do **not** introduce a transversal `reporting_mode` in the first reporting PR.
Produce two explicitly approved, basis-fixed management flow metrics and a
dedicated projected drilldown outside the native stack.

Frontier has the best existing contract vocabulary—`flow_or_stock`,
`currency_mode`, source, suitability, caveats—but is not yet a safe computation
or coexistence boundary. Register projected metrics there only after its
contract and series identities become basis- and completeness-aware.

## 1. Invariant being protected

1. Native semantic marts, frontier series, annual/professional tables, human
   reports and drilldowns remain byte/semantically unchanged.
2. Projected USD never masquerades as native USD:

   ```text
   Currency           = native currency
   valuation_currency = USD
   valuation_basis    = usd_ccl
   ```

3. ARS+USD combination happens only after selecting one projected measure and
   one valuation policy.
4. Incomplete projection is neither zero nor a reportable partial: reportable
   `value=NA`; a separately named available subtotal may be diagnostic only.
5. Transaction-date flow valuation never leaks into cash/debt stocks.
6. Every projected cell reconciles to the complete set of canonical rows and
   valuation-sidecar rows; sampled transaction IDs are insufficient.

## 2. Current behavior found in code

### 2.1 Semantic marts are native accounting boundaries

`monthly_flow_semantic_split.csv` provides native `Currency`, `amount_in`,
`amount_out`, `net_amount`, `amount_abs`, semantic dimensions, and sampled
lineage. `monthly_operating_statement.csv` provides native `Currency`, statement
lines, native `amount`, coverage and caveats.

Operating/funding/draw/Treasury formulas use already aggregated native semantic
measures. `source_tx_ids_sample` is useful diagnostically but is not full
membership evidence for a projected cell.

### 2.2 Frontier is structured but basis-blind

`FRONTIER_COLUMNS` already carries metric identity, semantic category,
flow/stock, period grain, currency mode, source/calculation/lineage, suitability,
caveat and validation. `SERIES_COLUMNS` carries period, native `Currency`, one
generic `value`, dimensions, source/run metadata and suitability.

It lacks:

- `valuation_basis`;
- `valuation_currency`;
- `valuation_policy_id`/hash;
- projection status and contributor/missing counters;
- a basis-aware primary-key contract.

Native values are copied from statement `amount`, rent `amount_in`, Treasury
`net_amount`, cash `close_amount`, and debt `open_amount`. Existing QA named
`no_cross_currency_sum` effectively checks that currency is populated; it does
not prove aggregation keys or safe coexistence with common-currency rows.

### 2.3 Annual/professional tables assume native currency and zero-fill

Professional annual builders group by native `Currency`, sum flows, and select
latest snapshots for stocks. `_annual_wide` and related helpers coerce invalid or
missing values to `0.0`, use pivot fill values, and add absent years as zero.

That is incompatible with projection completeness. A missing rate, stale
rejection, unsupported currency and a true economic zero are distinct states;
the current paths can collapse them into the same displayed zero.

### 2.4 Existing drilldowns are native

Metric drilldowns key cells by metric/period/native currency, filter canonical
rows by native `Currency`, sum native `amount`, and fill missing numeric values
with zero. Their manifest names only `ledger_canonical.csv`.

Professional drilldowns understand more formulas and the flow/stock distinction,
but still key/filter by native currency, zero-fill measures, and sometimes
reconstruct membership from sampled IDs. Their index lacks basis, valuation
currency/policy, completeness, and rate/valuation artifact identity. The
duplicated `_fx_treasury_measure_for_row` defect remains a separate blocker.

### 2.5 Human/professional reporting is not an automatic pass-through

Adding one projected column or switch would require auditing every formula,
pivot, formatter, label, cell ID, drilldown key and zero-fill default. A global
mode would also appear applicable to stocks even though flow and closing-rate
policies differ.

## 3. Hidden coupling and failure modes

### 3.1 Native/projected USD collision

If projected ARS and native USD both use `Currency=USD`, consumers can double
count native USD identity rows, combine bases, collide cell IDs, and produce a
drilldown containing only native USD while omitting projected ARS contributors.
Native currency must never be overwritten.

For a combined projected cell, use an explicit native-currency scope/breakdown
under a contract fixed to `valuation_basis=usd_ccl`; do not pretend the cell's
native currency is USD.

### 3.2 Frontier is a registry/selector, not yet the calculator

`currency_mode=by_currency` does not express valuation basis. Adding basis only
to frontier contracts would be false safety unless it also becomes part of
series identity, uniqueness QA, source allowlists, drilldown keys and every
consumer selector. Current basis-neutral metric IDs and keys assume one series
per metric/period/currency.

### 3.3 Zero-fill destroys incomplete-cell meaning

Projected reporting must distinguish:

1. valid zero;
2. no contributing rows;
3. missing rate;
4. stale rejection;
5. unsupported currency;
6. invalid native row;
7. missing source artifact.

Existing `fillna(0.0)` and pivot `fill_value=0.0` make a transversal migration
unsafe.

### 3.4 Derived formulas must fail closed

If either revenue or OPEX is incomplete, net operating is incomplete. If
funding/draws are required for a second formula and either is incomplete, that
figure is incomplete. Diagnostic available subtotals may exist, but reportable
formulas must never substitute zero for unavailable components.

### 3.5 Basis-neutral metric IDs can collide

Reusing `IS.NET.OPERATING` for native and projected values is safe only after
basis becomes part of every key and consumer selection. A smaller first PR can
use a dedicated `MGMT.CCL.*` namespace. Exact IDs require Matías because the two
requested figures are not named unambiguously.

### 3.6 Sampled lineage cannot support projected reconciliation

A numeric total can reconcile while omitting rows, multiplying a bad join, or
hiding unavailable rates. Projected drilldowns require deterministic complete
membership, with both anti-joins empty against the expected contributor set.

### 3.7 Global mode reintroduces unsupported outputs

Frontier contains native `TR.FX.NET`, cash and debt. A generic USD mode could
project an unsupported FX economic net or transaction-date-translate stocks.
This directly violates Engineer 2 and cash/debt boundaries.

### 3.8 Professional integration explodes scope

A reporting switch across professional packs requires new filenames/defaults,
public/internal decisions, basis-sensitive table contracts, drilldown handlers
and latest behavior. None is necessary to deliver two internal figures.

## 4. Disagreements with the existing packet

1. `reporting_mode=native|usd_ccl` is too broad for v1.
2. Frontier is the best eventual vocabulary, not the first computation boundary.
3. The prior PR3 combined metric contract, presentation and drilldown migrations;
   these should be staged.
4. Zero-fill hazards in professional annual tables need to be treated as a hard
   boundary, not merely a caveat.
5. `source_tx_ids_sample` is not adequate lineage.
6. Generic `value` is safe only in a basis-fixed artifact; shared series require
   basis/policy/completeness as part of identity.
7. Matías must name the exact two figures; code contains several plausible pairs
   and an agent cannot choose accounting meaning.

## 5. Recommended architectural choice

```text
canonical ledger
  + exact USD/CCL valuation sidecar
  + canonical semantic row classification
        |
        v
monthly_management_usd_ccl_components.csv
        |
        v
monthly_management_usd_ccl_metrics.csv     # exactly two approved figures
        |
        v
management_usd_ccl_drilldown_index.csv + detail CSVs
```

### Component artifact

Contains only approved flow components, native-currency contribution/breakdown,
projected value, basis/policy, completeness and full membership reference.
Operating, funding/distribution and gross Treasury diagnostics may coexist; no
stocks and no projected FX economic net.

### Metric artifact

Contains exactly the two approved metrics. Its basis is fixed:

```text
valuation_basis=usd_ccl
valuation_currency=USD
valuation_policy_id
projection_status
```

For incomplete cells:

```text
value                         = NA
available_value_usd_ccl       = diagnostic valued subtotal
projection_status             = incomplete
contributing_rows
valued_rows
identity_rows
missing_rate_rows
stale_rejection_rows
unsupported_currency_rows
invalid_native_rows
missing_tx_ids_artifact
```

A presentation may later render “N/A — 17 of 19 rows valued; 2 missing rates,”
but never display the available subtotal as the management figure.

### Where basis belongs

`valuation_basis` belongs at contract, cell/series identity, and drilldown lookup
levels. When frontier eventually registers these metrics, add valuation
currency/policy/status and a `projected_common_currency`-style currency mode to
both contracts and series keys.

### Exact projected drilldown evidence

Each detail row needs:

- canonical row identity;
- ledger and valuation artifact path/SHA;
- semantic artifact/rule version;
- native date, `Currency`, `amount`, direction and selected native measure;
- semantic bucket/subbucket;
- applied rate, date, age, source, series and policy;
- conversion status and projected amount;
- signed contribution to the cell or exclusion reason;
- Box/scope evidence.

The drilldown header/index needs metric/cell/period, basis/currency/policy,
ledger/rate/valuation SHAs, displayed value, recomputed detail sum/difference,
status counts and overall completeness.

## 6. Minimum PR boundary

After reliable sidecar and approved semantic eligibility exist, the first
reporting PR should add only:

1. one fixture-only projected management component artifact;
2. one fixture-only metric artifact with exactly two approved figures;
3. one dedicated projected drilldown index/detail CSV contract;
4. QA for basis identity, complete membership, formulas, completeness, source
   hashes and drilldown reconciliation.

It leaves unchanged:

- native semantic split/operating statement;
- legacy `metric_values` and native frontier files;
- annual/professional tables and human reports;
- native drilldowns;
- cash/debt reporting;
- publication and latest.

Frontier registration and a small internal presentation are separate later PRs.
A transversal mode is considered only after multiple consumers need it and
stock policies exist.

## 7. Tests that must fail before and pass afterward

| Test | Acceptance evidence |
|---|---|
| native artifact invariance | optional projected stage changes no native mart/frontier/table/report/drilldown bytes or values |
| basis identity | native USD and projected common USD cannot collide under declared keys |
| no implicit cross-sum | only basis-fixed projected builder combines native currencies under one policy |
| native USD identity | contributes exactly once at 1:1 |
| incomplete cell | `value=NA`, explicit incomplete status/counts, diagnostic subtotal separate, never zero |
| formula fail-closed | incomplete revenue/OPEX makes net operating NA; same for second figure components |
| zero versus unavailable | true zero remains complete zero; absent evidence remains NA/incomplete |
| two-figure allowlist | only approved IDs emitted; no cash/debt/legacy/projected `TR.FX.NET` |
| Treasury isolation | FX principal changes none of the two figures; gross diagnostics remain separate |
| full membership | detail IDs equal complete expected contributors; both anti-joins empty |
| drilldown arithmetic | signed detail sum equals complete displayed value; incomplete available sum equals diagnostic subtotal |
| drilldown provenance | ARS rows have full rate evidence; USD identity has rate 1 without fake market observation; SHAs match |
| scope isolation | FBPM/Household fixture includes exactly approved rows without ownership inference |
| metric-key uniqueness | duplicate basis/policy-aware cell key fails |
| zero-fill guard | annual/professional pivot rejects incomplete projected input instead of emitting zero |
| flow/stock boundary | cash/debt rows fail as unsupported in v1 builder |
| native drilldown regression | existing indexes/reconciliations unchanged |
| FX helper guard | projected professional integration blocked until duplicate helper is separately repaired |

## 8. Decisions that genuinely require Matías

1. Name the exact two management figures.
2. Approve each formula and component allowlist.
3. Decide whether funding/draws form the second headline, separate bridge lines,
   or are excluded from headlines.
4. Decide whether explicit FX cost affects either figure or remains Treasury-only.
5. Approve strict-NA incomplete-cell behavior and diagnostic subtotal disclosure.
6. Approve reporting scope: FBPM, Household, or separate explicit views.
7. Approve first artifacts as internal-only/non-public.
8. Approve metric IDs/labels and dedicated `MGMT.CCL.*` versus shared IDs.
9. Confirm transaction-date policy for each selected flow figure.
10. Decide whether unknown/review-required rows are excluded by rule or make the
    figure unavailable.
11. Approve row/aggregate rounding and reconciliation tolerance.
12. Decide whether frontier registration belongs in the first reporting PR
    (recommendation: no).
13. Decide when a transversal reporting mode becomes useful.
14. Decide whether gross Treasury diagnostics ship beside the two figures.
15. Approve disclosure that projected flows are management valuation, not native
    accounting currency or period-end stock translation.

## Evidence map

- Frontier contract/series/QA: `accounting/metrics/frontier.py`.
- Legacy zero-fill schema/formulas: `accounting/metrics/io.py`,
  `accounting/metrics/derive.py`.
- Native semantic measures/formulas/sampled lineage:
  `accounting/marts/semantic.py`.
- Annual values/status and zero-fill: `accounting/metrics/annual.py`.
- Professional flow/stock tables and zero-fill:
  `accounting/professional/annual_dashboard_tables.py`.
- Native metric drilldowns: `accounting/metrics/drilldown.py`.
- Rich native/professional drilldowns and duplicated FX helper:
  `accounting/professional/drilldown.py`.
- Human consumption/unavailable rendering: `accounting/human/*`.

## Completion record

```text
Changed: Engineer 4 metrics/reporting/drilldown packet; main investigation narrowed from transversal mode to two basis-fixed management artifacts and dedicated projected drilldown.
Accounting rule changed: None.
Fixture/test evidence: Existing code/tests inspected; proposed contract tests only.
Run ID: N/A.
Outputs inspected: Source-controlled code, tests and notes only.
Live inputs accessed: No.
Publication performed: No.
Totals/invariants checked: Static schema/formula/drilldown trace only; no live totals claimed.
Blocked accounting decision: Exact figures/formulas/IDs/scope/FX-cost treatment/incomplete disclosure/rounding/frontier timing/reporting mode/disclosure.
Next bounded action: Matías names and approves the two figures before the first projected reporting PR.
```
