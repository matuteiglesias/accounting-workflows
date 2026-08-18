# Traceable USD/CCL projection investigation

**Status:** decision packet only; no accounting rule or runtime behavior is changed.
**Date:** 2026-08-18
**Scope traced:** canonical ledger → Stage D materialization → semantic/cash marts → debt resolution and balances → metrics → human/professional outputs → drilldowns.

## Executive conclusion

The repository can support a useful USD-at-CCL management view without changing native truth, but the existing `base_amount` mechanism is not yet a sufficient contract. It is an exact `(Date, Currency)` multiplication with no source, effective-date, policy, or status provenance, and current Makefile live ingest does not pass either FX option. Most downstream stages also select fixed native measure columns, so additive ledger columns will normally be dropped rather than accidentally consumed. That is a useful safety property, not automatic propagation.

Engineer 1's follow-up architecture review changes the recommended storage
boundary: the projection should be a versioned sidecar, not columns on either
canonical ledger. The safe boundary is:

1. preserve `amount`, `amount_cents`, and `Currency` exactly;
2. leave both canonical ledger files byte-for-byte untouched and write a separately selected row-level **flow projection sidecar** with explicit provenance and NA behavior;
3. build projection-aware semantic flow outputs in parallel with native outputs;
4. exclude FX principal legs from operating/funding/distribution views and expose only a separate **gross** treasury bridge with valuation completeness; do not claim pairing, spread, or `FX economic net` in v1;
5. defer cash and debt **stock** conversion until an as-of/closing-rate contract exists.

A reporting selector must therefore choose both a monetary basis (`native` or `usd_ccl`) and an accounting measure. Renaming projected USD to `amount`, or changing `Currency` to `USD`, would defeat existing currency guards and is explicitly out of scope.

## A. Current architecture

### A1. Pipeline/control-plane reality

The operational path is `run-ingest` → `run-materialize` → `run-marts`; debt resolution and debt views follow, then metrics, dashboard/human report, publication, and optional professional drilldowns. `run-full` composes those stages. The live `run-ingest` recipe currently passes sheet, scope, run ID, and output arguments but **does not pass** `--fx-rates` or `--base-currency`, despite those CLI arguments existing. Smoke ingest likewise does not use rates. Consequently `base_amount` is normally an all-null additive column in canonical CSVs.

`SYSTEM.yaml` confirms that this repository owns canonical transformation/reporting, while intake, viewer presentation, and public documentation are external boundaries. A rate acquisition service belongs outside this first change unless separately approved; this repository should consume a reviewed rate artifact.

### A2. Canonical ledger

Relevant code is `accounting/ledger/ingest.py`:

- `_normalize_columns` canonicalizes source names; `_coerce_money_and_dates` parses `amount`, derives `amount_cents`, and uppercases `Currency`.
- `_build_tx_id` hashes date, parties, native currency, classifiers, native cents, and source provenance. A future projection must not participate in identity.
- `_load_fx_rates` accepts CSV/Parquet with columns normalizable to `Date` and `Currency`, plus the first column named `rate_to_base`, `rate`, `fx`, or `tc`. It returns only `Date`, `Currency`, `rate_to_base`; it does not validate positivity, uniqueness ahead of merge, base currency, source, quote convention, CCL instrument, timezone, publication time, or policy.
- `_attach_base_amount` treats rows whose `Currency == base_currency` as identity and otherwise left-merges on exact `(Date, Currency)` with `validate="m:1"`; it computes `amount * rate_to_base`. There is no weekend fallback. A missing rate stays nullable and creates a `missing_fx_rate` anomaly.
- `build_ledger_base` always emits `base_amount` (all NA without a rate path) and writes both status-filtered `ledger_canonical.csv` and scoped all-status `ledger_canonical_all_status.csv`.

The multiplication convention is only correct if the file expresses “units of base currency per one native unit.” For USD projection of ARS, that means USD/ARS (a small multiplier). Common Argentine CCL feeds instead quote ARS per USD, which requires division. The generic name `rate_to_base` does not make that convention safe.

`accounting/support/currency.py` is not a safe alternative. `convert_currency` mutates selected values and overwrites `Currency`; it assumes a `Rate` convention (ARS divided for USD; USD multiplied for ARS) and drops `Rate`. `_ensure_amount` also fills unparseable/missing money with zero. Both conflict with additive truth and missing-as-NA requirements. The Pydantic `Money.to_base` helper in `accounting/contracts/models.py` has yet another division/multiplication convention and is not the ledger ingest path.

### A3. Materialization

`accounting/stage_d/materialize.py` reloads `ledger_canonical.csv` and uses fixed output schemas:

| Output | Input measure | Output measures | Currency invariant | Projection behavior |
|---|---|---|---|---|
| `per_flow_time_long.freq=*.csv` | `amount` | `amount`, `n_tx` | grouped by `Box, Currency, Flujo, Tipo, period` in core timeseries | all added projection columns dropped |
| `per_party_time_long.freq=*.csv` | `amount` → expanded `signed_amount` | renamed aggregate `amount`, `n_tx` | includes `Currency` in party grain | dropped |
| `daily_cash_position.csv` | `amount` → payer/receiver signed rows | cumulative `balance` | cumulative by `Box, party, Currency` | dropped; this is an inferred party balance, not validated cash |
| `box_balance_time_long.freq=*.csv` | `amount` | `in_amt`, `out_amt`, `net`, `cum_net` | group/cumsum by `Box, Currency` | dropped |
| `box_flow_balance_time_long.freq=*.csv` | `amount` | `in_amt`, `out_amt`, `net`, `n_tx` | grouped by `Box, Currency, Flujo, Tipo` | dropped |
| `loans_time.freq=*.csv` | loan register/native schedules | loan schedule measures | loan/currency-specific | no natural ledger projection propagation |

Direction is inferred from the Box initials (`Family Business → FB`, `Property Management → PM`, `Household → HH`) versus payer/receiver. Rows matching neither are omitted from box balance artifacts with a warning. Projection work must not use FX conversion as an excuse to change that existing behavior.

Because schemas explicitly slice columns, merely adding six fields at ingest leaves all native materializations unchanged. A projection output needs a separate function/output or a parameterized internal helper whose native invocation is byte-regression-tested.

### A4. Semantic marts

`accounting/marts/semantic.py` prepares ledger rows from native `amount`, classifies each row, then emits:

- `classification_audit.csv`: row-level `amount` and native `Currency`, semantic/direction/funding metadata, but a fixed `AUDIT_COLUMNS` list drops `base_amount` and any future projection fields;
- `classification_audit_summary.csv`: `amount_total` and `amount_abs_total`, grouped by period, native `Currency`, semantic bucket/subbucket/status/rule;
- `monthly_flow_semantic_split.csv`: derives `amount_in`, `amount_out`, `net_amount = amount_in - amount_out`, and `amount_abs`, grouped by period, `Currency`, Box/party/funding dimensions and semantic bucket/subbucket;
- `monthly_operating_statement.csv`: consumes those four measures, groups by period and `Currency`, and emits a generic `amount` per statement line.

Operating revenue, property OPEX, funding, family distributions, debt movement, unknown, and treasury FX remain distinct buckets. Treasury lines include conversion in, conversion out, cost, and net; they are excluded from operating revenue, property OPEX, funding, draws, and debt. This semantic separation is the right place to decide economic-flow treatment, after row valuation—not in the rate join.

Projection columns will currently be dropped at the row audit and will not reach semantic aggregates. A v1 projection must retain row lineage (`tx_id`) and either add parallel projected measures (`amount_in_usd_ccl`, etc.) or emit parallel projection artifacts. Parallel artifacts are safer because many consumers assume `amount` means native.

### A5. Cash marts: inferred versus validated stocks

`accounting/marts/cash.py::build_monthly_cash_close` combines three materially different sources:

1. last daily `balance` by month/Box/party/Currency from `daily_cash_position.csv`; marked `internal_balance`, internal-only, not frontend-safe;
2. `cum_net` from box balance; renamed `close_amount`, marked `inferred_box_motor`, not frontend-safe;
3. externally supplied `validated_cash_close.csv`; `close_amount` is accepted only for explicit validation statuses, a named validator, and approved source types; marked frontend-safe.

Every row retains native `Currency`, and QA asserts no cross-currency total. Transaction-date conversion is **not** a true CCL valuation of any closing stock: it sums historical translated movements and embeds historical rates. A period-end stock needs `close_amount_usd_ccl = close_amount_native × rate(as_of_date or approved period-close date)`, with the rate date/policy recorded. Validated cash should normally use each row’s `as_of_date`; the inferred motor may be projected for analytical reconciliation only and must retain its suitability caveat.

### A6. Debt

`accounting/debt/resolve.py` reads the all-status scoped ledger, normalizes native `amount` and `Currency`, creates principal/interest open items, and allocates repayments only within matching debtor/creditor/currency keys. It produces `original_amount`, `open_amount`, `repayment_amount`, `allocated_amount`, and `leftover_amount` in native currency.

`accounting/debt/balance_views.py` builds daily native stock by debtor/creditor/currency/item type, renames aggregated `original_amount` to `open_amount`, pivots principal and interest, and derives `open_total`. Periodic views select last snapshots. `accounting/marts/debt.py` wraps these as:

- `monthly_debt_position.csv`: `open_amount`, `open_principal`, `open_interest`, `open_total`, by period/as-of/debtor/creditor/`Currency`/component;
- `monthly_debt_activity.csv`: flow-like `new_principal`, `interest_accrued`, `repayments`, `adjustments`, opening/closing totals and `net_change`, by pair and native currency.

Projection must not affect repayment allocation or debt status reconciliation. Transaction-date projection is meaningful for debt **activity** disclosures. Open debt is a stock and needs a closing/as-of CCL rate applied to the native balance, not a sum of projected origination and repayment flows. Principal and interest components must use the same snapshot rate so their projected sum equals projected total.

### A7. Metrics

Legacy metric builders consume fixed columns:

- flows: materialized/mart `amount` or `amount_out`, grouped by `TimePeriod` and `Currency`;
- cash: daily `balance`, selecting last observation within periodic currency groups;
- debt: `open_principal`, `open_interest`, or `open_total`, from periodic debt tables.

They normalize all results into `metric_values.csv` with a single `value` and lowercase `currency`. Derived formulas pivot on `(period_grain, period, currency, run_id, as_of_date)`, so native arithmetic is currency-isolated. However `ensure_metric_values_schema` coerces missing `value` to `0.0`; projected-unavailable rows must not be pushed through that schema unchanged or missing FX will become zero.

The newer frontier reads canonical semantic/cash/debt marts, emits `frontend_metric_series.csv` with `value` and `Currency`, and explicitly checks that money rows retain currency. It is the better eventual selector boundary. A new basis dimension (for example `valuation_basis=native|usd_ccl`) is required before a common-USD series can coexist safely; overloading native `Currency=USD` cannot distinguish native USD from projected USD and invites duplicate totals.

### A8. Human reports and professional pack

Human compact, tables, document, and front renderers consume `value`, native `currency`/`Currency`, and source-specific measures. They pivot/group by currency and label cards with currency. These can support `usd_ccl` with relatively small renderer changes **after** a selected, reconciled metric series exists; they should not perform row conversion themselves.

Professional annual companion builders already understand the critical flow/stock distinction:

- funding and debt activity are annual sums by explicit dimensions and `Currency`;
- cash close and debt stock select the latest monthly/as-of snapshot, never sum stocks;
- ARS and USD remain separate rows.

These builders could be generalized to accept an explicit value selector/basis. Existing professional CSV tables must remain native. A parallel pack namespace or manifest-declared reporting basis is safer than adding projected year columns to native tables.

### A9. Drilldowns and the shadowed FX helper

`accounting/professional/drilldown.py` reconciles displayed cells to semantic audit/split, operating statement, cash, debt, annual metrics, and ledger rows. It filters period and `Currency`, chooses fixed measures, sums with `fillna(0)`, and records displayed value, matched sum, residual, source, filters, and row context.

The file defines `FX_TREASURY_TABLE_IDS`, `FX_MEASURES`, and `_fx_treasury_measure_for_row` **twice**. Python uses the second definition. The shadowed first version deliberately refuses to default compact/all-measure rows and checks both `measure` and `metric`; the active second version checks `measure`, then label mapping, and defaults compact blank rows to `net_amount`. That can reconcile a compact cell against the wrong measure without an obvious exception. This open defect should be repaired and regression-tested before projection drilldowns are trusted, but separately from the valuation PR.

Projected drilldowns need to carry `valuation_basis`, select projected measure columns, expose rate provenance/status, and reconcile through `tx_id`. They must not filter projected rows by replacing native `Currency` with USD; the contributing rows’ native currency remains essential evidence.

## B. Existing reusable machinery

### Retain or adapt

- canonical date/currency normalization and status/scope filtering;
- exact merge cardinality validation (`m:1`), strengthened with explicit rate-key uniqueness checks;
- nullable result on unmatched rate and anomaly reporting;
- identity treatment for native USD;
- additive placement near ingest, so all downstream valuation derives from the same reviewed rate artifact;
- semantic FX buckets, flow/stock distinction in professional annual tables, run IDs, source-row/`tx_id` lineage, and current currency-grained QA.

### Replace or supersede

- generic `base_amount` as the public CCL contract: it lacks quote convention, provenance, effective date, policy and status;
- ambiguous `rate_to_base`: use a named direction and unit;
- exact-date-only behavior as an implicit policy;
- mutating `convert_currency` and any helper that fills unavailable monetary values with zero;
- reports choosing arbitrary numeric columns by name fallback;
- projected values in the legacy `metric_values` schema until NA semantics and basis are explicit.

`base_amount` may remain for backward compatibility, but v1 should not derive CCL reports from it. If retained, document it as legacy/generic and test that adding the CCL contract does not change it.

## C. Semantic hazards

### C1. Cross-currency aggregation

The repository is currently protected by pervasive `Currency` grouping. The largest new risk is that all projected rows are USD-denominated while their native `Currency` differs. Safe designs retain `Currency` as native and add `valuation_currency="USD"` plus `valuation_basis="usd_ccl"`. A projection aggregator may intentionally group across native currencies only when explicitly selecting the projected measure and basis. Native aggregators must continue requiring/grouping native `Currency`.

### C2. Actual FX transactions

Current semantic recognition uses `Cambio:FX`, an FX payer/receiver, or related text. An ARS proceeds row can be represented alone (`FX → PM`, `treasury_fx/fx_conversion_proceeds`). A source-currency outflow can be a separate row (`PM → FX`, `fx_conversion_outflow`). Cost/spread is a third possible row. Fixtures/tests demonstrate one-sided ARS proceeds, not a stable two-leg correlation contract.

There is a more immediate contract defect: ingest documents `amount` as signed,
but semantic code copies it into `amount_in`/`amount_out` according to
payer/receiver-derived direction and then subtracts out from in. Existing
fixtures normally use positive magnitudes. A negative outbound `amount` can
therefore yield a positive `net_amount`. V1 must fail closed on sign/direction
contradictions until Matías approves whether canonical amount is signed or a
nonnegative magnitude; an agent must not silently apply `abs`.

FX recognition is also ordered after several rent/OPEX/funding/debt early
returns. Merely excluding rows already labeled `treasury_fx` is not enough:
fixture-level leakage tests must prove how competing FX and semantic evidence is
handled or route the row to review under an approved precedence rule.

No canonical `fx_trade_id`, source-document ID, paired transaction ID, quantity pair, or execution-rate field exists. `tx_id` is row identity, not trade identity. Same date/Box/FX counterparty is insufficient: multiple trades, fees, settlement timing, or split legs can collide. Thus paired legs cannot currently be identified reliably.

Independently projecting each leg is mathematically necessary for a liquidity bridge but insufficient for an economic-flow statement:

- USD outflow leg: e.g. `amount=100 USD` → projected principal `100 USD`;
- ARS proceeds leg: e.g. `amount=120,000 ARS`, at policy CCL `1,200 ARS/USD` → projected principal `100 USD`;
- signed treasury bridge: proceeds `+100`, source leg `-100` → approximately zero;
- separately represented fee/spread remains the economic cost.

If both legs are instead summed as absolute “activity,” the conversion becomes `200 USD` and is double-counted. If only the ARS row exists, `+100 USD` explains ARS liquidity but must not become revenue/funding/economic gain. If both ledger amounts are positive and direction is expressed only by payer/receiver, projected net must use the semantic direction (`amount_in - amount_out`), not raw `amount` sum.

The following invariant is testable only for a future, explicitly paired fixture:

```text
projected_fx_net = Σ(projected amount_in) - Σ(projected amount_out) - separately classified cost treatment
abs(projected_fx_principal_net) <= approved tolerance
```

The tolerance must be policy-defined (absolute USD and/or bps), not guessed. Differences may represent feed timing, execution-vs-policy-rate basis, rounding, spread, or incomplete representation. Pairing is therefore **not** a prerequisite for v1 operating/funding projection or gross conversion-in/out visibility; it is a prerequisite for any later claim of trade completeness, execution spread, realized result, or economic FX net. V1 must not invent `paired`, `one_sided`, or `ambiguous` trade statuses from date/party/amount proximity.

### C3. Scope and ownership

Scope is selected solely by canonical `Box`. Projection and pairing must occur **after** that scoped evidence boundary and must not pair an FBPM row with an excluded Household row invisibly. A cross-scope candidate can be reported as incomplete but cannot be imported or used to infer ownership/entitlement.

### C4. Stocks versus flows

Transaction-date rates are coherent for flow-period management views if the policy is disclosed. They are not closing valuations. Cash/debt stock conversion needs a valuation date tied to the selected snapshot; historical projected movements do not equal translated closing native balance when rates move. Any translation difference belongs in a separate management reconciliation, not fabricated operating income.

### C5. Null destruction

Multiple downstream helpers use `fillna(0.0)` for native aggregation convenience. Projected availability must therefore be aggregated with an explicit completeness rule: if required contributing rows lack rates, the projected aggregate is NA/unavailable (plus missing counts/amount), not a partial total masquerading as complete and not zero.

## D. Recommended v1 boundary

The smallest useful v1 is a **fixture-only, transaction-date USD/CCL valuation sidecar contract**, limited to native USD identity and ARS rows with an approved synthetic rate artifact. It must consume an already-written `ledger_canonical.csv` read-only. Projected semantic flow reporting follows only after the sidecar's identity, cardinality, and reproducibility contracts pass.

Include:

- a deterministic `ledger_valuation_usd_ccl.csv` sidecar bound to an exact canonical-ledger artifact, valuation policy, and rate snapshot;
- a fail-closed uniqueness/cardinality check before claiming a 1:1 `tx_id` join (supplied `tx_id` values are not currently guaranteed unique);
- explicit exact/previous-available rate status under an approved transaction-flow policy;
- native USD identity;
- projected semantic flow measures and completeness metadata;
- FX principal excluded from projected operating income/OPEX/funding/draws, with row-level leakage QA and a gross treasury conversion-in/out/cost bridge;
- two basis-fixed, Matías-approved management flow metrics in dedicated projected artifacts; no transversal `reporting_mode` in the first reporting PR;
- FBPM/Household scope isolation and projected drilldowns.

Defer:

- validated cash, inferred cash, and debt stock USD totals;
- monetary/translation gain accounting;
- automated CCL acquisition or selection among instruments;
- monthly-average or month-end restatement;
- automatic FX pairing, projected `FX economic net`, computed spread, or completeness claims without a source-approved `fx_trade_id` contract;
- changing legacy native materializations, metric schema, or published pack defaults.
- any live rate acquisition or live FX activation.

This boundary produces useful comparable operating/funding/treasury flows while avoiding false claims about period-end net worth or debt exposure.

## E. Proposed data contract

### E1. Rate artifact contract

One row per `(rate_date, currency_native, valuation_currency, rate_type, source_series)` after validation:

| Column | Meaning |
|---|---|
| `rate_date` | market/effective date of the observation (ISO date) |
| `currency_native` | native unit being valued (`ARS`; USD identity need not be supplied) |
| `valuation_currency` | `USD` |
| `rate_type` | controlled value `ccl` |
| `ars_per_usd_ccl` | positive ARS per USD quote; naming removes divide/multiply ambiguity |
| `fx_rate_source` | stable provider/dataset identifier, not merely a URL |
| `fx_rate_source_reference` | observation/series/version reference |
| `observed_at` | publication/retrieval timestamp if known |
| `quality_status` | `approved`, `provisional`, or `rejected` |

For ARS, `fx_rate_to_usd_ccl = 1 / ars_per_usd_ccl`. Store the multiplier actually applied on each ledger row for reproducibility.

### E2. Valuation sidecar columns

The requested minimum contract belongs in `ledger_valuation_usd_ccl.csv`, not
in `ledger_canonical.csv` or `ledger_canonical_all_status.csv`. The sidecar is
one row per source canonical row for one declared policy/rate snapshot, subject
to a validated unique join key. Its minimum row contract should be:

| Column | Exact meaning / null behavior |
|---|---|
| `amount_usd_ccl` | native `amount × fx_rate_to_usd_ccl` under the separately approved canonical sign contract; nullable Float64; never filled with zero |
| `fx_rate_to_usd_ccl` | USD per one unit of native currency; `1.0` for native USD; null if unavailable/unsupported |
| `fx_rate_date` | effective observation date used; transaction date for USD identity; null if unavailable |
| `fx_rate_source` | stable source ID; `native_identity` for USD; null only when no applicable source exists |
| `fx_rate_policy` | versioned policy ID, e.g. `ccl_txn_prev_available_v1`, not free-form prose |
| `fx_conversion_status` | controlled status below |

Recommended additional columns are `valuation_currency="USD"`, `valuation_basis="usd_ccl"`, `fx_rate_source_reference`, and `fx_rate_age_days`. Do not add `amount_native`/`currency_native` as duplicative truth unless an external interface requires aliases; canonical `amount`/`Currency` already have those meanings.

The valuation manifest must independently bind `source_ledger_sha256`, a
native-business fingerprint, source row count/population, `valuation_policy_id`,
`rate_artifact_sha256`, and `valuation_artifact_sha256`. A rate correction must
change the latter two identities and projected outputs while leaving canonical
ledger bytes, canonical artifact SHA, native-business fingerprint, and all
native outputs unchanged.

Engineer 3's reliability review strengthens this contract. The accounting run
must accept only a local immutable rate snapshot—never a URL or network lookup—
verify its SHA before parsing, and record the resolved policy hash and code
identity. The deterministic sidecar and timestamped execution manifest are
different artifacts: sidecar bytes must reproduce exactly, while declared
volatile manifest fields such as `generated_at` may differ.

Minimum valuation-manifest evidence is:

```text
valuation_schema_version
valuation_policy_id
valuation_policy_sha256
source_ledger_artifact
source_ledger_sha256
source_ledger_rows
source_scope_tag
source_status_population
rate_artifact
rate_artifact_sha256
rate_source
rate_series
rate_observation_count
rate_raw_observation_count
rate_rejected_observation_count
rate_min_date
rate_max_date
valuation_rows
native_usd_identity_rows
exact_matches
previous_available_matches
stale_rejections
missing_rates
unsupported_currency_rows
invalid_native_rows
valuation_artifact_sha256
code_revision
code_dirty
implementation_id
generated_by_network_access=false
```

Match-status counters must be mutually exclusive and reconcile exactly to
`valuation_rows`. `rate_observation_count` means accepted canonical rate
observations, not raw input rows or ledger matches. Duplicate canonical
observation keys—including identical duplicates—fail validation rather than
being silently deduplicated.

Controlled statuses:

- `identity_native_usd` — rate 1, amount copied, rate source `native_identity`;
- `converted_exact_date` — approved ARS observation on transaction date;
- `converted_previous_available` — approved prior observation under policy; expose age;
- `unavailable_missing_history` — no eligible prior/exact observation;
- `unavailable_after_policy_limit` — prior observation is too stale;
- `unsupported_currency` — neither USD nor approved ARS;
- `invalid_native_amount` — native amount unavailable;
- `invalid_rate` — nonpositive/duplicate/rejected observation.

Unavailable statuses require null projected amount and null applied rate (except diagnostics may retain a rejected candidate separately). Status/provenance must exist for every row, including identity.

### E3. Aggregated projection contract

Parallel semantic outputs should carry `valuation_basis`, `valuation_currency`, projected measures, `projection_status`, `n_rows`, `n_converted`, `n_unavailable`, and sampled/machine-readable missing `tx_id`s. `projection_status=complete` only when all required contributing rows are converted/identity. A partial diagnostic sum may be separately named `available_amount_usd_ccl`; the reportable `amount_usd_ccl` remains NA when incomplete.

### E4. Rate policy decision packet

No policy is approved by this investigation. Recommended candidates for explicit owner approval are:

| Situation/use | Recommended explicit policy | Reason / unresolved choice |
|---|---|---|
| Native USD | identity at row/snapshot date | no market lookup; preserves 1:1 |
| Transaction flow, exact rate exists | exact transaction date | closest reproducible policy observation |
| Weekend/holiday flow | previous approved available business observation, with age | avoids look-ahead; owner must approve maximum staleness (candidate: 3–5 calendar days) |
| Missing internal date | unavailable | no silent interpolation |
| Before history | unavailable | no backfill/extrapolation |
| After history | previous available only within approved staleness; otherwise unavailable | prevents indefinite stale valuation |
| Monthly flow management | sum transaction-date projected rows for v1 | traceable to rows; monthly-average would be a different approximation |
| Period-end cash/debt stock | exact snapshot/as-of rate, or previous approved available close under a separately versioned close policy | stock valuation, not transaction projection |
| Comparative balance-sheet presentation | each period’s closing policy | do not use current rate for historical periods unless explicitly labeled restatement |
| Forecast/budget | out of v1 | requires a scenario/rate authority separate from observed CCL |

Monthly-average and month-end rates may be useful management conventions, but they must be separate modes/policy IDs, never silent fallbacks for transaction flows. “CCL” itself remains underspecified until the owner approves the instrument/series, source, bid/ask/mid/close convention, publication cutoff, timezone, corrections policy, and licensing/redistribution boundary.

## F. Change-impact map

### Must change for v1

- a new bounded valuation module: read an existing canonical ledger and versioned rate snapshot, validate identity/cardinality, and write the sidecar plus manifest without rewriting ingest outputs;
- artifact contracts/manifests: classify the sidecar as derived valuation evidence rather than transaction source of truth, and record separate ledger/rate/valuation hashes;
- a new explicit fixture-only Make target (if command plumbing is included); do not change live ingest or fetch rates over the network;
- contract checks: exact canonical schema/byte invariance, sidecar cardinality, source binding, status/NA, exclusive counter reconciliation, deterministic serialization, rate/policy/code identity, duplicate observation rejection, no-network enforcement, and distinct hash identities;
- `accounting/marts/semantic.py`: retain projected row lineage and build parallel projected flow measures/artifacts; do not alter native schemas/formulas.
- a small parallel management projection builder: exactly two approved flow metrics, explicit valuation basis/policy/completeness, and no legacy zero-fill behavior;
- `accounting/metrics/frontier.py` should not compute the first projected figures; register them only later after its contract/series keys become basis-aware;
- human/professional builders remain unchanged in the first reporting PR.
- `accounting/professional/drilldown.py`: remove the duplicate FX helper in its own bounded repair; later add projected measure/rate provenance reconciliation.
- `accounting/ledger/ingest.py` should not receive CCL columns in v1; a later bounded contract decision may add duplicate supplied-`tx_id` validation or a stronger governed row key.

### Should not need semantic changes

- `accounting/debt/resolve.py`, `accounting/debt/rules.py`, and repayment allocation: must remain native.
- `accounting/debt/balance_views.py`: native balances remain canonical; a future stock valuation wrapper should consume its snapshots.
- existing native materializers in `accounting/stage_d/materialize.py`: leave their fixed schemas/amount selection unchanged.
- scope selection in `accounting/scope.py`, publish atomicity, latest links, intake/viewer/docs repositories.
- existing native `metric_values.csv`, validation identities, and native human/professional pack defaults.

### Future stock PR only

- `accounting/marts/cash.py`: add a parallel close-valuation wrapper using `as_of_date`, never historical transaction projections.
- `accounting/marts/debt.py`: add parallel position projections at snapshot rates and transaction-date activity projections.
- professional annual cash/debt tables and drilldowns: select projected stock/value measures while retaining latest-snapshot logic.

## G. Reconciliation/test matrix

All tests use synthetic fixtures and bounded smoke stages; no live input or publication is required.

| Invariant | Fixture/check | Acceptance evidence |
|---|---|---|
| Native ledger unchanged | build canonical once, then run sidecar valuation against rate snapshots V1 and V2 | exact bytes, schema, canonical artifact SHA, and native-business fingerprint of both canonical ledgers remain unchanged; no projection columns appear |
| Native reports unchanged | run bounded fixture pipeline before/after; compare all existing native CSVs/HTML semantic source tables or golden hashes | byte equality where deterministic; otherwise keyed semantic equality and zero native reconciliation differences |
| Native USD 1:1 | positive, negative, zero and fractional USD rows | rate `1.0`, projected amount exactly native amount, identity source/status, no rate file dependency |
| Missing FX is NA | ARS dates before history, stale after history, gap beyond limit, invalid/duplicate rates | null applied rate and amount; precise unavailable status/anomaly; aggregates unavailable, never zero/partial-as-complete |
| Sidecar join is truly 1:1 | unique-key fixture plus a fixture with duplicate supplied `tx_id` | equal source/sidecar row counts, empty anti-joins, `validate="1:1"`; duplicate source IDs fail closed rather than deduplicate or multiply rows |
| Sidecar is bound to its source | attempt to join a sidecar to a different canonical artifact | source SHA/fingerprint mismatch fails before any total is calculated |
| Revaluation identities stay separate | value one canonical artifact with rate snapshots V1 and V2 | ledger bytes/SHA/fingerprint identical; rate snapshot SHA and valuation artifact SHA differ; repeated identical inputs reproduce identical sidecar bytes |
| Rate snapshot is immutable evidence | mutate bytes under the same filename/version and try a declared SHA mismatch | changed bytes produce a new valuation identity; expected-SHA mismatch fails before parsing; provenance never records `latest` |
| Rate observations validate fail-closed | duplicate/conflicting date-series keys, invalid date/rate, nonpositive/infinite rate, unsupported quote | no silent coercion/drop/deduplication; raw = accepted + rejected and min/max use accepted observations |
| Match counters reconcile | identity, exact, allowed previous, stale, missing, unsupported and invalid-native fixture rows | categories are mutually exclusive and sum exactly to `valuation_rows`; rate observations are not confused with ledger matches |
| Accounting run is offline | URL input plus network APIs patched to raise | URL/URI rejected; local fixture valuation succeeds without network; manifest asserts offline generation |
| Deterministic serialization | reorder ledger/rate inputs and rerun identical valuation | fixed row/column ordering, dates, nulls, decimals, newline and encoding yield identical sidecar bytes; only declared volatile manifest fields may differ |
| Immutable rerun behavior | rerun into an existing valuation identity | identical existing SHA is accepted without mutation; differing existing bytes fail rather than overwrite |
| ARS policy reconciliation | exact-date and weekend/holiday ARS rows with known quotes | `amount_usd_ccl = amount / ars_per_usd_ccl`; recorded effective date/source/policy/age; aggregate equals row sum within decimal tolerance |
| No native ARS+USD cross-sum | mixed-currency fixture through materialize, semantic, metrics and reports | every native money aggregate retains native currency; existing `no_cross_currency_*` QA passes; no new native total without currency |
| Explicit projected combination only | same mixed fixture in USD mode | cross-native-currency total exists only with `valuation_basis=usd_ccl`, `valuation_currency=USD`, complete status, and contributing native-currency breakdown |
| FBPM/Household isolation | matching FX-like legs split across Boxes plus existing scope fixture | FBPM output contains only scoped row IDs; no cross-scope pairing; Household run independently reconciles; missing counterpart is flagged rather than imported |
| Fully represented USD→ARS conversion has near-zero principal flow | two legs linked by fixture `fx_trade_id`: 100 USD out, 120,000 ARS in at 1,200; optional explicit fee | projected treasury principal net 0 within approved tolerance; gross activity 200 shown only as gross; fee separately classified; operating income/OPEX/funding/draws unchanged |
| One-sided FX is not income | only ARS proceeds leg | gross treasury/liquidity inflow only, no economic income/net or invented pairing status; projected operating result unchanged |
| Unlinked FX is not guessed into pairs | two same-day trades without link ID | gross row visibility only; no guessed pair, completeness label, computed spread, or zero-net claim |
| Cash stock policy separation | same native close with transactions at multiple historical rates and a known month-end rate | transaction-flow projection differs from close valuation; only snapshot policy value is labeled cash stock; internal/inferred/frontend-safe flags unchanged |
| Debt stock/activity separation | ARS principal and repayment across changing rates plus end balance | activity uses transaction dates; open balance uses snapshot close; projected principal + interest = total; native allocation/status outputs identical |
| Drilldown reconciliation | mixed native/ARS/USD semantic fixture, including unavailable and FX rows | displayed projected aggregate equals sum of contributing `amount_usd_ccl`; row IDs, native values/currencies, applied rate metadata and status visible; residual within tolerance; incomplete cells unavailable |
| Shadowed FX helper repaired | compact/all-measure rows with `measure`, `metric`, blank and unsupported values | one definition only; no blank default to `net_amount`; selected detail measure matches displayed measure or cell is unsupported |

The FX zero-net test proves a **policy-rate treasury bridge**, not realized accounting gain/loss. Execution price versus CCL reference and explicit spread require source evidence and an approved rule.

## H. Implementation sequence

### PR 1 — fixture-only valuation sidecar contract

Define the versioned rate artifact and `ledger_valuation_usd_ccl.csv` contracts,
then add a fixture-only generator which reads an existing canonical ledger
without rewriting it. Validate join-key uniqueness, emit one status-bearing row
per source row (including unavailable/unsupported rows), and write a manifest
with separate canonical-ledger, native-business, rate-snapshot, resolved-policy,
code, and valuation identities plus exclusive match counters. Reject duplicate
or invalid observations, URLs, mutable `latest` provenance, hash mismatches, and
conflicting rerun overwrites. Do not create a report mode, change ingest,
activate live rates, or perform network lookup.

Independently reversible: removing the new sidecar artifact/module leaves the
canonical ledger and every native consumer byte-for-byte untouched.

### PR 2 — projected semantic eligibility and gross treasury bridge

Add fixture-only semantic eligibility/leakage contracts, projected operating and funding/distribution flow artifacts, completeness propagation, and gross Treasury FX conversion-in/out/explicit-cost lines. Resolve the canonical `amount` sign contract and FX classification precedence through approved fixtures; do not normalize signs or precedence by guess. Emit no projected `FX economic net`, pairing status, computed spread, or completeness-of-trade claim. Separately eliminate the duplicated drilldown helper before relying on projected drilldowns. Do not add cash/debt stock valuation.

Independently testable: projected monthly lines reconcile to row valuation; all existing native semantic artifacts remain golden-equal.

### PR 3 — two basis-fixed management artifacts and dedicated drilldown

After the sidecar and semantic-eligibility contracts exist, add
`monthly_management_usd_ccl_components.csv` plus
`monthly_management_usd_ccl_metrics.csv` containing exactly the two
Matías-approved flow figures. Add a dedicated projected drilldown index/detail
CSV with complete transaction membership and rate provenance. Do not change
frontier, human/professional reports, native drilldowns, publication, or latest.
Register the metrics in frontier only in a later PR after `valuation_basis`,
valuation currency/policy, completeness, and basis-aware primary keys are part
of both contract and series schemas. Consider a transversal `reporting_mode`
only after multiple consumers require it and stock policies exist.

## Decisions still required from accounting authority

1. CCL source series/instrument, quote (bid/ask/mid/close), cutoff/timezone, correction/version policy, and redistribution rights.
2. Weekend/holiday carry-back and maximum staleness.
3. Whether transaction date or settlement date is authoritative when both exist.
4. Stable `fx_trade_id`/pair evidence and tolerance for a fully represented trade.
5. Treatment/name of residuals (spread, fee, policy-rate difference) without auto-reclassification.
6. Which management flow lines enter v1; specifically whether funding/support belongs in a combined management result or only a bridge.
7. Later close-rate conventions for validated cash, inferred cash, debt principal, accrued interest, and dates with no market observation.
8. Rate-series authority, quote/date meaning, correction/revision handling, precision/rounding, snapshot retention, external acquisition owner, and whether content-derived `valuation_id` accompanies operational `run_id`.
9. Exact two management figures, formulas/component allowlists, projected metric IDs, reporting scope, incomplete-cell disclosure, and whether explicit FX costs affect either figure.

## Completion record

```text
Changed: Added this investigation/decision packet only.
Accounting rule changed: None.
Fixture/test evidence: Existing fixture-safe tests inspected; no new fixture because implementation is explicitly deferred.
Commands run: Repository/code searches, focused source inspection, documentation checks listed in the commit report.
Run ID: N/A (no pipeline run).
Outputs inspected: Source-controlled code, tests, fixtures, and notes only; no generated accounting output inspected.
Live inputs accessed: No.
Publication performed: No.
Totals/invariants checked: Code-path and test-contract trace only; no live totals claimed.
Blocked accounting decision: Join key/population, rate series and quote/date meaning, fallback/staleness, revisions, precision, fingerprint/policy/code identity, partials, retention, acquisition authority, and valuation ID require explicit approval.
Next bounded action: Approve Engineer 1 identity/cardinality and Engineer 3 reproducibility decisions, then implement fixture-only PR 1.
```

## Tomorrow recommendation

**If we were to implement this tomorrow, the first PR should be a fixture-only `ledger_valuation_usd_ccl.csv` sidecar and valuation-manifest contract because CCL is a reproducible interpretation of canonical transactions, not part of transaction identity; its exact accounting invariants and acceptance tests are: both canonical ledger files remain byte-for-byte and schema-identical; their artifact SHA and native-business fingerprint do not change under rate correction; the source join key is proven unique or valuation fails closed; the sidecar has exactly one status-bearing row per source row with empty anti-joins; native USD is exactly 1:1; ARS follows the approved synthetic policy; missing/invalid/stale rates remain NA rather than zero; ledger, rate, and valuation hashes are independently recorded; identical inputs reproduce identical sidecar bytes; and no live lookup, semantic report, native pipeline change, or publication occurs.**
