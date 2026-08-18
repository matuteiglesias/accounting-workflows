# USD/CCL Engineer 2 review — accounting systems and FX semantics

**Status:** investigation only; no implementation or accounting-rule change.
**Mandate:** determine what “convert flows to USD” can safely mean when the ledger contains currency exchanges.

## Decision

The hypothesis survives, but only in a narrower form:

> Pairing is not required for a safe v1 Operating USD/CCL view, a separate
> funding/distributions bridge, and a **gross** Treasury FX bridge. Pairing is
> required before claiming trade completeness, execution spread, realized FX
> result, or an `FX economic net`.

Two more immediate blockers were found:

1. ingest documents `amount` as signed, while semantic marts use it as a
   positive magnitude selected into `amount_in`/`amount_out` by direction;
2. FX classification is not a hard boundary because rent, OPEX, funding and
   debt rules run before FX recognition.

Thus “keep `treasury_fx` out of operating result” is necessary but not
sufficient until row-level leakage and amount-direction contracts are approved.

## 1. Invariant being protected

1. Native `amount` and `Currency` remain unchanged; direction, semantic
   classification, and valuation remain separate concepts.
2. Projected operating reporting contains only operating revenue and true
   property OPEX; conversion principal never enters revenue, OPEX, funding,
   draws, debt, or operating result.
3. Funding/distributions remain a separate projected bridge.
4. Treasury v1 may show gross conversion inflows, gross conversion outflows,
   explicitly classified FX-cost rows, and valuation completeness.
5. V1 does not claim economic net, realized result, spread, or pair completeness.
6. Unpaired/ambiguous FX rows never become economic income merely because their
   projected USD amount is positive.
7. Native ARS and USD are never summed before explicit valuation.
8. Classification never implies ownership, entitlement, or legal meaning.

## 2. Current behavior found in code

### 2.1 `amount` has contradictory declared and practical semantics

The ingest docstring calls `amount` a signed native-currency amount.
Canonicalization preserves the source sign and derives `amount_cents` directly;
it does not take an absolute value.

The semantic mart instead derives direction from the Box-relative parties:

- receiver equals Box party → `in`;
- payer equals Box party → `out`;
- both → `internal`;
- otherwise a semantic rule may provide a fallback direction.

It then copies the original value into `amount_in` or `amount_out` and computes
`net_amount = amount_in - amount_out`. Only `amount_abs` calls `abs`.

Therefore the honest current answer is:

> The declared contract says signed amount, but executable semantic behavior
> and fixtures rely on a normally nonnegative magnitude whose direction comes
> from payer/receiver or a semantic fallback.

For an outbound canonical row with `amount=-100`, current behavior can produce
`amount_out=-100` and `net_amount=+100`, reversing the expected semantic sign.
An agent must not repair this with `abs` without an approved accounting rule.

### 2.2 FX recognition is heuristic, not trade-linked

FX conversion recognition uses `Cambio:FX`, `payer=FX`, `receiver=FX`, or
weaker cambio/FX text. `receiver=FX` normally becomes
`fx_conversion_outflow`; the other recognized direction becomes
`fx_conversion_proceeds`. FX cost uses textual evidence such as cost, spread,
commission, or loss.

There is no demonstrated canonical `fx_trade_id`, execution-rate field, paired
transaction ID, or trade-quantity contract. Consequently the current evidence
can support row classification and gross box-relative direction only. It cannot
support claims about which legs belong together, trade completeness, actual
execution spread, or realized gain/loss.

### 2.3 Current operating formulas already provide a useful boundary

The monthly statement computes by native currency:

```text
operating revenue = amount_in where bucket=operating_revenue
property OPEX     = amount_out where bucket=property_opex
net operating     = operating revenue - property OPEX
funding           = amount_in where bucket=funding_contribution
draws             = amount_out where bucket=family_withdrawal_candidate
```

Treasury FX is emitted separately as conversion in, conversion out, cost, and a
current native `treasury_fx_net`. Caveats say conversions alter liquidity but
are not operating income or funding. Static QA checks that statement filters do
not name Treasury FX.

### 2.4 Existing fixtures prove one-sided visibility, not economic netting

Professional drilldown fixtures contain ARS proceeds rows such as `FX → PM`
with positive `amount_in` and no corresponding USD leg. They prove visibility
and drilldown wiring, but not a paired conversion, zero projected principal,
execution spread, or protection against projected double counting.

## 3. Hidden coupling and failure modes

### 3.1 FX classification is not fail-closed

The classifier returns rent, taxes, services, maintenance, legal, contribution,
loan, repayment, or interest classifications **before** it tests FX evidence.
Examples under current precedence include:

| Row evidence | Current risk |
|---|---|
| `Cambio:FX` plus `Cobros/Renta` | may become operating revenue |
| `receiver=FX` plus `Tipo=Impuestos` | may become property OPEX |
| `payer=FX` plus `Flujo=Contribucion` | may become funding |
| FX markers plus debt type | debt early return suppresses FX classification |

Some precedence may be legitimate, but repository evidence does not authorize
an agent to choose. Existing QA examines statement filter strings; it cannot
detect an FX-principal row already misclassified into another bucket.

### 3.2 Amount-sign ambiguity affects every bridge

Negative values can invert operating revenue, OPEX, funding, draws, conversion
in/out, and costs because the semantic measures preserve source sign. Projected
reporting must fail closed on sign/direction contradictions until Matías defines
the canonical contract.

### 3.3 Independent projection is valid for gross visibility, not economics

For USD 100 out and ARS 120,000 in at 1,200 ARS/USD, independent valuation
correctly shows gross projected out 100 and gross projected in 100. Gross
activity can be 200 if explicitly labeled. It does not prove a paired trade or
economic result.

Pairing is unnecessary to show the two gross lines. It is necessary to say the
trade is complete, net is zero, a residual is spread, or a gain/loss was
realized. A monthly subtraction across all FX rows can mix unrelated trades,
boxes, dates, missing legs, and settlement timing.

### 3.4 `treasury_fx_net` overclaims after common-currency projection

The current native line is partitioned by `Currency`, so it is a per-currency
liquidity movement. Once all legs are projected to USD, “FX net treasury effect”
looks like an economic result. V1 must not create that projected line, even if
the native compatibility output remains.

### 3.5 FX cost and completeness are limited claims

Textual `fx_cost_or_spread` classification supports “explicitly classified FX
cost rows,” not a computed execution spread. Missing projected rows also require
line-level valued/missing counts and an incomplete status; a successful partial
sum must not appear complete.

### 3.6 Drilldown currently has a measure-selection defect

`_fx_treasury_measure_for_row` is defined twice. The shadowing version can
default compact FX rows to `net_amount`. Gross projected bridges cannot rely on
that machinery until the duplicate is repaired and each gross line selects its
explicit measure.

## 4. Disagreements with the existing packet

1. Pairing and a near-zero trade test should not block projected operating,
   funding/distribution, or gross Treasury visibility.
2. The existing paired near-zero fixture belongs to a later trade-reconciliation
   capability, because no current source contract supplies linkage.
3. Classifier leakage is a more immediate risk than absent pairing; row-level
   FX evidence can be consumed by earlier rent/OPEX/funding/debt rules.
4. The packet understated the contradiction between signed `amount` documentation
   and magnitude-plus-direction execution.
5. Projected `FX economic net` must be explicitly removed from v1. Native
   `treasury_fx_net` may remain unchanged only as compatibility output.

## 5. Recommended architectural/accounting choice

```text
Operating USD/CCL view
    projected operating revenue
  - projected true property OPEX
  = projected operating result

Funding / distributions bridge
    projected contributions
  - projected draws/distributions

Treasury FX bridge
    projected conversion inflows       # gross
    projected conversion outflows      # gross
    projected explicit FX-cost rows    # separately classified
    valuation completeness             # counts/status
```

Do not emit projected economic net, computed spread, paired-trade count, or
complete-conversion count.

Projection should join after native semantic classification, retain native
`Currency`, and carry `valuation_basis=usd_ccl`. Semantic classification remains
the selector, but v1 eligibility must be fail-closed for FX-marker leakage,
sign/direction contradictions, and incomplete valuation. Never infer pairs from
date, amount, Box, parties, or reciprocal projected values.

## 6. Minimum PR boundary

The future Engineer 2 implementation boundary is fixture-only semantic
eligibility—not a live report:

1. synthetic rows for rent, OPEX, funding, draws, ARS proceeds, USD outflow,
   explicit FX cost, one-sided/ambiguous FX, and positive/negative amounts;
2. an explicit projection-eligibility map for operating,
   funding/distribution, gross Treasury, and excluded/review rows;
3. row-level leakage validation for competing FX and semantic markers;
4. no automatic pairing and no projected FX net;
5. no live rates, network lookup, cash/debt stock valuation, or publication;
6. no change to canonical ledger or existing native formulas/outputs;
7. unresolved sign and precedence cases fail closed until approved.

## 7. Tests that must fail before and pass afterward

| Test | Required evidence |
|---|---|
| amount-direction characterization | outbound `amount=-100` cannot silently become `net_amount=+100`; final behavior awaits approved sign contract |
| FX precedence matrix | competing FX+rent/OPEX/funding/debt markers follow an explicit approved outcome or become review-required |
| clean operating isolation | adding any FX principal leaves projected revenue, OPEX, and net operating unchanged |
| funding/distribution isolation | adding Treasury rows leaves projected contributions and draws unchanged |
| one-sided proceeds | gross conversion in only; no revenue, funding, pairing status, or economic net |
| unlinked two-leg conversion | gross in=100, gross out=100; optionally gross activity=200; no paired/complete/net claim |
| future linked trade | only a later PR may assert near-zero principal under approved link/rate/tolerance rules |
| missing valuation | line is unavailable or unmistakably partial; valued/missing counts reconcile; never zero-filled |
| native currency isolation | native outputs stay by `Currency`; only explicit valuation measures combine currencies |
| projected drilldown | native amount/currency, direction evidence, bucket, valuation/rate/status shown; gross line selects exact measure |
| native regression | native statement and native `treasury_fx_net` remain unchanged if compatibility is retained |

## 8. Decisions that genuinely require Matías

1. Is canonical `amount` signed or a nonnegative magnitude? What happens when
   sign and party-derived direction disagree?
2. What precedence applies when FX evidence competes with rent, OPEX, funding,
   personal, or debt evidence? Which cases become `review_required`?
3. Does `fx_cost_or_spread` mean explicit fees only, separately recorded losses,
   or something broader? May embedded spread ever be computed?
4. Approve the v1 presentation: operating result, funding/distribution bridge,
   gross Treasury in/out, explicit cost rows, and no economic net.
5. Must an incomplete projected aggregate be NA, or may an unmistakably partial
   value be shown, and in which reports?
6. Does native `treasury_fx_net` remain for compatibility while projected net is
   prohibited?
7. Will the source eventually supply `fx_trade_id`, who may assign/correct it,
   and may one trade contain multiple principal and fee legs?
8. For a future near-zero assertion, which rate basis and tolerance apply, and
   what accounting meaning—if any—does the residual have?

## Evidence map

- Declared amount contract and sign-preserving parse:
  `accounting/ledger/ingest.py`.
- Direction, semantic precedence, amount-in/out, native statement formulas and
  QA: `accounting/marts/semantic.py`.
- One-sided FX and positive-magnitude fixtures:
  `tests/test_professional_drilldowns.py` and
  `tests/test_semantic_funding_dimensions.py`.
- Current projected-metric placeholders/native FX net contract:
  `accounting/metrics/frontier.py` and `accounting/metrics/annual.py`.
- Shadowed FX measure resolution and statement drilldowns:
  `accounting/professional/drilldown.py`.

## Completion record

```text
Changed: Engineer 2 FX-semantics decision packet; original investigation narrowed to gross Treasury v1 without pairing/net claims.
Accounting rule changed: None.
Fixture/test evidence: Existing fixtures and tests inspected; proposed characterization/contract tests only.
Run ID: N/A.
Outputs inspected: Source-controlled code, fixtures, tests, and notes only.
Live inputs accessed: No.
Publication performed: No.
Totals/invariants checked: Static formula/fixture trace only; no live totals claimed.
Blocked accounting decision: Amount sign, FX precedence, cost meaning, incomplete aggregation, compatibility net, future link/rate/tolerance.
Next bounded action: Matías approves sign and FX-precedence contracts before any projected semantic-flow PR.
```
