# Semantic measure authority characterization — 2026-08-19

## Scope

This is a fixture-safe characterization of current production behavior. It does
not choose a future authority, change a selector, or declare an accounting
interpretation. The machine-readable matrix is
`docs/semantic_measure_authorities_20260819.csv` and is pinned by
`tests/test_semantic_measure_authorities.py`.

The four observed surfaces are:

1. native monthly operating-statement construction;
2. USD/CCL management-flow selection;
3. professional/drilldown membership and measure selection; and
4. annual/metrics construction.

“Authority” means production code that independently names a bucket/subbucket
measure or independently translates it into another accounting measure. A layer
that only sums an already-governed statement line is still shown in the matrix,
but is labeled `statement_projection:<measure>`.

## Current semantic edit distance

For this baseline:

```text
semantic edit distance = number of production surfaces that independently encode
or translate the concept's selected measure and therefore require review for an
approved semantic-measure change
```

| concept family | current distance | result |
| --- | ---: | --- |
| rent | 2 | native, management, and professional atomic drilldowns share the contract; annual remains local |
| OPEX taxes/services/maintenance/legal | 2 | native, management, and professional atomic drilldowns share the contract; annual remains local |
| funding | 2 | native, management, and professional atomic drilldowns share the contract; annual remains local |
| withdrawals | 2 | native, management, and professional atomic drilldowns share the contract; annual remains local |
| debt principal/repayment | 3 | intentional debt-engine/cash-bridge translation |
| internal transfer | 1 | native visibility; management excludes, professional does not support it, and annual omits |
| known FX proceeds/outflow/cost | 2 | native, management, and professional atomic drilldowns share the contract; annual remains local |
| unknown FX | 3 | management fails closed; other surfaces retain net visibility |
| `review_required` | 3 | QA state with conditional behavior; annual only projects native visibility |

After the native, management, and professional atomic-drilldown migrations, the
repository-wide maximum and modal distance for ordinary flow concepts is **2**,
down from 4. The scoped distance across those three migrated surfaces is **1**;
annual semantic-detail metrics remain an independent authority.

## Agreement and intentional differences

### Full agreement

Rent, the four explicit property-OPEX subbuckets, funding contributions,
withdrawals, and the three approved FX subbuckets agree for their named totals. Annual FX
dimension metrics intentionally aggregate `net_amount` while named annual totals
project the native in/out statement lines; the matrix exposes both behaviors.

### Debt

The native statement exposes debt movement using `amount_abs` for visibility.
Management explicitly excludes debt from the approved USD/CCL component set.
Professional cash-bridge lineage selects signed `net_amount`, while annual debt
metrics use debt-specific `new_principal` and `repayments`. This is recorded as
an intentional current difference, not normalized by this audit.

### Internal transfers

The native statement retains `amount_abs` visibility. Management excludes the
bucket, professional semantic lineage does not support that statement line, and
the annual dashboard does not emit an internal-transfer metric. No missing value
is reinterpreted as zero.

### Unknown FX

The native statement includes unrecognized treasury-FX subbuckets in the net FX
visibility line using `net_amount`. Professional and annual net views preserve
that visibility. Management fails closed with `excluded_not_approved_v1` because
the subbucket has no approved management measure direction.

### Review-required rows

`review_required` is a state rather than a bucket. Native unknown/review
visibility uses `amount_out`, falling back to `amount_abs` if the outflow total is
zero. Professional QA lineage exposes absolute visibility. Annual metrics project
the native statement line. Management only selects a row when its semantic
bucket/subbucket has an approved direction, and then marks the projection
incomplete/review-required; otherwise it fails closed.

## Evidence and limitations

The regression test deliberately checks current production entrypoints/private
selectors rather than copying only the matrix values:

* native measures and numeric behavior come from
  `build_monthly_operating_statement_from_split`;
* management measures come from `_measure_direction` and approved-bucket logic;
* professional measures come from `_semantic_filter_for_statement_line`, the FX
  resolver, and cash-bridge line specifications; and
* annual/metrics measures are pinned to the current annual semantic-detail specs,
  FX statement projection, and debt-specific metric mappings.

The matrix is a change detector, not a future API. Private functions are used
intentionally because this PR characterizes where authority exists today. A
future contract PR must replace these assertions atomically with contract-level
tests and demonstrate unchanged fixture outputs.

```text
Changed: characterization matrix, decision note, and regression test only
Accounting rule changed: no
Fixture/test evidence: tests/test_semantic_measure_authorities.py
Run ID: none
Outputs inspected: fixture-built in-memory/native test outputs only
Live inputs accessed: no
Publication performed: no
Totals/invariants checked: authority agreement matrix and semantic edit distance
Blocked accounting decision: whether current intentional differences belong in the future contract
Next bounded action: seek approval of contract rows before centralizing selectors
```
