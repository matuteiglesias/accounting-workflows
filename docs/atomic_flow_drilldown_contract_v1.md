# Atomic-flow drilldown contract v1

## Scope

`accounting.contracts.atomic_flow_drilldowns` defines immutable, typed
`FlowCellSpec` records for governed atomic semantic flows. It is still a
contract-only surface: no production consumer imports the registry yet.

The v1 contract deliberately excludes:

- cash and debt snapshots;
- debt activity routing;
- derived formulas and quality ratios;
- compatibility fallbacks;
- unsupported-route policy;
- arbitrary Python predicates or callables.

Those families have different selection and fallback behavior and must not be
forced into a premature common `CellSpec`.

## Semantic membership and measure authority

A spec contains a non-empty tuple of `semantic_members`, where every member is
an exact `(semantic_bucket, semantic_subbucket)` pair. Most specs contain one
member. A governed union may contain several members when one professional cell
legitimately combines atomic semantic populations.

`measure_ref` must identify one of the declared members. Construction fails
closed unless:

1. every semantic member resolves through `semantic_measure_registry_v1`;
2. `measure_ref` resolves through the same registry; and
3. every member resolves to exactly the same governed physical measure as
   `measure_ref`.

The contract therefore supports the existing statement total that combines
`family_withdrawal_candidate/*` and `family_withdrawal/*`: both resolve to
`amount_out`, so the union is declarative and governed. A union such as revenue
plus OPEX is rejected because its members resolve to different measures.

The spec never stores `measure`, `amount_in`, `amount_out`, `amount_abs`, or
`net_amount` as fields. A future executor must resolve `measure_ref` at use time.
This keeps `semantic_measure_registry_v1` as the sole atomic-measure authority.

## Declarative boundary

Membership is limited to normalized semantic pairs and a typed grain. Every
grain begins with `period, Currency`; optional dimensions are drawn from the
explicit `FlowGrainDimension` vocabulary. `FlowCellSpec` is frozen and slotted,
the registry is read-only, duplicate members are rejected, and callable fields
are not allowed.

The funding entries remain `funding_contribution`, not generic funding/support.
Direct-obligation and debt-linked support require multi-semantic behavior with
different accounting measures or debt evidence and are intentionally **not**
authorized by the union feature. Equal-measure membership is necessary for a
union, not sufficient reason to add one; registry additions remain reviewed
contract decisions.

## Version and migration rule

The registry remains `atomic_flow_drilldown_specs_v1`. This refinement occurs
before any production consumer has adopted the contract and preserves the same
contract boundary while making legitimate same-measure unions explicit.

Adding a consumer remains a separate migration requiring fixture parity for
membership, matched values, fallback behavior, and reconciliation tolerance.
This change does not modify any professional route, accounting output, or
semantic measure rule.
