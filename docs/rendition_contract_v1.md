# Formal rendition contract v1

`accounting.rendition.v1` is a private reporting projection over one immutable governed accounting run. Phase 1 defines scope only: it does not render a report, recalculate accounting semantics, determine title, or create legal entitlement.

## Invariant

A rendition specification may select an exact run, period, governed Boxes and reporting properties. It must not create accounting or legal authority. In particular:

- property IDs are reporting identities only;
- property selectors are exact `Lugar x Box` selectors;
- ownership, usufruct, creditor status and percentages remain outside this contract;
- native currencies remain source-driven and ARS/USD are never synthesized by the rendition spec;
- the selected period cannot extend beyond an immutable Stage A cutoff;
- selected Boxes/properties cannot exceed the immutable run scope;
- one rendition binds to exactly one `source_run_id`.

## Private inputs

A private spec may be YAML or JSON:

```yaml
schema: accounting.rendition_spec.v1
rendition_id: RDC-MI-2026-09-30-001
source_run_id: 20260930T000000Z_FBPM
rendidor:
  display_name: Example Rendidor
period:
  from: 2026-01-01
  through: 2026-09-30
properties:
  - PROPERTY_A
boxes:
  - Property Management
recipients:
  - Example Recipient
purpose: private_account_rendition
language: es-AR
```

The private property registry is CSV with this bounded schema:

```text
property_id,display_name,lugar_selector,box_selector,optional_notes
```

Selectors are exact in v1. Wildcards and regex selectors are rejected. Two property IDs cannot resolve to the same normalized `Lugar x Box` selector.

## Validation

```bash
python -m accounting.rendition.validation \
  --spec private/rendition.yaml \
  --property-registry private/property_registry.csv \
  --run-root out/run/accounting/<RUN_ID>
```

A successful command prints a `pass` summary. It computes no accounting values and writes no output artifacts.

## Explicit non-goals of phase 1

Phase 1 does not implement rendering, evidence blobs, digital signatures, approval, delivery, ownership percentages, legal balances, FX conversion, setoff, or consignation. Those belong to later bounded waves and must consume this scope contract rather than bypass it.


## Wave 2 compiler

Wave 2 projects the validated scope into transaction-grain rendition datasets. It does not reclassify the ledger or create a second accounting engine.

Primary sources:

- `box_treasury_transaction_detail.csv` for governed actual Box cash;
- `box_treasury_transaction_detail_qa.csv` as an upstream hard gate;
- `classification_audit.csv` for governed non-cash/direct-payment events;
- `monthly_cash_accountability.csv` only for an optional governed prior Box close.

Outputs under `out/renditions/<RENDITION_ID>/compiled/`:

- `rendition_cash_tape.csv`;
- `rendition_non_cash_events.csv`;
- `rendition_summary.csv`;
- `rendition_opening_basis.csv`;
- `rendition_validation.csv`;
- `rendition_context.json`.

The cash tape is an exact property/period subset of the governed treasury transaction detail. Direct tenant payments, constructive settlements and other explicit non-cash events are kept in a separate annex population and never manufactured as Box cash.

The summary is derived only from those two compiled populations. Cash components remain separated by native currency and source semantic bucket. `activity_result` is a reporting result, not a legal balance.

Mandatory caveat for later rendering:

> El resultado de actividad no constituye por sí mismo determinación de deuda, saldo jurídicamente exigible, titularidad de fondos ni caja física disponible.

### Opening basis

The optional spec block is:

```yaml
opening_basis:
  type: unavailable
```

Allowed values:

- `zero_origin_from_management_start`: emits zero only if the exact run contains no earlier activity for any selected property;
- `governed_prior_close`: reads the latest prior `monthly_cash_accountability` close at **Box × Currency** grain and explicitly refuses to attribute that pooled control balance to a property;
- `verified_external`: reserved for the evidence wave; Wave 2 records the state as pending and does not invent an opening amount;
- `unavailable`: asserts no opening amount.

### Local compile

The governed Make public surface remains unchanged. Invoke the private rendition compiler directly against an existing exact run:

```bash
python -m accounting.rendition.build \
  --spec private/rendition.yaml \
  --property-registry private/property_registry.csv \
  --run-root out/run/accounting/<RUN_ID> \
  --out-base out/renditions
```

The compiler fails closed on upstream treasury QA failures, duplicate transaction identities, mixed cash/non-cash populations, cash arithmetic errors, invalid scope/run/cutoff, and summary reconciliation errors.
