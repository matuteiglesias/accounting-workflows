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
