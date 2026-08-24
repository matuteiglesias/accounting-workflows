---
id: notes/public_bundle_contract
title: "Accounting Public Bundle Contract"
sidebar_label: "Accounting Public Bundle Contract"
---

# Accounting Public Bundle Contract

Status: current contract
Last reviewed: 2026-08-24

## Purpose

`accounting.publish.latest` packages governed accounting artifacts for downstream consumers. It performs no accounting computation and owns no Flask/frontend runtime.

Canonical command:

```text
make publish-latest
```

Scope-qualified bundle:

```text
public/accounting/latest_<SCOPE_TAG>/
```

## Sources

Publication consumes only the matching latest metrics and debt roots:

```text
out/metrics/latest_<SCOPE_TAG>
out/debt_resolution/latest_<SCOPE_TAG>
```

The resolved run identities must match. `out/human_reports` is not a source.

## Manifest

`manifest.json` uses schema `accounting_public_bundle.v1` and contains source paths, source run identity, published files, metrics metadata, debt metadata, and publication mode. It intentionally has no report/navigation contract.

## Consumer rule

Consumers may use the published bundle or the professional-pack/drilldown surfaces appropriate to their job. They must not treat presentation HTML, legacy reconciliation tables, or raw debt diagnostics as new accounting authority.

The historical Python function name `build_frontend_snapshot_manifest` remains only as an explicitly deprecated external-import compatibility alias. Removal condition: an external-import census confirms no caller remains. The repository itself uses `build_public_bundle_manifest`.
