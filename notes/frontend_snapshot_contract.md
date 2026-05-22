---
id: notes/frontend_snapshot_contract
title: "Accounting Frontend Snapshot Contract"
sidebar_label: "Accounting Frontend Snapshot Contract"
---

# Accounting Frontend Snapshot Contract

Status: current contract
Last reviewed: 2026-05-10

## Purpose

`accounting.publish.latest` creates the frontend-safe handoff bundle for accounting UI/viewer consumers.
The viewer should consume the published snapshot only and should not read arbitrary producer internals under `out/`.

## Canonical producer and path

Canonical command:

```text
python -m accounting.publish.latest --project-root <repo> --clean
```

Canonical Make target:

```text
make publish-latest
```

Compatibility Make target:

```text
make publish
```

The old `python -m accounting.publish_latest` compatibility module has been removed; use `python -m accounting.publish.latest` directly.

Published snapshot path:

```text
public/accounting/latest/*
```

## Manifest

The required manifest lives at:

```text
public/accounting/latest/manifest.json
```

Minimum shape:

```json
{
  "schema_name": "accounting_frontend_snapshot.v1",
  "built_at": "...",
  "source_run_id": "...",
  "status": "ok",
  "source_paths": {},
  "files": [],
  "metrics": {},
  "debt": {},
  "reports": {}
}
```

Compatibility keys such as `surface_id`, `published_at_utc`, `publish_mode`, `run_id`, `as_of_date`, `months`, `include_statuses`, `report`, and `navigation` may also be present for existing consumers.
New consumers should prefer the explicit v1 keys above.

## Consumer rule

Frontend/viewer code should read only from `public/accounting/latest/*` and the manifest paths listed in `manifest.json`.
It should not depend directly on `out/run`, `out/metrics`, `out/debt_resolution`, or `out/human_reports` internal producer directories.
