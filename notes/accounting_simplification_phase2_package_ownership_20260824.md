# Accounting simplification Phase 2 — clean package ownership

Date: 2026-08-24  
Base: `bbd49ddf46991b1e664c8c51b8811956bd1ab401`  
Accounting-policy change: **none**

## Invariant

This phase is movement and ownership cleanup only. It must not change:

- canonical ledger records or scope/cutoff rules;
- semantic classification or funding/support meaning;
- monthly or annual metric values/statuses;
- debt position/activity semantics;
- validated-cash rules;
- professional displayed values;
- professional drilldown membership or reconciliation.

## Ownership rule

`accounting.professional` owns:

- professional table producers;
- professional contracts/adapters;
- drilldown execution;
- rendering of the professional pack / linked digest.

`accounting.diagnostics` owns:

- forensic audits;
- issue digests;
- migration characterization;
- read-only investigations over canonical, semantic, metric, debt, professional-pack, and drilldown artifacts.

Diagnostics may inspect professional outputs. They do not own the values they inspect and must not become an alternate semantic authority.

## Moves

The following source files were moved as identical Git blobs, with no algorithm edits:

| Before | After | Role |
|---|---|---|
| `accounting/professional/funding_lineage_audit.py` | `accounting/diagnostics/funding_lineage.py` | funding/support forensic characterization |
| `accounting/professional/issue_digest.py` | `accounting/diagnostics/professional_issues.py` | drilldown issue digest / QA triage |

The funding-lineage unit test now imports the diagnostic path. There was no repository caller of `accounting.professional.issue_digest` on the Phase-2 base.

Dated audit documents that mention the former paths are intentionally left unchanged as historical evidence.

## Capability preservation

The moved funding diagnostic still exposes `build_audit`, `write_outputs`, and `main` and writes the same audit/summary/HTML/markdown artifact names.

The moved professional-issues diagnostic still exposes `build_issue_rows`, `build_summary_rows`, and `main` and keeps the same issue/summary/HTML artifact contract.

No compatibility shim is retained in `accounting.professional`: retaining forensic modules there under alternate names would defeat the ownership boundary. Repository callers have been migrated instead.

## Regression boundary

`tests/test_phase2_package_ownership.py` requires:

1. the former professional module paths to be absent;
2. both diagnostic module paths to exist and expose their public capabilities;
3. moved diagnostics not to import `accounting.professional` runtime modules.

The existing funding-lineage behavioral fixture remains the before/after semantic check for that diagnostic.

## Validation required before merge

- `make validate`;
- normal repository CI;
- existing funding-lineage behavioral test;
- package-ownership regression;
- diff inspection confirming the two large source files are pure renames/moves rather than semantic rewrites.

No generated accounting artifacts, professional packs, caches, or confidential data should be committed.
