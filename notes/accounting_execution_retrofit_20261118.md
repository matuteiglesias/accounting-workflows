# Product-driven accounting execution: retrofit checkpoint (2026-11-18)

**Status:** architecture retrofit in progress. No scenario formula engine or scenario runner has been deployed. Existing production accounting semantics and historical reports are unchanged.

## Supported commands and boundaries

- `make run-compute`: performs **live** ingest, materialization, debt/treasury, metrics and reports on one exact run; does **not** advance latest or publish. It **must not** be used for simulated inputs because it fetches the real ledger.
- `make run-full`: calls run-compute, then advances latest and publishes the release. This preserves the existing operator workflow, but makes the release step explicit.
- `make run-materialize / run-debt / run-metrics / run-reports RUN_ID=...`: independent offline-stage commands; **no live ingest** and no publication on their own.
- `accounting.support.isolated_execution.isolated_accounting_execution()`: small temporary-workspace context and a command allowlist admitting only the four offline-stage targets. It supplies per-execution OUT and exact RUN_ID, can explicitly export selected files, and cleans up the workspace on normal exit or exception. It **is not an OS sandbox** and is not yet a user-facing runner.
- The isolated context currently accepts only scope FBPM. This is intentional; report closure/release contracts do not yet handle a simulated horizon independently of a real cutoff.

## Product ownership and dependencies

1. Stage A and D already append artifact records to the existing `meta/artifacts.jsonl`; metrics have `build_manifest.json`; report bundles have `report_catalog.json` and per-product report manifests.
2. The three mart CLIs for debt, treasury and accountability now append their outputs to that **same artifact log**. Identical reruns add nothing; changed replay outputs append a new checksum revision from the same producer; a conflicting owner fails.
3. `accounting.diagnostics.report_product_ownership` inventories observed reports using these existing manifests, verifies exact-run identity and hashes, and identifies gaps in registered producers.
4. A core report regression test captures actual renderer path arguments and requires that every existing file consumed by the renderers is listed in its report manifest. Report pack QA may declare extra gate sources.
5. The 2026-11-18 historical run reported 26 reports, 20 distinct declared input artifacts, eight sources not attributed in the then-existing stage log. **That historical run has NOT been backfilled.** Verify the gap closes on a new run before treating ownership P0 as complete.
6. Report manifests are **observed post-build contracts**, not yet an executable pre-build dependency DAG. Specialized report source views still have mappings distributed across three modules. They are deliberately not duplicated into a new global registry.

## Failure semantics

Stage D's existing contract-checker requires per-flow, per-party, loans, and daily-cash outputs. These four required materializers now **raise on failure** even when an old CSV exists. Optional diagnostic box-balance producers retain fail-soft behavior. Existing cash/semantic builders already fail closed.

A failed compute run never advances latest or publishes because release occurs after the compute prerequisite. A previously written partial file may remain in an exact run after a failed real-world execution; do not treat its existence as a success indicator—require stage success and corresponding checks. For scenario workspaces, the complete temporary directory is discarded on failure.

## Retention categories (policy; no GC implemented)

| Category | Source contract/owner | Retention |
|---|---|---|
| Canonical facts and validated source marts | Stage/metric manifest; artifact_role and source_authority | Retain with the approved real run under existing production policy |
| Release documents and public bundles | Report catalog and report manifests | Publish **only by explicit authorized release action**; preserve accepted release provenance |
| QA/diagnostic evidence | Stage QA, output manifests, checksums, diagnostic_only roles | Retain for real-run audit while useful; **do not delete** merely because reports do not read them |
| Scenario intermediate marts / metrics | Isolated execution context | Temporary; delete after requested product validated/exported or on failure |
| Explicitly saved scenario | Future scenario manifest + exported report(s) | Persist deliberately with formula version, parameters, base exact-run identity and hashes, native currencies, cutoff/horizon, and validations. Pin base data to remain reproducible |
| Abandoned temporary run after abrupt process death | Temporary workspace location | Future age-based cleanup limited to the unique isolated prefix, never the real out tree |

**No destructive pruning** has been performed. The workspace context cleans its owned temporary files, but it cannot magically remove orphaned directories after a machine crash; a bounded age-based cleanup can be added later.

## Remaining gates before scenario implementation

- **P0:** verify newly produced debt/treasury outputs are attributed in artifact manifests, and that every desired report has a complete declared dependency closure. No new catalog should compete with existing source contracts.
- **P0:** integrate a no-hypothesis canonical input into an isolated execution and prove key totals and HTML output match an equivalent real run (minus timestamps / execution provenance). This is not yet demonstrated.
- **P1:** for a selective run, required producers must fail closed and optional diagnostic failures may warn; define dependency completeness before optimizing selective materialization.
- **P1:** retain/export provenance with simulated reports so an HTML or PDF outside the Viewer cannot be misread as actual accounting.
- **P1:** decide the minimal artifact archive package for saved scenarios; avoid automatic copies of every mart.
- **Not now:** generic scenario DSL, new DAG orchestrator, caching, data marts rewrite, edits to Accounting Viewer, default publishing of simulations.

## Verification

Checkpoint after implementation (2026-11-18): **35 focused tests passed** across producer provenance, core report source declarations, isolated execution, required-stage failure propagation and Make target contracts. **`make smoke-core` passed** in a uniquely named `/tmp/accounting-retrofit-smoke-*` directory (309 all-status fixture rows, 306 canonical rows; ingest/materialize QA passed); the workspace was removed on exit. Repository-wide `make validate` passed compilation and contract/schema checks, but its full pytest suite was **deliberately interrupted around 15%** after more than four minutes of costly integration tests; do not describe that full suite as green. No release or `latest` pointers were touched. End-to-end reproduction of a scenario report is NOT yet tested.

- `tests/test_mart_producer_provenance.py`: same-run producer attribution, idempotency/replay revision, conflict detection.
- `tests/test_report_product_ownership_audit.py`: report identity, declared sources, checksums, safety.
- `tests/test_report_publication_spine.py`: report inputs declared; live release vs compute Make wiring.
- `tests/test_isolated_execution_context.py`: tmp lifecycle, allowlist, scoped output, safe explicit export.
- `tests/test_materialize_required_failure.py`: four required producers raise even with stale files.
- `tests/test_readme_command_contract.py`: public Make command contract.
