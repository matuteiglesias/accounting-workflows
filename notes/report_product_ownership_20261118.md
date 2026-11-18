# Report-product ownership: first executable inventory (2026-11-18)

Status: **P0 first slice delivered; producer attribution incomplete.**
Scope: **observed report products and their declared inputs from one exact run**.
This does not supersede pipeline_dag_contract.md, accounting_spine_runbook.md, the artifact authority in accounting/artifacts/manifest.py, or the specialized report specs.

## Command

Run without rebuilding or publishing accounting artifacts:

    python3 -m accounting.diagnostics.report_product_ownership \
      --run-root out/run/accounting/20261118T200115Z_FBPM \
      --metrics-root out/metrics/20261118T200115Z_FBPM \
      --reports-root out/reports/20261118T200115Z_FBPM \
      --output /tmp/accounting_report_product_ownership_20261118.json

**Observed on that exact run:** 26 report products, 20 distinct declared input artifacts, **8 inputs without producer attribution** in the stage/metrics manifests inspected. Input/output hashes and run/cutoff/scope identity passed. No accounting or report products were regenerated.

## Evidence-first ownership model

- Report **owner** is taken from its own report manifest's renderer_version; the report catalog supplies supported product IDs. The audit validates one manifest per report ID, input/output hashes, and unique ownership of the report's output files.
- Source **consumers** are taken from each report manifest's sources, not by scanning arbitrary report files.
- Run-level source **producers** are taken from the stage artifact log (meta/artifacts.jsonl). Metrics producers are obtained from the existing metrics/build_manifest.json.
- **Unattributed** means the observed source is present, valid, and consumed, but does not have a producer in those consulted manifests. It does NOT mean a producer doesn't exist.

### Producer-attribution gaps in this run

| Input artifact | Where its producer is currently wired | Required next step |
|---|---|---|
| run/monthly_debt_position.csv | Makefile run-debt → accounting.marts.debt | Record producer provenance in the existing execution artifact contract |
| run/monthly_debt_position_qa.csv | Same debt stage | Same |
| run/monthly_debt_activity.csv | Same debt stage | Same |
| run/monthly_debt_activity_qa.csv | Same debt stage | Same |
| run/monthly_debt_repayment_detail.csv | Same debt stage | Same |
| run/monthly_cash_accountability.csv | Makefile run-debt → accounting.marts.treasury | Same |
| run/monthly_cash_accountability_qa.csv | Same treasury stage | Same |
| run/family_business_accountability_cycles.csv | Makefile run-debt → accounting.marts.accountability | Same |

This was a **specific instrumentation gap**, not an invitation to create a competing product registry. **Implementation added 2026-11-18:** the three CLI producers now append output identities and hashes via accounting/artifacts/producer_provenance.py into the existing meta/artifacts.jsonl, using stages E.debt_position, E.treasury_accountability and E.accountability_cycles. Duplicate identical registration is idempotent; conflicting run identity/producer or changed hash fails closed. The eight gaps listed above remain in the immutable historical run; new runs should resolve them. No historical file was backfilled.

## Important boundary: declared is not necessarily exhaustive

The audit proves consistency for **manifest-declared** sources. It does not yet prove that every renderer dependency is in its manifest.

In the audited 20261118T200115Z_FBPM snapshot, render_treasury() consumes annual_metrics_path while the treasury report's source manifest does not list annual metrics. **Follow-up code correction:** accounting/reports/build.py now records this dependency; regression coverage was added to tests/test_report_publication_spine.py. Historical manifests remain unchanged, by design. Other missing dependency declarations are still possible; don't yet use manifests as a complete selective-execution DAG.

Likewise, artifacts not directly consumed by reports may still feed other marts, QA, diagnostics, or provenance. Do not delete them from a static nonconsumer list.

## Next bounded work item

**P0 dependency completeness / producer attribution**, not scenarios yet:

1. Make the debt/treasury/accountability stage record its outputs in the existing artifact provenance mechanism (exact run, one producer per output). Preserve current computational formulas.
2. Verify each report renderer's actual file dependencies are reflected in its report manifest, starting with treasury annual metrics.
3. Gate any proposed selective runner on a comparison between declared dependencies and runtime consumption; reject incomplete declarations.
4. Once the ownership picture is complete, proceed with an isolated execution context that is physically incapable of publishing or advancing latest.

This first slice is deliberately **read-only at runtime**: no source ledger edits, no changed report outputs, no publication, no retention changes.

## Regression gate

    python3 -m pytest -q tests/test_report_product_ownership_audit.py

The tests cover a valid fixture with missing attribution, duplicate IDs, mixed runs, checksum drift, missing outputs, path traversal, and the refusal to audit a latest pointer instead of an exact run.
