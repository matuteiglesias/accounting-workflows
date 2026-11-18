# Accounting scenarios — performance audit and profiling procedure
Date: 2026-11-18
Status: **first bottleneck confirmed, opt-in audit instrumentation implemented; no accounting formula optimization yet.**

## Evidence and confidence

1. **Prior Chino Juan pilot:** roughly 12 minutes end-to-end for two variants. Both kept all standard marts and metrics, but rendered only treasury HTML/PDF. This is an observed end-to-end order of magnitude, not an independently captured per-subprocess timing trace.
2. **Existing exact real run `20261118T210059Z_FBPM`:** file modification stamps give only approximate stage boundaries, not true `perf_counter` measurements. `ledger_canonical.csv` appeared at 18:01:03 local; `meta/stage_D_materialize.json` at 18:02:11; `monthly_cash_accountability.csv` at 18:04:13; metrics manifest at 18:04:58; treasury HTML 18:05:07; full report catalog 18:15:05. These indicate roughly 70 s for materialization, 124 s for debt+treasury, 43 s for metrics, and up to ten minutes for a full bundle of PDFs. Other work and process scheduling can affect wall clock. **Do not treat these as benchmark results.**
3. **Targeted `cProfile` of the existing `accounting.debt.balance_views` on actual exact-run input (125 obligations, 82 repayments/allocations, May 2023–Oct 2026, 1,270 calendar days):**
   - Instrumented process wall clock: **143.85 seconds**; CPU user 69.35 s; peak RSS 123,028 KB. The CPU vs wall gap is a warning about contention/overhead (another accounting process was running). Never compare this wall clock directly with an unprofiled run.
   - `build_debt_balance_daily()` had **137.37 s cumulative profile time**, 18,560,453 function calls across the entire profiled process.
   - Inside the daily loop: **1,271 DataFrame merges**, ~2,544 groupby aggregations, 1,270 Series `reset_index` calls. The implementation filters/aggregates all known debt and repayment state for every day. This is a clear repeated-work hotspot.
   - **Output parity: exact byte equality** for `debt_balance_daily.csv`, `debt_balance_monthly.csv`, `debt_balance_quarterly.csv`, and `debt_balance_yearly.csv` compared to the existing same-run reference. Therefore these four files are suitable regression targets for a future event-sweep/vectorized rewrite.
4. **Make scope overhead:** a dry run with opt-in Python tracing revealed `canonical_scope_tag` command is invoked twice even if `SCOPE_TAG=FBPM` is supplied. The current Makefile uses an eagerly evaluated `$(shell $(PY) -c ...)`; command-line `SCOPE_TAG` does **not** prevent that evaluation. This is a smaller potential cleanup. The dry-run timings were unstable under CPU load, so *no quantified saving is asserted*.
5. **Chromium Snap PDF behavior:** the old scenario workflow failed when the rendered PDF destination lived under host `/tmp`. The supported scenario workspace must be on an accessible non-hidden home path, and deleted afterward. PDF generation took tens of seconds in the first pilot. This is infrastructure latency rather than necessarily a reporting-logic issue.

## Recommended tool split

- **Default lightweight audit: Python standard library `time.perf_counter()` plus a temporary subprocess wrapper**, added here as `accounting.diagnostics.timed_python`. When explicitly enabled, this records one JSONL event per Python command launched by Make, alongside the stage wall times and separate treasury HTML and PDF wall times. No third-party profiling dependency; disabled by default.
- **Deep-call sampling (optional): Pyinstrument** is the best next tool when a module's internal call tree is not obvious. Pyinstrument can output HTML, and is not installed in the current environment. It is *not required* for stage checkpoints. VizTracer is a richer temporal trace option; defer unless necessary.
- **Built-in deterministic profiler: `python3 -m cProfile -o ... -m accounting.<module> ...`** and `pstats`. This was used for the debt hotspot confirmation; profiling adds overhead and must not be misrepresented as production timing.

## Running the next scenario with timing enabled

Only when intentionally launching the full two-variant pilot (it is costly):

```bash
cd ~/repos/accounting-workflows
python3 -m accounting.scenarios.runner \
  --scenario scenarios/chino_juan_no_paga_desde_2026_12.yaml \
  --outbox ~/Desktop/MEMOS/10_accounting_family/08_scenarios/outbox \
  --audit-file ~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/chino_pilot_timing_001.json

python3 -m accounting.diagnostics.summarize_scenario_timings \
  ~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/chino_pilot_timing_001.json \
  --html ~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/chino_pilot_timing_001.html
```

**Outbox default remains HTML/PDF only.** The optional performance JSON and HTML go to a separate diagnostics folder and only when requested; the temporary CSVs and intermediate metrics remain disposable. One JSON is written for the *complete successful run*; failure handling still primarily uses the transient stage log. This limitation should be addressed before relying on the audit as a failure record.

## Read the timing result correctly

- `make_stage`: wall clock for an entire stage, including child process launches and validations.
- `python_subprocess`: one program invoked via `$(PY)` (e.g. `accounting.debt.balance_views`). Nested within the corresponding Make stage; **do not add** stage and child elapsed times.
- `renderer`: separate treasury HTML and PDF times.
- `seed_and_transform`: ledger clone and scenario change overhead.
- `total_seconds`: end-to-end variant and overall run times.

The single-run critical path is sequential; prioritize the largest *exclusive bottleneck*. A subprocess above five seconds does **not** automatically qualify for omission from the accounting pipeline: core calculations and QA remain mandatory.

## Priority-ranked actions after measurement

1. **P0/P1: replace `accounting.debt.balance_views.build_debt_balance_daily` daily repeated pandas operations with a mathematically equivalent event-sweep or vectorized cumulative balance.** First demonstrate exact parity for the 4 reference CSVs and existing tests, including opening/closing dates, allocations, and interest. Keep implementation separate until parity passes. Do not assume that debt can be skipped merely because the requested scenario only changes future rent.
2. **P1: minimize redundant full-run work when inputs are provably invariant.** The current Chino case changes `planeado` data, but the treasury report reads a `pagado` subset. A scenario executor must eventually detect/declare when outputs are unaffected; avoid rebuilding identical marts and metrics in each variant only with explicit dependency/provenance proof.
3. **P1: measure Stage D / metrics internals** with optional Pyinstrument once the debt bottleneck has been addressed; do not optimize on guesses.
4. **P2: Make scope-tag parse overhead** should be addressed only with a validated fast path, preserving the consistency of BOXES and SCOPE_TAG.
5. **P2: PDF generation** may be classed high-resource when measured >5 seconds, but the requested HTML/PDF deliverable remains required, not optional.

## Acceptance gates

- No changes to actual or planned ledger rows.
- Baseline scenario returns identical treasury mart and HTML to exact-run reference.
- Changed scenario only alters the eight planned Chino rent rows; preserved paid-only history remains byte-identical.
- All treasury reports remain generated by the existing HTML/PDF renderers, not a secondary accounting interpretation.
- No writes to production `out`, `latest`, public reports, or real source spreadsheets.
- Default outbox contains only the requested HTML/PDF, and workspaces are removed on success.
