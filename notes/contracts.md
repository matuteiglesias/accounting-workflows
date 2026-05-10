
Purpose: Build canonical ledger dataframe and write ledger_canonical.csv (plus preserve anomalies).
Command: python -m src.accounting.ingest --fixture <csv> --out-dir <dir> [--require-tx-id]
Inputs: fixture CSV (preferred) OR Google Sheet URL + service account JSON.
Flags: --fixture, --out-dir, --sheet-url, --service-account, --sheet-name, --require-tx-id.
Env: FIXTURE, OUT_DIR, SHEET_URL, SERVICE_ACCOUNT (fallbacks).  (live mode requires sheet+SA)
Required files: fixture CSV must exist when provided; service account JSON must exist in live mode.
Outputs: <out-dir>/ledger_canonical.csv (canonical columns + stable ordering).
Artifacts: ledger_canonical.csv (and in-memory anomalies df in df.attrs["anomalies"]).
Manifest fields: N/A (this step does not write manifest; downstream materialize does).
Invariants: canonical_df.attrs["anomalies"] exists (possibly empty); Date coerced; amount present or flagged.
Failure modes: missing fixture path; missing sheet creds; parse/coercion creates NaT/NaN; tx_id missing if required.
Observability: logs include rows/anomalies count; exit !=0 on fatal config/import errors.
Log lines/counters: "Built ledger_base rows=%d anomalies=%d" (rows, anomalies). 


Purpose: Produce “public” CSV artifacts + partitions.json + manifest.json from ledger_canonical.csv.
Command: python -m src.accounting.materialize --out-dir <dir> --freq {W|M} --force {0|1}
Inputs: <out-dir>/ledger_canonical.csv (must exist).
Flags: --out-dir, --freq, --force.
Env: OUT_DIR, FREQ, FORCE (defaults).
Required files: ledger_canonical.csv; optional loan register (currently None in CLI path).
Outputs: ledger_canonical.csv (copy), per_flow_time_long.freq=<freq>.csv, per_party_time_long.freq=<freq>.csv,
         loans_time.freq=M.csv, daily_cash_position.csv, partitions.json, manifest.json, anomalies.csv (if any).
Artifacts: all CSVs written atomically; JSON files updated atomically.
Manifest fields: generated_at, aggregates{file->{path,rows,sha256}}, partitions{freq,last_materialized_at,last_period_end,outputs}, anomalies{...}|None.
Invariants: aggregates entries include sha256; partitions.json updated; manifest written even if some substeps fail (best-effort).
Failure modes: missing ledger file; per_flow/per_party exceptions; schema drift in ledger breaks aggregators; partial outputs if crash mid-run.
Observability: logs for each write; logs exceptions per substep; prints/check logs exist (should be tamed later).
Log lines/counters: "Writing ledger_canonical", "Materialization complete. Manifest: ...", rows per artifact.


Purpose: Generate higher-level report artifacts from materialized CSV folder (optionally validate totals).
Command: python -m src.accounting.reports -o <out_dir> -f <freq> [-w <write_dir>] [--parties ... | --top N] [--no-validate]
Inputs: Materialized CSVs in <out_dir> produced by materialize step; freq label must match filenames.
Flags: --out-dir/-o, --freq/-f, --write-dir/-w, --parties (space list or "A,B"), --top, --no-validate, --pretty-json.
Env: none required (CLI args dominate).
Required files: at least ledger_canonical + per_party_time_long (for top parties derivation) + other report dependencies.
Outputs: report files written under write_dir; JSON summary printed to stdout.
Artifacts: report CSVs + metadata summary; validation block when enabled.
Manifest fields: report summary JSON includes outputs + (optional) validation summary.
Invariants: when parties omitted, derives top parties by abs total movement on 'amount'; returns empty parties if missing/empty.
Failure modes: missing materialized files; freq mismatch; schema drift (missing amount/party); validation fails.
Observability: prints JSON summary; logs/exit code should be non-zero on fatal errors.
Log lines/counters: derived parties count; produced file count; validation pass/fail counts.

