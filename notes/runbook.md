Nice — this is tidy territory now. Let’s examine what the system actually produces, how the responsibilities should be split, and then give you a practical Makefile that runs the whole thing end-to-end (with incremental and full-rebuild options). I’ll also state concrete function-level responsibilities for the reporting layer so you can safely remove legacy bits.

# 1) What your system aims to produce (concise)

* Canonical ledger (ledger_canonical.parquet / parquet-like DataFrame) where each row is one transaction and required columns exist: Date, amount_cents (int64), payer, receiver, Currency, Flujo, Tipo, tx_id, plus enrichment columns (normalized_party, fx_rate, base_amount_cents).
* Materialized, partitionable, long-form Parquet aggregates (idempotent):

  * per_party_time_long.freq=<freq>.parquet (TimePeriod, TimePeriod_ts_end, party, role, amount_cents, base_amount_cents?, Currency, Flujo, Tipo, tx_id)
  * per_flow_time_long.freq=<freq>.parquet (TimePeriod, TimePeriod_ts_end, Flujo, Tipo, amount_cents, Currency)
  * loans_time.freq=<freq>.parquet (TimePeriod, TimePeriod_ts_end, lender, receiver, signed_for_lender_cents)
  * daily_cash_position.parquet (Date / as_of, party, balance_cents)
  * ledger_canonical.parquet (optional copy for upcoming payables views)
* Small metadata manifests and anomalies files with sha256, rows, partitions, and anomalies_count.

These artifacts are the single source for the reporting/view layer to read from. The ETL-materialization pipeline must guarantee invariants (amount_cents int64, no unexplained duplicate tx_id, TimePeriod_ts_end present).

# 2) Layer responsibilities (clear boundary)

* Ingest layer (already implemented)

  * Responsibility: read raw sources (CSV, Google Sheet), clean noise, standardize column names, coerce numeric columns, produce `ledger_canonical` DataFrame, call `enrich_canonical()` which guarantees `amount_cents` int64 and `Date` typed. Nothing here writes report CSVs or pivots.
* Materialization layer (new `src/accounting/materialize.py`)

  * Responsibility: pure aggregation and storage. Use small pure primitives from `core_timeseries` (aggregate_by_time, actor_signed_series, compute_loan_series_long). Write partitionable Parquet files and small manifest JSON and anomalies. Expose read helpers (read_aggregate_parquet) and manifest location. Idempotent, supports incremental appends by partition.
* Reports layer (thin, `src/accounting/reports.py`)

  * Responsibility: read materialized Parquet artifacts, perform lightweight pivots and `synthesize_wide_columns` for backward compatibility, convert cents->major-units for CSVs, generate final CSVs (fondos_report.csv, renta_*.csv, loan_PPA.csv, etc.), run validation checks against materialization manifest, write a manifest of outputs. No heavy groupby across raw ledger.
* Views layer (`src/accounting/views.py`)

  * Responsibility: UI representation and framing. Format tables for UI or spreadsheet export (string formatting, localised display of dates, pretty columns). This file may also contain the `transaction_specs` mapping that turns legacy filter names into FILTERS callables. Views should not re-run heavy aggregations.

Recommendation: keep reports and views separate. Combine only the tiny CLI glue if you want one entrypoint. Reports = data transforms + CSV writes. Views = formatting for human presentation / dashboards. That keeps testable boundaries and makes automation reliable.

# 3) What should be in `src/accounting/reports.py` (minimal scope)

* IO helpers: load_materialized_folder(out_dir, freq)
* Converters: _period_to_timestamp_end helper (prefer TimePeriod_ts_end)
* High-level constructors:

  * build_fondos_report_from_materialized(materialized, parties, freq) -> DataFrame
  * build_renta_series_from_materialized(materialized, target, freq) -> Series
  * materialized_daily_cash_position_as_of(materialized, as_of) -> DataFrame
  * rolling_forecast_90d_from_daily(daily_df, days=90) -> DataFrame
  * upcoming_payables_receivables(ledger_df, days=90) -> DataFrame
* Validation: validate_materialization_totals(ledger_df, materialized) -> dict
* CLI: run_write_all(out_dir, freq, parties, write_dir) -> summary (calls constructors and writes CSVs using atomic_write_df)

Keep all heavy work in materialize.py primitives. Those are your invariants: amount_cents integer, TimePeriod_ts_end present, etc.

# 4) Small, defensible choices you’ve already made (and why they are good)

* Materialize long-form aggregates (not wide). Good: re-pivoting is cheap, recomputing full joins is expensive.
* Materialize both native `amount_cents` and `base_amount_cents` when FX is present. Good: avoids float drift.
* Write TimePeriod as ISO string and also write TimePeriod_ts_end as timestamp. Good: parquet friendly, avoids pd.Period portability issues.

(I pulled the config loader for context; see the project config and YAML used to set base_dir, freq, fixtures. Use it for paths and default OUT_DIR).  

# 5) Removal plan: what you can delete from legacy pipeline/reports

* Remove heavy pivot/groupby code that re scans raw ledger across multiple places. Replace calls by:

  * ingest -> produce ledger_canonical
  * materialize -> produce per_party_time_long etc
  * reports -> read materialized artifacts and compute CSVs
* Replace every `Credit X`/`Debit X` ad-hoc generation with `synthesize_wide_columns_from_materialized` performed once in reports for backwards compatibility.

# 6) How to run everything from Makefile (practical)

Below is a Makefile that is opinionated but practical. It:

* uses environment vars to override config values
* supports run: etl (ingest+materialize), reports (generate CSVs), validate, rebuild (force), quick (minimal debug)
* writes outputs to OUT_DIR (defaults to ./out)

Drop this Makefile at repo root as `Makefile`.

```makefile
# Makefile for RAG_Sync_Accounting pipeline
PY := python3
OUT_DIR ?= ./out
FIXTURE ?= $(shell python -c "from accounting.config import load_config; print(load_config().fixture_file)")
FREQ ?= $(shell python -c "from accounting.config import load_config; print(load_config().freq)")
PARTIES ?= PM,FB,MI,Primos,Alejandro
FORCE ?= 0
LOG := logs/pipeline.log

.PHONY: all etl materialize reports validate rebuild clean quick manifest

all: etl reports validate
	@echo "done. outputs in $(OUT_DIR)"

# full ETL run: ingest -> materialize -> write parquet + manifest
etl:
	@echo "Running ETL -> materialize (OUT_DIR=$(OUT_DIR) FREQ=$(FREQ) FIXTURE=$(FIXTURE))"
	@mkdir -p $(OUT_DIR)
	$(PY) ./scripts/run_etl_pipeline.py --out-dir $(OUT_DIR) --freq $(FREQ) --fixture $(FIXTURE) || (echo "ETL failed; see $(LOG)"; false)

# If you have a dedicated materialize CLI, call it here. Otherwise ETL writes aggregates.
materialize: etl
	@echo "Materialization completed (via etl). Check $(OUT_DIR) for parquet files."

# Reports: read materialized data and write CSVs
reports:
	@echo "Generating reports (OUT_DIR=$(OUT_DIR) FREQ=$(FREQ) PARTIES=$(PARTIES))"
	$(PY) ./scripts/run_report.py --out-dir $(OUT_DIR) --freq $(FREQ) --parties "$(PARTIES)" || (echo "reports failed"; false)

# Validate invariants: sums, schema, manifest
validate:
	@echo "Validating materialization and reports"
	$(PY) - <<'PY'
from pathlib import Path
from accounting.reports import load_materialized_folder, validate_materialization_totals
p=Path("$(OUT_DIR)")
m=load_materialized_folder(p, freq="$(FREQ)")
ledger=m.get("ledger_canonical") or None
res=validate_materialization_totals(ledger if ledger is not None else {}, m)
import json, sys
print(json.dumps(res, indent=2))
if not res.get("ok", False):
    print("Validation failed", file=sys.stderr)
    sys.exit(2)
PY

# force full rebuild re-run all steps (clean + all)
rebuild:
	@$(MAKE) clean
	@ACCOUNT_FORCE=1 $(MAKE) all

# quick developer run (small fixture)
quick:
	@OUT_DIR=./out/quick FIXTURE=tests/fixtures/ledger_fixture.csv FREQ=W $(MAKE) all

# manifest: print manifest file for current OUT_DIR
manifest:
	@python - <<'PY'
import json, sys
from pathlib import Path
p=Path("$(OUT_DIR)")/"manifest.json"
if p.exists():
    print(p.read_text())
else:
    print("No manifest at", p)
PY

clean:
	@echo "Cleaning outputs"
	@rm -rf $(OUT_DIR)/* || true

# convenience: show config summary
config:
	@python - <<'PY'
from accounting.config import load_config
print(load_config())
PY
```

Notes about this Makefile

* `scripts/run_etl_pipeline.py` and `scripts/run_report.py` are the actual entry points used. They already exist in your repo; adjust flags if different. If you want, create a small `scripts/materialize.py` that calls the new materialize functions directly for faster iteration.
* Use `FORCE=1 make rebuild` to force full re-materialization and overwrite partitions.
* `PARTIES` is a comma-separated string the CLI `run_report.py` should parse into a list.

# 7) Example CLI flows (what I expect you to run)

* Full run (daily/cron): `make all`
* Just regenerate reports (materialized artifacts already present): `make reports`
* Force full rebuild (corrected ingest or FX rates changed): `make rebuild`
* Validate invariants after run: `make validate`

# 8) Small checklist for correctness after you run

* All materialized Parquet files exist in `OUT_DIR` with expected names:

  * per_party_time_long.freq=<freq>.parquet
  * per_flow_time_long.freq=<freq>.parquet
  * loans_time.freq=<freq>.parquet
  * daily_cash_position.parquet
  * ledger_canonical.parquet
* `manifest.json` exists and lists outputs and sha256 sums
* `validate` returns ok true
* Reports CSVs generated in `OUT_DIR/reports` (or write_dir configured)

# 9) Final recommended tidy-ups (low-effort, high ROI)

* Expose a small `scripts/materialize.py` wrapper that imports `src.accounting.materialize` and runs all materialize functions for a given freq and out_dir. That makes Makefile calls simpler and separates full-ETL from pure aggregation.
* Keep `src/accounting/reports.py` focused on transforms and CSV writing. Keep `views.py` for UI/table templating only.
* Add unit tests for invariants: sum(amount_cents) equality, no NaN TimePeriod, amount_cents dtype int64.
* Persist a small `partitions.json` written by materialize to track which partitions were appended and their last processed date. The materialize code you requested earlier should already implement it.

# 10) Quick decisions I made for you

* Keep reports and views as separate files. Reports is the single CLI entrypoint to export CSVs for other systems. Views remain for front-end formatting or notebook presentation.
* Prefer TimePeriod_ts_end timestamp in materialized data and only use TimePeriod ISO string as a human tag. Reports will use the timestamp for indices.

If you want I can:

* generate the `scripts/materialize.py` wrapper now so Makefile calls it directly, or
* produce the exact `Makefile` as a file in repo (I can paste it here ready to copy), or
* convert your `run_report.py` to accept `--parties` as comma list and show a minimal CLI help block.

Which of those three should I produce right now? (I will not wait — I can produce the code right here in the next message.)
