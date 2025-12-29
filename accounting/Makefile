# Makefile.v2 — Accounting spine (A ingest → D materialize → E reports)
# Design goals:
# - Two modes: smoke (fixture/offline) vs run (live/bounded)
# - Explicit out_dir always
# - Timestamped run outputs (no stale-file illusions)
# - Content checks (not only presence)
# - Wrapper manifests for ingest + reports (materialize already emits manifest.json)

SHELL := /bin/bash
PY ?= python3

# ----------------------------------------
# Paths / import roots
# ----------------------------------------
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))      # folder containing this Makefile
WORKDIR := $(abspath $(ROOT)/../..)                         # repo root containing "src/"
export PYTHONPATH := $(WORKDIR)

# ----------------------------------------
# User-tunable parameters
# ----------------------------------------
OUT ?= out
FREQ ?= W
TOP ?= 6

ACCOUNT_SHEET_URL ?=
RENTALS_SHEET_URL ?=
ACCOUNT_SA ?=

define require_var
	@if [ -z "$($(1))" ]; then \
		echo "[ERROR] Missing required var: $(1)"; \
		echo "        Set it via environment or private/.env (not committed)."; \
		exit 2; \
	fi
endef


# Then, in targets that need Sheets access, add:

# 	$(call require_var,ACCOUNT_SHEET_URL)
# 	$(call require_var,ACCOUNT_SA)


# And replace args like:

# --sheet-url "$(ACCOUNT_SHEET_URL)" \
# --service-account "$(ACCOUNT_SA)" \





# Smoke fixture (override if your fixture lives elsewhere)
FIXTURE ?= /home/matias/RAG_Sync/Accounting/4_Analysis_Workflows/src/fixtures/ledger_fixture.csv

# Live sheet config (RUN mode)
ACCOUNT_SHEET_URL ?= $(ACCOUNT_SHEET_URL)
ACCOUNT_SERVICE_ACCOUNT ?= $(ACCOUNT_SA)
ACCOUNT_SHEET_NAME ?= C. Long Ledger

# Optional explicit parties for reports (RUN boundedness)
# Example: REPORT_PARTIES="PM FB"  (space-separated)
# or:      REPORT_PARTIES="PM,FB" (comma-separated)
REPORT_PARTIES ?=

# ----------------------------------------
# Derived output dirs
# ----------------------------------------
SMOKE_OUT := $(OUT)/smoke/accounting
RUN_STAMP ?= $(shell date -u +%Y%m%dT%H%M%SZ)
RUN_OUT := $(OUT)/run/accounting/$(RUN_STAMP)

SMOKE_REPORTS_DIR := $(SMOKE_OUT)/reports
RUN_REPORTS_DIR := $(RUN_OUT)/reports



ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
OUT_DIR ?= $(ROOT)/../out

export PYTHONPATH := $(ROOT)/../..


# ----------------------------------------
# Help
# ----------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "Accounting spine v2:"
	@echo "  make smoke-accounting         # fixture → ingest → materialize → reports (+checks)"
	@echo "  make run-accounting           # live sheet → ingest → materialize → reports (+checks)"
	@echo ""
	@echo "Per-step targets:"
	@echo "  make smoke-ingest | smoke-materialize | smoke-reports"
	@echo "  make run-ingest   | run-materialize   | run-reports"
	@echo ""
	@echo "Key vars:"
	@echo "  OUT=out  FREQ=W|M  TOP=6"
	@echo "  FIXTURE=$(WORKDIR)/src/fixtures/ledger_fixture.csv"
	@echo "  ACCOUNT_SERVICE_ACCOUNT=...  ACCOUNT_SHEET_URL=...  ACCOUNT_SHEET_NAME='C. Long Ledger'"
	@echo "  REPORT_PARTIES='PM FB'  (or 'PM,FB')"
	@echo ""

# ----------------------------------------
# Guardrails helpers
# ----------------------------------------
define _guard_out_dir
	@test -n "$(1)" || (echo "ERROR: out_dir is empty"; exit 2)
	@test "$(1)" != "/" || (echo "ERROR: refusing to write to /"; exit 2)
endef

# ----------------------------------------
# Meta targets
# ----------------------------------------
.PHONY: smoke-accounting run-accounting
smoke-accounting: smoke-reports
run-accounting: run-reports

# ========================================
# SMOKE MODE (fixture, offline, deterministic-ish)
# ========================================

.PHONY: smoke-ingest
smoke-ingest:
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@echo "[SMOKE][INGEST] fixture=$(FIXTURE) -> out=$(SMOKE_OUT)"
	@echo "$(FIXTURE)"

	@mkdir -p "$(SMOKE_OUT)"
	@$(PY) -m src.accounting.ingest \
		--fixture "$(FIXTURE)" \
		--out-dir "$(SMOKE_OUT)" \
		--require-tx-id
		--service-account "$(ACCOUNT_SERVICE_ACCOUNT)" \ 
		--sheet-url "$(ACCOUNT_SHEET_URL)" \ 

	@$(MAKE) _check_ingest OUT_DIR="$(SMOKE_OUT)" MODE="smoke" FIXTURE="$(FIXTURE)"

.PHONY: smoke-materialize
smoke-materialize: smoke-ingest
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@echo "[SMOKE][MATERIALIZE] freq=$(FREQ) -> out=$(SMOKE_OUT)"
	@$(PY) -m src.accounting.materialize \
		--out-dir "$(SMOKE_OUT)" \
		--freq "$(FREQ)" \
		--force 1
	@$(MAKE) _check_materialize OUT_DIR="$(SMOKE_OUT)" MODE="smoke" FREQ="$(FREQ)"

.PHONY: smoke-reports
smoke-reports: smoke-materialize
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@echo "[SMOKE][REPORTS] top=$(TOP) freq=$(FREQ) -> out=$(SMOKE_REPORTS_DIR)"
	@mkdir -p "$(SMOKE_REPORTS_DIR)"
	@$(PY) -m src.accounting.reports \
		--out-dir "$(SMOKE_OUT)" \
		--freq "$(FREQ)" \
		--write-dir "$(SMOKE_REPORTS_DIR)" \
		--top "$(TOP)" \
		--pretty-json > "$(SMOKE_OUT)/reports_summary.json"
	@$(MAKE) _check_reports OUT_DIR="$(SMOKE_OUT)" MODE="smoke" REPORTS_DIR="$(SMOKE_REPORTS_DIR)"

# ========================================
# RUN MODE (live sheet, bounded, timestamped output)
# ========================================

.PHONY: run-ingest
run-ingest:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -n "$(ACCOUNT_SERVICE_ACCOUNT)" || (echo "ERROR: missing ACCOUNT_SERVICE_ACCOUNT (or ACCOUNT_SA)"; exit 2)
	@test -f "$(ACCOUNT_SERVICE_ACCOUNT)" || (echo "ERROR: service account file not found: $(ACCOUNT_SERVICE_ACCOUNT)"; exit 3)
	@test -n "$(ACCOUNT_SHEET_URL)" || (echo "ERROR: missing ACCOUNT_SHEET_URL"; exit 2)
	@echo "[RUN][INGEST] sheet='$(ACCOUNT_SHEET_NAME)' -> out=$(RUN_OUT)"
	@mkdir -p "$(RUN_OUT)"
	@$(PY) -m src.accounting.ingest \
		--service-account "$(ACCOUNT_SERVICE_ACCOUNT)" \
		--sheet-url "$(ACCOUNT_SHEET_URL)" \
		--sheet-name "$(ACCOUNT_SHEET_NAME)" \
		--out-dir "$(RUN_OUT)" \
		--require-tx-id
	@$(MAKE) _check_ingest OUT_DIR="$(RUN_OUT)" MODE="run" FIXTURE=""

.PHONY: run-materialize
run-materialize: run-ingest
	@$(call _guard_out_dir,$(RUN_OUT))
	@echo "[RUN][MATERIALIZE] freq=$(FREQ) -> out=$(RUN_OUT)"
	@$(PY) -m src.accounting.materialize \
		--out-dir "$(RUN_OUT)" \
		--freq "$(FREQ)" \
		--force 1
	@$(MAKE) _check_materialize OUT_DIR="$(RUN_OUT)" MODE="run" FREQ="$(FREQ)"

.PHONY: run-reports
run-reports: run-materialize
	@$(call _guard_out_dir,$(RUN_OUT))
	@echo "[RUN][REPORTS] freq=$(FREQ) parties='$(REPORT_PARTIES)' top=$(TOP) -> out=$(RUN_REPORTS_DIR)"
	@mkdir -p "$(RUN_REPORTS_DIR)"
	@set -e; \
	if [ -n "$(REPORT_PARTIES)" ]; then \
		$(PY) -m src.accounting.reports \
			--out-dir "$(RUN_OUT)" \
			--freq "$(FREQ)" \
			--write-dir "$(RUN_REPORTS_DIR)" \
			--parties $(REPORT_PARTIES) \
			--pretty-json > "$(RUN_OUT)/reports_summary.json"; \
	else \
		$(PY) -m src.accounting.reports \
			--out-dir "$(RUN_OUT)" \
			--freq "$(FREQ)" \
			--write-dir "$(RUN_REPORTS_DIR)" \
			--top "$(TOP)" \
			--pretty-json > "$(RUN_OUT)/reports_summary.json"; \
	fi
	@$(MAKE) _check_reports OUT_DIR="$(RUN_OUT)" MODE="run" REPORTS_DIR="$(RUN_REPORTS_DIR)"

# ========================================
# CHECKS (content checks + wrapper manifests)
# ========================================

.PHONY: _check_ingest
_check_ingest:
	@$(PY) - <<'PY'
	import json, hashlib
	from pathlib import Path
	import pandas as pd

	out_dir = Path("$(OUT_DIR)")
	mode = "$(MODE)"
	fixture = Path("$(FIXTURE)") if "$(FIXTURE)" else None

	ledger = out_dir / "ledger_canonical.csv"
	assert ledger.exists(), f"missing {ledger}"

	df = pd.read_csv(ledger)

	# Minimal canonical contract: required fields for downstream stages
	required = [
	"tx_id","Date","amount","Currency","payer","receiver","Flujo","Tipo"
	]
	missing = [c for c in required if c not in df.columns]
	assert not missing, f"missing columns: {missing}"

	n = len(df)
	assert n > 0, "ledger_canonical.csv is empty"
	assert n < 5_000_000, f"suspiciously large n={n}"

	# require-tx-id invariant
	assert df["tx_id"].notna().all(), "null tx_id exists"
	assert df["tx_id"].is_unique, "duplicate tx_id exists"

	anoms_path = out_dir / "anomalies.csv"
	anoms_rows = 0
	if anoms_path.exists():
		anoms_rows = len(pd.read_csv(anoms_path))

	# Input hash:
	# - smoke: hash fixture (if present)
	# - run: hash ledger output (proxy for sheet snapshot)
	if mode == "smoke" and fixture and fixture.exists():
		input_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
	else:
		input_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()

	manifest = {
	"stage": "A.ingest",
	"mode": mode,
	"input_hash": input_hash,
	"outputs": {
		"ledger_canonical": {
		"path": str(ledger),
		"rows": n,
		"sha256": hashlib.sha256(ledger.read_bytes()).hexdigest()
		},
		"anomalies": {
		"path": str(anoms_path),
		"rows": anoms_rows
		} if anoms_path.exists() else None,
	},
	"schema": {"columns": df.columns.tolist()},
	}

	(out_dir / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"[OK] ingest {mode}: rows={n} anomalies={anoms_rows}")
	PY

.PHONY: _check_materialize
_check_materialize:
	@$(PY) - <<'PY'
	import json
	from pathlib import Path
	import pandas as pd

	out_dir = Path("$(OUT_DIR)")
	freq = "$(FREQ)"
	mode = "$(MODE)"

	manifest_path = out_dir / "manifest.json"
	partitions_path = out_dir / "partitions.json"
	assert manifest_path.exists(), f"missing {manifest_path}"
	assert partitions_path.exists(), f"missing {partitions_path}"

	m = json.loads(manifest_path.read_text(encoding="utf-8"))
	assert isinstance(m.get("aggregates"), dict), "manifest missing aggregates dict"
	assert isinstance(m.get("partitions"), dict), "manifest missing partitions dict"

	ledger = out_dir / "ledger_canonical.csv"
	per_flow = out_dir / f"per_flow_time_long.freq={freq}.csv"
	per_party = out_dir / f"per_party_time_long.freq={freq}.csv"
	daily_cash = out_dir / "daily_cash_position.csv"

	for p in (ledger, per_flow, per_party, daily_cash):
		assert p.exists(), f"missing {p}"

	ld = pd.read_csv(ledger)
	pf = pd.read_csv(per_flow)

	# Totals reconciliation (smoke/run): ledger amount sum equals per_flow amount sum within tight tol
	ld_amt = pd.to_numeric(ld.get("amount"), errors="coerce").fillna(0.0).sum()
	pf_amt = pd.to_numeric(pf.get("amount"), errors="coerce").fillna(0.0).sum()

	tol = max(1e-6, abs(ld_amt) * 1e-9)
	assert abs(ld_amt - pf_amt) <= tol, f"totals drift: ledger={ld_amt} per_flow={pf_amt} tol={tol}"

	print(f"[OK] materialize {mode}: ledger_sum={ld_amt} per_flow_sum={pf_amt}")
	PY

.PHONY: _check_reports
_check_reports:
	@$(PY) - <<'PY'
	import json, hashlib
	from pathlib import Path

	out_dir = Path("$(OUT_DIR)")
	reports_dir = Path("$(REPORTS_DIR)")
	mode = "$(MODE)"

	summary_path = out_dir / "reports_summary.json"
	assert summary_path.exists(), f"missing {summary_path}"

	summary = json.loads(summary_path.read_text(encoding="utf-8"))
	assert isinstance(summary, dict), "reports_summary.json not a JSON object"
	assert "outputs" in summary, "reports summary missing 'outputs'"

	csvs = sorted(reports_dir.glob("*.csv"))
	assert len(csvs) >= 1, "no report CSVs produced"

	outputs = []
	for p in csvs:
		b = p.read_bytes()
		outputs.append({"path": str(p), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)})

	manifest = {
	"stage": "E.reports",
	"mode": mode,
	"summary_path": str(summary_path),
	"outputs": outputs,
	}

	(out_dir / "reports_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"[OK] reports {mode}: files={len(csvs)}")
	PY

# ----------------------------------------
# Cleaning (safe-ish)
# ----------------------------------------
.PHONY: clean-smoke clean-runs
clean-smoke:
	@echo "[CLEAN] rm -rf $(SMOKE_OUT)"
	@rm -rf "$(SMOKE_OUT)" || true

clean-runs:
	@echo "[CLEAN] rm -rf $(OUT)/run/accounting/*"
	@rm -rf "$(OUT)/run/accounting" || true





# # Makefile — weekly accounting pipeline (uses CLI modules, no embedded Python)
# PY ?= python3

# # CLI modules (calls python -m <module>)
# INGEST_MODULE ?= src.accounting.ingest
# MATERIALIZE_MODULE ?= src.accounting.materialize

# # overridable run-time vars (export ACCOUNT_* env vars to override)
# # # OUT_DIR ?= ./../out
# ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# OUT_DIR ?= $(ROOT)/../out

# export PYTHONPATH := $(ROOT)/../..

# FIXTURE ?= ./../fixtures/ledger_fixture.csv
# FREQ ?= M
# FORCE ?= 0          # 1 to force materialize full rebuild
# PARTIES ?= FB,PM



# SERVICE_ACCOUNT ?= $(ACCOUNT_SERVICE_ACCOUNT)

# SHEET_URL ?= $(ACCOUNT_SHEET_URL)

# LOG_DIR := logs
# ANOM_DIR := $(OUT_DIR)/anomalies

# # lightweight report targets for accounting reports
# WRITE_DIR ?= $(OUT_DIR)/reports
# TOP ?= 6



# .PHONY: help all weekly ingest ingest-live materialize materialize-force quick clean manifest validate

# help:
# 	@echo ""
# 	@echo "Make targets (lean, CLI-driven):"
# 	@echo "  make ingest            -> run ingest using FIXTURE (writes ledger_canonical.csv)"
# 	@echo "  make ingest-live       -> run ingest against Google Sheet (requires SERVICE_ACCOUNT & SHEET_URL)"
# 	@echo "  make materialize       -> run materialize (reads ledger_canonical.csv -> writes CSV artifacts)"
# 	@echo "  make materialize-force -> run materialize with FORCE=1 (full rebuild)"
# 	@echo "  make weekly            -> ingest + materialize"
# 	@echo "  make quick             -> quick dev run using tests fixture -> out/quick"
# 	@echo "  make manifest          -> show existing manifest.json if present"
# 	@echo "  make validate          -> basic existence checks of key outputs"
# 	@echo "  make clean             -> remove OUT_DIR/*"
# 	@echo ""

# all: weekly

# # Combined weekly run: ingest (fixture) then materialize
# weekly: ingest materialize
# 	@echo "WEEKLY: done. outputs in $(OUT_DIR)"

# # INGEST (fixture mode)
# ingest:
# 	@echo "[INGEST] fixture -> OUT_DIR=$(OUT_DIR)"
# 	@mkdir -p $(OUT_DIR) $(ANOM_DIR) $(LOG_DIR)
# 	@$(PY) -m $(INGEST_MODULE) --fixture "$(FIXTURE)" --out-dir "$(OUT_DIR)" || (echo "ingest failed"; false)

# # INGEST LIVE (Google Sheets mode)
# ingest-live:
# 	@if [ -z "$(SERVICE_ACCOUNT)" ]; then \
# 		echo "ERROR: set SERVICE_ACCOUNT (or export ACCOUNT_SERVICE_ACCOUNT)"; exit 2; \
# 	fi
# 	@if [ ! -f "$(SERVICE_ACCOUNT)" ]; then \
# 		echo "ERROR: SERVICE_ACCOUNT file '$(SERVICE_ACCOUNT)' not found"; exit 3; \
# 	fi
# 	@if [ -z "$(SHEET_URL)" ]; then \
# 		echo "ERROR: set SHEET_URL (or export ACCOUNT_SHEET_URL)"; exit 2; \
# 	fi
# 	@echo "[INGEST-LIVE] sheet -> OUT_DIR=$(OUT_DIR)"
# 	@mkdir -p $(OUT_DIR) $(ANOM_DIR) $(LOG_DIR)
# 	@$(PY) -m $(INGEST_MODULE) --service-account "$(SERVICE_ACCOUNT)" --sheet-url "$(SHEET_URL)" --out-dir "$(OUT_DIR)" || (echo "ingest-live failed"; false)

# # MATERIALIZE (reads ledger_canonical.csv)
# materialize:
# 	@echo "[MATERIALIZE] freq=$(FREQ) -> OUT_DIR=$(OUT_DIR)"
# 	@mkdir -p $(OUT_DIR) $(LOG_DIR)
# 	@$(PY) -m $(MATERIALIZE_MODULE) --out-dir "$(OUT_DIR)" --freq "$(FREQ)" $(if $(filter 1,$(FORCE)),--force)  || (echo "materialize failed"; false)

# # FORCE materialize convenience target
# materialize-force:
# 	@$(MAKE) materialize FORCE=1

# # QUICK dev run using local test fixture
# quick:
# 	@OUT_DIR=./out/quick FIXTURE=./tests/fixtures/ledger_fixture.csv FREQ=W $(MAKE) weekly

# # Show manifest if present
# manifest:
# 	@if [ -f "$(OUT_DIR)/manifest.json" ]; then \
# 		echo "Manifest: $(OUT_DIR)/manifest.json"; \
# 		cat "$(OUT_DIR)/manifest.json"; \
# 	else \
# 		echo "No manifest found at $(OUT_DIR)/manifest.json"; \
# 	fi

# # Basic validate: check canonical ledger + at least per_flow + per_party CSV existence
# validate:
# 	@echo "[VALIDATE] quick existence checks in $(OUT_DIR)"
# 	@if [ -f "$(OUT_DIR)/ledger_canonical.csv" ]; then echo "OK: ledger_canonical.csv"; else echo "MISSING: ledger_canonical.csv" >&2; fi
# 	@if [ -f "$(OUT_DIR)/per_flow_time_long.freq=$(FREQ).csv" ]; then echo "OK: per_flow_time_long.freq=$(FREQ).csv"; else echo "MISSING: per_flow_time_long.freq=$(FREQ).csv" >&2; fi
# 	@if [ -f "$(OUT_DIR)/per_party_time_long.freq=$(FREQ).csv" ]; then echo "OK: per_party_time_long.freq=$(FREQ).csv"; else echo "MISSING: per_party_time_long.freq=$(FREQ).csv" >&2; fi

# clean:
# 	@echo "[CLEAN] rm -rf $(OUT_DIR)/*"
# 	@rm -rf $(OUT_DIR)/* || true


# .PHONY: reports reports-parties clean-reports

# # default: derive top $(TOP) parties and produce fondos + renta_{party}.csv in $(WRITE_DIR)
# reports:
# 	mkdir -p $(WRITE_DIR)
# 	PYTHONPATH=$(PYTHONPATH) $(PY) -m src.accounting.reports --out-dir $(OUT_DIR) --freq $(FREQ) --write-dir $(WRITE_DIR) --top $(TOP)

# # explicit parties: pass comma-separated list, e.g. make reports-parties PARTIES="PM,FB"
# reports-parties:
# ifndef PARTIES
# 	$(error PARTIES variable required, e.g. make reports-parties PARTIES="PM,FB")
# endif
# 	mkdir -p $(WRITE_DIR)
# 	PYTHONPATH=$(PYTHONPATH) $(PY) -m src.accounting.reports --out-dir $(OUT_DIR) --freq $(FREQ) --write-dir $(WRITE_DIR) --parties "$(PARTIES)"

# # quick cleanup of generated reports
# clean-reports:
# 	-rm -rf $(WRITE_DIR)/*


# .PHONY: views plots

# views:
# 	mkdir -p $(OUT_DIR)/views
# 	PYTHONPATH=$(PYTHONPATH) python3 -m src.accounting.views --reports-dir $(OUT_DIR)/reports --write-dir $(OUT_DIR)/views

# plots:
# 	mkdir -p $(OUT_DIR)/figs
# 	PYTHONPATH=$(PYTHONPATH) python3 -m src.accounting.plots --views-dir $(OUT_DIR)/views --out-dir $(OUT_DIR)/figs


# # lightweight smoke target for CI / local quick checks
# .PHONY: smoke
# smoke:
# 	@echo "[SMOKE] creating tmp smoke out and running ingest+materialize+validate"
# 	@TMP_OUT=./tmp_smoke_out
# 	@mkdir -p $(PWD)/tmp_smoke_out
# 	@OUT_DIR=$$TMP_OUT FIXTURE=./tests/fixtures/ledger_fixture.csv FREQ=W FORCE=1 $(MAKE) weekly
# 	@OUT_DIR=$$TMP_OUT FREQ=W $(MAKE) validate
# 	@echo "[SMOKE] artifacts in $$(realpath $$TMP_OUT)"


# # Rebuild whole monthly pipeline (clean + ingest + materialize + reports + views + plots)
# .PHONY: rebuild-monthly
# rebuild-monthly: clean
# 	@echo "[REBUILD-MONTHLY] starting full pipeline (monthly)"
# 	@$(MAKE) ingest-live OUT_DIR="$(OUT_DIR)" FIXTURE="$(FIXTURE)"
# 	@$(MAKE) materialize FREQ=M FORCE=1 OUT_DIR="$(OUT_DIR)"
# 	@$(MAKE) reports WRITE_DIR="$(WRITE_DIR)" OUT_DIR="$(OUT_DIR)" FREQ=M TOP=$(TOP)
# 	@$(MAKE) views OUT_DIR="$(OUT_DIR)"
# 	@$(MAKE) plots OUT_DIR="$(OUT_DIR)"
# 	@echo "[REBUILD-MONTHLY] done. outputs in $(OUT_DIR)"
