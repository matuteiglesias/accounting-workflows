# Makefile.v3 - Accounting spine
# Official path: run-ingest -> run-materialize -> run-views -> run-metrics -> run-human-balance
# Design goals:
# - Two modes: smoke (fixture/offline) vs run (live/bounded)
# - Explicit out-dir passed to all Python entrypoints
# - Timestamped run outputs (avoid stale-file illusions)
# - Content checks (not only presence)
# - Views consumes Stage D; reports/ is optional legacy input only, never a required stage

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory

PY ?= python3
export PYTHONUNBUFFERED := 1

# ----------------------------------------
# Resolve repo root (assumes Makefile in repo root)
# ----------------------------------------
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(ROOT)

# ----------------------------------------
# User-tunable parameters
# ----------------------------------------
OUT  ?= out
FREQ ?= M
TOP  ?= 10
METRIC_MONTHS ?= 6
RENT_PLACE_COL ?= Lugar
RENT_DETAIL_COL ?= Detalle
FLOW_ROLLUP_GROUPBY ?= Flujo,Tipo,Currency
INCLUDE_STATUSES ?= pagado
NOISE_FLOOR ?= ARS:5000,USD:10

# Smoke fixture (override if your fixture lives elsewhere)
FIXTURE ?= $(ROOT)/fixtures/ledger_fixture.csv

# Live ingest vars (export before running, or use run-env wrapper)
ACCOUNT_SA ?=
ACCOUNT_SHEET_URL ?=
ACCOUNT_SHEET_NAME ?= C. Long Ledger

# ----------------------------------------
# Helpers
# ----------------------------------------
define require_var
	@if [ -z "$($(1))" ]; then echo "ERROR: missing required var: $(1)"; exit 2; fi
endef

define _guard_out_dir
	@if [ -z "$(1)" ]; then echo "ERROR: OUT_DIR empty"; exit 2; fi
endef

# Assert views sanity exists and invariant errors are empty
define _check_views_sanity
	@sanity="$(1)"; \
	test -s "$$sanity" || (echo "ERROR: views_sanity.json missing/empty at $$sanity"; exit 2); \
	$(PY) -c 'import json,sys; d=json.load(open(sys.argv[1],"r",encoding="utf-8")); errs=(d.get("invariants") or {}).get("errors") or []; assert len(errs)==0, "views invariant errors: "+str(errs)' "$$sanity"
endef


# ----------------------------------------
# Derived output dirs
# ----------------------------------------
SMOKE_OUT := $(OUT)/smoke/accounting
RUN_STAMP ?= $(shell date -u +%Y%m%dT%H%M%SZ)

RUN_BASE := $(OUT)/run/accounting

RUN_OUT   := $(OUT)/run/accounting/$(RUN_STAMP)
RUN_REL   := $(notdir $(RUN_OUT))
# Per-run out dir (you probably already have this, keep your existing one)
# RUN_OUT := $(RUN_BASE)/$(RUN_RUN_ID)

# NOTE: keep reports dirs only as anchors for legacy files and for loader heuristics.
SMOKE_REPORTS_DIR := $(SMOKE_OUT)/reports
RUN_REPORTS_DIR   := $(RUN_OUT)/reports

SMOKE_VIEWS_DIR   := $(SMOKE_OUT)/views
RUN_VIEWS_DIR     := $(RUN_OUT)/views

SMOKE_VIEWS_SANITY := $(SMOKE_VIEWS_DIR)/views_sanity.json
RUN_VIEWS_SANITY   := $(RUN_VIEWS_DIR)/views_sanity.json

SMOKE_RUN_ID := smoke
RUN_RUN_ID   := $(RUN_STAMP)

RUN_METRICS_DIR := $(OUT)/metrics/$(RUN_RUN_ID)
RUN_HUMAN_DIR   := $(OUT)/human_reports/$(RUN_RUN_ID)/balance_human_v2

METRICS_LATEST := $(OUT)/metrics/latest
HUMAN_LATEST   := $(OUT)/human_reports/latest



DEBT_CURRENCIES ?= USD
DEBT_REPAYMENT_STATUSES ?= pagado
DEBT_FULL_ONLY ?= 1
RUN_DEBT_DIR := $(OUT)/debt_resolution/$(RUN_RUN_ID)
DEBT_LATEST := $(OUT)/debt_resolution/latest



# RUN_LATEST := $(OUT)/run/accounting/latest
# STORY_LATEST := $(OUT)/storypack/latest

RUN_REL := $(notdir $(RUN_OUT))

.PHONY: _update_latest
_update_latest:
	@echo "[RUN][LATEST] run=$(RUN_REL)"
	@bash -eu -o pipefail -c '\
		link_swap () { \
			base="$$1"; \
			target="$$2"; \
			latest="$$base/latest"; \
			mkdir -p "$$base"; \
			if [ -d "$$latest" ] && [ ! -L "$$latest" ]; then \
				echo "[LATEST] WARN: $$latest is a directory. Moving aside."; \
				rm -rf "$$latest.bak"; \
				mv "$$latest" "$$latest.bak"; \
			fi; \
			tmp="$$base/.latest_tmp"; \
			ln -sfn "$$target" "$$tmp"; \
			rm -f "$$latest"; \
			mv -f "$$tmp" "$$latest"; \
			ls -lah "$$latest"; \
		}; \
		link_swap "$(RUN_BASE)" "$(RUN_REL)"; \
		link_swap "$(OUT)/metrics" "$(RUN_REL)"; \
		link_swap "$(OUT)/human_reports" "$(RUN_REL)"; \
	'	link_swap "$(OUT)/debt_resolution" "$(RUN_REL)";


# ----------------------------------------
# Help
# ----------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "Accounting spine v3:"
	@echo "  make run-ingest"
	@echo "  make run-materialize"
	@echo "  make run-views"
	@echo "  make run-debt                # resolve internal debt artifacts"
	@echo "  make run-metrics"
	@echo "  make run-human-balance"
	@echo "  make run-accounting           # wrapper for the official full path above"
	@echo "  make run-accounting-full      # explicit full chain alias"
	@echo "  make run-downstream-from-ledger # materialize -> views -> metrics -> human (reuse existing ledger_canonical.csv)"
	@echo "  make run-metrics-and-human    # metrics -> human (reuse existing views)"
	@echo "  make run-human-balance-only   # human balance only (reuse existing metrics)"
	@echo "  make smoke-accounting         # fixture path through views only"
	@echo ""
	@echo "Per-step targets:"
	@echo "  make smoke-ingest | smoke-materialize | smoke-views"
	@echo "  make run-ingest   | run-materialize   | run-views | run-metrics | run-human-balance"
	@echo ""	
	@echo "Key vars:"
	@echo "  OUT=out  FREQ=W|M  TOP=6  METRIC_MONTHS=6"
	@echo "  FIXTURE=$(ROOT)/fixtures/ledger_fixture.csv"
	@echo "  ACCOUNT_SA=/path/to/sa.json  ACCOUNT_SHEET_URL=...  ACCOUNT_SHEET_NAME='C. Long Ledger'"
	@echo ""

# ----------------------------------------
# Meta targets
# ----------------------------------------
.PHONY: smoke-accounting run-accounting run-accounting-full run-downstream-from-ledger run-metrics-and-human run-human-balance-only
smoke-accounting: smoke-views
run-accounting: run-accounting-full
run-accounting-full: run-human-balance

run-downstream-from-ledger:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/ledger_canonical.csv" || (echo "ERROR: missing ledger_canonical.csv at $(RUN_OUT). Run make run-ingest first or point RUN_STAMP to an existing run."; exit 2)
	@$(MAKE) _run_materialize_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)"
	@$(MAKE) _run_views_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)"
	@$(MAKE) _run_debt_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)"
	@$(MAKE) _run_metrics_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"
	@$(MAKE) _run_human_balance_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"

run-metrics-and-human:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_VIEWS_SANITY)" || (echo "ERROR: missing views_sanity.json at $(RUN_VIEWS_SANITY). Run make run-views first or point RUN_STAMP to an existing run."; exit 2)
	@$(MAKE) _run_metrics_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"
	@$(MAKE) _run_human_balance_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"

run-human-balance-only:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_METRICS_DIR)/metric_values.csv" || (echo "ERROR: missing metric_values.csv at $(RUN_METRICS_DIR). Run make run-metrics first or point RUN_STAMP to an existing run."; exit 2)
	@$(MAKE) _run_human_balance_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"

# ========================================
# SMOKE MODE
# ========================================

.PHONY: smoke-ingest
smoke-ingest:
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@mkdir -p "$(SMOKE_OUT)"
	@$(PY) -m accounting.ingest \
		--mode smoke \
		--fixture "$(FIXTURE)" \
		--out-dir "$(SMOKE_OUT)" \
		--run-id "$(SMOKE_RUN_ID)"
	@$(MAKE) _check_ingest OUT_DIR="$(SMOKE_OUT)" MODE="smoke" FIXTURE="$(FIXTURE)"

.PHONY: smoke-materialize
smoke-materialize: smoke-ingest
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@$(PY) -m accounting.materialize \
		--out-dir "$(SMOKE_OUT)" \
		--freq "$(FREQ)" \
		--force 1 \
		--mode smoke \
		--run-id "$(SMOKE_RUN_ID)"
	@$(MAKE) _check_materialize OUT_DIR="$(SMOKE_OUT)" MODE="smoke" FREQ="$(FREQ)"

.PHONY: smoke-views
smoke-views: smoke-materialize
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@mkdir -p "$(SMOKE_VIEWS_DIR)"
	@mkdir -p "$(SMOKE_REPORTS_DIR)"  # anchor for loader heuristics / optional legacy files
	@$(PY) -m accounting.views \
		--reports-dir "$(SMOKE_REPORTS_DIR)" \
		--write-dir "$(SMOKE_VIEWS_DIR)" \
		--freq "$(FREQ)" \
		--mode smoke \
		--run-id "$(SMOKE_RUN_ID)"

	test -s "$(SMOKE_VIEWS_SANITY)" || (echo "ERROR: views_sanity.json missing/empty"; exit 2); \
	$(PY) -c 'import json,sys; d=json.load(open(sys.argv[1],"r",encoding="utf-8")); errs=(d.get("invariants") or {}).get("errors") or []; assert len(errs)==0, "views invariant errors: "+str(errs)' "$(SMOKE_VIEWS_SANITY)"; \


	@$(call _check_views_sanity,$(SMOKE_VIEWS_SANITY))
	@$(MAKE) _check_views OUT_DIR="$(SMOKE_OUT)" MODE="smoke"


# ========================================
# RUN MODE (LIVE)
# ========================================

.PHONY: run-ingest
run-ingest:
	@$(call _guard_out_dir,$(RUN_OUT))
	@$(call require_var,ACCOUNT_SHEET_URL)
	@mkdir -p "$(RUN_OUT)"
	@$(PY) -m accounting.ingest \
		--mode run \
		--out-dir "$(RUN_OUT)" \
		--run-id "$(RUN_RUN_ID)" \
		--service-account "$(ACCOUNT_SA)" \
		--sheet-url "$(ACCOUNT_SHEET_URL)" \
		--sheet-name "$(ACCOUNT_SHEET_NAME)"
	@$(MAKE) _check_ingest OUT_DIR="$(RUN_OUT)" MODE="run"

.PHONY: run-materialize _run_materialize_action
run-materialize: run-ingest _run_materialize_action

_run_materialize_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/ledger_canonical.csv" || (echo "ERROR: missing ledger_canonical.csv at $(RUN_OUT)"; exit 2)
	@$(PY) -m accounting.materialize \
		--out-dir "$(RUN_OUT)" \
		--freq "$(FREQ)" \
		--force 0 \
		--mode run \
		--run-id "$(RUN_RUN_ID)"
	@$(MAKE) _check_materialize OUT_DIR="$(RUN_OUT)" MODE="run" FREQ="$(FREQ)"

.PHONY: run-views _run_views_action
run-views: run-materialize _run_views_action

_run_views_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/per_flow_time_long.freq=$(FREQ).csv" || (echo "ERROR: missing materialized outputs at $(RUN_OUT). Run make run-materialize first or use run-downstream-from-ledger."; exit 2)
	@mkdir -p "$(RUN_VIEWS_DIR)"
	@mkdir -p "$(RUN_REPORTS_DIR)"  # anchor for loader heuristics / optional legacy files
	@$(PY) -m accounting.views \
		--reports-dir "$(RUN_REPORTS_DIR)" \
		--write-dir "$(RUN_VIEWS_DIR)" \
		--freq "$(FREQ)" \
		--mode run \
		--run-id "$(RUN_RUN_ID)"


	test -s "$(RUN_VIEWS_SANITY)" || (echo "ERROR: views_sanity.json missing/empty"; exit 2); \
	$(PY) -c 'import json,sys; d=json.load(open(sys.argv[1],"r",encoding="utf-8")); errs=(d.get("invariants") or {}).get("errors") or []; assert len(errs)==0, "views invariant errors: "+str(errs)' "$(RUN_VIEWS_SANITY)"; \


	@$(call _check_views_sanity,$(RUN_VIEWS_SANITY))
	@$(MAKE) _check_views OUT_DIR="$(RUN_OUT)" MODE="run"





.PHONY: run-debt _run_debt_action
run-debt: run-views _run_debt_action

_run_debt_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@$(call require_var,ACCOUNT_SHEET_URL)
	@$(call require_var,ACCOUNT_SA)
	@mkdir -p "$(RUN_DEBT_DIR)"
	@bash -eu -o pipefail -c '\
		args=( \
			--sheet-url "$(ACCOUNT_SHEET_URL)" \
			--service-account "$(ACCOUNT_SA)" \
			--sheet-name "$(ACCOUNT_SHEET_NAME)" \
			--write-dir "$(RUN_DEBT_DIR)" \
			--exclude-household \
			--currencies "$(DEBT_CURRENCIES)" \
			--repayment-statuses "$(DEBT_REPAYMENT_STATUSES)" \
		); \
		if [ "$(DEBT_FULL_ONLY)" = "1" ]; then \
			args+=( --full-only ); \
		fi; \
		$(PY) -m accounting.resolve_internal_debt_v2 "$${args[@]}"; \
		test -s "$(RUN_DEBT_DIR)/debt_open_items.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_allocations.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_repayment_events.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_resolution_timeline.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_status_reconciliation.csv"; \
	'





.PHONY: run-metrics _run_metrics_action
run-metrics: run-debt _run_metrics_action

_run_metrics_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_VIEWS_SANITY)" || (echo "ERROR: missing views_sanity.json at $(RUN_VIEWS_SANITY)"; exit 2)
	@mkdir -p "$(RUN_METRICS_DIR)"
	@bash -eu -o pipefail -c '\
		$(PY) -m accounting.build_metric_values \
			--run-root "$(RUN_OUT)" \
			--out-dir "$(RUN_METRICS_DIR)" \
			--months "$(METRIC_MONTHS)" \
			--rent-place-col "$(RENT_PLACE_COL)" \
			--rent-detail-col "$(RENT_DETAIL_COL)" \
			--flow-rollup-groupby "$(FLOW_ROLLUP_GROUPBY)" \
			--include-statuses "$(INCLUDE_STATUSES)" \
			--noise-floor "$(NOISE_FLOOR)"; \
		test -s "$(RUN_METRICS_DIR)/metric_registry.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_values.csv"; \
		test -s "$(RUN_METRICS_DIR)/validation_report.csv"; \
		test -s "$(RUN_METRICS_DIR)/build_manifest.json"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/income_statement_monthly_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/rent_rollup_by_place_m_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/rent_rollup_by_detail_m_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/flow_type_rollup_m_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/draws_discipline_monthly_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/metric_views_manifest.csv"; \
	'



.PHONY: run-human-balance _run_human_balance_action
run-human-balance: run-metrics _run_human_balance_action

_run_human_balance_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_METRICS_DIR)/metric_values.csv" || (echo "ERROR: missing metric_values.csv at $(RUN_METRICS_DIR)"; exit 2)
	@mkdir -p "$(RUN_HUMAN_DIR)"
	@bash -eu -o pipefail -c '\
		$(PY) -m accounting.human_balance_document_factory \
			--run-root "$(RUN_OUT)" \
			--metrics-dir "$(RUN_METRICS_DIR)" \
			--write-dir "$(RUN_HUMAN_DIR)" \
			--months "$(METRIC_MONTHS)" \
			--rent-place-col "$(RENT_PLACE_COL)" \
			--rent-detail-col "$(RENT_DETAIL_COL)" \
			--flow-rollup-groupby "$(FLOW_ROLLUP_GROUPBY)" \
			--include-statuses "$(INCLUDE_STATUSES)" \
			--noise-floor "$(NOISE_FLOOR)"; \
		test -s "$(RUN_HUMAN_DIR)/balance_humano_v2.html"; \
		test -s "$(RUN_HUMAN_DIR)/story_manifest.json"; \
	'
	@$(MAKE) _update_latest \
		RUN_STAMP="$(RUN_STAMP)" \
		RUN_OUT="$(RUN_OUT)" \
		RUN_RUN_ID="$(RUN_RUN_ID)" \
		RUN_REL="$(RUN_REL)" \
		OUT="$(OUT)" \
		RUN_BASE="$(RUN_BASE)"

	
# ========================================
# LEGACY REPORTING LAYER (deprecated)
# - Kept only for comparison / rescue
# - Not part of official accounting spine
# ========================================
	

# # Add after run-views:

# # run-storypack-cashflow_v1: run-views

# # call $(PY) report_cashflow.py --in-dir "$(RUN_OUT)" --out-dir "$(RUN_OUT)/storypack/latest" or equivalent

# # run-storypack-balance_v1: run-views

# # same pattern

# # ========================================
# # STORYPACK (REPORT ARTIFACT CREATORS)
# # ========================================

# .PHONY: run-storypack-cashflow
# run-storypack-cashflow: run-views
# 	@$(call _guard_out_dir,$(RUN_OUT))
# 	@echo "[LEGACY][RUN][STORYPACK][cashflow_v1] -> out=$(RUN_STORYPACK_DIR)/cashflow_v1"
# 	@mkdir -p "$(RUN_STORYPACK_DIR)/cashflow_v1"
# 	@bash -eu -o pipefail -c '\
# 		err="$(RUN_OUT)/storypack_cashflow.stderr.log"; \
# 		$(PY) accounting/report_cashflow.py \
# 			--accounting-root "$(RUN_OUT)" \
# 			--write-dir "$(RUN_STORYPACK_DIR)/cashflow_v1" \
# 			--mode run \
# 			--run-id "$(RUN_RUN_ID)" \
# 			> /dev/null 2> "$$err"; \
# 		test -s "$(RUN_STORYPACK_DIR)/cashflow_v1/story_manifest.json"; \
# 	'

# .PHONY: run-storypack-balance
# run-storypack-balance: run-views
# 	@$(call _guard_out_dir,$(RUN_OUT))
# 	@echo "[LEGACY][RUN][STORYPACK][balance_v1] -> out=$(RUN_STORYPACK_DIR)/balance_v1"
# 	@mkdir -p "$(RUN_STORYPACK_DIR)/balance_v1"
# 	@bash -eu -o pipefail -c '\
# 		err="$(RUN_OUT)/storypack_balance.stderr.log"; \
# 		$(PY) accounting/report_balance.py \
# 			--accounting-root "$(RUN_OUT)" \
# 			--write-dir "$(RUN_STORYPACK_DIR)/balance_v1" \
# 			--mode run \
# 			--run-id "$(RUN_RUN_ID)" \
# 			> /dev/null 2> "$$err"; \
# 		test -s "$(RUN_STORYPACK_DIR)/balance_v1/story_manifest.json"; \
# 	'

# .PHONY: run-storypack
# run-storypack: run-storypack-cashflow run-storypack-balance
# 	@echo "[LEGACY][RUN][STORYPACK] done -> $(RUN_STORYPACK_DIR)"




# # ========================================


# .PHONY: run-compile-cashflow
# run-compile-cashflow: run-storypack
# 	@$(call _guard_out_dir,$(RUN_OUT))
# 	@echo "[LEGACY][RUN][COMPILE][cashflow_v1] -> out=$(RUN_DOCS_DIR)/cashflow_v1"
# 	@mkdir -p "$(RUN_DOCS_DIR)/cashflow_v1"
# 	@$(PY) accounting/compile_reports.py \
# 		--storypack-root "$(RUN_STORYPACK_DIR)" \
# 		--template "templates/cashflow_template.md" \
# 		--out-dir "$(RUN_DOCS_DIR)/cashflow_v1" \
# 		--css "$(RUN_ASSETS_CSS)"

# .PHONY: run-compile-balance
# run-compile-balance: run-storypack
# 	@$(call _guard_out_dir,$(RUN_OUT))
# 	@echo "[LEGACY][RUN][COMPILE][balance_v1] -> out=$(RUN_DOCS_DIR)/balance_v1"
# 	@mkdir -p "$(RUN_DOCS_DIR)/balance_v1"
# 	@$(PY) accounting/compile_reports.py \
# 		--storypack-root "$(RUN_STORYPACK_DIR)" \
# 		--template "templates/balance_template.md" \
# 		--out-dir "$(RUN_DOCS_DIR)/balance_v1" \
# 		--css "$(RUN_ASSETS_CSS)"

# .PHONY: run-compile
# run-compile: run-compile-cashflow run-compile-balance
# 	@echo "[RUN][COMPILE] done -> $(RUN_DOCS_DIR)"





# ========================================
# CHECKS
# ========================================

.PHONY: _check_ingest
_check_ingest:
	@$(call _guard_out_dir,$(OUT_DIR))
	@OUT_DIR="$(OUT_DIR)" MODE="$(MODE)" FIXTURE="$(FIXTURE)" $(PY) scripts/check_ingest.py

.PHONY: _check_materialize
_check_materialize:
	@$(call _guard_out_dir,$(OUT_DIR))
	@OUT_DIR="$(OUT_DIR)" MODE="$(MODE)" FREQ="$(FREQ)" $(PY) scripts/check_materialize.py

.PHONY: _check_views
_check_views:
	@$(call _guard_out_dir,$(OUT_DIR))
	@sanity="$(OUT_DIR)/views/views_sanity.json"; \
	test -s "$$sanity" || (echo "ERROR: views_sanity.json missing/empty at $$sanity"; exit 2); \
	$(PY) -c 'import json,sys; json.load(open(sys.argv[1],"r",encoding="utf-8"))' "$$sanity"




# ========================================
# Aliases / convenience
# ========================================

.PHONY: smoke run-all run
smoke: smoke-accounting
run: run-accounting
run-all: run-accounting
	@echo "[RUN] done. latest -> $(RUN_REL)"


# ----------------------------------------
# Explicit env wrappers (no implicit .env include)
# ----------------------------------------
ENV_FILE ?= private/accounting.env

.PHONY: run-env smoke-env
run-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) run-accounting'

smoke-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) smoke-accounting'