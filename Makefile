# Makefile.v3 - Accounting spine
# Official path: run-ingest -> run-materialize -> run-marts -> run-metrics -> run-human-report
# Design goals:
# - Two modes: smoke (fixture/offline) vs run (live/bounded)
# - Explicit out-dir passed to all Python entrypoints
# - Timestamped run outputs (avoid stale-file illusions)
# - Content checks (not only presence)
# - Marts consumes Stage D; reports/ is optional legacy input only, never a required stage

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory

PY ?= python3
export PYTHONUNBUFFERED := 1

-include .env
export ACCOUNT_SHEET_URL ACCOUNT_SA ACCOUNT_SHEET_NAME


# ----------------------------------------
# Explicit env wrappers (no implicit .env include)
# ----------------------------------------
ENV_FILE ?= private/accounting.env

.PHONY: run-env smoke-env
run-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) run-accounting'

smoke-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) smoke-accounting'

	
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

# Assert marts/views sanity exists and invariant errors are empty
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

RUN_DEBT_DIR          := $(OUT)/debt_resolution/$(RUN_RUN_ID)
RUN_DEBT_BALANCE_DIR  := $(OUT)/debt_resolution/$(RUN_RUN_ID)

DEBT_LATEST := $(OUT)/debt_resolution/latest

DEBT_CURRENCIES ?= USD
DEBT_REPAYMENT_STATUSES ?= pagado
DEBT_FULL_ONLY ?= 1
DEBT_EXCLUDE_HOUSEHOLD ?= 1
DRY_RUN ?= 0



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
		link_swap "$(OUT)/debt_resolution" "$(RUN_REL)"; \
		link_swap "$(OUT)/metrics" "$(RUN_REL)"; \
		link_swap "$(OUT)/human_reports" "$(RUN_REL)"; \
		echo "[LATEST] updated run,debt,metrics,human latest links"; \
	'

# ----------------------------------------
# Help
# ----------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "Accounting backend control plane"
	@echo ""
	@echo "Doctor / validate:"
	@echo "  make doctor             # env + compile checks; no Google Sheets required"
	@echo "  make validate           # static + artifact contract validation; no private env"
	@echo ""
	@echo "Fixture / smoke:"
	@echo "  make smoke-core         # fixture ingest -> materialize -> semantic/cash wrappers"
	@echo "  make smoke-full         # fixture-safe smoke bundle; currently core + validation + publish dry-run"
	@echo "  make smoke              # compatibility alias for smoke-core"
	@echo ""
	@echo "Live canonical / metrics / dashboard / human:"
	@echo "  make run-canonical      # live ingest -> materialize -> marts -> debt wrappers"
	@echo "  make metrics-from-run   # metrics from existing RUN_OUT/RUN_STAMP; no upstream live ingest"
	@echo "  make run-metrics        # alias for metrics-from-run"
	@echo "  make run-metrics-live   # orchestrate live upstream then metrics"
	@echo "  make run-dashboard      # assert annual dashboard outputs from metrics"
	@echo "  make run-human          # build human report from existing metrics/run artifacts"
	@echo "  make run-full           # full live pipeline -> publish -> release-check"
	@echo ""
	@echo "Publish / release:"
	@echo "  make publish-latest     # package latest artifacts only"
	@echo "  make release-check      # validate public/accounting/latest readiness"
	@echo ""
	@echo "Legacy compatibility aliases:"
	@echo "  make ledger | materialize | debt | debt-views | metrics | human-report | publish | build-all"
	@echo "  make run-accounting | run-accounting-full | run-human-balance | run-debt-balance"
	@echo ""
	@echo "Diagnostics / cleanup / experimental:"
	@echo "  make clean-derived      # remove derived accounting outputs"
	@echo "  make front-report       # presentation-only report factory stub"
	@echo ""
	@echo "Key vars:"
	@echo "  OUT=out  RUN_STAMP=<run-id>  FREQ=W|M  TOP=6  METRIC_MONTHS=6"
	@echo "  FIXTURE=$(ROOT)/fixtures/ledger_fixture.csv"
	@echo "  ACCOUNT_SA=/path/to/sa.json  ACCOUNT_SHEET_URL=...  ACCOUNT_SHEET_NAME='C. Long Ledger'"
	@echo ""

# ----------------------------------------
# Meta targets
# ----------------------------------------
# Canonical names: these are the preferred operational surface.
.PHONY: ledger materialize debt debt-views metrics human-report publish-latest publish
ledger: run-ingest

materialize: run-materialize

debt: run-debt

debt-views: run-debt-views

metrics: run-metrics

human-report: run-human-report

publish-latest:
	@bash -eu -o pipefail -c '\
		args=( --project-root "$(ROOT)" --clean ); \
		if [ "$(DRY_RUN)" = "1" ]; then args+=( --dry-run ); fi; \
		$(PY) -m accounting.publish.latest "$${args[@]}"; \
	'

# Compatibility alias; prefer publish-latest.
publish: publish-latest

# Composite names: one clear path for full builds and frontend handoff.
.PHONY: build-all build-report build-front
build-all: run-full

build-report: human-report

build-front: publish-latest

# Support and experimental command surface.
.PHONY: doctor validate clean-derived front-report
doctor:
	@$(PY) --version
	@$(PY) -m py_compile \
		accounting/config.py \
		accounting/logging_utils.py \
		accounting/ledger/ingest.py \
		accounting/stage_d/materialize.py \
		accounting/core/timeseries.py \
		accounting/contracts/models.py \
		accounting/viz/plots.py \
		accounting/artifacts/manifest.py \
		accounting/support/run_id.py \
		accounting/support/io.py \
		accounting/support/currency.py \
		accounting/support/env.py \
		accounting/support/hashing.py \
		accounting/support/partitions.py \
		accounting/marts/build.py \
		accounting/marts/cash.py \
		accounting/marts/debt.py \
		accounting/marts/semantic.py \
		accounting/debt/resolve.py \
		accounting/debt/balance_views.py \
		accounting/metrics/io.py \
		accounting/metrics/registry.py \
		accounting/metrics/builders.py \
		accounting/metrics/derive.py \
		accounting/metrics/validate.py \
		accounting/metrics/views.py \
		accounting/metrics/drilldown.py \
		accounting/metrics/frontier.py \
		accounting/metrics/build.py \
		accounting/human/tables.py \
		accounting/human/document.py \
		accounting/human/front.py \
		accounting/human/reports.py \
		accounting/publish/latest.py \
		accounting/publish/manifest.py \
		accounting/publish/snapshot.py
	@echo "accounting command modules compile ok"

validate: doctor
	@$(MAKE) help >/dev/null
	@$(PY) scripts/check_contracts.py
	@echo "make help and contract validation ok"

clean-derived:
	rm -rf "$(OUT)/smoke/accounting" "$(OUT)/run/accounting" "$(OUT)/metrics" "$(OUT)/human_reports" "$(OUT)/debt_resolution" "$(ROOT)/public/accounting/latest"

front-report:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_METRICS_DIR)/metric_values.csv" || (echo "ERROR: missing metric_values.csv at $(RUN_METRICS_DIR). Run make metrics first or point RUN_STAMP to an existing run."; exit 2)
	@mkdir -p "$(OUT)/front/$(RUN_RUN_ID)"
	@$(PY) -m accounting.human.front \
		--run-root "$(RUN_OUT)" \
		--metrics-dir "$(RUN_METRICS_DIR)" \
		--write-dir "$(OUT)/front/$(RUN_RUN_ID)" \
		--months "$(METRIC_MONTHS)" \
		--rent-place-col "$(RENT_PLACE_COL)" \
		--rent-detail-col "$(RENT_DETAIL_COL)" \
		--flow-rollup-groupby "$(FLOW_ROLLUP_GROUPBY)" \
		--include-statuses "$(INCLUDE_STATUSES)" \
		--noise-floor "$(NOISE_FLOOR)"

.PHONY: smoke-core smoke-full run-canonical run-full run-dashboard run-human metrics-from-run run-metrics-live smoke-accounting run-accounting run-accounting-full run-downstream-from-ledger run-metrics-and-human run-human-balance-only

smoke-core: smoke-ingest
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@$(PY) -m accounting.stage_d.materialize --out-dir "$(SMOKE_OUT)" --freq "$(FREQ)" --force 1 --mode smoke --run-id "$(SMOKE_RUN_ID)"
	@test -s "$(SMOKE_OUT)/classification_audit.csv"
	@test -s "$(SMOKE_OUT)/classification_audit_summary.csv"
	@test -s "$(SMOKE_OUT)/monthly_flow_semantic_split.csv"
	@test -s "$(SMOKE_OUT)/monthly_operating_statement.csv"
	@test -s "$(SMOKE_OUT)/monthly_operating_statement_qa.csv"
	@test -s "$(SMOKE_OUT)/semantic_leakage_qa.csv"
	@test -s "$(SMOKE_OUT)/monthly_cash_close.csv"
	@test -s "$(SMOKE_OUT)/monthly_cash_close_qa.csv"
	@echo "smoke-core passed fixture ingest/materialize semantic and cash wrapper checks"

smoke-full: smoke-core validate
	@$(PY) -m accounting.publish.latest --project-root "$(ROOT)" --dry-run >/dev/null
	@echo "smoke-full partial: fixture core + validation + publish dry-run passed; fixture debt/human publish remains documented follow-up"

smoke-accounting: smoke-core

run-canonical: run-debt-views

run-full: run-canonical run-metrics run-dashboard run-human publish-latest release-check

run-accounting: run-accounting-full
run-accounting-full: run-full

run-downstream-from-ledger:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/ledger_canonical.csv" || (echo "ERROR: missing ledger_canonical.csv at $(RUN_OUT). Run make run-ingest first or point RUN_STAMP to an existing run."; exit 2)
	@$(MAKE) _run_materialize_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)"
	@$(MAKE) _run_views_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)"
	@$(MAKE) _run_debt_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
	@$(MAKE) _run_debt_balance_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
	@$(MAKE) _run_metrics_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"
	@$(MAKE) _run_human_balance_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)" METRIC_MONTHS="$(METRIC_MONTHS)" RENT_PLACE_COL="$(RENT_PLACE_COL)" RENT_DETAIL_COL="$(RENT_DETAIL_COL)" FLOW_ROLLUP_GROUPBY="$(FLOW_ROLLUP_GROUPBY)" INCLUDE_STATUSES="$(INCLUDE_STATUSES)" NOISE_FLOOR="$(NOISE_FLOOR)"

run-metrics-and-human:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_VIEWS_SANITY)" || (echo "ERROR: missing views_sanity.json at $(RUN_VIEWS_SANITY). Run make run-marts first or point RUN_STAMP to an existing run."; exit 2)
	@$(MAKE) _run_debt_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
	@$(MAKE) _run_debt_balance_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
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
	@$(PY) -m accounting.ledger.ingest \
		--mode smoke \
		--fixture "$(FIXTURE)" \
		--out-dir "$(SMOKE_OUT)" \
		--run-id "$(SMOKE_RUN_ID)"
	@$(MAKE) _check_ingest OUT_DIR="$(SMOKE_OUT)" MODE="smoke" FIXTURE="$(FIXTURE)"

.PHONY: smoke-materialize
smoke-materialize: smoke-ingest
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@$(PY) -m accounting.stage_d.materialize \
		--out-dir "$(SMOKE_OUT)" \
		--freq "$(FREQ)" \
		--force 1 \
		--mode smoke \
		--run-id "$(SMOKE_RUN_ID)"
	@$(MAKE) _check_materialize OUT_DIR="$(SMOKE_OUT)" MODE="smoke" FREQ="$(FREQ)"
	@test -s "$(SMOKE_OUT)/classification_audit.csv"
	@test -s "$(SMOKE_OUT)/classification_audit_summary.csv"
	@test -s "$(SMOKE_OUT)/monthly_flow_semantic_split.csv"
	@test -s "$(SMOKE_OUT)/monthly_operating_statement.csv"
	@test -s "$(SMOKE_OUT)/monthly_operating_statement_qa.csv"
	@test -s "$(SMOKE_OUT)/semantic_leakage_qa.csv"
	@test -s "$(SMOKE_OUT)/monthly_cash_close.csv"
	@test -s "$(SMOKE_OUT)/monthly_cash_close_qa.csv"

.PHONY: smoke-views
smoke-views: smoke-materialize
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@mkdir -p "$(SMOKE_VIEWS_DIR)"
	@mkdir -p "$(SMOKE_REPORTS_DIR)"  # anchor for loader heuristics / optional legacy files
	@$(PY) -m accounting.marts.build \
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
	@$(PY) -m accounting.ledger.ingest \
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
	@$(PY) -m accounting.stage_d.materialize \
		--out-dir "$(RUN_OUT)" \
		--freq "$(FREQ)" \
		--force 0 \
		--mode run \
		--run-id "$(RUN_RUN_ID)"
	@$(MAKE) _check_materialize OUT_DIR="$(RUN_OUT)" MODE="run" FREQ="$(FREQ)"
	@test -s "$(RUN_OUT)/classification_audit.csv"
	@test -s "$(RUN_OUT)/classification_audit_summary.csv"
	@test -s "$(RUN_OUT)/monthly_flow_semantic_split.csv"
	@test -s "$(RUN_OUT)/monthly_operating_statement.csv"
	@test -s "$(RUN_OUT)/monthly_operating_statement_qa.csv"
	@test -s "$(RUN_OUT)/semantic_leakage_qa.csv"
	@test -s "$(RUN_OUT)/monthly_cash_close.csv"
	@test -s "$(RUN_OUT)/monthly_cash_close_qa.csv"

.PHONY: run-marts _run_views_action
run-marts: run-materialize _run_views_action

_run_views_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/per_flow_time_long.freq=$(FREQ).csv" || (echo "ERROR: missing materialized outputs at $(RUN_OUT). Run make run-materialize first or use run-downstream-from-ledger."; exit 2)
	@mkdir -p "$(RUN_VIEWS_DIR)"
	@mkdir -p "$(RUN_REPORTS_DIR)"  # anchor for loader heuristics / optional legacy files
	@$(PY) -m accounting.marts.build \
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
run-debt: run-marts _run_debt_action

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
			--currencies "$(DEBT_CURRENCIES)" \
			--repayment-statuses "$(DEBT_REPAYMENT_STATUSES)" \
		); \
		if [ "$(DEBT_EXCLUDE_HOUSEHOLD)" = "1" ]; then \
			args+=( --exclude-household ); \
		fi; \
		if [ "$(DEBT_FULL_ONLY)" = "1" ]; then \
			args+=( --full-only ); \
		fi; \
		$(PY) -m accounting.debt.resolve "$${args[@]}"; \
		test -s "$(RUN_DEBT_DIR)/debt_open_items.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_allocations.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_repayment_events.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_resolution_timeline.csv"; \
	'

.PHONY: run-debt-views run-debt-balance _run_debt_balance_action
run-debt-views: run-debt _run_debt_balance_action

# Compatibility alias; prefer run-debt-views.
run-debt-balance: run-debt-views

_run_debt_balance_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_DEBT_DIR)/debt_open_items.csv" || (echo "ERROR: missing debt_open_items.csv at $(RUN_DEBT_DIR)"; exit 2)
	@bash -eu -o pipefail -c '\
		$(PY) -m accounting.debt.balance_views \
			--open-items "$(RUN_DEBT_DIR)/debt_open_items.csv" \
			--write-dir "$(RUN_DEBT_BALANCE_DIR)"; \
		test -s "$(RUN_DEBT_BALANCE_DIR)/debt_balance_daily.csv"; \
		test -s "$(RUN_DEBT_BALANCE_DIR)/debt_balance_monthly.csv"; \
		test -s "$(RUN_DEBT_BALANCE_DIR)/debt_balance_quarterly.csv"; \
		test -s "$(RUN_DEBT_BALANCE_DIR)/debt_balance_yearly.csv"; \
		$(PY) -m accounting.marts.debt \
			--debt-dir "$(RUN_DEBT_BALANCE_DIR)" \
			--write-dir "$(RUN_OUT)"; \
		test -s "$(RUN_OUT)/monthly_debt_position.csv"; \
		test -s "$(RUN_OUT)/monthly_debt_position_qa.csv"; \
		test -s "$(RUN_OUT)/monthly_debt_activity.csv"; \
		test -s "$(RUN_OUT)/monthly_debt_activity_qa.csv"; \
	'


.PHONY: run-metrics _run_metrics_action
run-metrics: metrics-from-run

metrics-from-run: _run_metrics_action

run-metrics-live: run-debt-views _run_metrics_action

_run_metrics_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_VIEWS_SANITY)" || (echo "ERROR: missing views_sanity.json at $(RUN_VIEWS_SANITY)"; exit 2)
	@mkdir -p "$(RUN_METRICS_DIR)"
	@bash -eu -o pipefail -c '\
		$(PY) -m accounting.metrics.build \
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
		test -s "$(RUN_METRICS_DIR)/metric_contract_frontier.csv"; \
		test -s "$(RUN_METRICS_DIR)/frontend_metric_series.csv"; \
		test -s "$(RUN_METRICS_DIR)/metrics_frontier_qa.csv"; \
		test -s "$(RUN_METRICS_DIR)/frontier_source_qa.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/income_statement_monthly_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/rent_rollup_by_place_m_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/rent_rollup_by_detail_m_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/flow_type_rollup_m_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/draws_discipline_monthly_last6.csv"; \
		test -s "$(RUN_METRICS_DIR)/metric_views/metric_views_manifest.csv"; \
	'


.PHONY: run-human-report run-human-balance _run_human_balance_action
run-human-report: run-human

run-human: _run_human_balance_action

run-dashboard: run-metrics
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_metrics.csv"
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_contract.csv"
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_qa.csv"

# Compatibility alias; prefer run-human-report.
run-human-balance: run-human-report

_run_human_balance_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_METRICS_DIR)/metric_values.csv" || (echo "ERROR: missing metric_values.csv at $(RUN_METRICS_DIR)"; exit 2)
	@mkdir -p "$(RUN_HUMAN_DIR)"
	@bash -eu -o pipefail -c '\
		$(PY) -m accounting.human.document \
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

.PHONY: release-check smoke run-all run
release-check:
	@$(PY) scripts/check_release.py --public-root "$(ROOT)/public/accounting/latest"

smoke: smoke-core
run: run-accounting
run-all: run-accounting
	@echo "[RUN] done. latest -> $(RUN_REL)"
