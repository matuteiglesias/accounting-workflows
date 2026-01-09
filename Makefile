# Makefile.v2 - Accounting spine (A ingest -> D materialize -> E reports -> V views)
# Design goals:
# - Two modes: smoke (fixture/offline) vs run (live/bounded)
# - Explicit out-dir passed to all Python entrypoints
# - Timestamped run outputs (avoid stale-file illusions)
# - Content checks (not only presence)
# - Materialize emits manifest.json; ingest/reports may emit wrappers

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

# Smoke fixture (override if your fixture lives elsewhere)
FIXTURE ?= $(ROOT)/fixtures/ledger_fixture.csv

# Live ingest vars (export before running, or use run-env wrapper)
ACCOUNT_SA ?=
ACCOUNT_SHEET_URL ?=
ACCOUNT_SHEET_NAME ?= C. Long Ledger

# Optional explicit parties for reports (RUN boundedness)
# Examples:
#   REPORT_PARTIES="PM FB"
#   REPORT_PARTIES="PM,FB"
REPORT_PARTIES ?= PM,FB

# ----------------------------------------
# Helpers
# ----------------------------------------
define require_var
	@if [ -z "$($(1))" ]; then echo "ERROR: missing required var: $(1)"; exit 2; fi
endef

define _guard_out_dir
	@if [ -z "$(1)" ]; then echo "ERROR: OUT_DIR empty"; exit 2; fi
endef

# ----------------------------------------
# Derived output dirs
# ----------------------------------------
SMOKE_OUT := $(OUT)/smoke/accounting
RUN_STAMP ?= $(shell date -u +%Y%m%dT%H%M%SZ)
RUN_OUT   := $(OUT)/run/accounting/$(RUN_STAMP)

SMOKE_REPORTS_DIR := $(SMOKE_OUT)/reports
RUN_REPORTS_DIR   := $(RUN_OUT)/reports

SMOKE_VIEWS_DIR   := $(SMOKE_OUT)/views
RUN_VIEWS_DIR     := $(RUN_OUT)/views

SMOKE_VIEWS_SANITY := $(SMOKE_VIEWS_DIR)/views_sanity.json
RUN_VIEWS_SANITY   := $(RUN_VIEWS_DIR)/views_sanity.json

SMOKE_RUN_ID := smoke
RUN_RUN_ID   := $(RUN_STAMP)

SMOKE_META_DIR := $(SMOKE_OUT)/meta
RUN_META_DIR   := $(RUN_OUT)/meta

SMOKE_REPORTS_SUMMARY := $(SMOKE_META_DIR)/reports_summary.json
RUN_REPORTS_SUMMARY   := $(RUN_META_DIR)/reports_summary.json

# ----------------------------------------
# Help
# ----------------------------------------
.PHONY: help
help:
	@echo ""
	@echo "Accounting spine v2:"
	@echo "  make smoke-accounting         # fixture -> ingest -> materialize -> reports -> views (+checks)"
	@echo "  make run-accounting           # live sheet -> ingest -> materialize -> reports -> views (+checks)"
	@echo ""
	@echo "Per-step targets:"
	@echo "  make smoke-ingest | smoke-materialize | smoke-reports | smoke-views"
	@echo "  make run-ingest   | run-materialize   | run-reports   | run-views"
	@echo ""
	@echo "Key vars:"
	@echo "  OUT=out  FREQ=W|M  TOP=6"
	@echo "  FIXTURE=$(ROOT)/fixtures/ledger_fixture.csv"
	@echo "  ACCOUNT_SA=/path/to/sa.json  ACCOUNT_SHEET_URL=...  ACCOUNT_SHEET_NAME='C. Long Ledger'"
	@echo "  REPORT_PARTIES='PM FB'  (or 'PM,FB')"
	@echo ""
	@echo "Env wrapper targets (explicit, no implicit .env include):"
	@echo "  make run-env ENV_FILE=private/accounting.env"
	@echo "  make smoke-env ENV_FILE=private/accounting.env"
	@echo "  make run-story-env ENV_FILE=private/accounting.env STORY_YEAR=2025"
	@echo ""

# ----------------------------------------
# Meta targets
# ----------------------------------------
.PHONY: smoke-accounting run-accounting
smoke-accounting: smoke-views
run-accounting: run-views

# ========================================
# SMOKE MODE
# ========================================

.PHONY: smoke-ingest
smoke-ingest:
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@echo "[SMOKE][INGEST] fixture=$(FIXTURE) -> out=$(SMOKE_OUT)"
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
	@echo "[SMOKE][MATERIALIZE] freq=$(FREQ) -> out=$(SMOKE_OUT)"
	@$(PY) -m accounting.materialize \
		--out-dir "$(SMOKE_OUT)" \
		--freq "$(FREQ)" \
		--force 1 \
		--mode smoke \
		--run-id "$(SMOKE_RUN_ID)"
	@$(MAKE) _check_materialize OUT_DIR="$(SMOKE_OUT)" MODE="smoke" FREQ="$(FREQ)"

.PHONY: smoke-reports
smoke-reports: smoke-materialize
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@echo "[SMOKE][REPORTS] top=$(TOP) freq=$(FREQ) -> out=$(SMOKE_REPORTS_DIR)"
	@mkdir -p "$(SMOKE_REPORTS_DIR)"

	@bash -eu -o pipefail -c '\
	mkdir -p "$(SMOKE_META_DIR)"; \
	err="$(SMOKE_OUT)/reports.stderr.log"; \
	$(PY) -m accounting.reports \
		--out-dir "$(SMOKE_OUT)" \
		--freq "$(FREQ)" \
		--write-dir "$(SMOKE_REPORTS_DIR)" \
		--top "$(TOP)" \
		--summary-path "$(SMOKE_REPORTS_SUMMARY)" \
		--mode smoke \
		--run-id "$(SMOKE_RUN_ID)" \
		> /dev/null 2> "$$err"; \
	test -s "$(SMOKE_REPORTS_SUMMARY)" || (echo "ERROR: reports_summary.json missing/empty"; exit 2); \
	$(PY) -c "import json,sys; json.load(open(sys.argv[1],\"r\",encoding=\"utf-8\"))" "$(SMOKE_REPORTS_SUMMARY)"; \
	'
	@$(MAKE) _check_reports OUT_DIR="$(SMOKE_OUT)" MODE="smoke" REPORTS_DIR="$(SMOKE_REPORTS_DIR)"

.PHONY: smoke-views
smoke-views: smoke-reports
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@echo "[SMOKE][VIEWS] freq=$(FREQ) -> out=$(SMOKE_VIEWS_DIR)"
	@mkdir -p "$(SMOKE_VIEWS_DIR)"
	@bash -eu -o pipefail -c '\
	err="$(SMOKE_OUT)/views.stderr.log"; \
	$(PY) -m accounting.views \
		--reports-dir "$(SMOKE_REPORTS_DIR)" \
		--write-dir "$(SMOKE_VIEWS_DIR)" \
		--freq "$(FREQ)" \
		> /dev/null 2> "$$err"; \
	test -s "$(SMOKE_VIEWS_SANITY)" || (echo "ERROR: views_sanity.json missing/empty"; exit 2); \
	$(PY) -c "import json,sys; json.load(open(sys.argv[1],\"r\",encoding=\"utf-8\"))" "$(SMOKE_VIEWS_SANITY)"; \
	'
	@$(MAKE) _check_views OUT_DIR="$(SMOKE_OUT)" MODE="smoke"

# ========================================
# RUN MODE (LIVE)
# ========================================

.PHONY: run-ingest
run-ingest:
	@$(call _guard_out_dir,$(RUN_OUT))
	# @$(call require_var,ACCOUNT_SA)
	@$(call require_var,ACCOUNT_SHEET_URL)
	@echo "[RUN][INGEST] sheet='$(ACCOUNT_SHEET_NAME)' -> out=$(RUN_OUT)"
	@mkdir -p "$(RUN_OUT)"
	@$(PY) -m accounting.ingest \
		--mode run \
		--out-dir "$(RUN_OUT)" \
		--run-id "$(RUN_RUN_ID)" \
		--service-account "$(ACCOUNT_SA)" \
		--sheet-url "$(ACCOUNT_SHEET_URL)" \
		--sheet-name "$(ACCOUNT_SHEET_NAME)"
	@$(MAKE) _check_ingest OUT_DIR="$(RUN_OUT)" MODE="run"

.PHONY: run-materialize
run-materialize: run-ingest
	@$(call _guard_out_dir,$(RUN_OUT))
	@echo "[RUN][MATERIALIZE] freq=$(FREQ) -> out=$(RUN_OUT)"
	@$(PY) -m accounting.materialize \
		--out-dir "$(RUN_OUT)" \
		--freq "$(FREQ)" \
		--force 0 \
		--mode run \
		--run-id "$(RUN_RUN_ID)"
	@$(MAKE) _check_materialize OUT_DIR="$(RUN_OUT)" MODE="run" FREQ="$(FREQ)"

.PHONY: run-reports
run-reports: run-materialize
	@$(call _guard_out_dir,$(RUN_OUT))
	@echo "[RUN][REPORTS] freq=$(FREQ) parties='$(REPORT_PARTIES)' top=$(TOP) -> out=$(RUN_REPORTS_DIR)"
	@mkdir -p "$(RUN_REPORTS_DIR)"

	@bash -eu -o pipefail -c '\
	mkdir -p "$(RUN_META_DIR)"; \
	err="$(RUN_OUT)/reports.stderr.log"; \
	$(PY) -m accounting.reports \
		--out-dir "$(RUN_OUT)" \
		--freq "$(FREQ)" \
		--write-dir "$(RUN_REPORTS_DIR)" \
		--top "$(TOP)" \
		--parties "$(REPORT_PARTIES)" \
		--summary-path "$(RUN_REPORTS_SUMMARY)" \
		--mode run \
		--run-id "$(RUN_RUN_ID)" \
		> /dev/null 2> "$$err"; \
	test -s "$(RUN_REPORTS_SUMMARY)" || (echo "ERROR: reports_summary.json missing/empty"; exit 2); \
	$(PY) -c "import json,sys; json.load(open(sys.argv[1],\"r\",encoding=\"utf-8\"))" "$(RUN_REPORTS_SUMMARY)"; \
	'
	@$(MAKE) _check_reports OUT_DIR="$(RUN_OUT)" MODE="run" REPORTS_DIR="$(RUN_REPORTS_DIR)"

.PHONY: run-views
run-views: run-reports
	@$(call _guard_out_dir,$(RUN_OUT))
	@echo "[RUN][VIEWS] freq=$(FREQ) -> out=$(RUN_VIEWS_DIR)"
	@mkdir -p "$(RUN_VIEWS_DIR)"
	@bash -eu -o pipefail -c '\
	err="$(RUN_OUT)/views.stderr.log"; \
	$(PY) -m accounting.views \
		--reports-dir "$(RUN_REPORTS_DIR)" \
		--write-dir "$(RUN_VIEWS_DIR)" \
		--freq "$(FREQ)" \
		> /dev/null 2> "$$err"; \
	test -s "$(RUN_VIEWS_SANITY)" || (echo "ERROR: views_sanity.json missing/empty"; exit 2); \
	$(PY) -c "import json,sys; json.load(open(sys.argv[1],\"r\",encoding=\"utf-8\"))" "$(RUN_VIEWS_SANITY)"; \
	'
	@$(MAKE) _check_views OUT_DIR="$(RUN_OUT)" MODE="run"

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

.PHONY: _check_reports
_check_reports:
	@$(call _guard_out_dir,$(OUT_DIR))
	@OUT_DIR="$(OUT_DIR)" MODE="$(MODE)" REPORTS_DIR="$(REPORTS_DIR)" $(PY) scripts/check_reports.py

.PHONY: _check_views
_check_views:
	@$(call _guard_out_dir,$(OUT_DIR))
	@echo "[CHECK][VIEWS] MODE=$(MODE) OUT_DIR=$(OUT_DIR)"
	@sanity="$(OUT_DIR)/views/views_sanity.json"; \
	test -s "$$sanity" || (echo "ERROR: views_sanity.json missing/empty at $$sanity"; exit 2); \
	$(PY) -c 'import json,sys; json.load(open(sys.argv[1],"r",encoding="utf-8"))' "$$sanity"

# ========================================
# Aliases / convenience
# ========================================

.PHONY: smoke run-all run caps
smoke: smoke-accounting
run-all: run-accounting

# optional alias if you want the runner to call `run` not `run-all`
run: run-accounting

# ----------------------------------------
# Explicit env wrappers (no implicit .env include)
# ----------------------------------------
ENV_FILE ?= private/accounting.env

.PHONY: run-env smoke-env run-story-env smoke-story-env
run-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) run-accounting'

smoke-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) smoke-accounting'

run-story-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) run-storypack STORY_YEAR="$(STORY_YEAR)" STORY_FREQ="$(STORY_FREQ)"'

smoke-story-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) smoke-storypack STORY_YEAR="$(STORY_YEAR)" STORY_FREQ="$(STORY_FREQ)"'

# ========================================
# Storypack layer (F) - depends on views (V)
# ========================================

STORY_YEAR ?= 2025
STORY_FREQ ?= $(FREQ)
STORY_WRITE_DIR ?= $(RUN_OUT)/storypack

.PHONY: run-storypack smoke-storypack
run-storypack: run-views
	@echo "[RUN][STORY] year=$(STORY_YEAR) freq=$(STORY_FREQ) -> $(RUN_OUT)/storypack"
	@$(PY) scripts/run_storypack.py \
		--run-out "$(RUN_OUT)" \
		--year "$(STORY_YEAR)" \
		--freq "$(STORY_FREQ)" \
		--report-parties "$(REPORT_PARTIES)" \
		--write-dir "$(RUN_OUT)/storypack"

smoke-storypack: smoke-views
	@echo "[SMOKE][STORY] -> $(SMOKE_OUT)/storypack"
	@$(PY) scripts/run_storypack.py \
		--run-out "$(SMOKE_OUT)" \
		--year "$(STORY_YEAR)" \
		--freq "$(STORY_FREQ)" \
		--report-parties "$(REPORT_PARTIES)" \
		--write-dir "$(SMOKE_OUT)/storypack"

# Notes:
# notebooks should stay as assembly/visualization
# reusable logic lives in accounting/views.py / reports.py
