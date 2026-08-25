# Accounting backend control plane
# Official path:
#   run-ingest -> run-materialize -> run-debt-views -> run-metrics -> run-dashboard -> publish-latest
#
# Materialization owns the canonical monthly semantic and cash artifacts.
# There is no separate generic views stage and no parallel metric_values engine.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory

PY ?= python3
export PYTHONUNBUFFERED := 1
export BOXES SCOPE_TAG

-include .env
export ACCOUNT_SHEET_URL ACCOUNT_SA ACCOUNT_SHEET_NAME

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(ROOT)

ENV_FILE ?= private/accounting.env
OUT ?= out
FREQ ?= M
BOXES := $(if $(strip $(BOXES)),$(BOXES),Family Business,Property Management)

ifneq (,$(filter smoke-usd-ccl-valuation smoke-usd-ccl-management-flows,$(MAKECMDGOALS)))
SCOPE_TAG ?= FBPM
else
SCOPE_TAG := $(shell PYTHONPATH='$(ROOT)' $(PY) -c 'from accounting.scope import canonical_scope_tag, parse_box_scope; print(canonical_scope_tag(parse_box_scope("$(BOXES)")))')
endif

FIXTURE ?= $(ROOT)/fixtures/ledger_fixture.csv
USD_CCL_LEDGER_FIXTURE ?= $(ROOT)/fixtures/ledger_valuation_fixture.csv
USD_CCL_RATE_FIXTURE ?= $(ROOT)/fixtures/synthetic_ccl_rates.csv
USD_CCL_POLICY_FIXTURE ?= $(ROOT)/fixtures/valuation_policy_v1.json
USD_CCL_MANAGEMENT_LEDGER_FIXTURE ?= $(ROOT)/fixtures/management_usd_ccl_flow_fixture.csv
USD_CCL_FLOW_POLICY ?= $(ROOT)/reference/fx/ccl_txn_prev_available_v1.json
CCL_RATES ?=

ACCOUNT_SA ?=
ACCOUNT_SHEET_URL ?=
ACCOUNT_SHEET_NAME ?= C. Long Ledger

SMOKE_OUT := $(OUT)/smoke/accounting
SMOKE_RUN_ID := smoke
USD_CCL_SMOKE_OUT ?= $(SMOKE_OUT)/usd_ccl_valuation
USD_CCL_MANAGEMENT_SMOKE_OUT ?= $(SMOKE_OUT)/usd_ccl_management_flows

RUN_STAMP ?= $(shell date -u +%Y%m%dT%H%M%SZ)
RUN_BASE := $(OUT)/run/accounting
RUN_OUT := $(RUN_BASE)/$(RUN_STAMP)_$(SCOPE_TAG)
RUN_REL := $(notdir $(RUN_OUT))
RUN_RUN_ID := $(RUN_REL)

RUN_METRICS_DIR := $(OUT)/metrics/$(RUN_RUN_ID)
METRICS_LATEST := $(OUT)/metrics/latest_$(SCOPE_TAG)

RUN_DEBT_DIR := $(OUT)/debt_resolution/$(RUN_RUN_ID)
RUN_DEBT_BALANCE_DIR := $(RUN_DEBT_DIR)
DEBT_LATEST := $(OUT)/debt_resolution/latest_$(SCOPE_TAG)

DEBT_CURRENCIES ?= USD
DEBT_REPAYMENT_STATUSES ?= pagado
DEBT_FULL_ONLY ?= 1
DRY_RUN ?= 0


define require_var
	@if [ -z "$($(1))" ]; then echo "ERROR: missing required var: $(1)"; exit 2; fi
endef

define _guard_out_dir
	@if [ -z "$(1)" ]; then echo "ERROR: OUT_DIR empty"; exit 2; fi
endef


# ---------------------------------------------------------------------------
# Help / validation / aliases
# ---------------------------------------------------------------------------

.PHONY: help
help:
	@echo ""
	@echo "Accounting backend control plane"
	@echo ""
	@echo "Fixture / validation:"
	@echo "  make smoke-core         # fixture ingest -> governed materialization"
	@echo "  make smoke-full         # smoke-core + validate + publish dry-run"
	@echo "  make validate           # compile + contracts + regression suite"
	@echo ""
	@echo "Live canonical path:"
	@echo "  make run-canonical      # live ingest -> governed materialization"
	@echo "  make run-debt-views     # canonical -> debt resolution/position/activity"
	@echo "  make run-metrics        # governed frontier + annual metrics from RUN_OUT"
	@echo "  make run-metrics-live   # live canonical/debt then governed metrics"
	@echo "  make run-dashboard      # assert governed annual dashboard artifacts"
	@echo "  make run-full           # full live path -> publish -> release-check"
	@echo ""
	@echo "Existing-run / sidecar operations:"
	@echo "  make metrics-from-run RUN_STAMP=<existing stamp>"
	@echo "  make run-usd-ccl-valuation RUN_ROOT=<exact-run> CCL_RATES=<local.csv>"
	@echo "  make run-usd-ccl-management-flows RUN_ROOT=<exact-run> CCL_RATES=<local.csv>"
	@echo ""
	@echo "Publication / professional:"
	@echo "  make publish-latest"
	@echo "  make release-check"
	@echo "  make professional-drilldowns"
	@echo "  make professional-linked-digest"
	@echo ""
	@echo "Compatibility aliases:"
	@echo "  make ledger | materialize | debt | debt-views | metrics | publish | build-all"
	@echo "  make run-accounting | run-accounting-full | run-debt-balance"
	@echo ""
	@echo "Key vars: OUT=out RUN_STAMP=<timestamp> BOXES='Family Business,Property Management'"
	@echo ""

.PHONY: run-env smoke-env
run-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) run-accounting'

smoke-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) smoke-accounting'

.PHONY: doctor validate clean-derived
doctor:
	@$(PY) --version
	@$(PY) -m compileall -q accounting scripts tests
	@echo "accounting command modules compile ok"

validate: doctor
	@$(MAKE) help >/dev/null
	@$(PY) scripts/check_contracts.py
	@$(PY) -m pytest -q
	@echo "compile, contract, and regression validation ok"

clean-derived:
	rm -rf "$(OUT)/smoke/accounting" "$(OUT)/run/accounting" "$(OUT)/metrics" "$(OUT)/debt_resolution" "$(ROOT)/public/accounting/latest" "$(ROOT)/public/accounting/latest_$(SCOPE_TAG)"

.PHONY: ledger materialize debt debt-views metrics publish build-all
ledger: run-ingest
materialize: run-materialize
debt: run-debt
debt-views: run-debt-views
metrics: run-metrics
publish: publish-latest
build-all: run-full


# ---------------------------------------------------------------------------
# Latest pointers / publication
# ---------------------------------------------------------------------------

.PHONY: _update_latest update-latest-light
_update_latest:
	@echo "[RUN][LATEST] run=$(RUN_REL)"
	@$(PY) -m accounting.support.latest --scope-tag "$(SCOPE_TAG)" --target "$(RUN_REL)" \
		--base "$(RUN_BASE)" --base "$(OUT)/debt_resolution" --base "$(OUT)/metrics"

update-latest-light:
	@echo "[RUN][LATEST-LIGHT] run=$(RUN_REL)"
	@$(PY) -m accounting.support.latest --scope-tag "$(SCOPE_TAG)" --target "$(RUN_REL)" \
		--base "$(RUN_BASE)" --base "$(OUT)/metrics"

.PHONY: publish-latest release-check
publish-latest:
	@bash -eu -o pipefail -c '\
		args=( --project-root "$(ROOT)" --scope-tag "$(SCOPE_TAG)" --clean ); \
		if [ "$(DRY_RUN)" = "1" ]; then args+=( --dry-run ); fi; \
		$(PY) -m accounting.publish.latest "$${args[@]}"; \
	'

release-check:
	@$(PY) scripts/check_release.py --public-root "$(ROOT)/public/accounting/latest_$(SCOPE_TAG)"


# ---------------------------------------------------------------------------
# Fixture / smoke path
# ---------------------------------------------------------------------------

.PHONY: smoke-ingest smoke-materialize smoke-core smoke-full smoke smoke-accounting
smoke-ingest:
	@$(call _guard_out_dir,$(SMOKE_OUT))
	@mkdir -p "$(SMOKE_OUT)"
	@$(PY) -m accounting.ledger.ingest \
		--mode smoke \
		--fixture "$(FIXTURE)" \
		--out-dir "$(SMOKE_OUT)" \
		--run-id "$(SMOKE_RUN_ID)"
	@$(MAKE) _check_ingest OUT_DIR="$(SMOKE_OUT)" MODE="smoke" FIXTURE="$(FIXTURE)"

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

smoke-core: smoke-materialize
	@echo "smoke-core passed fixture ingest/materialize semantic and cash checks"

smoke-full: smoke-core validate
	@$(PY) -m accounting.publish.latest --project-root "$(ROOT)" --dry-run >/dev/null
	@echo "smoke-full partial: fixture core + validation + publish dry-run passed"

smoke: smoke-core
smoke-accounting: smoke-core


# ---------------------------------------------------------------------------
# USD-CCL sidecars
# ---------------------------------------------------------------------------

.PHONY: smoke-usd-ccl-valuation smoke-usd-ccl-management-flows run-usd-ccl-valuation run-usd-ccl-management-flows
smoke-usd-ccl-valuation:
	@$(call _guard_out_dir,$(USD_CCL_SMOKE_OUT))
	@mkdir -p "$(USD_CCL_SMOKE_OUT)"
	@$(PY) -m accounting.valuation.usd_ccl \
		--ledger "$(USD_CCL_LEDGER_FIXTURE)" \
		--rates "$(USD_CCL_RATE_FIXTURE)" \
		--policy "$(USD_CCL_POLICY_FIXTURE)" \
		--output-dir "$(USD_CCL_SMOKE_OUT)" \
		--run-id "smoke-usd-ccl-valuation" \
		--source-scope-tag "FBPM"
	@test -s "$(USD_CCL_SMOKE_OUT)/ledger_valuation_usd_ccl.csv"
	@test -s "$(USD_CCL_SMOKE_OUT)/valuation_manifest.json"
	@test -s "$(USD_CCL_SMOKE_OUT)/valuation_validation.json"
	@test -s "$(USD_CCL_SMOKE_OUT)/valuation_coverage_by_year.csv"
	@echo "smoke-usd-ccl-valuation passed isolated fixture-only valuation checks"

smoke-usd-ccl-management-flows:
	@$(call _guard_out_dir,$(USD_CCL_MANAGEMENT_SMOKE_OUT))
	@mkdir -p "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/semantic" "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/valuation"
	@$(PY) -c 'from pathlib import Path; import pandas as pd; from accounting.marts.semantic import build_semantic_outputs; build_semantic_outputs(pd.read_csv("$(USD_CCL_MANAGEMENT_LEDGER_FIXTURE)"), Path("$(USD_CCL_MANAGEMENT_SMOKE_OUT)/semantic"))'
	@$(PY) -m accounting.valuation.usd_ccl \
		--ledger "$(USD_CCL_MANAGEMENT_LEDGER_FIXTURE)" \
		--rates "$(USD_CCL_RATE_FIXTURE)" \
		--policy "$(USD_CCL_POLICY_FIXTURE)" \
		--output-dir "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/valuation" \
		--run-id "smoke-usd-ccl-management-flows" \
		--source-scope-tag "synthetic"
	@$(PY) -m accounting.management.usd_ccl_flows \
		--ledger "$(USD_CCL_MANAGEMENT_LEDGER_FIXTURE)" \
		--semantic-audit "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/semantic/classification_audit.csv" \
		--valuation-sidecar "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/valuation/ledger_valuation_usd_ccl.csv" \
		--valuation-manifest "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/valuation/valuation_manifest.json" \
		--output-dir "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/management"
	@test -s "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/management/management_usd_ccl_flow_audit.csv"
	@test -s "$(USD_CCL_MANAGEMENT_SMOKE_OUT)/management/monthly_management_usd_ccl_components.csv"
	@echo "smoke-usd-ccl-management-flows passed isolated fixture-only eligibility checks"

run-usd-ccl-valuation:
	@$(call require_var,RUN_ROOT)
	@$(call require_var,CCL_RATES)
	@test -s "$(RUN_ROOT)/ledger_canonical.csv" || (echo "ERROR: missing $(RUN_ROOT)/ledger_canonical.csv"; exit 2)
	@test -s "$(CCL_RATES)" || (echo "ERROR: missing or empty CCL_RATES=$(CCL_RATES)"; exit 2)
	@$(PY) -m accounting.valuation.usd_ccl \
		--ledger "$(RUN_ROOT)/ledger_canonical.csv" \
		--rates "$(CCL_RATES)" \
		--policy "$(USD_CCL_FLOW_POLICY)" \
		--output-dir "$(RUN_ROOT)/valuations/usd_ccl" \
		--run-id "offline-usd-ccl-$(notdir $(RUN_ROOT))" \
		--source-scope-tag "$(notdir $(RUN_ROOT))" \
		--content-addressed \
		--mode offline

run-usd-ccl-management-flows:
	@$(call require_var,RUN_ROOT)
	@$(call require_var,CCL_RATES)
	@test -s "$(RUN_ROOT)/ledger_canonical.csv" || (echo "ERROR: missing $(RUN_ROOT)/ledger_canonical.csv"; exit 2)
	@test -s "$(RUN_ROOT)/classification_audit.csv" || (echo "ERROR: missing $(RUN_ROOT)/classification_audit.csv"; exit 2)
	@test -s "$(CCL_RATES)" || (echo "ERROR: missing or empty CCL_RATES=$(CCL_RATES)"; exit 2)
	@$(PY) -m accounting.management.usd_ccl_run \
		--run-root "$(RUN_ROOT)" \
		--rates "$(CCL_RATES)" \
		--policy "$(USD_CCL_FLOW_POLICY)"


# ---------------------------------------------------------------------------
# Live canonical path
# ---------------------------------------------------------------------------

.PHONY: run-ingest run-materialize _run_materialize_action run-canonical
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
		--sheet-name "$(ACCOUNT_SHEET_NAME)" \
		--boxes "$(BOXES)"
	@$(MAKE) _check_ingest OUT_DIR="$(RUN_OUT)" MODE="run"

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

run-canonical: run-materialize


# ---------------------------------------------------------------------------
# Debt stock/activity path
# ---------------------------------------------------------------------------

.PHONY: run-debt _run_debt_action run-debt-views run-debt-balance _run_debt_balance_action
run-debt: run-canonical _run_debt_action

_run_debt_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/ledger_canonical_all_status.csv" || (echo "ERROR: missing scoped all-status debt evidence at $(RUN_OUT)"; exit 2)
	@mkdir -p "$(RUN_DEBT_DIR)"
	@bash -eu -o pipefail -c '\
		args=( \
			--ledger-csv "$(RUN_OUT)/ledger_canonical_all_status.csv" \
			--write-dir "$(RUN_DEBT_DIR)" \
			--currencies "$(DEBT_CURRENCIES)" \
			--repayment-statuses "$(DEBT_REPAYMENT_STATUSES)" \
		); \
		if [ "$(DEBT_FULL_ONLY)" = "1" ]; then args+=( --full-only ); fi; \
		$(PY) -m accounting.debt.resolve "$${args[@]}"; \
		test -s "$(RUN_DEBT_DIR)/debt_open_items.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_allocations.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_repayment_events.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_resolution_timeline.csv"; \
	'

run-debt-views: run-debt _run_debt_balance_action
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
		$(PY) -m accounting.marts.treasury --run-root "$(RUN_OUT)"; \
		test -s "$(RUN_OUT)/monthly_cash_accountability.csv"; \
		test -s "$(RUN_OUT)/monthly_cash_accountability_qa.csv"; \
	'


# ---------------------------------------------------------------------------
# Governed metrics / annual dashboard
# ---------------------------------------------------------------------------

.PHONY: run-metrics metrics-from-run run-metrics-live _run_metrics_action run-dashboard
run-metrics: metrics-from-run
metrics-from-run: _run_metrics_action
run-metrics-live: run-debt-views _run_metrics_action _update_latest

_run_metrics_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/monthly_flow_semantic_split.csv" || (echo "ERROR: missing governed semantic split at $(RUN_OUT)"; exit 2)
	@test -s "$(RUN_OUT)/monthly_operating_statement.csv" || (echo "ERROR: missing governed operating statement at $(RUN_OUT)"; exit 2)
	@test -s "$(RUN_OUT)/monthly_cash_close.csv" || (echo "ERROR: missing governed cash close at $(RUN_OUT)"; exit 2)
	@mkdir -p "$(RUN_METRICS_DIR)"
	@$(PY) -m accounting.metrics.build \
		--run-root "$(RUN_OUT)" \
		--out-dir "$(RUN_METRICS_DIR)"
	@test -s "$(RUN_METRICS_DIR)/build_manifest.json"
	@test -s "$(RUN_METRICS_DIR)/metric_contract_frontier.csv"
	@test -s "$(RUN_METRICS_DIR)/frontend_metric_series.csv"
	@test -s "$(RUN_METRICS_DIR)/metrics_frontier_qa.csv"
	@test -s "$(RUN_METRICS_DIR)/frontier_source_qa.csv"
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_metrics.csv"
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_contract.csv"
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_qa.csv"
	@test -s "$(RUN_METRICS_DIR)/annual_flow_membership.csv"
	@test -s "$(RUN_METRICS_DIR)/artifact_contracts.csv"
	@test -s "$(RUN_METRICS_DIR)/source_contract_qa.csv"

run-dashboard: run-metrics
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_metrics.csv"
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_contract.csv"
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_qa.csv"


# ---------------------------------------------------------------------------
# Composite operations
# ---------------------------------------------------------------------------

.PHONY: run-full run-accounting run-accounting-full run-downstream-from-ledger run-live-light assert-live-light-no-debt
run-full: run-debt-views run-dashboard _update_latest publish-latest release-check

run-accounting: run-accounting-full
run-accounting-full: run-full

run-downstream-from-ledger:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/ledger_canonical.csv" || (echo "ERROR: missing ledger_canonical.csv at $(RUN_OUT)"; exit 2)
	@$(MAKE) _run_materialize_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)" FREQ="$(FREQ)"
	@$(MAKE) _run_debt_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
	@$(MAKE) _run_debt_balance_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
	@$(MAKE) _run_metrics_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
	@$(MAKE) _update_latest RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"

run-live-light: run-materialize
	@$(MAKE) _run_metrics_action RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_metrics.csv"
	@$(MAKE) update-latest-light RUN_STAMP="$(RUN_STAMP)" OUT="$(OUT)"
	@echo "[LIVE-LIGHT] refreshed governed non-debt outputs"

assert-live-light-no-debt:
	@before=$$(find "$(OUT)/debt_resolution" -maxdepth 1 -type d -name 'LIVE_*' 2>/dev/null | wc -l); \
	$(MAKE) run-live-light; \
	after=$$(find "$(OUT)/debt_resolution" -maxdepth 1 -type d -name 'LIVE_*' 2>/dev/null | wc -l); \
	if [ "$$before" != "$$after" ]; then \
		echo "ERROR: run-live-light created debt artifacts: before=$$before after=$$after"; \
		exit 2; \
	fi; \
	echo "OK: run-live-light did not create debt artifacts"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

.PHONY: _check_ingest _check_materialize
_check_ingest:
	@$(call _guard_out_dir,$(OUT_DIR))
	@OUT_DIR="$(OUT_DIR)" MODE="$(MODE)" FIXTURE="$(FIXTURE)" $(PY) scripts/check_ingest.py

_check_materialize:
	@$(call _guard_out_dir,$(OUT_DIR))
	@OUT_DIR="$(OUT_DIR)" MODE="$(MODE)" FREQ="$(FREQ)" $(PY) scripts/check_materialize.py


# ---------------------------------------------------------------------------
# Professional presentation over governed artifacts
# ---------------------------------------------------------------------------

.PHONY: professional-drilldowns professional-linked-digest
professional-drilldowns:
	@$(PY) -m accounting.professional.drilldown \
		--repo-root "$(ROOT)" \
		--pack "$(ROOT)/out/professional_pack/latest_FBPM" \
		--run-root "$(ROOT)/out/run/accounting/latest_FBPM"

professional-linked-digest:
	@$(PY) -m accounting.professional.render_linked_digest \
		--repo-root "$(ROOT)" \
		--pack "$(ROOT)/out/professional_pack/latest_FBPM"

.PHONY: run run-all
run: run-accounting
run-all: run-accounting
	@echo "[RUN] done. latest -> $(RUN_REL)"
