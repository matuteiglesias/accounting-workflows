# Accounting backend control plane
# Official live path:
#   run-canonical -> run-debt -> run-metrics -> run-reports -> publication
#
# Only run-ingest/run-canonical/run-full pull live source inputs. Downstream stage
# targets operate on the exact RUN_ID selected by the caller and are replayable.
# Materialization owns the canonical monthly semantic and cash artifacts.
# Reports consume governed artifacts and do not introduce accounting authority.
# There is no generic views stage, parallel metric_values engine, or command alias layer.

.DEFAULT_GOAL := help
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
RUN_STAMP := $(RUN_STAMP)
RUN_BASE := $(OUT)/run/accounting
RUN_ID ?= $(RUN_STAMP)_$(SCOPE_TAG)
RUN_ID := $(RUN_ID)
RUN_OUT := $(RUN_BASE)/$(RUN_ID)
RUN_METRICS_DIR := $(OUT)/metrics/$(RUN_ID)
RUN_DEBT_DIR := $(OUT)/debt_resolution/$(RUN_ID)
RUN_REPORTS_BASE := $(OUT)/reports
RUN_REPORTS_DIR := $(RUN_REPORTS_BASE)/$(RUN_ID)
REPORT_BROWSER_BIN ?=

DEBT_CURRENCIES ?= USD
DEBT_REPAYMENT_STATUSES ?= pagado
DRY_RUN ?= 0


define require_var
	@if [ -z "$($(1))" ]; then echo "ERROR: missing required var: $(1)"; exit 2; fi
endef

define _guard_out_dir
	@if [ -z "$(1)" ]; then echo "ERROR: OUT_DIR empty"; exit 2; fi
endef


# ---------------------------------------------------------------------------
# Help / validation
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
	@echo "Live operations:"
	@echo "  make run-canonical      # live ingest -> governed materialization"
	@echo "  make run-full           # ordered full live path -> reports -> publication"
	@echo "  make run-env            # load ENV_FILE, then run-full"
	@echo ""
	@echo "Exact-run stage replay (set RUN_ID=<existing run id>):"
	@echo "  make run-materialize    # canonical ledger -> governed materialization"
	@echo "  make run-debt           # debt resolution -> position/activity + treasury"
	@echo "  make run-metrics        # governed frontier + annual metrics"
	@echo "  make run-reports        # annual + treasury HTML/PDF report bundle"
	@echo ""
	@echo "Focused source / sidecar operations:"
	@echo "  make run-ingest"
	@echo "  make run-usd-ccl-valuation RUN_ROOT=<exact-run> CCL_RATES=<local.csv>"
	@echo "  make run-usd-ccl-management-flows RUN_ROOT=<exact-run> CCL_RATES=<local.csv>"
	@echo ""
	@echo "Publication / professional:"
	@echo "  make publish-latest"
	@echo "  make publish-reports"
	@echo "  make release-check"
	@echo "  make professional-drilldowns"
	@echo "  make professional-linked-digest"
	@echo ""
	@echo "Key vars: OUT=out RUN_ID=<exact-run-id> BOXES='Family Business,Property Management' REPORT_BROWSER_BIN=<chromium>"
	@echo ""

.PHONY: run-env
run-env:
	@bash -lc 'set -a; source "$(ENV_FILE)"; set +a; $(MAKE) run-full'

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
	rm -rf "$(OUT)/smoke/accounting" "$(OUT)/run/accounting" "$(OUT)/metrics" "$(OUT)/debt_resolution" "$(OUT)/reports" "$(ROOT)/public/accounting/latest" "$(ROOT)/public/accounting/latest_$(SCOPE_TAG)" "$(ROOT)/public/reports/latest" "$(ROOT)/public/reports/latest_$(SCOPE_TAG)"


# ---------------------------------------------------------------------------
# Latest pointers / publication
# ---------------------------------------------------------------------------

.PHONY: _update_latest
_update_latest:
	@echo "[RUN][LATEST] run=$(RUN_ID) including governed reports"
	@$(PY) -m accounting.support.latest --scope-tag "$(SCOPE_TAG)" --target "$(RUN_ID)" \
		--base "$(RUN_BASE)" --base "$(OUT)/debt_resolution" --base "$(OUT)/metrics" --base "$(RUN_REPORTS_BASE)"

.PHONY: publish-latest publish-reports release-check
publish-latest:
	@bash -eu -o pipefail -c '\
		args=( --project-root "$(ROOT)" --scope-tag "$(SCOPE_TAG)" --clean ); \
		if [ "$(DRY_RUN)" = "1" ]; then args+=( --dry-run ); fi; \
		$(PY) -m accounting.publish.latest "$${args[@]}"; \
	'

publish-reports:
	@bash -eu -o pipefail -c '\
		args=( --project-root "$(ROOT)" --scope-tag "$(SCOPE_TAG)" ); \
		if [ "$(DRY_RUN)" = "1" ]; then args+=( --dry-run ); fi; \
		$(PY) -m accounting.reports.publish "$${args[@]}"; \
	'

release-check:
	@$(PY) scripts/check_release.py --public-root "$(ROOT)/public/accounting/latest_$(SCOPE_TAG)"


# ---------------------------------------------------------------------------
# Fixture / smoke path
# ---------------------------------------------------------------------------

.PHONY: smoke-ingest smoke-materialize smoke-core smoke-full
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
# Canonical ledger / materialization
# ---------------------------------------------------------------------------

.PHONY: run-ingest run-materialize _run_materialize_action run-canonical
run-ingest:
	@$(call _guard_out_dir,$(RUN_OUT))
	@$(call require_var,ACCOUNT_SHEET_URL)
	@mkdir -p "$(RUN_OUT)"
	@$(PY) -m accounting.ledger.ingest \
		--mode run \
		--out-dir "$(RUN_OUT)" \
		--run-id "$(RUN_ID)" \
		--service-account "$(ACCOUNT_SA)" \
		--sheet-url "$(ACCOUNT_SHEET_URL)" \
		--sheet-name "$(ACCOUNT_SHEET_NAME)" \
		--boxes "$(BOXES)"
	@$(MAKE) _check_ingest OUT_DIR="$(RUN_OUT)" MODE="run"

run-materialize: _run_materialize_action

_run_materialize_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_OUT)/ledger_canonical.csv" || (echo "ERROR: missing ledger_canonical.csv at $(RUN_OUT)"; exit 2)
	@$(PY) -m accounting.stage_d.materialize \
		--out-dir "$(RUN_OUT)" \
		--freq "$(FREQ)" \
		--force 0 \
		--mode run \
		--run-id "$(RUN_ID)"
	@$(MAKE) _check_materialize OUT_DIR="$(RUN_OUT)" MODE="run" FREQ="$(FREQ)"
	@test -s "$(RUN_OUT)/classification_audit.csv"
	@test -s "$(RUN_OUT)/classification_audit_summary.csv"
	@test -s "$(RUN_OUT)/monthly_flow_semantic_split.csv"
	@test -s "$(RUN_OUT)/monthly_operating_statement.csv"
	@test -s "$(RUN_OUT)/monthly_operating_statement_qa.csv"
	@test -s "$(RUN_OUT)/semantic_leakage_qa.csv"
	@test -s "$(RUN_OUT)/monthly_cash_close.csv"
	@test -s "$(RUN_OUT)/monthly_cash_close_qa.csv"

run-canonical: run-ingest
	@$(MAKE) run-materialize RUN_ID="$(RUN_ID)" OUT="$(OUT)" FREQ="$(FREQ)" BOXES="$(BOXES)"


# ---------------------------------------------------------------------------
# Debt position/activity + treasury
# ---------------------------------------------------------------------------

.PHONY: run-debt _run_debt_resolution_action _run_debt_products_action
run-debt: _run_debt_resolution_action
	@$(MAKE) _run_debt_products_action RUN_ID="$(RUN_ID)" OUT="$(OUT)" BOXES="$(BOXES)"

_run_debt_resolution_action:
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
		$(PY) -m accounting.debt.resolve "$${args[@]}"; \
		test -s "$(RUN_DEBT_DIR)/debt_open_items.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_allocations.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_repayment_events.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_resolution_timeline.csv"; \
	'

_run_debt_products_action:
	@$(call _guard_out_dir,$(RUN_OUT))
	@test -s "$(RUN_DEBT_DIR)/debt_open_items.csv" || (echo "ERROR: missing debt_open_items.csv at $(RUN_DEBT_DIR)"; exit 2)
	@bash -eu -o pipefail -c '\
		$(PY) -m accounting.debt.balance_views \
			--open-items "$(RUN_DEBT_DIR)/debt_open_items.csv" \
			--allocations "$(RUN_DEBT_DIR)/debt_allocations.csv" \
			--write-dir "$(RUN_DEBT_DIR)"; \
		test -s "$(RUN_DEBT_DIR)/debt_balance_daily.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_balance_monthly.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_balance_quarterly.csv"; \
		test -s "$(RUN_DEBT_DIR)/debt_balance_yearly.csv"; \
		$(PY) -m accounting.marts.debt \
			--debt-dir "$(RUN_DEBT_DIR)" \
			--write-dir "$(RUN_OUT)"; \
		test -s "$(RUN_OUT)/monthly_debt_position.csv"; \
		test -s "$(RUN_OUT)/monthly_debt_position_qa.csv"; \
		test -s "$(RUN_OUT)/monthly_debt_activity.csv"; \
		test -s "$(RUN_OUT)/monthly_debt_activity_qa.csv"; \
		test -s "$(RUN_OUT)/monthly_debt_repayment_detail.csv"; \
		$(PY) -m accounting.marts.treasury --run-root "$(RUN_OUT)"; \
		test -s "$(RUN_OUT)/monthly_cash_accountability.csv"; \
		test -s "$(RUN_OUT)/monthly_cash_accountability_qa.csv"; \
		$(PY) -m accounting.marts.accountability --run-root "$(RUN_OUT)"; \
		test -f "$(RUN_OUT)/family_business_accountability_cycles.csv"; \
		test -f "$(RUN_OUT)/household_monthly_control.csv"; \
		$(PY) -m accounting.grooming --ledger "$(RUN_OUT)/ledger_canonical_all_status.csv" \
			--debt-reconciliation "$(RUN_DEBT_DIR)/debt_status_reconciliation.csv" \
			--out-dir "$(RUN_OUT)/private_review"; \
	'


# ---------------------------------------------------------------------------
# Governed metrics / annual dashboard
# ---------------------------------------------------------------------------

.PHONY: run-metrics _run_metrics_action
run-metrics: _run_metrics_action

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


# ---------------------------------------------------------------------------
# Governed human reports
# ---------------------------------------------------------------------------

.PHONY: run-reports _run_reports_action
run-reports: _run_reports_action

_run_reports_action:
	@$(call _guard_out_dir,$(RUN_REPORTS_DIR))
	@test -s "$(RUN_OUT)/monthly_cash_accountability.csv" || (echo "ERROR: missing treasury accountability at $(RUN_OUT)"; exit 2)
	@test -s "$(RUN_OUT)/monthly_cash_accountability_qa.csv" || (echo "ERROR: missing treasury accountability QA at $(RUN_OUT)"; exit 2)
	@test -s "$(RUN_METRICS_DIR)/annual_balance_dashboard_metrics.csv" || (echo "ERROR: missing annual dashboard metrics at $(RUN_METRICS_DIR)"; exit 2)
	@bash -eu -o pipefail -c '\
		args=( \
			--run-root "$(RUN_OUT)" \
			--metrics-dir "$(RUN_METRICS_DIR)" \
			--out-dir "$(RUN_REPORTS_DIR)" \
			--scope-tag "$(SCOPE_TAG)" \
		); \
		if [ -n "$(REPORT_BROWSER_BIN)" ]; then args+=( --browser-bin "$(REPORT_BROWSER_BIN)" ); fi; \
		$(PY) -m accounting.reports.build "$${args[@]}"; \
		test -s "$(RUN_REPORTS_DIR)/report_catalog.json"; \
		test -s "$(RUN_REPORTS_DIR)/annual_management/report.html"; \
		test -s "$(RUN_REPORTS_DIR)/annual_management/report.pdf"; \
		test -s "$(RUN_REPORTS_DIR)/annual_management/report_manifest.json"; \
		test -s "$(RUN_REPORTS_DIR)/treasury_accountability/report.html"; \
		test -s "$(RUN_REPORTS_DIR)/treasury_accountability/report.pdf"; \
		test -s "$(RUN_REPORTS_DIR)/treasury_accountability/report_manifest.json"; \
		test -s "$(RUN_REPORTS_DIR)/debt_accountability/report.html"; \
		test -s "$(RUN_REPORTS_DIR)/debt_accountability/report.pdf"; \
		test -s "$(RUN_REPORTS_DIR)/debt_accountability/report_manifest.json"; \
	'


# ---------------------------------------------------------------------------
# Ordered live composite
# ---------------------------------------------------------------------------

.PHONY: run-full
run-full: run-canonical
	@$(MAKE) run-debt RUN_ID="$(RUN_ID)" OUT="$(OUT)" BOXES="$(BOXES)"
	@$(MAKE) run-metrics RUN_ID="$(RUN_ID)" OUT="$(OUT)" BOXES="$(BOXES)"
	@$(MAKE) run-reports RUN_ID="$(RUN_ID)" OUT="$(OUT)" BOXES="$(BOXES)" REPORT_BROWSER_BIN="$(REPORT_BROWSER_BIN)"
	@$(MAKE) _update_latest RUN_ID="$(RUN_ID)" OUT="$(OUT)" BOXES="$(BOXES)"
	@$(MAKE) publish-latest OUT="$(OUT)" BOXES="$(BOXES)"
	@$(MAKE) publish-reports OUT="$(OUT)" BOXES="$(BOXES)"
	@$(MAKE) release-check OUT="$(OUT)" BOXES="$(BOXES)"
	@echo "[RUN] done. latest -> $(RUN_ID)"


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

.PHONY: _professional-debt-tables professional-drilldowns professional-linked-digest
_professional-debt-tables:
	@$(PY) -m accounting.professional.debt_tables \
		--run-root "$(ROOT)/out/run/accounting/latest_FBPM" \
		--tables-dir "$(ROOT)/out/professional_pack/latest/tables"

professional-drilldowns: _professional-debt-tables
	@$(PY) -m accounting.professional.drilldown \
		--repo-root "$(ROOT)" \
		--pack "$(ROOT)/out/professional_pack/latest" \
		--run-root "$(ROOT)/out/run/accounting/latest_FBPM"

professional-linked-digest:
	@$(PY) -m accounting.professional.render_linked_digest \
		--repo-root "$(ROOT)" \
		--pack "$(ROOT)/out/professional_pack/latest"
