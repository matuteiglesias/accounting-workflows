---
id: notes/environment_bootstrap
title: "Environment Bootstrap (Wave 1)"
sidebar_label: "Environment Bootstrap (Wave 1)"
---

# Environment Bootstrap (Wave 1)

Status: authority draft
Last reviewed: 2026-05-22
Audience: operators, developers, coding agents

## Purpose

Make local and automation environments deterministic enough to separate:
- bootstrap/runtime failures, from
- real accounting pipeline regressions.

This is the first-response environment contract for this repository.

## Source-of-truth anchors

This guide is anchored to:
- `Makefile` (command surface, env vars, fixture defaults, run/smoke orchestration)
- `README.md` (run and logging operational expectations)
- stage entrypoints (`accounting.ledger.ingest`, `accounting.stage_d.materialize`, `accounting.views`, metrics/debt/human/publish modules)

## Python runtime policy

Observed command behavior:
- `make doctor` prints active Python version and compile-checks command modules.

Policy (current):
- Use Python **3.12.x** for operator and automation runtime.
- If you change runtime major/minor, re-run full command-surface checks before promoting.

## Dependency policy

Current repo state:
- Pipeline modules import `pandas` across ingest/materialize/views/metrics/debt/human.
- `make smoke` fails immediately when `pandas` is missing.
- No dependency manifest (`requirements*.txt` or `pyproject.toml`) is currently present.

Operational implication:
- Environment bootstrap is currently sensitive to local machine history.
- This is the highest-likelihood cause of "looks broken after refactor" false alarms.

## Bootstrap steps

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

Because there is no authoritative dependency manifest yet, install minimum required packages iteratively from import failures.

Recommended immediate follow-up for repo maintainers:
- add one authoritative dependency manifest (`requirements.txt` or `pyproject.toml`) and pin this guide to it.

### 3) Configure required environment variables

Variables used by Makefile and runtime modules:
- `ACCOUNT_SA`
- `ACCOUNT_SHEET_URL`
- `ACCOUNT_SHEET_NAME` (default: `C. Long Ledger`)
- `FIXTURE` (default: `fixtures/ledger_fixture.csv`)

Example `.env`:

```bash
ACCOUNT_SA=/absolute/path/to/service-account.json
ACCOUNT_SHEET_URL=https://docs.google.com/spreadsheets/d/...
ACCOUNT_SHEET_NAME=C. Long Ledger
FIXTURE=/workspace/accounting-workflows/fixtures/ledger_fixture.csv
```

## Sanity command sequence

Run in this exact order:

```bash
make help
make doctor
make smoke
```

Interpretation:
- `make help`: command surface discovery and canonical target visibility.
- `make doctor`: python runtime + compile/import surface verification.
- `make smoke`: fixture/offline pipeline confidence through views.

Failure classification:
- Missing module/import error -> bootstrap/dependency issue.
- Invariant/contract output failure -> probable logic regression.

## Optional deeper CLI checks

```bash
python -m accounting.ledger.ingest --help
python -m accounting.stage_d.materialize --help
python -m accounting.views --help
python -m accounting.metrics.build --help
python -m accounting.debt.resolve --help
python -m accounting.human.document --help
python -m accounting.publish.latest --help
```

## Evidence map

### Commands validated (2026-05-22)

- `make help` -> pass
- `make doctor` -> pass
- `make smoke` -> fails in current environment when `pandas` is missing
- `rg --files -g 'requirements*.txt' -g 'pyproject.toml'` -> no manifest found

### Code anchors

- `Makefile`
- `README.md`
- `accounting/ledger/ingest.py`
- `accounting/stage_d/materialize.py`
- `accounting/views.py`
- `accounting/metrics/build.py`
- `accounting/debt/resolve.py`
- `accounting/human/document.py`
- `accounting/publish/latest.py`
- `accounting/support/env.py`

### Known assumptions

- Python 3.12.x is policy-by-observation until pinned in repo config.
- Final dependency set remains unpinned until a manifest is added.
