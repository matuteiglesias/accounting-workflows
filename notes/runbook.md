**Runbook: Output Contracts (v1)**

It captures the **expected outputs by stage**, and the **governance rules** (naming, currency-safety, Box consistency, and artifact registration).

---

# 📘 Runbook — Output Contracts v1

**Repo:** `RAG_Sync/Accounting/4_Analysis_Workflows/src`
**Date:** 2026-01-09
**Purpose:** Freeze reference behavior of ingestion, materialization, and visualization stages to ensure reproducibility and currency-safe auditability.

---

## 1. Stage A — Ingest

### Purpose

Convert raw ledgers (CSV or GSheet) into canonical Pandas DataFrame with normalized fields and consistent typing.

### Primary Output

| File                                               | Description                                                           |
| -------------------------------------------------- | --------------------------------------------------------------------- |
| `out/run/accounting/<RUN_ID>/ledger_canonical.csv` | Canonical ledger containing all transactions, post-ingest validation. |

### Expected Columns

```
Date, Box, payer, receiver, amount, Currency, Flujo, Tipo,
Lugar, Detalle, issuer, account_id, status, medio,
due_date, posted_date, rate (optional), transaction_id
```

### Key Rules

* **Currency-safe**: all rows include `Currency`; FX conversions optional but logged in `rate`.
* **Presence of Box**: every transaction must map to one and only one `Box` (“Family Business”, “Property Management”, etc.).
* **Normalization**:

  * Dates → ISO or `%Y-%m-%d`
  * `payer`, `receiver` → canonical strings (no whitespace, no aliases)
  * `status` normalized via `TxStatus` enum: `{pagado, pendiente, cancelado}`
* **Validation checks**:

  * Must have at least one of `payer` or `receiver`.
  * Amounts must be nonnegative (Pydantic `ge=0` rule in `Money`).

---

## 2. Stage D — Materialize

### Purpose

Generate derived CSV aggregates (flows, parties, boxes) from the canonical ledger.

### Primary Outputs

| Output File                             | Description                                                        |
| --------------------------------------- | ------------------------------------------------------------------ |
| `per_flow_time_long.freq=<freq>.csv`    | Aggregated inflows/outflows per flow type, with counts and totals. |
| `per_party_time_long.freq=<freq>.csv`   | Aggregated signed amounts per party and role.                      |
| `box_balance_time_long.freq=<freq>.csv` | Net and cumulative balances per Box, currency, and time period.    |
| `daily_cash_position.csv`               | Rolling balances per Box/party/day.                                |

### Common Columns

```
TimePeriod, TimePeriod_ts_end, Box, Currency,
[Flujo | Tipo | party | role], amount, n_tx
```

### Special for `box_balance_time_long`

```
TimePeriod, Date_end, Box, Currency,
in_amt, out_amt, net, cum_net
```

Used for high-level audits and cross-checking flows vs party aggregates.

### Artifact Registration

Each output must append to `out_arts` list:

```python
artifact_from_path(
  name="<basename>",
  path=<Path>,
  stage="D.materialize",
  mode=args.mode,
  run_id=run_id,
  role="derived",
  root_dir=out_dir,
  content_type="text/csv"
)
```

### Rules

* **Currency safety**: All aggregates grouped by `Currency` before summation.
* **Box integrity**: Each row must carry `Box`; group-level sums without Box are invalid.
* **File naming**:
  `*_time_long.freq=<freq>.csv` (W, M, Q allowed)
  or `daily_cash_position.csv` (fixed name).
* **Atomic write**: `_atomic_write_csv` ensures safe overwrites.
* **Logging**: `LOG.info` confirms written row counts.

---

## 3. Stage E — Reports

### Purpose

Aggregate the materialized outputs into analytic views (summary tables, report packs).

### Primary Outputs

| File                   | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `reports_summary.json` | Summary of report generation, counts by Box, flows, anomalies. |
| `reports/`             | Folder with final CSVs or markdowns for dashboards and PDFs.   |

### Rules

* Each report must be traceable to a materialized input (`meta/stage_D_materialize.json` references).
* JSON includes `dataset_hash`, `timestamp`, `run_id`.

---

## 4. Stage V — Views

### Purpose

Visualization and cross-stage diagnostics (plot data, sanity checks).

### Primary Outputs

| File                | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| `views_sanity.json` | QA and validation results on aggregates (missing boxes, currency mismatches, etc.). |
| `storypack/`        | Plot-ready data and figures for narrative synthesis.                                |

### View Models

`views.py` reads from materialized CSVs and enforces:

* **presence of Box** and **Currency**,
* **matching of sums** between Box aggregates and per-party aggregates,
* **visual coherence** (per Box, per Flow).

### Rules

* `views_sanity.json` must include keys:

  ```
  { "missing_boxes": [], "fx_inconsistencies": [], "unmatched_pairs": [], "drift_detected": false }
  ```
* **Currency normalization**: conversions always logged via `rate` if cross-currency aggregates are performed.
* **Naming convention for plots**:
  `out/storypack/<RUN_ID>/<plot_name>.png|svg|json`

---

## 5. Meta Artifacts and Governance

### `artifacts.jsonl`

Located at `out/run/accounting/<RUN_ID>/meta/artifacts.jsonl`

Each artifact line (JSONL record) must contain:

```json
{
  "name": "per_party_time_long",
  "path": "out/run/accounting/<RUN_ID>/per_party_time_long.freq=W.csv",
  "stage": "D.materialize",
  "run_id": "<RUN_ID>",
  "mode": "<mode>",
  "role": "derived",
  "root_dir": "out/run/accounting/<RUN_ID>",
  "content_type": "text/csv",
  "timestamp": "2026-01-09T10:00:00"
}
```

**Completeness rule**:
All Stage D and E outputs must appear here.
Downstream pipelines (reports, views) must not depend on unregistered files.

---

## 6. Cross-Stage Invariants

| Invariant            | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| Box completeness     | Each transaction’s Box is propagated through all derived views.                 |
| Currency consistency | No aggregate mixes ARS and USD. Conversion requires explicit rate.              |
| Deterministic Run ID | Every run produces a unique `<RUN_ID>` (UTC timestamped).                       |

---


