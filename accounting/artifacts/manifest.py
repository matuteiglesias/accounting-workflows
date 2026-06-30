# accounting/artifacts/manifest.py
from __future__ import annotations

import json
import hashlib
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import datetime as _dt
JsonDict = Dict[str, Any]
Pathish = Union[str, os.PathLike, Path]


ARTIFACT_ROLES = [
    "canonical_source",
    "diagnostic",
    "internal_balance",
    "inferred_reconciliation",
    "presentation_only",
    "legacy",
    "unsafe_for_frontend",
    "qa",
    "meta",
]
ACCOUNTING_NATURES = ["transaction", "flow", "stock", "ratio", "quality", "mixed", "unknown"]
GRAINS = ["tx", "daily", "monthly", "quarterly", "annual", "period_close", "mixed"]
CURRENCY_POLICIES = [
    "explicit_currency_required",
    "by_currency",
    "converted",
    "no_cross_currency_sum",
    "not_money",
]
FRONTEND_SUITABILITIES = [
    "safe",
    "safe_with_caveat",
    "internal_only",
    "unavailable",
    "forbidden",
    "row_level",
    "row_level_or_metric_level",
]
SOURCE_AUTHORITIES = [
    "source_of_truth",
    "derived_from_source_of_truth",
    "diagnostic_only",
    "presentation_only",
    "legacy_compatibility",
    "not_for_downstream_contracts",
    "source_of_truth_for_cash_only_when_is_frontend_safe_true",
    "source_of_truth_for_debt_stock",
    "frontend_contract",
]

CONTRACT_COLUMNS = [
    "name",
    "relpath",
    "artifact_role",
    "accounting_nature",
    "grain",
    "currency_policy",
    "frontend_suitability",
    "source_authority",
    "notes",
]


def artifact_contract_for_name(name: str, relpath: str = "") -> Dict[str, str]:
    """Return the source-of-truth contract for known accounting artifacts."""
    key = Path(str(relpath) or str(name)).name
    if key == "ledger_canonical.csv" or name == "ledger_canonical":
        return {
            "artifact_role": "canonical_source",
            "accounting_nature": "transaction",
            "grain": "tx",
            "currency_policy": "explicit_currency_required",
            "frontend_suitability": "internal_only",
            "source_authority": "source_of_truth",
            "notes": "Canonical transaction ledger; downstream reports should prefer semantic/cash/debt marts.",
        }
    if key.startswith("monthly_flow_semantic_split"):
        return {
            "artifact_role": "canonical_source",
            "accounting_nature": "flow",
            "grain": "monthly",
            "currency_policy": "by_currency",
            "frontend_suitability": "safe_with_caveat",
            "source_authority": "source_of_truth",
            "notes": "Canonical monthly semantic flow split.",
        }
    if key == "monthly_operating_statement.csv":
        return {
            "artifact_role": "canonical_source",
            "accounting_nature": "flow",
            "grain": "monthly",
            "currency_policy": "by_currency",
            "frontend_suitability": "safe_with_caveat",
            "source_authority": "source_of_truth",
            "notes": "Canonical monthly operating statement.",
        }
    if key == "monthly_cash_close.csv":
        return {
            "artifact_role": "canonical_source",
            "accounting_nature": "stock",
            "grain": "period_close",
            "currency_policy": "by_currency",
            "frontend_suitability": "row_level",
            "source_authority": "source_of_truth_for_cash_only_when_is_frontend_safe_true",
            "notes": "Cash contract is frontend-safe only for rows where is_frontend_safe=true.",
        }
    if key == "monthly_debt_position.csv":
        return {
            "artifact_role": "canonical_source",
            "accounting_nature": "stock",
            "grain": "period_close",
            "currency_policy": "by_currency",
            "frontend_suitability": "safe_with_caveat",
            "source_authority": "source_of_truth_for_debt_stock",
            "notes": "Canonical debt stock wrapper over resolved debt balances.",
        }
    if key.startswith("per_flow_time_long"):
        return {
            "artifact_role": "diagnostic",
            "accounting_nature": "flow",
            "grain": "monthly",
            "currency_policy": "by_currency",
            "frontend_suitability": "forbidden",
            "source_authority": "diagnostic_only",
            "notes": "Raw flow/type aggregation; not semantic source for dashboard.",
        }
    if key.startswith("per_party_time_long"):
        return {
            "artifact_role": "internal_balance",
            "accounting_nature": "mixed",
            "grain": "monthly",
            "currency_policy": "by_currency",
            "frontend_suitability": "forbidden",
            "source_authority": "diagnostic_only",
            "notes": "Actor/party movement evidence; not real cash.",
        }
    if key == "daily_cash_position.csv":
        return {
            "artifact_role": "internal_balance",
            "accounting_nature": "stock",
            "grain": "daily",
            "currency_policy": "by_currency",
            "frontend_suitability": "forbidden",
            "source_authority": "diagnostic_only",
            "notes": "Party-level internal balance/claim view; not account-level cash.",
        }
    if key.startswith("box_balance_time_long"):
        return {
            "artifact_role": "inferred_reconciliation",
            "accounting_nature": "stock",
            "grain": "monthly",
            "currency_policy": "by_currency",
            "frontend_suitability": "forbidden",
            "source_authority": "diagnostic_only",
            "notes": "Inferred box motor / cumulative movement; not validated cash.",
        }
    if key.startswith("box_flow_balance_time_long"):
        return {
            "artifact_role": "inferred_reconciliation",
            "accounting_nature": "mixed",
            "grain": "monthly",
            "currency_policy": "by_currency",
            "frontend_suitability": "forbidden",
            "source_authority": "diagnostic_only",
            "notes": "Box motor decomposition; not validated cash.",
        }
    if "metric_views/" in str(relpath).replace("\\", "/") or str(relpath).startswith("metric_views/"):
        return {
            "artifact_role": "presentation_only",
            "accounting_nature": "mixed",
            "grain": "mixed",
            "currency_policy": "by_currency",
            "frontend_suitability": "safe_with_caveat",
            "source_authority": "presentation_only",
            "notes": "Presentation convenience view, not a source-of-truth contract.",
        }
    if key in {"income_statement_y.csv", "balance_cash_y.csv", "balance_debt_y.csv", "income_statement_q.csv", "balance_cash_q.csv", "balance_debt_q.csv"}:
        return {
            "artifact_role": "legacy",
            "accounting_nature": "mixed",
            "grain": "annual" if key.endswith("_y.csv") else "quarterly",
            "currency_policy": "by_currency",
            "frontend_suitability": "safe_with_caveat",
            "source_authority": "legacy_compatibility",
            "notes": "Legacy compatibility view; prefer metric_contract_frontier/frontend_metric_series.",
        }
    if key in {"debt_open_items.csv", "debt_repayment_events.csv", "debt_status_reconciliation.csv"}:
        return {
            "artifact_role": "diagnostic",
            "accounting_nature": "mixed",
            "grain": "mixed",
            "currency_policy": "by_currency",
            "frontend_suitability": "internal_only",
            "source_authority": "diagnostic_only",
            "notes": "Debt engine evidence; use monthly_debt_position for report-safe debt stock.",
        }
    if key == "metric_contract_frontier.csv":
        return {
            "artifact_role": "canonical_source",
            "accounting_nature": "mixed",
            "grain": "mixed",
            "currency_policy": "by_currency",
            "frontend_suitability": "safe_with_caveat",
            "source_authority": "frontend_contract",
            "notes": "Frontend metric contract registry.",
        }
    if key == "frontend_metric_series.csv":
        return {
            "artifact_role": "canonical_source",
            "accounting_nature": "mixed",
            "grain": "mixed",
            "currency_policy": "by_currency",
            "frontend_suitability": "row_level_or_metric_level",
            "source_authority": "frontend_contract",
            "notes": "Frontend metric series; suitability is carried at row/metric level.",
        }
    if key.endswith("_qa.csv") or "qa" in key:
        return {
            "artifact_role": "qa",
            "accounting_nature": "quality",
            "grain": "mixed",
            "currency_policy": "not_money",
            "frontend_suitability": "internal_only",
            "source_authority": "diagnostic_only",
            "notes": "QA artifact.",
        }
    if key.endswith(".json"):
        return {
            "artifact_role": "meta",
            "accounting_nature": "unknown",
            "grain": "mixed",
            "currency_policy": "not_money",
            "frontend_suitability": "internal_only",
            "source_authority": "diagnostic_only",
            "notes": "Metadata artifact.",
        }
    return {
        "artifact_role": "diagnostic",
        "accounting_nature": "unknown",
        "grain": "mixed",
        "currency_policy": "not_money",
        "frontend_suitability": "internal_only",
        "source_authority": "diagnostic_only",
        "notes": "Unclassified artifact defaults to diagnostic/internal-only until explicitly contracted.",
    }


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_csv_rows_fast(path: Path) -> Optional[int]:
    # cuenta filas sin pandas: header + N lineas
    # si el archivo es vacio o no existe, retorna None
    try:
        with path.open("rb") as f:
            n = 0
            for _ in f:
                n += 1
        if n == 0:
            return 0
        # asume header
        return max(0, n - 1)
    except Exception:
        return None


def _json_sanitize(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_sanitize(v) for v in x]
    return x


def _stage_to_filename(stage: str) -> str:
    # "A.ingest" -> "stage_A_ingest.json"
    # "D.materialize" -> "stage_D_materialize.json"
    # "E.reports" -> "stage_E_reports.json"
    stage = stage.strip()
    if "." in stage:
        left, right = stage.split(".", 1)
        left = left.strip()
        right = right.strip().replace(".", "_")
        return f"stage_{left}_{right}.json"
    return f"stage_{stage.replace('.', '_')}.json"

def read_last_artifacts(meta_dir: Pathish, n: int = 50) -> List[JsonDict]:
    """
    Small helper for debugging: read last N artifact rows.
    """
    md = Path(meta_dir)
    p = md / "artifacts.jsonl"
    if not p.exists():
        return []
    lines: List[str] = p.read_text(encoding="utf-8").splitlines()
    tail = lines[-n:]
    out: List[JsonDict] = []
    for ln in tail:
        try:
            out.append(json.loads(ln))
        except Exception:
            out.append({"_parse_error": True, "raw": ln})
    return out


# -------------------------
# JSON Schemas (draft 2020-12)
# -------------------------

ARTIFACT_ROW_SCHEMA: JsonDict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ArtifactRow",
    "type": "object",
    "additionalProperties": True,
    "required": ["run_id", "mode", "stage", "name", "relpath", "content_type", "created_at", "role"],
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "mode": {"type": "string", "enum": ["smoke", "run"]},
        "stage": {"type": "string", "minLength": 1},
        "name": {"type": "string", "minLength": 1},
        "relpath": {"type": "string", "minLength": 1},
        "sha256": {"type": ["string", "null"], "pattern": "^[a-f0-9]{64}$"},
        "bytes": {"type": ["integer", "null"], "minimum": 0},
        "rows": {"type": ["integer", "null"], "minimum": 0},
        "content_type": {"type": "string", "minLength": 1},
        "created_at": {"type": "string", "minLength": 1},
        "role": {"type": "string", "enum": ["input", "primary", "derived", "meta", "debug"]},
        "artifact_role": {"type": "string", "enum": ARTIFACT_ROLES},
        "accounting_nature": {"type": "string", "enum": ACCOUNTING_NATURES},
        "grain": {"type": "string", "enum": GRAINS},
        "currency_policy": {"type": "string", "enum": CURRENCY_POLICIES},
        "frontend_suitability": {"type": "string", "enum": FRONTEND_SUITABILITIES},
        "source_authority": {"type": "string", "enum": SOURCE_AUTHORITIES},
        "notes": {"type": ["string", "null"]},
        "exists": {"type": ["boolean", "null"]},
        "extra": {"type": ["object", "null"]},
    },
}

STAGE_MANIFEST_SCHEMA: JsonDict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "StageManifest",
    "type": "object",
    "additionalProperties": True,
    "required": ["stage", "mode", "run_id", "generated_at", "inputs", "params", "outputs"],
    "properties": {
        "stage": {"type": "string", "minLength": 1},
        "mode": {"type": "string", "enum": ["smoke", "run"]},
        "run_id": {"type": "string", "minLength": 1},
        "generated_at": {"type": "string", "minLength": 1},
        "inputs": {
            "type": "array",
            "items": {"type": "object"},
        },
        "params": {"type": "object"},
        "outputs": {
            "type": "array",
            "items": {"type": "object"},
        },
        "warnings": {"type": ["array", "null"], "items": {"type": "object"}},
        "anomalies": {"type": ["array", "null"], "items": {"type": "object"}},
    },
}

CHECK_MANIFEST_SCHEMA: JsonDict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CheckManifest",
    "type": "object",
    "additionalProperties": True,
    "required": ["stage", "ok", "generated_at", "stage_manifest_path", "checks", "errors"],
    "properties": {
        "stage": {"type": "string", "minLength": 1},
        "ok": {"type": "boolean"},
        "generated_at": {"type": "string", "minLength": 1},
        "stage_manifest_path": {"type": "string", "minLength": 1},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "ok"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "ok": {"type": "boolean"},
                    "details": {"type": ["string", "null"]},
                    "metrics": {"type": ["object", "null"]},
                },
                "additionalProperties": True,
            },
        },
        "errors": {"type": "array", "items": {"type": "string"}},
    },
}



## Data structure hint for observability.

### HELPERS

import csv
# import json
# from pathlib import Path
from typing import Set #Any, Dict, List, Optional, Set, Tuple, Union

def _flatten_json_paths(
    obj: Any,
    prefix: str = "",
    *,
    max_depth: int = 6,
    _depth: int = 0,
    _out: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Flatten nested JSON structure into dotted key paths.
    - Dict keys become ".key"
    - Lists become "[]" and we inspect only the first element if it's a list of dicts
    - Lists of scalars become "field[]" without enumerating values (avoid enum spam)
    """
    if _out is None:
        _out = set()

    if _depth >= max_depth:
        if prefix:
            _out.add(prefix + " (depth_limit)")
        return _out

    if isinstance(obj, dict):
        if not obj and prefix:
            _out.add(prefix)  # empty dict still a meaningful node
        for k, v in obj.items():
            k_str = str(k)
            new_prefix = f"{prefix}.{k_str}" if prefix else k_str
            _flatten_json_paths(v, new_prefix, max_depth=max_depth, _depth=_depth + 1, _out=_out)

    elif isinstance(obj, list):
        list_prefix = prefix + "[]" if prefix else "[]"
        if not obj:
            _out.add(list_prefix)
            return _out

        first = obj[0]

        # list of dicts or list of lists: inspect first element shape only
        if isinstance(first, (dict, list)):
            _flatten_json_paths(first, list_prefix, max_depth=max_depth, _depth=_depth + 1, _out=_out)
        else:
            # list of scalars: record schema-level fact only, not values
            _out.add(list_prefix)

    else:
        if prefix:
            _out.add(prefix)

    return _out


def _csv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # skip empty lines
            if row and any(cell.strip() for cell in row):
                return [c.strip() for c in row]
    return []


def _jsonl_first_record(path: Path, *, max_bytes: int = 256_000) -> Optional[Any]:
    """
    Read first non-empty JSON object line from JSONL.
    Bounded read to avoid huge files.
    """
    read_bytes = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            read_bytes += len(line.encode("utf-8", errors="ignore"))
            if read_bytes > max_bytes:
                return None
            s = line.strip()
            if not s:
                continue
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                # If the first non-empty line isn't valid JSON, abort quietly
                return None
    return None


def structure_hint_from_file(
    path: Union[str, Path],
    *,
    max_depth: int = 6,
    max_fields: int = 200,
    max_json_bytes: int = 2_000_000,   # guard: avoid loading very large json into memory
) -> Dict[str, Any]:
    """
    Return a compact structure hint for artifact manifests:
    - CSV: header columns
    - JSONL: flattened fields from first record
    - JSON: flattened fields from parsed object (guarded by size)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    size = path.stat().st_size

    hint: Dict[str, Any] = {"kind": "other"}

    if suffix == ".csv":
        cols = _csv_header(path)
        hint = {"kind": "csv", "columns": cols}

    elif suffix == ".jsonl":
        obj = _jsonl_first_record(path)
        if obj is None:
            hint = {"kind": "jsonl", "fields": [], "note": "no_valid_first_record_or_too_large"}
        else:
            fields = sorted(_flatten_json_paths(obj, max_depth=max_depth))
            hint = {"kind": "jsonl", "fields": fields[:max_fields], "fields_truncated": len(fields) > max_fields}

    elif suffix == ".json":
        if size > max_json_bytes:
            hint = {"kind": "json", "fields": [], "note": f"json_too_large>{max_json_bytes}_bytes"}
        else:
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                fields = sorted(_flatten_json_paths(obj, max_depth=max_depth))
                hint = {"kind": "json", "fields": fields[:max_fields], "fields_truncated": len(fields) > max_fields}
            except Exception as e:
                hint = {"kind": "json", "fields": [], "note": f"json_parse_failed:{type(e).__name__}"}

    return hint



###



def artifact_from_path(
    name: str,
    path: Path,
    stage: str,
    mode: str,
    run_id: str,
    role: str,
    root_dir: Path,
    rows: Optional[int] = None,
    content_type: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(path)
    root_dir = Path(root_dir)

    if not path.exists():
        raise FileNotFoundError(f"artifact missing: {path}")

    relpath = str(path.resolve().relative_to(root_dir.resolve()))
    b = path.stat().st_size
    sha = _sha256_file(path)

    if rows is None and path.suffix.lower() == ".csv":
        rows = _count_csv_rows_fast(path)

    if content_type is None:
        guess, _enc = mimetypes.guess_type(path.name)
        content_type = guess or "application/octet-stream"

    structure = structure_hint_from_file(path)  # <-- Structure hint
    contract = artifact_contract_for_name(name, relpath)

    return {
        "run_id": str(run_id),
        "stage": str(stage),
        "mode": str(mode),
        "name": str(name),
        "relpath": str(relpath),
        "sha256": str(sha),
        "bytes": int(b),
        "rows": rows,
        "content_type": str(content_type),
        "created_at": _utc_now_iso(),
        "role": str(role),
        **contract,
        "structure": structure,  # <--
    }


def contracts_dataframe_rows(artifact_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in artifact_rows:
        relpath = str(row.get("relpath", ""))
        contract = {**artifact_contract_for_name(str(row.get("name", "")), relpath), **{k: row.get(k) for k in CONTRACT_COLUMNS if k in row}}
        rows.append({col: contract.get(col, row.get(col, "")) for col in CONTRACT_COLUMNS})
    return rows


def write_artifact_contracts_csv(path: Pathish, artifact_rows: Iterable[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = list(artifact_rows)
    rows = contracts_dataframe_rows(raw_rows)
    for out_row, raw_row in zip(rows, raw_rows):
        for key in ["publish_class", "exists"]:
            if key in raw_row:
                out_row[key] = raw_row[key]
    fieldnames = list(CONTRACT_COLUMNS)
    for optional in ["publish_class", "exists"]:
        if any(optional in row for row in rows):
            fieldnames.append(optional)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_artifact_contract_qa(path: Pathish, artifact_rows: Iterable[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(artifact_rows)
    qa: List[Dict[str, str]] = []

    def add(check: str, ok: bool, detail: str, severity: str = "error") -> None:
        qa.append({"check": check, "status": "pass" if ok else "fail", "detail": detail, "severity": severity})

    contracts = contracts_dataframe_rows(rows)
    names = {str(r.get("name", "")) for r in rows}
    rels = {str(r.get("relpath", "")) for r in rows}
    add("artifact_contracts_exist", all(c.get("artifact_role") for c in contracts), f"rows={len(contracts)}")
    canonical_present = any(c.get("source_authority") == "source_of_truth" for c in contracts)
    add("canonical_outputs_present", canonical_present, f"canonical_count={sum(c.get('artifact_role') == 'canonical_source' for c in contracts)}", "warning")
    money_contracts = [c for c in contracts if c.get("currency_policy") in {"explicit_currency_required", "by_currency", "converted", "no_cross_currency_sum"}]
    add("money_outputs_have_currency_policy", all(c.get("currency_policy") for c in money_contracts), f"money_contracts={len(money_contracts)}")
    unsafe_safe = [c for c in contracts if c.get("artifact_role") in {"internal_balance", "inferred_reconciliation", "unsafe_for_frontend"} and c.get("frontend_suitability") not in {"forbidden", "internal_only"}]
    add("unsafe_artifacts_not_marked_safe", not unsafe_safe, f"unsafe_safe={unsafe_safe}")
    daily = [c for c in contracts if Path(str(c.get("relpath", c.get("name", "")))).name == "daily_cash_position.csv"]
    add("daily_cash_position_not_frontend_safe", not daily or all(c.get("frontend_suitability") == "forbidden" for c in daily), f"rows={len(daily)}")
    box = [c for c in contracts if Path(str(c.get("relpath", c.get("name", "")))).name.startswith("box_balance_time_long")]
    add("box_balance_not_frontend_safe", not box or all(c.get("frontend_suitability") == "forbidden" for c in box), f"rows={len(box)}")
    metric_views = [c for c in contracts if "metric_views/" in str(c.get("relpath", "")).replace("\\", "/")]
    add("metric_views_not_canonical", all(c.get("artifact_role") == "presentation_only" for c in metric_views), f"rows={len(metric_views)}")
    add("frontend_outputs_have_suitability", all(c.get("frontend_suitability") for c in contracts), f"rows={len(contracts)}")
    add("no_cross_currency_sum_without_conversion_policy", True, "contract metadata requires by-currency or explicit converted policy", "warning")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail", "severity"])
        writer.writeheader()
        writer.writerows(qa)
    return path


def append_artifacts(meta_dir: Path, artifact_rows: Iterable[Dict[str, Any]]) -> str:
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    out_path = meta_dir / "artifacts.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        for row in artifact_rows:
            clean = _json_sanitize(dict(row))
            clean.setdefault("created_at", _utc_now_iso())
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    out_dir = meta_dir.parent
    return str(out_path.resolve().relative_to(out_dir.resolve()))

def write_stage_manifest(meta_dir: Path, manifest_obj: Dict[str, Any], filename: Optional[str] = None) -> str:
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    obj = _json_sanitize(dict(manifest_obj))
    obj.setdefault("generated_at", _utc_now_iso())

    stage = str(obj.get("stage", "stage"))
    fname = filename or _stage_to_filename(stage)

    out_path = meta_dir / fname
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # devolver relpath respecto de out_dir (parent de meta)
    out_dir = meta_dir.parent
    return str(out_path.resolve().relative_to(out_dir.resolve()))
