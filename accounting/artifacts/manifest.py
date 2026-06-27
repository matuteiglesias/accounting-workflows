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
        "role": {"type": "string", "enum": ["primary", "derived", "meta", "debug"]},
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
        "structure": structure,  # <--
    }


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

