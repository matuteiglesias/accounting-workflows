from __future__ import annotations
"""YAML -> isolated ledger clones -> existing offline accounting -> HTML/PDF only.

YAML is a reviewable business declaration, never Python to execute.
The pilot deliberately leaves the paid-only canonical history unchanged:
future `planeado` items occur only in the all-status evidence stream.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

import yaml

from accounting.artifacts.manifest import artifact_from_path, append_artifacts, write_stage_manifest
from accounting.support.isolated_execution import isolated_accounting_execution

SCHEMA = "accounting.scenario.v1"
ID = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
RUN = re.compile(r"^\d{8}T\d{6}Z_FBPM$")
MONTH = re.compile(r"^20\d{2}-(?:0[1-9]|1[0-2])$")
FIELDS = frozenset({"Box", "payer", "receiver", "Flujo", "Tipo", "Lugar", "Detalle", "tag", "status"})
STAGES = ("run-materialize", "run-debt", "run-metrics", "run-reports")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_definition(path: Path) -> dict:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise ValueError("invalid scenario YAML schema")
    if not ID.fullmatch(str(obj.get("id", ""))):
        raise ValueError("invalid scenario id")
    base = obj.get("base") or {}
    if not RUN.fullmatch(str(base.get("run_id", ""))):
        raise ValueError("scenario base must be one exact FBPM run, not latest")
    if not re.fullmatch(r"[a-f0-9]{64}", str(base.get("all_status_sha256", ""))):
        raise ValueError("YAML must pin the base all-status ledger SHA256")
    if obj.get("reports") != ["treasury_accountability"]:
        raise ValueError("pilot supports the existing treasury_accountability report")
    variants = obj.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise ValueError("pilot requires two variants")
    labels = [v.get("id") for v in variants if isinstance(v, dict)]
    if len(labels) != 2 or len(set(labels)) != 2 or any(not ID.fullmatch(str(x)) for x in labels):
        raise ValueError("variant ids must be unique and safe")
    if not any(not v.get("operations") for v in variants):
        raise ValueError("at least one unchanged baseline variant is required")
    for variant in variants:
        for op in variant.get("operations", []):
            if op.get("op") != "suppress_planned":
                raise ValueError("unsupported operator (no code execution allowed)")
            equals = op.get("equals")
            if not isinstance(equals, dict) or not equals or not set(equals) <= FIELDS:
                raise ValueError("invalid field selector")
            if equals.get("status") != "planeado":
                raise ValueError("suppress_planned may only select status=planeado")
            tags = op.get("tags")
            if not isinstance(tags, list) or not tags or len(tags) != len(set(tags)):
                raise ValueError("must select unique explicit tags")
            a, b = str(op.get("from_month", "")), str(op.get("through_month", ""))
            if not MONTH.fullmatch(a) or not MONTH.fullmatch(b) or b < a:
                raise ValueError("invalid monthly interval")
            expected = op.get("expect")
            if not isinstance(expected, dict) or type(expected.get("rows")) is not int or expected["rows"] < 1:
                raise ValueError("positive matched-row assertion required")
            if not isinstance(expected.get("amount_by_currency"), dict) or not expected["amount_by_currency"]:
                raise ValueError("currency-specific amount assertions required")
    return obj


def transform_all_status(source: Path, output: Path, operations: list[dict]) -> list[dict]:
    if not operations:
        shutil.copy2(source, output)
        return []
    with source.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        cols = reader.fieldnames
        if not cols or not {"Date", "status", "tag", "tx_id", "Currency", "amount"} <= set(cols):
            raise ValueError("canonical ledger schema mismatch")
        rows = list(reader)
    audit = []
    for rule in operations:
        matched, keep = [], []
        for row in rows:
            selected = (
                all(row.get(k, "") == str(v) for k, v in rule["equals"].items())
                and row["tag"] in rule["tags"]
                and rule["from_month"] <= row["Date"][:7] <= rule["through_month"]
            )
            (matched if selected else keep).append(row)
        expected = rule["expect"]
        if len(matched) != expected["rows"] or len({x["tx_id"] for x in matched}) != len(matched):
            raise ValueError(f"unsafe scenario selector: matched {len(matched)}, expected {expected['rows']}")
        totals = {}
        for record in matched:
            if record["status"] != "planeado":
                raise ValueError("scenario selected realized history")
            cur = record["Currency"]
            totals[cur] = totals.get(cur, 0.0) + float(record["amount"])
        if set(totals) != set(expected["amount_by_currency"]) or any(
            abs(totals[k] - float(expected["amount_by_currency"][k])) > 0.005 for k in totals
        ):
            raise ValueError(f"unexpected currency amounts: {totals}")
        audit.append({"operator": "suppress_planned", "rows": len(matched), "totals": totals})
        rows = keep
    with output.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return audit


def seed_execution(base: Path, out: Path, run_id: str, scenario: str, variant: dict) -> list[dict]:
    out.mkdir(parents=True, exist_ok=False)
    meta = out / "meta"
    meta.mkdir()
    paid_source = base / "ledger_canonical.csv"
    all_source = base / "ledger_canonical_all_status.csv"
    shutil.copy2(paid_source, out / paid_source.name)
    removed = transform_all_status(all_source, out / all_source.name, variant.get("operations", []))
    if digest(out / paid_source.name) != digest(paid_source):
        raise ValueError("paid-only historical ledger was modified")
    original = json.loads((base / "meta" / "stage_A_ingest.json").read_text(encoding="utf-8"))
    if original.get("run_id") != base.name:
        raise ValueError("base Stage A manifest identity mismatch")
    original["run_id"] = run_id
    original["generated_at"] = datetime.now(timezone.utc).isoformat()
    original["mode"] = "run"
    original["inputs"] = [{
        "source_run_id": base.name, "canonical_sha256": digest(paid_source),
        "all_status_sha256": digest(all_source), "source_type": "exact_run_snapshot",
    }]
    original["params"].update({
        "simulation": True, "scenario_id": scenario,
        "scenario_variant": variant["id"], "base_run_id": base.name,
    })
    artifacts = [
        artifact_from_path(name=name.removesuffix(".csv"), path=out / name, stage="A.ingest",
                           mode="run", run_id=run_id, role="output", root_dir=out)
        for name in ("ledger_canonical_all_status.csv", "ledger_canonical.csv")
    ]
    original["outputs"] = artifacts
    stage_file = out / write_stage_manifest(meta, original)
    artifacts.append(artifact_from_path(
        name="stage_A_ingest", path=stage_file, stage="A.ingest",
        mode="run", run_id=run_id, role="meta", root_dir=out,
    ))
    append_artifacts(meta, artifacts)
    return removed


def run_stages(execution) -> None:
    log = execution.workspace / "stages.log"
    for stage in STAGES:
        print(f"  {stage}", flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== {stage} ===\n")
            handle.flush()
            result = subprocess.run(
                execution.make_command(stage), cwd=execution.repo_root,
                stdout=handle, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{stage} failed (exit {result.returncode}):\n" +
                               log.read_text(encoding="utf-8")[-6000:])


def execute(scenario_file: Path, repo: Path, outbox: Path) -> Path:
    spec = read_definition(scenario_file)
    repo = Path(repo).resolve(strict=True)
    base = repo / "out" / "run" / "accounting" / spec["base"]["run_id"]
    if not base.is_dir() or base.is_symlink():
        raise ValueError("base run must be an existing non-symlinked exact-run directory")
    if digest(base / "ledger_canonical_all_status.csv") != spec["base"]["all_status_sha256"]:
        raise ValueError("base ledger checksum differs from human-approved YAML")
    outbox = Path(outbox).resolve()
    if any(outbox.is_relative_to(path.resolve()) for path in (repo / "out", repo / "public", repo / "scenarios")):
        raise ValueError("scenario outbox must be outside production and code")
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final = outbox / f"{spec['id']}__{stamp}"
    hashes = {}
    with tempfile.TemporaryDirectory(prefix=".scenario-export-", dir=outbox) as folder:
        deliverables = Path(folder) / "deliverables"
        deliverables.mkdir()
        for variant in spec["variants"]:
            print(f"Scenario {spec['id']}: {variant['id']}", flush=True)
            with isolated_accounting_execution(repo) as execution:
                removed = seed_execution(base, execution.canonical_root, execution.run_id,
                                         spec["id"], variant)
                print("  matched transformations:", removed, flush=True)
                hashes[variant["id"]] = digest(execution.canonical_root / "ledger_canonical_all_status.csv")
                run_stages(execution)
                report = execution.reports_root / "treasury_accountability"
                manifest = json.loads((report / "report_manifest.json").read_text(encoding="utf-8"))
                if manifest["source_run_id"] != execution.run_id or manifest["validation_status"] == "fail":
                    raise ValueError("scenario report failed identity or validation")
                for extension in ("html", "pdf"):
                    source = report / f"report.{extension}"
                    if extension == "pdf":
                        with source.open("rb") as handle:
                            if handle.read(5) != b"%PDF-":
                                raise ValueError("invalid PDF")
                    execution.export(source, deliverables / f"SIMULACION__{spec['id']}__{variant['id']}.{extension}")
        if len(set(hashes.values())) != 2:
            raise ValueError("scenario and baseline yielded identical input evidence")
        exports = list(deliverables.iterdir())
        if len(exports) != 4 or {x.suffix for x in exports} != {".html", ".pdf"}:
            raise ValueError("the outbox accepts exactly four HTML/PDF artifacts")
        if final.exists():
            raise FileExistsError(final)
        os.replace(deliverables, final)
    print(f"Four reports exported to {final}", flush=True)
    return final


def main() -> None:
    p = argparse.ArgumentParser(description="Run a declarative scenario over an exact accounting run.")
    p.add_argument("--scenario", required=True, type=Path)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--outbox", required=True, type=Path)
    a = p.parse_args()
    execute(a.scenario, a.repo_root, a.outbox)


if __name__ == "__main__":
    main()
