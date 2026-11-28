from __future__ import annotations
"""YAML -> isolated ledger clones -> existing offline accounting -> HTML/PDF only.

YAML is a reviewable business declaration, never Python to execute.
The production paid-only canonical history remains unchanged. Isolated
simulations derive an analytic ledger from paid rows plus eligible future plans,
retaining each row's status; the same accounting stages and report renderer run.
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
from datetime import date, datetime, timezone
from time import perf_counter

import yaml

from accounting.artifacts.manifest import artifact_from_path, append_artifacts, write_stage_manifest
from accounting.support.isolated_execution import isolated_accounting_execution
from accounting.scenarios.projection import build_projection_ledger

SCHEMA = "accounting.scenario.v1"
ID = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
RUN = re.compile(r"^\d{8}T\d{6}Z_FBPM$")
MONTH = re.compile(r"^20\d{2}-(?:0[1-9]|1[0-2])$")
FIELDS = frozenset({"Box", "payer", "receiver", "Flujo", "Tipo", "Lugar", "Detalle", "tag", "status"})
STAGES = ("run-materialize", "run-debt", "run-metrics")


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
    projection = obj.get("projection")
    if projection is not None:
        if not isinstance(projection, dict) or set(projection) != {"as_of_date", "through_date"}:
            raise ValueError("projection needs exact as_of_date and through_date")
        start = date.fromisoformat(projection["as_of_date"])
        end = date.fromisoformat(projection["through_date"])
        if start > end or end.year > start.year + 3:
            raise ValueError("invalid projection horizon")
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


def seed_execution(base: Path, out: Path, run_id: str, scenario: str, variant: dict, *, projection: dict | None = None, projection_source: Path | None = None) -> list[dict]:
    out.mkdir(parents=True, exist_ok=False)
    meta = out / "meta"
    meta.mkdir()
    paid_source = base / "ledger_canonical.csv"
    all_source = base / "ledger_canonical_all_status.csv"
    if projection_source is not None and (not projection or variant.get("operations")):
        raise ValueError("separate planning snapshot is supported only for unchanged projected baseline")
    removed = transform_all_status(all_source, out / all_source.name, variant.get("operations", []))
    if projection:
        summary = build_projection_ledger(
            paid_source, projection_source if projection_source is not None else out / all_source.name, out / paid_source.name,
            as_of_date=projection["as_of_date"], through_date=projection["through_date"],
        )
        print("  analytic simulation ledger:", summary, flush=True)
    else:
        shutil.copy2(paid_source, out / paid_source.name)
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
    if projection:
        original["params"].update({
            "only_status": ["pagado", "planeado"],
            "projection_as_of_date": projection["as_of_date"],
            "projection_through_date": projection["through_date"],
            "projection_semantics": "paid_observed_plus_future_planned_not_realized",
            **({"planning_source_sha256": digest(projection_source)} if projection_source is not None else {}),
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


def run_stages(execution, *, measurements: list[dict] | None = None) -> None:
    log = execution.workspace / "stages.log"
    python_timings = execution.workspace / "python_timings.jsonl"
    for stage in STAGES:
        print(f"  {stage}", flush=True)
        command = execution.make_command(stage)
        env = None
        if measurements is not None:
            # No dependency or instrumentation in ordinary execution. Make's
            # existing $(PY) expansion becomes a transparent timed subprocess.
            command.append("PY=python3 -m accounting.diagnostics.timed_python")
            env = dict(os.environ, ACCOUNTING_TIMINGS_JSONL=str(python_timings))
        start = perf_counter()
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== {stage} ===\n")
            handle.flush()
            result = subprocess.run(command, cwd=execution.repo_root,
                                    stdout=handle, stderr=subprocess.STDOUT, env=env)
        if measurements is not None:
            measurements.append({"kind": "make_stage", "stage": stage,
                                 "elapsed_seconds": round(perf_counter() - start, 6),
                                 "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{stage} failed (exit {result.returncode}):\n" +
                               log.read_text(encoding="utf-8")[-6000:])
    if measurements is not None and python_timings.is_file():
        measurements.extend(json.loads(line) for line in python_timings.read_text(encoding="utf-8").splitlines())


def render_requested_treasury(execution, *, measurements: list[dict] | None = None, projection: dict | None = None) -> Path:
    """Use the existing official renderer, but don't build unrelated PDFs."""
    from accounting.reports.pdf import render_pdf
    from accounting.reports.treasury_accountability.render import render_report

    run = execution.canonical_root
    metric = execution.metrics_root
    report = execution.reports_root / "treasury_accountability"
    sources = {
        "accountability_path": run / "monthly_cash_accountability.csv",
        "qa_path": run / "monthly_cash_accountability_qa.csv",
        "cycles_path": run / "family_business_accountability_cycles.csv",
        "stakeholder_support_path": run / "monthly_stakeholder_support.csv",
        "semantic_audit_path": run / "classification_audit.csv",
        "annual_metrics_path": metric / "annual_balance_dashboard_metrics.csv",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required treasury source missing: {missing}")
    print("  render_treasury (existing renderer only)", flush=True)
    start = perf_counter()
    if projection:
        # These optional sections have observed-evidence contracts; the single
        # monthly treasury renderer remains the authority for projected rows.
        for key in ("cycles_path", "stakeholder_support_path", "semantic_audit_path", "annual_metrics_path"):
            sources.pop(key)
    output = render_report(
        **sources, out_dir=report,
        **({"projection_as_of_date": projection["as_of_date"],
            "projection_through_date": projection["through_date"]} if projection else {}),
    )
    if measurements is not None:
        measurements.append({"kind": "renderer", "stage": "treasury_html",
                             "elapsed_seconds": round(perf_counter() - start, 6)})
    if not output["html"].is_file() or not output["validation"].is_file():
        raise ValueError("existing renderer failed to produce report and validation")
    print("  render_pdf (existing PDF engine)", flush=True)
    start = perf_counter()
    render_pdf(output["html"], report / "report.pdf")
    if measurements is not None:
        measurements.append({"kind": "renderer", "stage": "treasury_pdf",
                             "elapsed_seconds": round(perf_counter() - start, 6)})
    return report


def execute(scenario_file: Path, repo: Path, outbox: Path, *, audit_file: Path | None = None) -> Path:
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
    # Validate the optional diagnostic destination *before* a costly execution.
    audit_path = Path(audit_file).resolve() if audit_file is not None else None
    if audit_path is not None:
        if (audit_path.is_relative_to(outbox) or
                audit_path.is_relative_to((repo / "out").resolve()) or
                audit_path.is_relative_to((repo / "public").resolve()) or
                audit_path.is_relative_to((repo / "scenarios").resolve())):
            raise ValueError("performance audit must be outside outbox, production and YAML definitions")
        if audit_path.exists():
            raise FileExistsError(f"performance audit already exists: {audit_path}")
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final = outbox / f"{spec['id']}__{stamp}"
    hashes = {}
    audit_events: list[dict] | None = [] if audit_file is not None else None
    audit_start = perf_counter()
    with tempfile.TemporaryDirectory(prefix=".scenario-export-", dir=outbox) as folder:
        deliverables = Path(folder) / "deliverables"
        deliverables.mkdir()
        for variant in spec["variants"]:
            print(f"Scenario {spec['id']}: {variant['id']}", flush=True)
            # Snap Chromium uses a private /tmp namespace, so PDFs rendered in
            # host /tmp can appear missing even when Chromium reports exit 0.
            # The temporary workspace must live in the user's non-hidden home
            # tree; the context still deletes it when each variant finishes.
            with isolated_accounting_execution(repo, temp_parent=outbox.parent) as execution:
                variant_events: list[dict] | None = [] if audit_events is not None else None
                variant_start = perf_counter()
                start = perf_counter()
                removed = seed_execution(base, execution.canonical_root, execution.run_id,
                                         spec["id"], variant, projection=spec.get("projection"))
                if variant_events is not None:
                    variant_events.append({"kind": "scenario", "stage": "seed_and_transform",
                                           "elapsed_seconds": round(perf_counter() - start, 6)})
                print("  matched transformations:", removed, flush=True)
                hashes[variant["id"]] = digest(execution.canonical_root / "ledger_canonical_all_status.csv")
                run_stages(execution, measurements=variant_events)
                # Do not invoke run-reports here: it produces many unrelated PDFs,
                # and failure of an optional specialized renderer would block the
                # requested treasury document. The official treasury renderer and
                # official PDF engine are reused unchanged.
                report = render_requested_treasury(execution, measurements=variant_events, projection=spec.get("projection"))
                for extension in ("html", "pdf"):
                    source = report / f"report.{extension}"
                    if extension == "pdf":
                        with source.open("rb") as handle:
                            if handle.read(5) != b"%PDF-":
                                raise ValueError("invalid PDF")
                    execution.export(source, deliverables / f"SIMULACION__{spec['id']}__{variant['id']}.{extension}")
                if audit_events is not None and variant_events is not None:
                    audit_events.append({"variant": variant["id"], "run_id": execution.run_id,
                                         "total_seconds": round(perf_counter() - variant_start, 6),
                                         "events": variant_events})
        if len(set(hashes.values())) != 2:
            raise ValueError("scenario and baseline yielded identical input evidence")
        exports = list(deliverables.iterdir())
        if len(exports) != 4 or {x.suffix for x in exports} != {".html", ".pdf"}:
            raise ValueError("the outbox accepts exactly four HTML/PDF artifacts")
        if final.exists():
            raise FileExistsError(final)
        os.replace(deliverables, final)
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({
            "schema": "accounting.scenario.perf.v1", "scenario_id": spec["id"],
            "base_run_id": spec["base"]["run_id"],
            "total_seconds": round(perf_counter() - audit_start, 6),
            "variants": audit_events,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Four reports exported to {final}", flush=True)
    return final


def main() -> None:
    p = argparse.ArgumentParser(description="Run a declarative scenario over an exact accounting run.")
    p.add_argument("--scenario", required=True, type=Path)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--outbox", required=True, type=Path)
    p.add_argument("--audit-file", type=Path, default=None,
                   help="Optional JSON timings outside outbox; off by default")
    a = p.parse_args()
    execute(a.scenario, a.repo_root, a.outbox, audit_file=a.audit_file)


if __name__ == "__main__":
    main()
