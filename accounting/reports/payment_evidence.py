from __future__ import annotations

"""Reproducible private report for an approved transaction-evidence snapshot."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any

import pandas as pd

from accounting.reports.catalog import (
    ReportCatalogItem,
    build_report_catalog,
    validate_catalog_files,
    write_report_catalog,
)
from accounting.reports.common import sha256_file
from accounting.reports.pdf import render_pdf


SCHEMA = "acct.transaction-evidence@1"
REPORT_ID = "pm_payment_evidence"
REPORT_TITLE = "Registro de comprobantes — Property Management"
REPORT_DESCRIPTION = "Detalle de operaciones de Property Management y estado de su respaldo documental."
PAYMENT_RELATIONS = {"payment_proof", "transfer_proof"}
ALLOWED_RELATIONS = PAYMENT_RELATIONS | {
    "statement_context", "liability_source", "other_support"
}
ALLOWED_STATUSES = {"approved", "candidate", "rejected"}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _relative_inside(root: Path, value: str) -> Path:
    rel = PurePosixPath(str(value).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe evidence href: {value!r}")
    candidate = (root / Path(*rel.parts)).resolve(strict=True)
    root = root.resolve(strict=True)
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"evidence href escapes snapshot: {value!r}")
    return candidate


def _required_hash(tx_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(tx_ids)).encode()).hexdigest()


def _read_snapshot(snapshot_dir: Path, run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    snapshot_dir = snapshot_dir.resolve(strict=True)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact") != SCHEMA:
        raise ValueError(f"unsupported evidence snapshot: {manifest.get('artifact')!r}")
    ledger_path = run_root.resolve(strict=True) / "ledger_canonical.csv"
    if manifest.get("source_ledger_sha256") != _sha256(ledger_path):
        raise ValueError("evidence snapshot is not bound to the requested canonical ledger")
    required = pd.read_csv(snapshot_dir / "required_transactions.csv", dtype=str, keep_default_na=False)
    docs = pd.read_csv(snapshot_dir / "evidence_documents.csv", dtype=str, keep_default_na=False)
    relations = pd.read_csv(snapshot_dir / "transaction_evidence.csv", dtype=str, keep_default_na=False)
    required_ids = required["tx_id"].astype(str).tolist()
    if len(required_ids) != len(set(required_ids)):
        raise ValueError("required transaction population contains duplicate tx_id values")
    if len(required_ids) != int(manifest["required_transaction_count"]):
        raise ValueError("required transaction count differs from snapshot manifest")
    if _required_hash(required_ids) != manifest.get("required_tx_ids_sha256"):
        raise ValueError("required transaction identity hash differs from snapshot manifest")
    if len(required) != 183:
        raise ValueError(f"PM evidence report requires the frozen 183 rows; got {len(required)}")
    expected_doc_columns = {"evidence_id", "content_sha256", "media_type", "display_name", "href"}
    expected_relation_columns = {"tx_id", "evidence_id", "relation", "status"}
    if not expected_doc_columns.issubset(docs.columns):
        raise ValueError(f"evidence_documents missing columns: {sorted(expected_doc_columns - set(docs.columns))}")
    if not expected_relation_columns.issubset(relations.columns):
        raise ValueError(f"transaction_evidence missing columns: {sorted(expected_relation_columns - set(relations.columns))}")
    if docs["evidence_id"].duplicated().any():
        raise ValueError("evidence_documents contains duplicate evidence_id")
    if relations.duplicated(["tx_id", "evidence_id", "relation", "status"]).any():
        raise ValueError("transaction_evidence contains duplicate relation rows")
    if not set(relations["tx_id"]).issubset(set(required_ids)):
        raise ValueError("transaction_evidence references a transaction outside the frozen population")
    if not set(relations["evidence_id"]).issubset(set(docs["evidence_id"])):
        raise ValueError("transaction_evidence references an unknown evidence_id")
    if not set(relations["relation"]).issubset(ALLOWED_RELATIONS):
        raise ValueError("transaction_evidence contains an unsupported relation")
    if not set(relations["status"]).issubset(ALLOWED_STATUSES):
        raise ValueError("transaction_evidence contains an unsupported status")
    for _, doc in docs.iterrows():
        evidence_id = str(doc["evidence_id"])
        sha = str(doc["content_sha256"])
        if not SHA256_RE.fullmatch(evidence_id) or evidence_id.lower() != sha.lower():
            raise ValueError(f"evidence identity/hash mismatch: {evidence_id}")
        source = _relative_inside(snapshot_dir, str(doc["href"]))
        if _sha256(source) != sha.lower():
            raise ValueError(f"snapshot evidence hash mismatch: {evidence_id}")
    if set(relations["status"]) - {"approved", "candidate"}:
        raise ValueError("rejected relations are not supported in the report source")
    return required, docs, relations, manifest


def _concept(row: pd.Series) -> str:
    values = []
    for column in ("receiver", "Detalle", "notes", "issuer", "Lugar"):
        value = str(row.get(column, "")).strip()
        if value and value not in values:
            values.append(value)
    return " · ".join(values)


def _format_links(tx_id: str, tx_links: list[dict[str, str]], docs: dict[str, dict[str, str]]) -> str:
    links = []
    for relation in tx_links:
        if relation["status"] != "approved" or relation["relation"] not in PAYMENT_RELATIONS:
            continue
        doc = docs[relation["evidence_id"]]
        links.append(
            f'<a class="evidence-link" href="evidence/{html.escape(relation["evidence_id"], quote=True)}.pdf">'
            f'Ver comprobante<span class="sr-only"> {html.escape(doc["display_name"])}</span></a>'
        )
    return "<br>".join(links) if links else "—"


def _status_for(tx_links: list[dict[str, str]]) -> tuple[str, str]:
    approved = [r for r in tx_links if r["status"] == "approved" and r["relation"] in PAYMENT_RELATIONS]
    candidate = [r for r in tx_links if r["status"] == "candidate"]
    if approved:
        return "approved", "Comprobante aprobado"
    if candidate:
        return "candidate", "Candidato pendiente de revisión"
    return "missing", "Sin comprobante aprobado"


def _html(required: pd.DataFrame, docs: pd.DataFrame, relations: pd.DataFrame, *, source_run_id: str, cutoff: str, snapshot_id: str) -> str:
    doc_map = {str(row["evidence_id"]): row.to_dict() for _, row in docs.iterrows()}
    links_by_tx: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, row in relations.iterrows():
        links_by_tx[str(row["tx_id"])].append({key: str(row[key]) for key in ("tx_id", "evidence_id", "relation", "status")})
    rows = []
    for _, row in required.iterrows():
        tx_id = str(row["tx_id"])
        status, label = _status_for(links_by_tx.get(tx_id, []))
        year = str(row["Date"])[:4]
        rows.append(
            f'<tr data-status="{status}" data-year="{html.escape(year)}" data-payer="{html.escape(str(row.get("payer", "")))}" data-tipo="{html.escape(str(row.get("Tipo", "")))}">'
            f'<td class="tx-id">{html.escape(tx_id)}</td>'
            f'<td>{html.escape(str(row.get("Date", "")))}</td>'
            f'<td>{html.escape(str(row.get("payer", row.get("Payer", ""))))}</td>'
            f'<td>{html.escape(str(row.get("Tipo", "")))}</td>'
            f'<td>{html.escape(_concept(row))}</td>'
            f'<td class="amount">{html.escape(str(row.get("amount", "")))}</td>'
            f'<td>{html.escape(str(row.get("Currency", "")))}</td>'
            f'<td><span class="status status-{status}">{html.escape(label)}</span></td>'
            f'<td>{_format_links(tx_id, links_by_tx.get(tx_id, []), doc_map)}</td></tr>'
        )
    approved_count = sum(_status_for(links_by_tx.get(str(tx), []))[0] == "approved" for tx in required["tx_id"])
    candidate_count = sum(_status_for(links_by_tx.get(str(tx), []))[0] == "candidate" for tx in required["tx_id"])
    missing_count = len(required) - approved_count - candidate_count
    approved_relations = relations.query("status == 'approved' and relation in @PAYMENT_RELATIONS")
    attachment_count = approved_relations["evidence_id"].nunique()
    return f'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(REPORT_TITLE)}</title>
<style>
:root{{--ink:#1d2733;--muted:#64748b;--line:#dbe2ea;--green:#147a4b;--green-bg:#e8f6ee;--amber:#956d00;--amber-bg:#fff7d6;--gray:#667085;--gray-bg:#f1f3f5}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:#f8fafc}}main{{max-width:1500px;margin:0 auto;padding:28px}}h1{{margin:0 0 8px;font-size:28px}}h2{{font-size:18px;margin:24px 0 10px}}.meta,.note{{color:var(--muted)}}.panel{{background:white;border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:180px}}.card strong{{display:block;font-size:24px}}.filters{{display:flex;gap:10px;flex-wrap:wrap;align-items:end}}label{{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:12px}}input,select{{font:inherit;padding:8px;border:1px solid #bcc7d3;border-radius:6px;background:white;color:var(--ink)}}table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#eef2f6;z-index:1}}.tx-id{{font-family:ui-monospace,SFMono-Regular,monospace;white-space:nowrap}}.amount{{text-align:right;white-space:nowrap}}.status{{display:inline-block;border-radius:999px;padding:3px 8px;white-space:nowrap;font-weight:650;font-size:12px}}.status-approved{{color:var(--green);background:var(--green-bg)}}.status-candidate{{color:var(--amber);background:var(--amber-bg)}}.status-missing{{color:var(--gray);background:var(--gray-bg)}}.evidence-link{{color:#0b5cad;font-weight:650;white-space:nowrap}}.sr-only{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}}.hidden{{display:none!important}}@media print{{.filters{{display:none}}body{{background:white}}main{{padding:8px}}table{{font-size:9px}}th{{position:static}}.evidence-link{{color:#000}}}}
</style></head><body><main>
<header><h1>{html.escape(REPORT_TITLE)}</h1><p class="meta">Corrida contable: <strong>{html.escape(source_run_id)}</strong> · Corte contable: <strong>{html.escape(cutoff)}</strong> · Snapshot de evidencia: <strong>{html.escape(snapshot_id)}</strong></p></header>
<section class="panel"><h2>Cobertura documental</h2><div class="cards"><div class="card"><strong>{approved_count}</strong>Comprobantes aprobados</div><div class="card"><strong>{candidate_count}</strong>Candidatos pendientes</div><div class="card"><strong>{missing_count}</strong>Sin comprobante aprobado</div><div class="card"><strong>{attachment_count}</strong>PDFs únicos adjuntos</div></div><p class="note">Los estados reflejan únicamente relaciones de tipo <code>payment_proof</code> o <code>transfer_proof</code>. “Sin comprobante aprobado” no implica que la operación esté impaga.</p></section>
<section class="panel"><div class="filters"><label>Buscar<input id="search" type="search" placeholder="tx_id, concepto…"></label><label>Estado<select id="status"><option value="">Todos</option><option value="approved">Aprobado</option><option value="candidate">Candidato</option><option value="missing">Sin comprobante aprobado</option></select></label><label>Año<select id="year"><option value="">Todos</option></select></label><label>Payer<select id="payer"><option value="">Todos</option></select></label><label>Tipo<select id="tipo"><option value="">Todos</option></select></label></div></section>
<section class="panel"><table id="transactions"><thead><tr><th>tx_id</th><th>Fecha</th><th>Payer</th><th>Tipo</th><th>Concepto / beneficiario</th><th>Importe</th><th>Moneda</th><th>Estado documental</th><th>Evidencia</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<script>(function(){{const table=document.querySelector('#transactions'), rows=[...table.tBodies[0].rows]; const fields={{status:document.querySelector('#status'),year:document.querySelector('#year'),payer:document.querySelector('#payer'),tipo:document.querySelector('#tipo'),search:document.querySelector('#search')}}; function fill(key){{const vals=[...new Set(rows.map(r=>r.dataset[key]).filter(Boolean))].sort(); for(const v of vals){{const o=document.createElement('option');o.value=v;o.textContent=v;fields[key].append(o)}}}} ['year','payer','tipo'].forEach(fill); function apply(){{const q=fields.search.value.toLowerCase();rows.forEach(r=>{{const hay=r.textContent.toLowerCase();const ok=(!fields.status.value||r.dataset.status===fields.status.value)&&(!fields.year.value||r.dataset.year===fields.year.value)&&(!fields.payer.value||r.dataset.payer===fields.payer.value)&&(!fields.tipo.value||r.dataset.tipo===fields.tipo.value)&&(!q||hay.includes(q));r.classList.toggle('hidden',!ok)}})}} Object.values(fields).forEach(e=>e.addEventListener('input',apply));}})();</script>
</main></body></html>'''


def build_payment_evidence_report(*, run_root: Path, snapshot_dir: Path, out_root: Path, browser_bin: str | Path | None = None) -> dict[str, Path | int | str]:
    run_root = Path(run_root).resolve(strict=True)
    snapshot_dir = Path(snapshot_dir).resolve(strict=True)
    required, docs, relations, snapshot_manifest = _read_snapshot(snapshot_dir, run_root)
    source_run_id = run_root.name
    snapshot_id = str(snapshot_manifest.get("snapshot_id") or snapshot_dir.name)
    source_ledger = run_root / "ledger_canonical.csv"
    cutoff = str(pd.to_datetime(required["Date"], errors="coerce").max().date())
    approved = relations[(relations["status"] == "approved") & relations["relation"].isin(PAYMENT_RELATIONS)]
    approved_evidence = sorted(set(approved["evidence_id"]))
    report_root = Path(out_root).resolve() / REPORT_ID
    if report_root.exists():
        shutil.rmtree(report_root)
    evidence_root = report_root / "evidence"
    evidence_root.mkdir(parents=True)
    docs_by_id = {str(row["evidence_id"]): row for _, row in docs.iterrows()}
    attachment_rows = []
    for evidence_id in approved_evidence:
        doc = docs_by_id[evidence_id]
        source = _relative_inside(snapshot_dir, str(doc["href"]))
        target = evidence_root / f"{evidence_id}.pdf"
        shutil.copy2(source, target)
        if _sha256(target) != evidence_id.lower():
            raise ValueError(f"copied attachment hash mismatch: {evidence_id}")
        attachment_rows.append({"path": f"{REPORT_ID}/evidence/{evidence_id}.pdf", "sha256": evidence_id.lower()})
    report_html = report_root / "report.html"
    report_html.write_text(_html(required, docs, relations, source_run_id=source_run_id, cutoff=cutoff, snapshot_id=snapshot_id), encoding="utf-8")
    report_pdf = render_pdf(report_html, report_root / "report.pdf", browser_bin=browser_bin)
    generated_at = str(snapshot_manifest.get("generated_at_utc") or snapshot_manifest.get("run_id") or source_run_id)
    report_manifest = {
        "schema": "accounting_report_manifest.v1",
        "report_id": REPORT_ID,
        "renderer_version": "pm_payment_evidence.v1",
        "source_run_id": source_run_id,
        "scope_tag": "PM_EVIDENCE",
        "as_of_date": cutoff,
        "sources": [
            {"logical_path": "run/ledger_canonical.csv", "path": str(source_ledger), "sha256": _sha256(source_ledger), "rows": int(len(pd.read_csv(source_ledger)))},
            {"logical_path": "evidence/manifest.json", "path": str(snapshot_dir / "manifest.json"), "sha256": _sha256(snapshot_dir / "manifest.json"), "rows": None},
            {"logical_path": "evidence/required_transactions.csv", "path": str(snapshot_dir / "required_transactions.csv"), "sha256": _sha256(snapshot_dir / "required_transactions.csv"), "rows": int(len(required))},
            {"logical_path": "evidence/evidence_documents.csv", "path": str(snapshot_dir / "evidence_documents.csv"), "sha256": _sha256(snapshot_dir / "evidence_documents.csv"), "rows": int(len(docs))},
            {"logical_path": "evidence/transaction_evidence.csv", "path": str(snapshot_dir / "transaction_evidence.csv"), "sha256": _sha256(snapshot_dir / "transaction_evidence.csv"), "rows": int(len(relations))},
        ],
        "outputs": {"html": {"path": f"{REPORT_ID}/report.html", "sha256": _sha256(report_html)}, "pdf": {"path": f"{REPORT_ID}/report.pdf", "sha256": _sha256(report_pdf)}},
        "attachments": attachment_rows,
        "coverage": {"required_transactions": int(len(required)), "approved_relationships": int(len(approved)), "approved_payment_coverage_tx": int(approved["tx_id"].nunique()), "candidate_relationships": int(((relations["status"] == "candidate")).sum()), "without_approved_payment_proof": int(len(required) - approved["tx_id"].nunique())},
        "accounting_authority_changed": False,
        "matching_rerun": False,
    }
    (report_root / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    catalog = build_report_catalog(source_run_id=source_run_id, scope_tag="PM_EVIDENCE", as_of_date=cutoff, generated_at_utc=generated_at, reports=[ReportCatalogItem(report_id=REPORT_ID, title=REPORT_TITLE, description=REPORT_DESCRIPTION, period_label=f"Corte {cutoff}", sort_order=10, html=f"{REPORT_ID}/report.html", pdf=f"{REPORT_ID}/report.pdf", manifest=None, attachments=tuple(attachment_rows))])
    catalog_path = Path(out_root).resolve() / "report_catalog.json"
    write_report_catalog(catalog_path, catalog)
    validate_catalog_files(catalog, bundle_root=Path(out_root).resolve())
    return {"catalog": catalog_path, "report_html": report_html, "report_pdf": report_pdf, "report_manifest": report_root / "report_manifest.json", "approved_attachments": len(approved_evidence), "required_rows": len(required), "approved_relations": len(approved), "candidate_relations": int((relations["status"] == "candidate").sum()), "approved_payment_coverage_tx": int(approved["tx_id"].nunique())}
