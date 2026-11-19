from __future__ import annotations

"""Reproducible private report for an approved transaction-evidence snapshot."""

from collections import Counter, defaultdict
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


def _attachment_extension(doc: Any) -> str:
    """Preserve original evidence media types in the private report bundle."""
    media_type = str(doc["media_type"])
    allowed = {"application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpeg"}
    if media_type not in allowed:
        raise ValueError(f"unsupported evidence media_type: {media_type!r}")
    return allowed[media_type]


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
    # Population size is a ledger-run property, not a renderer constant.
    # Frozen universe identity and cardinality are checked against manifest above.
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
        _attachment_extension(doc)
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


def _format_links(tx_id: str, tx_links: list[dict[str, str]], docs: dict[str, dict[str, str]], *, section: str = "payment") -> str:
    """Priority: approved receipt > receipt to review > bill supporting the obligation."""
    links = []
    shown: set[tuple[str, str]] = set()
    def priority(relation: dict[str, str]) -> tuple[int, str]:
        if relation["relation"] in PAYMENT_RELATIONS:
            return (0 if relation["status"] == "approved" else 1, relation["evidence_id"])
        if relation["relation"] == "liability_source":
            return (2, relation["evidence_id"])
        return (3, relation["evidence_id"])

    if section not in {"payment", "statement"}:
        raise ValueError(f"unknown evidence link section: {section}")
    for relation in sorted(tx_links, key=priority):
        kind = relation["relation"]
        if section == "payment" and kind in {"liability_source", "statement_context"}:
            continue
        if section == "statement" and kind not in {"liability_source", "statement_context"}:
            continue
        if kind in PAYMENT_RELATIONS:
            label = "Ver comprobante" if relation["status"] == "approved" else "Revisar comprobante (no aprobado)"
            css = "evidence-link" if relation["status"] == "approved" else "evidence-link evidence-pending"
        elif kind == "liability_source":
            label = "Ver boleta (no acredita pago)"
            css = "evidence-link evidence-bill"
        elif kind == "statement_context":
            label = "Ver estado de cuenta (no es recibo)"
            css = "evidence-link evidence-statement"
        elif kind == "other_support":
            label = "Recibo de la contrapartida"
            css = "evidence-link evidence-counterpart"
        else:
            continue
        eid = relation["evidence_id"]
        marker = (kind, eid)
        if marker in shown:
            continue
        shown.add(marker)
        doc = docs[eid]
        links.append(
            f'<a class="{css}" href="evidence/{html.escape(eid, quote=True)}.{_attachment_extension(doc)}" '
            f'title="{html.escape(doc["display_name"], quote=True)}">'
            f'{html.escape(label)}<span class="sr-only"> {html.escape(doc["display_name"])}</span></a>'
        )
    return "<br>".join(links) if links else "—"


def _status_for(tx_links: list[dict[str, str]]) -> tuple[str, str]:
    approved = [r for r in tx_links if r["status"] == "approved" and r["relation"] in PAYMENT_RELATIONS]
    candidate = [r for r in tx_links if r["status"] == "candidate" and r["relation"] in PAYMENT_RELATIONS]
    liability = [r for r in tx_links if r["relation"] == "liability_source"]
    statement = [r for r in tx_links if r["relation"] == "statement_context"]
    if approved:
        return "approved", "Comprobante aprobado"
    if candidate:
        return "candidate", "Comprobante por revisar"
    if liability:
        return "bill_only", "Solo boleta; falta prueba de pago"
    if statement:
        return "statement_only", "Solo estado de cuenta; pago no validado"
    return "missing", "Sin candidato documental"


def _html(required: pd.DataFrame, docs: pd.DataFrame, relations: pd.DataFrame, *, source_run_id: str, cutoff: str, snapshot_id: str, review_preview: bool = False) -> str:
    doc_map = {str(row["evidence_id"]): row.to_dict() for _, row in docs.iterrows()}
    links_by_tx: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, row in relations.iterrows():
        links_by_tx[str(row["tx_id"])].append({key: str(row[key]) for key in ("tx_id", "evidence_id", "relation", "status")})
    groups = {str(r["tx_id"]): str(r.get("evidence_review_group", "")) for _, r in required.iterrows()}

    def review_status(tx_id: str) -> tuple[str, str]:
        status, label = _status_for(links_by_tx.get(tx_id, []))
        if review_preview and groups.get(tx_id) == "COUNTERPART_OF_ALREADY_APPROVED_PAYMENT":
            if status == "approved":
                raise ValueError("counterpart cannot be a separately approved payment")
            if status == "candidate":
                raise ValueError("counterpart still incorrectly classified as a payment candidate")
            return "counterpart", "Comprobante en otra operación; revisar imputación"
        return status, label

    required_display = required.copy()
    if review_preview and "evidence_review_group" in required_display.columns:
        priorities = {
            "NEW_PAYMENT_DATE_DIFFERENCE": 0,
            "SINGLE_RECEIPT_REVIEW": 1,
            "PAYMENT_ACCOUNT_AMBIGUITY": 2,
            "TWO_LEDGER_LEGS_ONE_UNAPPROVED_PAYMENT": 3,
            "ARBA_ACCOUNT_MISMATCH_REJECT_RECOMMENDED": 4,
            "WRONG_ISSUER_ACCOUNT_REJECT_RECOMMENDED": 5,
            "COUNTERPART_OF_ALREADY_APPROVED_PAYMENT": 6,
            "NEW_VERIFIED_PAYMENT_APPROVED": 7,
            "HISTORICAL_APPROVAL_RESTORED": 8,
        }
        required_display["_review_priority"] = required_display["evidence_review_group"].map(priorities).fillna(9)
        required_display = required_display.sort_values(by=["_review_priority", "Date", "tx_id"],
                                                        ascending=[True, False, True], kind="mergesort")
    rows = []
    for _, row in required_display.iterrows():
        tx_id = str(row["tx_id"])
        status, label = review_status(tx_id)
        year = str(row["Date"])[:4]
        review_note = str(row.get("evidence_review_note", "")).strip()
        state_markup = f'<span class="status status-{status}">{html.escape(label)}</span>'
        if review_note:
            state_markup += f'<div class="review-note">{html.escape(review_note)}</div>'
        rows.append(
            f'<tr data-status="{status}" data-year="{html.escape(year)}" data-payer="{html.escape(str(row.get("payer", "")))}" data-tipo="{html.escape(str(row.get("Tipo", "")))}">'
            f'<td>{html.escape(str(row.get("transaction_id", "")))}</td>'
            f'<td class="tx-id">{html.escape(tx_id)}</td>'
            f'<td>{html.escape(str(row.get("Date", "")))}</td>'
            f'<td>{html.escape(str(row.get("payer", row.get("Payer", ""))))}</td>'
            f'<td>{html.escape(str(row.get("Tipo", "")))}</td>'
            f'<td>{html.escape(_concept(row))}</td>'
            f'<td class="amount">{html.escape(str(row.get("amount", "")))}</td>'
            f'<td>{html.escape(str(row.get("Currency", "")))}</td>'
            f'<td>{state_markup}</td>'
            f'<td>{_format_links(tx_id, links_by_tx.get(tx_id, []), doc_map, section="payment")}</td>'
            f'<td>{_format_links(tx_id, links_by_tx.get(tx_id, []), doc_map, section="statement")}</td></tr>'
        )
    approved_count = sum(review_status(str(tx))[0] == "approved" for tx in required["tx_id"])
    candidate_count = sum(review_status(str(tx))[0] == "candidate" for tx in required["tx_id"])
    counterpart_count = sum(review_status(str(tx))[0] == "counterpart" for tx in required["tx_id"])
    bill_only_count = sum(review_status(str(tx))[0] == "bill_only" for tx in required["tx_id"])
    statement_only_count = sum(review_status(str(tx))[0] == "statement_only" for tx in required["tx_id"])
    missing_count = len(required) - approved_count - candidate_count - counterpart_count - bill_only_count - statement_only_count
    attachment_count = relations["evidence_id"].nunique()
    preview_banner = ('<div class="preview-banner"><strong>REPORTE PRIVADO EN REVISIÓN — NO PUBLICAR.</strong> '
                      'Los comprobantes aprobados se identificaron y registraron con auditoría; las contrapartidas no representan pagos adicionales. '
                      'Las decisiones contables y el corte oficial de publicación permanecen independientes.</div>') if review_preview else ""
    return f'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(REPORT_TITLE)}</title>
<style>
:root{{--ink:#1d2733;--muted:#64748b;--line:#dbe2ea;--green:#147a4b;--green-bg:#e8f6ee;--amber:#956d00;--amber-bg:#fff7d6;--gray:#667085;--gray-bg:#f1f3f5}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:#f8fafc}}main{{max-width:1500px;margin:0 auto;padding:28px}}h1{{margin:0 0 8px;font-size:28px}}h2{{font-size:18px;margin:24px 0 10px}}.meta,.note{{color:var(--muted)}}.panel{{background:white;border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:180px}}.card strong{{display:block;font-size:24px}}.filters{{display:flex;gap:10px;flex-wrap:wrap;align-items:end}}label{{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:12px}}input,select{{font:inherit;padding:8px;border:1px solid #bcc7d3;border-radius:6px;background:white;color:var(--ink)}}table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#eef2f6;z-index:1}}.tx-id{{font-family:ui-monospace,SFMono-Regular,monospace;white-space:nowrap}}.amount{{text-align:right;white-space:nowrap}}.status{{display:inline-block;border-radius:999px;padding:3px 8px;white-space:nowrap;font-weight:650;font-size:12px}}.status-approved{{color:var(--green);background:var(--green-bg)}}.status-candidate{{color:var(--amber);background:var(--amber-bg)}}.status-missing{{color:var(--gray);background:var(--gray-bg)}}.status-counterpart{{color:#385d78;background:#e3eef9}}.status-bill_only{{color:#315e7e;background:#e9f2fa}}.status-statement_only{{color:#315e7e;background:#e9f2fa}}.evidence-link{{color:#0b5cad;font-weight:650;white-space:nowrap}}.evidence-pending{{color:#996400}}.evidence-bill{{color:#4f6273}}.evidence-counterpart{{color:#385d78}}.review-note{{font-size:11px;color:#64748b;max-width:310px;margin-top:4px;line-height:1.25}}.preview-banner{{padding:12px 16px;background:#ffedd5;border:2px solid #ed9b39;border-radius:8px;font-size:14px;margin:12px 0}}.sr-only{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}}.hidden{{display:none!important}}@media print{{@page{{size:landscape;margin:7mm}}.filters{{display:none}}body{{background:white}}main{{padding:4px;max-width:none}}table{{width:100%;table-layout:fixed;font-size:7px}}th,td{{padding:3px;overflow-wrap:anywhere;word-break:break-word}}.tx-id{{white-space:normal;font-size:6px}}#transactions tr>:nth-child(2){{display:none}}.review-note{{display:none}}.status{{white-space:normal;font-size:7px;line-height:1.2;padding:2px 3px;max-width:130px}}th{{position:static}}.evidence-link{{color:#000;white-space:normal}}}}
</style></head><body><main>
{preview_banner}<header><h1>{html.escape(REPORT_TITLE)}</h1><p class="meta">Corrida contable: <strong>{html.escape(source_run_id)}</strong> · Corte contable: <strong>{html.escape(cutoff)}</strong> · Snapshot de evidencia: <strong>{html.escape(snapshot_id)}</strong></p></header>
<section class="panel"><h2>Cobertura documental</h2><div class="cards"><div class="card"><strong>{approved_count}</strong>Comprobantes aprobados</div><div class="card"><strong>{candidate_count}</strong>Recibos pendientes</div><div class="card"><strong>{counterpart_count}</strong>Contrapartidas (no pagos adicionales)</div><div class="card"><strong>{bill_only_count}</strong>Solo boleta</div><div class="card"><strong>{statement_only_count}</strong>Solo estado de cuenta</div><div class="card"><strong>{missing_count}</strong>Sin candidato</div><div class="card"><strong>{attachment_count}</strong>Documentos adjuntos</div></div><p class="note">Una boleta acredita una obligación, no el desembolso; una reliquidación no crea otra deuda. Los recibos candidatos no se confunden con recibos aprobados. “Sin candidato” no significa impago. Un pago puede cubrir varias cuotas y una cuota varios pagos.</p></section>
<section class="panel"><div class="filters"><label>Buscar<input id="search" type="search" placeholder="tx_id, concepto…"></label><label>Estado<select id="status"><option value="">Todos</option><option value="approved">Aprobado</option><option value="candidate">Comprobante por revisar</option><option value="counterpart">Contrapartida</option><option value="bill_only">Solo boleta</option><option value="statement_only">Solo estado de cuenta</option><option value="missing">Sin candidato</option></select></label><label>Año<select id="year"><option value="">Todos</option></select></label><label>Payer<select id="payer"><option value="">Todos</option></select></label><label>Tipo<select id="tipo"><option value="">Todos</option></select></label></div></section>
<section class="panel"><table id="transactions"><thead><tr><th>ID operación</th><th>tx_id</th><th>Fecha</th><th>Payer</th><th>Tipo</th><th>Concepto / beneficiario</th><th>Importe</th><th>Moneda</th><th>Estado documental</th><th>Comprobante / respaldo</th><th>Statement / boleta (opcional)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
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
    report_evidence = sorted(set(relations["evidence_id"]))
    # Exclusive summary states must use the same precedence as the visible HTML.
    # A counterpart can retain a liability bill, but cannot also count as bill-only.
    manifest_relation_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, relation in relations.iterrows():
        manifest_relation_map[str(relation["tx_id"])].append({
            "status": str(relation["status"]),
            "relation": str(relation["relation"]),
        })
    counterpart_ids = set(
        required.loc[required.get("evidence_review_group", pd.Series("", index=required.index))
                     == "COUNTERPART_OF_ALREADY_APPROVED_PAYMENT", "tx_id"].astype(str)
    ) if snapshot_manifest.get("review_preview", False) else set()
    states = Counter(
        "counterpart" if str(tid) in counterpart_ids
        else _status_for(manifest_relation_map.get(str(tid), []))[0]
        for tid in required["tx_id"]
    )
    report_root = Path(out_root).resolve() / REPORT_ID
    if report_root.exists():
        shutil.rmtree(report_root)
    evidence_root = report_root / "evidence"
    evidence_root.mkdir(parents=True)
    docs_by_id = {str(row["evidence_id"]): row for _, row in docs.iterrows()}
    attachment_rows = []
    for evidence_id in report_evidence:
        doc = docs_by_id[evidence_id]
        source = _relative_inside(snapshot_dir, str(doc["href"]))
        target = evidence_root / f"{evidence_id}.{_attachment_extension(doc)}"
        shutil.copy2(source, target)
        if _sha256(target) != evidence_id.lower():
            raise ValueError(f"copied attachment hash mismatch: {evidence_id}")
        attachment_rows.append({"path": f"{REPORT_ID}/evidence/{evidence_id}.{_attachment_extension(doc)}", "sha256": evidence_id.lower()})
    report_html = report_root / "report.html"
    report_html.write_text(_html(required, docs, relations, source_run_id=source_run_id, cutoff=cutoff, snapshot_id=snapshot_id, review_preview=bool(snapshot_manifest.get("review_preview", False))), encoding="utf-8")
    report_pdf = render_pdf(report_html, report_root / "report.pdf", browser_bin=browser_bin)
    generated_at = str(snapshot_manifest.get("generated_at_utc") or snapshot_manifest.get("run_id") or source_run_id)
    report_manifest = {
        "schema": "accounting_report_manifest.v1",
        "report_id": REPORT_ID,
        "renderer_version": "pm_payment_evidence.v2_precedence",
        "review_preview": bool(snapshot_manifest.get("review_preview", False)),
        "publication_allowed": not bool(snapshot_manifest.get("review_preview", False)),
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
        "coverage": {
            "required_transactions": int(len(required)),
            "approved_relationships": int(len(approved)),
            "approved_payment_coverage_tx": int(approved["tx_id"].nunique()),
            "candidate_relationships": int((relations["status"] == "candidate").sum()),
            "candidate_payment_coverage_tx": int(states["candidate"]),
            "counterpart_coverage_tx": int(states["counterpart"]),
            "bill_only_coverage_tx": int(states["bill_only"]),
            "statement_only_coverage_tx": int(states["statement_only"]),
            "without_approved_payment_proof": int(len(required) - approved["tx_id"].nunique()),
        },
        "accounting_authority_changed": bool(snapshot_manifest.get("source_corrections")),
        "source_corrections": list(snapshot_manifest.get("source_corrections", [])),
        "matching_rerun": False,
        "statement_column_optional": True,
        "statement_context_relations": int((relations["relation"] == "statement_context").sum()),
    }
    (report_root / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    catalog = build_report_catalog(source_run_id=source_run_id, scope_tag="PM_EVIDENCE", as_of_date=cutoff, generated_at_utc=generated_at, reports=[ReportCatalogItem(report_id=REPORT_ID, title=REPORT_TITLE, description=REPORT_DESCRIPTION, period_label=f"Corte {cutoff}", sort_order=10, html=f"{REPORT_ID}/report.html", pdf=f"{REPORT_ID}/report.pdf", manifest=None, attachments=tuple(attachment_rows))])
    catalog_path = Path(out_root).resolve() / "report_catalog.json"
    write_report_catalog(catalog_path, catalog)
    validate_catalog_files(catalog, bundle_root=Path(out_root).resolve())
    return {"catalog": catalog_path, "report_html": report_html, "report_pdf": report_pdf, "report_manifest": report_root / "report_manifest.json", "approved_attachments": len(approved_evidence), "included_attachments": len(report_evidence), "required_rows": len(required), "approved_relations": len(approved), "candidate_relations": int((relations["status"] == "candidate").sum()), "approved_payment_coverage_tx": int(approved["tx_id"].nunique())}
