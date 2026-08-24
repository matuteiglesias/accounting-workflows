from __future__ import annotations

import argparse
import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

AUDIT_COLUMNS = [
    "source_layer", "source_artifact", "period", "date", "Currency", "Box", "cash_path",
    "semantic_bucket", "semantic_subbucket", "rule_id", "rule_ids", "actor", "payer", "receiver",
    "counterparty", "funding_actor", "funding_channel", "source_box", "target_box", "beneficiary_box",
    "obligation_box", "payment_channel", "is_rent", "is_cash_box_inflow", "is_direct_obligation_payment",
    "is_internal_support", "is_family_business_support", "is_property_management_support",
    "is_household_support", "creates_debt", "settles_debt", "linked_debt_id", "amount_in", "amount_out",
    "net_amount", "amount_abs", "tx_id", "source_tx_ids_sample", "description", "raw_text_hint",
    "classification_confidence", "classification_problem", "recommended_semantic_bucket",
    "recommended_semantic_subbucket", "recommended_extra_dimensions", "recommended_fix",
]
SUMMARY_COLUMNS = [
    "summary_type", "source_layer", "semantic_bucket", "semantic_subbucket", "funding_actor",
    "funding_channel", "target_box", "obligation_box", "cash_effect", "classification_problem",
    "rows", "amount_in", "amount_out", "net_amount", "amount_abs", "recommended_fix",
]
TEXT_COLUMNS = [
    "description", "raw_text_hint", "Detalle", "notes", "Flujo", "Tipo", "rule_id", "rule_ids",
    "actor", "payer", "receiver", "counterparty", "Box", "cash_path", "source_tx_ids_sample",
]
KEYWORD_RE = re.compile(
    r"fund|funding|aporte|aportes|contrib|contribution|support|soporte|inquil|\bInq\b|"
    r"impuesto|tax|servicio|service|Mat[ií]as|Matias|Alejandro|Primos|H[eé]ctor|Hector|"
    r"Household|Family Business|Property Management|deuda|debt",
    re.IGNORECASE,
)
ACTOR_PATTERNS = [
    ("Matías", re.compile(r"mat[ií]as|matias|\bmi\b", re.I)),
    ("Alejandro", re.compile(r"alejandro|\balen\b|\bale\b", re.I)),
    ("Primos", re.compile(r"primos", re.I)),
    ("Héctor", re.compile(r"h[eé]ctor|hector", re.I)),
    ("Inquilino", re.compile(r"inquil|\binq\b", re.I)),
    ("Household", re.compile(r"household|\bhh\b", re.I)),
]
BOXES = {"property management", "family business", "household"}


def _now_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _num(value: Any) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return 0.0
    return float(parsed)


def _read_csv(path: Path | None) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _find_source(repo_root: Path, pack: Path, run_root: Path, name: str) -> Path | None:
    candidates = [
        pack / name,
        pack / "tables" / name,
        pack / "drilldown" / name,
        pack / "digest" / name,
        run_root / name,
        repo_root / "out" / "professional_pack" / "latest_FBPM" / name,
        repo_root / "out" / "professional_pack" / "latest_FBPM" / "tables" / name,
        repo_root / "out" / "professional_pack" / "latest_FBPM" / "drilldown" / name,
        repo_root / "out" / "run" / "accounting" / "latest_FBPM" / name,
        repo_root / "public" / "accounting" / "latest_FBPM" / "canonical_dashboard" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _blob(row: pd.Series) -> str:
    return " | ".join(_str(row.get(c)) for c in TEXT_COLUMNS if c in row.index)


def _contains_keyword(row: pd.Series) -> bool:
    return bool(KEYWORD_RE.search(_blob(row)))


def _derive_actor(row: pd.Series) -> str:
    existing = _str(row.get("funding_actor"))
    if existing:
        return existing
    blob = _blob(row)
    for actor, pattern in ACTOR_PATTERNS:
        if pattern.search(blob):
            return actor
    return ""


def _derive_channel(row: pd.Series) -> str:
    existing = _str(row.get("funding_channel"))
    if existing:
        return existing
    blob = _blob(row).casefold()
    bucket = _str(row.get("semantic_bucket"))
    sub = _str(row.get("semantic_subbucket")).casefold()
    if "debt" in blob or "deuda" in blob or bucket == "debt_movement":
        if "repago" in blob or "repayment" in blob or "settle" in blob or "sald" in blob:
            return "debt_settlement"
        return "debt_creation"
    if "impuesto" in blob or "tax" in blob or sub == "taxes":
        if re.search(r"inquil|\binq\b", blob):
            return "tenant_direct_tax_payment"
        return "named_actor_support"
    if "servicio" in blob or "service" in blob or sub == "services":
        if re.search(r"inquil|\binq\b", blob):
            return "tenant_direct_service_payment"
        return "named_actor_support"
    if "household" in blob or re.search(r"\bhh\b", blob):
        return "household_to_pm" if "property management" in blob or re.search(r"\bpm\b", blob) else "other"
    if "family business" in blob or re.search(r"\bfb\b", blob) or re.search(r"alejandro|primos", blob):
        return "family_business_contribution"
    if re.search(r"inquil|\binq\b", blob):
        return "tenant_to_box"
    if bucket == "funding_contribution" or "fund" in blob or "aporte" in blob or "contrib" in blob:
        return "cash_to_box" if _num(row.get("amount_in")) > 0 else "other"
    return ""


def _is_rent(row: pd.Series) -> bool:
    return _str(row.get("semantic_bucket")) == "operating_revenue" and _str(row.get("semantic_subbucket")) == "rent"


def _target_box(row: pd.Series) -> str:
    for col in ["target_box", "beneficiary_box", "Box"]:
        value = _str(row.get(col))
        if value:
            return value
    return ""


def _obligation_box(row: pd.Series, channel: str) -> str:
    existing = _str(row.get("obligation_box"))
    if existing:
        return existing
    if channel in {"tenant_direct_tax_payment", "tenant_direct_service_payment", "household_to_pm", "named_actor_support"}:
        target = _target_box(row)
        return target if target else "Property Management"
    return ""


def _cash_effect(row: pd.Series, channel: str) -> str:
    existing = _str(row.get("cash_effect"))
    if existing:
        return existing
    if channel in {"tenant_direct_tax_payment", "tenant_direct_service_payment"}:
        return "no_cash_in_box_direct_payment"
    if _num(row.get("amount_in")) > 0:
        return "cash_in_box"
    if _num(row.get("amount_out")) > 0:
        return "cash_out_box"
    if channel:
        return "non_cash_support"
    return ""


def _classification_problem(row: pd.Series, channel: str, actor: str, cash_effect: str) -> str:
    bucket = _str(row.get("semantic_bucket"))
    sub = _str(row.get("semantic_subbucket"))
    problems: list[str] = []
    if bucket == "funding_contribution" and sub in {"", "family_or_tenant_contribution"}:
        problems.append("collapsed_funding_subbucket")
    if bucket == "funding_contribution" and not actor:
        problems.append("missing_funding_actor")
    if bucket == "funding_contribution" and not channel:
        problems.append("missing_funding_channel")
    if cash_effect == "no_cash_in_box_direct_payment" and _num(row.get("amount_in")) > 0:
        problems.append("direct_payment_modeled_as_amount_in")
    if channel in {"debt_creation", "debt_settlement"} and not _str(row.get("linked_debt_id")):
        problems.append("debt_link_missing")
    if bucket == "property_opex" and channel in {"tenant_direct_tax_payment", "tenant_direct_service_payment", "named_actor_support", "household_to_pm"}:
        problems.append("third_party_obligation_payment_not_explicit")
    return ";".join(problems) if problems else "ok_candidate"


def _candidate_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    bucket = df.get("semantic_bucket", pd.Series("", index=df.index)).fillna("").astype(str)
    sub = df.get("semantic_subbucket", pd.Series("", index=df.index)).fillna("").astype(str)
    rule = (df.get("rule_id", pd.Series("", index=df.index)).fillna("").astype(str) + " " + df.get("rule_ids", pd.Series("", index=df.index)).fillna("").astype(str))
    text = pd.Series([_blob(r) for _, r in df.iterrows()], index=df.index)
    amount_in = pd.to_numeric(df.get("amount_in", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0)
    box = df.get("Box", pd.Series("", index=df.index)).fillna("").astype(str).str.casefold()
    mask = bucket.eq("funding_contribution")
    mask |= sub.str.contains(r"fund|contrib|aporte|support", case=False, na=False, regex=True)
    mask |= rule.str.contains(r"fund|contrib|aporte|support", case=False, na=False, regex=True)
    mask |= text.str.contains(KEYWORD_RE, na=False)
    mask |= box.isin(BOXES) & amount_in.gt(0) & ~bucket.eq("operating_revenue") & ~sub.eq("rent")
    mask |= bucket.isin(["property_opex", "taxes", "services"]) & text.str.contains(r"inquil|\binq\b|mat[ií]as|matias|alejandro|primos|h[eé]ctor|hector|household", case=False, na=False, regex=True)
    return mask.fillna(False)


def _normalize_candidate_rows(df: pd.DataFrame, layer: str, artifact: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    data = df.loc[_candidate_mask(df)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        actor = _derive_actor(row)
        channel = _derive_channel(row)
        target = _target_box(row)
        obligation = _obligation_box(row, channel)
        cash_effect = _cash_effect(row, channel)
        creates_debt = channel == "debt_creation"
        settles_debt = channel == "debt_settlement"
        direct = cash_effect == "no_cash_in_box_direct_payment"
        is_fb = target == "Family Business" or channel == "family_business_contribution"
        is_pm = target == "Property Management" or obligation == "Property Management"
        is_hh = actor == "Household" or target == "Household"
        out = {col: "" for col in AUDIT_COLUMNS}
        out.update({
            "source_layer": layer,
            "source_artifact": artifact,
            "period": _str(row.get("period")) or _str(row.get("year")),
            "date": _str(row.get("Date")) or _str(row.get("date")) or _str(row.get("period_end")),
            "Currency": _str(row.get("Currency")),
            "Box": _str(row.get("Box")),
            "cash_path": _str(row.get("cash_path")),
            "semantic_bucket": _str(row.get("semantic_bucket")),
            "semantic_subbucket": _str(row.get("semantic_subbucket")),
            "rule_id": _str(row.get("rule_id")),
            "rule_ids": _str(row.get("rule_ids")),
            "actor": _str(row.get("actor")),
            "payer": _str(row.get("payer")),
            "receiver": _str(row.get("receiver")),
            "counterparty": _str(row.get("counterparty")),
            "funding_actor": actor,
            "funding_channel": channel,
            "source_box": _str(row.get("source_box")),
            "target_box": target,
            "beneficiary_box": _str(row.get("beneficiary_box")) or target,
            "obligation_box": obligation,
            "payment_channel": _str(row.get("payment_channel")) or _str(row.get("channel")),
            "is_rent": str(_is_rent(row)).lower(),
            "is_cash_box_inflow": str(cash_effect == "cash_in_box").lower(),
            "is_direct_obligation_payment": str(direct).lower(),
            "is_internal_support": str(channel == "household_to_pm").lower(),
            "is_family_business_support": str(is_fb).lower(),
            "is_property_management_support": str(is_pm).lower(),
            "is_household_support": str(is_hh).lower(),
            "creates_debt": str(creates_debt).lower(),
            "settles_debt": str(settles_debt).lower(),
            "linked_debt_id": _str(row.get("linked_debt_id")),
            "amount_in": _num(row.get("amount_in")),
            "amount_out": _num(row.get("amount_out")),
            "net_amount": _num(row.get("net_amount")) or _num(row.get("amount")),
            "amount_abs": _num(row.get("amount_abs")) or abs(_num(row.get("amount"))),
            "tx_id": _str(row.get("tx_id")),
            "source_tx_ids_sample": _str(row.get("source_tx_ids_sample")),
            "description": _str(row.get("description")) or _str(row.get("Detalle")) or _str(row.get("label")) or _str(row.get("metric")),
            "raw_text_hint": _blob(row),
            "classification_confidence": _str(row.get("classification_confidence")),
            "recommended_semantic_bucket": "funding_contribution" if channel and channel not in {"debt_creation", "debt_settlement"} else ("debt_movement" if channel else ""),
            "recommended_semantic_subbucket": channel,
            "recommended_extra_dimensions": "funding_actor;funding_channel;target_box;beneficiary_box;obligation_box;cash_effect;debt_effect;linked_debt_id",
            "recommended_fix": "Add explicit funding/support dimensions upstream; keep renderer label mappings out of business inference.",
        })
        out["classification_problem"] = _classification_problem(pd.Series(out), channel, actor, cash_effect)
        rows.append(out)
    return rows


def _professional_rows(pack: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    table_dir = pack / "tables"
    if not table_dir.exists():
        return rows
    for path in sorted(table_dir.glob("*.csv")):
        if path.name in {"funding_lineage_audit.csv", "funding_lineage_summary.csv"}:
            continue
        df = _read_csv(path)
        if df.empty:
            continue
        text_cols = [c for c in df.columns if df[c].dtype == object or c in {"metric", "line", "label", "section"}]
        if not text_cols:
            continue
        mask = pd.Series(False, index=df.index)
        for col in text_cols:
            mask |= df[col].fillna("").astype(str).str.contains(KEYWORD_RE, na=False)
        for _, row in df.loc[mask].iterrows():
            out = {col: "" for col in AUDIT_COLUMNS}
            desc = " | ".join(_str(row.get(c)) for c in ["section", "metric", "line", "label", "statement_line", "metric_id"] if c in row.index and _str(row.get(c)))
            out.update({
                "source_layer": "professional_table",
                "source_artifact": str(path.relative_to(pack)),
                "period": "",
                "Currency": _str(row.get("Currency")),
                "description": desc,
                "raw_text_hint": " | ".join(_str(row.get(c)) for c in row.index),
                "funding_actor": _derive_actor(row),
                "funding_channel": _derive_channel(row),
                "classification_problem": "professional_label_needs_metric_id_mapping",
                "recommended_extra_dimensions": "metric_id;funding_actor;funding_channel;target_box;obligation_box;cash_effect",
                "recommended_fix": "Wire professional row to annual metric_id plus explicit funding dimensions; do not infer semantics from label text.",
            })
            rows.append(out)
    return rows


def build_audit(repo_root: Path, pack: Path, run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = [
        ("raw_or_canonical", "ledger_canonical.csv"),
        ("classification", "classification_audit.csv"),
        ("semantic_monthly", "monthly_flow_semantic_split.csv"),
        ("operating_statement", "monthly_operating_statement.csv"),
        ("annual_metrics", "annual_balance_dashboard_metrics.csv"),
        ("drilldown", "professional_drilldown_index.csv"),
        ("drilldown", "professional_drilldown_issues.csv"),
        ("debt_activity", "monthly_debt_activity.csv"),
        ("debt_position", "monthly_debt_position.csv"),
    ]
    rows: list[dict[str, Any]] = []
    for layer, name in sources:
        path = _find_source(repo_root, pack, run_root, name)
        df = _read_csv(path)
        artifact = str(path.relative_to(repo_root)) if path and path.exists() else name
        rows.extend(_normalize_candidate_rows(df, layer, artifact))
    rows.extend(_professional_rows(pack))
    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    summary = _summary(audit)
    return audit, summary


def _summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    work = audit.copy()
    for col in ["amount_in", "amount_out", "net_amount", "amount_abs"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    work["cash_effect"] = work.apply(lambda r: _cash_effect(r, _str(r.get("funding_channel"))), axis=1)
    group_cols = ["source_layer", "semantic_bucket", "semantic_subbucket", "funding_actor", "funding_channel", "target_box", "obligation_box", "cash_effect", "classification_problem"]
    grouped = work.groupby(group_cols, dropna=False).agg(
        rows=("source_layer", "size"), amount_in=("amount_in", "sum"), amount_out=("amount_out", "sum"),
        net_amount=("net_amount", "sum"), amount_abs=("amount_abs", "sum"), recommended_fix=("recommended_fix", "first")
    ).reset_index()
    grouped.insert(0, "summary_type", "by_semantic_funding_dimensions")
    return grouped[SUMMARY_COLUMNS].sort_values(["rows", "amount_abs"], ascending=[False, False])


def _render_html(audit: pd.DataFrame, summary: pd.DataFrame) -> str:
    def table(df: pd.DataFrame, n: int = 80) -> str:
        if df.empty:
            return "<p><em>No rows.</em></p>"
        return df.head(n).to_html(index=False, escape=True)
    counts = audit["classification_problem"].value_counts(dropna=False).reset_index() if not audit.empty else pd.DataFrame(columns=["classification_problem", "count"])
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Funding lineage audit</title>
<style>body{{font-family:Arial,sans-serif;margin:28px;color:#111}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #ddd;padding:5px;vertical-align:top}}th{{background:#f4f4f4}}.note{{color:#555}}</style></head>
<body><h1>Funding lineage audit</h1><p class='note'>Diagnostic-only audit for funding/support/direct-payment/debt lineage. It does not change public report values.</p>
<h2>Classification problems</h2>{table(counts)}
<h2>Summary</h2>{table(summary)}
<h2>Audit sample</h2>{table(audit)}
</body></html>"""


def _write_markdown(path: Path, audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    problems = audit["classification_problem"].value_counts(dropna=False).head(30) if not audit.empty else pd.Series(dtype=int)
    lines = [
        "# Professional funding semantics audit",
        "",
        "Diagnostic-only report for contributions / funding / support / direct payments / debt-linked flows.",
        "",
        "## Current semantic conclusions",
        "",
        "1. `funding_contribution` is currently visible, but actor/channel/cash-effect/debt dimensions are incomplete unless supplied upstream.",
        "2. Direct obligation payments must not be treated as simple cash inflows.",
        "3. Professional labels need explicit metric IDs and dimensions before renderer wiring.",
        "",
        "## Classification problems",
        "",
    ]
    if problems.empty:
        lines.append("No candidate rows found in available artifacts.")
    else:
        for key, count in problems.items():
            lines.append(f"- `{key}`: {count}")
    lines.extend([
        "",
        "## Answers to required questions",
        "",
        "1. Current `funding_contribution` rows are rows explicitly classified as funding plus candidate funding/support rows detected by text, actor, non-rent PM/FB inflows, direct obligation payments, or debt hints.",
        "2. Non-rent PM inflows are rows where target/Box is Property Management, amount_in is positive, and the semantic bucket is not operating rent; see `funding_lineage_audit.csv`.",
        "3. Non-rent FB inflows are rows where target/Box/party evidence indicates Family Business and the flow is not rent; see `funding_lineage_audit.csv`.",
        "4. Direct obligation payments are candidates with tax/service wording and tenant/family actor evidence; these are flagged with `is_direct_obligation_payment` when detectable.",
        "5. Rows invisible in annual metrics are those without a stable annual `metric_id` or those represented only as generic OPEX/debt without funding dimensions.",
        "6. Rows visible in semantic split but lost in professional tables are rows whose dimensions collapse to generic `funding_in` or `FUND.CONTRIB.TOTAL`.",
        "7. Dashboard labels needing explicit mappings include Funding / aportes, Matías funding, Inquilinos directo a pagar impuestos, Inquilinos a la caja, Alejandro funding, Primos funding, Héctor funding, Household funding PM, Retiros / gasto personal, Dividendos, and Cobertura después de funding y retiros.",
        "8. Needed subbuckets/channels include cash_to_box, tenant_to_box, tenant_direct_tax_payment, tenant_direct_service_payment, household_to_pm, family_business_contribution, named_actor_support, debt_creation, and debt_settlement.",
        "9. Needed dimensions include funding_actor, funding_channel, source_box, target_box, beneficiary_box, obligation_box, cash_effect, debt_effect, and linked_debt_id.",
        "10. Debt-affecting cases are rows with debt/deuda/prestamo/repago/settlement evidence or future linked debt IDs.",
        "11. Unsupported flow drilldowns should become supported for funding totals and funding by actor/channel/cash effect.",
        "12. Stock/debt lineage should be used for debt balances, settlements, and debt-linked funding rows.",
        "",
        "## Prioritized implementation plan",
        "",
        "### Patch 1 — semantic classifier / rule IDs",
        "Files: `accounting/marts/semantic.py`. Expected behavior: classify contribution/support channels explicitly. Risks: historical migration and double counting. Acceptance: classification audit shows specific funding rule IDs.",
        "",
        "### Patch 2 — monthly_flow_semantic_split dimensions",
        "Files: `accounting/marts/semantic.py`. Expected behavior: propagate funding_actor, funding_channel, target_box, obligation_box, cash_effect, and debt fields. Risks: changed aggregation grain. Acceptance: monthly split has explicit funding dimensions.",
        "",
        "### Patch 3 — annual_balance_dashboard_metrics generation",
        "Files: `accounting/metrics/annual.py`. Expected behavior: annual funding totals by channel/actor/cash effect. Risks: new IDs need frontend mapping. Acceptance: annual metrics include dimensioned FUND rows.",
        "",
        "### Patch 4 — professional table labels / metric IDs",
        "Files: professional table builders/exporters. Expected behavior: rows carry stable metric IDs. Risks: notebook/code drift. Acceptance: labels no longer drive semantics.",
        "",
        "### Patch 5 — drilldown mapping for funding labels",
        "Files: `accounting/professional/drilldown.py`. Expected behavior: drilldowns filter by metric ID plus semantic dimensions. Risks: legacy rows may remain unsupported. Acceptance: funding rows produce supported lineage.",
        "",
        "### Patch 6 — debt linkage / stock lineage",
        "Files: `accounting/debt/resolve.py`, `accounting/marts/semantic.py`, `accounting/professional/drilldown.py`. Expected behavior: debt-linked support routes to debt activity/position. Risks: double-counting debt and funding. Acceptance: linked_debt_id connects funding rows to debt evidence.",
        "",
        "### Patch 7 — tests and QA checks",
        "Files: `tests/`. Expected behavior: fixtures cover rent, cash funding, direct obligation payments, HH→PM, FB support, and debt. Risks: insufficient real-ledger coverage. Acceptance: CLI and acceptance script pass.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(audit: pd.DataFrame, summary: pd.DataFrame, pack: Path, docs_dir: Path) -> dict[str, Path]:
    drilldown_dir = pack / "drilldown"
    digest_dir = pack / "digest"
    drilldown_dir.mkdir(parents=True, exist_ok=True)
    digest_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": drilldown_dir / "funding_lineage_audit.csv",
        "summary": drilldown_dir / "funding_lineage_summary.csv",
        "html": digest_dir / "accounting_professional_funding_lineage_audit.html",
        "markdown": docs_dir / f"professional_funding_semantics_audit_{_now_date()}.md",
    }
    audit.to_csv(paths["audit"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["html"].write_text(_render_html(audit, summary), encoding="utf-8")
    _write_markdown(paths["markdown"], audit, summary)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build funding/support lineage diagnostics for professional accounting reports.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--pack", type=Path, default=Path("out/professional_pack/latest_FBPM"))
    parser.add_argument("--run-root", type=Path, default=Path("out/run/accounting/latest_FBPM"))
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    pack = (repo_root / args.pack).resolve() if not args.pack.is_absolute() else args.pack.resolve()
    run_root = (repo_root / args.run_root).resolve() if not args.run_root.is_absolute() else args.run_root.resolve()
    audit, summary = build_audit(repo_root, pack, run_root)
    paths = write_outputs(audit, summary, pack, repo_root / "docs")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"audit rows: {len(audit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
