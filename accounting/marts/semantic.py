from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Tuple

import pandas as pd

RULE_VERSION = "semantic_pr9_treasury_fx_2026-07-01"
RULE_REGISTRY_COLUMNS = [
    "rule_id", "rule_version", "priority", "rule_name", "match_fields", "match_pattern",
    "semantic_bucket", "semantic_subbucket", "direction", "direction_source",
    "classification_confidence", "review_required", "warning", "notes", "active",
]
SEMANTIC_RULES = [
    {"rule_id":"R011_personal_expense_text","priority":10,"rule_name":"Personal expense text","match_fields":"Detalle,notes","match_pattern":"gastos personales","semantic_bucket":"family_withdrawal_candidate","semantic_subbucket":"personal_expense","direction":"out","direction_source":"rule_default","classification_confidence":"medium","review_required":False,"warning":"review family/informal withdrawal candidate","notes":"Personal/family expenses are distribution candidates, never property OPEX.","active":True},
    {"rule_id":"R001_rent_collections","priority":20,"rule_name":"Rent collections","match_fields":"Flujo,Tipo","match_pattern":"Flujo=cobros; Tipo=renta","semantic_bucket":"operating_revenue","semantic_subbucket":"rent","direction":"in","direction_source":"semantic_fallback","classification_confidence":"high","review_required":False,"warning":"","notes":"Operating rent revenue only.","active":True},
    {"rule_id":"R002_property_taxes","priority":30,"rule_name":"Property taxes","match_fields":"Tipo","match_pattern":"impuestos","semantic_bucket":"property_opex","semantic_subbucket":"taxes","direction":"out","direction_source":"semantic_fallback","classification_confidence":"high","review_required":False,"warning":"","notes":"True property OPEX.","active":True},
    {"rule_id":"R003_property_services","priority":40,"rule_name":"Property services","match_fields":"Tipo","match_pattern":"servicio|servicios","semantic_bucket":"property_opex","semantic_subbucket":"services","direction":"out","direction_source":"semantic_fallback","classification_confidence":"high","review_required":False,"warning":"","notes":"True property OPEX.","active":True},
    {"rule_id":"R004_property_maintenance","priority":50,"rule_name":"Property maintenance","match_fields":"Tipo","match_pattern":"mantenimiento","semantic_bucket":"property_opex","semantic_subbucket":"maintenance","direction":"out","direction_source":"semantic_fallback","classification_confidence":"high","review_required":False,"warning":"","notes":"True property OPEX.","active":True},
    {"rule_id":"R005_property_legal","priority":60,"rule_name":"Property legal","match_fields":"Tipo","match_pattern":"legal","semantic_bucket":"property_opex","semantic_subbucket":"legal","direction":"out","direction_source":"semantic_fallback","classification_confidence":"high","review_required":False,"warning":"","notes":"True property OPEX.","active":True},
    {"rule_id":"R006_contribution","priority":70,"rule_name":"Funding contribution","match_fields":"Flujo,Tipo","match_pattern":"contribucion|contribuciones","semantic_bucket":"funding_contribution","semantic_subbucket":"family_or_tenant_contribution","direction":"in","direction_source":"semantic_fallback","classification_confidence":"high","review_required":False,"warning":"","notes":"Funding is not operating revenue.","active":True},
    {"rule_id":"R007_debt_principal","priority":80,"rule_name":"Debt principal","match_fields":"Tipo","match_pattern":"prestamo","semantic_bucket":"debt_movement","semantic_subbucket":"principal","direction":"unknown","direction_source":"unknown","classification_confidence":"high","review_required":False,"warning":"","notes":"Debt principal is not property OPEX.","active":True},
    {"rule_id":"R008_debt_repayment","priority":90,"rule_name":"Debt repayment","match_fields":"Tipo","match_pattern":"repago","semantic_bucket":"debt_movement","semantic_subbucket":"repayment","direction":"unknown","direction_source":"unknown","classification_confidence":"high","review_required":False,"warning":"","notes":"Debt repayment is not property OPEX.","active":True},
    {"rule_id":"R009_debt_interest","priority":100,"rule_name":"Debt interest","match_fields":"Tipo","match_pattern":"interes","semantic_bucket":"debt_movement","semantic_subbucket":"interest","direction":"unknown","direction_source":"unknown","classification_confidence":"high","review_required":False,"warning":"","notes":"Debt interest remains separate from property OPEX.","active":True},
    {"rule_id":"R010_dividend","priority":110,"rule_name":"Dividend/distribution","match_fields":"Tipo","match_pattern":"dividendo","semantic_bucket":"family_withdrawal_candidate","semantic_subbucket":"dividend","direction":"out","direction_source":"rule_default","classification_confidence":"medium","review_required":False,"warning":"review family/informal withdrawal candidate","notes":"Dividend-like outflows are not property OPEX.","active":True},
    {"rule_id":"R012_transfer_expense","priority":120,"rule_name":"Transfer to family expense","match_fields":"Flujo,Tipo","match_pattern":"Flujo=transfer; Tipo=gasto","semantic_bucket":"family_withdrawal_candidate","semantic_subbucket":"transfer_to_family_expense","direction":"out","direction_source":"rule_default","classification_confidence":"medium","review_required":False,"warning":"review family/informal withdrawal candidate","notes":"Transfer/gasto is not property OPEX unless later explicitly substantiated.","active":True},
    {"rule_id":"R013_internal_transfer","priority":130,"rule_name":"Internal transfer","match_fields":"Flujo","match_pattern":"transfer","semantic_bucket":"internal_transfer","semantic_subbucket":"transfer","direction":"internal","direction_source":"rule_default","classification_confidence":"medium","review_required":False,"warning":"review if this transfer crosses economic owners","notes":"Internal transfers excluded from operating result.","active":True},
    {"rule_id":"R014_fx_conversion_proceeds","priority":140,"rule_name":"FX conversion proceeds","match_fields":"Flujo,Tipo,payer,receiver,cash_path","match_pattern":"Cambio:FX|payer=FX|receiver=FX","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_conversion_proceeds","direction":"unknown","direction_source":"rule_default","classification_confidence":"medium","review_required":False,"warning":"FX conversion proceeds; not revenue, not funding, not property OPEX","notes":"ARS proceeds from currency conversion. Treasury movement only.","active":True},
    {"rule_id":"R015_fx_cost_or_spread","priority":150,"rule_name":"FX cost or spread","match_fields":"Flujo,Tipo,cash_path","match_pattern":"Costo Operativo:FX|Tipo=FX","semantic_bucket":"treasury_fx","semantic_subbucket":"fx_cost_or_spread","direction":"out","direction_source":"rule_default","classification_confidence":"medium","review_required":False,"warning":"FX cost/spread/loss; not property OPEX unless explicitly reclassified","notes":"Treasury/financial FX cost.","active":True},
    {"rule_id":"R999_unknown_review_required","priority":999,"rule_name":"Unknown review required","match_fields":"*","match_pattern":"no conservative semantic rule matched","semantic_bucket":"unknown","semantic_subbucket":"review_required","direction":"unknown","direction_source":"unknown","classification_confidence":"low","review_required":True,"warning":"review_required","notes":"Ambiguous rows stay visible and are not forced into OPEX.","active":True},
]


AUDIT_COLUMNS = [
    "tx_id", "Date", "period", "period_end", "Currency", "amount", "Box", "Lugar",
    "payer", "receiver", "Flujo", "Tipo", "Detalle", "semantic_bucket", "semantic_subbucket", "direction",
    "direction_source", "direction_confidence", "actor", "counterparty", "funding_actor",
    "funding_channel", "source_box", "target_box", "beneficiary_box", "obligation_box",
    "payment_channel", "cash_effect", "debt_effect", "linked_debt_id", "channel", "cash_path",
    "rule_id", "rule_version", "classification_confidence", "classification_status",
    "review_required", "warning", "notes",
]
SUMMARY_COLUMNS = [
    "period", "Currency", "semantic_bucket", "semantic_subbucket", "classification_status",
    "review_required", "amount_total", "amount_abs_total", "n_tx", "rule_id",
    "sample_detalle", "sample_payer", "sample_receiver",
]
MONTHLY_COLUMNS = [
    "period", "period_end", "Currency", "Box", "Lugar", "actor", "counterparty", "payer",
    "receiver", "funding_actor", "funding_channel", "source_box", "target_box",
    "beneficiary_box", "obligation_box", "payment_channel", "cash_effect", "debt_effect",
    "linked_debt_id", "channel", "cash_path", "semantic_bucket", "semantic_subbucket",
    "amount_in", "amount_out", "net_amount", "amount_abs", "n_tx", "classification_status",
    "classification_confidence", "review_required", "source_table", "source_tx_ids_sample",
    "rule_ids", "notes",
]
VALIDATION_COLUMNS = ["check", "period", "Currency", "amount", "n_tx", "warning"]
OPERATING_STATEMENT_COLUMNS = [
    "period", "period_end", "Currency", "statement_line", "label", "semantic_category",
    "amount", "source_table", "source_filter", "n_tx", "classification_coverage_ratio",
    "unknown_amount", "review_required_amount", "caveat", "frontend_suitability",
]
OPERATING_STATEMENT_QA_COLUMNS = ["check", "status", "detail", "severity"]
SEMANTIC_LEAKAGE_QA_COLUMNS = ["tx_id", "period", "Currency", "amount", "Box", "Lugar", "Flujo", "Tipo", "Detalle", "payer", "receiver", "semantic_bucket", "semantic_subbucket", "rule_id", "leakage_pattern", "severity", "recommended_bucket", "notes"]


def _norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _norm_key(value: Any) -> str:
    return _norm(value).casefold()


def _infer_box_party(box: Any) -> str:
    b = _norm(box)
    if not b:
        return ""
    if b.casefold() == "household":
        return "HH"
    return "".join(part[0].upper() for part in b.split() if part and part[0].isalpha())


def _semantic_blob(row: pd.Series) -> str:
    fields = [
        "payer", "receiver", "actor", "counterparty", "Box", "Lugar", "Flujo",
        "Tipo", "Detalle", "notes", "channel", "cash_path", "semantic_bucket",
        "semantic_subbucket", "rule_id",
    ]
    return " ".join(_norm(row.get(c)) for c in fields if c in row.index)


def _actor_from_text(row: pd.Series) -> str:
    blob = _semantic_blob(row).casefold()
    checks = [
        ("Matías", r"mat[ií]as|matias|\bmi\b"),
        ("Alejandro", r"alejandro|\balen\b|\bale\b"),
        ("Primos", r"primos"),
        ("Héctor", r"h[eé]ctor|hector"),
        ("Inquilino", r"inquil|\binq\b"),
        ("Household", r"household|\bhh\b"),
    ]
    for actor, pattern in checks:
        if re.search(pattern, blob, flags=re.IGNORECASE):
            return actor
    return ""


def _is_funding_support_candidate(row: pd.Series) -> bool:
    bucket = _norm(row.get("semantic_bucket"))
    subbucket = _norm(row.get("semantic_subbucket")).casefold()
    blob = _semantic_blob(row).casefold()
    if bucket in {"funding_contribution", "debt_movement"}:
        return True
    if bucket == "property_opex" and subbucket in {"taxes", "services"} and _actor_from_text(row):
        return True
    return bool(re.search(r"fund|aporte|contrib|support|soporte|deuda|debt", blob, flags=re.IGNORECASE))


def _box_for_row(row: pd.Series) -> str:
    return _norm(row.get("Box"))


def _derive_funding_dimensions(row: pd.Series) -> dict[str, str]:
    bucket = _norm(row.get("semantic_bucket"))
    subbucket = _norm(row.get("semantic_subbucket")).casefold()
    direction = _norm(row.get("direction"))
    box = _box_for_row(row)
    blob = _semantic_blob(row).casefold()
    is_support = _is_funding_support_candidate(row)
    actor = _actor_from_text(row) if is_support else ""

    source_box = box if direction == "out" else ""
    target_box = box if direction == "in" or is_support else ""
    beneficiary_box = target_box
    obligation_box = ""
    payment_channel = _norm(row.get("channel"))
    linked_debt_id = _norm(row.get("linked_debt_id"))
    funding_channel = ""
    cash_effect = ""
    debt_effect = "none"

    if direction == "in":
        cash_effect = "cash_in_box"
    elif direction == "out":
        cash_effect = "cash_out_box"
    elif direction == "internal":
        cash_effect = "non_cash_support"

    if bucket == "debt_movement":
        if subbucket == "repayment" or re.search(r"repago|repayment|settle|sald", blob):
            debt_effect = "settles_debt"
            funding_channel = "debt_settlement"
        elif subbucket == "principal" or re.search(r"prestamo|principal|loan|deuda|debt", blob):
            debt_effect = "creates_debt"
            funding_channel = "debt_creation"
        else:
            debt_effect = "ambiguous"
            funding_channel = "other"
        cash_effect = cash_effect or "non_cash_support"
        beneficiary_box = beneficiary_box or box
        target_box = target_box or box
    elif bucket == "funding_contribution":
        if actor == "Inquilino":
            funding_channel = "tenant_to_box"
        elif actor == "Household" and (box == "Property Management" or re.search(r"property management|\bpm\b", blob)):
            funding_channel = "household_to_pm"
        elif actor in {"Alejandro", "Primos"} or box == "Family Business" or re.search(r"family business|\bfb\b", blob):
            funding_channel = "family_business_contribution"
        elif actor:
            funding_channel = "named_actor_support"
        else:
            funding_channel = "cash_to_box" if direction == "in" else "other"
        cash_effect = cash_effect or ("cash_in_box" if direction == "in" else "non_cash_support")
        beneficiary_box = beneficiary_box or box
        target_box = target_box or box
    elif bucket == "property_opex" and subbucket in {"taxes", "services"} and actor:
        if actor == "Inquilino" and subbucket == "taxes":
            funding_channel = "tenant_direct_tax_payment"
        elif actor == "Inquilino" and subbucket == "services":
            funding_channel = "tenant_direct_service_payment"
        elif actor == "Household" and box == "Property Management":
            funding_channel = "household_to_pm"
        else:
            funding_channel = "named_actor_support"
        cash_effect = "no_cash_in_box_direct_payment"
        obligation_box = box
        beneficiary_box = box
        target_box = box
        source_box = ""
        if debt_effect == "none" and re.search(r"deuda|debt|prestamo|repago|repayment", blob):
            debt_effect = "ambiguous"
    elif is_support:
        funding_channel = "named_actor_support" if actor else "other"
        beneficiary_box = beneficiary_box or box
        target_box = target_box or box
        cash_effect = cash_effect or "non_cash_support"

    return {
        "funding_actor": actor,
        "funding_channel": funding_channel,
        "source_box": source_box,
        "target_box": target_box,
        "beneficiary_box": beneficiary_box,
        "obligation_box": obligation_box,
        "payment_channel": payment_channel,
        "cash_effect": cash_effect,
        "debt_effect": debt_effect,
        "linked_debt_id": linked_debt_id,
    }


def semantic_rule_registry_frame() -> pd.DataFrame:
    return pd.DataFrame([{**r, "rule_version": RULE_VERSION} for r in SEMANTIC_RULES], columns=RULE_REGISTRY_COLUMNS)


def _classify_row(row: pd.Series) -> Tuple[str, str, str, str, bool, str, str]:
    flujo = _norm_key(row.get("Flujo"))
    tipo = _norm_key(row.get("Tipo"))
    detail_blob = " ".join(_norm(row.get(c)) for c in ("Detalle", "notes") if c in row.index).casefold()

    if "gastos personales" in detail_blob:
        return ("family_withdrawal_candidate", "personal_expense", "R011_personal_expense_text", "medium", False, "classified", "review family/informal withdrawal candidate")
    if flujo == "cobros" and tipo == "renta":
        return ("operating_revenue", "rent", "R001_rent_collections", "high", False, "classified", "")
    if tipo == "impuestos":
        return ("property_opex", "taxes", "R002_property_taxes", "high", False, "classified", "")
    if tipo in {"servicio", "servicios"}:
        return ("property_opex", "services", "R003_property_services", "high", False, "classified", "")
    if tipo == "mantenimiento":
        return ("property_opex", "maintenance", "R004_property_maintenance", "high", False, "classified", "")
    if tipo == "legal":
        return ("property_opex", "legal", "R005_property_legal", "high", False, "classified", "")
    if flujo == "contribucion" or tipo in {"contribucion", "contribuciones"}:
        return ("funding_contribution", "family_or_tenant_contribution", "R006_contribution", "high", False, "classified", "")
    if tipo == "prestamo":
        return ("debt_movement", "principal", "R007_debt_principal", "high", False, "classified", "")
    if tipo == "repago":
        return ("debt_movement", "repayment", "R008_debt_repayment", "high", False, "classified", "")
    if tipo == "interes":
        return ("debt_movement", "interest", "R009_debt_interest", "high", False, "classified", "")

    cash_path = _norm_key(row.get("cash_path"))
    payer = _norm_key(row.get("payer"))
    receiver = _norm_key(row.get("receiver"))
    fx_blob = " ".join([flujo, tipo, cash_path, detail_blob, payer, receiver])
    is_debt = tipo in {"prestamo", "repago", "interes"} or any(token in fx_blob for token in ["principal", "repago", "repayment", "interest", "interes"])
    is_fx_cost = (
        "costo operativo:fx" in fx_blob
        or (receiver == "costos" and "fx" in fx_blob)
        or any(token in fx_blob for token in ["fx cost", "spread", "commission", "comision", "comisión", "loss", "perdida", "pérdida"])
    )
    if is_fx_cost and not is_debt:
        return ("treasury_fx", "fx_cost_or_spread", "R015_fx_cost_or_spread", "high", False, "classified", "FX cost/spread/loss; not property OPEX unless explicitly reclassified")
    is_fx_conversion = (
        "cambio:fx" in fx_blob
        or payer == "fx"
        or receiver == "fx"
        or ("cambio" in fx_blob and "fx" in fx_blob)
    )
    if is_fx_conversion and not is_debt:
        subbucket = "fx_conversion_outflow" if receiver == "fx" and payer != "fx" else "fx_conversion_proceeds"
        confidence = "high" if "cambio:fx" in fx_blob or payer == "fx" or receiver == "fx" else "medium"
        return ("treasury_fx", subbucket, "R014_fx_conversion_proceeds", confidence, False, "classified", "FX conversion proceeds; not revenue, not funding, not property OPEX")

    if tipo == "dividendo":
        return ("family_withdrawal_candidate", "dividend", "R010_dividend", "medium", False, "classified", "review family/informal withdrawal candidate")
    if flujo == "transfer" and tipo == "gasto":
        return ("family_withdrawal_candidate", "transfer_to_family_expense", "R012_transfer_expense", "medium", False, "classified", "review family/informal withdrawal candidate")
    if flujo == "transfer":
        return ("internal_transfer", "transfer", "R013_internal_transfer", "medium", False, "classified", "review if this transfer crosses economic owners")
    return ("unknown", "review_required", "R999_unknown_review_required", "low", True, "review_required", "no conservative semantic rule matched")


def _prepare_ledger(ledger: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    df = ledger.copy()
    if "amount" not in df.columns and "amount_cents" in df.columns:
        df["amount"] = pd.to_numeric(df["amount_cents"], errors="coerce").fillna(0.0) / 100.0
    df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df = df[df["Date"].notna()].copy()
    period = df["Date"].dt.to_period(freq)
    df["period"] = period.astype(str)
    df["period_end"] = period.dt.end_time.dt.date.astype(str)
    for col in ["tx_id", "Currency", "Box", "Lugar", "Flujo", "Tipo", "Detalle", "payer", "receiver", "status", "notes", "channel", "cash_path"]:
        if col not in df.columns:
            df[col] = ""
    return df


def build_semantic_outputs(ledger: pd.DataFrame, out_dir: Path, freq: str = "M") -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _prepare_ledger(ledger, freq=freq)

    rule_registry = semantic_rule_registry_frame()
    classified = df.apply(_classify_row, axis=1, result_type="expand")
    classified.columns = ["semantic_bucket", "semantic_subbucket", "rule_id", "classification_confidence", "review_required", "classification_status", "warning"]
    audit = pd.concat([df, classified], axis=1)
    audit["rule_version"] = RULE_VERSION
    if "channel" not in audit.columns or audit["channel"].astype(str).str.strip().eq("").all():
        audit["channel"] = audit["Box"]
    if "cash_path" not in audit.columns or audit["cash_path"].astype(str).str.strip().eq("").all():
        audit["cash_path"] = audit["Flujo"].astype(str) + ":" + audit["Tipo"].astype(str)

    payer = audit["payer"].map(_norm).str.upper()
    receiver = audit["receiver"].map(_norm).str.upper()
    box_party = audit["Box"].map(_infer_box_party).str.upper()
    audit["direction"] = "unknown"
    audit["direction_source"] = "unknown"
    audit["direction_confidence"] = "low"
    matched_in = receiver.eq(box_party) & box_party.ne("")
    matched_out = payer.eq(box_party) & box_party.ne("")
    audit.loc[matched_in, ["direction", "direction_source", "direction_confidence"]] = ["in", "box_party_match", "high"]
    audit.loc[matched_out, ["direction", "direction_source", "direction_confidence"]] = ["out", "box_party_match", "high"]
    audit.loc[matched_in & matched_out, ["direction", "direction_source", "direction_confidence"]] = ["internal", "box_party_match", "high"]

    default_direction = audit["rule_id"].map(rule_registry.set_index("rule_id")["direction"]).fillna("unknown")
    default_source = audit["rule_id"].map(rule_registry.set_index("rule_id")["direction_source"]).fillna("unknown")
    unknown_direction = audit["direction"].eq("unknown") & default_direction.ne("unknown")
    audit.loc[unknown_direction, "direction"] = default_direction[unknown_direction]
    audit.loc[unknown_direction, "direction_source"] = default_source[unknown_direction].where(default_source[unknown_direction].ne("unknown"), "semantic_fallback")
    audit.loc[unknown_direction, "direction_confidence"] = audit.loc[unknown_direction, "direction_source"].map({"semantic_fallback":"medium", "rule_default":"medium", "explicit_direction":"high"}).fillna("low")
    audit["actor"] = audit["Box"].where(audit["Box"].astype(str).str.strip().ne(""), box_party)
    audit["counterparty"] = audit["receiver"].where(audit["direction"].eq("out"), audit["payer"])
    if "linked_debt_id" not in audit.columns:
        audit["linked_debt_id"] = ""
    funding_dimensions = audit.apply(_derive_funding_dimensions, axis=1, result_type="expand")
    for col in [
        "funding_actor", "funding_channel", "source_box", "target_box", "beneficiary_box",
        "obligation_box", "payment_channel", "cash_effect", "debt_effect", "linked_debt_id",
    ]:
        audit[col] = funding_dimensions[col] if col in funding_dimensions.columns else ""

    audit["Date"] = audit["Date"].dt.date.astype(str)
    audit["review_required"] = audit["review_required"].astype(bool)
    period_end_lookup = audit[["period", "period_end"]].drop_duplicates()
    audit = audit[AUDIT_COLUMNS]

    summary = audit.groupby(["period", "Currency", "semantic_bucket", "semantic_subbucket", "classification_status", "review_required", "rule_id"], dropna=False).agg(
        amount_total=("amount", "sum"), amount_abs_total=("amount", lambda s: s.abs().sum()), n_tx=("tx_id", "count"),
        sample_detalle=("Detalle", "first"), sample_payer=("payer", "first"), sample_receiver=("receiver", "first"),
    ).reset_index()[SUMMARY_COLUMNS]

    work = audit.copy()
    work["amount_in"] = work["amount"].where(work["direction"].eq("in"), 0.0)
    work["amount_out"] = work["amount"].where(work["direction"].eq("out"), 0.0)
    work["net_amount"] = work["amount_in"] - work["amount_out"]
    work["amount_abs"] = work["amount"].abs()
    monthly = work.groupby([
        "period", "Currency", "Box", "Lugar", "actor", "counterparty", "payer", "receiver",
        "funding_actor", "funding_channel", "source_box", "target_box", "beneficiary_box",
        "obligation_box", "payment_channel", "cash_effect", "debt_effect", "linked_debt_id",
        "channel", "cash_path", "semantic_bucket", "semantic_subbucket",
    ], dropna=False).agg(
        amount_in=("amount_in", "sum"), amount_out=("amount_out", "sum"), net_amount=("net_amount", "sum"), amount_abs=("amount_abs", "sum"),
        n_tx=("tx_id", "count"), classification_status=("classification_status", lambda s: "review_required" if (s == "review_required").any() else "classified"),
        classification_confidence=("classification_confidence", lambda s: "low" if "low" in set(s) else ("medium" if "medium" in set(s) else "high")),
        review_required=("review_required", "max"), source_tx_ids_sample=("tx_id", lambda s: ";".join(s.astype(str).head(10))),
        rule_ids=("rule_id", lambda s: ";".join(sorted(set(s.astype(str))))), notes=("warning", lambda s: "; ".join(sorted(set(x for x in s.astype(str) if x)))),
    ).reset_index()
    monthly = monthly.merge(period_end_lookup, on="period", how="left")
    monthly["source_table"] = "ledger_canonical.csv"
    monthly = monthly[MONTHLY_COLUMNS]

    validations = _build_validation_rows(audit, monthly)
    leakage = build_semantic_leakage_qa(audit)
    paths = {
        "semantic_rule_registry": out_dir / "semantic_rule_registry.csv",
        "classification_audit": out_dir / "classification_audit.csv",
        "classification_audit_summary": out_dir / "classification_audit_summary.csv",
        "monthly_flow_semantic_split": out_dir / "monthly_flow_semantic_split.csv",
        "classification_validation": out_dir / "classification_validation.csv",
        "semantic_leakage_qa": out_dir / "semantic_leakage_qa.csv",
        "semantic_dashboard_coverage": out_dir / "semantic_dashboard_coverage.csv",
    }
    rule_registry.to_csv(paths["semantic_rule_registry"], index=False)
    audit.to_csv(paths["classification_audit"], index=False)
    summary.to_csv(paths["classification_audit_summary"], index=False)
    monthly.to_csv(paths["monthly_flow_semantic_split"], index=False)
    validations.to_csv(paths["classification_validation"], index=False)
    leakage.to_csv(paths["semantic_leakage_qa"], index=False)
    build_semantic_dashboard_coverage().to_csv(paths["semantic_dashboard_coverage"], index=False)
    return paths


def build_semantic_dashboard_coverage() -> pd.DataFrame:
    rows = [
        ("Renta total", "operating_revenue", "rent", "Currency", "monthly_flow_semantic_split.csv", "supported", "", "Aggregate amount_in for rent revenue."),
        ("Renta CABA", "operating_revenue", "rent", "Lugar", "monthly_flow_semantic_split.csv", "supported_if_present", "", "Filter rent by Lugar when populated."),
        ("Renta Torcuato", "operating_revenue", "rent", "Lugar", "monthly_flow_semantic_split.csv", "supported_if_present", "", "Filter rent by Lugar when populated."),
        ("Impuestos", "property_opex", "taxes", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line taxes."),
        ("Servicios", "property_opex", "services", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line services."),
        ("Mantenimiento", "property_opex", "maintenance", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line maintenance."),
        ("Legal", "property_opex", "legal", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line legal."),
        ("OPEX patrimonial", "property_opex", "*", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line property_opex_true excludes funding, debt, and distributions."),
        ("Resultado operativo", "operating_result", "net_operating", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line net_operating."),
        ("Contribuciones por actor", "funding_contribution", "family_or_tenant_contribution", "funding_actor,funding_channel,target_box,obligation_box,cash_effect,debt_effect", "monthly_flow_semantic_split.csv", "supported_if_present", "", "Use explicit funding dimensions; do not infer funding semantics from renderer labels."),
        ("Gasto personal", "family_withdrawal_candidate", "personal_expense", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line personal_expenses."),
        ("Dividendos", "family_withdrawal_candidate", "dividend", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line dividends."),
        ("Retiros/distribución", "family_withdrawal_candidate", "*", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line family_draws_or_distributions."),
        ("Cobertura después de retiros", "coverage", "coverage_after_draws", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line coverage_after_draws."),
        ("Unknown / review-required", "unknown", "review_required", "Currency", "monthly_operating_statement.csv", "supported", "", "Statement line unknown_or_ambiguous_outflows plus audit rows."),
        ("Treasury FX", "treasury_fx", "*", "Currency,Box", "monthly_operating_statement.csv", "supported", "", "Treasury FX bridge lines; excluded from operating result, funding, draws, debt, and property OPEX."),
        ("Caja FB", "cash", "cash_close", "cash mart", "monthly_cash_close.csv", "not_semantic_mart", "cash mart responsibility", "Semantic mart does not validate cash."),
        ("Deuda fin", "debt", "debt_position", "debt mart", "monthly_debt_position.csv", "not_semantic_mart", "debt mart responsibility", "Semantic mart only excludes debt movement from OPEX."),
    ]
    return pd.DataFrame(rows, columns=["dashboard_line", "required_semantic_bucket", "required_semantic_subbucket", "required_dimension", "source_output", "status", "missing_reason", "notes"])


def build_semantic_leakage_qa(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame(columns=SEMANTIC_LEAKAGE_QA_COLUMNS)
    text_cols = [c for c in ["Flujo", "Tipo", "Detalle", "notes", "payer", "receiver", "rule_id"] if c in audit.columns]
    patterns = {
        "Gastos Personales": r"gastos\s+personales",
        "Personal": r"\bpersonal\b",
        "Dividendo": r"dividendo|dividend",
        "Retiro": r"retiro|withdrawal",
        "Distribucion": r"distribuci[oó]n|distribution",
        "Transfer + Gasto": r"transfer.*gasto|gasto.*transfer",
        "family withdrawal": r"family\s+withdrawal",
        "draw": r"\bdraw\b",
    }
    opex = audit.loc[audit["semantic_bucket"].astype(str).eq("property_opex")].copy()
    rows = []
    for _, r in opex.iterrows():
        blob = " ".join(_norm(r.get(c)) for c in text_cols).casefold()
        hits = [name for name, pat in patterns.items() if re.search(pat, blob, flags=re.IGNORECASE)]
        for hit in hits:
            rows.append({
                "tx_id": r.get("tx_id", ""), "period": r.get("period", ""), "Currency": r.get("Currency", ""),
                "amount": r.get("amount", pd.NA), "Box": r.get("Box", ""), "Lugar": r.get("Lugar", ""), "Flujo": r.get("Flujo", ""),
                "Tipo": r.get("Tipo", ""), "Detalle": r.get("Detalle", ""), "payer": r.get("payer", ""),
                "receiver": r.get("receiver", ""), "semantic_bucket": r.get("semantic_bucket", ""),
                "semantic_subbucket": r.get("semantic_subbucket", ""), "rule_id": r.get("rule_id", ""),
                "leakage_pattern": hit, "severity": "error",
                "recommended_bucket": "family_withdrawal_candidate_review",
                "notes": "Property OPEX contains personal/family/distribution-like text; review classification rule before frontend/reporting use.",
            })
    return pd.DataFrame(rows, columns=SEMANTIC_LEAKAGE_QA_COLUMNS)


def build_monthly_operating_statement_from_split(
    split: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "period", "period_end", "Currency", "semantic_bucket", "semantic_subbucket",
        "amount_in", "amount_out", "net_amount", "amount_abs", "n_tx", "review_required",
    ]
    missing = [c for c in required if c not in split.columns]
    if missing:
        raise ValueError(f"monthly_flow_semantic_split.csv missing required columns for operating statement: {missing}")

    df = split.copy()
    for col in ["amount_in", "amount_out", "net_amount", "amount_abs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["n_tx"] = pd.to_numeric(df["n_tx"], errors="coerce").fillna(0).astype(int)
    df["review_required"] = df["review_required"].astype(str).str.lower().isin({"true", "1", "yes", "y"})

    scope_label = "supplied_semantic_universe"
    keys = ["period", "period_end", "Currency"]
    groups = df[keys].drop_duplicates().sort_values(keys).to_dict("records")
    rows = []

    def mask_for(bucket: str | None = None, subbucket: str | None = None, review_required: bool | None = None):
        mask = pd.Series(True, index=df.index)
        if bucket is not None:
            mask &= df["semantic_bucket"].eq(bucket)
        if subbucket is not None:
            mask &= df["semantic_subbucket"].eq(subbucket)
        if review_required is not None:
            mask &= df["review_required"].eq(review_required)
        return mask

    def add_row(base: dict[str, Any], line: str, label: str, category: str, amount: float, source_filter: str, n_tx: int, coverage: float, unknown: float, review: float, caveat: str = "", frontend: str = "safe_canonical") -> None:
        rows.append({
            "period": base["period"],
            "period_end": base["period_end"],
            "Currency": base["Currency"],
            "statement_line": line,
            "label": label,
            "semantic_category": category,
            "amount": float(amount),
            "source_table": "monthly_flow_semantic_split.csv",
            "source_filter": f"{scope_label}; {source_filter}",
            "n_tx": int(n_tx),
            "classification_coverage_ratio": float(coverage),
            "unknown_amount": float(unknown),
            "review_required_amount": float(review),
            "caveat": caveat,
            "frontend_suitability": frontend,
        })

    for base in groups:
        gmask = (df["period"].eq(base["period"]) & df["period_end"].eq(base["period_end"]) & df["Currency"].eq(base["Currency"]))
        g = df.loc[gmask].copy()
        eligible_abs = float(g["amount_abs"].sum())
        unknown_mask = g["semantic_bucket"].eq("unknown") | g["review_required"]
        unknown_amount = float(g.loc[unknown_mask, "amount_abs"].sum())
        review_amount = float(g.loc[g["review_required"], "amount_abs"].sum())
        classified_abs = float(g.loc[~unknown_mask, "amount_abs"].sum())
        coverage = classified_abs / eligible_abs if eligible_abs else 1.0

        op_rev = g.loc[g["semantic_bucket"].eq("operating_revenue"), "amount_in"].sum()
        rent = g.loc[g["semantic_bucket"].eq("operating_revenue") & g["semantic_subbucket"].eq("rent"), "amount_in"].sum()
        opex = g.loc[g["semantic_bucket"].eq("property_opex"), "amount_out"].sum()
        taxes = g.loc[g["semantic_bucket"].eq("property_opex") & g["semantic_subbucket"].eq("taxes"), "amount_out"].sum()
        services = g.loc[g["semantic_bucket"].eq("property_opex") & g["semantic_subbucket"].eq("services"), "amount_out"].sum()
        maintenance = g.loc[g["semantic_bucket"].eq("property_opex") & g["semantic_subbucket"].eq("maintenance"), "amount_out"].sum()
        legal = g.loc[g["semantic_bucket"].eq("property_opex") & g["semantic_subbucket"].eq("legal"), "amount_out"].sum()
        explicit_opex = {"taxes", "services", "maintenance", "legal"}
        other_opex = g.loc[g["semantic_bucket"].eq("property_opex") & ~g["semantic_subbucket"].isin(explicit_opex), "amount_out"].sum()
        funding = g.loc[g["semantic_bucket"].eq("funding_contribution"), "amount_in"].sum()
        draws_mask = g["semantic_bucket"].isin(["family_withdrawal_candidate", "family_withdrawal"])
        draws = g.loc[draws_mask, "amount_out"].sum()
        personal_expenses = g.loc[draws_mask & g["semantic_subbucket"].eq("personal_expense"), "amount_out"].sum()
        dividends = g.loc[draws_mask & g["semantic_subbucket"].eq("dividend"), "amount_out"].sum()
        transfer_family_expense = g.loc[draws_mask & g["semantic_subbucket"].eq("transfer_to_family_expense"), "amount_out"].sum()
        debt = g.loc[g["semantic_bucket"].eq("debt_movement"), "amount_abs"].sum()
        transfers = g.loc[g["semantic_bucket"].eq("internal_transfer"), "amount_abs"].sum()
        fx_mask = g["semantic_bucket"].eq("treasury_fx")
        fx_conversion_in = g.loc[fx_mask & g["semantic_subbucket"].eq("fx_conversion_proceeds"), "amount_in"].sum()
        fx_conversion_out = g.loc[fx_mask & g["semantic_subbucket"].eq("fx_conversion_outflow"), "amount_out"].sum()
        fx_cost = g.loc[fx_mask & g["semantic_subbucket"].eq("fx_cost_or_spread"), "amount_out"].sum()
        fx_other = g.loc[fx_mask & ~g["semantic_subbucket"].isin(["fx_conversion_proceeds", "fx_conversion_outflow", "fx_cost_or_spread"]), "net_amount"].sum()
        fx_net = float(fx_conversion_in - fx_conversion_out - fx_cost + fx_other)
        unknown_out = g.loc[unknown_mask, "amount_out"].sum()
        if unknown_out == 0:
            unknown_out = unknown_amount
        net_operating = float(op_rev - opex)
        coverage_after_draws = float(net_operating + funding - draws)

        def ntx(bucket: str | None = None, subbucket: str | None = None) -> int:
            m = pd.Series(True, index=g.index)
            if bucket is not None:
                m &= g["semantic_bucket"].eq(bucket)
            if subbucket is not None:
                m &= g["semantic_subbucket"].eq(subbucket)
            return int(g.loc[m, "n_tx"].sum())

        caveat = "Excludes funding, family withdrawals, debt movements, internal transfers, and unknown/review-required flows from net operating."
        add_row(base, "operating_revenue", "Operating revenue", "operating", op_rev, "semantic_bucket=operating_revenue; amount_in", ntx("operating_revenue"), coverage, unknown_amount, review_amount, caveat)
        add_row(base, "rent_revenue", "Rent revenue", "operating_detail", rent, "semantic_bucket=operating_revenue; semantic_subbucket=rent; amount_in", ntx("operating_revenue", "rent"), coverage, unknown_amount, review_amount, caveat)
        add_row(base, "property_opex_true", "True property operating expense", "operating", opex, "semantic_bucket=property_opex; amount_out", ntx("property_opex"), coverage, unknown_amount, review_amount, caveat)
        add_row(base, "taxes", "Taxes", "property_opex_detail", taxes, "semantic_bucket=property_opex; semantic_subbucket=taxes; amount_out", ntx("property_opex", "taxes"), coverage, unknown_amount, review_amount, caveat)
        add_row(base, "services", "Services", "property_opex_detail", services, "semantic_bucket=property_opex; semantic_subbucket=services; amount_out", ntx("property_opex", "services"), coverage, unknown_amount, review_amount, caveat)
        add_row(base, "maintenance", "Maintenance", "property_opex_detail", maintenance, "semantic_bucket=property_opex; semantic_subbucket=maintenance; amount_out", ntx("property_opex", "maintenance"), coverage, unknown_amount, review_amount, caveat)
        add_row(base, "legal", "Legal", "property_opex_detail", legal, "semantic_bucket=property_opex; semantic_subbucket=legal; amount_out", ntx("property_opex", "legal"), coverage, unknown_amount, review_amount, caveat)
        add_row(base, "other_property_opex", "Other property OPEX", "property_opex_detail", other_opex, "semantic_bucket=property_opex; semantic_subbucket not in explicit property opex list; amount_out", ntx("property_opex"), coverage, unknown_amount, review_amount, "Should remain zero unless future explicit OPEX rules are added.", "review_before_frontend")
        add_row(base, "net_operating", "Net operating", "operating_result", net_operating, "operating_revenue - property_opex_true", ntx("operating_revenue") + ntx("property_opex"), coverage, unknown_amount, review_amount, caveat)
        add_row(base, "funding_contributions", "Funding contributions", "non_operating_funding", funding, "semantic_bucket=funding_contribution; amount_in", ntx("funding_contribution"), coverage, unknown_amount, review_amount, "Shown separately; not operating revenue.")
        add_row(base, "family_draws_or_distributions", "Family draws or distributions", "non_operating_distribution", draws, "semantic_bucket in family_withdrawal_candidate,family_withdrawal; amount_out", int(g.loc[draws_mask, "n_tx"].sum()), coverage, unknown_amount, review_amount, "Candidate line for review; not property OPEX.", "review_before_frontend")
        add_row(base, "personal_expenses", "Personal expenses", "non_operating_distribution_detail", personal_expenses, "semantic_subbucket=personal_expense; amount_out", ntx("family_withdrawal_candidate", "personal_expense"), coverage, unknown_amount, review_amount, "Personal expenses are distribution candidates; not property OPEX.", "review_before_frontend")
        add_row(base, "dividends", "Dividends", "non_operating_distribution_detail", dividends, "semantic_subbucket=dividend; amount_out", ntx("family_withdrawal_candidate", "dividend"), coverage, unknown_amount, review_amount, "Dividend-like flows are not property OPEX.", "review_before_frontend")
        add_row(base, "transfer_to_family_expense", "Transfer to family expense", "non_operating_distribution_detail", transfer_family_expense, "semantic_subbucket=transfer_to_family_expense; amount_out", ntx("family_withdrawal_candidate", "transfer_to_family_expense"), coverage, unknown_amount, review_amount, "Transfer/gasto candidate; not property OPEX.", "review_before_frontend")
        add_row(base, "coverage_after_draws", "Coverage after draws", "coverage", coverage_after_draws, "net_operating + funding_contributions - family_draws_or_distributions", int(g["n_tx"].sum()), coverage, unknown_amount, review_amount, "Coverage-like cash view; not a pure operating result.", "review_before_frontend")
        add_row(base, "treasury_fx_conversion_in", "FX conversion proceeds", "treasury", fx_conversion_in, "semantic_bucket=treasury_fx; semantic_subbucket=fx_conversion_proceeds; amount_in", ntx("treasury_fx", "fx_conversion_proceeds"), coverage, unknown_amount, review_amount, "FX conversion changes liquidity by currency but is not operating income or funding.", "safe_with_caveat")
        add_row(base, "treasury_fx_conversion_out", "FX conversion outflow", "treasury", fx_conversion_out, "semantic_bucket=treasury_fx; semantic_subbucket=fx_conversion_outflow; amount_out", ntx("treasury_fx", "fx_conversion_outflow"), coverage, unknown_amount, review_amount, "FX conversion changes liquidity by currency but is not operating income or funding.", "safe_with_caveat")
        add_row(base, "treasury_fx_cost", "FX cost / spread", "treasury", fx_cost, "semantic_bucket=treasury_fx; semantic_subbucket=fx_cost_or_spread; amount_out", ntx("treasury_fx", "fx_cost_or_spread"), coverage, unknown_amount, review_amount, "Treasury/financial FX cost; not property OPEX unless explicitly reclassified.", "safe_with_caveat")
        add_row(base, "treasury_fx_net", "FX net treasury effect", "treasury", fx_net, "treasury_fx_conversion_in - treasury_fx_conversion_out - treasury_fx_cost +/- other_fx", int(g.loc[fx_mask, "n_tx"].sum()), coverage, unknown_amount, review_amount, "FX conversion changes liquidity by currency but is not operating income or funding.", "safe_with_caveat")
        add_row(base, "unknown_or_ambiguous_outflows", "Unknown or ambiguous outflows", "data_quality", unknown_out, "semantic_bucket=unknown or review_required=true; amount_out else amount_abs", int(g.loc[unknown_mask, "n_tx"].sum()), coverage, unknown_amount, review_amount, "Requires accounting review before decision-grade reporting.", "not_frontend_ready")
        add_row(base, "classification_coverage", "Classification coverage", "data_quality", coverage, "classified_amount_abs / eligible_amount_abs", int(g["n_tx"].sum()), coverage, unknown_amount, review_amount, "Ratio, not money.", "safe_canonical")
        add_row(base, "debt_movements", "Debt movements", "non_operating_debt", debt, "semantic_bucket=debt_movement; amount_abs", ntx("debt_movement"), coverage, unknown_amount, review_amount, "Excluded from property OPEX and net operating.", "review_before_frontend")
        add_row(base, "internal_transfers", "Internal transfers", "non_operating_transfer", transfers, "semantic_bucket=internal_transfer; amount_abs", ntx("internal_transfer"), coverage, unknown_amount, review_amount, "Excluded from operating revenue, property OPEX, and net operating.", "review_before_frontend")

    statement = pd.DataFrame(rows, columns=OPERATING_STATEMENT_COLUMNS)
    qa = build_monthly_operating_statement_qa(statement)
    return statement, qa


def build_monthly_operating_statement(out_dir: Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    split_path = out_dir / "monthly_flow_semantic_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"monthly_operating_statement requires {split_path}; run semantic classification first and do not fall back to legacy reports"
        )
    split = pd.read_csv(split_path)
    statement, qa = build_monthly_operating_statement_from_split(split)
    statement_path = out_dir / "monthly_operating_statement.csv"
    qa_path = out_dir / "monthly_operating_statement_qa.csv"
    statement.to_csv(statement_path, index=False)
    qa.to_csv(qa_path, index=False)
    return {"monthly_operating_statement": statement_path, "monthly_operating_statement_qa": qa_path}


def build_monthly_operating_statement_qa(statement: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lines = set(statement["statement_line"].astype(str)) if not statement.empty else set()

    def add(check: str, ok: bool, detail: str, severity: str = "error") -> None:
        rows.append({"check": check, "status": "pass" if ok else "fail", "detail": detail, "severity": severity})

    add("monthly_operating_statement_exists", not statement.empty, f"rows={len(statement)}")
    for line in [
        "operating_revenue", "property_opex_true", "net_operating", "funding_contributions",
        "family_draws_or_distributions", "personal_expenses", "dividends", "transfer_to_family_expense", "coverage_after_draws",
        "treasury_fx_conversion_in", "treasury_fx_conversion_out", "treasury_fx_cost", "treasury_fx_net",
    ]:
        add(f"has_{line}", line in lines, line)
    add("classification_coverage_present", "classification_coverage" in lines and statement["classification_coverage_ratio"].notna().all(), "coverage ratio populated")
    add("unknown_amount_present", "unknown_amount" in statement.columns, "unknown_amount column populated", "warning")

    def source_filter(line: str) -> str:
        vals = statement.loc[statement["statement_line"].eq(line), "source_filter"].astype(str).unique().tolist()
        return "; ".join(vals)

    add("no_funding_in_operating_revenue", "funding_contribution" not in source_filter("operating_revenue"), source_filter("operating_revenue"))
    add("no_family_draws_in_property_opex", "family_withdrawal" not in source_filter("property_opex_true"), source_filter("property_opex_true"))
    add("no_personal_expense_in_property_opex", "personal" not in source_filter("property_opex_true").lower(), source_filter("property_opex_true"))
    add("no_dividend_in_property_opex", "dividend" not in source_filter("property_opex_true").lower() and "dividendo" not in source_filter("property_opex_true").lower(), source_filter("property_opex_true"))
    add("no_transfer_gasto_in_property_opex", "transfer" not in source_filter("property_opex_true").lower() and "gasto" not in source_filter("property_opex_true").lower(), source_filter("property_opex_true"))
    add("no_debt_principal_in_property_opex", "debt_movement" not in source_filter("property_opex_true") and "principal" not in source_filter("property_opex_true"), source_filter("property_opex_true"))
    add("fx_not_in_operating_revenue", "treasury_fx" not in source_filter("operating_revenue"), source_filter("operating_revenue"))
    add("fx_not_in_property_opex", "treasury_fx" not in source_filter("property_opex_true") and "fx" not in source_filter("property_opex_true").lower(), source_filter("property_opex_true"))
    add("fx_not_funding", "treasury_fx" not in source_filter("funding_contributions"), source_filter("funding_contributions"))
    add("fx_not_draws", "treasury_fx" not in source_filter("family_draws_or_distributions"), source_filter("family_draws_or_distributions"))
    return pd.DataFrame(rows, columns=OPERATING_STATEMENT_QA_COLUMNS)


def _build_validation_rows(audit: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"check": "classification_audit_exists", "period": "", "Currency": "", "amount": len(audit), "n_tx": len(audit), "warning": ""},
        {"check": "monthly_flow_semantic_split_exists", "period": "", "Currency": "", "amount": len(monthly), "n_tx": int(monthly["n_tx"].sum()) if not monthly.empty else 0, "warning": ""},
    ]
    def add(check: str, mask):
        g = audit.loc[mask].groupby(["period", "Currency"], dropna=False).agg(amount=("amount", "sum"), n_tx=("tx_id", "count")).reset_index()
        for _, r in g.iterrows():
            rows.append({"check": check, "period": r["period"], "Currency": r["Currency"], "amount": r["amount"], "n_tx": r["n_tx"], "warning": "review" if r["n_tx"] else ""})
    add("unknown_amount_by_currency", audit["semantic_bucket"].eq("unknown"))
    add("unknown_tx_count", audit["semantic_bucket"].eq("unknown"))
    add("property_opex_amount_by_month", audit["semantic_bucket"].eq("property_opex"))
    add("family_withdrawal_candidate_amount_by_month", audit["semantic_bucket"].eq("family_withdrawal_candidate"))
    add("funding_amount_by_month", audit["semantic_bucket"].eq("funding_contribution"))
    add("rent_amount_by_month", audit["semantic_bucket"].eq("operating_revenue") & audit["semantic_subbucket"].eq("rent"))
    fx_cols = [c for c in ["Flujo", "Tipo", "Detalle", "notes", "payer", "receiver", "cash_path"] if c in audit.columns]
    if fx_cols:
        fx_text_blob = audit[fx_cols].fillna("").astype(str).apply(lambda row: " ".join(row.tolist()), axis=1)
        fx_text = fx_text_blob.str.contains("FX|Cambio", case=False, na=False)
    else:
        fx_text = pd.Series(False, index=audit.index)
    add("fx_rows_not_unknown", fx_text & (audit["semantic_bucket"].astype(str).eq("unknown") | audit["review_required"]))
    add("treasury_fx_amount_by_month", audit["semantic_bucket"].eq("treasury_fx"))
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)
