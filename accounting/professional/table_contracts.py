from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from accounting.contracts.atomic_flow_drilldowns import resolve_flow_cell_spec

FUNDING_CONTRACT_TABLE_IDS = {
    "overview_balance_dashboard",
    "income_operating_statement",
    "cash_annual_box_flow_bridge_wide",
    "cash_annual_box_flow_bridge_long",
}

FUNDING_CONTRACT_EXCLUDED_TABLE_IDS = {
    "monthly_tables_debt_position_matrix",
    "monthly_tables_debt_activity_matrix",
    "monthly_tables_diagnostic_box_level_matrix",
}

ATOMIC_FLOW_METADATA_TABLE_IDS = {
    *FUNDING_CONTRACT_TABLE_IDS,
    "monthly_tables_operating_statement_matrix",
    "monthly_tables_operating_statement_matrix_ars",
    "monthly_tables_draws_by_box_amount_out",
    "monthly_tables_draws_by_type_amount_out",
    "monthly_tables_opex_by_type_amount_out",
    "monthly_tables_fb_bridge_matrix",
    "monthly_tables_pm_stress_matrix",
    "monthly_tables_household_bridge_matrix",
    "annual_funding_by_actor_channel_wide",
}

OVERVIEW_PRESENTATION_METRIC_IDS = {
    "funding / aportes": "FUND.CONTRIB.TOTAL",
    "aportes": "FUND.CONTRIB.TOTAL",
    "retiros / gasto personal": "DIST.DRAWS.PERSONAL",
    "gasto personal": "DIST.DRAWS.PERSONAL",
    "dividendos": "DIST.DIVIDENDS",
    "cobertura después de funding y retiros": "COV.NET.AFTER_DRAWS",
    "cobertura despues de funding y retiros": "COV.NET.AFTER_DRAWS",
    "retiros / resultado operativo": "RATIO.DRAWS_TO_OPERATING_RESULT",
    "deuda total abierta": "ID.DEBT.TOTAL.OPEN",
    "principal abierto": "ID.DEBT.PRINCIPAL.OPEN",
    "interés abierto": "ID.DEBT.INTEREST.OPEN",
    "interes abierto": "ID.DEBT.INTEREST.OPEN",
}

CONTRACT_COLUMNS = [
    "metric_id",
    "line_id",
    "dimension_name",
    "dimension_value",
    "funding_channel",
    "funding_actor",
    "cash_effect",
]
DRILLDOWN_CELL_ID_COLUMN = "drilldown_cell_id"
TABLE_METADATA_FRONT_COLUMNS = [DRILLDOWN_CELL_ID_COLUMN, *CONTRACT_COLUMNS]

EXPLICIT_METRIC_FLOW_CELL_IDS = {
    "IS.REVENUE.OPERATING": "flow.operating_revenue",
    "IS.RENT.TOTAL": "flow.rent.total",
    "IS.RENT.BY_PROPERTY": "flow.rent.by_property",
    "IS.OPEX.PROPERTY": "flow.property_opex.total",
    "IS.OPEX.BY_CATEGORY": "flow.property_opex.by_category",
    "FUND.CONTRIB.TOTAL": "flow.funding_contribution.total",
    "FUND.CONTRIB.BY_FUNDING_ACTOR": "flow.funding_contribution.by_actor",
    "FUND.CONTRIB.BY_CHANNEL": "flow.funding_contribution.by_channel",
    "FUND.CONTRIB.BY_CASH_EFFECT": "flow.funding_contribution.by_cash_effect",
    "FUND.CONTRIB.BY_TARGET_BOX": "flow.funding_contribution.by_target_box",
    "DIST.DRAWS.PERSONAL": "flow.family_draws_or_distributions.total",
    "DIST.DRAWS.BY_TYPE": "flow.draws.by_type",
    "TR.FX.CONVERSION.IN": "flow.fx.conversion_proceeds",
    "TR.FX.CONVERSION.OUT": "flow.fx.conversion_outflow",
    "TR.FX.COST.OUT": "flow.fx.cost_or_spread",
}

EXPLICIT_STATEMENT_LINE_FLOW_CELL_IDS = {
    "operating_revenue": "flow.operating_revenue",
    "rent_revenue": "flow.rent.total",
    "property_opex_true": "flow.property_opex.total",
    "taxes": "flow.property_opex.taxes",
    "services": "flow.property_opex.services",
    "maintenance": "flow.property_opex.maintenance",
    "legal": "flow.property_opex.legal",
    "funding_contributions": "flow.funding_contribution.total",
    "family_draws_or_distributions": "flow.family_draws_or_distributions.total",
    "treasury_fx_conversion_in": "flow.fx.conversion_proceeds",
    "treasury_fx_conversion_out": "flow.fx.conversion_outflow",
    "treasury_fx_cost": "flow.fx.cost_or_spread",
}

EXPLICIT_TABLE_FLOW_CELL_IDS = {
    "monthly_tables_draws_by_box_amount_out": "flow.draws.by_box",
    "monthly_tables_draws_by_type_amount_out": "flow.draws.by_type",
    "monthly_tables_opex_by_type_amount_out": "flow.property_opex.by_box_category",
}

YEAR_RE = re.compile(r"^20\d{2}$")
MONTH_RE = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")
    return token[:120] or "line"


def _row_blob(row: pd.Series) -> str:
    fields = [
        "metric_id", "line_id", "metric", "line", "label", "statement_line", "measure",
        "section", "dashboard_section", "dimension_name", "dimension_value", "funding_channel",
        "funding_actor", "cash_effect",
    ]
    return " | ".join(_text(row.get(c)) for c in fields if c in row.index).casefold()


def _first(row: pd.Series, *cols: str) -> str:
    for col in cols:
        value = _text(row.get(col))
        if value:
            return value
    return ""


def _overview_presentation_metric_id(row: pd.Series) -> str:
    label = _first(row, "line", "metric", "label", "statement_line")
    return OVERVIEW_PRESENTATION_METRIC_IDS.get(label.casefold(), "")


def _is_cash_bridge_net_debt_movement(row: pd.Series) -> bool:
    label = _first(row, "line", "metric", "label", "statement_line").casefold()
    return (
        "movimiento neto de deuda" in label
        or "debt net" in label
        or "net debt" in label
        or ("deuda" in label and "neto" in label)
        or ("debt" in label and "net" in label)
    )


def _validated_flow_cell_id(cell_id: str) -> str:
    value = _text(cell_id)
    if not value:
        return ""
    if resolve_flow_cell_spec(value) is None:
        raise ValueError(f"Unknown atomic-flow drilldown_cell_id: {value!r}")
    return value


for _flow_cell_id in {
    *EXPLICIT_METRIC_FLOW_CELL_IDS.values(),
    *EXPLICIT_STATEMENT_LINE_FLOW_CELL_IDS.values(),
    *EXPLICIT_TABLE_FLOW_CELL_IDS.values(),
}:
    _validated_flow_cell_id(_flow_cell_id)


def _stable_atomic_flow_cell_id(
    table_id: str,
    raw_row: pd.Series,
    final_row: pd.Series,
) -> str:
    """Resolve explicit producer metadata to one governed atomic-flow cell ID.

    Human-label inference is deliberately excluded. A raw ``metric_id`` only
    counts when enrichment leaves that exact ID unchanged, so compatibility
    repair/inference cannot accidentally opt a row into the governed executor.
    """

    explicit_cell_id = _text(raw_row.get(DRILLDOWN_CELL_ID_COLUMN))
    candidates: list[str] = []

    table_cell_id = EXPLICIT_TABLE_FLOW_CELL_IDS.get(table_id, "")
    if table_cell_id:
        candidates.append(table_cell_id)

    explicit_metric_id = _text(raw_row.get("metric_id"))
    final_metric_id = _text(final_row.get("metric_id"))
    if explicit_metric_id and explicit_metric_id == final_metric_id:
        metric_cell_id = EXPLICIT_METRIC_FLOW_CELL_IDS.get(explicit_metric_id, "")
        if metric_cell_id:
            candidates.append(metric_cell_id)

    statement_line = _text(raw_row.get("statement_line"))
    statement_cell_id = EXPLICIT_STATEMENT_LINE_FLOW_CELL_IDS.get(statement_line, "")
    if statement_cell_id:
        candidates.append(statement_cell_id)

    unique_candidates = tuple(dict.fromkeys(candidates))

    if explicit_cell_id:
        explicit_cell_id = _validated_flow_cell_id(explicit_cell_id)
        if unique_candidates and any(
            candidate != explicit_cell_id for candidate in unique_candidates
        ):
            raise ValueError(
                "Explicit drilldown_cell_id conflicts with stable producer metadata: "
                f"table_id={table_id!r}; drilldown_cell_id={explicit_cell_id!r}; "
                f"candidates={unique_candidates!r}"
            )
        return explicit_cell_id

    if len(unique_candidates) == 1:
        return _validated_flow_cell_id(unique_candidates[0])

    # No stable identity, or conflicting structured metadata: leave the row on
    # its existing compatibility path rather than guessing from presentation
    # labels.
    return ""


def _attach_atomic_flow_cell_ids(
    raw: pd.DataFrame,
    enriched: pd.DataFrame,
    table_id: str,
) -> pd.DataFrame:
    if (
        table_id not in ATOMIC_FLOW_METADATA_TABLE_IDS
        and DRILLDOWN_CELL_ID_COLUMN not in raw.columns
    ):
        return enriched

    out = enriched.copy()
    if DRILLDOWN_CELL_ID_COLUMN not in out.columns:
        out[DRILLDOWN_CELL_ID_COLUMN] = ""

    if out.empty:
        return out

    raw_aligned = raw.reindex(out.index)
    out[DRILLDOWN_CELL_ID_COLUMN] = [
        _stable_atomic_flow_cell_id(table_id, raw_aligned.loc[idx], out.loc[idx])
        for idx in out.index
    ]

    nonblank = out[DRILLDOWN_CELL_ID_COLUMN].fillna("").astype(str).str.strip()
    for cell_id in sorted(set(nonblank[nonblank.ne("")])):
        _validated_flow_cell_id(cell_id)

    return out


def _infer_metric_contract(table_id: str, row: pd.Series) -> dict[str, str]:
    existing_metric = _text(row.get("metric_id"))
    existing_dim_name = _text(row.get("dimension_name"))
    existing_dim_value = _text(row.get("dimension_value"))
    existing_channel = _text(row.get("funding_channel"))
    existing_actor = _text(row.get("funding_actor"))
    existing_cash = _text(row.get("cash_effect"))

    blob = _row_blob(row)
    metric = existing_metric
    dim_name = existing_dim_name
    dim_value = existing_dim_value
    channel = existing_channel
    actor = existing_actor
    cash = existing_cash

    presentation_metric = _overview_presentation_metric_id(row) if table_id == "overview_balance_dashboard" else ""
    bridge_net_debt = table_id == "cash_annual_box_flow_bridge_wide" and _is_cash_bridge_net_debt_movement(row)
    if bridge_net_debt:
        metric = ""
        dim_name = ""
        dim_value = ""
        channel = ""
        actor = ""
        cash = ""

    if presentation_metric:
        metric = presentation_metric
        dim_name = ""
        dim_value = ""
        channel = ""
        actor = ""
        cash = ""

    if not metric and not bridge_net_debt:
        if "funding" in blob or "aporte" in blob or "contrib" in blob or _text(row.get("metric")) == "funding_in":
            metric = "FUND.CONTRIB.TOTAL"
        if "inquil" in blob or re.search(r"\binq\b", blob):
            actor = actor or "Inquilino"
            if "impuesto" in blob or "tax" in blob:
                metric = "FUND.CONTRIB.BY_CHANNEL"
                dim_name = dim_name or "funding_channel"
                dim_value = dim_value or "tenant_direct_tax_payment"
                channel = channel or "tenant_direct_tax_payment"
                cash = cash or "no_cash_in_box_direct_payment"
            elif "servicio" in blob or "service" in blob:
                metric = "FUND.CONTRIB.BY_CHANNEL"
                dim_name = dim_name or "funding_channel"
                dim_value = dim_value or "tenant_direct_service_payment"
                channel = channel or "tenant_direct_service_payment"
                cash = cash or "no_cash_in_box_direct_payment"
            elif "caja" in blob or "box" in blob:
                metric = "FUND.CONTRIB.BY_CHANNEL"
                dim_name = dim_name or "funding_channel"
                dim_value = dim_value or "tenant_to_box"
                channel = channel or "tenant_to_box"
                cash = cash or "cash_in_box"
        for name, canonical in [
            ("mat", "Matías"),
            ("alejandro", "Alejandro"),
            ("alen", "Alejandro"),
            ("primos", "Primos"),
            ("héctor", "Héctor"),
            ("hector", "Héctor"),
        ]:
            if name in blob:
                metric = "FUND.CONTRIB.BY_FUNDING_ACTOR"
                dim_name = dim_name or "funding_actor"
                dim_value = dim_value or canonical
                actor = actor or canonical
                break
        if "household" in blob or re.search(r"\bhh\b", blob):
            metric = "FUND.CONTRIB.BY_CHANNEL"
            dim_name = dim_name or "funding_channel"
            dim_value = dim_value or "household_to_pm"
            channel = channel or "household_to_pm"
            actor = actor or "Household"
        if "direct" in blob and ("obligation" in blob or "impuesto" in blob or "tax" in blob or "servicio" in blob or "service" in blob):
            metric = metric or "FUND.CONTRIB.DIRECT_OBLIGATION"
            cash = cash or "no_cash_in_box_direct_payment"
        if "debt" in blob or "deuda" in blob:
            metric = metric or "FUND.CONTRIB.DEBT_LINKED"

    line_seed = _first(row, "line_id", "metric_id", "metric", "line", "label", "statement_line", "measure") or table_id
    line_id = _text(row.get("line_id")) or _safe_token(f"{table_id}:{line_seed}")

    if metric == "FUND.CONTRIB.BY_CHANNEL" and not dim_name and channel:
        dim_name, dim_value = "funding_channel", channel
    if metric == "FUND.CONTRIB.BY_FUNDING_ACTOR" and not dim_name and actor:
        dim_name, dim_value = "funding_actor", actor
    if metric == "FUND.CONTRIB.BY_CASH_EFFECT" and not dim_name and cash:
        dim_name, dim_value = "cash_effect", cash

    return {
        "metric_id": metric,
        "line_id": line_id,
        "dimension_name": dim_name,
        "dimension_value": dim_value,
        "funding_channel": channel,
        "funding_actor": actor,
        "cash_effect": cash,
    }


def enrich_professional_table(df: pd.DataFrame, table_id: str) -> pd.DataFrame:
    raw = df.copy()
    out = df.copy()

    # Funding contract inference is intentionally limited to tables whose
    # presentation rows can contain funding/contribution lines. Matrix tables
    # such as monthly debt position use labels like ``open_total`` and must not
    # receive stale FUND.* metadata from a previous enrichment pass.
    if table_id in FUNDING_CONTRACT_TABLE_IDS:
        for col in CONTRACT_COLUMNS:
            if col not in out.columns:
                out[col] = ""
        if not out.empty:
            inferred = out.apply(
                lambda row: _infer_metric_contract(table_id, row),
                axis=1,
                result_type="expand",
            )
            presentation_metric = (
                out.apply(_overview_presentation_metric_id, axis=1)
                .fillna("")
                .astype(str)
                if table_id == "overview_balance_dashboard"
                else pd.Series("", index=out.index)
            )
            bridge_net_debt = (
                out.apply(_is_cash_bridge_net_debt_movement, axis=1)
                .fillna(False)
                .astype(bool)
                if table_id == "cash_annual_box_flow_bridge_wide"
                else pd.Series(False, index=out.index)
            )
            for col in CONTRACT_COLUMNS:
                current = out[col].fillna("").astype(str).str.strip()
                out[col] = out[col].where(current.ne(""), inferred[col])

            # Curated annual overview presentation labels are authoritative.
            # This repairs older generated packs where a broad funding heuristic
            # may have stamped debt stock rows as FUND.CONTRIB.DEBT_LINKED.
            curated = presentation_metric.ne("")
            if curated.any():
                out.loc[curated, "metric_id"] = presentation_metric.loc[curated]
                for col in [
                    "dimension_name",
                    "dimension_value",
                    "funding_channel",
                    "funding_actor",
                    "cash_effect",
                ]:
                    out.loc[curated, col] = ""

            # Net debt movement is a signed cash-bridge line. Do not let the
            # generic debt/funding heuristic convert it into
            # FUND.CONTRIB.DEBT_LINKED, which uses gross amount_abs semantics.
            if bridge_net_debt.any():
                for col in [
                    "metric_id",
                    "dimension_name",
                    "dimension_value",
                    "funding_channel",
                    "funding_actor",
                    "cash_effect",
                ]:
                    out.loc[bridge_net_debt, col] = ""

    elif table_id in FUNDING_CONTRACT_EXCLUDED_TABLE_IDS:
        for col in [
            "metric_id",
            "dimension_name",
            "dimension_value",
            "funding_channel",
            "funding_actor",
            "cash_effect",
            DRILLDOWN_CELL_ID_COLUMN,
        ]:
            if col in out.columns:
                out[col] = ""
        return out

    metadata_applicable = (
        table_id in FUNDING_CONTRACT_TABLE_IDS
        or table_id in ATOMIC_FLOW_METADATA_TABLE_IDS
        or DRILLDOWN_CELL_ID_COLUMN in raw.columns
    )
    if not metadata_applicable:
        return out

    out = _attach_atomic_flow_cell_ids(raw, out, table_id)

    period_cols = [
        c for c in out.columns if YEAR_RE.match(str(c)) or MONTH_RE.match(str(c))
    ]
    front = [c for c in TABLE_METADATA_FRONT_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in front and c not in period_cols]
    return out[front + rest + period_cols]


def enrich_professional_table_contracts(tables_dir: Path) -> list[Path]:
    tables_dir = Path(tables_dir)
    if not tables_dir.exists():
        return []
    written: list[Path] = []
    for path in sorted(tables_dir.glob("*.csv")):
        df = pd.read_csv(path)
        enriched = enrich_professional_table(df, path.stem)
        if list(enriched.columns) != list(df.columns) or not enriched.equals(df):
            enriched.to_csv(path, index=False)
            written.append(path)
    return written
