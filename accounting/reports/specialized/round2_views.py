from __future__ import annotations

"""Round-2 governed professional views for specialized human reports.

These builders consume already-governed run artifacts. They do not classify
ledger transactions, infer legal responsibility, or reconstruct cash paths.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TOLERANCE = 0.01


@dataclass(frozen=True)
class Round2ViewResult:
    frame: pd.DataFrame
    metric_id: str
    dimension: str
    table_columns: tuple[tuple[str, str], ...]


SOURCE_LOCATIONS = {
    "stakeholder_support": ("run", "monthly_stakeholder_support.csv"),
    "cash_accountability": ("run", "monthly_cash_accountability.csv"),
    "accountability_cycles": ("run", "family_business_accountability_cycles.csv"),
    "debt_position": ("run", "monthly_debt_position.csv"),
    "debt_activity": ("run", "monthly_debt_activity.csv"),
    "repayment_detail": ("run", "monthly_debt_repayment_detail.csv"),
}

VIEW_REQUIREMENTS = {
    "support_by_target_box": ("stakeholder_support",),
    "prior_period_clearing": ("stakeholder_support",),
    "physical_inflows_by_box": ("cash_accountability",),
    "physical_outflows_by_box": ("cash_accountability",),
    "accountability_balance": ("accountability_cycles",),
    "accountability_cycles": ("accountability_cycles",),
    "open_debt_positions": ("debt_position",),
    "debt_activity": ("debt_activity",),
    "repayment_allocations": ("repayment_detail",),
}


def _path_for(source_key: str, run_root: Path, metrics_dir: Path) -> Path:
    root_key, filename = SOURCE_LOCATIONS[source_key]
    return (run_root if root_key == "run" else metrics_dir) / filename


def source_paths_for_view(
    view_key: str,
    run_root: Path,
    metrics_dir: Path,
) -> tuple[tuple[Path, str], ...]:
    paths: list[tuple[Path, str]] = []
    for key in VIEW_REQUIREMENTS[view_key]:
        path = _path_for(key, run_root, metrics_dir)
        prefix = "run" if SOURCE_LOCATIONS[key][0] == "run" else "metrics"
        paths.append((path, f"{prefix}/{path.name}"))
    return tuple(paths)


def _read(source_key: str, run_root: Path, metrics_dir: Path) -> pd.DataFrame:
    return pd.read_csv(_path_for(source_key, run_root, metrics_dir))


def _require(frame: pd.DataFrame, columns: set[str], source_name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} missing required specialized-view columns: {missing}")


def _text(series: pd.Series, fallback: str) -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    return out.mask(out.eq("") | out.eq("nan"), fallback)


def _scope_boxes(scope: str) -> set[str]:
    if scope == "FBPM":
        return {"Family Business", "Property Management"}
    if scope in {"Family Business", "Property Management"}:
        return {scope}
    raise ValueError(f"unsupported Box scope for specialized report: {scope}")


def _annual_rows(
    grouped: pd.DataFrame,
    *,
    metric_id: str,
    dimension: str,
    scope: str,
    source_table: str,
    source_filter: str,
    calculation_rule: str,
) -> pd.DataFrame:
    rows = grouped.copy()
    rows["metric_id"] = metric_id
    rows["scope"] = scope
    rows["period_basis"] = "annual"
    rows["line_id"] = rows.apply(
        lambda row: f"{metric_id}|{row['period']}|{row['Currency']}|{row[dimension]}",
        axis=1,
    )
    rows["source_table"] = source_table
    rows["source_filter"] = source_filter
    rows["calculation_rule"] = calculation_rule
    return rows


def _support_by_target_box(support: pd.DataFrame, scope: str) -> Round2ViewResult:
    _require(
        support,
        {"period", "Currency", "target_box", "recognized_amount"},
        "monthly_stakeholder_support.csv",
    )
    work = support.copy()
    work["target_box"] = _text(work["target_box"], "")
    work = work.loc[work["target_box"].isin(_scope_boxes(scope))].copy()
    work["value"] = pd.to_numeric(work["recognized_amount"], errors="coerce")
    if work["value"].isna().any() or work["value"].lt(-TOLERANCE).any():
        raise ValueError("stakeholder support by target Box contains unavailable or negative recognized amounts")
    work["period"] = work["period"].astype(str).str[:4]
    grouped = (
        work.groupby(["period", "Currency", "target_box"], as_index=False, sort=True)["value"]
        .sum()
    )
    grouped = grouped.loc[grouped["value"].gt(TOLERANCE)].copy()
    out = _annual_rows(
        grouped,
        metric_id="SUPPORT.BY_TARGET_BOX",
        dimension="target_box",
        scope=scope,
        source_table="monthly_stakeholder_support.csv",
        source_filter="target_box in reporting scope; recognized_amount",
        calculation_rule="annual governed stakeholder support summed by target Box; no physical-cash inference",
    )
    return Round2ViewResult(
        out,
        "SUPPORT.BY_TARGET_BOX",
        "target_box",
        (("target_box", "Box objetivo"), ("value", "Apoyo reconocido"), ("Currency", "Moneda")),
    )


def _prior_period_clearing(support: pd.DataFrame, scope: str) -> Round2ViewResult:
    required = {
        "period", "Currency", "target_box", "funding_actor", "recognized_amount",
        "settlement_nature", "obligation_period", "settlement_period",
    }
    _require(support, required, "monthly_stakeholder_support.csv")
    work = support.loc[support["settlement_nature"].astype(str).eq("prior_period_clearing")].copy()
    work["target_box"] = _text(work["target_box"], "")
    work = work.loc[work["target_box"].isin(_scope_boxes(scope))].copy()
    work["funding_actor"] = _text(work["funding_actor"], "Actor no identificado")
    work["obligation_period"] = _text(work["obligation_period"], "Período de obligación no informado")
    work["settlement_period"] = _text(work["settlement_period"], "")
    work["settlement_period"] = work["settlement_period"].mask(
        work["settlement_period"].eq(""), work["period"].astype(str)
    )
    work["period"] = work["settlement_period"].astype(str).str[:4]
    work["value"] = pd.to_numeric(work["recognized_amount"], errors="coerce")
    if work["value"].isna().any() or work["value"].lt(-TOLERANCE).any():
        raise ValueError("prior-period clearing contains unavailable or negative recognized amounts")
    work = work.loc[work["value"].gt(TOLERANCE)].copy()
    if work.empty:
        return Round2ViewResult(
            pd.DataFrame(),
            "SUPPORT.PRIOR_PERIOD_CLEARING",
            "clearing_line",
            (("funding_actor", "Actor"), ("target_box", "Box objetivo"), ("obligation_period", "Período obligación"), ("settlement_period", "Período aplicación"), ("value", "Importe reconocido"), ("Currency", "Moneda")),
        )
    group_cols = [
        "period", "Currency", "funding_actor", "target_box", "obligation_period", "settlement_period"
    ]
    grouped = work.groupby(group_cols, as_index=False, sort=True)["value"].sum()
    grouped["clearing_line"] = (
        grouped["funding_actor"]
        + " → "
        + grouped["target_box"]
        + " · obligación "
        + grouped["obligation_period"]
    )
    grouped["metric_id"] = "SUPPORT.PRIOR_PERIOD_CLEARING"
    grouped["scope"] = scope
    grouped["period_basis"] = "settlement_year"
    grouped["line_id"] = grouped.apply(
        lambda row: "SUPPORT.PRIOR_PERIOD_CLEARING|"
        f"{row['period']}|{row['Currency']}|{row['funding_actor']}|{row['target_box']}|"
        f"{row['obligation_period']}|{row['settlement_period']}",
        axis=1,
    )
    grouped["source_table"] = "monthly_stakeholder_support.csv"
    grouped["source_filter"] = "settlement_nature=prior_period_clearing; target_box in reporting scope"
    grouped["calculation_rule"] = (
        "sum recognized clearing support at governed settlement grain; obligation period remains explicit; no debt extinguishment inferred"
    )
    return Round2ViewResult(
        grouped,
        "SUPPORT.PRIOR_PERIOD_CLEARING",
        "clearing_line",
        (("funding_actor", "Actor"), ("target_box", "Box objetivo"), ("obligation_period", "Período obligación"), ("settlement_period", "Período aplicación"), ("value", "Importe reconocido"), ("Currency", "Moneda")),
    )


def _cash_by_box(cash: pd.DataFrame, scope: str, measure: str) -> Round2ViewResult:
    _require(
        cash,
        {"period", "Box", "Currency", "total_cash_in", "total_cash_out"},
        "monthly_cash_accountability.csv",
    )
    work = cash.copy()
    work["Box"] = _text(work["Box"], "")
    work = work.loc[work["Box"].isin(_scope_boxes(scope))].copy()
    work["total_cash_in"] = pd.to_numeric(work["total_cash_in"], errors="coerce")
    work["total_cash_out"] = pd.to_numeric(work["total_cash_out"], errors="coerce")
    if work[["total_cash_in", "total_cash_out"]].isna().any().any():
        raise ValueError("monthly cash accountability contains unavailable physical cash totals")
    if work[["total_cash_in", "total_cash_out"]].lt(-TOLERANCE).any().any():
        raise ValueError("monthly cash accountability contains negative physical cash magnitudes")
    if "net_cash_flow" in work.columns:
        net = pd.to_numeric(work["net_cash_flow"], errors="coerce")
        gap = work["total_cash_in"] - work["total_cash_out"] - net
        if net.isna().any() or gap.abs().gt(TOLERANCE).any():
            raise ValueError("monthly cash accountability does not reconcile total_cash_in - total_cash_out to net_cash_flow")
    work["period"] = work["period"].astype(str).str[:4]
    grouped = (
        work.groupby(["period", "Currency", "Box"], as_index=False, sort=True)[measure]
        .sum()
        .rename(columns={"Box": "box", measure: "value"})
    )
    grouped = grouped.loc[grouped["value"].gt(TOLERANCE)].copy()
    is_in = measure == "total_cash_in"
    metric_id = "TREASURY.PHYSICAL.IN.BY_BOX" if is_in else "TREASURY.PHYSICAL.OUT.BY_BOX"
    label = "Entradas físicas" if is_in else "Salidas físicas"
    out = _annual_rows(
        grouped,
        metric_id=metric_id,
        dimension="box",
        scope=scope,
        source_table="monthly_cash_accountability.csv",
        source_filter=f"Box in reporting scope; measure={measure}",
        calculation_rule=f"annual sum of governed monthly {measure}; constructive settlements remain non-cash",
    )
    return Round2ViewResult(
        out,
        metric_id,
        "box",
        (("box", "Box"), ("value", label), ("Currency", "Moneda")),
    )


def _prepare_cycles(cycles: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cycle_id", "cycle_start", "cycle_end", "view_type", "as_of_date", "Box", "Currency",
        "opening_accountability_balance", "accountable_receipts", "documented_distributions",
        "supported_uses", "documented_transfers_out", "closing_accountability_balance",
        "validated_cash_status", "accountability_gap_status", "n_months",
    }
    _require(cycles, required, "family_business_accountability_cycles.csv")
    work = cycles.loc[cycles["Box"].astype(str).eq("Family Business")].copy()
    numeric = [
        "opening_accountability_balance", "accountable_receipts", "documented_distributions",
        "supported_uses", "documented_transfers_out", "closing_accountability_balance",
    ]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[numeric].isna().any().any():
        raise ValueError("family-business accountability cycle contains unavailable control amounts")
    for column in ["accountable_receipts", "documented_distributions", "supported_uses", "documented_transfers_out"]:
        if work[column].lt(-TOLERANCE).any():
            raise ValueError(f"accountability cycle contains negative flow magnitude: {column}")
    expected = (
        work["opening_accountability_balance"]
        + work["accountable_receipts"]
        - work["documented_distributions"]
        - work["supported_uses"]
        - work["documented_transfers_out"]
    )
    gap = expected - work["closing_accountability_balance"]
    if gap.abs().gt(TOLERANCE).any():
        bad = work.loc[gap.abs().gt(TOLERANCE), ["cycle_id", "Currency"]].copy()
        bad["gap"] = gap.loc[bad.index]
        raise ValueError(f"accountability cycle equation does not reconcile: {bad.to_dict('records')}")
    work["cycle_start_date"] = pd.to_datetime(work["cycle_start"], errors="coerce")
    if work["cycle_start_date"].isna().any():
        raise ValueError("accountability cycle has invalid cycle_start")
    work["period"] = work["as_of_date"].astype(str).str[:4]
    return work


def _balance_state(value: float, *, initial: bool = False) -> str:
    prefix = "inicial" if initial else "final"
    if value > TOLERANCE:
        return f"Saldo {prefix} a rendir"
    if value < -TOLERANCE:
        return f"Déficit {prefix} de rendición"
    return f"Balance {prefix} conciliado en cero"


def _accountability_balance(cycles: pd.DataFrame, _scope: str) -> Round2ViewResult:
    work = _prepare_cycles(cycles)
    if work.empty:
        return Round2ViewResult(pd.DataFrame(), "ACCOUNTABILITY.BALANCE.COMPONENTS", "concept", ())
    max_start = work.groupby("Currency", dropna=False)["cycle_start_date"].transform("max")
    latest = work.loc[work["cycle_start_date"].eq(max_start)].copy()
    rows: list[dict[str, object]] = []
    for _, row in latest.iterrows():
        components = [
            ("opening", _balance_state(float(row["opening_accountability_balance"]), initial=True), "balance", abs(float(row["opening_accountability_balance"]))),
            ("receipts", "Ingresos sujetos a rendición", "+", float(row["accountable_receipts"])),
            ("distributions", "Distribuciones documentadas", "−", float(row["documented_distributions"])),
            ("uses", "Usos respaldados", "−", float(row["supported_uses"])),
            ("transfers", "Transferencias documentadas", "−", float(row["documented_transfers_out"])),
            ("closing", _balance_state(float(row["closing_accountability_balance"])), "balance", abs(float(row["closing_accountability_balance"]))),
        ]
        for key, concept, direction, value in components:
            rows.append({
                "period": row["period"],
                "Currency": row["Currency"],
                "concept": concept,
                "direction": direction,
                "value": value,
                "cycle_start": row["cycle_start"],
                "cycle_end": row["cycle_end"],
                "view_type": row["view_type"],
                "validated_cash_status": row["validated_cash_status"],
                "accountability_gap_status": row["accountability_gap_status"],
                "metric_id": "ACCOUNTABILITY.BALANCE.COMPONENTS",
                "scope": "Family Business",
                "period_basis": "accountability_cycle",
                "line_id": f"ACCOUNTABILITY.BALANCE.COMPONENTS|{row['cycle_id']}|{row['Currency']}|{key}",
                "source_table": "family_business_accountability_cycles.csv",
                "source_filter": f"cycle_id={row['cycle_id']}; Box=Family Business",
                "calculation_rule": (
                    "present governed cycle components; opening + receipts - distributions - supported uses - transfers = closing; balance sign is encoded in the concept label"
                ),
            })
    out = pd.DataFrame(rows)
    return Round2ViewResult(
        out,
        "ACCOUNTABILITY.BALANCE.COMPONENTS",
        "concept",
        (("concept", "Componente"), ("direction", "Efecto"), ("value", "Importe"), ("Currency", "Moneda"), ("cycle_start", "Inicio ciclo"), ("cycle_end", "Fin ciclo"), ("validated_cash_status", "Caja validada")),
    )


def _accountability_cycles(cycles: pd.DataFrame, _scope: str) -> Round2ViewResult:
    work = _prepare_cycles(cycles)
    rows: list[dict[str, object]] = []
    for _, row in work.iterrows():
        closing = float(row["closing_accountability_balance"])
        state = _balance_state(closing)
        cycle_label = f"{str(row['cycle_start'])[:7]} → {str(row['cycle_end'])[:7]} · {state}"
        rows.append({
            "period": row["period"],
            "Currency": row["Currency"],
            "cycle": cycle_label,
            "value": abs(closing),
            "view_type": row["view_type"],
            "n_months": row["n_months"],
            "validated_cash_status": row["validated_cash_status"],
            "metric_id": "ACCOUNTABILITY.CYCLE.CLOSING",
            "scope": "Family Business",
            "period_basis": "accountability_cycle",
            "line_id": f"ACCOUNTABILITY.CYCLE.CLOSING|{row['cycle_id']}|{row['Currency']}",
            "source_table": "family_business_accountability_cycles.csv",
            "source_filter": "Box=Family Business; six-month governed cycle authority",
            "calculation_rule": "one governed closing accountability balance per Mar–Aug / Sep–Feb cycle; magnitude shown with balance state in label",
        })
    out = pd.DataFrame(rows)
    return Round2ViewResult(
        out,
        "ACCOUNTABILITY.CYCLE.CLOSING",
        "cycle",
        (("cycle", "Ciclo"), ("value", "Magnitud saldo al cierre"), ("Currency", "Moneda"), ("view_type", "Estado ciclo"), ("n_months", "Meses"), ("validated_cash_status", "Caja validada")),
    )


def _open_debt_positions(position: pd.DataFrame, scope: str) -> Round2ViewResult:
    required = {
        "period", "as_of_date", "debtor", "creditor", "Currency", "component",
        "position_status", "open_amount", "open_principal", "open_interest", "n_open_items",
    }
    _require(position, required, "monthly_debt_position.csv")
    work = position.copy()
    valid_period = work["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
    work = work.loc[valid_period].copy()
    if work.empty:
        return Round2ViewResult(pd.DataFrame(), "DEBT.OPEN.POSITION", "relation", ())
    latest_period = work["period"].astype(str).max()
    latest = work.loc[
        work["period"].astype(str).eq(latest_period)
        & work["component"].astype(str).eq("total")
    ].copy()
    if latest.empty:
        return Round2ViewResult(pd.DataFrame(), "DEBT.OPEN.POSITION", "relation", ())
    unavailable = latest.loc[~latest["position_status"].astype(str).eq("available")]
    if not unavailable.empty:
        raise ValueError(
            "latest debt position contains unavailable governed relation(s); open-position report will not backfill: "
            f"{unavailable[['debtor', 'creditor', 'Currency', 'position_status']].to_dict('records')}"
        )
    latest["value"] = pd.to_numeric(latest["open_amount"], errors="coerce")
    latest["open_principal"] = pd.to_numeric(latest["open_principal"], errors="coerce")
    latest["open_interest"] = pd.to_numeric(latest["open_interest"], errors="coerce")
    if latest[["value", "open_principal", "open_interest"]].isna().any().any():
        raise ValueError("latest debt position contains unavailable amounts")
    duplicate = latest.duplicated(["debtor", "creditor", "Currency"], keep=False)
    if duplicate.any():
        raise ValueError("latest debt total position is not singular by debtor/creditor/currency")
    latest = latest.loc[latest["value"].gt(TOLERANCE)].copy()
    latest["relation"] = _text(latest["debtor"], "?") + " → " + _text(latest["creditor"], "?")
    latest["period"] = latest_period[:4]
    latest["metric_id"] = "DEBT.OPEN.POSITION"
    latest["scope"] = scope
    latest["period_basis"] = "closing_stock"
    latest["line_id"] = latest.apply(
        lambda row: f"DEBT.OPEN.POSITION|{latest_period}|{row['Currency']}|{row['debtor']}|{row['creditor']}",
        axis=1,
    )
    latest["source_table"] = "monthly_debt_position.csv"
    latest["source_filter"] = f"period={latest_period}; component=total; position_status=available; open_amount>0"
    latest["calculation_rule"] = "consume latest governed closing debt stock; closed zero relations excluded; no monthly stock summation"
    return Round2ViewResult(
        latest,
        "DEBT.OPEN.POSITION",
        "relation",
        (("relation", "Relación registrada"), ("value", "Saldo abierto"), ("Currency", "Moneda"), ("open_principal", "Principal abierto"), ("open_interest", "Interés abierto"), ("n_open_items", "Partidas abiertas"), ("as_of_date", "Fecha de posición")),
    )


def _assert_debt_activity_reconciles(activity: pd.DataFrame) -> None:
    needed = {
        "period", "debtor", "creditor", "Currency", "opening_total", "closing_total",
        "new_principal", "interest_accrued", "repayments", "adjustments", "reconciliation_status",
    }
    _require(activity, needed, "monthly_debt_activity.csv")
    keys = ["period", "debtor", "creditor", "Currency"]
    numeric = ["new_principal", "interest_accrued", "repayments", "adjustments"]
    work = activity.copy()
    for column in ["opening_total", "closing_total", *numeric]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    grouped = work.groupby(keys, dropna=False, sort=True)
    flows = grouped[numeric].sum(min_count=1)
    opening = grouped["opening_total"].first()
    closing = grouped["closing_total"].first()
    statuses = grouped["reconciliation_status"].agg(lambda values: set(values.astype(str)))
    expected = opening + flows["new_principal"] + flows["interest_accrued"] - flows["repayments"] + flows["adjustments"]
    gap = expected - closing
    for key in flows.index:
        if "unavailable_position" in statuses.loc[key]:
            continue
        if pd.isna(opening.loc[key]) or pd.isna(closing.loc[key]) or pd.isna(gap.loc[key]):
            raise ValueError(f"reconcilable debt activity has unavailable opening/closing position: {key}")
        if abs(float(gap.loc[key])) > TOLERANCE:
            raise ValueError(f"debt activity does not reconcile to position for {key}: gap={float(gap.loc[key])}")


def _debt_activity(activity: pd.DataFrame, scope: str) -> Round2ViewResult:
    required = {
        "period", "Currency", "debtor", "creditor", "activity_type",
        "new_principal", "interest_accrued", "repayments", "adjustments",
        "reconciliation_status", "n_items",
    }
    _require(activity, required, "monthly_debt_activity.csv")
    _assert_debt_activity_reconciles(activity)
    specs = {
        "new_claim": ("new_principal", "Nueva obligación / principal"),
        "interest_accrual": ("interest_accrued", "Interés devengado"),
        "repayment": ("repayments", "Repago aplicado"),
        "adjustment": ("adjustments", "Ajuste"),
    }
    rows: list[dict[str, object]] = []
    for _, row in activity.loc[activity["activity_type"].astype(str).isin(specs)].iterrows():
        measure, label = specs[str(row["activity_type"])]
        amount = pd.to_numeric(pd.Series([row[measure]]), errors="coerce").iloc[0]
        if pd.isna(amount):
            raise ValueError(f"debt activity measure unavailable: {measure}")
        amount = float(amount)
        if abs(amount) <= TOLERANCE:
            continue
        if str(row["activity_type"]) == "adjustment":
            label = "Ajuste que aumenta deuda" if amount > 0 else "Ajuste que reduce deuda"
        relation = f"{str(row['debtor']).strip()} → {str(row['creditor']).strip()}"
        rows.append({
            "period": str(row["period"])[:4],
            "Currency": row["Currency"],
            "debtor": row["debtor"],
            "creditor": row["creditor"],
            "relation": relation,
            "activity_label": label,
            "activity_line": f"{relation} · {label}",
            "value": abs(amount),
            "n_items": int(pd.to_numeric(pd.Series([row["n_items"]]), errors="coerce").fillna(0).iloc[0]),
            "reconciliation_status": row["reconciliation_status"],
        })
    if not rows:
        return Round2ViewResult(pd.DataFrame(), "DEBT.ACTIVITY", "activity_line", ())
    work = pd.DataFrame(rows)
    grouped = (
        work.groupby(
            ["period", "Currency", "debtor", "creditor", "relation", "activity_label", "activity_line"],
            as_index=False,
            sort=True,
        )
        .agg(
            value=("value", "sum"),
            n_items=("n_items", "sum"),
            reconciliation_status=("reconciliation_status", lambda values: ";".join(sorted(set(values.astype(str))))),
        )
    )
    grouped["metric_id"] = "DEBT.ACTIVITY"
    grouped["scope"] = scope
    grouped["period_basis"] = "annual_activity_flow"
    grouped["line_id"] = grouped.apply(
        lambda row: f"DEBT.ACTIVITY|{row['period']}|{row['Currency']}|{row['debtor']}|{row['creditor']}|{row['activity_label']}",
        axis=1,
    )
    grouped["source_table"] = "monthly_debt_activity.csv"
    grouped["source_filter"] = "activity_type in new_claim,interest_accrual,repayment,adjustment; nonzero governed flow"
    grouped["calculation_rule"] = (
        "sum governed debt activity flows by relation and type; opening/closing stocks are excluded from annual aggregation; adjustment direction remains explicit"
    )
    return Round2ViewResult(
        grouped,
        "DEBT.ACTIVITY",
        "activity_line",
        (("relation", "Relación"), ("activity_label", "Actividad"), ("value", "Magnitud"), ("Currency", "Moneda"), ("n_items", "Partidas"), ("reconciliation_status", "Reconciliación")),
    )


def _repayment_allocations(detail: pd.DataFrame, scope: str) -> Round2ViewResult:
    required = {
        "period", "repayment_tx_id", "repayment_date", "debtor", "creditor", "Currency",
        "repayment_amount", "allocated_amount", "leftover_amount", "allocation_status",
        "target_debt_id", "target_item_type", "target_opened_at", "target_detail",
    }
    _require(detail, required, "monthly_debt_repayment_detail.csv")
    work = detail.copy()
    for column in ["repayment_amount", "allocated_amount", "leftover_amount"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[["repayment_amount", "allocated_amount", "leftover_amount"]].isna().any().any():
        raise ValueError("debt repayment allocation detail contains unavailable amounts")
    if work[["repayment_amount", "allocated_amount", "leftover_amount"]].lt(-TOLERANCE).any().any():
        raise ValueError("debt repayment allocation detail contains negative magnitudes")
    for tx_id, event in work.groupby("repayment_tx_id", dropna=False, sort=False):
        repayment_values = event["repayment_amount"].unique()
        leftover_values = event["leftover_amount"].unique()
        if len(repayment_values) != 1 or len(leftover_values) != 1:
            raise ValueError(f"repayment allocation metadata conflicts within event {tx_id}")
        allocated = float(event["allocated_amount"].sum())
        repayment = float(repayment_values[0])
        leftover = float(leftover_values[0])
        if abs(allocated + leftover - repayment) > TOLERANCE:
            raise ValueError(
                f"repayment allocations do not reconcile for {tx_id}: allocated={allocated}; leftover={leftover}; repayment={repayment}"
            )
    work["repayment_period"] = work["period"].astype(str)
    work["period"] = work["period"].astype(str).str[:4]
    work["relation"] = _text(work["debtor"], "?") + " → " + _text(work["creditor"], "?")
    target_type = _text(work["target_item_type"], "")
    target_date = _text(work["target_opened_at"], "")
    target_detail = _text(work["target_detail"], "")
    fallback = (target_type + " " + target_date).str.strip()
    target_label = target_detail.mask(target_detail.eq(""), fallback)
    target_label = target_label.mask(target_label.eq(""), "Sin asignación gobernada")
    work["allocation_target"] = target_label
    work["allocation_line"] = work["relation"] + " · " + work["allocation_target"]
    work["value"] = work["allocated_amount"].astype(float)
    work["metric_id"] = "DEBT.REPAYMENT.ALLOCATION"
    work["scope"] = scope
    work["period_basis"] = "repayment_allocation"
    work["line_id"] = work.apply(
        lambda row: "DEBT.REPAYMENT.ALLOCATION|"
        f"{row['repayment_tx_id']}|{row['Currency']}|{row['target_debt_id']}|{row['target_item_type']}|{row['target_opened_at']}",
        axis=1,
    )
    work["source_table"] = "monthly_debt_repayment_detail.csv"
    work["source_filter"] = "allocation grain; one row per governed repayment-to-obligation allocation"
    work["calculation_rule"] = (
        "display allocated_amount only; repayment_amount is not repeated into report totals; allocated + leftover reconciles to each repayment event"
    )
    return Round2ViewResult(
        work,
        "DEBT.REPAYMENT.ALLOCATION",
        "allocation_line",
        (("repayment_date", "Fecha repago"), ("relation", "Relación"), ("allocation_status", "Estado"), ("target_item_type", "Tipo obligación"), ("target_opened_at", "Apertura obligación"), ("allocation_target", "Detalle destino"), ("value", "Importe asignado"), ("Currency", "Moneda"), ("leftover_amount", "Remanente del repago")),
    )


def build_view(
    view_key: str,
    *,
    run_root: Path,
    metrics_dir: Path,
    scope: str,
) -> Round2ViewResult:
    frames = {
        key: _read(key, run_root, metrics_dir)
        for key in VIEW_REQUIREMENTS[view_key]
    }
    if view_key == "support_by_target_box":
        return _support_by_target_box(frames["stakeholder_support"], scope)
    if view_key == "prior_period_clearing":
        return _prior_period_clearing(frames["stakeholder_support"], scope)
    if view_key == "physical_inflows_by_box":
        return _cash_by_box(frames["cash_accountability"], scope, "total_cash_in")
    if view_key == "physical_outflows_by_box":
        return _cash_by_box(frames["cash_accountability"], scope, "total_cash_out")
    if view_key == "accountability_balance":
        return _accountability_balance(frames["accountability_cycles"], scope)
    if view_key == "accountability_cycles":
        return _accountability_cycles(frames["accountability_cycles"], scope)
    if view_key == "open_debt_positions":
        return _open_debt_positions(frames["debt_position"], scope)
    if view_key == "debt_activity":
        return _debt_activity(frames["debt_activity"], scope)
    if view_key == "repayment_allocations":
        return _repayment_allocations(frames["repayment_detail"], scope)
    raise KeyError(f"unknown round-2 specialized view: {view_key}")
