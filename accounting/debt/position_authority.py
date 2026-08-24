from __future__ import annotations

"""Single governed selector for debt-position snapshots.

Debt position is a stock. Selection is therefore temporal, not additive:
monthly cells use the latest valid ``as_of_date`` in the requested period;
annual cells use the latest period in the year and then the latest valid
``as_of_date`` inside that closing period. If the closing scope contains no
valid as-of observation the result is unavailable. Older periods and lexical
invalid-date ordering are never substitutes.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class DebtPositionSelection:
    available: bool
    selected_period: str
    selected_as_of_date: str
    reason: str
    candidate_rows: int
    valid_as_of_rows: int
    selected_positions: tuple[int, ...]


def _period_mask(frame: pd.DataFrame, period: str, *, annual: bool) -> pd.Series:
    values = frame.get("period", pd.Series("", index=frame.index)).fillna("").astype(str)
    token = str(period).strip()
    return values.str.startswith(token[:4]) if annual else values.eq(token)


def _latest_period(frame: pd.DataFrame) -> str:
    if frame.empty or "period" not in frame.columns:
        return ""
    periods = sorted(
        value
        for value in frame["period"].fillna("").astype(str).unique().tolist()
        if value
    )
    return periods[-1] if periods else ""


def select_debt_position(
    rows: pd.DataFrame,
    *,
    period: str = "",
    annual: bool = False,
    as_of_field: str = "as_of_date",
) -> DebtPositionSelection:
    """Select one governed debt-position snapshot from an already scoped frame.

    ``rows`` should already be scoped to Currency/debtor/creditor/component when
    those dimensions are relevant. The function deliberately does not infer or
    collapse those dimensions.
    """

    if rows is None or rows.empty:
        return DebtPositionSelection(
            False, "", "", "no debt-position candidates", 0, 0, ()
        )
    if "period" not in rows.columns:
        return DebtPositionSelection(
            False,
            "",
            "",
            "missing period column",
            int(len(rows)),
            0,
            (),
        )
    if as_of_field not in rows.columns:
        return DebtPositionSelection(
            False,
            "",
            "",
            f"missing {as_of_field} column",
            int(len(rows)),
            0,
            (),
        )

    scope = rows.copy()
    if period:
        scope = scope.loc[_period_mask(scope, period, annual=annual)].copy()
    if scope.empty:
        return DebtPositionSelection(
            False,
            "",
            "",
            "no debt-position candidates in requested period scope",
            0,
            0,
            (),
        )

    selected_period = _latest_period(scope) if annual else (
        str(period).strip() or _latest_period(scope)
    )
    if not selected_period:
        return DebtPositionSelection(
            False,
            "",
            "",
            "no usable debt-position period",
            int(len(scope)),
            0,
            (),
        )

    closing = scope.loc[
        scope["period"].fillna("").astype(str).eq(selected_period)
    ].copy()
    candidate_rows = int(len(closing))
    parsed = pd.to_datetime(closing[as_of_field], errors="coerce")
    valid_mask = parsed.notna()
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        reason = (
            "latest debt-position period has no valid as_of_date; prior periods are not substituted"
            if annual
            else "no valid as_of_date in selected monthly debt-position candidates"
        )
        return DebtPositionSelection(
            False,
            selected_period,
            "",
            reason,
            candidate_rows,
            0,
            (),
        )

    latest = parsed.loc[valid_mask].max()
    selected_positions = tuple(
        int(position)
        for position in range(len(closing))
        if bool(valid_mask.iloc[position]) and parsed.iloc[position] == latest
    )
    selected_as_of = pd.Timestamp(latest).date().isoformat()
    return DebtPositionSelection(
        True,
        selected_period,
        selected_as_of,
        "latest valid as_of_date selected",
        candidate_rows,
        valid_count,
        selected_positions,
    )


def selected_debt_position_rows(
    rows: pd.DataFrame,
    selection: DebtPositionSelection,
) -> pd.DataFrame:
    """Materialize selected rows without re-running selection semantics."""

    if rows is None or rows.empty or not selection.available:
        return rows.iloc[0:0].copy() if rows is not None else pd.DataFrame()
    closing = rows.loc[
        rows["period"].fillna("").astype(str).eq(selection.selected_period)
    ].copy()
    if not selection.selected_positions:
        return closing.iloc[0:0].copy()
    return closing.iloc[list(selection.selected_positions)].copy()
