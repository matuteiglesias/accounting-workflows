from __future__ import annotations

"""Reusable, self-contained SVG time-series charts for finished accounting reports.

Inputs are already-governed point series. This module renders; it never reads a
ledger, imputes missing values, aggregates money, or infers an entitlement.
"""
from dataclasses import dataclass
from html import escape
import math
import re

import pandas as pd


HISTORY_START_MONTH = "2022-01"

PALETTE = ("#21456f", "#4a81ac", "#78a7c4", "#d09b63", "#7e7999")
GRID = "#e3e9ee"
INK = "#203a58"
MUTED = "#62758a"


@dataclass(frozen=True)
class TimeSeriesSpec:
    chart_id: str
    title: str
    frequency: str  # M or Y only
    mark: str  # stacked_bars, bars, line or dots
    unit: str
    scope: str
    series_ids: tuple[str, ...]
    series_labels: tuple[str, ...]
    subtitle: str = ""
    note: str = ""
    max_points: int = 60
    period_domain: tuple[str, ...] = ()  # visible missing periods, never zero-filled

    def __post_init__(self) -> None:
        if self.frequency not in ("M", "Y"):
            raise ValueError("time series frequency must be M or Y")
        if self.mark not in ("stacked_bars", "bars", "line", "dots"):
            raise ValueError("unknown time series mark")
        if not self.chart_id or not re.fullmatch(r"[a-z0-9_]+", self.chart_id):
            raise ValueError("unsafe time series chart ID")
        if not self.unit or not self.scope:
            raise ValueError("time series unit and scope are required")
        if not (1 <= len(self.series_ids) <= 5) or len(self.series_ids) != len(self.series_labels):
            raise ValueError("time series must have 1-5 named series with labels")
        if len(set(self.series_ids)) != len(self.series_ids):
            raise ValueError("duplicate time series identities")


def _compact(value: float, unit: str) -> str:
    scale, suffix = (1_000_000, " M") if abs(value) >= 1_000_000 else (1_000, " k") if abs(value) >= 1_000 else (1, "")
    short = value / scale
    number = f"{short:,.1f}" if abs(short - round(short)) > .04 else f"{short:,.0f}"
    number = number.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{number}{suffix}"


def _period_valid(period: str, frequency: str) -> bool:
    if frequency == "Y":
        return re.fullmatch(r"(19|20)\d{2}", period) is not None
    if re.fullmatch(r"(19|20)\d{2}-(0[1-9]|1[0-2])", period) is None:
        return False
    return True


def _prepare(spec: TimeSeriesSpec, points: pd.DataFrame) -> pd.DataFrame:
    fields = {"period", "series_id", "value", "unit", "scope", "phase", "source_table"}
    absent = fields - set(points.columns)
    if absent:
        raise ValueError(f"time series missing fields: {sorted(absent)}")
    if points.empty:
        raise ValueError("time series cannot be empty")
    rows = points[list(fields)].copy()
    if not rows["period"].map(lambda p: _period_valid(str(p), spec.frequency)).all():
        raise ValueError("invalid annual/monthly period")
    if not rows["series_id"].isin(spec.series_ids).all():
        raise ValueError("unexpected series id")
    if rows[["period", "series_id"]].duplicated().any():
        raise ValueError("duplicate time series point at period x series")
    if set(rows["unit"].astype(str)) != {spec.unit} or set(rows["scope"].astype(str)) != {spec.scope}:
        raise ValueError("time series mixes unit or scope")
    if not rows["phase"].isin(("observed", "current", "projected")).all():
        raise ValueError("time series phase must be observed, current or projected")
    if rows["source_table"].isna().any() or rows["source_table"].astype(str).str.strip().eq("").any():
        raise ValueError("missing governed source table")
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    if not rows["value"].map(lambda x: math.isfinite(x) if pd.notna(x) else False).all():
        raise ValueError("time series has invalid/nonfinite value; missing is not zero")
    if spec.mark in ("stacked_bars", "bars") and rows["value"].lt(0).any():
        raise ValueError("bar series must be nonnegative; use a line for signed cash flow")
    present_periods = sorted(rows["period"].astype(str).unique())
    periods = list(spec.period_domain) if spec.period_domain else present_periods
    if periods != sorted(set(periods)) or any(not _period_valid(str(p), spec.frequency) for p in periods):
        raise ValueError("period domain must be unique, sorted and of correct frequency")
    if not set(present_periods).issubset(periods):
        raise ValueError("chart values appear outside the declared period domain")
    if len(periods) > spec.max_points:
        raise ValueError("too many periods for a single panel")
    if len(periods) < 2:
        raise ValueError("time series requires at least two periods")
    # A single period is either historical or projected, not a mix of both.
    if rows.groupby("period")["phase"].nunique().gt(1).any():
        raise ValueError("observed and projected values mixed within one period")
    if spec.mark == "stacked_bars":
        expected = set(spec.series_ids)
        if any(set(group["series_id"]) != expected for _, group in rows.groupby("period")):
            raise ValueError("stacked bars require all component series every period")
    return rows.sort_values(["period", "series_id"], kind="stable").reset_index(drop=True)


def render_timeseries_svg(
    spec: TimeSeriesSpec, points: pd.DataFrame, *, width: int = 940, height: int = 278,
) -> tuple[str, pd.DataFrame]:
    """Return inline SVG and exact chart-point trace (no external image dependencies)."""
    if width < 400 or height < 190:
        raise ValueError("chart dimensions too small for legible labels")
    rows = _prepare(spec, points)
    periods = list(spec.period_domain) if spec.period_domain else sorted(rows["period"].astype(str).unique())
    count = len(periods)
    xmin, xmax = 92.0, float(width - 34)
    ymin, ymax = 28.0, float(height - 47)
    plot_h = ymax - ymin
    lookup = {
        (str(r.period), str(r.series_id)): (float(r.value), str(r.phase))
        for r in rows.itertuples(index=False)
    }
    vals = [float(v) for v in rows["value"]]
    if spec.mark == "stacked_bars":
        ylim = max(sum(lookup[(p, s)][0] for s in spec.series_ids) for p in periods if all((p, s) in lookup for s in spec.series_ids))
        bottom = 0.0
    else:
        ylim = max(vals + [0.0])
        bottom = min(vals + [0.0])
    span = max(ylim - bottom, abs(ylim) * 0.1, 1.0)
    top = ylim + span * 0.12
    if bottom < 0:
        bottom -= span * 0.12
    else:
        bottom = 0.0

    def y(value: float) -> float:
        return ymin + (top - value) / (top - bottom) * plot_h

    def x(index: int) -> float:
        return xmin + (xmax - xmin) * (index + 0.5) / count

    svg = [
        f'<svg class="governed-timeseries" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(spec.title, quote=True)}">'
    ]
    # Four gently marked horizontal references.
    for tick in range(5):
        value = bottom + (top - bottom) * tick / 4
        yy = y(value)
        svg.append(
            f'<line x1="{xmin:.1f}" x2="{xmax:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        label = f"{value * 100:.0f}%" if spec.unit == "%" else _compact(value, spec.unit)
        svg.append(
            f'<text x="{xmin - 11:.1f}" y="{yy + 4:.1f}" text-anchor="end" '
            f'fill="{MUTED}" font-size="12">{escape(label)}</text>'
        )
    for index, period in enumerate(periods):
        visible = count <= 13 or index == 0 or index == count - 1 or (
            spec.frequency == "M" and period.endswith("-01")
        ) or (spec.frequency == "Y" and index % 2 == 0)
        if visible:
            label = period if spec.frequency == "Y" else period[2:4] + "/" + period[5:7]
            svg.append(
                f'<text x="{x(index):.1f}" y="{height - 25}" fill="{MUTED}" '
                f'text-anchor="middle" font-size="12">{escape(label)}</text>'
            )
    # Legend is intentionally inside the width of the panel.
    xlegend = xmin
    for index, label in enumerate(spec.series_labels):
        color = PALETTE[index]
        svg.append(
            f'<rect x="{xlegend:.1f}" y="{height - 13}" width="11" height="5" '
            f'fill="{color}" rx="2"/>'
            f'<text x="{xlegend + 16:.1f}" y="{height - 7}" '
            f'fill="{INK}" font-size="12">{escape(label)}</text>'
        )
        xlegend += 28 + 7.1 * min(len(label), 33)
    if spec.mark in ("stacked_bars", "bars"):
        group_space = (xmax - xmin) / count
        bar_width = min(64.0, group_space * (.58 if spec.mark == "stacked_bars" else .69))
        for i, period in enumerate(periods):
            current = 0.0
            for index, series_id in enumerate(spec.series_ids):
                point = lookup.get((period, series_id))
                if point is None:
                    continue
                value, phase = point
                start = current if spec.mark == "stacked_bars" else 0.0
                current += value
                bar_y = y(current) if spec.mark == "stacked_bars" else y(value)
                base_y = y(start)
                slot_width = bar_width if spec.mark == "stacked_bars" else bar_width / len(spec.series_ids)
                xx = x(i) - bar_width / 2 + (index * slot_width if spec.mark == "bars" else 0)
                svg.append(
                    f'<rect x="{xx:.2f}" y="{min(base_y, bar_y):.2f}" '
                    f'width="{slot_width:.2f}" height="{max(abs(base_y - bar_y), .6):.2f}" '
                    f'fill="{PALETTE[index]}" opacity="{.42 if phase == "projected" else .72 if phase == "current" else .94}">'
                    f'<title>{escape(series_id)} · {escape(period)} · {_compact(value, spec.unit)}</title></rect>'
                )
    else:
        for index, series_id in enumerate(spec.series_ids):
            segments: list[tuple[str, float, float, str]] = []
            for i, period in enumerate(periods):
                if (period, series_id) in lookup:
                    value, phase = lookup[(period, series_id)]
                    segments.append((period, x(i), y(value), phase))
                else:
                    # Missing values break the path; never interpolate them.
                    if len(segments) > 1 and spec.mark == "line":
                        _append_line(svg, segments, PALETTE[index])
                    segments = []
            if len(segments) > 1 and spec.mark == "line":
                _append_line(svg, segments, PALETTE[index])
            for i, period in enumerate(periods):
                if (period, series_id) not in lookup:
                    continue
                value, phase = lookup[(period, series_id)]
                svg.append(
                    f'<circle cx="{x(i):.2f}" cy="{y(value):.2f}" r="{4.3 if spec.mark == "dots" else 3.7}" '
                    f'fill="{PALETTE[index]}" stroke="white" stroke-width="1.4" '
                    f'opacity="{.52 if phase == "projected" else .72 if phase == "current" else 1.0}">'
                    f'<title>{escape(series_id)} · {escape(period)} · {_compact(value, spec.unit)}</title></circle>'
                )
    svg.append("</svg>")
    trace = rows.copy()
    trace.insert(0, "chart_id", spec.chart_id)
    trace["source_metric"] = trace["series_id"]
    trace["frequency"] = spec.frequency
    return "".join(svg), trace


def _append_line(svg: list[str], segments: list[tuple[str, float, float, str]], color: str) -> None:
    for i in range(1, len(segments)):
        _, x0, y0, _ = segments[i - 1]
        _, x1, y1, phase = segments[i]
        dash = ' stroke-dasharray="5 5"' if phase in ("projected", "current") else ""
        svg.append(
            f'<path d="M{x0:.2f},{y0:.2f} L{x1:.2f},{y1:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round"{dash}/>'
        )


def timeseries_figure(spec: TimeSeriesSpec, svg: str) -> str:
    """Reusable HTML wrapper; report CSS sizes SVG and keeps PDF panels intact."""
    return (
        f'<figure class="ts-figure" data-chart-id="{escape(spec.chart_id, quote=True)}">'
        f'<figcaption><strong>{escape(spec.title)}</strong>'
        f'<span>{escape(spec.subtitle)}</span></figcaption>'
        + svg
        + (f'<p class="ts-note">{escape(spec.note)}</p>' if spec.note else "")
        + "</figure>"
    )


TIMESERIES_CSS = """
.ts-figure{margin:0 0 10px;padding:10px 12px 7px;border:1px solid #d7dde7;
 border-radius:7px;background:white;break-inside:avoid;page-break-inside:avoid}
.ts-figure figcaption{display:flex;flex-direction:column;gap:2px;margin-bottom:2px}
.ts-figure figcaption strong{font-size:12px;font-weight:760;color:#102a56}
.ts-figure figcaption span,.ts-figure .ts-note{font-size:9px;color:#667085;
 line-height:1.35}
.ts-figure .governed-timeseries{display:block;width:100%;height:auto;
 max-height:235px}
.ts-figure .ts-note{margin:3px 0 0}
@media print{.ts-figure{break-inside:avoid;page-break-inside:avoid}
 .ts-figure .governed-timeseries{max-height:190px}}
"""
