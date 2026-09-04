from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from accounting.reports.annual_management.render import render_report as render_annual
from accounting.reports.catalog import (
    ReportCatalogItem,
    build_report_catalog,
    validate_catalog_files,
    write_report_catalog,
)
from accounting.reports.manifest import (
    ReportOutput,
    ReportSource,
    build_report_manifest,
    write_report_manifest,
)
from accounting.reports.pdf import render_pdf
from accounting.reports.debt_accountability.render import render_report as render_debt
from accounting.reports.treasury_accountability.render import (
    render_report as render_treasury,
)


def _csv_rows(path: Path) -> int:
    return int(len(pd.read_csv(path)))


def _source(path: Path, logical_path: str) -> ReportSource:
    return ReportSource.from_file(path, rows=_csv_rows(path), logical_path=logical_path)


def _validation_status(path: Path) -> str:
    frame = pd.read_csv(path)
    statuses = set(frame.get("status", pd.Series(dtype=str)).astype(str).str.lower())
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _assert_annual_run_identity(metrics_path: Path, expected_run_id: str) -> None:
    frame = pd.read_csv(metrics_path, usecols=lambda col: col == "run_id")
    if "run_id" not in frame.columns:
        raise ValueError("annual report source is missing run_id provenance")
    run_ids = {
        str(value).strip()
        for value in frame["run_id"].dropna()
        if str(value).strip()
    }
    if run_ids != {expected_run_id}:
        raise ValueError(
            "annual report source run_id does not match exact run directory: "
            f"expected={expected_run_id} actual={sorted(run_ids)}"
        )


def _annual_metadata(metrics_path: Path) -> tuple[str, str]:
    frame = pd.read_csv(metrics_path, usecols=lambda col: col in {"period", "as_of_date"})
    periods = sorted(
        {
            str(value).strip().removesuffix(".0")
            for value in frame.get("period", pd.Series(dtype=str)).dropna()
            if str(value).strip().removesuffix(".0").isdigit()
        }
    )
    if not periods:
        raise ValueError("annual report source has no year periods")
    dates = sorted(
        value
        for value in frame.get("as_of_date", pd.Series(dtype=str)).dropna().astype(str).str.strip()
        if value
    )
    as_of_date = dates[-1] if dates else ""
    first, last = periods[0], periods[-1]
    if as_of_date == f"{last}-06-30":
        last_label = f"{last} H1"
    elif as_of_date.startswith(last):
        last_label = f"{last} YTD"
    else:
        last_label = last
    return as_of_date, f"{first}–{last_label}"


def _treasury_period_label(accountability_path: Path) -> str:
    periods = pd.read_csv(accountability_path, usecols=["period"])["period"].dropna().astype(str)
    if periods.empty:
        raise ValueError("treasury accountability source has no periods")
    return f"{periods.min()} – {periods.max()}"


def _manifest_outputs(
    outputs: dict[str, Path],
    *,
    bundle_root: Path,
) -> dict[str, ReportOutput]:
    return {
        name: ReportOutput.from_file(path, bundle_root=bundle_root)
        for name, path in outputs.items()
    }


def _build_pack_validation(*, run_id: str, scope_tag: str, as_of_date: str, annual_metrics: Path, debt_position: Path, source_paths: list[Path], report_html: list[Path]) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of_date)
    rows: list[dict[str, str]] = []
    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": check, "status": "pass" if ok else "fail", "severity": "error", "detail": detail})
    add("exact_release_identity", scope_tag == "FBPM" and run_id.endswith("_FBPM") and as_of_date == "2026-08-31", f"run_id={run_id}; scope={scope_tag}; cutoff={as_of_date}")
    future_periods, household_box_rows = [], 0
    for path in source_paths:
        frame = pd.read_csv(path)
        for col in ("period", "cycle_start", "Date", "as_of_date"):
            if col not in frame.columns: continue
            dates = pd.to_datetime(frame[col], errors="coerce") if col != "period" else pd.to_datetime(frame[col].astype(str)+"-01", errors="coerce")
            future_periods.extend(f"{path.name}:{value}" for value in frame.loc[dates.gt(cutoff), col].astype(str).unique())
        for box_col in ("Box", "target_box", "obligation_box"):
            if box_col in frame.columns:
                household_box_rows += int(frame[box_col].astype(str).str.strip().eq("Household").sum())
    add("no_period_after_cutoff", not future_periods, f"future={future_periods[:10]}")
    add("no_household_reporting_box_membership", household_box_rows == 0, f"rows={household_box_rows}; Household remains allowed as a participant dimension")
    position = pd.read_csv(debt_position); latest = position["period"].astype(str).max()
    debt_total = pd.to_numeric(position.loc[(position.period.astype(str)==latest)&position.component.eq("total"),"open_amount"],errors="coerce").sum()
    metrics = pd.read_csv(annual_metrics); year=as_of_date[:4]
    metric_period = metrics.period.astype(str).str.removesuffix(".0")
    annual_rows=metrics.loc[(metrics.metric_id.eq("ID.DEBT.TOTAL.OPEN"))&metric_period.eq(year)&(metrics.Currency.eq("USD"))]
    annual_total=pd.to_numeric(annual_rows["value"],errors="coerce").sum()
    add("annual_debt_equals_debt_report", len(annual_rows)==1 and abs(float(annual_total)-float(debt_total))<=0.01, f"annual={annual_total}; debt={debt_total}")
    pm_primos=position.loc[(position.period.astype(str)==latest)&position.debtor.eq("PM")&position.creditor.eq("Primos")&position.component.eq("total"),"open_amount"]
    add("pm_primos_closes_zero", len(pm_primos)==1 and abs(float(pm_primos.iloc[0]))<=0.01, f"rows={len(pm_primos)}; close={pm_primos.tolist()}")
    visible="\n".join(path.read_text(encoding="utf-8") for path in report_html)
    household_page = bool(__import__("re").search(
        r"<h[12][^>]*>[^<]*HOUSEHOLD\s*(?:·|<|$)", visible,
        flags=__import__("re").I,
    ))
    add("no_household_reporting_page", not household_page, "Household may appear only as an actor within an in-scope Box table")
    add("no_visible_raw_debt_hash", not bool(__import__("re").search(r">\s*(?:prestamo|interes)::[0-9a-f]+\s*<", visible, flags=__import__("re").I)), "visible text nodes checked")
    return pd.DataFrame(rows)


def build_report_bundle(
    *,
    run_root: Path,
    metrics_dir: Path,
    out_dir: Path,
    scope_tag: str,
    browser_bin: str | Path | None = None,
    require_pdf: bool = True,
    generated_at_utc: str | None = None,
) -> dict[str, Path]:
    run_root = Path(run_root).resolve(strict=True)
    metrics_dir = Path(metrics_dir).resolve(strict=True)
    out_dir = Path(out_dir)
    source_run_id = run_root.name
    if metrics_dir.name != source_run_id:
        raise ValueError(
            "report inputs mix accounting runs: "
            f"run_root={source_run_id} metrics_dir={metrics_dir.name}"
        )

    annual_metrics = metrics_dir / "annual_balance_dashboard_metrics.csv"
    annual_contract = metrics_dir / "annual_balance_dashboard_contract.csv"
    annual_qa = metrics_dir / "annual_balance_dashboard_qa.csv"
    treasury_accountability = run_root / "monthly_cash_accountability.csv"
    treasury_qa = run_root / "monthly_cash_accountability_qa.csv"
    accountability_cycles = run_root / "family_business_accountability_cycles.csv"
    stakeholder_support = run_root / "monthly_stakeholder_support.csv"
    stakeholder_support_qa = run_root / "monthly_stakeholder_support_qa.csv"
    semantic_audit = run_root / "classification_audit.csv"
    repayment_detail = run_root / "monthly_debt_repayment_detail.csv"
    debt_position = run_root / "monthly_debt_position.csv"
    debt_position_qa = run_root / "monthly_debt_position_qa.csv"
    debt_activity = run_root / "monthly_debt_activity.csv"
    debt_activity_qa = run_root / "monthly_debt_activity_qa.csv"
    cost_gaps = run_root / "cost_allocation_gaps.csv"
    cost_gaps_qa = run_root / "cost_allocation_gaps_qa.csv"
    required = [
        annual_metrics,
        annual_contract,
        annual_qa,
        treasury_accountability,
        treasury_qa,
        accountability_cycles,
        stakeholder_support,
        stakeholder_support_qa,
        repayment_detail,
        debt_position, debt_position_qa, debt_activity, debt_activity_qa,
        cost_gaps, cost_gaps_qa,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"report source artifact(s) missing: {missing}")

    _assert_annual_run_identity(annual_metrics, source_run_id)
    _annual_as_of_date, annual_period_label = _annual_metadata(annual_metrics)
    debt_dates = pd.read_csv(debt_position, usecols=["period", "as_of_date"])
    latest_debt_period = debt_dates["period"].astype(str).max()
    latest_debt_dates = sorted(set(debt_dates.loc[debt_dates["period"].astype(str).eq(latest_debt_period), "as_of_date"].dropna().astype(str)))
    if len(latest_debt_dates) != 1:
        raise ValueError(f"debt report cutoff is not singular: {latest_debt_dates}")
    as_of_date = latest_debt_dates[0]
    treasury_period_label = _treasury_period_label(treasury_accountability)
    generated_at_utc = generated_at_utc or datetime.now(timezone.utc).isoformat()

    out_dir.mkdir(parents=True, exist_ok=True)
    annual_dir = out_dir / "annual_management"
    treasury_dir = out_dir / "treasury_accountability"
    debt_dir = out_dir / "debt_accountability"

    annual_outputs = render_annual(
        metrics_path=annual_metrics,
        contract_path=annual_contract,
        qa_path=annual_qa,
        out_dir=annual_dir,
        repayment_detail_path=repayment_detail,
    )
    treasury_outputs = render_treasury(
        accountability_path=treasury_accountability,
        qa_path=treasury_qa,
        cycles_path=accountability_cycles,
        stakeholder_support_path=stakeholder_support,
        semantic_audit_path=semantic_audit,
        annual_metrics_path=annual_metrics,
        out_dir=treasury_dir,
    )
    debt_outputs = render_debt(
        position_path=debt_position,
        position_qa_path=debt_position_qa,
        activity_path=debt_activity,
        activity_qa_path=debt_activity_qa,
        repayment_detail_path=repayment_detail,
        gaps_path=cost_gaps,
        gaps_qa_path=cost_gaps_qa,
        out_dir=debt_dir,
        as_of_date=as_of_date,
    )

    if require_pdf:
        annual_outputs["pdf"] = render_pdf(
            annual_outputs["html"], annual_dir / "report.pdf", browser_bin=browser_bin
        )
        treasury_outputs["pdf"] = render_pdf(
            treasury_outputs["html"], treasury_dir / "report.pdf", browser_bin=browser_bin
        )
        debt_outputs["pdf"] = render_pdf(
            debt_outputs["html"], debt_dir / "report.pdf", browser_bin=browser_bin
        )

    annual_status = _validation_status(annual_outputs["validation"])
    treasury_status = _validation_status(treasury_outputs["validation"])
    debt_status = _validation_status(debt_outputs["validation"])
    if annual_status == "fail" or treasury_status == "fail" or debt_status == "fail":
        raise ValueError(
            "report bundle cannot be cataloged with failed report validation: "
            f"annual={annual_status} treasury={treasury_status} debt={debt_status}"
        )

    annual_manifest = build_report_manifest(
        report_id="annual_management",
        renderer_version="annual_management.v1",
        source_run_id=source_run_id,
        scope_tag=scope_tag,
        as_of_date=as_of_date,
        sources=[
            _source(annual_metrics, "metrics/annual_balance_dashboard_metrics.csv"),
            _source(annual_contract, "metrics/annual_balance_dashboard_contract.csv"),
            _source(annual_qa, "metrics/annual_balance_dashboard_qa.csv"),
            _source(repayment_detail, "run/monthly_debt_repayment_detail.csv"),
        ],
        outputs=_manifest_outputs(annual_outputs, bundle_root=out_dir),
        validation_status=annual_status,
    )
    annual_manifest_path = annual_dir / "report_manifest.json"
    write_report_manifest(annual_manifest_path, annual_manifest)

    treasury_manifest = build_report_manifest(
        report_id="treasury_accountability",
        renderer_version="treasury_accountability.v1",
        source_run_id=source_run_id,
        scope_tag=scope_tag,
        as_of_date=as_of_date,
        sources=[
            _source(treasury_accountability, "run/monthly_cash_accountability.csv"),
            _source(treasury_qa, "run/monthly_cash_accountability_qa.csv"),
            _source(accountability_cycles, "run/family_business_accountability_cycles.csv"),
            _source(stakeholder_support, "run/monthly_stakeholder_support.csv"),
            _source(stakeholder_support_qa, "run/monthly_stakeholder_support_qa.csv"),
            *([_source(semantic_audit, "run/classification_audit.csv")] if semantic_audit.is_file() else []),
        ],
        outputs=_manifest_outputs(treasury_outputs, bundle_root=out_dir),
        validation_status=treasury_status,
    )
    treasury_manifest_path = treasury_dir / "report_manifest.json"
    write_report_manifest(treasury_manifest_path, treasury_manifest)

    debt_manifest = build_report_manifest(
        report_id="debt_accountability",
        renderer_version="debt_accountability.v1",
        source_run_id=source_run_id,
        scope_tag=scope_tag,
        as_of_date=as_of_date,
        sources=[
            _source(debt_position, "run/monthly_debt_position.csv"),
            _source(debt_position_qa, "run/monthly_debt_position_qa.csv"),
            _source(debt_activity, "run/monthly_debt_activity.csv"),
            _source(debt_activity_qa, "run/monthly_debt_activity_qa.csv"),
            _source(repayment_detail, "run/monthly_debt_repayment_detail.csv"),
            _source(cost_gaps, "run/cost_allocation_gaps.csv"),
            _source(cost_gaps_qa, "run/cost_allocation_gaps_qa.csv"),
        ],
        outputs=_manifest_outputs(debt_outputs, bundle_root=out_dir),
        validation_status=debt_status,
    )
    debt_manifest_path = debt_dir / "report_manifest.json"
    write_report_manifest(debt_manifest_path, debt_manifest)

    pack_validation = _build_pack_validation(
        run_id=source_run_id, scope_tag=scope_tag, as_of_date=as_of_date,
        annual_metrics=annual_metrics, debt_position=debt_position,
        source_paths=[annual_metrics, treasury_accountability, accountability_cycles, stakeholder_support, debt_position, debt_activity, repayment_detail, cost_gaps],
        report_html=[annual_outputs["html"], treasury_outputs["html"], debt_outputs["html"]],
    )
    pack_validation_path = out_dir / "report_pack_validation.csv"
    pack_validation.to_csv(pack_validation_path, index=False)
    if pack_validation["status"].eq("fail").any():
        raise ValueError(f"report pack validation failed; inspect {pack_validation_path}")

    # The catalog is the viewer boundary. Internal provenance manifests and
    # trace/validation CSVs remain under out/reports and are deliberately not
    # part of the public document-discovery contract.
    catalog = build_report_catalog(
        source_run_id=source_run_id,
        scope_tag=scope_tag,
        as_of_date=as_of_date,
        generated_at_utc=generated_at_utc,
        reports=[
            ReportCatalogItem(
                report_id="annual_management",
                title="Informe patrimonial y de gestión",
                description=(
                    "Visión anual de rentas, operación, aplicación del resultado, "
                    "tesorería, deuda y control."
                ),
                period_label=annual_period_label,
                sort_order=10,
                html="annual_management/report.html",
                pdf="annual_management/report.pdf" if require_pdf else None,
                manifest=None,
            ),
            ReportCatalogItem(
                report_id="debt_accountability",
                title="Posición y movimientos de deuda",
                description="Obligaciones registradas, actividad, repagos y trazabilidad.",
                period_label=f"{as_of_date[:4]} YTD · cierre {as_of_date}",
                sort_order=30,
                html="debt_accountability/report.html",
                pdf="debt_accountability/report.pdf" if require_pdf else None,
                manifest=None,
            ),
            ReportCatalogItem(
                report_id="treasury_accountability",
                title="Rendición mensual de tesorería",
                description=(
                    "Movimientos mensuales y evolución del control acumulado de "
                    "Family Business y Property Management."
                ),
                period_label=treasury_period_label,
                sort_order=20,
                html="treasury_accountability/report.html",
                pdf="treasury_accountability/report.pdf" if require_pdf else None,
                manifest=None,
            ),
        ],
    )
    catalog_path = out_dir / "report_catalog.json"
    write_report_catalog(catalog_path, catalog)
    validate_catalog_files(catalog, bundle_root=out_dir)

    return {
        "catalog": catalog_path,
        "annual_html": annual_outputs["html"],
        "annual_pdf": annual_outputs.get("pdf", Path()),
        "annual_manifest": annual_manifest_path,
        "treasury_html": treasury_outputs["html"],
        "treasury_pdf": treasury_outputs.get("pdf", Path()),
        "treasury_manifest": treasury_manifest_path,
        "debt_html": debt_outputs["html"],
        "debt_pdf": debt_outputs.get("pdf", Path()),
        "debt_manifest": debt_manifest_path,
        "pack_validation": pack_validation_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build governed human report documents from one exact accounting run."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--scope-tag", required=True)
    parser.add_argument("--browser-bin")
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Render HTML/manifests/catalog without PDF; not suitable for public report publication.",
    )
    args = parser.parse_args()
    outputs = build_report_bundle(
        run_root=args.run_root,
        metrics_dir=args.metrics_dir,
        out_dir=args.out_dir,
        scope_tag=args.scope_tag,
        browser_bin=args.browser_bin,
        require_pdf=not args.html_only,
    )
    for name, path in outputs.items():
        if path:
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
