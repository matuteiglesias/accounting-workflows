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
    repayment_detail = run_root / "monthly_debt_repayment_detail.csv"
    required = [
        annual_metrics,
        annual_contract,
        annual_qa,
        treasury_accountability,
        treasury_qa,
        accountability_cycles,
        repayment_detail,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"report source artifact(s) missing: {missing}")

    _assert_annual_run_identity(annual_metrics, source_run_id)
    as_of_date, annual_period_label = _annual_metadata(annual_metrics)
    treasury_period_label = _treasury_period_label(treasury_accountability)
    generated_at_utc = generated_at_utc or datetime.now(timezone.utc).isoformat()

    out_dir.mkdir(parents=True, exist_ok=True)
    annual_dir = out_dir / "annual_management"
    treasury_dir = out_dir / "treasury_accountability"

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
        out_dir=treasury_dir,
    )

    if require_pdf:
        annual_outputs["pdf"] = render_pdf(
            annual_outputs["html"], annual_dir / "report.pdf", browser_bin=browser_bin
        )
        treasury_outputs["pdf"] = render_pdf(
            treasury_outputs["html"], treasury_dir / "report.pdf", browser_bin=browser_bin
        )

    annual_status = _validation_status(annual_outputs["validation"])
    treasury_status = _validation_status(treasury_outputs["validation"])
    if annual_status == "fail" or treasury_status == "fail":
        raise ValueError(
            "report bundle cannot be cataloged with failed report validation: "
            f"annual={annual_status} treasury={treasury_status}"
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
        ],
        outputs=_manifest_outputs(treasury_outputs, bundle_root=out_dir),
        validation_status=treasury_status,
    )
    treasury_manifest_path = treasury_dir / "report_manifest.json"
    write_report_manifest(treasury_manifest_path, treasury_manifest)

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
