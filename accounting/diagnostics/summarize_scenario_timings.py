from __future__ import annotations

"""Render timing checkpoints from the opt-in scenario --audit-file JSON.

No artifacts are produced unless --html is given. The source JSON stays
outside the scenario HTML/PDF outbox by contract.
"""

import argparse
from html import escape
import json
from pathlib import Path


def module_label(args: list[str]) -> str:
    if "-m" in args:
        idx = args.index("-m")
        if idx + 1 < len(args):
            return args[idx + 1]
    if "-c" in args:
        return "python -c (Make scope/import)"
    return "python " + " ".join(args[:2])


def summarize(audit: dict) -> tuple[list[dict], list[dict]]:
    stages: list[dict] = []
    commands: list[dict] = []
    for variant in audit["variants"]:
        name = variant["variant"]
        for record in variant["events"]:
            kind = record["kind"]
            if kind == "python_subprocess":
                commands.append({
                    "variant": name, "operation": module_label(record["command"]),
                    "seconds": float(record["elapsed_seconds"]),
                    "exit": record.get("exit_code", 0),
                })
            else:
                stages.append({
                    "variant": name, "operation": record.get("stage", kind),
                    "seconds": float(record["elapsed_seconds"]),
                    "exit": record.get("exit_code", 0),
                })
    return stages, commands


def render_html(audit: dict) -> str:
    stages, commands = summarize(audit)
    def table(rows: list[dict], *, title: str) -> str:
        maximum = max((row["seconds"] for row in rows), default=1.0)
        trs = []
        for row in sorted(rows, key=lambda x: (x["variant"], -x["seconds"])):
            width = 100 * row["seconds"] / maximum
            trs.append(
                "<tr><td>" + escape(row["variant"]) + "</td><td>" +
                escape(row["operation"]) + "</td><td class='number'>" +
                f'{row["seconds"]:.3f}' + "</td><td><div class='bar' style='width:" +
                f"{width:.2f}%" + "'></div></td><td>" + str(row["exit"]) + "</td></tr>"
            )
        return "<h2>" + title + "</h2><table><thead><tr><th>Variant</th><th>Stage</th><th>Seconds</th><th>Relative duration</th><th>Exit</th></tr></thead><tbody>" + "".join(trs) + "</tbody></table>"
    return (
        "<!DOCTYPE html><html lang='en'><meta charset='utf-8'><title>Accounting scenario timing</title>"
        "<style>body{font:15px system-ui;margin:2rem auto;max-width:1150px;color:#163046;background:#f7fafb}"
        "h1{color:#123e4f}table{border-collapse:collapse;width:100%;background:white}"
        "th,td{padding:.65rem;border-bottom:1px solid #dce5e9;text-align:left}"
        "th{background:#e9f1f3}.number{text-align:right;font-variant-numeric:tabular-nums}"
        ".bar{height:12px;background:#268e9d;border-radius:2px}</style>"
        "<h1>Accounting scenario — performance audit</h1><p><b>Scenario:</b> " +
        escape(audit["scenario_id"]) + " · <b>Total:</b> " +
        f'{audit["total_seconds"]:.2f}' + " seconds</p>"
        "<p>Make stages and Python subprocesses are different nesting levels; do not add both totals.</p>" +
        table(stages, title="Stage checkpoints") +
        table(commands, title="Python subprocesses") +
        "</html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize opt-in scenario timing JSON")
    parser.add_argument("audit", type=Path)
    parser.add_argument("--html", type=Path, default=None)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("schema") != "accounting.scenario.perf.v1":
        raise ValueError("unsupported timing schema")
    stages, commands = summarize(audit)
    print("Scenario", audit["scenario_id"], "overall seconds", audit["total_seconds"])
    for variant in audit["variants"]:
        print(" ", variant["variant"], "elapsed seconds", variant["total_seconds"])
        for stage in sorted(
            [s for s in stages if s["variant"] == variant["variant"]],
            key=lambda item: -item["seconds"],
        ):
            print("    %-24s %8.2fs" % (stage["operation"], stage["seconds"]))
    print("Largest Python commands:")
    for item in sorted(commands, key=lambda item: -item["seconds"])[:15]:
        print("  %-18s %-45s %8.2fs" % (
            item["variant"], item["operation"], item["seconds"]
        ))
    if args.html:
        if args.html.exists():
            raise FileExistsError(args.html)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_html(audit), encoding="utf-8")
        print("HTML audit", args.html)


if __name__ == "__main__":
    main()
