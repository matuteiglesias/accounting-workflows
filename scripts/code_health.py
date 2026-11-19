#!/usr/bin/env python3
"""Read-only code health: existing correctness, advisory diagnostics, static imports."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "accounting"
TOP = 15


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def run(command: list[str], *, cwd: Path = ROOT, timeout: int = 120) -> dict:
    start = perf_counter()
    executable = shutil.which(command[0])
    if executable is None:
        return {"status": "missing", "command": command, "message": f"Tool unavailable: {command[0]}",
                "elapsed_seconds": 0}
    try:
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
        return {"status": "ok" if proc.returncode == 0 else "error", "command": command,
                "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "elapsed_seconds": round(perf_counter() - start, 3)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "command": command, "message": str(exc),
                "elapsed_seconds": round(perf_counter() - start, 3)}


def version(cmd: str, arg: str = "--version") -> str:
    result = run([cmd, arg], timeout=10)
    if result["status"] == "missing":
        return "unavailable"
    if result["status"] != "ok":
        return "error: " + result.get("message", result.get("stderr", "")).strip()[:120]
    return (result.get("stdout", "") or result.get("stderr", "")).strip().splitlines()[0][:120]


def metadata_info() -> dict:
    head = run(["git", "rev-parse", "--short", "HEAD"], timeout=10)
    commit = head.get("stdout", "").strip() if head["status"] == "ok" else "unknown"
    return {"generated_utc": now(), "git_commit": commit, "python": sys.version.split()[0],
            "tools": {"radon": version("radon"), "vulture": version("vulture"),
                      "graphviz": version("dot", "-V"), "pytest": version("pytest") }}


def source_paths(root: Path) -> dict[str, Path]:
    """Only first-party accounting Python; no fixtures, caches, venv, vendor, outputs."""
    paths = {}
    for file in (root / SOURCE).rglob("*.py"):
        if any(part in {"__pycache__", ".venv", "venv", "vendor", "generated"} for part in file.parts):
            continue
        rel = file.relative_to(root)
        bits = list(rel.with_suffix("").parts)
        if bits[-1] == "__init__":
            bits.pop()
        paths[".".join(bits)] = file
    return paths


def package(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 2 else "root"


def internal_imports(root: Path) -> tuple[dict[str, Path], dict[str, set[str]]]:
    modules = source_paths(root)
    edges: dict[str, set[str]] = {name: set() for name in modules}
    for name, file in modules.items():
        syntax = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(syntax):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # __init__.py represents the package itself; other files start at parent package.
                    base = name if file.name == "__init__.py" else name.rsplit(".", 1)[0]
                    parts = base.split(".")
                    offset = node.level - 1
                    if offset >= len(parts):
                        continue
                    prefix = ".".join(parts[:len(parts) - offset])
                    target = prefix + ("." + node.module if node.module else "")
                else:
                    target = node.module or ""
                candidates = [target]
                candidates.extend(target + "." + alias.name for alias in node.names if alias.name != "*")
            for candidate in candidates:
                while candidate.startswith(SOURCE + "."):
                    if candidate in modules and candidate != name:
                        edges[name].add(candidate)
                        break
                    candidate = candidate.rsplit(".", 1)[0]
    return modules, edges


def cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan SCCs on MODULE imports; aggregate package cycles can be spurious."""
    idx = 0
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[list[str]] = []

    def visit(v: str) -> None:
        nonlocal idx
        index[v] = low[v] = idx
        idx += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(edges[v]):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            group = []
            while stack:
                w = stack.pop()
                on_stack.remove(w)
                group.append(w)
                if w == v:
                    break
            if len(group) > 1:
                result.append(sorted(group))

    for module in sorted(edges):
        if module not in index:
            visit(module)
    return sorted(result, key=lambda c: (-len(c), c))


def structural(root: Path) -> dict:
    modules, edges = internal_imports(root)
    sizes = []
    incoming = Counter()
    package_edges = Counter()
    pkg_modules = Counter()
    for name, path in modules.items():
        pkg_modules[package(name)] += 1
        length = len(path.read_text(encoding="utf-8").splitlines())
        sizes.append({"file": str(path.relative_to(root)), "lines": length, "outgoing_imports": len(edges[name])})
        for target in edges[name]:
            incoming[target] += 1
            if package(name) != package(target):
                package_edges[(package(name), package(target))] += 1
    dependencies = {
        "generated_utc": now(), "analysis": "static Python AST imports; not a runtime call graph",
        "module_count": len(modules), "module_import_edges": sum(map(len, edges.values())),
        "package_count": len(pkg_modules),
        "cross_package_import_edges": sum(package_edges.values()),
        "module_cycles": cycles(edges),
        "modules": [{"name": k, "file": str(modules[k].relative_to(root)), "imports": sorted(edges[k]),
                     "incoming_count": incoming[k]} for k in sorted(modules)],
        "package_sizes": dict(sorted(pkg_modules.items())),
        "package_imports": [
            {"from": a, "to": b, "count": count} for (a, b), count in sorted(package_edges.items())
        ],
        "largest_files": sorted(sizes, key=lambda d: (-d["lines"], d["file"]))[:TOP],
        "highest_fanout": sorted(
            [{"module": name, "imports": len(refs)} for name, refs in edges.items()],
            key=lambda d: (-d["imports"], d["module"])
        )[:TOP],
        "highest_fanin": [{"module": m, "imported_by": n} for m, n in incoming.most_common(TOP)],
    }
    return dependencies


def inspect(root: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {"metadata": metadata_info(), "tools": {}, "analysis": {}, "summary": {}}
    commands = {
        "complexity": ["radon", "cc", "-j", SOURCE],
        "maintainability": ["radon", "mi", "-j", SOURCE],
        "unused_candidates": ["vulture", SOURCE, "--min-confidence", "80"],
    }
    for name, command in commands.items():
        info = run(command, cwd=root)
        stdout = info.pop("stdout", "")
        stderr = info.pop("stderr", "")
        if name in {"complexity", "maintainability"} and info["status"] == "ok":
            try:
                parsed = json.loads(stdout)
                save(out / (name + ".json"), parsed)
                info["output"] = name + ".json"
                info["items"] = len(parsed)
            except ValueError as exc:
                info.update(status="error", message=f"Malformed Radon JSON: {exc}")
        elif name == "unused_candidates" and info["status"] != "missing":
            # Vulture emits absolute paths; normalize for comparable reports
            # across workstations and avoid revealing local home directories.
            stdout = stdout.replace(str(root.resolve()) + "/", "")
            (out / "unused_candidates.txt").write_text(stdout, encoding="utf-8")
            info["output"] = "unused_candidates.txt"
            # Vulture 2.3 exits 1 for candidates; some versions use 3.
            # Recognize only well-formed candidate lines, never arbitrary errors.
            findings = [line for line in stdout.splitlines() if line.strip()]
            pattern = re.compile(r"^.+:\d+: .+ \(\d+% confidence\)$")
            if info.get("exit_code") in (1, 3) and findings and all(pattern.match(line) for line in findings):
                info["status"] = "advisory"
            elif info.get("exit_code") != 0:
                info["status"] = "error"
            info["items"] = len(stdout.splitlines())
        if info["status"] == "error":
            info["message"] = info.get("message", stderr.strip()[:1200] or stdout[:1200])
        results["tools"][name] = info
        print(f"{name}: {info['status']} ({info.get('items', '?')} entries; {info['elapsed_seconds']}s)", flush=True)
    try:
        graph = structural(root)
        save(out / "imports.json", graph)
        results["analysis"]["imports"] = "imports.json"
        results["summary"] = {key: graph[key] for key in (
            "module_count", "module_import_edges", "package_count",
            "cross_package_import_edges", "module_cycles", "largest_files",
            "highest_fanout", "highest_fanin"
        )}
    except (SyntaxError, UnicodeError, OSError) as exc:
        results["analysis"]["imports_error"] = str(exc)
        print(f"static imports: error: {exc}", file=sys.stderr)
    save(out / "inspection.json", results)
    write_report(out)
    return 0  # Advisory tools never hide a failed invocation in the report.


def dot_graph(graph: dict, *, min_weight: int = 2) -> str:
    sizes = graph["package_sizes"]
    lines = [
        "digraph accounting_imports {",
        '  graph [rankdir=LR, bgcolor="#ffffff", fontname="Helvetica", pad="0.3", splines=polyline, nodesep="0.32", ranksep="1.0"];',
        '  node [shape=box, style="rounded,filled", fillcolor="#eef3f9", color="#607997", fontname="Helvetica", fontsize=11];',
        '  edge [color="#8293a8", arrowsize="0.6", fontname="Helvetica", fontsize=9];',
        '  label="Accounting package imports — static AST, not calls\\nOnly links with >=2 module-level import relationships shown";',
        '  labelloc=t; fontsize=14;',
    ]
    for pkg, count in sorted(sizes.items()):
        lines.append(f'  "{pkg}" [label="{pkg}\\n{count} modules"];')
    for entry in graph["package_imports"]:
        if entry["count"] >= min_weight:
            src, dest, weight = entry["from"], entry["to"], entry["count"]
            lines.append(f'  "{src}" -> "{dest}" [label="{weight}", penwidth={min(4, 1 + weight / 6):.2f}];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def graph_cmd(root: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {"metadata": metadata_info(), "status": "error"}
    try:
        graph = structural(root)
        save(out / "imports.json", graph)
        dot_file = out / "architecture_packages.dot"
        dot_file.write_text(dot_graph(graph), encoding="utf-8")
        status = run(["dot", "-Tsvg", str(dot_file), "-o", str(out / "architecture_packages.svg")], cwd=root)
        result.update(status=status["status"], module_cycles=graph["module_cycles"],
                      package_count=graph["package_count"], module_count=graph["module_count"],
                      full_graph="imports.json", dot=dot_file.name,
                      svg="architecture_packages.svg" if status["status"] == "ok" else None,
                      detail="Graph shows static AST imports aggregated to packages; links with weight<2 hidden in visualization.")
        if status["status"] != "ok":
            result["error"] = status.get("message", status.get("stderr", "")).strip()
    except (SyntaxError, UnicodeError, OSError) as exc:
        result["error"] = str(exc)
    save(out / "graph.json", result)
    print(f"graph: {result['status']}; {result.get('svg') or result.get('error', 'no SVG')}", flush=True)
    write_report(out)
    return 0 if result["status"] == "ok" else 2


def check(root: Path, out: Path, python: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    command = ["make", "--no-print-directory", "validate", f"PY={python}"]
    result: dict = {"metadata": metadata_info(), "command": command, "status": "error"}
    log_path = out / "check.log"
    start = perf_counter()
    print(f"check: executing existing make validate (log: {log_path})", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = proc.wait()
        result.update(exit_code=returncode, status="pass" if returncode == 0 else "fail",
                      elapsed_seconds=round(perf_counter() - start, 3), log=log_path.name)
    except OSError as exc:
        result.update(error=str(exc), elapsed_seconds=round(perf_counter() - start, 3))
    save(out / "checks.json", result)
    write_report(out)
    return result.get("exit_code", 2)


def top_complexity(data: dict) -> list[dict]:
    selected = []
    def collect(file: str, block: dict, prefix: str = "") -> None:
        kind = block.get("type")
        if kind in {"function", "method"}:
            selected.append({"file": file, "name": prefix + block["name"],
                             "complexity": block.get("complexity", 0),
                             "rank": block.get("rank", "?"), "line": block.get("lineno", 0)})
        for sub in block.get("methods", []):
            collect(file, sub, prefix + block.get("name", "") + ".")
        for sub in block.get("closures", []):
            collect(file, sub, prefix + block.get("name", "") + ".")
    for file, blocks in data.items():
        if isinstance(blocks, list):
            for block in blocks:
                collect(file, block)
    # Radon class nodes contain methods that may also appear as standalone
    # entries in a file's JSON list. Keep one record per source location.
    unique = {}
    for row in selected:
        key = (row["file"], row["line"])
        if key not in unique or len(row["name"]) > len(unique[key]["name"]):
            unique[key] = row
    return sorted(unique.values(), key=lambda row: (-row["complexity"], row["file"]))[:TOP]


def top_maintainability(data: dict) -> list[dict]:
    return sorted(({"file": f, "mi": round(float(v["mi"]), 2), "rank": v["rank"]}
                   for f, v in data.items() if isinstance(v, dict) and "mi" in v),
                  key=lambda row: (row["mi"], row["file"]))[:TOP]


def fmt_status(value: str) -> str:
    return {"pass": "PASS", "ok": "OK", "fail": "FAIL", "error": "ERROR",
            "missing": "MISSING", "advisory": "ADVISORY"}.get(value, value.upper())


def write_report(out: Path) -> None:
    check_data = load(out / "checks.json")
    inspect_data = load(out / "inspection.json")
    graph_data = load(out / "graph.json")
    info = metadata_info()
    lines = [
        "# Codebase health & architecture inspection",
        "",
        f"- Generated (UTC): {now()}",
        f"- Git commit: {info['git_commit']}",
        f"- Python: {info['python']}",
        "- Scope: first-party `accounting/**/*.py`; excludes tests, generated outputs, caches, vendor, venv.",
        "- Diagnostic only: no accounting computations, publication, or source modification.",
        "",
        "## 1. Correctness gate",
        "",
    ]
    if check_data:
        state = fmt_status(check_data["status"])
        lines.append(f"- **{state}**: existing `make validate` (`make check`).")
        lines.append(f"- Exit code: {check_data.get('exit_code', 'unavailable')}; duration: {check_data.get('elapsed_seconds', '?')} s; [log](check.log).")
        checked_commit = check_data.get("metadata", {}).get("git_commit", "unknown")
        if checked_commit != info["git_commit"]:
            lines.append(f"- **STALE CHECK:** latest recorded check is for commit `{checked_commit}`, current HEAD is `{info['git_commit']}`. Re-run `make check` before asserting correctness of this revision.")
    else:
        lines.append("- **NOT RUN**: `make check` has not been executed in this output directory. No correctness claim.")
    lines += ["", "## 2. Tool availability and analysis outcomes", ""]
    lines.append("| Tool / analysis | Version | Outcome | Detail |")
    lines.append("| --- | --- | --- | --- |")
    for name, vkey in [("complexity", "radon"), ("maintainability", "radon"), ("unused_candidates", "vulture")]:
        item = inspect_data.get("tools", {}).get(name)
        result = fmt_status(item["status"]) if item else "NOT RUN"
        detail = f"{item.get('items', '?')} entries" if item and "items" in item else item.get("message", "") if item else "No output"
        lines.append(f"| {name} | {info['tools'][vkey]} | **{result}** | {detail.replace('|', '/')} |")
    lines.append(f"| dependency visualization | {info['tools']['graphviz']} | **{fmt_status(graph_data['status']) if graph_data else 'NOT RUN'}** | [diagram](architecture_packages.svg) when available |")
    lines += ["", "Tool `error` means the analysis command failed; `missing` means it was unavailable; `advisory` denotes Vulture findings, **not proven dead code**. Mandatory correctness is the separate check gate.", ""]
    cc = load(out / "complexity.json")
    mi = load(out / "maintainability.json")
    lines += ["## 3. Highest cyclomatic complexity (Radon)", ""]
    worst_cc = top_complexity(cc)
    if worst_cc:
        lines += ["| Function | CC | Rank |", "| --- | ---: | --- |"]
        for item in worst_cc:
            lines.append(f"| `{item['file']}:{item['line']} {item['name']}` | {item['complexity']} | {item['rank']} |")
    else:
        lines.append("Unavailable: Radon CC did not produce parseable results.")
    lines += ["", "Complexity measures branch count, not a correctness defect. Inspect high scores together with change frequency and tests.", "",
              "## 4. Lowest maintainability scores (Radon)", ""]
    worst_mi = top_maintainability(mi)
    if worst_mi:
        lines += ["| Module | MI | Rank |", "| --- | ---: | --- |"]
        for item in worst_mi:
            lines.append(f"| `{item['file']}` | {item['mi']} | {item['rank']} |")
    else:
        lines.append("Unavailable: Radon MI did not produce parseable results.")
    lines += ["", "Maintainability index is a heuristic influenced by file size and formatting; do not use it as a gate without calibration.", "",
              "## 5. Potentially unused code (Vulture >=80% confidence)", ""]
    vulture_file = out / "unused_candidates.txt"
    if vulture_file.is_file() and inspect_data.get("tools", {}).get("unused_candidates", {}).get("status") in {"ok", "advisory"}:
        candidates = [s for s in vulture_file.read_text(encoding="utf-8").splitlines() if s.strip()]
        lines.append(f"- Candidates: **{len(candidates)}**. Static candidates only; decorators, CLI entrypoints, reflection, and compatibility may be missed.")
        for item in candidates[:TOP]:
            lines.append(f"  - `{item}`")
        if len(candidates) > TOP:
            lines.append(f"- … {len(candidates)-TOP} additional candidates in [full list](unused_candidates.txt).")
    else:
        lines.append("- Unavailable or not run; see analysis outcome above.")
    lines += ["", "## 6. Static dependency architecture", ""]
    imports = load(out / "imports.json")
    if imports:
        lines.append(f"- Modules: **{imports['module_count']}**; module-level edges: **{imports['module_import_edges']}**; packages: **{imports['package_count']}**; cross-package edges: **{imports['cross_package_import_edges']}**.")
        sc = imports.get("module_cycles", [])
        lines.append(f"- Static **module** import cycles: **{len(sc)}**.")
        for group in sc[:6]:
            lines.append(f"  - `{' -> '.join(group)}`")
        lines.append("- [Package diagram](architecture_packages.svg) (filtered edges >=2) | [Graphviz source](architecture_packages.dot) | [complete machine-readable module imports](imports.json).")
        lines.append("- The diagram aggregates **static import declarations**, not runtime calls or dataflow. Package-level reciprocal edges do not establish a module import cycle.")
        lines += ["", "Highest static fan-out (number of imported first-party modules):", ""]
        for m in imports.get("highest_fanout", [])[:10]:
            lines.append(f"- `{m['module']}`: {m['imports']}")
        lines += ["", "Largest files (physical lines, including comments):", ""]
        for f in imports.get("largest_files", [])[:10]:
            lines.append(f"- `{f['file']}`: {f['lines']}")
    else:
        lines.append("- Unavailable; no valid static import inventory.")
    lines += ["", "## 7. Architecture observations / prioritized review queue", ""]
    priorities = []
    if worst_cc:
        h = worst_cc[0]
        priorities.append(f"**Review hotspot (high confidence in complexity, unknown business risk):** `{h['file']}:{h['line']}` has CC={h['complexity']}. Establish ownership and branch tests before proposing a split.")
    if imports and imports.get("highest_fanout"):
        h = imports["highest_fanout"][0]
        priorities.append(f"**Review coupling (high confidence in static fan-out):** `{h['module']}` imports {h['imports']} first-party modules. Examine whether orchestration and business decisions are mixed; no change implied.")
    if worst_mi:
        h = worst_mi[0]
        priorities.append(f"**Review maintainability (heuristic):** `{h['file']}` MI={h['mi']}; combine with activity and test coverage before prioritizing.")
    if imports and imports.get("module_cycles"):
        priorities.append("**Module cycles (static evidence):** investigate SCCs before introducing cross-module features.")
    if (inspect_data.get("tools", {}).get("unused_candidates", {}).get("status") == "advisory"
            and vulture_file.is_file() and vulture_file.read_text(encoding="utf-8").strip()):
        priorities.append("**Investigate dead-code candidates (low confidence):** check runtime consumers and compatibility obligations; do not delete based on Vulture alone.")
    for number, item in enumerate(priorities, 1):
        lines.append(f"{number}. {item}")
    lines += ["", "## 8. Limitations and interpretation", "",
              "- Static AST misses dynamic imports, monkey patches, reflection and runtime dependencies; it may overstate conditional imports.",
              "- No dynamic call graph, actual execution profile, or dependency-based selective runner is inferred here.",
              "- CI executes `make validate` with its own dependencies; Radon, Vulture and Graphviz are locally optional for advisory commands.",
              "- The complete codebase is not type-checked: no configured type checker was found; none was added.",
              "- Missing/failed analyzers do not imply clean code, and advisory findings do not fail the mandatory gate.",
              "- Generated diagnostics are not tracked by Git. Run `make inspect graph` to refresh evidence before making decisions.",
              "",
              "## 9. Raw artifacts", "",
              "- [Check log](check.log) (only after `make check`); [checks.json](checks.json).",
              "- [complexity.json](complexity.json); [maintainability.json](maintainability.json); [unused_candidates.txt](unused_candidates.txt).",
              "- [inspection.json](inspection.json); [graph.json](graph.json); [imports.json](imports.json).", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "inspect", "graph"))
    parser.add_argument("--output-dir", default=".reports/code-health")
    parser.add_argument("--python", default=sys.executable, help="Python command passed to existing make validate")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if args.action == "check":
        return check(ROOT, out, args.python)
    if args.action == "inspect":
        return inspect(ROOT, out)
    return graph_cmd(ROOT, out)


if __name__ == "__main__":
    raise SystemExit(main())
