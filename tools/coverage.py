"""Test coverage of the C++ core (clang source-based coverage) and the Python package (coverage.py
through pytest-cov), with a report per area and the list of branches the tests never take.

    .venv/bin/python tools/coverage.py all [--golden] [PYTEST ARGS ...]   # build, run, report
    .venv/bin/python tools/coverage.py build        # the instrumented extension, in build/coverage/cmake
    .venv/bin/python tools/coverage.py run [--golden] [PYTEST ARGS ...]   # the tests (default: all of tests/)
    .venv/bin/python tools/coverage.py report       # merge the profiles, write the reports

Everything goes under build/coverage/ (ignored by git); the normal build (build/<wheel tag>) and
CMakeLists.txt are untouched. The instrumented build is the same CMake project configured with
-fprofile-instr-generate -fcoverage-mapping on top of its own flags (Release, -O3).

run: the tests import stardust from src/ with the instrumented _core in place of the normal one
(build/coverage/site/sitecustomize.py does it in every process that has SD_COVERAGE_CORE set, so
subprocesses count too). --golden also runs `tools/golden.py check --any-nan` with the
instrumented core: its code differs, and in a few cases so does the sign of a NaN result
(2026-11-20: 10 cases, a/b with complex NaN, StudentT's inverses); every other bit must agree.

report writes
    build/coverage/summary.md       line, function and branch coverage per area, C++ and Python
    build/coverage/<area>.md        per file, then every branch some direction of which never ran
    build/coverage/html/cpp/        llvm-cov's annotated sources (branch counts)
    build/coverage/html/py/         coverage.py's annotated sources
A C++ branch counts once per source location (template instantiations summed): "true never" and
"false never" name the direction no test took, "never reached" a condition no test evaluated.

The terminal shows a header block, a progress bar per long step (ninja's steps, the tests, the
golden cases), a result line per step and the summary as a table (bench/console.py); the tools'
own output goes to build/coverage/logs/<stamp>.log.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import procrun

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench import console as C  # noqa: E402

OUT = ROOT / "build" / "coverage"
CMAKE_DIR = OUT / "cmake"
SITE = OUT / "site"
PROFRAW = OUT / "profraw"
PYDATA = OUT / "py"
RCFILE = OUT / "coveragerc"
PY = ROOT / ".venv" / "bin" / "python"
JOBS = 8  # at most 8 processes (compile jobs included)
FLAGS = "-fprofile-instr-generate -fcoverage-mapping"
IGNORE = r"(third_party|/Applications/|/opt/homebrew/|/Library/Developer/|nanobind|\.venv)"
# Tests that crash the instrumented core, deselected so one crash does not end the whole run.
# Each is a library bug the normal build survives by chance; remove the entry once it is fixed.
CRASHES = {
    # corrcoef of 0 variables divides by p = 0 (kParallelTask / p, csrc/array/ops/covariance.cpp):
    # 0 on arm64 in the normal build, a bus error in the instrumented one (2026-11-20).
    "tests/array/test_statistics.py::test_cov_rowvar_false_always_transposes_and_no_variables_is_empty",
}

SITECUSTOMIZE = '''"""Written by tools/coverage.py: load the instrumented stardust._core in place of the editable
install's (only when SD_COVERAGE_CORE names it)."""

import os
import sys

_core = os.environ.get("SD_COVERAGE_CORE")
if _core:
    import importlib.util

    # The editable install's finder would load build/<wheel tag>'s _core (and rebuild it);
    # without it, stardust comes from src/ (on sys.path through the install's .pth file).
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
    _spec = importlib.util.spec_from_file_location("stardust._core", _core)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["stardust._core"] = _mod
    _spec.loader.exec_module(_mod)
'''

COVERAGERC = """# Written by tools/coverage.py
[run]
branch = True
source_pkgs = stardust
parallel = True
data_file = {data}
patch = subprocess

[paths]
source =
    {src}
"""


GOLDEN_STEP = re.compile(r"golden: ([\d,]+)/([\d,]+) cases")  # tools/golden.py's plain progress lines


def golden_progress(line):
    m = GOLDEN_STEP.search(line)
    return (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))) if m else None


class Failed(Exception):
    """A step that failed; its output's end has been shown."""


def logged(con, cmd):
    con.log.write(f"{time.strftime('%H:%M:%S')} $ {' '.join(shown_arg(c) for c in cmd)}\n")


def step(con, label, cmd, env=None, stdout=None, progress=None, quiet=False):
    """cmd as one step: its output to the log (a bar from progress(line) when given), then a
    result line; on failure the end of its output and Failed. stdout: a file for the output
    instead (llvm-cov export), the log then gets stderr only. Returns (output lines, seconds)."""
    logged(con, cmd)
    if stdout is None:
        code, lines, secs = procrun.run(cmd, con, label, progress, env)
    else:
        t0 = time.perf_counter()
        r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, stdout=stdout, stderr=subprocess.PIPE, text=True)
        code, lines, secs = r.returncode, r.stderr.splitlines(), time.perf_counter() - t0
        for line in lines:
            con.log.write(f"{time.strftime('%H:%M:%S')} | {line}\n")
    if code:
        con.print(con.color(f"  {label}: failed (exit {code}, {C.fmt_duration(secs)})", "31"))
        con.lines([f"    {line}" for line in lines[-15:]])
        raise Failed(label)
    if not quiet:
        con.print(f"  {label}  ({C.fmt_duration(secs)})")
    return lines, secs


def shown_arg(c):
    c = str(c).replace(str(ROOT) + "/", "")
    return f"'{c}'" if " " in c else c


def xcrun(tool):
    return subprocess.check_output(["xcrun", "-f", tool], text=True).strip()


def core_path(con=None):
    found = sorted(CMAKE_DIR.glob("_core*.so"))
    if not found:
        if con is not None:
            con.print(con.color("  no instrumented _core yet: run tools/coverage.py build first", "31"))
        raise Failed("finding the instrumented core")
    return found[0]


# --------------------------------------------------------------------------------------------
# build


def cmd_build(con):
    CMAKE_DIR.mkdir(parents=True, exist_ok=True)
    if not (CMAKE_DIR / "CMakeCache.txt").exists():
        step(con, "configure", [
            "cmake", "-S", ROOT, "-B", CMAKE_DIR, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={PY}", f"-DCMAKE_CXX_FLAGS={FLAGS}",
            f"-DCMAKE_MODULE_LINKER_FLAGS={FLAGS}", f"-DCMAKE_SHARED_LINKER_FLAGS={FLAGS}",
        ])  # fmt: skip
    lines, secs = step(con, "compile", ["cmake", "--build", CMAKE_DIR, "-j", str(JOBS)],
                       progress=procrun.ninja_progress, quiet=True)  # fmt: skip
    n = max((p[1] for p in map(procrun.ninja_progress, lines) if p), default=0)
    what = f"{n} ninja steps" if n else "up to date"
    con.print(f"  compile: {what}  ({C.fmt_duration(secs)}) -> {core_path(con).relative_to(ROOT)}")


# --------------------------------------------------------------------------------------------
# run


def env_for_run():
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "sitecustomize.py").write_text(SITECUSTOMIZE)
    RCFILE.write_text(COVERAGERC.format(data=PYDATA / ".coverage", src=ROOT / "src" / "stardust"))
    env = dict(os.environ)
    env["SD_COVERAGE_CORE"] = str(core_path())
    env["PYTHONPATH"] = os.pathsep.join([str(SITE)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["LLVM_PROFILE_FILE"] = str(PROFRAW / "core-%p-%m.profraw")
    env["COVERAGE_RCFILE"] = str(RCFILE)
    return env


def cmd_run(con, golden, pytest_args):
    """(exit status, pytest's counts, its FAILED lines, the golden check's result)."""
    core_path(con)
    for d in (PROFRAW, PYDATA):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    env = env_for_run()
    args = pytest_args or ["tests"]
    deselect = [a for t in sorted(CRASHES) for a in ("--deselect", t)]
    cmd = [PY, "-m", "pytest", "-v", "-p", "no:cacheprovider", "--cov", "--cov-branch",
           f"--cov-config={RCFILE}", "--cov-report=", *deselect, *args]  # fmt: skip
    logged(con, cmd)
    code, lines, secs = procrun.run(cmd, con, "tests", procrun.PytestProgress(), dict(env, COLUMNS="100"))
    counts, fails = procrun.pytest_summary(lines)
    parts = [f"{counts['passed']:,} passed", f"{counts['failed']:,} failed", f"{counts['skipped']:,} skipped"]
    parts += [f"{counts[k]:,} {k}" for k in ("errors", "deselected") if counts[k]]
    text = "  tests: " + " · ".join(parts) + f"  ({C.fmt_duration(secs)})"
    con.print(con.color(text, "31") if code not in (0, 5) else text)
    for f in fails:
        con.lines([con.color(x, "31") for x in C.split_label("    " + f, 8, con.width())])
    if code not in (0, 5) and not fails:  # pytest stopped without a summary
        con.lines([f"    {line}" for line in lines[-12:]])
    status = 0 if code in (0, 5) else code
    golden_result = "not run"
    if golden:
        g_cmd = [PY, "-m", "coverage", "run", f"--rcfile={RCFILE}", "tools/golden.py", "check", "--any-nan"]
        logged(con, g_cmd)
        g_code, g_lines, g_secs = procrun.run(g_cmd, con, "golden", golden_progress, env)
        result = next((ln for ln in g_lines if "Result:" in ln), "")
        golden_result = result.split("Result:", 1)[1].strip() if result else f"exit {g_code}"
        text = f"  golden (instrumented core, --any-nan): {golden_result}  ({C.fmt_duration(g_secs)})"
        if g_code:
            con.print(con.color(text, "31"))
            diffs = [ln for ln in g_lines if ln.startswith(("CHANGED", "NEW", "GONE"))]
            con.lines([f"    {ln}" for ln in (diffs or g_lines[-12:])[:20]])
            status = status or g_code
        else:
            con.print(text)
    n = len(list(PROFRAW.glob("*.profraw")))
    con.print(f"  profiles: {n:,} file{'s' if n != 1 else ''} in {PROFRAW.relative_to(ROOT)}")
    return status, counts, fails, golden_result


# --------------------------------------------------------------------------------------------
# report

COMMON = "common"


def cpp_area(rel):
    parts = rel.split("/")
    if parts[0] != "csrc":
        return None
    d = parts[1]
    if d == "bind":
        name = parts[-1]
        for area in ("array", "linalg", "frame", "special", "stats", "regression", "optimize", "random"):
            if name.startswith(area):
                return area
        return COMMON
    if d in ("base", "parallel", "accelerate"):
        return COMMON
    return d


def py_area(rel):
    parts = rel.split("/")
    if len(parts) >= 4 and parts[:2] == ["src", "stardust"]:
        return parts[2]
    if rel == "src/stardust/__init__.py":
        return "array"
    return COMMON


def pct(hit, total):
    return f"{100.0 * hit / total:5.1f}%" if total else "    -"


def cmd_report(con):
    OUT.mkdir(parents=True, exist_ok=True)
    areas = {}  # area -> {"cpp": [file rows], "py": [file rows]}
    raws = sorted(PROFRAW.glob("*.profraw"))
    if raws:
        prof = OUT / "core.profdata"
        step(
            con,
            f"merge {len(raws):,} C++ profile{'s' if len(raws) != 1 else ''}",
            [xcrun("llvm-profdata"), "merge", "-sparse", *raws, "-o", prof],
        )
        cov = xcrun("llvm-cov")
        obj = core_path(con)
        common = ["-instr-profile", prof, f"-ignore-filename-regex={IGNORE}"]
        exported = OUT / "cpp.json"
        with open(exported, "w") as fh:
            step(con, "export C++ coverage", [cov, "export", obj, "-format=text", *common], stdout=fh)
        shutil.rmtree(OUT / "html" / "cpp", ignore_errors=True)
        step(con, "C++ annotated sources (html)", [cov, "show", obj, "-format=html", "-show-branches=count",
             "-output-dir", OUT / "html" / "cpp", f"-Xdemangler={xcrun('c++filt')}", *common],
             stdout=subprocess.DEVNULL)  # fmt: skip
        for row in cpp_rows(json.loads(exported.read_text())):
            areas.setdefault(row["area"], {"cpp": [], "py": []})["cpp"].append(row)
    else:
        con.print(con.color("  no C++ profiles (run first)", "33"))
    data = list(PYDATA.glob(".coverage*"))
    if data:
        env = dict(os.environ, COVERAGE_RCFILE=str(RCFILE))
        if any(p.name != ".coverage" for p in data):
            step(con, f"combine {len(data):,} Python data files",
                 [PY, "-m", "coverage", "combine", "--append", f"--rcfile={RCFILE}"], env=env)  # fmt: skip
        pyjson = OUT / "py.json"
        step(con, "export Python coverage", [PY, "-m", "coverage", "json", f"--rcfile={RCFILE}", "-o", pyjson], env=env)
        shutil.rmtree(OUT / "html" / "py", ignore_errors=True)
        step(con, "Python annotated sources (html)",
             [PY, "-m", "coverage", "html", f"--rcfile={RCFILE}", "-d", OUT / "html" / "py"], env=env)  # fmt: skip
        for row in py_rows(json.loads(pyjson.read_text())):
            areas.setdefault(row["area"], {"cpp": [], "py": []})["py"].append(row)
    else:
        con.print(con.color("  no Python coverage data (run first)", "33"))
    return write_reports(areas) if areas else None


def cpp_rows(doc):
    rows = []
    for f in doc["data"][0]["files"]:
        path = pathlib.Path(f["filename"])
        try:
            rel = str(path.resolve().relative_to(ROOT))
        except ValueError:
            continue
        area = cpp_area(rel)
        if area is None:
            continue
        s = f["summary"]
        # branches: [line, col, end line, end col, true count, false count, file id, expanded id, kind]
        by_loc = {}
        for b in f.get("branches", []):
            key = (b[0], b[1])
            t, fl = by_loc.get(key, (0, 0))
            by_loc[key] = (t + b[4], fl + b[5])
        missed = []
        for (line, col), (t, fl) in sorted(by_loc.items()):
            if t and fl:
                continue
            missed.append((line, col, "never reached" if not (t or fl) else ("true never" if not t else "false never")))
        rows.append({
            "area": area, "file": rel, "lines": (s["lines"]["covered"], s["lines"]["count"]),
            "functions": (s["functions"]["covered"], s["functions"]["count"]),
            "branches": (s["branches"]["covered"], s["branches"]["count"]), "missed": missed,
        })  # fmt: skip
    return rows


def py_rows(doc):
    rows = []
    for name, f in doc["files"].items():
        path = pathlib.Path(name)
        path = path if path.is_absolute() else ROOT / path
        try:
            rel = str(path.resolve().relative_to(ROOT))
        except ValueError:
            continue
        s = f["summary"]
        missed = [(a, b) for a, b in f.get("missing_branches", [])]
        rows.append({
            "area": py_area(rel), "file": rel,
            "lines": (s["covered_lines"], s["num_statements"]),
            "branches": (s.get("covered_branches", 0), s.get("num_branches", 0)),
            "missing_lines": f.get("missing_lines", []), "missed": missed,
        })  # fmt: skip
    return rows


def total(rows, key):
    return sum(r[key][0] for r in rows), sum(r[key][1] for r in rows)


def ranges(nums):
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ", ".join(out)


def source_line(rel, line, cache={}):  # noqa: B006 -- a cache of file lines
    if rel not in cache:
        try:
            cache[rel] = (ROOT / rel).read_text().splitlines()
        except OSError:
            cache[rel] = []
    lines = cache[rel]
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def write_reports(areas):
    """Writes summary.md and <area>.md; returns the summary's rows for the terminal."""
    order = sorted(areas, key=lambda a: (a == COMMON, a))
    lines = [
        "# 커버리지 요약",
        "",
        "`tools/coverage.py report`가 만든다. 영역별 파일과 실행되지 않은 분기는 `<영역>.md`에 있다.",
        "",
    ]
    lines += ["| 영역 | C++ 줄 | C++ 함수 | C++ 분기 | Python 줄 | Python 분기 |", "|---|---|---|---|---|---|"]
    for a in order:
        c, p = areas[a]["cpp"], areas[a]["py"]
        cells = [pct(*total(c, k)) if c else "-" for k in ("lines", "functions", "branches")]
        cells += [pct(*total(p, k)) if p else "-" for k in ("lines", "branches")]
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    allc = [r for a in order for r in areas[a]["cpp"]]
    allp = [r for a in order for r in areas[a]["py"]]
    cells = [pct(*total(allc, k)) for k in ("lines", "functions", "branches")]
    cells += [pct(*total(allp, k)) for k in ("lines", "branches")]
    lines.append("| 전체 | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[6:]]
    rows[-1][0] = "all"
    for a in order:
        out = [f"# 커버리지: {a}", ""]
        c, p = areas[a]["cpp"], areas[a]["py"]
        if c:
            out += ["## C++", "", "| 파일 | 줄 | 함수 | 분기 | 실행 안 된 분기 |", "|---|---|---|---|---|"]
            for r in sorted(c, key=lambda r: r["file"]):
                out.append(f"| {r['file']} | {pct(*r['lines'])} | {pct(*r['functions'])} | {pct(*r['branches'])} | "
                           f"{len(r['missed'])} |")  # fmt: skip
            out.append("")
            for r in sorted(c, key=lambda r: r["file"]):
                if not r["missed"]:
                    continue
                out += [f"### {r['file']}", ""]
                for line, col, what in r["missed"]:
                    out.append(f"- `{r['file']}:{line}:{col}` {what}: `{source_line(r['file'], line)}`")
                out.append("")
        if p:
            out += ["## Python", "", "| 파일 | 줄 | 분기 | 실행 안 된 줄 |", "|---|---|---|---|"]
            for r in sorted(p, key=lambda r: r["file"]):
                out.append(
                    f"| {r['file']} | {pct(*r['lines'])} | {pct(*r['branches'])} | {ranges(r['missing_lines'])} |"
                )
            out.append("")
            for r in sorted(p, key=lambda r: r["file"]):
                if not r["missed"]:
                    continue
                out += [f"### {r['file']}", ""]
                for a_, b_ in r["missed"]:
                    to = "함수 끝" if b_ < 0 else f"{b_}행"
                    out.append(f"- `{r['file']}:{a_}` → {to} 안 감: `{source_line(r['file'], a_)}`")
                out.append("")
        (OUT / f"{a}.md").write_text("\n".join(out) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["all", "build", "run", "report"])
    ap.add_argument("--golden", action="store_true", help="run: also tools/golden.py check, instrumented")
    args, rest = ap.parse_known_args()
    log = procrun.log_path("coverage/logs")
    stages = {"all": ["build", "run", "report"]}.get(args.command, [args.command])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    with open(log, "w") as fh:
        con = C.Console(log=fh, stream=sys.stdout)
        wide = [("Tests", " ".join(rest) or "tests (every test)")] if "run" in stages else []
        wide += [("Output", str(OUT.relative_to(ROOT))), ("Log", str(log.relative_to(ROOT)))]
        con.lines(
            C.block(
                f"stardust coverage {args.command}",
                [
                    ("Steps", " → ".join(stages)),
                    ("Commit", commit.stdout.strip() or "?"),
                    ("Compile jobs", str(JOBS)),
                    ("Golden", "yes" if args.golden and "run" in stages else "no"),
                ],
                wide,
            )  # fmt: skip
        )
        status, ran, rows, t0 = 0, None, None, time.perf_counter()
        try:
            for k, stage in enumerate(stages, 1):
                con.print(f"[{k}/{len(stages)}] {stage}")
                if stage == "build":
                    cmd_build(con)
                elif stage == "run":
                    status, *ran = cmd_run(con, args.golden, rest)
                else:
                    rows = cmd_report(con)
        except Failed as e:
            con.lines(
                ["", con.color(f"Stopped: {e} failed.", "31")] + C.notes([f"The whole output: {log.relative_to(ROOT)}"])
            )
            return 1
        con.print("")
        if ran:
            counts, fails, golden = ran
            con.lines(
                C.block(
                    "Run summary",
                    [
                        ("Passed", f"{counts['passed']:,}"),
                        ("Failed", f"{counts['failed'] + counts['errors']:,}"),
                        ("Skipped", f"{counts['skipped']:,}"),
                        ("Exit status", str(status)),
                        ("Golden", golden),
                    ],
                )  # fmt: skip
            )
        if rows:
            con.lines(
                C.block(
                    "Coverage summary",
                    [("Areas", str(len(rows) - 1)), ("Time", C.fmt_duration(time.perf_counter() - t0))],
                )
                + C.table(
                    ["Area", "C++ lines", "C++ funcs", "C++ branches", "Py lines", "Py branches"], rows, "lrrrrr"
                )  # fmt: skip
            )
        notes = []
        if rows:
            notes.append(
                f"Per area, every file and every branch no test took: {OUT.relative_to(ROOT)}/<area>.md\n"
                f"(the table above is {(OUT / 'summary.md').relative_to(ROOT)})"
            )
            notes.append(f"Annotated sources: {(OUT / 'html' / 'cpp').relative_to(ROOT)}/index.html,\n"
                         f"{(OUT / 'html' / 'py').relative_to(ROOT)}/index.html")  # fmt: skip
        if "run" in stages:
            notes.append(
                "Deselected, as they crash the instrumented core (library bugs):\n" + "\n".join(sorted(CRASHES))
            )
            if "report" not in stages:
                notes.append("Write the reports with: .venv/bin/python tools/coverage.py report")
        notes.append(f"The tools' own output: {log.relative_to(ROOT)}")
        con.lines(C.notes(notes))
    return status


if __name__ == "__main__":
    sys.exit(main())
