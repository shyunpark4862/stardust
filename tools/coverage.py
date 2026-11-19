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
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
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


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)


def xcrun(tool):
    return subprocess.check_output(["xcrun", "-f", tool], text=True).strip()


def core_path():
    found = sorted(CMAKE_DIR.glob("_core*.so"))
    if not found:
        raise SystemExit("coverage: no instrumented _core yet (run: tools/coverage.py build)")
    return found[0]


# --------------------------------------------------------------------------------------------
# build


def cmd_build():
    CMAKE_DIR.mkdir(parents=True, exist_ok=True)
    if not (CMAKE_DIR / "CMakeCache.txt").exists():
        r = run([
            "cmake", "-S", ROOT, "-B", CMAKE_DIR, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={PY}", f"-DCMAKE_CXX_FLAGS={FLAGS}",
            f"-DCMAKE_MODULE_LINKER_FLAGS={FLAGS}", f"-DCMAKE_SHARED_LINKER_FLAGS={FLAGS}",
        ])  # fmt: skip
        if r.returncode:
            raise SystemExit("coverage: cmake configure failed")
    r = run(["cmake", "--build", CMAKE_DIR, "-j", str(JOBS)])
    if r.returncode:
        raise SystemExit("coverage: build failed")
    print(f"instrumented core: {core_path().relative_to(ROOT)}")


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


def cmd_run(golden, pytest_args):
    core_path()
    for d in (PROFRAW, PYDATA):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    env = env_for_run()
    args = pytest_args or ["tests"]
    deselect = [a for t in sorted(CRASHES) for a in ("--deselect", t)]
    for t in sorted(CRASHES):
        print(f"coverage: deselected (crashes the instrumented core): {t}")
    r = run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--cov", "--cov-branch",
             f"--cov-config={RCFILE}", "--cov-report=", *deselect, *args], env=env)  # fmt: skip
    status = r.returncode
    if golden:
        g = run([PY, "-m", "coverage", "run", f"--rcfile={RCFILE}", "tools/golden.py", "check", "--any-nan"], env=env)
        if g.returncode:
            print("coverage: golden check with the instrumented core found differences (see above)")
            status = status or g.returncode
    print(f"profiles: {len(list(PROFRAW.glob('*.profraw')))} files in {PROFRAW.relative_to(ROOT)}")
    return status


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


def cmd_report():
    OUT.mkdir(parents=True, exist_ok=True)
    areas = {}  # area -> {"cpp": [file rows], "py": [file rows]}
    raws = sorted(PROFRAW.glob("*.profraw"))
    if raws:
        prof = OUT / "core.profdata"
        r = run([xcrun("llvm-profdata"), "merge", "-sparse", *raws, "-o", prof])
        if r.returncode:
            raise SystemExit("coverage: llvm-profdata merge failed")
        cov = xcrun("llvm-cov")
        obj = core_path()
        common = ["-instr-profile", prof, f"-ignore-filename-regex={IGNORE}"]
        exported = OUT / "cpp.json"
        with open(exported, "w") as fh:
            r = run([cov, "export", obj, "-format=text", *common], stdout=fh)
        if r.returncode:
            raise SystemExit("coverage: llvm-cov export failed")
        shutil.rmtree(OUT / "html" / "cpp", ignore_errors=True)
        run([cov, "show", obj, "-format=html", "-show-branches=count", "-output-dir", OUT / "html" / "cpp",
             f"-Xdemangler={xcrun('c++filt')}", *common], stdout=subprocess.DEVNULL)  # fmt: skip
        for row in cpp_rows(json.loads(exported.read_text())):
            areas.setdefault(row["area"], {"cpp": [], "py": []})["cpp"].append(row)
    else:
        print("coverage: no C++ profiles (run first)")
    data = list(PYDATA.glob(".coverage*"))
    if data:
        env = dict(os.environ, COVERAGE_RCFILE=str(RCFILE))
        if any(p.name != ".coverage" for p in data):
            run([PY, "-m", "coverage", "combine", "--append", f"--rcfile={RCFILE}"], env=env)
        pyjson = OUT / "py.json"
        run([PY, "-m", "coverage", "json", f"--rcfile={RCFILE}", "-o", pyjson], env=env, stdout=subprocess.DEVNULL)
        shutil.rmtree(OUT / "html" / "py", ignore_errors=True)
        run([PY, "-m", "coverage", "html", f"--rcfile={RCFILE}", "-d", OUT / "html" / "py"], env=env,
            stdout=subprocess.DEVNULL)  # fmt: skip
        for row in py_rows(json.loads(pyjson.read_text())):
            areas.setdefault(row["area"], {"cpp": [], "py": []})["py"].append(row)
    else:
        print("coverage: no Python coverage data (run first)")
    write_reports(areas)


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
    print("\n".join(lines))
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
    print(f"reports: {(OUT / 'summary.md').relative_to(ROOT)}, {OUT.relative_to(ROOT)}/<area>.md, "
          f"{(OUT / 'html').relative_to(ROOT)}/")  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["all", "build", "run", "report"])
    ap.add_argument("--golden", action="store_true", help="run: also tools/golden.py check, instrumented")
    args, rest = ap.parse_known_args()
    if args.command == "build":
        return cmd_build()
    if args.command == "report":
        return cmd_report()
    if args.command == "all":
        cmd_build()
    status = cmd_run(args.golden, rest)
    if args.command == "all":
        cmd_report()
    return status


if __name__ == "__main__":
    sys.exit(main())
