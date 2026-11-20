"""The test suite in parallel: the tests without the serial mark on pytest-xdist workers, then
the serial ones alone (tests/README.md §1; tests/conftest.py has the tiers and the marks).

    .venv/bin/python tools/test.py [-n N] [PYTEST ARGS ...]     # e.g. tests/frame, -k merge, --tier deep

-n is the worker count (default 8, at most 10: timed 2026-11-20 on this Mac with nothing else
running, the whole quick tier with its 44 s serial part: 6 workers 76.6 s / 7.6 GB summed RSS,
8 workers 69.6 s / 9.7 GB, 10 workers 75.7 s / 12.9 GB; tests/README.md §1 has the current
count and how to take it). Every worker keeps sd's default thread count (2 threads per worker
measured slower). Plain `pytest` stays serial. The exit status is the worse of the two
runs; "no tests collected" in either counts as success. The terminal shows a header block, a
progress bar per phase and a summary (bench/console.py); pytest's own output goes to
build/test/<stamp>.log.
"""

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import procrun  # noqa: E402

from bench import console as C  # noqa: E402

MAX_WORKERS = 10
NO_TESTS = 5  # pytest's exit status when nothing was selected


def phase_line(label, counts, secs, con) -> str:
    parts = [f"{counts['passed']:,} passed"]
    for k in ("failed", "errors", "skipped", "deselected", "xfailed", "xpassed"):
        if counts[k] or k in ("failed", "skipped"):
            parts.append(f"{counts[k]:,} {'error' if k == 'errors' and counts[k] == 1 else k}")
    text = f"  {label}: " + " · ".join(parts) + f"  ({C.fmt_duration(secs)})"
    return con.color(text, "31") if counts["failed"] or counts["errors"] else text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-n", type=int, default=8, help=f"workers (default 8, at most {MAX_WORKERS})")
    args, rest = ap.parse_known_args()
    n = max(1, min(args.n, MAX_WORKERS))
    tier = os.environ.get("SD_TIER", "quick")
    for k, v in enumerate(rest):
        if v == "--tier" and k + 1 < len(rest):
            tier = rest[k + 1]
        elif v.startswith("--tier="):
            tier = v.split("=", 1)[1]
    log = procrun.log_path("test")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    phases = (("parallel", ["-n", str(n), "-m", "not serial"]), ("serial", ["-m", "serial"]))
    base = [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", *rest]
    status, results, failed = 0, [], []
    env = {**os.environ, "COLUMNS": "100"}  # pytest's short summary lines are cut at this width
    with open(log, "w") as fh:
        con = C.Console(log=fh, stream=sys.stdout)
        con.lines(
            C.block(
                "stardust tests",
                [("Tier", tier), ("Workers", str(n)), ("Commit", commit.stdout.strip() or "?"), ("Phases", "2")],
                [
                    ("Selection", " ".join(rest) or "every test of the tier"),
                    ("Parallel", f"pytest -n {n} -m 'not serial'"),
                    ("Serial", "pytest -m serial (one process)"),
                    ("Log", str(log.relative_to(ROOT))),
                ],
            )  # fmt: skip
        )
        for k, (label, extra) in enumerate(phases, 1):
            shown = " ".join(f"'{x}'" if " " in x else x for x in extra + rest)
            con.print(f"[{k}/{len(phases)}] {label}: pytest {shown}")
            code, lines, secs = procrun.run(base + extra, con, label, procrun.PytestProgress(), env=env)
            counts, fails = procrun.pytest_summary(lines)
            con.print(phase_line(label, counts, secs, con))
            if code not in (0, NO_TESTS) and not fails:  # pytest stopped without a summary (a crash, an error)
                con.lines([con.color(f"  {ln}", "31") for ln in lines[-12:]])
            results.append((label, counts, secs))
            failed += fails
            if code not in (0, NO_TESTS):
                status = status or code
        total = {key: sum(r[1][key] for r in results) for key in procrun.COUNTS}
        times = {label: secs for label, _, secs in results}
        con.lines(
            [""]
            + C.block(
                "Test summary",
                [
                    ("Passed", f"{total['passed']:,}"),
                    ("Failed", f"{total['failed']:,}"),
                    ("Errors", f"{total['errors']:,}"),
                    ("Skipped", f"{total['skipped']:,}"),
                    ("Deselected", f"{total['deselected']:,}"),
                    ("Exit status", str(status)),
                    ("Parallel", C.fmt_duration(times.get("parallel", 0))),
                    ("Serial", C.fmt_duration(times.get("serial", 0))),
                    ("Total", C.fmt_duration(sum(times.values()))),
                    ("Tier", tier),
                ],
            )  # fmt: skip
        )
        ids = [f.split()[1] for f in failed]
        if failed:
            w = con.width()
            con.print("Failed:")
            for f in failed:
                con.lines([con.color(x, "31") for x in C.split_label("  " + f, 6, w)])
        notes = []
        if ids:
            more = " ..." if len(ids) > 5 else ""
            tier_env = f"SD_TIER={tier} " if tier != "quick" else ""
            again = " ".join(f"'{i}'" for i in ids[:5]) + more
            notes.append(f"Run the failed tests again, alone:\n{tier_env}.venv/bin/python -m pytest -q {again}")
            notes.append(f"pytest's own output (the tracebacks) is in the log:\n{log.relative_to(ROOT)}")
        notes += [
            "The deep tier (ten times the hypothesis examples, the golden and accuracy\n"
            "comparisons): SD_TIER=deep .venv/bin/python tools/test.py",
            "The serial tests (tests/conftest.py SERIAL) run alone after the parallel phase:\n"
            "they time wall clocks or use most of the memory.",
        ]
        if not ids:
            notes.append(f"pytest's own output: {log.relative_to(ROOT)}")
        con.lines(C.notes(notes))
    return status


if __name__ == "__main__":
    sys.exit(main())
