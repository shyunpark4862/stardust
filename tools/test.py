"""The test suite in parallel: the tests without the serial mark on pytest-xdist workers, then
the serial ones alone (tests/README.md §1; tests/conftest.py has the tiers and the marks).

    .venv/bin/python tools/test.py [AREA ...] [--match PATTERN] [--jobs N] [--tier T] [-- PYTEST ARGS]

    .venv/bin/python tools/test.py                        # quick, every test
    .venv/bin/python tools/test.py frame --match csv      # tests/frame, the tests whose id has "csv"
    .venv/bin/python tools/test.py --tier deep            # deep (or SD_TIER=deep)
    .venv/bin/python tools/test.py -- tests/bench -x      # pytest's own arguments after --

The options are bench/cli.py's: AREA (tests/<area>; default every test, the areas and the tests
outside them), --match PATTERN (the tests whose id, tests/<area>/test_<topic>.py::test_<name>[...], matches: a
glob, or a substring without wildcards; repeatable; tests/conftest.py applies it, so plain
pytest takes --match too), --jobs N (xdist workers, default 8, at most 10: timed 2026-11-20 on
this Mac with nothing else running, the whole quick tier with its 44 s serial part: 6 workers
76.6 s / 7.6 GB summed RSS, 8 workers 69.6 s / 9.7 GB, 10 workers 75.7 s / 12.9 GB). --tier
picks quick, deep or all (default SD_TIER, else quick). After --, arguments go to pytest as they
are, except -k and -m (--match selects; the tiers and the two phases set the marks) and -n
(--jobs). Every worker keeps sd's default thread count (2 threads per worker measured slower).
Plain `pytest` stays serial.

Exit status (bench/cli.py): 0 every selected test passed; 1 a test failed, or neither phase
selected any test; 2 a usage error, or pytest stopped on an error of its own (an interrupt, an
internal or usage error). The worse of the two phases decides. The terminal shows a header
block, a progress bar per phase and a summary (bench/console.py); pytest's own output goes to
build/logs/test/<stamp>.log.
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

from bench import cli  # noqa: E402
from bench import console as C  # noqa: E402

TIERS = ("quick", "deep", "all")  # tests/_tier.py


def phase_line(label, counts, secs, con) -> str:
    parts = [f"{counts['passed']:,} passed"]
    for k in ("failed", "errors", "skipped", "deselected", "xfailed", "xpassed"):
        if counts[k] or k in ("failed", "skipped"):
            parts.append(f"{counts[k]:,} {'error' if k == 'errors' and counts[k] == 1 else k}")
    text = f"  {label}: " + " · ".join(parts) + f"  ({C.fmt_duration(secs)})"
    return con.color(text, "31") if counts["failed"] or counts["errors"] else text


def parser():
    ap = argparse.ArgumentParser(prog="tools/test.py", description=__doc__.split("\n\n")[0])
    cli.add_areas(ap)
    cli.add_match(ap)
    cli.add_jobs(ap)
    ap.add_argument("--tier", choices=TIERS, help="quick, deep or all (default SD_TIER, else quick)")
    return ap


def parse(argv):
    """(options, pytest arguments): the arguments after -- go to pytest; ap.error on refused ones."""
    ap = parser()
    if "--" in argv:
        k = argv.index("--")
        mine, rest = argv[:k], argv[k + 1 :]
    else:
        mine, rest = argv, []
    args = ap.parse_args(mine)
    why = procrun.refused_pytest_args(rest)
    if why:
        ap.error(why)
    if args.areas:  # in AREAS order; none given means every test, not just the areas' folders
        args.areas = list(cli.areas(ap, args))
    return args, rest


def main(argv=None):
    args, rest = parse(sys.argv[1:] if argv is None else argv)
    tier = args.tier or os.environ.get("SD_TIER", "quick")
    paths = [f"tests/{a}" for a in args.areas] if args.areas else []
    select = paths + [x for p in args.match for x in ("--match", p)] + (["--tier", args.tier] if args.tier else [])
    log = cli.log_file("test")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    phases = (("parallel", ["-n", str(args.jobs), "-m", "not serial"]), ("serial", ["-m", "serial"]))
    base = [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", *select, *rest]
    codes, results, failed = [], [], []
    env = {**os.environ, "COLUMNS": "100"}  # pytest's short summary lines are cut at this width
    shown_sel = " ".join([*args.areas, *(f"--match {p}" for p in args.match), *(["--"] + rest if rest else [])])
    with open(log, "w") as fh:
        con = C.Console(log=fh, stream=sys.stdout)
        con.lines(
            C.block(
                "stardust tests",
                [
                    ("Tier", tier),
                    ("Workers", str(args.jobs)),
                    ("Commit", commit.stdout.strip() or "?"),
                    ("Phases", "2"),
                ],
                [
                    ("Selection", shown_sel or "every test of the tier"),
                    ("Parallel", f"pytest -n {args.jobs} -m 'not serial'"),
                    ("Serial", "pytest -m serial (one process)"),
                    ("Log", str(log.relative_to(ROOT))),
                ],
            )  # fmt: skip
        )
        for k, (label, extra) in enumerate(phases, 1):
            shown = " ".join(f"'{x}'" if " " in x else x for x in extra + select + rest)
            con.print(f"[{k}/{len(phases)}] {label}: pytest {shown}")
            code, lines, secs = procrun.run(base + extra, con, label, procrun.PytestProgress(), env=env)
            counts, fails = procrun.pytest_summary(lines)
            con.print(phase_line(label, counts, secs, con))
            if code not in (0, procrun.NO_TESTS) and not fails:  # pytest stopped without a summary (a crash, an error)
                con.lines([con.color(f"  {ln}", "31") for ln in lines[-12:]])
            results.append((label, counts, secs))
            failed += fails
            codes.append(code)
        status = procrun.pytest_status(codes)
        nothing = all(c == procrun.NO_TESTS for c in codes)
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
                    ("Exit status", str(status) + (" (no tests selected)" if nothing else "")),
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
        if nothing:
            notes.append(con.color("No tests selected: check the areas and the --match patterns.", "31"))
        if ids:
            more = " ..." if len(ids) > 5 else ""
            tier_env = f"SD_TIER={tier} " if tier != "quick" else ""
            again = " ".join(f"'{i}'" for i in ids[:5]) + more
            notes.append(f"Run the failed tests again, alone:\n{tier_env}.venv/bin/python -m pytest -q {again}")
            notes.append(f"pytest's own output (the tracebacks) is in the log:\n{log.relative_to(ROOT)}")
        notes += [
            "The deep tier (ten times the hypothesis examples, the golden and accuracy\n"
            "comparisons): .venv/bin/python tools/test.py --tier deep",
            "The serial tests (pytest -m serial --co lists them) run alone after the parallel\n"
            "phase: they time wall clocks or use most of the memory.",
        ]
        if not ids:
            notes.append(f"pytest's own output: {log.relative_to(ROOT)}")
        con.lines(C.notes(notes))
    return status


if __name__ == "__main__":
    sys.exit(main())
