"""The test suite in parallel: the tests without the serial mark on pytest-xdist workers, then
the serial ones alone (tests/README.md §1; tests/conftest.py has the tiers and the marks).

    .venv/bin/python tools/test.py [-n N] [PYTEST ARGS ...]     # e.g. tests/frame, -k merge, --tier deep

-n is the worker count (default 8, at most 10: timed 2026-11-20 on this Mac with nothing else
running, the whole quick tier with its 44 s serial part: 6 workers 76.6 s / 7.6 GB summed RSS,
8 workers 69.6 s / 9.7 GB, 10 workers 75.7 s / 12.9 GB; tests/README.md §1 has the current
count and how to take it). Every worker keeps sd's default thread count (2 threads per worker
measured slower). Plain `pytest` stays serial. The exit status is the worse of the two
runs; "no tests collected" in either counts as success.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import console as C  # noqa: E402

MAX_WORKERS = 10
NO_TESTS = 5  # pytest's exit status when nothing was selected
COUNTS = ("passed", "failed", "errors", "skipped", "deselected", "xfailed", "xpassed")


def run_phase(cmd, con) -> tuple[int, dict, list[str], float]:
    """pytest's own output as it comes; (exit status, counts, failed node ids, seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=ROOT)
    tail, failed = [], []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail = (tail + [line])[-5:]
        m = re.match(r"(FAILED|ERROR) (\S+)", line)
        if m:
            failed.append(m.group(2))
    code = proc.wait()
    counts = dict.fromkeys(COUNTS, 0)
    for line in tail:  # pytest's last line: '3 failed, 10451 passed, 142 deselected in 25.09s'
        for k, word in re.findall(r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b", line):
            counts["errors" if word.startswith("error") else word] = int(k)
    return code, counts, failed, time.perf_counter() - t0


def phase_line(label, counts, secs, con) -> str:
    parts = [f"{counts['passed']:,} passed"]
    for k in ("failed", "errors", "skipped", "deselected", "xfailed", "xpassed"):
        if counts[k] or k in ("failed", "skipped"):
            parts.append(f"{counts[k]:,} {k}")
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
    con = C.Console(stream=sys.stdout)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    phases = (("parallel", ["-n", str(n), "-m", "not serial"]), ("serial", ["-m", "serial"]))
    con.lines(
        C.block(
            "stardust tests",
            [("Tier", tier), ("Workers", str(n)), ("Commit", commit.stdout.strip() or "?"), ("Phases", "2")],
            [
                ("Selection", " ".join(rest) or "every test of the tier"),
                ("Parallel", f"pytest -n {n} -m 'not serial'"),
                ("Serial", "pytest -m serial (one process)"),
            ],
        )  # fmt: skip
    )
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *rest]
    status, results, failed = 0, [], []
    for k, (label, extra) in enumerate(phases, 1):
        shown = " ".join(f"'{x}'" if " " in x else x for x in extra + rest)
        con.print(f"[{k}/{len(phases)}] {label}: pytest {shown}")
        code, counts, ids, secs = run_phase(base + extra, con)
        con.print(phase_line(label, counts, secs, con))
        results.append((label, counts, secs, code))
        failed += ids
        if code not in (0, NO_TESTS):
            status = status or code
    total = {k: sum(r[1][k] for r in results) for k in COUNTS}
    times = {label: secs for label, _, secs, _ in results}
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
    if failed:
        con.lines(["Failed:"] + [f"  {con.color(i, '31')}" for i in failed])
    notes = []
    if failed:
        more = " ..." if len(failed) > 5 else ""
        notes.append("Run the failed tests again, alone:\n.venv/bin/python -m pytest -q -p no:cacheprovider "
                     + " ".join(failed[:5]) + more)  # fmt: skip
    notes += [
        "The deep tier (ten times the hypothesis examples, the golden and accuracy\n"
        "comparisons): SD_TIER=deep .venv/bin/python tools/test.py",
        "The serial tests (tests/conftest.py SERIAL) run alone after the parallel phase:\n"
        "they time wall clocks or use most of the memory.",
    ]
    con.lines(C.notes(notes))
    return status


if __name__ == "__main__":
    sys.exit(main())
