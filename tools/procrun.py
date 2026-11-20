"""Running a tool's subprocess (pytest, cmake, ninja) behind the console of bench/console.py: its
output goes to the run's log, the terminal shows a progress bar read from that output and a
result line, so a tool prints the same blocks as bench.speed and bench.accuracy instead of the
subprocess's own stream. Used by tools/test.py and tools/coverage.py; their log files come from
bench/cli.py's log_file (build/logs/<tool>/, the newest kept)."""

from __future__ import annotations

import codecs
import datetime
import os
import pathlib
import re
import select
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

NINJA = re.compile(r"^\[(\d+)/(\d+)\]")  # ninja's '[12/340] Building CXX ...'
PYTEST_COUNTS = re.compile(r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b")
PYTEST_TOTAL = (
    re.compile(r"^\d+ workers \[(\d+) items?\]"),  # pytest-xdist
    re.compile(r"^collect(?:ing \.\.\. collect)?ed (\d+) items?(?: / \d+ deselected / (\d+) selected)?"),
)
# One line per finished test under pytest -v: '[gw3] [ 41%] PASSED tests/<file>.py::<test>' (xdist)
# or 'tests/<file>.py::<test> PASSED   [ 41%]'.
PYTEST_DONE = re.compile(
    r"^(?:\[gw\d+\] \[\s*\d+%\] (?P<o1>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (?P<t1>\S.*)"
    r"|(?P<t2>\S+::\S.*?) (?P<o2>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+\[\s*\d+%\])$"
)
NO_TESTS = 5  # pytest's exit status when nothing was selected
# pytest arguments tools/test.py and tools/coverage.py do not pass on (after --), and why
REFUSED = {
    "-k": "select with --match",
    "-m": "the tier and the two phases set the marks; select with --match",
    "-n": "set the workers with --jobs",
    "--numprocesses": "set the workers with --jobs",
    "--tier": "give --tier before --",
}
COUNTS = ("passed", "failed", "errors", "skipped", "deselected", "xfailed", "xpassed")


class PytestProgress:
    """Progress of a pytest -v run: the total from the collection line, one step per finished
    test, the failures counted as they come and the last test shown on the bar."""

    def __init__(self):
        self.total, self.done, self.bad, self.last = None, 0, 0, ""

    def __call__(self, line: str):
        if self.total is None:
            for rx in PYTEST_TOTAL:
                m = rx.match(line)
                if m:
                    self.total = int(m.group(m.lastindex))
            return None
        m = PYTEST_DONE.match(line)
        if not m:
            return None
        self.done += 1
        outcome = m.group("o1") or m.group("o2")
        self.bad += outcome in ("FAILED", "ERROR")
        self.last = (m.group("t1") or m.group("t2")).rsplit("/", 1)[-1]
        return self.done, max(self.total, self.done)

    def partial(self, text: str) -> bool:
        """The test that has started (pytest -v without workers prints its name first)."""
        m = re.match(r"^(\S+::\S.*?)\s*$", text)
        if not m or re.search(r" (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b", text):
            return False
        self.last = m.group(1).rsplit("/", 1)[-1]
        return True

    def what(self, room: int = 34) -> str:
        bad = f"{self.bad} failed · " if self.bad else ""
        name = self.last if len(self.last) <= room else self.last[: room - 1] + "…"
        return bad + name


def ninja_progress(line: str):
    m = NINJA.match(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


def run(cmd, con, desc: str, progress=None, env=None) -> tuple[int, list[str], float]:
    """cmd with its output in the log only; a bar over progress(line) -> (done, total) or None.
    Returns (exit status, every output line, seconds). progress.what(), when it has one, is shown
    after the bar; progress.partial(text), when it has one, sees the unfinished last line (pytest
    -v writes a test's name before running it, its outcome after)."""
    t0 = time.perf_counter()
    lines: list[str] = []
    state = {"p": None, "tenth": -1, "drawn": 0.0}

    def draw(force=False):
        done, total = state["p"]
        el = time.perf_counter() - t0
        left = el / done * (total - done) if done else None
        what = progress.what() if hasattr(progress, "what") else ""
        tenth = done * 10 // max(total, 1)  # without a bar: a plain line every tenth
        plain = f"  {desc}: {done}/{total}  ({_dur(el)} elapsed)" if (tenth != state["tenth"] and not con.tty) else ""
        state["tenth"] = tenth if plain else state["tenth"]
        if force or plain or done == total or el - state["drawn"] >= 0.1:  # a test can take a millisecond
            con.step(done, desc, what, el, left, plain)
            state["drawn"] = el

    proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    decode = codecs.getincrementaldecoder("utf-8")(errors="replace").decode
    fd, buf = proc.stdout.fileno(), ""
    while True:
        if not select.select([fd], [], [], 1.0)[0]:  # quiet for a second: the clock still moves
            if state["p"] is not None:
                draw(force=True)
            continue
        chunk = os.read(fd, 1 << 16)
        buf += decode(chunk, final=not chunk)
        *full, buf = buf.split("\n") if chunk else [*buf.split("\n"), ""]
        for line in full:
            line = line.rstrip("\r")
            lines.append(line)
            if con.log is not None:
                con.log.write(f"{datetime.datetime.now():%H:%M:%S} | {line}\n")
            p = progress(line) if progress else None
            if p is None:
                continue
            if state["p"] is None:
                con.start(p[1], desc)
            state["p"] = p
            draw()
        if buf and state["p"] is not None and hasattr(progress, "partial") and progress.partial(buf):
            draw(force=True)
        if not chunk:
            break
    code = proc.wait()
    con.close()
    return code, lines, time.perf_counter() - t0


def _dur(s):
    from bench.console import fmt_duration

    return fmt_duration(s)


def pytest_summary(lines: list[str]) -> tuple[dict, list[str]]:
    """The counts of pytest's last summary line and its 'FAILED id - message' / 'ERROR id'
    lines (the short test summary)."""
    counts = dict.fromkeys(COUNTS, 0)
    for line in lines[-8:]:
        if " in " in line and PYTEST_COUNTS.search(line):
            for k, word in PYTEST_COUNTS.findall(line):
                counts["errors" if word.startswith("error") else word] = int(k)
    failed = [ln for ln in lines if ln.startswith(("FAILED ", "ERROR "))]
    return counts, failed


def refused_pytest_args(args) -> str | None:
    """Why one of the pytest arguments is not passed on (REFUSED), or None."""
    for a in args:
        name = a.split("=", 1)[0]
        short = a[:2] if a[:2] in ("-k", "-m", "-n") else None  # also -kexpr, -mserial, -n4
        if short or name in REFUSED:
            return f"{a!r} after --: {REFUSED[short or name]}"
    return None


def pytest_status(codes) -> int:
    """The exit status (bench/cli.py: 0, 1, 2) of pytest runs with exit statuses `codes`: the
    worse of them (1 a test failed; 2 pytest stopped on an error of its own: an interrupt, an
    internal or usage error), and 1 when none selected a test."""
    states = [None if c == NO_TESTS else {0: 0, 1: 1}.get(c, 2) for c in codes]
    if all(s is None for s in states):
        return 1
    return max(s for s in states if s is not None)
