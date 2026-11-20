"""The supervisor: run cases in a worker process and write a results file.

Rounds: every round walks all selected cases, and within a case times stardust and its
comparators one after the other, so a slow spell of the machine hits both sides of a
ratio. Each measurement keeps every batch of every round; the reported time is the
minimum over all of them and the noise band is median / min - 1.

Long runs:
- The measuring happens in a worker process (bench/speed/worker.py) in its own process
  group. The supervisor watches each case's wall time and the resident memory of the
  worker and its children (the R session); past a limit it kills the group, records
  the case as "timeout" or "memory", starts a new worker and goes on. Workers are also
  replaced every `restart_every` cases, so memory a case leaks does not pile up, and at
  every round boundary of a run with several rounds: a short parallel call may settle in
  a faster or slower mode for a whole process (2.4x apart for a + b at 2^16 elements,
  10 threads), so each round's minimum comes from another process.
- After every case the run file gets one more line (the case with its raw batch times;
  results.py), so an interrupted run loses at most one case: `run --resume FILE`
  continues it.
- Progress lines show the elapsed time and the expected rest, from the baselines'
  seconds per case (scaled by how this run compares so far).
- Before the run, the CPU time other processes use is measured for a few seconds
  (proc.idle_check); when it is above `quiet_cores`, the busiest processes are listed and
  the run waits for a quiet machine (up to `wait_quiet_s`; then it starts with a warning,
  or stops with quiet="strict"). During the run, the cores other processes kept busy are
  recorded per case (proc.CpuMeter, no process started); a case above `busy_cores` is
  marked busy, in the progress line, the run file, the tables and compare, and
  `run --rerun-busy RUN.json` measures just those cases again.
- caffeinate keeps the machine awake while the supervisor lives; the log file
  (build/logs/speed/<stamp>.log, bench/cli.py) gets the progress lines and the worker's
  stderr; the 1-minute load is recorded per case; a macOS notification says when a long
  run is done.
- A smoke run also checks the results (check.py): a case whose check fails counts as a
  failure, and any failure (an error, a timeout, a memory stop, a failed check) makes the
  command's exit code 1.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import re
import select
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

from .. import cli
from ..console import WIDTH, Console, block, case_label, fmt_duration, ljust, rjust, rule, split_label, table, width
from . import harness, proc, results
from .harness import fmt_time
from .registry import Case, RRef


@dataclasses.dataclass
class Limits:
    case_timeout_s: float = 15 * 60  # one case, one round (all threads and comparators)
    rss_bytes: int = 16 * 2**30  # the worker and its children (R)
    poll_s: float = 2.0
    restart_every: int = 40  # cases per worker
    # An idle worker holding more than this is closed before another worker measures:
    # macOS keeps large freed blocks resident, so after a case at n = 10^6 a worker may sit
    # on GBs that would count against rss_bytes while the next worker runs.
    idle_rss_bytes: int = 2 * 2**30
    startup_timeout_s: float = 30 * 60  # importing stardust may rebuild the C++ core
    load_warn: float = 3.0  # without CPU counters: warn when the 1-minute load is above this
    quiet_cores: float = 1.0  # before the run: other processes may keep at most this many cores busy
    idle_window_s: float = 3.0  # ... measured over this window
    wait_quiet_s: float = 10 * 60  # wait this long for a quiet machine before starting anyway
    wait_poll_s: float = 10.0
    busy_cores: float = 1.0  # a case during which other processes used more is marked busy
    notify_after_s: float = 60.0  # a notification only for runs at least this long


# sd's default thread count (the performance cores): measured in the worker that leaves
# Accelerate at its default, next to the "fixed" comparators.
DEFAULT_THREADS = int(proc.perf_cores() or 10)


# ---------------------------------------------------------------- the worker handle


class WorkerStartError(RuntimeError):
    def __init__(self, status):
        super().__init__(f"the worker did not start ({status}); see the log")
        self.status = status


class WorkerHandle:
    """A worker process. blas_threads: VECLIB_MAXIMUM_THREADS for it (None: Accelerate's
    default, all cores)."""

    live: set[int] = set()  # pids of the workers alive, whose memory counts against the limit

    def __init__(self, areas, log, limits: Limits, smoke: bool = False, blas_threads: int | None = None):
        self.limits = limits
        self.blas_threads = blas_threads
        env = dict(os.environ)
        env.pop("VECLIB_MAXIMUM_THREADS", None)
        if blas_threads is not None:
            env["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
        self.p = subprocess.Popen(
            [sys.executable, "-m", "bench.speed.worker", *areas, *(["--smoke"] if smoke else [])],
            cwd=str(results.ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )
        WorkerHandle.live.add(self.p.pid)
        self.buf = b""
        self.served = 0
        _, status, _ = self._read(limits.startup_timeout_s, starting=True)
        if status != "ok":
            self.kill()
            raise WorkerStartError(status)

    def _read(self, timeout_s, starting=False):
        """One reply line: (dict | None, "ok" | "timeout" | "memory" | "died", peak RSS).

        The memory limit counts every live worker (idle ones too, and parallel lanes) with
        its R session. Past it, only the worker holding the most stops its case: the others
        wait for that one to be killed. A starting worker counts only itself."""
        t0 = time.monotonic()
        peak = 0
        fd = self.p.stdout.fileno()
        while b"\n" not in self.buf:
            own = proc.tree_rss(self.p.pid)
            others = [] if starting else [proc.tree_rss(pid) for pid in list(WorkerHandle.live) if pid != self.p.pid]
            rss = own + sum(others)
            peak = max(peak, rss)
            if rss > self.limits.rss_bytes and own >= max(others, default=0):
                return None, "memory", peak
            if time.monotonic() - t0 > timeout_s:
                return None, "timeout", peak
            r, _, _ = select.select([fd], [], [], self.limits.poll_s)
            if r:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    return None, "died", peak
                self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line), "ok", peak

    def request(self, obj, timeout_s):
        try:
            self.p.stdin.write((json.dumps(obj) + "\n").encode())
            self.p.stdin.flush()
        except BrokenPipeError, OSError:
            return None, "died", 0
        return self._read(timeout_s)

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.write(b'{"cmd": "quit"}\n')
                self.p.stdin.flush()
                self.p.wait(timeout=30)
            except OSError, subprocess.SubprocessError:
                self.kill()
        WorkerHandle.live.discard(self.p.pid)

    def kill(self):
        """Kill the worker's process group: the worker and the R session it started."""
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except ProcessLookupError, PermissionError:
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.SubprocessError:
            pass
        WorkerHandle.live.discard(self.p.pid)


# ---------------------------------------------------------------- run state


class State:
    """Everything a run file holds, built up case by case."""

    def __init__(self, cases: list[Case], threads, use_r, busy_cores: float = 1.0):
        self.busy_cores = busy_cores
        self.cases = cases
        self.threads = threads
        self.use_r = use_r
        self.samples: dict[str, dict[tuple, harness.Sample]] = {c.id: {} for c in cases}
        self.checks: dict[str, dict[str, tuple[bool, str]]] = {c.id: {} for c in cases}
        self.libs: dict[str, dict[str, str]] = {c.id: {} for c in cases}
        self.errors: dict[str, dict[tuple, str]] = {c.id: {} for c in cases}
        self.seconds: dict[str, float] = {c.id: 0.0 for c in cases}
        self.rounds_done: dict[str, int] = {c.id: 0 for c in cases}
        self.load: dict[str, float] = {c.id: 0.0 for c in cases}
        self.other: dict[str, float] = {}  # cores other processes kept busy (max over rounds)
        self.rss: dict[str, int] = {c.id: 0 for c in cases}
        self.failure: dict[str, str] = {}
        self.setup_error: dict[str, str] = {}

    def add_reply(self, reply):
        cid = reply["id"]
        for label, lib, t, times, calls, nan in reply["samples"]:
            key = (label, t)
            s = harness.Sample(list(times), calls, nan)
            old = self.samples[cid].get(key)
            self.samples[cid][key] = s if old is None else old.merged(s)
            self.libs[cid][label] = lib
            self.errors[cid].pop(key, None)
        for label, t, err in reply["errors"]:
            if (label, t) not in self.samples[cid]:
                self.errors[cid][(label, t)] = err
        if reply.get("setup_error"):
            self.setup_error[cid] = reply["setup_error"]
        for label, ok, text in reply.get("checks") or []:
            old = self.checks[cid].get(label)
            if old is None or old[0]:  # a failed check stays failed
                self.checks[cid][label] = (bool(ok), text)

    def record(self, case: Case) -> dict:
        cid = case.id
        ths = [t for t in case.threads if self.threads is None or t in self.threads]
        want = [("sd", "sd", t, case.n) for t in ths]
        for ref in case.refs:
            if isinstance(ref, RRef) and not self.use_r:
                continue
            n = ref.n or case.n
            if ref.threads == "follow":
                want += [(ref.label, ref.library, t, n) for t in ths]
            else:
                want.append((ref.label, ref.library, None, n))
        timings = []
        for label, lib, t, n in want:
            s = self.samples[cid].get((label, t))
            rec = {"label": label, "lib": lib, "threads": t, "n": n}
            if s is not None:
                rec.update(
                    min=s.min,
                    median=s.median,
                    noise=round(s.noise, 4),
                    batches=len(s.times),
                    calls=s.calls,
                    times=s.times,
                )
                if s.nan is not None:
                    rec["nan"] = round(s.nan, 4)
            else:
                rec.update(min=None, error=self.errors[cid].get((label, t)) or self.failure.get(cid) or "not measured")
            if label in self.checks[cid] and not any(t["label"] == label for t in timings):
                ok, text = self.checks[cid][label]
                rec["check"] = {"ok": ok, "text": text}  # on the label's first timing
            timings.append(rec)
        rd = self.rounds_done[cid]
        out = {
            "id": cid,
            "area": case.area,
            "group": case.group,
            "name": case.name,
            "size": case.size,
            "n": case.n,
            "cost": list(case.cost) if case.cost else None,
            "note": case.notes,
            "rounds": rd,
            "seconds_per_round": round(self.seconds[cid] / rd, 2) if rd else None,
            "load1": round(self.load[cid], 2),
            "peak_rss_mb": round(self.rss[cid] / 2**20),
            "other_cpu": round(self.other[cid], 2) if cid in self.other else None,
            "busy": self.other.get(cid, 0.0) > self.busy_cores,
            "timings": timings,
        }
        if cid in self.failure:
            out["failure"] = self.failure[cid]
        if cid in self.setup_error:
            out["setup_error"] = self.setup_error[cid]
        return out

    @classmethod
    def from_file(cls, data, cases):
        st = cls(
            cases,
            tuple(data["options"]["threads"]) if data["options"].get("threads") else None,
            data["options"].get("r", True),
        )
        for c in data["cases"]:
            cid = c["id"]
            if cid not in st.samples:
                continue
            for t in c["timings"]:
                if t.get("times"):
                    sample = harness.Sample(t["times"], t.get("calls", 1), t.get("nan"))
                    st.samples[cid][(t["label"], t["threads"])] = sample
                    st.libs[cid][t["label"]] = t["lib"]
                if t.get("check"):
                    st.checks[cid][t["label"]] = (t["check"]["ok"], t["check"]["text"])
            rd = c.get("rounds") or 0
            st.rounds_done[cid] = rd
            st.seconds[cid] = (c.get("seconds_per_round") or 0.0) * rd
            st.load[cid] = c.get("load1") or 0.0
            if c.get("other_cpu") is not None:
                st.other[cid] = c["other_cpu"]
            st.rss[cid] = int((c.get("peak_rss_mb") or 0) * 2**20)
            if c.get("failure"):
                st.failure[cid] = c["failure"]
        return st


# ---------------------------------------------------------------- estimates


def expected_seconds(cases: list[Case]) -> dict[str, float | None]:
    """Seconds per round of each case, from the baselines (None: never measured)."""
    base = results.baseline_cases(sorted({c.area for c in cases}))
    return {c.id: (base.get(c.id) or {}).get("seconds_per_round") for c in cases}


def estimate(cases: list[Case], rounds: int) -> tuple[float, int]:
    """(expected seconds of the known cases over `rounds` rounds, cases without an estimate)."""
    exp = expected_seconds(cases)
    known = sum(v for v in exp.values() if v)
    return known * rounds, sum(1 for v in exp.values() if not v)


class Eta:
    def __init__(self, pairs, expected):
        self.expected = expected
        self.todo = list(pairs)
        self.known_actual = self.known_expected = 0.0
        self.unknown_actual, self.unknown_n = 0.0, 0

    def done(self, cid, seconds):
        e = self.expected.get(cid)
        if e:
            self.known_actual += seconds
            self.known_expected += e
        else:
            self.unknown_actual += seconds
            self.unknown_n += 1

    def remaining(self, pairs_left) -> float | None:
        factor = self.known_actual / self.known_expected if self.known_expected else 1.0
        mean_unknown = self.unknown_actual / self.unknown_n if self.unknown_n else None
        total = 0.0
        for _, cid in pairs_left:
            e = self.expected.get(cid)
            if e:
                total += e * factor
            elif mean_unknown is not None:
                total += mean_unknown
            else:
                return None
        return total


# ---------------------------------------------------------------- a quiet machine


def wait_for_quiet(con: Console, limits: Limits, mode: str = "wait") -> dict | None:
    """Measure the CPU other processes use; when it is above limits.quiet_cores, list the
    busiest and (mode "wait" or "strict") wait for it to drop. None when the counters
    cannot be read. Returns {"other_cores", "busiest", "waited_s", "quiet", "stopped"}."""
    other, top = proc.idle_check(limits.idle_window_s)
    if other is None:
        return None
    t0 = time.monotonic()
    first = (other, top)
    if other <= limits.quiet_cores:
        return {"other_cores": round(other, 2), "busiest": top, "waited_s": 0, "quiet": True, "stopped": False}
    con.print(f"CPU check: other processes keep {other:.1f} cores busy (limit {limits.quiet_cores:g}).")
    for row in top or ["(processes that cannot be read)"]:
        con.print(f"  {row}")
    if mode == "no":
        return {"other_cores": round(other, 2), "busiest": top, "waited_s": 0, "quiet": False, "stopped": False}
    con.print(f"Waiting up to {fmt_duration(limits.wait_quiet_s)} for a quiet machine (--no-wait starts now).")
    last_report = time.monotonic()
    while time.monotonic() - t0 < limits.wait_quiet_s:
        time.sleep(limits.wait_poll_s)
        other, top = proc.idle_check(limits.idle_window_s)
        if other is not None and other <= limits.quiet_cores:
            waited = time.monotonic() - t0
            con.print(f"Quiet after {fmt_duration(waited)} ({other:.1f} cores). Starting.")
            return {
                "other_cores": round(other, 2),
                "busiest": top,
                "waited_s": round(waited),
                "quiet": True,
                "stopped": False,
                "at_first": round(first[0], 2),
            }
        if time.monotonic() - last_report >= 60:
            con.print(f"  still busy: {other:.1f} cores ({', '.join(top[:3])})")
            last_report = time.monotonic()
    waited = time.monotonic() - t0
    stopped = mode == "strict"
    con.print(
        f"Still busy after {fmt_duration(waited)} ({other:.1f} cores): "
        + ("stopping (--strict)." if stopped else "starting anyway; expect more noise.")
    )
    return {
        "other_cores": round(other, 2),
        "busiest": top,
        "waited_s": round(waited),
        "quiet": False,
        "stopped": stopped,
        "at_first": round(first[0], 2),
    }


# ---------------------------------------------------------------- what the console shows


def _round_desc(rnd: int, rounds: int, smoke: bool) -> str:
    return "Smoke" if smoke and rounds == 1 else f"Round {rnd + 1}/{rounds}"


def _header(cases, rounds, threads, use_r, have_r, smoke, lanes, jobs, limits, env, run_id, quiet, first, out, log):
    git = env.get("git", {})
    machine = env.get("machine", {})
    ths = threads or sorted({t for c in cases for t in c.threads})
    est, unknown = estimate(cases, rounds)
    if unknown == len(cases):
        expected = "unknown (no baseline)"
    elif unknown:
        expected = f"{fmt_duration(est)} + {unknown} unknown"
    else:
        expected = fmt_duration(est)
    if quiet is None:
        cpu = "not checked" if smoke else "no counters"
    else:
        cpu = f"{quiet['other_cores']:.1f} cores" + ("" if quiet.get("quiet") else " (busy)")
    has_r = any(isinstance(r, RRef) for c in cases for r in c.refs)
    r_text = "-" if not has_r else "yes" if use_r else "not found" if shutil.which("R") is None else "off"
    pairs = [
        ("Run", run_id),
        ("Commit", f"{git.get('commit') or '-'} ({'dirty' if git.get('dirty') else 'clean'})"),
        ("Cases", f"{len(cases):,}"),
        ("Rounds", str(rounds)),
        ("sd threads", ", ".join(map(str, ths))),
        ("R comparators", r_text),
        ("Expected", "-" if smoke else expected),
        ("Other CPU", cpu),
        ("Case limits", f"{fmt_duration(limits.case_timeout_s)}, {limits.rss_bytes / 2**30:g} GB"),
        ("Machine", f"{machine.get('chip', '?')}, {machine.get('cpus', '?')} cores"),
    ]
    if lanes > 1:
        pairs.append(("Lanes", f"{lanes} (at most {jobs} processes)"))
    if first:
        pairs.append(("Resumed at", f"{first:,} of {rounds * len(cases):,}"))
    areas = ", ".join(sorted({c.area for c in cases}))
    title = "stardust smoke run (every case called once, results checked)" if smoke else "stardust speed run"
    files = [("Results", _show(out)), ("Log", _show(log))]
    return [""] + block(title, pairs, [("Areas", areas), *files])


def _show(path) -> str:
    """A path relative to the repository when it is inside it."""
    path = pathlib.Path(path).resolve()
    try:
        return str(path.relative_to(results.ROOT))
    except ValueError:
        return str(path)


def _ratios(c) -> dict[str, float]:
    """comparator label -> comparator time / sd time, per element (a comparator timed on a
    sample has its own n), at sd's lowest thread count: a "follow" comparator at that count,
    a "fixed" one at its own threads."""
    ts = c["timings"]
    sd = [t for t in ts if t["label"] == "sd" and t.get("min")]
    if not sd:
        return {}
    s = min(sd, key=lambda t: t["threads"])
    per_sd = s["min"] / (s.get("n") or 1)
    return {
        t["label"]: (t["min"] / (t.get("n") or 1)) / per_sd
        for t in ts
        if t["label"] != "sd" and t.get("min") and t["threads"] in (None, s["threads"])
    }


def _threads_text(t) -> str:
    if t is None:
        return "its default threads"
    return f"{t} thread" + ("" if t == 1 else "s")


def _result_line(con: Console, case, rec, rnd: int, rounds: int) -> str:
    """One case's result so far (the minimum over the rounds done), a few lines in words:

        regression · ridge(y, X, alpha=0.1) [(100,000, 100)]
            sd, 1 thread                 96.5 ms
            sd, 10 threads               22.9 ms   4.21x faster than 1 thread
            scikit-learn Ridge           13.6 ms   sd is 7.09x slower
            scikit-learn Ridge svd        340 ms   sd is 3.52x faster

    A comparator is set against sd at the same thread count (one that runs at its own
    threads against sd's lowest), per element: a comparator timed on a smaller sample is
    shown scaled to the case's size. Slower verdicts are red, a busy note is yellow."""
    W = con.width()
    lead = f"[round {rnd + 1}/{rounds}] " if rounds > 1 else ""
    head = split_label(lead + case_label(case), 4, W - 2)
    lines = ["  " + head[0]] + ["  " + h for h in head[1:]]
    ts = rec["timings"]
    sd = {t["threads"]: t for t in ts if t["label"] == "sd"}
    low = min(sd) if sd else None
    rows = []  # (label, time text, verdict, verdict is bad)

    def per(t):
        return t["min"] / (t.get("n") or 1)

    for th in sorted(sd):
        t = sd[th]
        label = f"sd, {_threads_text(th)}" if len(sd) > 1 or th != 1 else "sd"
        if t.get("min") is None:
            rows.append((label, "failed", "", True))
            continue
        verdict, bad = "", False
        if th != low and sd[low].get("min"):
            x = sd[low]["min"] / t["min"]
            verdict = (f"{x:.2f}x faster" if x >= 1 else f"{1 / x:.2f}x slower") + f" than {_threads_text(low)}"
            bad = x < 1
        rows.append((label, fmt_time(t["min"]), verdict, bad))
    labels = list(dict.fromkeys(t["label"] for t in ts if t["label"] != "sd"))
    for label in labels:
        mine = [t for t in ts if t["label"] == label]
        for t in sorted(mine, key=lambda t: (t["threads"] is not None, t["threads"] or 0)):
            name = label if len(mine) == 1 else f"{label}, {_threads_text(t['threads'])}"
            if t.get("min") is None:
                rows.append((name, "failed", "", True))
                continue
            scale = (case.n or 1) / (t.get("n") or 1) if t.get("n") and case.n and t["n"] != case.n else 1.0
            shown = fmt_time(t["min"] * scale) + ("*" if scale != 1.0 else "")
            ref_sd = sd.get(t["threads"]) or sd.get(low)
            if not ref_sd or ref_sd.get("min") is None:
                rows.append((name, shown, "", False))
                continue
            r = per(t) / per(ref_sd)
            who = "sd" if t["threads"] is not None or len(sd) == 1 else f"sd ({_threads_text(low)})"
            verdict = f"{who} is {r:.2f}x faster" if r >= 1 else f"{who} is {1 / r:.2f}x slower"
            rows.append((name, shown, verdict, r < 1))
    name_w = min(max((width(r[0]) for r in rows), default=0), max(20, W - 6 - 11 - 3 - 30))
    for name, tm, verdict, bad in rows:
        name_lines = split_label(name, 2, name_w) if width(name) > name_w else [name]
        for extra in name_lines[:-1]:
            lines.append("      " + extra)
        cell = "      " + ljust(name_lines[-1], name_w) + "  " + rjust(tm, 10)
        if verdict:
            cell += "   " + (con.color(verdict, "31") if bad else verdict)
        elif tm == "failed":
            cell = cell.replace("failed", con.color("failed", "31"))
        lines.append(cell)
    if any(t.get("n") and case.n and t["n"] != case.n for t in ts if t["label"] != "sd"):
        lines.append("      " + con.color("* timed on a smaller sample, scaled to this size", "2"))
    if rec.get("busy"):
        lines.append("      " + con.color(f"busy: other processes kept {rec['other_cpu']:.1f} cores busy", "33"))
    return "\n".join(lines)


def _strip(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _summary(data, total, interrupted, out, log_path) -> list[str]:
    cs = data["cases"]
    opts = data["options"]
    smoke = opts.get("smoke", False)
    timings = [t for c in cs for t in c["timings"]]
    failed = [t for t in timings if t.get("min") is None and t.get("error") != "not measured"]
    bad_checks = [t for t in timings if t.get("check") and not t["check"]["ok"]]
    stopped = [c for c in cs if c.get("failure")]
    busy = [c for c in cs if c.get("busy")]
    peak = max((c.get("peak_rss_mb") or 0 for c in cs), default=0)
    pairs = [
        ("Cases", f"{len(cs):,}"),
        ("Elapsed", fmt_duration(total)),
        ("Measurements", f"{sum(1 for t in timings if t.get('min') is not None):,}"),
        ("Failed", f"{len(failed):,}"),
        *(
            [("Checks failed", f"{len(bad_checks):,} of {sum(1 for t in timings if t.get('check')):,}")]
            if smoke
            else []
        ),
        ("Stopped cases", f"{len(stopped):,}"),
        ("Busy cases", f"{len(busy):,}"),
        ("Peak memory", f"{peak / 1024:.1f} GB"),
        ("Rounds done", f"{min((c.get('rounds') or 0 for c in cs), default=0)} of {opts['rounds']}"),
    ]
    title = "Run summary" + (" (interrupted)" if interrupted else "")
    lines = [""] + block(title, pairs)
    by_area: dict[str, list] = {}
    for c in cs:
        by_area.setdefault(c["area"], []).append(c)
    rows = []
    for area, group in by_area.items():
        bad = sum(1 for c in group if case_failed(c))
        row = [
            area,
            f"{len(group):,}",
            f"{bad:,}",
            f"{sum(1 for c in group if c.get('busy')):,}",
            fmt_duration(sum(c.get("seconds_per_round") or 0 for c in group)),
        ]
        rows.append(row)
    headers = ["Area", "Cases", "Failed", "Busy", "Case time" if smoke else "Time per round"]
    lines += table(headers, rows, "lrrrr")
    lines.append(rule())
    if not smoke:
        per: dict[tuple, list[float]] = {}
        for c in cs:
            for label, r in _ratios(c).items():
                per.setdefault((c["area"], label), []).append(r)
        if per:
            crow = [
                [area, label, f"{len(rs):,}", f"{statistics.median(rs):.2f}x", f"{sum(r < 1 for r in rs):,}"]
                for (area, label), rs in per.items()
            ]
            lines += table(["Area", "Compared with", "Cases", "Others / sd", "sd slower"], crow, "llrrr")
            lines.append(rule())
    problems = (
        [(c, c["failure"]) for c in stopped]
        + [(c, f"setup: {c['setup_error']}") for c in cs if c.get("setup_error") and not c.get("failure")]
        + [
            (c, f"{t['label']} ({t['threads'] or 'default'}): {t['error']}")
            for c in cs
            if not c.get("failure")
            for t in c["timings"]
            if t.get("min") is None and t.get("error") and t["error"] != "not measured"
        ]
        + [
            (c, f"check against {t['label']}: {t['check']['text']}")
            for c in cs
            for t in c["timings"]
            if t.get("check") and not t["check"]["ok"]
        ]
    )
    if problems:
        lines.append("Failures:")
        for c, why in problems[:12]:
            lines.append(f"  {c['area']} · {c['name']}{' [' + c['size'] + ']' if c.get('size') else ''}")
            lines.append(f"      {why.strip().splitlines()[-1][: WIDTH - 6]}")
        if len(problems) > 12:
            lines.append(f"  ... and {len(problems) - 12} more (see the run file)")
        lines.append(rule())
    notes = []
    if interrupted:
        notes.append(f"Interrupted. Resume with:\n    .venv/bin/python -m bench.speed run --resume {_show(out)}")
    if busy:
        notes.append(
            f"{len(busy)} cases ran while other processes kept the CPU busy (marked in the tables).\n"
            f"    Measure them again:\n    .venv/bin/python -m bench.speed run --rerun-busy {_show(out)}"
        )
    if not smoke and not interrupted:
        notes.append(
            "Others / sd is the median over the cases of comparator time / sd time, per\n"
            "    element, at sd's lowest thread count: above 1, sd is faster. 'sd slower'\n"
            "    counts the cases below 1."
        )
    if smoke:
        notes.append(
            "A check compares sd's result with the first Python comparator's and each R and\n"
            "    Boost one's (bench/speed/check.py); NaN shares that differ are reported, not failed."
        )
    notes.append(f"Results: {_show(out)}\n    Log:     {_show(log_path)}")
    lines.append("Notes:")
    lines += [f"[{k}] {n}" for k, n in enumerate(notes, 1)]
    return lines


# ---------------------------------------------------------------- the run


def _notify(title, text):
    if shutil.which("osascript") is None:
        return
    script = f'display notification "{text}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10, check=False)


def run(
    cases: list[Case],
    *,
    rounds: int = 1,
    threads: tuple[int, ...] | None = None,
    use_r: bool = True,
    out=None,
    label: str = "",
    limits: Limits | None = None,
    notify: bool = True,
    resume=None,
    progress: bool = True,
    keep_awake: bool = True,
    smoke: bool = False,
    quiet: str = "wait",
    jobs: int = 1,
) -> dict:
    """Time `cases` over `rounds` rounds; write and return the results dict.

    threads: restrict each case's thread counts to these (None: the case's own).
    resume: a run file to continue (its cases, rounds and options win).
    smoke: call everything once, untimed in effect (does every case run?); such a run
    cannot be promoted.
    jobs: processes a smoke run may use at once (workers and their R sessions); above 1 the
    cases are spread over that many lanes, each with one worker (half as many when R
    runs). Measuring runs ignore it: their cases run one at a time.
    quiet: "wait" for a quiet machine before starting (at most limits.wait_quiet_s), "no"
    (only report), "strict" (stop if it does not get quiet)."""
    limits = limits or Limits()
    lock = threading.RLock()  # the state, the run file and the log, shared by the lanes
    stop = threading.Event()
    if resume is not None:
        prev = results.load(resume)
        by_id = {c.id: c for c in cases}
        missing = [cid for cid in prev["options"]["case_ids"] if cid not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} cases of {resume} are no longer registered, e.g. {missing[0]}")
        cases = [by_id[cid] for cid in prev["options"]["case_ids"]]
        rounds = prev["options"]["rounds"]
        label = prev["options"].get("label", "")
        smoke = prev["options"].get("smoke", False)
        state = State.from_file(prev, cases)
        state.busy_cores = limits.busy_cores
        threads, use_r = state.threads, state.use_r
        start_round, start_index = prev["options"]["next"]
        out = resume
        run_id, env = prev["run"], prev["env"]
        started = datetime.datetime.fromisoformat(prev["options"]["started"])
        elapsed_before = prev["options"].get("elapsed_s", 0.0)
        env.setdefault("resumed", []).append(datetime.datetime.now().isoformat(timespec="seconds"))
    else:
        state = State(cases, threads, use_r, limits.busy_cores)
        start_round, start_index = 0, 0
        started = datetime.datetime.now()
        elapsed_before = 0.0
        git = harness.git_info(str(results.ROOT))
        env = {"git": git, "machine": harness.machine_info()}
        stamp = started.strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{git['commit'] or 'nogit'}{'+' if git['dirty'] else ''}"
        out = out or results.RUN_DIR / f"{stamp}{'-' + label if label else ''}.jsonl"
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = cli.log_file("speed")
    log = open(log_path, "a", buffering=1)  # noqa: SIM115  (the workers write to it too)
    runfile = results.RunFile(out, resume=resume is not None)

    con = Console(log, show=progress)
    say = con.print

    areas = sorted({c.area for c in cases})
    have_r = use_r and any(isinstance(r, RRef) for c in cases for r in c.refs) and shutil.which("R") is not None
    if use_r and not have_r and any(isinstance(r, RRef) for c in cases for r in c.refs):
        say("R not found: skipping the R comparators.")
    state.use_r = have_r or (resume is not None and state.use_r)
    lanes = max(1, jobs // (2 if state.use_r else 1)) if smoke else 1
    parallel = lanes > 1
    pairs = [(r, c.id) for r in range(rounds) for c in cases]
    first = start_round * len(cases) + start_index
    expected = expected_seconds(cases)
    eta = Eta(pairs, expected)
    t_run = time.monotonic()
    caff = None
    if keep_awake and shutil.which("caffeinate"):
        caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    load0 = proc.load1()
    env.setdefault("load1_at_start", round(load0, 2))
    quiet_check = None if smoke else wait_for_quiet(con, limits, quiet)
    if quiet_check is not None:
        env.setdefault("cpu_checks", []).append(quiet_check)
        if quiet_check.get("stopped"):
            log.close()
            runfile.close()
            if caff is not None:
                caff.terminate()
            return {"path": str(out), "interrupted": True, "stopped_busy": True, "cases": [], "log": str(log_path)}
    elif not smoke and load0 > limits.load_warn:
        busy = proc.busy_processes()
        say(f"Warning: the 1-minute load is {load0:.1f} (a quiet machine is wanted). Busiest: {', '.join(busy) or '-'}")
    con.lines(
        _header(
            cases,
            rounds,
            threads,
            state.use_r,
            have_r,
            smoke,
            lanes,
            jobs,
            limits,
            env,
            run_id,
            quiet_check,
            first,
            out,
            log_path,
        )  # fmt: skip
    )

    env["blas_control"] = (
        "one worker per BLAS thread count: sd at 1 thread and the 'follow' comparators at 1 run with "
        "VECLIB_MAXIMUM_THREADS=1; sd at its default thread count and the 'fixed' comparators run "
        "without it (Accelerate's default); R runs on one thread (OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1)"
    )
    options = {
        "rounds": rounds,
        "threads": list(threads) if threads else None,
        "r": state.use_r,
        "label": label,
        "smoke": smoke,
        "started": started.isoformat(timespec="seconds"),
        "limits": dataclasses.asdict(limits),
        "case_ids": [c.id for c in cases],
        "log": str(log_path),
    }
    if resume is None:
        runfile.start(run_id, env, dict(options, next=[0, 0], complete=False, elapsed_s=0.0))
    else:
        runfile.options({"log": str(log_path)})
        runfile.env(env)
    env_written = [json.dumps(env, sort_keys=True, default=str)]

    def snapshot(next_round, next_index, cids=(), full=False):
        """Append the cases `cids` (their records so far), the env if it changed, and the
        progress; with full=True return the run as one dict."""
        for cid in cids:
            runfile.case(state.record(by_id[cid]))
        now = json.dumps(env, sort_keys=True, default=str)
        if now != env_written[0]:
            runfile.env(env)
            env_written[0] = now
        progress = {
            "elapsed_s": round(elapsed_before + time.monotonic() - t_run, 1),
            "next": [next_round, next_index],
            "complete": next_round >= rounds,
        }
        runfile.options(progress)
        if not full:
            return None
        return {
            "schema": results.SCHEMA,
            "run": run_id,
            "env": env,
            "options": dict(options, **progress),
            "cases": [state.record(c) for c in cases],
        }

    workers: dict = {}  # BLAS thread setting (None: default) -> WorkerHandle
    by_id = {c.id: c for c in cases}

    def worker_for(ws, key):
        w = ws.get(key)
        if w is not None and (w.served >= limits.restart_every or w.p.poll() is not None):
            w.close()
            w = None
        if w is None:
            w = WorkerHandle(areas, log, limits, smoke=smoke, blas_threads=key)
            ws[key] = w
            if key is None and "versions" not in env:
                pkgs = sorted({p for c in cases for r in c.refs if isinstance(r, RRef) for p in r.packages})
                reply, status, _ = w.request(
                    {"cmd": "env", "r": state.use_r, "r_packages": pkgs if state.use_r else []},
                    limits.startup_timeout_s,
                )
                if status == "ok":
                    with lock:
                        env.update(reply["env"])
        return w

    def kill_worker(ws, key):
        w = ws.pop(key, None)
        if w is not None:
            w.kill()

    def trim_idle(ws, active):
        """Close the idle workers that hold much memory (they start again when needed)."""
        for key in [k for k in ws if k != active]:
            if proc.tree_rss(ws[key].p.pid) > limits.idle_rss_bytes:
                ws.pop(key).close()

    def parts(case):
        """(BLAS key, sd thread counts, measure the fixed comparators) per worker, in order. A
        smoke run calls everything in one worker: nothing is timed, so the BLAS setting does
        not matter; one worker per lane keeps the process count down, and the results check
        has sd's result next to every comparator's."""
        ths = [t for t in case.threads if threads is None or t in threads]
        if smoke:
            return [(None, ths, True)]
        out = {}
        for t in ths:
            out.setdefault(None if t == DEFAULT_THREADS else t, []).append(t)
        has_fixed = any(r.threads == "fixed" for r in case.refs)
        keys = sorted(k for k in out if k is not None)
        if None in out or has_fixed:
            keys.append(None)
        return [(k, out.get(k, []), k is None) for k in keys]

    def run_case(cid, ws) -> tuple[str, float]:
        """One round of one case in the workers `ws` (BLAS key -> WorkerHandle); records the
        result in `state` and returns (status, wall seconds)."""
        with lock:
            state.load[cid] = max(state.load[cid], proc.load1())
        meter = None if smoke else proc.CpuMeter().start()
        t0 = time.monotonic()
        status, seconds, key, peak = "ok", 0.0, None, 0
        for key, ths, fixed in parts(by_id[cid]):
            budget = limits.case_timeout_s - (time.monotonic() - t0)
            trim_idle(ws, key)
            try:
                worker = worker_for(ws, key)
            except WorkerStartError as e:
                status = e.status
                break
            reply, status, peak = worker.request(
                {"cmd": "case", "id": cid, "threads": ths, "use_r": state.use_r, "fixed": fixed},
                max(budget, 1.0),
            )
            worker.served += 1
            with lock:
                state.rss[cid] = max(state.rss[cid], peak)
                if status == "ok":
                    state.add_reply(reply)
                    seconds += reply.get("seconds", 0.0)
            if status != "ok":
                break
            if reply.get("setup_error"):
                con.event("setup", case_label(by_id[cid]), reply["setup_error"].strip().splitlines()[-1])
            for label_, t, err in reply["errors"]:
                th = "default threads" if t is None else f"{t} thread{'s' if t != 1 else ''}"
                con.event("failed", case_label(by_id[cid]), f"{label_} ({th}): {err}")
        dt = time.monotonic() - t0
        other = meter.stop() if meter is not None and status == "ok" else None
        if status != "ok":
            kill_worker(ws, key)
            if stop.is_set():  # interrupted: the case is not done, not failed
                return status, dt
        with lock:
            if other is not None:
                state.other[cid] = max(state.other.get(cid, 0.0), other)
            if status == "ok":
                state.seconds[cid] += seconds or dt
                state.rounds_done[cid] += 1
            else:
                why = {
                    "timeout": f"timeout: over {fmt_duration(limits.case_timeout_s)}",
                    "memory": f"memory: over {limits.rss_bytes / 2**30:g} GB (peak {peak / 2**30:.1f} GB)",
                    "died": "the worker died",
                }[status]
                state.failure[cid] = why
                if parallel and status == "memory":
                    con.event("deferred", case_label(by_id[cid]), f"{why}; runs again alone at the end")
                else:
                    con.event("stopped", case_label(by_id[cid]), f"{why}; starting a new worker")
        return status, dt

    def elapsed():
        return fmt_duration(elapsed_before + time.monotonic() - t_run)

    next_index = [first]  # the first (round, case) not done yet, for the run file

    def run_parallel():
        """The pairs from `first` on, spread over `lanes` lanes. A case stopped for memory is
        run again at the end on its own: the limit counts every lane's worker, so a case may
        be stopped for another's memory."""
        todo = list(range(first, len(pairs)))
        total = len(todo)
        done: set[int] = set()
        retry: list[int] = []
        con.start(total, "Smoke")
        t_lanes = time.monotonic()

        def finish(idx, status, dt):
            with lock:
                if stop.is_set() and status != "ok":
                    return
                done.add(idx)
                eta.done(pairs[idx][1], dt)
                while next_index[0] in done:
                    next_index[0] += 1
                mark = "" if status == "ok" else f" ({status})"
                spent = time.monotonic() - t_lanes
                left = spent / len(done) * (total - len(done))
                what = case_label(by_id[pairs[idx][1]]) + mark
                con.step(
                    len(done),
                    "Smoke",
                    what,
                    elapsed_before + time.monotonic() - t_run,
                    left,
                    f"{len(done)}/{total} {what}  ({elapsed()} elapsed)",
                )
                snapshot(*divmod(next_index[0], len(cases)), cids=(pairs[idx][1],))

        def lane(ws):
            while not stop.is_set():
                with lock:
                    if not todo:
                        return
                    idx = todo.pop(0)
                status, dt = run_case(pairs[idx][1], ws)
                if status == "memory" and not stop.is_set():
                    with lock:
                        state.failure.pop(pairs[idx][1], None)
                        retry.append(idx)
                    continue
                finish(idx, status, dt)
                w = ws.get(None)  # macOS keeps freed blocks: do not sit on them next to the other lanes
                if w is not None and proc.tree_rss(w.p.pid) > limits.idle_rss_bytes:
                    ws.pop(None).close()

        lane_ws[:] = [{} for _ in range(min(lanes, len(todo)))]
        threads_ = [threading.Thread(target=lane, args=(ws,), daemon=True) for ws in lane_ws]
        for t in threads_:
            t.start()
        while any(t.is_alive() for t in threads_):
            for t in threads_:
                t.join(timeout=0.5)
        for ws in lane_ws:
            for key in list(ws):
                ws.pop(key).close()
        if retry:
            say(f"Running {len(retry)} cases again one at a time (stopped at the shared memory limit).")
            lane_ws[:] = [{}]
            for idx in sorted(retry):
                if stop.is_set():
                    break
                status, dt = run_case(pairs[idx][1], lane_ws[0])
                finish(idx, status, dt)
        with lock:
            return snapshot(*divmod(next_index[0], len(cases)), full=True)

    def new_round(i) -> bool:
        """At a round boundary of a run with several rounds: the next case starts in new
        workers (each round's minimum then comes from other processes)."""
        return rounds > 1 and i > first and i % len(cases) == 0

    data = None
    interrupted = False
    i = first
    lane_ws: list[dict] = [workers]
    try:
        if not parallel:
            con.start(len(pairs), _round_desc(pairs[i][0] if i < len(pairs) else rounds - 1, rounds, smoke))
            while i < len(pairs):
                rnd, cid = pairs[i]
                if new_round(i):
                    for key in list(workers):
                        workers.pop(key).close()
                left = eta.remaining(pairs[i:])
                what = case_label(by_id[cid])
                eta_s = f"~{fmt_duration(left)} left" if left is not None else "time left unknown"
                con.step(
                    i,
                    _round_desc(rnd, rounds, smoke),
                    what,
                    elapsed_before + time.monotonic() - t_run,
                    left,
                    f"[{rnd + 1}/{rounds}] {i % len(cases) + 1}/{len(cases)} {what}  ({elapsed()} elapsed, {eta_s})",
                )
                status, dt = run_case(cid, workers)
                if status == "ok" and not smoke:
                    with lock:
                        rec = state.record(by_id[cid])
                    con.print(_result_line(con, by_id[cid], rec, rnd, rounds))
                eta.done(cid, dt)
                i += 1
                snapshot(*divmod(i, len(cases)), cids=(cid,))
            con.step(i, _round_desc(rounds - 1, rounds, smoke), "", elapsed_before + time.monotonic() - t_run, 0, "")
        else:
            data = run_parallel()
            i = len(pairs)
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        for ws in lane_ws:
            for key in list(ws):
                kill_worker(ws, key)
        with lock:
            nxt = divmod(next_index[0] if parallel else i, len(cases))
            data = snapshot(*nxt, full=True)
    finally:
        con.close()
        for ws in lane_ws:
            for w in list(ws.values()):
                w.close()
        if caff is not None:
            caff.terminate()
    if data is None:
        data = snapshot(*divmod(len(pairs), len(cases)), full=True)
    runfile.close()
    total = elapsed_before + time.monotonic() - t_run
    failed = sum(1 for c in data["cases"] if case_failed(c))
    con.lines(_summary(data, total, interrupted, out, log_path))
    if not interrupted and notify and total >= limits.notify_after_s:
        _notify("stardust speed run", f"Done: {len(cases)} cases, {fmt_duration(total)}, {failed} failed")
    log.close()
    data["path"] = str(out)
    data["log"] = str(log_path)
    data["interrupted"] = interrupted
    data["failed"] = failed
    return data


def case_failed(c: dict) -> bool:
    """Whether a run's case failed: stopped (timeout, memory, a dead worker), its setup or a
    call raised, or (smoke) a results check failed."""
    if c.get("failure") or c.get("setup_error"):
        return True
    for t in c["timings"]:
        if t.get("min") is None and t.get("error") and t["error"] != "not measured":
            return True
        if t.get("check") and not t["check"]["ok"]:
            return True
    return False
