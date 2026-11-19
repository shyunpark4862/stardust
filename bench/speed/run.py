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
  replaced every `restart_every` cases, so memory a case leaks does not pile up.
- After every case the run file is rewritten (with the raw batch times), so an
  interrupted run loses at most one case: `run --resume FILE` continues it.
- Progress lines show the elapsed time and the expected rest, from the baselines'
  seconds per case (scaled by how this run compares so far).
- Before the run, the CPU time other processes use is measured for a few seconds
  (proc.idle_check); when it is above `quiet_cores`, the busiest processes are listed and
  the run waits for a quiet machine (up to `wait_quiet_s`; then it starts with a warning,
  or stops with quiet="strict"). During the run, the cores other processes kept busy are
  recorded per case (proc.CpuMeter, no process started); a case above `busy_cores` is
  marked busy, in the progress line, the run file, the tables and compare, and
  `run --rerun-busy RUN.json` measures just those cases again.
- caffeinate keeps the machine awake while the supervisor lives; the log file next to
  the run file gets the progress lines and the worker's stderr; the 1-minute load is
  recorded per case; a macOS notification says when a long run is done.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import select
import shutil
import signal
import subprocess
import sys
import time

from . import harness, proc, results
from .registry import Case, RRef


@dataclasses.dataclass
class Limits:
    case_timeout_s: float = 15 * 60  # one case, one round (all threads and comparators)
    rss_bytes: int = 16 * 2**30  # the worker and its children (R)
    poll_s: float = 2.0
    restart_every: int = 40  # cases per worker
    startup_timeout_s: float = 30 * 60  # importing stardust may rebuild the C++ core
    load_warn: float = 3.0  # without CPU counters: warn when the 1-minute load is above this
    quiet_cores: float = 1.0  # before the run: other processes may keep at most this many cores busy
    idle_window_s: float = 3.0  # ... measured over this window
    wait_quiet_s: float = 10 * 60  # wait this long for a quiet machine before starting anyway
    wait_poll_s: float = 10.0
    busy_cores: float = 1.0  # a case during which other processes used more is marked busy
    notify_after_s: float = 60.0  # a notification only for runs at least this long


# ---------------------------------------------------------------- the worker handle


class WorkerHandle:
    def __init__(self, areas, log, limits: Limits, smoke: bool = False):
        self.limits = limits
        self.p = subprocess.Popen(
            [sys.executable, "-m", "bench.speed.worker", *areas, *(["--smoke"] if smoke else [])],
            cwd=str(results.ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            start_new_session=True,
        )
        self.buf = b""
        self.served = 0
        _, status, _ = self._read(limits.startup_timeout_s)
        if status != "ok":
            self.kill()
            raise RuntimeError(f"the worker did not start ({status}); see the log")

    def _read(self, timeout_s):
        """One reply line: (dict | None, "ok" | "timeout" | "memory" | "died", peak RSS)."""
        t0 = time.monotonic()
        peak = 0
        fd = self.p.stdout.fileno()
        while b"\n" not in self.buf:
            rss = proc.tree_rss(self.p.pid)
            peak = max(peak, rss)
            if rss > self.limits.rss_bytes:
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
        except (BrokenPipeError, OSError):
            return None, "died", 0
        return self._read(timeout_s)

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.write(b'{"cmd": "quit"}\n')
                self.p.stdin.flush()
                self.p.wait(timeout=30)
            except (OSError, subprocess.SubprocessError):
                self.kill()

    def kill(self):
        """Kill the worker's process group: the worker and the R session it started."""
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.SubprocessError:
            pass


# ---------------------------------------------------------------- run state


class State:
    """Everything a run file holds, built up case by case."""

    def __init__(self, cases: list[Case], threads, use_r, busy_cores: float = 1.0):
        self.busy_cores = busy_cores
        self.cases = cases
        self.threads = threads
        self.use_r = use_r
        self.samples: dict[str, dict[tuple, harness.Sample]] = {c.id: {} for c in cases}
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
        for label, lib, t, times, calls in reply["samples"]:
            key = (label, t)
            s = harness.Sample(list(times), calls)
            old = self.samples[cid].get(key)
            self.samples[cid][key] = s if old is None else old.merged(s)
            self.libs[cid][label] = lib
            self.errors[cid].pop(key, None)
        for label, t, err in reply["errors"]:
            if (label, t) not in self.samples[cid]:
                self.errors[cid][(label, t)] = err
        if reply.get("setup_error"):
            self.setup_error[cid] = reply["setup_error"]

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
                rec.update(min=s.min, median=s.median, noise=round(s.noise, 4), batches=len(s.times),
                           calls=s.calls, times=s.times)
            else:
                rec.update(min=None, error=self.errors[cid].get((label, t)) or self.failure.get(cid) or "not measured")
            timings.append(rec)
        rd = self.rounds_done[cid]
        out = {"id": cid, "area": case.area, "group": case.group, "name": case.name, "size": case.size,
               "n": case.n, "cost": list(case.cost) if case.cost else None, "note": case.note,
               "rounds": rd, "seconds_per_round": round(self.seconds[cid] / rd, 2) if rd else None,
               "load1": round(self.load[cid], 2), "peak_rss_mb": round(self.rss[cid] / 2**20),
               "other_cpu": round(self.other[cid], 2) if cid in self.other else None,
               "busy": self.other.get(cid, 0.0) > self.busy_cores,
               "timings": timings}
        if cid in self.failure:
            out["failure"] = self.failure[cid]
        if cid in self.setup_error:
            out["setup_error"] = self.setup_error[cid]
        return out

    @classmethod
    def from_file(cls, data, cases):
        st = cls(cases, tuple(data["options"]["threads"]) if data["options"].get("threads") else None,
                 data["options"].get("r", True))
        for c in data["cases"]:
            cid = c["id"]
            if cid not in st.samples:
                continue
            for t in c["timings"]:
                if t.get("times"):
                    st.samples[cid][(t["label"], t["threads"])] = harness.Sample(t["times"], t.get("calls", 1))
                    st.libs[cid][t["label"]] = t["lib"]
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


def fmt_duration(s: float) -> str:
    s = max(0, int(s))
    if s < 60:
        return f"{s}초"
    if s < 3600:
        return f"{s // 60}분 {s % 60:02d}초"
    return f"{s // 3600}시간 {s % 3600 // 60:02d}분"


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


def wait_for_quiet(say, limits: Limits, mode: str = "wait") -> dict | None:
    """Measure the CPU other processes use; when it is above limits.quiet_cores, list the
    busiest and (mode "wait" or "strict") wait for it to drop. None when the counters
    cannot be read. Returns {"other_cores", "busiest", "waited_s", "quiet", "stopped"}."""
    other, top = proc.idle_check(limits.idle_window_s)
    if other is None:
        return None
    t0 = time.monotonic()
    first = (other, top)
    if other <= limits.quiet_cores:
        say(f"CPU: 다른 프로세스가 {other:.1f}코어를 쓴다(기준 {limits.quiet_cores:g}코어 이하). 시작한다")
        return {"other_cores": round(other, 2), "busiest": top, "waited_s": 0, "quiet": True, "stopped": False}
    say(f"경고: 다른 프로세스가 CPU {other:.1f}코어를 쓴다(기준 {limits.quiet_cores:g}코어). "
        f"바쁜 프로세스: {', '.join(top) or '(읽을 수 없는 프로세스)'}")
    if mode == "no":
        return {"other_cores": round(other, 2), "busiest": top, "waited_s": 0, "quiet": False, "stopped": False}
    say(f"조용해질 때까지 최대 {fmt_duration(limits.wait_quiet_s)} 기다린다(--no-wait: 바로 시작)")
    last_report = time.monotonic()
    while time.monotonic() - t0 < limits.wait_quiet_s:
        time.sleep(limits.wait_poll_s)
        other, top = proc.idle_check(limits.idle_window_s)
        if other is not None and other <= limits.quiet_cores:
            waited = time.monotonic() - t0
            say(f"CPU: 다른 프로세스 {other:.1f}코어로 조용해졌다({fmt_duration(waited)} 기다림). 시작한다")
            return {"other_cores": round(other, 2), "busiest": top, "waited_s": round(waited), "quiet": True,
                    "stopped": False, "at_first": round(first[0], 2)}
        if time.monotonic() - last_report >= 60:
            say(f"  아직 바쁘다: {other:.1f}코어 ({', '.join(top[:3])})")
            last_report = time.monotonic()
    waited = time.monotonic() - t0
    stopped = mode == "strict"
    say(f"{fmt_duration(waited)} 기다렸지만 바쁘다({other:.1f}코어). "
        + ("멈춘다(--strict)." if stopped else "경고를 남기고 시작한다. 결과의 잡음이 클 수 있다."))
    return {"other_cores": round(other, 2), "busiest": top, "waited_s": round(waited), "quiet": False,
            "stopped": stopped, "at_first": round(first[0], 2)}


# ---------------------------------------------------------------- the run


def _notify(title, text):
    if shutil.which("osascript") is None:
        return
    script = f'display notification "{text}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10, check=False)


def run(cases: list[Case], *, rounds: int = 1, threads: tuple[int, ...] | None = None, use_r: bool = True,
        out=None, label: str = "", limits: Limits | None = None, notify: bool = True, resume=None,
        progress: bool = True, keep_awake: bool = True, smoke: bool = False, quiet: str = "wait") -> dict:
    """Time `cases` over `rounds` rounds; write and return the results dict.

    threads: restrict each case's thread counts to these (None: the case's own).
    resume: a run file to continue (its cases, rounds and options win).
    smoke: call everything once, untimed in effect (does every case run?); such a run
    cannot be promoted.
    quiet: "wait" for a quiet machine before starting (at most limits.wait_quiet_s), "no"
    (only report), "strict" (stop if it does not get quiet)."""
    limits = limits or Limits()
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
        out = out or results.RUN_DIR / f"{stamp}{'-' + label if label else ''}.json"
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log")
    log = open(log_path, "a", buffering=1)  # noqa: SIM115  (the workers write to it too)

    def say(msg):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        log.write(f"{stamp} {msg}\n")
        if progress:
            harness.eprint(msg)

    areas = sorted({c.area for c in cases})
    have_r = use_r and any(isinstance(r, RRef) for c in cases for r in c.refs) and shutil.which("R") is not None
    if use_r and not have_r and any(isinstance(r, RRef) for c in cases for r in c.refs):
        say("R이 없어 R 비교는 건너뛴다")
    state.use_r = have_r or (resume is not None and state.use_r)
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
    check = None if smoke else wait_for_quiet(say, limits, quiet)
    if check is not None:
        env.setdefault("cpu_checks", []).append(check)
        if check.get("stopped"):
            log.close()
            if caff is not None:
                caff.terminate()
            return {"path": str(out), "interrupted": True, "stopped_busy": True, "cases": []}
    elif not smoke and load0 > limits.load_warn:
        busy = proc.busy_processes()
        say(f"경고: 1분 부하가 {load0:.1f}이다(조용한 기계를 바란다). 바쁜 프로세스: {', '.join(busy) or '없음'}")
    if first == 0:
        est, unknown = estimate(cases, rounds)
        tail = f", 기준선에 없는 사례 {unknown}개" if unknown else ""
        say(f"사례 {len(cases)}개 × {rounds}라운드, 예상 {fmt_duration(est)}{tail}. 결과 {out}, 로그 {log_path}")

    def snapshot(next_round, next_index):
        data = {
            "schema": results.SCHEMA, "run": run_id, "env": env,
            "options": {"rounds": rounds, "threads": list(threads) if threads else None, "r": state.use_r,
                        "label": label, "smoke": smoke, "started": started.isoformat(timespec="seconds"),
                        "elapsed_s": round(elapsed_before + time.monotonic() - t_run, 1),
                        "next": [next_round, next_index], "complete": next_round >= rounds,
                        "limits": dataclasses.asdict(limits), "case_ids": [c.id for c in cases]},
            "cases": [state.record(c) for c in cases],
        }
        results.save(data, out)
        return data

    worker = None
    data = None
    failures = 0
    interrupted = False
    i = first
    try:
        while i < len(pairs):
            rnd, cid = pairs[i]
            if worker is None or worker.served >= limits.restart_every or worker.p.poll() is not None:
                if worker is not None:
                    worker.close()
                worker = WorkerHandle(areas, log, limits, smoke=smoke)
                if "versions" not in env:
                    pkgs = sorted({p for c in cases for r in c.refs if isinstance(r, RRef) for p in r.packages})
                    reply, status, _ = worker.request({"cmd": "env", "r_packages": pkgs if state.use_r else []},
                                                      limits.startup_timeout_s)
                    if status == "ok":
                        env.update(reply["env"])
            left = eta.remaining(pairs[i:])
            say(f"[{rnd + 1}/{rounds}] {i % len(cases) + 1}/{len(cases)} {cid}  "
                f"(경과 {fmt_duration(elapsed_before + time.monotonic() - t_run)}, "
                f"남은 예상 {fmt_duration(left) if left is not None else '?'})")
            state.load[cid] = max(state.load[cid], proc.load1())
            meter = proc.CpuMeter().start()
            t0 = time.monotonic()
            reply, status, peak = worker.request({"cmd": "case", "id": cid, "threads": list(threads) if threads else None,
                                                  "use_r": state.use_r}, limits.case_timeout_s)
            worker.served += 1
            dt = time.monotonic() - t0
            state.rss[cid] = max(state.rss[cid], peak)
            other = meter.stop() if status == "ok" else None
            if other is not None:
                state.other[cid] = max(state.other.get(cid, 0.0), other)
                if other > limits.busy_cores:
                    say(f"  바빴다: 다른 프로세스가 평균 {other:.1f}코어를 썼다 (--rerun-busy로 다시 잴 수 있다)")
            if status == "ok":
                state.add_reply(reply)
                state.seconds[cid] += reply.get("seconds", dt)
                state.rounds_done[cid] += 1
                if reply.get("setup_error"):
                    say(f"  준비 실패: {reply['setup_error'].strip().splitlines()[-1]}")
                for label_, t, err in reply["errors"]:
                    say(f"  {label_} ({t if t is not None else '기본'} 스레드) 실패: {err}")
            else:
                why = {"timeout": f"제한 시간 {fmt_duration(limits.case_timeout_s)} 초과",
                       "memory": f"메모리 {limits.rss_bytes / 2**30:.0f} GB 초과(최대 {peak / 2**30:.1f} GB)",
                       "died": "작업 프로세스가 죽었다"}[status]
                state.failure[cid] = why
                failures += 1
                say(f"  중단: {why}. 작업 프로세스를 새로 띄운다")
                worker.kill()
                worker = None
            eta.done(cid, dt)
            i += 1
            data = snapshot(*divmod(i, len(cases)))
    except KeyboardInterrupt:
        interrupted = True
        if worker is not None:
            worker.kill()
            worker = None
        nxt = divmod(i, len(cases))
        data = snapshot(*nxt)
        say(f"중단했다. 이어 달리기: .venv/bin/python -m bench.speed run --resume {out}")
    finally:
        if worker is not None:
            worker.close()
        if caff is not None:
            caff.terminate()
    if data is None:
        data = snapshot(*divmod(len(pairs), len(cases)))
    total = elapsed_before + time.monotonic() - t_run
    errs = sum(1 for c in data["cases"] for t in c["timings"] if t.get("min") is None)
    if not interrupted:
        say(f"끝: 사례 {len(cases)}개, {fmt_duration(total)}, 실패한 측정 {errs}개, 중단된 사례 {failures}개. {out}")
        n_busy = sum(1 for c in data["cases"] if c.get("busy"))
        if n_busy:
            say(f"바쁠 때 잰 사례 {n_busy}개. 다시 재기: .venv/bin/python -m bench.speed run --rerun-busy {out}")
        if notify and total >= limits.notify_after_s:
            _notify("stardust 속도 측정", f"끝났다: 사례 {len(cases)}개, {fmt_duration(total)}, 실패 {errs}개")
    log.close()
    data["path"] = str(out)
    data["interrupted"] = interrupted
    return data
