"""The measuring process. The supervisor (run.py) starts it, sends one request per line
on stdin and reads one JSON reply per line from the protocol stream; it kills the
whole process group (this process and its R session) when a case passes its time or
memory limit, and starts a new worker for the next case.

Anything the measured code prints to stdout goes to stderr (the log): the protocol
stream is a copy of the original stdout, taken before fd 1 is redirected.

Requests:
    {"cmd": "env", "r": bool, "r_packages": [...]} -> {"env": {...}}
    {"cmd": "case", "id": ..., "threads": [..] | null, "use_r": bool, "fixed": bool}
        -> {"id": ..., "samples": [[label, lib, threads, times, calls, nan, abs_err], ...],
            "errors": [[label, threads, text], ...], "setup_error": text | null, "seconds": s,
            "checks": [[label, ok, text], ...]}
        (abs_err: the warm-up result's error against the case's reference, or null)
    {"cmd": "quit"}

    python -m bench.speed.worker AREA [AREA ...] [--smoke]

The supervisor starts one worker per BLAS thread setting: Accelerate (numpy's, scipy's and
sd's BLAS/LAPACK) reads VECLIB_MAXIMUM_THREADS once per process, and threadpoolctl cannot
change it. A request names the sd thread counts this worker measures (with the "follow"
comparators at the same counts) and whether it also measures the "fixed" comparators and R.

nan in a sample is the NaN share of the warm-up call's result (None when not numeric). A
smoke worker also compares the results (check.py): sd's against its first Python comparator
and every R and Boost comparator, unless the case's Check skips it.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback

from . import check, harness, registry
from .registry import RRef


class Worker:
    def __init__(self, areas, smoke=False):
        self.smoke = smoke
        self.cases = {c.id: c for c in registry.load(areas)}
        self._r = None
        self._r_failed = None

    def r_session(self):
        if self._r is None and self._r_failed is None:
            try:
                from .rsession import RSession

                self._r = RSession()
            except Exception as e:  # noqa: BLE001
                self._r_failed = f"{type(e).__name__}: {e}"
        return self._r

    def env(self, r_packages, use_r: bool = False) -> dict:
        import stardust as sd

        env = {
            "versions": harness.library_versions(),
            "blas": harness.blas_info(),
            "sd_default_threads": sd.get_num_threads(),
        }
        if use_r or r_packages:
            r = self.r_session()
            if r is not None:
                info = r.info()
                env["versions"].update(r.version())
                env["r_packages"] = r.package_versions(r_packages)
                env["r_blas"] = {k: info.get(k) for k in ("blas", "lapack", "threads")}
            else:
                env["r_error"] = self._r_failed
        return env

    def _one(self, case, label, threads, make, data, blas, rref=None, keep=None):
        """One measurement: (Sample | None, error | None). keep: a list the result of the
        warm-up call goes to (the smoke run's results check)."""
        # The batches run with gc off: collect the cycles the previous measurement left
        # (statsmodels' results hold GBs in cycles at n = 10^6) before the next one.
        gc.collect()
        try:
            if rref is not None:
                r = self.r_session()
                if r is None:
                    raise RuntimeError(self._r_failed or "no R session")
                arrays, sets = _r_inputs(rref, data)
                timing = dict(case.timing, batches=1, min_batches=1, target_s=0.0) if self.smoke else case.timing
                sample = r.time(rref.expr, arrays, setup=rref.setup, packages=rref.packages, sets=sets, **timing)
                if keep is not None or case.abs_err is not None:
                    v = r.value(
                        rref.expr, arrays, setup=rref.setup, packages=rref.packages, sets=sets, value=rref.value
                    )
                    if keep is not None:
                        keep.append(v)
                    sample.abs_err = _abs_err(case, data, v)
                return sample, None
            with harness.sd_threads(threads if label == "sd" else None), harness.blas_threads(blas):
                fn = make(data)
                got: list = []
                sample = harness.measure(fn, **case.timing, smoke=self.smoke, keep=got)
                if got:
                    sample.nan = check.nan_share(got[0])
                    sample.abs_err = _abs_err(case, data, got[0])
                    if keep is not None:
                        keep.append(got[0])
                return sample, None
        except Exception as e:  # noqa: BLE001  (one failing call must not stop the case)
            return None, f"{type(e).__name__}: {e}".strip()[:500]

    def case(self, cid, threads, use_r, fixed=True) -> dict:
        """One round of one case. numpy's error state is restored afterwards and the
        case's teardown(data) runs, so a case's global settings do not leak into the next."""
        import numpy as np

        err = np.geterr()
        try:
            return self._case(cid, threads, use_r, fixed)
        finally:
            np.seterr(**err)

    def _case(self, cid, threads, use_r, fixed) -> dict:
        case = self.cases[cid]
        t0 = time.perf_counter()
        out = {"id": cid, "samples": [], "errors": [], "setup_error": None}

        def add(label, lib, t, res):
            sample, err = res
            if sample is not None:
                out["samples"].append([label, lib, t, sample.times, sample.calls, sample.nan, sample.abs_err])
            else:
                out["errors"].append([label, t, err])

        try:
            data = case.setup()
        except Exception:  # noqa: BLE001
            out["setup_error"] = traceback.format_exc(limit=4)[-800:]
            out["seconds"] = time.perf_counter() - t0
            return out
        checked = _checked(case, use_r) if self.smoke else set()
        mine: list = []  # sd's result (the first thread count's), for the check
        theirs: dict[str, list] = {}

        def keep_for(label):
            if label == "sd":
                return mine if not mine else None
            if label in checked and label not in theirs:
                theirs[label] = []
                return theirs[label]
            return None

        ths = [t for t in case.threads if threads is None or t in threads]
        for t in ths:
            add("sd", "sd", t, self._one(case, "sd", t, case.sd, data, None, keep=keep_for("sd")))
            for ref in case.refs:
                if ref.threads == "follow" and not isinstance(ref, RRef):
                    k = keep_for(ref.label)
                    add(ref.label, ref.library, t, self._one(case, ref.label, t, ref.make, data, t, keep=k))
        for ref in case.refs:
            if ref.threads != "fixed" or not fixed:
                continue
            k = keep_for(ref.label)
            if isinstance(ref, RRef):
                if use_r:
                    add(ref.label, "R", None, self._one(case, ref.label, None, None, data, None, rref=ref, keep=k))
            else:
                add(ref.label, ref.library, None, self._one(case, ref.label, None, ref.make, data, None, keep=k))
        if self.smoke:
            out["checks"] = _compare(case, mine, theirs)
        if case.teardown is not None:
            try:
                case.teardown(data)
            except Exception as e:  # noqa: BLE001
                out["errors"].append(["teardown", None, f"{type(e).__name__}: {e}"[:500]])
        del data
        gc.collect()
        out["seconds"] = time.perf_counter() - t0
        return out

    def close(self):
        if self._r is not None:
            self._r.close()


def _abs_err(case, data, result) -> float | None:
    """The case's abs_err of a result (None without one); it raises like the call would, so
    a reference that does not fit shows as the measurement's error."""
    if case.abs_err is None:
        return None
    e = case.abs_err(data, result)
    return None if e is None else float(e)


def _r_inputs(rref, data):
    """(arrays, sets) for the R session: a rotated case's sets as dicts of R names."""
    if rref.sets:
        return {}, [{rname: one[key] for key, rname in rref.data.items()} for one in data[rref.sets]]
    return {rname: data[key] for key, rname in rref.data.items()}, None


def _checked(case, use_r) -> set[str]:
    """The comparator labels the smoke run's check compares with sd: the first Python
    comparator, every R one (when R runs) and every Boost one, but none with `differs`."""
    if case.check.skip:
        return set()
    out = set()
    python = [r for r in case.refs if not isinstance(r, RRef) and r.library != "Boost"]
    if python and not python[0].differs:
        out.add(python[0].label)
    for r in case.refs:
        if r.differs:
            continue
        if (isinstance(r, RRef) and use_r) or (not isinstance(r, RRef) and r.library == "Boost"):
            out.add(r.label)
    return out


def _compare(case, mine, theirs) -> list:
    """[[label, ok, text], ...] for the comparators kept for the check."""
    import numpy as np

    ck = case.check
    rows = []
    if not mine:
        return [[label, False, "no sd result to compare"] for label in theirs if theirs[label]]
    try:
        a = check.normalise(ck.sd(mine[0]) if ck.sd else mine[0])
    except Exception as e:  # noqa: BLE001
        return [[label, False, f"sd's result: {type(e).__name__}: {e}"[:300]] for label in theirs]
    ref_by_label = {r.label: r for r in case.refs}
    for label, got in theirs.items():
        if not got:
            continue  # the comparator failed: its error is reported already
        try:
            b = check.normalise(ck.ref(label, got[0]) if ck.ref else got[0])
            k = ref_by_label[label].n
            fa = a
            if k and isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                fa = a.ravel()[: b.size]  # a comparator timed on a sample of the inputs
            res = check.compare(fa, b, rtol=ck.rtol, atol=ck.atol, agree=ck.agree)
            rows.append([label, res["ok"], res["text"]])
        except Exception as e:  # noqa: BLE001
            rows.append([label, False, f"{type(e).__name__}: {e}"[:300]])
    return rows


def main(argv) -> None:
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # the measured code's prints go to the log, not the protocol
    sys.stdout = sys.stderr
    smoke = "--smoke" in argv
    w = Worker([a for a in argv if a != "--smoke"], smoke=smoke)
    proto.write(json.dumps({"ready": True, "cases": len(w.cases)}) + "\n")
    try:
        for line in sys.stdin:
            req = json.loads(line)
            if req["cmd"] == "quit":
                break
            if req["cmd"] == "env":
                reply = {"env": w.env(req.get("r_packages") or [], use_r=req.get("r", False))}
            elif req["cmd"] == "case":
                reply = w.case(req["id"], req.get("threads"), req.get("use_r", True), req.get("fixed", True))
            else:
                reply = {"error": f"unknown command {req['cmd']!r}"}
            proto.write(json.dumps(reply) + "\n")
    finally:
        w.close()


if __name__ == "__main__":
    main(sys.argv[1:])
