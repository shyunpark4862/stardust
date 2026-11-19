"""The measuring process. The supervisor (run.py) starts it, sends one request per line
on stdin and reads one JSON reply per line from the protocol stream; it kills the
whole process group (this process and its R session) when a case passes its time or
memory limit, and starts a new worker for the next case.

Anything the measured code prints to stdout goes to stderr (the log): the protocol
stream is a copy of the original stdout, taken before fd 1 is redirected.

Requests:
    {"cmd": "env", "r_packages": [...]}         -> {"env": {...}}
    {"cmd": "case", "id": ..., "threads": [..] | null, "use_r": bool}
        -> {"id": ..., "samples": [[label, lib, threads, times, calls], ...],
            "errors": [[label, threads, text], ...], "setup_error": text | null, "seconds": s}
    {"cmd": "quit"}

    python -m bench.speed.worker AREA [AREA ...] [--smoke]
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback

from . import harness, registry
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

    def env(self, r_packages) -> dict:
        import stardust as sd

        env = {"versions": harness.library_versions(), "blas": harness.blas_info(),
               "sd_default_threads": sd.get_num_threads()}
        if r_packages:
            r = self.r_session()
            if r is not None:
                env["versions"].update(r.version())
                env["r_packages"] = r.package_versions(r_packages)
            else:
                env["r_error"] = self._r_failed
        return env

    def _one(self, case, label, threads, make, data, blas, rref=None):
        # The batches run with gc off: collect the cycles the previous measurement left
        # (statsmodels' results hold GBs in cycles at n = 10^6) before the next one.
        gc.collect()
        try:
            if rref is not None:
                r = self.r_session()
                if r is None:
                    raise RuntimeError(self._r_failed or "no R session")
                arrays = {rname: data[key] for key, rname in rref.data.items()}
                timing = dict(case.timing, batches=1, min_batches=1, target_s=0.0) if self.smoke else case.timing
                return r.time(rref.expr, arrays, setup=rref.setup, packages=rref.packages, **timing), None
            with harness.sd_threads(threads if label == "sd" else None), harness.blas_threads(blas):
                fn = make(data)
                return harness.measure(fn, **case.timing, smoke=self.smoke), None
        except Exception as e:  # noqa: BLE001  (one failing call must not stop the case)
            return None, f"{type(e).__name__}: {e}".strip()[:500]

    def case(self, cid, threads, use_r) -> dict:
        """One round of one case. numpy's error state is restored afterwards and the
        case's teardown(data) runs, so a case's global settings do not leak into the next."""
        import numpy as np

        err = np.geterr()
        try:
            return self._case(cid, threads, use_r)
        finally:
            np.seterr(**err)

    def _case(self, cid, threads, use_r) -> dict:
        case = self.cases[cid]
        t0 = time.perf_counter()
        out = {"id": cid, "samples": [], "errors": [], "setup_error": None}

        def add(label, lib, t, res):
            sample, err = res
            if sample is not None:
                out["samples"].append([label, lib, t, sample.times, sample.calls])
            else:
                out["errors"].append([label, t, err])

        try:
            data = case.setup()
        except Exception:  # noqa: BLE001
            out["setup_error"] = traceback.format_exc(limit=4)[-800:]
            out["seconds"] = time.perf_counter() - t0
            return out
        ths = [t for t in case.threads if threads is None or t in threads]
        for t in ths:
            add("sd", "sd", t, self._one(case, "sd", t, case.sd, data, None))
            for ref in case.refs:
                if ref.threads == "follow" and not isinstance(ref, RRef):
                    add(ref.label, ref.library, t, self._one(case, ref.label, t, ref.make, data, t))
        for ref in case.refs:
            if ref.threads != "fixed":
                continue
            if isinstance(ref, RRef):
                if use_r:
                    add(ref.label, "R", None, self._one(case, ref.label, None, None, data, None, rref=ref))
            else:
                add(ref.label, ref.library, None, self._one(case, ref.label, None, ref.make, data, None))
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
                reply = {"env": w.env(req.get("r_packages") or [])}
            elif req["cmd"] == "case":
                reply = w.case(req["id"], req.get("threads"), req.get("use_r", True))
            else:
                reply = {"error": f"unknown command {req['cmd']!r}"}
            proto.write(json.dumps(reply) + "\n")
    finally:
        w.close()


if __name__ == "__main__":
    main(sys.argv[1:])
