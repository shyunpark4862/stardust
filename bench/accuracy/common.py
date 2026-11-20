"""What every area's measurement shares: the summary of one region and the generic sweep."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np


@dataclasses.dataclass
class Custom:
    """A sweep that measures itself: run() -> {region: entry} (entries from region())."""

    area: str
    name: str
    run: Callable
    compare: Callable | None = None  # () -> {library: {region: entry}} (python -m bench.accuracy run --compare)

    @property
    def id(self):
        return f"{self.area}/{self.name}"


def finite_max(e):
    e = np.where(np.isnan(e), np.inf, np.asarray(e, dtype=np.float64))  # a NaN error is the worst
    return float(e.max()) if e.size else None


def p999(e):
    e = np.where(np.isnan(e), np.inf, np.asarray(e, dtype=np.float64))
    return float(np.quantile(e, 0.999, method="higher")) if e.size else None


def region(
    where,
    *,
    ulp=None,
    rel=None,
    mixed=None,
    abs_err=None,
    norm_rel=None,
    contract=None,
    bound=None,
    excluded=None,
    **extra,
) -> dict:
    """One region's entry: the maxima (and 99.9% quantiles of ULP and mixed) of the per-point
    errors given, contract = max(contract) / bound (None without a bound), and worst = the
    description (where) of the point with the largest contract error, else relative error.
    extra: further per-point measures, stored as <name>_max."""
    where = np.asarray(where)
    entry = {"points": int(where.size)}
    if ulp is not None:
        entry["ulp_max"], entry["ulp_p999"] = finite_max(ulp), p999(ulp)
    if mixed is not None:
        entry["mixed_max"], entry["mixed_p999"] = finite_max(mixed), p999(mixed)
    if rel is not None:
        entry["rel_max"] = finite_max(rel)
    if abs_err is not None:
        entry["abs_max"] = finite_max(abs_err)
    if norm_rel is not None:
        entry["norm_rel"] = finite_max(norm_rel)
    for k, v in extra.items():
        entry[f"{k}_max"] = finite_max(v)
    worst_c = finite_max(contract) if contract is not None else None
    entry["contract"] = worst_c / bound if bound and worst_c is not None else None
    entry["bound"] = bound
    if excluded is not None:  # a comparator's points left out (NaN or inf where the reference is finite)
        entry["excluded"] = int(excluded)
    key = next((e for e in (contract, rel, mixed, ulp, abs_err, norm_rel, *extra.values()) if e is not None), None)
    if key is not None and where.size:
        k = np.where(np.isnan(key), np.inf, np.asarray(key, dtype=np.float64))
        entry["worst"] = [str(where[int(np.argmax(k))])]
    else:
        entry["worst"] = []
    return entry


class Guarded(Exception):
    """A comparator call that ran past its time or memory limit (isolated below)."""


def _rss_bytes(pid) -> int:
    import subprocess

    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else 0


INLINE = ("numpy",)  # Accelerate's BLAS is not fork-safe: numpy (and scipy.linalg under its name) runs in place


def compare_libs(measure, libs, seconds=60.0):
    """A Custom sweep's compare: () -> {library: measure(library)}, each in a watched child
    (isolated; numpy in place: bounded, and its BLAS dies in a forked child); a library without
    the function (measure returns None) is left out ("—"), R without R is "not run"."""

    def run():
        from . import rsession

        out = {}
        for lib in libs:
            if lib == "R" and not rsession.available():
                out[lib] = "not run"
                continue
            try:
                import numpy as np

                with np.errstate(all="ignore"):
                    r = measure(lib) if lib in INLINE else isolated(lambda lib=lib: measure(lib), seconds=seconds)
            except Exception as e:
                out[lib] = {"error": f"{type(e).__name__}: {e}"[:300]}
                continue
            if r is not None:
                out[lib] = r
        return out

    return run


def isolated(fn, seconds=30.0, max_bytes=2 << 30):
    """fn() in a forked child the parent watches: killed when it runs past `seconds` or its
    RSS passes `max_bytes` (polled every 50 ms; the child's C code may hold the GIL, so no
    thread or signal inside it could stop it). Returns fn's result (pickled through a file);
    raises Guarded on a kill, and the child's exception otherwise."""
    import os
    import pickle
    import tempfile
    import time

    with tempfile.TemporaryDirectory(prefix="sd-acc-") as d:
        path = os.path.join(d, "out.pkl")
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                from . import rsession

                rsession.forget()  # the parent's R session is the parent's: start its own if needed
                try:
                    res = ("ok", fn())
                except BaseException as e:  # noqa: BLE001 - carried to the parent
                    res = ("error", f"{type(e).__name__}: {e}"[:300])
                with open(path, "wb") as f:
                    pickle.dump(res, f)
            except BaseException:  # noqa: BLE001
                code = 1
            finally:
                try:
                    rsession.close()
                except BaseException:  # noqa: BLE001
                    pass
                os._exit(code)
        end = time.monotonic() + seconds
        reason = None
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() > end:
                reason = f"over {seconds:g} s"
            elif _rss_bytes(pid) > max_bytes:
                reason = f"over {max_bytes >> 30} GB of memory"
            if reason:
                _kill_tree(pid)
                os.waitpid(pid, 0)
                raise Guarded(reason)
            time.sleep(0.05)
        if not os.path.exists(path):
            raise RuntimeError("the comparator's process died")
        with open(path, "rb") as f:
            kind, val = pickle.load(f)
    if kind == "error":
        raise RuntimeError(val)
    return val


def _kill_tree(pid):
    """SIGKILL pid and its children (an R session or a Boost child it started)."""
    import os
    import signal
    import subprocess

    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    for c in out:
        _kill_tree(int(c))
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
