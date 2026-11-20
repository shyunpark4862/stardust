"""Timing, thread control and the facts recorded next to every measurement.

The timing rule is the one the area benches converged on (bench/frame/_harness.py (at commit 4151d2a),
bench/special/common.py): call the function in batches long enough that the timer's
resolution and the loop's overhead do not matter, and keep the per-call time of every
batch. The minimum is the run least disturbed by the OS; the median of the batches
gives the noise band (median / min - 1) that a comparison between runs must exceed
before it counts.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import importlib
import importlib.metadata
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence

# Default timing parameters (seconds and counts); a case may override them.
TARGET_S = 0.02  # one batch takes at least this long
BATCHES = 7  # batches per measurement when the calls are cheap
MIN_BATCHES = 3  # at least this many, unless one call is very slow
BUDGET_S = 2.0  # aim for at most this much time per measurement
SLOW_CALL_S = 5.0  # a single call slower than this is timed once per round
MAX_CALLS = 1 << 24


@dataclasses.dataclass
class Sample:
    """The per-call times (seconds) of the batches of one measurement."""

    times: list[float]
    calls: int  # calls per batch

    def merged(self, other: Sample) -> Sample:
        return Sample(self.times + other.times, max(self.calls, other.calls))

    @property
    def min(self) -> float:
        return min(self.times)

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def noise(self) -> float:
        """The noise band: how far the median batch lies above the best one."""
        return self.median / self.min - 1.0 if self.min > 0 else 0.0


def _batch(fn: Callable[[], object], n: int) -> float:
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    return (time.perf_counter_ns() - t0) * 1e-9


def measure(
    fn: Callable[[], object],
    *,
    target_s: float = TARGET_S,
    batches: int = BATCHES,
    min_batches: int = MIN_BATCHES,
    budget_s: float = BUDGET_S,
    smoke: bool = False,
) -> Sample:
    """Time fn() and return the per-call time of each batch.

    One warm-up call first (caches, lazy imports, the allocator's first touch). Then the
    batch size doubles from one call until a batch takes target_s; those calibration
    batches also warm the code up. The number of batches is cut to fit budget_s, but
    never below min_batches, except for a call slower than SLOW_CALL_S, which is timed
    once (rounds give it more samples). smoke=True: one call only (does the case run?)."""
    first = _batch(fn, 1)
    if smoke:
        return Sample([first], 1)
    if first >= SLOW_CALL_S:
        return Sample([_batch(fn, 1)], 1)
    n = 1
    per = first
    if first < target_s:
        while n < MAX_CALLS:
            dt = _batch(fn, n)
            per = dt / n
            if dt >= target_s:
                break
            n *= 2
    k = batches
    if per * n * k > budget_s:
        k = max(min_batches, int(budget_s / (per * n)))
    gc_was = gc.isenabled()
    gc.disable()  # a collection inside one batch would only add noise
    try:
        times = [_batch(fn, n) / n for _ in range(k)]
    finally:
        if gc_was:
            gc.enable()
    return Sample(times, n)


def rotate(calls: Sequence[Callable[[], object]]) -> Callable[[], object]:
    """One callable that calls `calls` in turn: time a branchy kernel on several inputs of
    one shape, so the branch predictor and the caches cannot learn one input (repeating
    one input made numpy's argsort of 10^3 elements 3.8x faster; docs/design/frame.md
    §12.1). The turn costs about 30-40 ns per call, the same on both sides of a ratio."""
    calls = list(calls)
    if len(calls) == 1:
        return calls[0]
    k = [0]
    n = len(calls)

    def call():
        i = k[0]
        k[0] = i + 1 if i + 1 < n else 0
        return calls[i]()

    return call


def fresh_count(n: int) -> int:
    """How many inputs to rotate for a branchy case of n elements: many small ones, a
    few big ones (about 4·10^6 elements in all, at least 4, at most 64)."""
    return min(64, max(4, 4_000_000 // max(n, 1)))


# ---------------------------------------------------------------- threads


@contextlib.contextmanager
def sd_threads(n: int | None) -> Iterator[None]:
    """Run with stardust's pool at n threads (None: leave it), restoring it after."""
    import stardust as sd

    before = sd.get_num_threads()
    if n is not None:
        sd.set_num_threads(n)
    try:
        yield
    finally:
        sd.set_num_threads(before)


@contextlib.contextmanager
def blas_threads(n: int | None) -> Iterator[None]:
    """Limit the BLAS / OpenMP pools of numpy, scipy, scikit-learn (threadpoolctl) to n
    threads; None leaves them at their defaults."""
    if n is None:
        yield
        return
    from threadpoolctl import threadpool_limits

    with threadpool_limits(n):
        yield


# ---------------------------------------------------------------- environment


def _run(cmd: Sequence[str], cwd: str | None = None, strip: bool = True) -> str:
    try:
        out = subprocess.run(list(cmd), capture_output=True, text=True, cwd=cwd, timeout=30).stdout
        return out.strip() if strip else out
    except OSError, subprocess.SubprocessError:
        return ""


def git_info(root: str) -> dict:
    commit = _run(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, strip=False)
    changed = [ln[3:] for ln in status.splitlines() if ln.strip()]
    return {"commit": commit or None, "dirty": bool(changed), "changed": changed}


def machine_info() -> dict:
    def sysctl(key: str) -> str:
        return _run(["sysctl", "-n", key])

    return {
        "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "macos": platform.mac_ver()[0] or None,
        "cpus": os.cpu_count(),
        "perf_cores": int(sysctl("hw.perflevel0.physicalcpu") or 0) or None,
        "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30) or None,
        "python": platform.python_version(),
    }


PY_LIBS = ("stardust", "numpy", "scipy", "pandas", "pyarrow", "statsmodels", "sklearn", "threadpoolctl")


def library_versions(libs: Sequence[str] = PY_LIBS) -> dict:
    out = {}
    for name in libs:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        version = getattr(mod, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        out[name] = version
    return out


def blas_info() -> list[dict]:
    """The BLAS libraries threadpoolctl sees (numpy's and scipy's), with their default
    thread counts."""
    try:
        import numpy  # noqa: F401  (load numpy's BLAS so threadpoolctl lists it)
        import scipy.linalg  # noqa: F401
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    return [{k: d.get(k) for k in ("internal_api", "num_threads", "version", "filepath")} for d in threadpool_info()]


def _sig3(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_time(s: float | None) -> str:
    """A time in seconds, in the unit that gives about three significant digits."""
    if s is None or s != s:
        return "-"
    for scale, unit in ((1e9, "ns"), (1e6, "µs"), (1e3, "ms")):
        if s * scale < 999.5:
            return f"{_sig3(s * scale)} {unit}"
    return f"{_sig3(s)} s"


def fmt_ratio(r: float | None) -> str:
    if r is None or r != r:
        return "-"
    return _sig3(r)


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)
