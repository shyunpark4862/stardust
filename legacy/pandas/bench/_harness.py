"""Shared helpers for benchmarks and experiments.

Timing rule (as in stardust-numpy): run a call in batches, and report the
minimum per-call time over several batches. The minimum is the run least
disturbed by the OS; means and medians drift with background load.
"""

import platform
import subprocess
import sys
import time


def per_call_ns(fn, *, target_s=0.02, repeats=7):
    """Minimum per-call time of fn() in nanoseconds."""
    # Pick a batch size that takes about target_s, so timer resolution and
    # loop overhead do not matter.
    n = 1
    while True:
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        dt = time.perf_counter_ns() - t0
        if dt >= target_s * 1e9 or n >= 1 << 24:
            break
        n *= 2
    best = dt / n
    for _ in range(repeats - 1):
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter_ns() - t0) / n)
    return best


def per_call_ns_cycle(fns, *, target_s=0.02, repeats=7):
    """per_call_ns over a list of calls on different inputs of one shape,
    called in turn (M12). Calling one input over and over lets the branch
    predictor learn that input's branches: M9's linear probing ran 3.2x
    faster on a repeated 1k-row input than on fresh ones. Real calls see new
    data, so code that branches on the data is timed on inputs it has not
    just seen."""
    k = [0]

    def call():
        fns[k[0]]()
        k[0] = (k[0] + 1) % len(fns)

    return per_call_ns(call, target_s=target_s, repeats=repeats)


def fmt_ns(ns):
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} us"
    return f"{ns / 1e6:.2f} ms"


def env_info():
    """Versions and machine facts to record next to every measurement."""
    import numpy
    import pandas
    import polars

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {
        "python": sys.version.split()[0],
        "machine": f"{chip} ({platform.machine()})",
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "polars": polars.__version__,
    }
