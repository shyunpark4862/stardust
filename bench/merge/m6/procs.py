"""The rows the M6 table flagged, timed in a fresh process (ROADMAP: flagged
rows are checked over several processes). Prints min/median µs of 15 runs.
    python bench/merge/m6/procs.py   (old packages, or stardust through bench/merge/shim)"""

import statistics
import time

import numpy as np

import stardust_linalg as la
import stardust_numpy as sdnp
import stardust_pandas as sdpd


def best(fn, reps=200):
    t = []
    for _ in range(15):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        t.append((time.perf_counter() - t0) / reps * 1e6)
    return f"{min(t):.1f}/{statistics.median(t):.1f}"


rng = np.random.default_rng(0)
a3 = sdnp.asarray(rng.standard_normal((1000, 3, 3)) + 3 * np.eye(3))
lu3 = la.lu_factor(a3) if hasattr(la, "lu_factor") else la.lu(a3)
b3 = sdnp.asarray(rng.standard_normal((1000, 3, 1)))
m32 = sdnp.asarray(rng.standard_normal((100, 32, 32)))
s = sdpd.Series(rng.standard_normal(1_000_000))
rows = []
try:
    rows.append(("lu_solve (1000,3,3)", best(lambda: la.lu_solve(*lu3, b3))))
except Exception as e:  # noqa: BLE001
    rows.append(("lu_solve (1000,3,3)", f"skip: {e}"))
rows.append(("matrix_norm fro (100,32,32)", best(lambda: la.matrix_norm(m32, ord="fro"))))
rows.append(("frame f + f 1M", best(lambda: s + s, reps=20)))
print("  ".join(f"{k}={v}" for k, v in rows))
