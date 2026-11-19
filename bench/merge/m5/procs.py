"""One flagged row timed in a fresh process (ROADMAP: flagged rows are checked
over several processes). Prints min/median µs of 15 runs.
    python bench/merge/m5/procs.py   (stardust_pandas: the old package, or the shim)"""

import statistics
import time

import numpy as np
import stardust_pandas as sdpd

sdpd.set_num_threads(1)
a = np.random.default_rng(0).standard_normal(1_000_000)
s = sdpd.Series(a)
t = []
for _ in range(15):
    t0 = time.perf_counter()
    for _ in range(20):
        s + s
    t.append((time.perf_counter() - t0) / 20 * 1e6)
print(f"{min(t):.0f}/{statistics.median(t):.0f}")
