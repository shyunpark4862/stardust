"""Group E: the univariate samplers. `hash`: a digest of each case's samples (several sizes, 1 and
3 threads) to compare two builds (a bit-identical change keeps every line); `time`: 10^6
samples, one thread, against numpy's Generator(PCG64) (the cases of bench/stats/t7/speed.py).

    .venv/bin/python bench/stats/opt/e/sample_speed.py hash|time
"""

import hashlib
import importlib.util
import sys
import timeit

import numpy as np

import stardust as sd

spec = importlib.util.spec_from_file_location("t7speed", "bench/stats/t7/speed.py")
t7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7)
S = sd.stats
mode = sys.argv[1]
if mode == "hash":
    for name, kw, _ in t7.CASES:
        D = getattr(S, name)(**kw)
        h = hashlib.sha1()
        for th in (1, 3):
            sd.set_num_threads(th)
            r = sd.random.default_rng(7)
            for n in (5, 1, 1000, 33, 200000):
                h.update(np.asarray(D.sample(n, rng=r)).tobytes())
            h.update(np.float64(D.sample(rng=r)).tobytes())
        print(name, kw, h.hexdigest()[:16])
else:
    sd.set_num_threads(1)
    g = np.random.Generator(np.random.PCG64(1))
    for name, kw, f in t7.CASES:
        D = getattr(S, name)(**kw)
        r = sd.random.default_rng(1)
        t1 = min(timeit.repeat(lambda: D.sample(10**6, rng=r), number=1, repeat=7))
        tn = min(timeit.repeat(lambda: f(g, 10**6), number=1, repeat=5)) if f else float("nan")
        print(f"{name:22s} {str(kw):45s} sd {t1 * 1e3:7.2f}  numpy {tn * 1e3:7.2f}  ratio {tn / t1:5.2f}")
