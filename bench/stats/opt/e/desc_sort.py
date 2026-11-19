"""Group E: rankdata / mode / differential_entropy over one long lane (the samplesort on the
pool). `hash`: a digest over sizes, kinds of data, NaN policies and 1, 3 and 10 threads, to
compare two builds; `time`: 10^6 values on 1 and 10 threads.

    .venv/bin/python bench/stats/opt/e/desc_sort.py hash|time
"""

import hashlib
import sys
import timeit

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats
rng = np.random.default_rng(5)
if sys.argv[1] == "hash":
    h = hashlib.sha1()
    for n in (1000, 65535, 65536, 300001, 1000000):
        for kind in range(4):
            x = [
                rng.normal(size=n),
                rng.integers(0, 50, n).astype(float),
                np.round(rng.normal(size=n), 2),
                np.zeros(n),
            ][kind]
            if kind == 2:
                x[rng.integers(0, n, 5)] = np.nan
            X = sd.asarray(x)
            for th in (1, 3, 10):
                sd.set_num_threads(th)
                for pol in ("omit", "propagate"):
                    for m in ("average", "min", "dense", "ordinal"):
                        h.update(np.asarray(S.rankdata(X, method=m, nan_policy=pol)).tobytes())
                    md = S.mode(X, nan_policy=pol)
                    h.update(np.float64(md[0]).tobytes())
                    h.update(np.int64(md[1]).tobytes())
                    h.update(np.float64(S.differential_entropy(X, nan_policy=pol)).tobytes())
    print(h.hexdigest())
else:
    x = rng.normal(size=10**6)
    X = sd.asarray(x)
    x2 = x.reshape(1000, 1000)
    X2 = sd.asarray(x2)
    xi = rng.integers(0, 1000, 10**6).astype(float)
    XI = sd.asarray(xi)
    for th in (1, 10):
        sd.set_num_threads(th)
        for nm, f, a in (
            ("rankdata all", lambda: S.rankdata(X), None),
            ("rankdata axis0", lambda: S.rankdata(X2, axis=0), None),
            ("mode all", lambda: S.mode(X), None),
            ("mode ints all", lambda: S.mode(XI), None),
            ("diffent all", lambda: S.differential_entropy(X), None),
            ("diffent axis0", lambda: S.differential_entropy(X2, axis=0), None),
        ):
            print(th, f"{nm:16s} {min(timeit.repeat(f, number=1, repeat=5)) * 1e3:7.2f} ms")
