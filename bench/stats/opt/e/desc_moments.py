"""Group E: sem / zscore / skew / kurtosis (the moments kernel). `hash`: a digest of the results
over sizes, shifts, NaN policies, layouts and 1 and 4 threads, to compare two builds (the
NEON loops and the side-by-side lanes keep every bit); `time`: 10^6 values against scipy.

    .venv/bin/python bench/stats/opt/e/desc_moments.py hash|time
"""

import sys, hashlib, numpy as np, timeit, scipy.stats as st
import stardust as sd

S = sd.stats
rng = np.random.default_rng(3)
if sys.argv[1] == "hash":
    h = hashlib.sha1()
    for n in (1, 7, 8, 9, 1000, 8192, 8193, 100003):
        for scale, shift in ((1, 0), (1, 1e8), (1e-3, -5)):
            x = rng.normal(size=n) * scale + shift
            if n > 20:
                x[rng.integers(0, n, 3)] = np.nan
            for th in (1, 4):
                sd.set_num_threads(th)
                for pol in ("omit", "propagate"):
                    X = sd.asarray(x)
                    for f in (S.sem, S.skew, S.kurtosis):
                        h.update(np.float64(f(X, nan_policy=pol)).tobytes())
                    h.update(np.asarray(S.zscore(X, nan_policy=pol)).tobytes())
                    if n >= 1000:
                        X2 = sd.asarray(x[: (n // 10) * 10].reshape(10, -1))
                        h.update(np.asarray(S.sem(X2, axis=1, nan_policy=pol)).tobytes())
                        h.update(np.asarray(S.kurtosis(X2, axis=0, nan_policy=pol)).tobytes())
    print(h.hexdigest())
else:
    sd.set_num_threads(1)
    x = rng.normal(size=10**6)
    X = sd.asarray(x)
    x2 = x.reshape(1000, 1000)
    X2 = sd.asarray(x2)
    for nm in ("sem", "zscore", "skew", "kurtosis"):
        f, g = getattr(S, nm), getattr(st, nm)
        for lab, a, b, kw in (("all", X, x, dict(axis=None)), ("axis 0", X2, x2, dict(axis=0))):
            t1 = min(timeit.repeat(lambda: f(a, **kw), number=1, repeat=9))
            t2 = min(timeit.repeat(lambda: g(b, **kw), number=1, repeat=5))
            print(f"{nm:10s} {lab:7s} sd {t1 * 1e3:6.2f} scipy {t2 * 1e3:6.2f} ratio {t2 / t1:5.2f}")
