"""Speed of the multivariate normal and t cdfs (T9, csrc/stats/mvcdf.hpp) against scipy.stats.

    .venv/bin/python bench/stats/t9/cdf_speed.py      # writes cdf_speed.md beside it

Per point, one covariance matrix per dimension (a random correlation matrix), x = z + 0.5
with z standard normal. Ours on 1 and 10 threads (sd.set_num_threads) through
stardust._core.stats.mv_cdf; scipy single-threaded with its defaults:
  * multivariate_normal.cdf: d = 2 exact (TVPACK's bvnu), d >= 3 QMC to abseps = releps = 1e-5,
    maxpts = 10^6 d;
  * multivariate_t.cdf: QMC with maxpts = 1000 d and no error target.
Ours: atol = 1e-6 (the default) and, for the QMC, also 1e-5 (scipy's normal target) and 1e-4
(about the accuracy scipy's t reaches: it does not control its error, and is ~5e-5 off). The
"max |ours - scipy|" column shows how far scipy's answer is from ours. The QMC cost varies
~25x between problems of one dimension; these are 20-40 random points per case. The d = 2, 3
t's are exact here (error 0, ~1e-15 relative), scipy's are its QMC.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats as st

import stardust as sd
from stardust._core import stats as C


def case(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d + 2))
    S = A @ A.T
    s = np.sqrt(np.diag(S))
    return S / s[:, None] / s[None, :], rng


def ours(S, X, df, atol, threads):
    sd.set_num_threads(threads)
    args = (
        sd.asarray(S[None]),
        sd.asarray(X),
        sd.asarray(np.zeros(len(X), np.int64)),
        sd.asarray(np.array([df])),
        atol,
        0,
    )
    C.mv_cdf(*args)  # warm
    t = time.perf_counter()
    v, e = C.mv_cdf(*args)
    return (time.perf_counter() - t) / len(X), np.asarray(v), np.asarray(e)


def theirs(S, X, df):
    d = S.shape[0]
    t = time.perf_counter()
    if math.isinf(df):
        v = np.atleast_1d(st.multivariate_normal.cdf(X, np.zeros(d), S, rng=0))
    else:
        dist = st.multivariate_t(np.zeros(d), S, df=df, seed=0)
        v = np.array([dist.cdf(x) for x in X])
    return (time.perf_counter() - t) / len(X), v


def fmt(t):
    return f"{t * 1e6:.1f} us" if t < 1e-3 else f"{t * 1e3:.2f} ms"


def main():
    out = [
        "# Multivariate normal and t cdf: speed (bench/stats/t9/cdf_speed.py)\n",
        "Per point; ours 1 / 10 threads vs scipy (single-threaded, its defaults). Apple M4 Pro.\n",
        "| case | atol | points | ours 1 thr | ours 10 thr | scipy | scipy / ours 1 thr | max error estimate | max abs(ours - scipy) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in (2, 3, 5, 10):
        S, rng = case(d, d)
        for df in (math.inf, 4.0):
            exact = d <= 3  # the exact formulas (normal and t)
            n_ours = (20000 if math.isinf(df) else 200) if exact else (40 if math.isinf(df) else 20)
            n_sc = 400 if exact or not math.isinf(df) else 40
            X = rng.normal(size=(n_ours, d)) + 0.5
            ts, vs = theirs(S, X[:n_sc], df)
            for atol in (1e-6,) if exact else (1e-6, 1e-5, 1e-4):
                t1, v, e = ours(S, X, df, atol, 1)
                t10, v10, _ = ours(S, X, df, atol, 10)
                assert np.array_equal(v, v10)
                name = f"{'normal' if math.isinf(df) else 't df=4'} d={d}"
                out.append(
                    f"| {name} | {atol:.0e} | {n_ours} | {fmt(t1)} | {fmt(t10)} | {fmt(ts)} | {ts / t1:.1f} | "
                    f"{e.max():.1e} | {np.abs(v[:n_sc] - vs).max():.1e} |"
                )
                sys.stdout.write(out[-1] + "\n")
                sys.stdout.flush()
    sd.set_num_threads(10)
    text = "\n".join(out) + "\n"
    Path(__file__).with_suffix(".md").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
