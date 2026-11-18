"""G4: the speed of rlm (M: Huber and bisquare against statsmodels' RLM; MM: absolute, and R's
lmrob once for scale) and of theilslopes / siegelslopes against scipy
(docs/design/regression.md §3.6). Results: bench/regression/g4/speed.md (the raw output in
speed.log).

Data: X standard normal (n, k), y = X beta + 0.5 + standard normal noise with 5% of the rows
shifted by +10 (vertical outliers); MM's data the same with 20% of the rows moved to a
leverage cluster. statsmodels: RLM(y, add_constant(X), M=...).fit() at its defaults (tol 1e-8
on the objective's absolute change, maxiter 50); its BLAS threads by threadpoolctl (the same
count as sd's). The slopes: x, y standard normal-based, n points. Each case is the best of
several runs (two for the largest). The coefficients' largest difference to sd's, relative to
sd's largest coefficient, is printed.

Run: uv run --no-sync python bench/regression/g4/speed.py
"""

import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

import stardust as sd


def best(f, reps):
    out = []
    v = None
    for _ in range(reps):
        t0 = time.perf_counter()
        v = f()
        out.append(time.perf_counter() - t0)
    return min(out), v


def ms(t):
    return f"{t * 1e3:,.3g}" if t < 10 else f"{t * 1e3:,.0f}"


def data(n, k, seed=0, leverage=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    y = X @ np.linspace(1.0, 2.0, k) + 0.5 + rng.standard_normal(n)
    m = n // 20
    y[:m] += 10.0
    if leverage:
        m = n // 5
        X[:m] = rng.normal(5.0, 0.5, (m, k))
        y[:m] = rng.normal(-10.0, 1.0, m)
    return y, X


def rlm_m():
    import statsmodels.api as sm
    from statsmodels.robust import norms
    from threadpoolctl import threadpool_limits

    print("## rlm(method='M') vs statsmodels RLM (ms)")
    print("| (n, k) | norm | threads | sd | statsmodels | ratio | sd n_iter | sm iterations | max diff |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for n, k in ((100_000, 10), (1_000_000, 50)):
        y, X = data(n, k)
        ys, Xs = sd.asarray(y), sd.asarray(X)
        Xc = sm.add_constant(X)
        reps = 5 if n <= 100_000 else 2
        for norm, M in (("huber", norms.HuberT()), ("bisquare", norms.TukeyBiweight())):
            for th in (1, 10):
                sd.set_num_threads(th)
                ts, r = best(lambda: sd.regression.rlm(ys, Xs, norm=norm), reps)
                with threadpool_limits(th), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tm, s = best(lambda: sm.RLM(y, Xc, M=M).fit(), reps)
                p = np.asarray(r.params.tolist())
                d = np.max(np.abs(p - s.params)) / np.max(np.abs(p))
                print(f"| ({n:,}, {k}) | {norm} | {th} | {ms(ts)} | {ms(tm)} | {tm / ts:.1f} | {r.n_iter} | "
                      f"{s.fit_history['iteration']} | {d:.1e} |", flush=True)
    sd.set_num_threads(10)


def lmrob_run(y, X):
    """robustbase::lmrob on the same data (one call after a warm-up): (seconds, coefficients, scale),
    or None."""
    if shutil.which("Rscript") is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(f"{d}/a.csv", np.column_stack([y, X]), delimiter=",", fmt="%.17g")
        script = (f'suppressMessages(library(robustbase)); A <- as.matrix(read.csv("{d}/a.csv", header=FALSE)); '
                  'y <- A[,1]; X <- A[,-1]; invisible(lmrob(y ~ X)); '
                  't <- system.time(f <- lmrob(y ~ X))[["elapsed"]]; '
                  'cat(t, sprintf("%.17g", f$scale), sprintf("%.17g", coef(f)))')
        out = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
        try:
            v = [float(w) for w in out.stdout.split()]
        except ValueError:
            return None
        return v[0], np.array(v[2:]), v[1]


def rlm_mm():
    print("\n## rlm(method='MM') (ms): the large-n fast-S beyond n = 2000 (G58), the small-n search on all rows "
          "for comparison, R's lmrob (its defaults)")
    print("| (n, k) | threads | sd | sd, small-n search | S steps | M steps | lmrob (R) | coef diff to lmrob | "
          "S scale: sd / lmrob |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    K = sd._core.regression
    for n, k in ((10_000, 5), (100_000, 10), (1_000_000, 10)):
        y, X = data(n, k, leverage=True)
        ys, Xs = sd.asarray(y), sd.asarray(X)
        lr = lmrob_run(y, X)
        reps = 5 if n <= 10_000 else (3 if n <= 100_000 else 2)
        for th in (1, 10):
            sd.set_num_threads(th)
            ts, r = best(lambda: sd.regression.rlm(ys, Xs, method="MM", rng=sd.random.default_rng(1)), reps)
            small = ""
            if n <= 100_000:
                keys = sd.random.default_rng(1).integers(-2**63, 2**63 - 1, endpoint=True, shape=(2,))
                tsm, _ = best(lambda: K.rlm_fit("rlm", ys, Xs, keys, True, -1.0, 1, 1, 0.0, 0.0, 4.685061, 2, 50, 1e-8,
                                                500, False), reps)
                small = ms(tsm)
            siter = int(float(r._raw[6].tolist()[6]))
            if lr is None:
                lt = dc = sc = ""
            else:
                p = np.asarray(r.params.tolist())
                lt, dc, sc = ms(lr[0]), f"{np.max(np.abs(p - lr[1])) / np.max(np.abs(p)):.1e}", f"{r.scale:.10g} / {lr[2]:.10g}"
            print(f"| ({n:,}, {k}) | {th} | {ms(ts)} | {small} | {siter} | {r.n_iter} | {lt} | {dc} | {sc} |", flush=True)
    sd.set_num_threads(10)


def slopes():
    import scipy.stats as st

    print("\n## theilslopes / siegelslopes vs scipy (ms)")
    print("| n | function | threads | sd | scipy | ratio | same bits |")
    print("|---:|---|---:|---:|---:|---:|---|")
    for n in (1_000, 10_000, 100_000):
        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n)
        reps = 5 if n <= 1000 else (2 if n <= 10_000 else 1)
        for name, f, g in (("theilslopes", sd.regression.theilslopes, st.theilslopes),
                           ("siegelslopes", sd.regression.siegelslopes, st.siegelslopes)):
            if n > 10_000 and name == "siegelslopes":
                reps = 1
            tg = None
            if n <= 10_000:
                tg, sv = best(lambda: g(y, x), reps)
            for th in (1, 10):
                sd.set_num_threads(th)
                ts, v = best(lambda: f(y, x), reps)
                same = "" if tg is None else str(all(float(a) == float(b) + 0.0 for a, b in zip(v, sv)))
                print(f"| {n:,} | {name} | {th} | {ms(ts)} | {'' if tg is None else ms(tg)} | "
                      f"{'' if tg is None else f'{tg / ts:.1f}'} | {same} |", flush=True)
    sd.set_num_threads(10)


if __name__ == "__main__":
    which = sys.argv[1:] or ["m", "mm", "slopes"]
    if "m" in which:
        rlm_m()
    if "mm" in which:
        rlm_mm()
    if "slopes" in which:
        slopes()
