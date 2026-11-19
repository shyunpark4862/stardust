"""G1b: the speed of the HAC and cluster covariances, the diagnostics and summary() against
statsmodels (docs/design/regression.md §4). Results: bench/regression/g1b/speed.md.

Each case is timed best of several runs, on inputs already in each library's arrays (the
constant column made beforehand for statsmodels).
- HAC: sd.regression.ols(y, X, cov_type='HAC') (default lags floor(4 (n/100)^(2/9))) against
  statsmodels OLS(y, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': L}), both reading bse and
  pvalues.
- cluster: 1,000 clusters (int64 labels), cov_type='cluster', groups=g against statsmodels'
  cov_kwds={'groups': g}, use_t=True, both reading bse and pvalues.
- diagnostics: a fit, then leverage, internal studentized residuals, Cook's distance,
  Durbin-Watson, Jarque-Bera and Breusch-Pagan. statsmodels: OLSInfluence's
  hat_matrix_diag, resid_studentized_internal, cooks_distance, durbin_watson(wresid),
  jarque_bera(wresid), het_breuschpagan(resid, exog). The external studentized residuals are
  timed for sd only: statsmodels' resid_studentized_external refits without each row (n fits).
- summary: a fit and summary() (statsmodels: str(summary())).
sd's thread count is set with sd.set_num_threads; numpy's BLAS (under statsmodels) with
threadpoolctl.

Run: uv run --no-sync python bench/regression/g1b/speed.py
"""

# Closures read loop variables that the iteration's `del` removes afterwards; ruff takes them as undefined.
# ruff: noqa: F821

import gc
import math
import time

import numpy as np

import stardust as sd

SIZES = ((100_000, 10), (1_000_000, 50))


def best(f, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        out.append(time.perf_counter() - t0)
    return min(out)


def ms(t):
    return f"{t * 1e3:.3g}"


def main():
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import OLSInfluence
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from threadpoolctl import threadpool_limits

    R = sd.regression
    rng = np.random.default_rng(0)
    rows = []
    for n, k in SIZES:
        X = rng.standard_normal((n, k))
        e = rng.standard_normal(n)
        for t in range(1, n):
            e[t] += 0.5 * e[t - 1]
        y = X @ rng.standard_normal(k) + e
        g = rng.integers(0, 1000, n)
        Xc = sm.add_constant(X)
        Xs, ys, gs = sd.asarray(X), sd.asarray(y), sd.asarray(g)
        L = int(math.floor(4 * (n / 100) ** (2 / 9)))
        reps = 5 if n * k <= 1e6 else 3
        for threads in (1, 10):
            sd.set_num_threads(threads)
            with threadpool_limits(limits=threads):

                def sd_fit():
                    r = R.ols(ys, Xs)
                    return r.bse, r.pvalues

                def sm_fit():
                    r = sm.OLS(y, Xc).fit()
                    return r.bse, r.pvalues

                def sd_hac():
                    r = R.ols(ys, Xs, cov_type="HAC")
                    return r.bse, r.pvalues

                def sm_hac():
                    r = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True)
                    return r.bse, r.pvalues

                def sd_cl():
                    r = R.ols(ys, Xs, cov_type="cluster", groups=gs)
                    return r.bse, r.pvalues

                def sm_cl():
                    r = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": g}, use_t=True)
                    return r.bse, r.pvalues

                def sd_diag():
                    r = R.ols(ys, Xs)
                    return (
                        r.leverage,
                        r.resid_studentized_internal,
                        r.cooks_distance,
                        r.durbin_watson,
                        r.jarque_bera(),
                        r.breusch_pagan(),
                    )

                def sd_diag_ext():
                    r = R.ols(ys, Xs)
                    return r.resid_studentized_external

                def sm_diag():
                    r = sm.OLS(y, Xc).fit()
                    inf = OLSInfluence(r)
                    return (
                        inf.hat_matrix_diag,
                        inf.resid_studentized_internal,
                        inf.cooks_distance,
                        durbin_watson(r.wresid),
                        jarque_bera(r.wresid),
                        het_breuschpagan(r.resid, Xc),
                    )

                def sd_sum():
                    return R.ols(ys, Xs).summary()

                def sm_sum():
                    return str(sm.OLS(y, Xc).fit().summary())

                t = {
                    name: best(f, reps)
                    for name, f in (
                        ("sd_fit", sd_fit),
                        ("sm_fit", sm_fit),
                        ("sd_hac", sd_hac),
                        ("sm_hac", sm_hac),
                        ("sd_cl", sd_cl),
                        ("sm_cl", sm_cl),
                        ("sd_diag", sd_diag),
                        ("sd_ext", sd_diag_ext),
                        ("sm_diag", sm_diag),
                        ("sd_sum", sd_sum),
                        ("sm_sum", sm_sum),
                    )
                }
                gc.collect()
            rows.append((f"({n:,}, {k})", threads, L, t))
            print(rows[-1], flush=True)
        del X, Xc, Xs, y, ys, g, gs
        gc.collect()
    print()
    print("| (n, k) | threads | case | sd | statsmodels | ratio |")
    print("|---|---:|---|---:|---:|---:|")
    for size, th, L, t in rows:
        for label, a, b in (
            ("fit (nonrobust)", "sd_fit", "sm_fit"),
            (f"HAC ({L} lags)", "sd_hac", "sm_hac"),
            ("cluster (1,000)", "sd_cl", "sm_cl"),
            ("fit + diagnostics", "sd_diag", "sm_diag"),
            ("fit + summary()", "sd_sum", "sm_sum"),
        ):
            print(f"| {size} | {th} | {label} | {ms(t[a])} | {ms(t[b])} | {t[b] / t[a]:.2g} |")
        print(f"| {size} | {th} | fit + external studentized | {ms(t['sd_ext'])} | (n refits) | |")


if __name__ == "__main__":
    main()
