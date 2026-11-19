"""G1: the speed of sd.regression.ols against statsmodels OLS and scikit-learn
LinearRegression (docs/design/regression.md §4). Results: bench/regression/g1/speed.md.

Each case is timed best of several runs, on inputs already in each library's own arrays (the
constant column made beforehand for statsmodels). "fit" is the call that fits:
sd.regression.ols (the kernel runs in full: coefficients, residuals, sums of squares,
covariance), statsmodels' OLS(...).fit() (its default method, pinv) and scikit-learn's fit.
"+inference" also reads params, bse, pvalues and rsquared (statsmodels computes those on
first access; so does sd for bse and pvalues). The batch case fits 10,000 regressions of
(50, 3) in one sd call, against a Python loop over statsmodels (and over sd, for reference).
sd's thread count is set with sd.set_num_threads; numpy's BLAS (under statsmodels and
scikit-learn) with threadpoolctl.

Run: uv run --no-sync python bench/regression/g1/speed.py
"""

import time

import numpy as np

import stardust as sd

SIZES = ((1_000, 10), (100_000, 10), (1_000_000, 50), (10_000, 1_000))


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
    from sklearn.linear_model import LinearRegression
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(0)
    rows = []
    for n, k in SIZES:
        X = rng.standard_normal((n, k))
        y = X @ rng.standard_normal(k) + rng.standard_normal(n)
        Xc = sm.add_constant(X)
        Xs, ys = sd.asarray(X), sd.asarray(y)
        reps = 7 if n * k <= 1e6 else 3
        for threads in (1, 10):
            sd.set_num_threads(threads)
            with threadpool_limits(limits=threads):
                t_sd = best(lambda: sd.regression.ols(ys, Xs), reps)

                def sd_inf():
                    r = sd.regression.ols(ys, Xs)
                    return r.params, r.bse, r.pvalues, r.rsquared

                t_sdi = best(sd_inf, reps)
                t_sm = best(lambda: sm.OLS(y, Xc).fit(), reps)

                def sm_inf():
                    r = sm.OLS(y, Xc).fit()
                    return r.params, r.bse, r.pvalues, r.rsquared

                t_smi = best(sm_inf, reps)
                t_sk = best(lambda: LinearRegression().fit(X, y), reps)
            rows.append((f"({n:,}, {k})", threads, t_sd, t_sdi, t_sm, t_smi, t_sk))
            print(rows[-1], flush=True)
    # batch: 10,000 regressions of (50, 3)
    B, n, k = 10_000, 50, 3
    X = rng.standard_normal((B, n, k))
    y = np.einsum("bnk,k->bn", X, [1.0, 2.0, 3.0]) + rng.standard_normal((B, n))
    Xc = [sm.add_constant(X[i]) for i in range(B)]
    Xs, ys = sd.asarray(X), sd.asarray(y)
    Xl = [sd.asarray(X[i]) for i in range(B)]
    yl = [sd.asarray(y[i]) for i in range(B)]
    batch = []
    for threads in (1, 10):
        sd.set_num_threads(threads)
        with threadpool_limits(limits=threads):

            def sd_batch():
                r = sd.regression.ols(ys, Xs)
                return r.params, r.bse, r.pvalues, r.rsquared

            def sd_loop():
                for i in range(B):
                    r = sd.regression.ols(yl[i], Xl[i])
                    r.params, r.bse, r.pvalues, r.rsquared

            def sm_loop():
                for i in range(B):
                    r = sm.OLS(y[i], Xc[i]).fit()
                    r.params, r.bse, r.pvalues, r.rsquared

            t_b = best(sd_batch, 5)
            t_l = best(sd_loop, 2)
            t_s = best(sm_loop, 2)
        batch.append((threads, t_b, t_l, t_s))
        print(batch[-1], flush=True)
    print()
    print("| (n, k) | threads | sd fit | sd +inference | statsmodels fit | statsmodels +inference | scikit-learn fit |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for size, th, a, b, c, d, e in rows:
        print(f"| {size} | {th} | {ms(a)} | {ms(b)} | {ms(c)} | {ms(d)} | {ms(e)} |")
    print()
    print(
        "| batch 10,000 x (50, 3), fit + inference | threads | sd one call | sd Python loop | statsmodels Python loop |"
    )
    print("|---|---:|---:|---:|---:|")
    for th, a, b, c in batch:
        print(f"| | {th} | {ms(a)} | {ms(b)} | {ms(c)} |")


if __name__ == "__main__":
    main()
