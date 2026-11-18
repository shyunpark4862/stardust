"""G2: the speed of glm against statsmodels' GLM and scikit-learn (docs/design/regression.md §4).
Results: bench/regression/g2/speed.md.

Each case is timed best of several runs, on inputs already in each library's arrays (the
constant column made beforehand for statsmodels).
- logistic: sd.regression.glm(y, X, family='binomial') reading params, bse and pvalues
  (and the same fit with the inner iterations by QR instead of the normal equations, G40);
  statsmodels GLM(y, Xc, Binomial()).fit() reading the same; scikit-learn
  LogisticRegression(penalty=None) with its default solver (lbfgs, tol 1e-4) and with
  solver='newton-cholesky' (fit only: no inference).
- poisson: the same with family='poisson', Poisson(), PoissonRegressor(alpha=0) (lbfgs and
  newton-cholesky).
- batch: 10,000 logistic regressions of (50, 3) in one glm call (y (10000, 50), X (10000,
  50, 3)) against a loop of statsmodels fits (timed once) and a loop of sd fits.
The coefficients' largest relative difference to sd's is printed for each competitor.
sd's thread count is set with sd.set_num_threads; numpy's BLAS (under statsmodels and
scikit-learn) with threadpoolctl.

Run: uv run --no-sync python bench/regression/g2/speed.py
"""

import gc
import time
import warnings

import numpy as np

import stardust as sd

SIZES = ((100_000, 10), (1_000_000, 50))


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


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def main():
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from threadpoolctl import threadpool_limits

    warnings.simplefilter("ignore")
    R = sd.regression
    rng = np.random.default_rng(0)
    rows = []
    for n, k in SIZES:
        X = rng.standard_normal((n, k))
        beta = rng.standard_normal(k) * (0.6 / np.sqrt(k))
        eta = 0.2 + X @ beta
        yl = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
        yp = rng.poisson(np.exp(0.3 + 0.5 * (X @ beta))).astype(float)
        Xc = sm.add_constant(X)
        Xs = sd.asarray(X)
        reps = 3 if n * k <= 1e6 else 2
        for fam, y, skl in (("binomial", yl, LogisticRegression), ("poisson", yp, PoissonRegressor)):
            ys = sd.asarray(y)
            smf = sm.families.Binomial() if fam == "binomial" else sm.families.Poisson()
            kw = dict(penalty=None) if fam == "binomial" else dict(alpha=0.0)
            for threads in (1, 10):
                sd.set_num_threads(threads)
                with threadpool_limits(limits=threads):
                    def sd_fit():
                        r = R.glm(ys, Xs, family=fam)
                        return r.params, r.bse, r.pvalues, r.n_iter

                    aux = sd.asarray(np.r_[np.ones((3, n)), np.zeros((1, n))])
                    fcode = 1 if fam == "binomial" else 2
                    lcode = 2 if fam == "binomial" else 1

                    def sd_qr():
                        # the same fit with the inner iterations by QR (G2 before G40)
                        raw = sd._core.regression.glm_fit("glm", ys, Xs, aux, None, 0, fcode, lcode, False, False,
                                                          False, False, True, -1.0, 0, -1, 100, 1e-8, 0, 0.0, 0, 3,
                                                          False)
                        return raw[0]

                    def sm_fit():
                        r = sm.GLM(y, Xc, family=smf).fit()
                        return r.params, r.bse, r.pvalues

                    def sk_lbfgs():
                        m = skl(**kw, max_iter=10000).fit(X, y)
                        return np.r_[m.intercept_, m.coef_.ravel()]

                    def sk_newton():
                        m = skl(**kw, solver="newton-cholesky").fit(X, y)
                        return np.r_[m.intercept_, m.coef_.ravel()]

                    t_sd, v_sd = best(sd_fit, reps)
                    t_qr, _ = best(sd_qr, reps)
                    t_sm, v_sm = best(sm_fit, reps)
                    t_l, v_l = best(sk_lbfgs, reps)
                    t_n, v_n = best(sk_newton, reps)
                    p = np.asarray(v_sd[0])
                    rows.append((f"({n:,}, {k})", fam, threads, int(v_sd[3]), t_sd, t_qr, t_sm, t_l, t_n,
                                 rel(v_sm[0], p), rel(v_l, p), rel(v_n, p)))
                    print(rows[-1], flush=True)
                gc.collect()
        del X, Xc, Xs
        gc.collect()

    # a batch of small logistic regressions
    B, n, k = 10_000, 50, 3
    Xb = rng.standard_normal((B, n, k))
    Yb = (rng.uniform(size=(B, n)) < 1 / (1 + np.exp(-(0.2 + Xb @ np.array([0.5, -0.3, 0.2]))))).astype(float)
    Xbs, Ybs = sd.asarray(Xb), sd.asarray(Yb)
    Xbc = [sm.add_constant(Xb[i]) for i in range(B)]
    brows = []
    for threads in (1, 10):
        sd.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            def sd_batch():
                r = R.glm(Ybs, Xbs, family="binomial")
                return r.params, r.bse, r.pvalues, r.separation

            def sd_loop():
                for i in range(B):
                    r = R.glm(Yb[i], Xb[i], family="binomial")
                    r.bse, r.pvalues

            def sm_loop():
                for i in range(B):
                    r = sm.GLM(Yb[i], Xbc[i], family=sm.families.Binomial()).fit()
                    r.bse, r.pvalues

            t_b, v_b = best(sd_batch, 5)
            t_l, _ = best(sd_loop, 1)
            t_s, _ = best(sm_loop, 1)
            nsep = int(np.sum(np.asarray(v_b[3])))
            brows.append((threads, t_b, t_l, t_s, nsep))
            print(brows[-1], flush=True)

    print()
    print("| (n, k) | family | threads | sd (iterations) | sd, QR inner | statsmodels | ratio | scikit-learn lbfgs | "
          "ratio | scikit-learn newton-cholesky | ratio |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for size, fam, th, it, t_sd, t_qr, t_sm, t_l, t_n, e_sm, e_l, e_n in rows:
        print(f"| {size} | {fam} | {th} | {ms(t_sd)} ({it}) | {ms(t_qr)} | {ms(t_sm)} | {t_sm / t_sd:.2g} | {ms(t_l)} | "
              f"{t_l / t_sd:.2g} | {ms(t_n)} | {t_n / t_sd:.2g} |")
    print()
    print("| (n, k) | family | statsmodels | lbfgs | newton-cholesky |  (largest relative difference of the coefficients to sd)")
    for size, fam, th, it, t_sd, t_qr, t_sm, t_l, t_n, e_sm, e_l, e_n in rows:
        if th == 1:
            print(f"| {size} | {fam} | {e_sm:.1e} | {e_l:.1e} | {e_n:.1e} |")
    print()
    print("| threads | sd batch | sd loop | statsmodels loop | ratio (statsmodels / batch) | separated |")
    for th, t_b, t_l, t_s, nsep in brows:
        print(f"| {th} | {ms(t_b)} | {ms(t_l)} | {ms(t_s)} | {t_s / t_b:.3g} | {nsep} |")


if __name__ == "__main__":
    main()
