"""G3: the speed of ridge, lasso, elastic_net, the path, 5-fold cross-validation and the
penalized logistic regression against scikit-learn (docs/design/regression.md §3.5). Results:
bench/regression/g3/speed.md (the raw output in speed.log).

Data: X standard normal (n, k), beta with 10 nonzero standard normal entries, y = X beta +
standard normal noise; the logistic y Bernoulli with eta = X beta / 2. alpha_max is the
smallest alpha whose lasso solution is 0.
- ridge: sd.ridge(alpha=0.1) / scikit-learn Ridge(alpha=0.1 n) (solver 'auto': Cholesky of
  X^T X) and solver='svd'.
- lasso, elastic net (l1_ratio 0.5) at alpha_max / 20: sd / scikit-learn Lasso, ElasticNet
  (their defaults: precompute=False, tol 1e-4).
- path: 100 alphas from alpha_max to 1e-3 alpha_max: sd.lasso(alpha=grid) / lasso_path
  (precompute='auto': the Gram).
- 5-fold CV: sd.lasso_cv(rng=..., alphas=100) / LassoCV(cv=the same folds, alphas=the same
  grid, n_jobs=None).
- ridge GCV over 100 alphas: sd.ridge_cv(cv='gcv', alphas=grid) / RidgeCV(alphas=grid) (its
  efficient leave-one-out).
- logistic L1 and L2 at alpha = 1e-3 (C = 1 / (n alpha)): sd's penalized GLM / scikit-learn
  LogisticRegression with saga (L1), liblinear (L1; it penalizes the intercept) and lbfgs,
  newton-cholesky (L2), their default tol.
Each case is the best of several runs. sd's threads by sd.set_num_threads; scikit-learn's
BLAS by threadpoolctl (the same count). The coefficients' largest difference to sd's,
relative to sd's largest coefficient, is printed.

Run: uv run --no-sync python bench/regression/g3/speed.py
"""

import time
import warnings

import numpy as np

import stardust as sd

SIZES = ((100_000, 10), (100_000, 100), (10_000, 1000))


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


def diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(a)))


def main():
    from sklearn.linear_model import ElasticNet, Lasso, LassoCV, LogisticRegression, Ridge, RidgeCV, lasso_path
    from threadpoolctl import threadpool_limits

    warnings.simplefilter("ignore")
    R = sd.regression
    GLMNET = R.penalized_glm
    for n, k in SIZES:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, k))
        beta = np.zeros(k)
        beta[:10] = rng.standard_normal(10)
        y = X @ beta + rng.standard_normal(n)
        yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta / 2))).astype(float)
        Xs, ys, ybs = sd.asarray(X), sd.asarray(y), sd.asarray(yb)
        amax = sd._core.regression.alpha_max("x", ys, Xs, None, 1.0, 0, True, False)
        grid = amax * np.geomspace(1, 1e-3, 100)
        gs = sd.asarray(grid)
        a20 = amax / 20
        fold = np.asarray(
            sd.take(sd.asarray([i % 5 for i in range(n)], dtype=sd.int64), sd.random.default_rng(1).permutation(n))
        )
        splits = [(np.nonzero(fold != q)[0], np.nonzero(fold == q)[0]) for q in range(5)]
        rgrid = np.geomspace(10.0, 1e-4, 100)
        reps = 5 if n * k <= 10**6 else 3
        for th in (1, 10):
            sd.set_num_threads(th)
            with threadpool_limits(th):
                rows = []
                t, r = best(lambda: R.ridge(ys, Xs, alpha=0.1), reps)
                t1, m = best(lambda: Ridge(alpha=0.1 * n).fit(X, y), reps)
                t2, m2 = best(lambda: Ridge(alpha=0.1 * n, solver="svd").fit(X, y), reps)
                rows.append(
                    (
                        "ridge",
                        t,
                        [
                            (t1, diff(r.params, np.r_[m.intercept_, m.coef_])),
                            (t2, diff(r.params, np.r_[m2.intercept_, m2.coef_])),
                        ],
                    )
                )
                t, r = best(lambda: R.lasso(ys, Xs, alpha=a20), reps)
                t1, m = best(lambda: Lasso(alpha=a20).fit(X, y), reps)
                rows.append((f"lasso ({r.n_iter})", t, [(t1, diff(r.params, np.r_[m.intercept_, m.coef_]))]))
                t, r = best(lambda: R.elastic_net(ys, Xs, alpha=a20), reps)
                t1, m = best(lambda: ElasticNet(alpha=a20).fit(X, y), reps)
                rows.append((f"elastic net ({r.n_iter})", t, [(t1, diff(r.params, np.r_[m.intercept_, m.coef_]))]))
                t, r = best(lambda: R.lasso(ys, Xs, alpha=gs), reps)
                t1, (al, co, _) = best(lambda: lasso_path(X - X.mean(0), y - y.mean(), alphas=grid), reps)
                rows.append(
                    (
                        f"path, 100 alphas ({int(np.sum(np.asarray(r.n_iter)))})",
                        t,
                        [(t1, diff(np.asarray(r.params)[:, 1:], co.T))],
                    )
                )
                t, r = best(lambda: R.lasso_cv(ys, Xs, rng=sd.random.default_rng(1), alphas=gs), 2)
                t1, m = best(lambda: LassoCV(alphas=grid, cv=splits).fit(X, y), 2)
                rows.append(("lasso_cv, 5 folds x 100", t, [(t1, diff(r.params, np.r_[m.intercept_, m.coef_]))]))
                t, r = best(lambda: R.ridge_cv(ys, Xs, cv="gcv", alphas=sd.asarray(rgrid)), reps)
                t1, m = best(lambda: RidgeCV(alphas=rgrid * n).fit(X, y), 2)
                rows.append(("ridge_cv gcv, 100 alphas", t, [(t1, float("nan"))]))
                C = 1 / (n * 1e-3)
                t, r = best(lambda: GLMNET(ybs, Xs, family="binomial", alpha=1e-3, l1_ratio=1.0), 2)
                t1, m = best(lambda: LogisticRegression(C=C, l1_ratio=1.0, solver="saga").fit(X, yb), 1)
                t2, m2 = best(lambda: LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear").fit(X, yb), 2)
                rows.append(
                    (
                        f"logistic L1 ({r.n_iter})",
                        t,
                        [
                            (t1, diff(r.params, np.r_[m.intercept_, m.coef_.ravel()])),
                            (t2, diff(r.params, np.r_[m2.intercept_, m2.coef_.ravel()])),
                        ],
                    )
                )
                t, r = best(lambda: GLMNET(ybs, Xs, family="binomial", alpha=1e-3, l1_ratio=0.0), 2)
                t1, m = best(lambda: LogisticRegression(C=C).fit(X, yb), 2)
                t2, m2 = best(lambda: LogisticRegression(C=C, solver="newton-cholesky").fit(X, yb), 2)
                rows.append(
                    (
                        f"logistic L2 ({r.n_iter})",
                        t,
                        [
                            (t1, diff(r.params, np.r_[m.intercept_, m.coef_.ravel()])),
                            (t2, diff(r.params, np.r_[m2.intercept_, m2.coef_.ravel()])),
                        ],
                    )
                )
                for name, t, comp in rows:
                    cells = " | ".join(f"{ms(tc)} | {tc / t:.2g} | {d:.1e}" for tc, d in comp)
                    print(f"| ({n:,}, {k}) | {th} | {name} | {ms(t)} | {cells} |", flush=True)


if __name__ == "__main__":
    main()
