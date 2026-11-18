"""G3: the accuracy of ridge, lasso, elastic_net and the penalized GLM against Arb, and how far
scikit-learn is from the same references (docs/design/regression.md §3.5). Results:
bench/regression/g3/accuracy.md (the raw output in accuracy.log).

The reference is tools/regref.py's penalized: given the active set and signs of sd's
solution, the stationarity conditions are solved exactly in Arb (a linear system; Newton and
Krawczyk's test for the GLMs) and certified by the signs and the strict KKT inequalities of
every inactive column. Errors are normwise relative errors of the whole params vector
(intercept first).

1. The problems of tests/regression/ref/penalized.json: sd, and scikit-learn at its default
   tol (1e-4) and at tol=1e-12 (ElasticNet/Lasso/Ridge(svd), LogisticRegression saga or
   newton-cholesky, PoissonRegressor newton-cholesky).
2. The condition number: X = U diag(s) V^T + 3 (n = 200, k = 8, s from 1 to 1/kappa
   geometrically after centering), y = X beta + noise; lasso at alpha = alpha_max / 20 and
   1e-3 alpha_max, elastic net (l1_ratio 0.5) and ridge at alpha 1e-3; two seeds, the max.
3. The number of rows (the Gram's sums): n = 1,000 and 20,000, k = 5.
4. ridge near least squares: the problems of 2 at kappa = 1e8 and 1e10, alpha = 1e-12 and 1e-18
   (the refinement with double-double residuals, G51).

Run: uv run --no-sync python bench/regression/g3/accuracy.py
"""

import os
import sys
import warnings

import numpy as np

import stardust as sd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import regref  # noqa: E402

from tests.regression import refdata_penalized as rp  # noqa: E402

R = sd.regression
GLMNET = R.penalized_glm


def err(got, ref):
    hi = np.array([v[0] for v in ref])
    lo = np.array([v[1] for v in ref])
    return float(np.linalg.norm((np.asarray(got, float) - hi) - lo) / np.linalg.norm(hi + lo))


def reference(y, X, *, alpha, l1_ratio, r, family="gaussian", weights=None, intercept=True, standardize=False):
    return regref.penalized(list(map(float, y)), np.asarray(X, float).tolist(), alpha=alpha, l1_ratio=l1_ratio,
                            beta0=[float(v) for v in np.asarray(r.params)], family=family,
                            weights=None if weights is None else list(map(float, weights)), intercept=intercept,
                            standardize=standardize)


def sk_fit(p, tol):
    """scikit-learn's fit of a problem (None where it has no equivalent: standardize, no intercept GLM)."""
    from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, PoissonRegressor, Ridge

    X, y = np.asarray(p["X"]), np.asarray(p["y"])
    w = None if p["weights"] is None else np.asarray(p["weights"])
    if p["standardize"]:
        return None
    fi = p["intercept"]
    if p["kind"] == "ridge":
        sw = len(y) if w is None else w.sum()
        m = Ridge(alpha=p["alpha"] * sw, fit_intercept=fi, solver="svd" if tol < 1e-6 else "auto").fit(X, y, w)
    elif p["kind"] in ("lasso", "elastic_net"):
        cls = Lasso if p["kind"] == "lasso" else ElasticNet
        kw = {} if p["kind"] == "lasso" else {"l1_ratio": p["l1_ratio"]}
        m = cls(alpha=p["alpha"], fit_intercept=fi, tol=tol, max_iter=10**6 if tol < 1e-6 else 1000, **kw).fit(X, y, w)
    elif p["family"] == "binomial":
        n = len(y) if w is None else w.sum()
        solver = "newton-cholesky" if p["l1_ratio"] == 0 else "saga"
        m = LogisticRegression(C=1 / (n * p["alpha"]), l1_ratio=p["l1_ratio"], solver=solver, tol=tol,
                               max_iter=10**6 if tol < 1e-6 else 100).fit(X, y, w)
    else:
        if p["l1_ratio"] != 0:
            return None
        m = PoissonRegressor(alpha=p["alpha"], solver="newton-cholesky", tol=tol, max_iter=10**4).fit(X, y, w)
    return np.r_[np.ravel(m.intercept_) if fi else [], np.ravel(m.coef_)]


def section1():
    print("## 1. the reference problems (n = 60)")
    print("| problem | sd | sd dual gap | scikit-learn tol 1e-4 | tol 1e-12 |")
    print("|---|---:|---:|---:|---:|")
    for name, p in rp.load().items():
        r = rp.fit(p)
        ref = p["ref"]["params"]
        gap = getattr(r, "dual_gap", float("nan"))
        cols = []
        for tol in (1e-4, 1e-12):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = sk_fit(p, tol)
            cols.append("" if s is None else f"{err(s, ref):.1e}")
        print(f"| {name} | {err(r.params, ref):.1e} | {gap:.1e} | {cols[0]} | {cols[1]} |", flush=True)


def kappa_problem(kappa, seed, n=200, k=8):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    U -= U.mean(axis=0)
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    X = (U * np.geomspace(1.0, 1.0 / kappa, k)) @ V.T * np.sqrt(n) + 3.0
    beta = rng.standard_normal(k)
    y = X @ beta + 0.5 * rng.standard_normal(n)
    return y, X


def section2():
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    print("\n## 2. the condition number (n = 200, k = 8; max of 2 seeds; sd / scikit-learn tol 1e-4)")
    print("| kappa | lasso alpha_max/20 | lasso 1e-3 alpha_max | elastic net 1e-3 | ridge 1e-3 | ridge 1e-8 |")
    print("|---:|---|---|---|---|---|")
    for kappa in (1e0, 1e2, 1e4, 1e6):
        cells = {}
        for seed in (0, 1):
            y, X = kappa_problem(kappa, seed)
            amax = sd._core.regression.alpha_max("x", sd.asarray(y), sd.asarray(X), None, 1.0, 0, True, False)
            cases = [("l20", lambda: R.lasso(y, X, alpha=amax / 20, max_iter=10**6), amax / 20, 1.0,
                      lambda: Lasso(alpha=amax / 20).fit(X, y)),
                     ("l1e3", lambda: R.lasso(y, X, alpha=amax * 1e-3, max_iter=10**6), amax * 1e-3, 1.0,
                      lambda: Lasso(alpha=amax * 1e-3).fit(X, y)),
                     ("e1e3", lambda: R.elastic_net(y, X, alpha=1e-3, max_iter=10**6), 1e-3, 0.5,
                      lambda: ElasticNet(alpha=1e-3).fit(X, y)),
                     ("r1e3", lambda: R.ridge(y, X, alpha=1e-3), 1e-3, 0.0,
                      lambda: Ridge(alpha=1e-3 * len(y)).fit(X, y)),
                     ("r1e8", lambda: R.ridge(y, X, alpha=1e-8), 1e-8, 0.0,
                      lambda: Ridge(alpha=1e-8 * len(y)).fit(X, y))]
            for key, fit, alpha, rho, skf in cases:
                r = fit()
                if hasattr(r, "converged") and not r.converged:
                    cells.setdefault(key, []).append((float("nan"), float("nan")))
                    continue
                ref = reference(y, X, alpha=alpha, l1_ratio=rho, r=r)["params"]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = skf()
                cells.setdefault(key, []).append((err(r.params, ref), err(np.r_[m.intercept_, m.coef_], ref)))
        row = []
        for key in ("l20", "l1e3", "e1e3", "r1e3", "r1e8"):
            v = np.array(cells[key])
            row.append("not converged" if np.isnan(v).any() else f"{v[:, 0].max():.1e} / {v[:, 1].max():.1e}")
        print(f"| {kappa:.0e} | " + " | ".join(row) + " |", flush=True)


def section3():
    print("\n## 3. the number of rows (k = 5; sd)")
    print("| n | lasso | elastic net | ridge | logistic L1 | poisson L2 |")
    print("|---:|---:|---:|---:|---:|---:|")
    for n in (1000, 20000):
        rng = np.random.default_rng(n)
        X = rng.standard_normal((n, 5)) * [1, 2, 0.5, 3, 1] + 1.0
        b = np.array([1.0, -0.5, 0.0, 0.2, 0.0])
        y = X @ b + rng.standard_normal(n)
        yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X - 1.0) @ b))).astype(float)
        yp = rng.poisson(np.exp(0.3 * (X - 1.0) @ b)).astype(float)
        out = []
        for fit, rho, fam, yy, alpha in ((R.lasso, 1.0, "gaussian", y, 0.02), (R.elastic_net, 0.5, "gaussian", y, 0.02),
                                         (R.ridge, 0.0, "gaussian", y, 0.02), (None, 1.0, "binomial", yb, 0.005),
                                         (None, 0.0, "poisson", yp, 0.005)):
            if fit is None:
                r = GLMNET(yy, X, family=fam, alpha=alpha, l1_ratio=rho)
            elif fit is R.elastic_net:
                r = fit(yy, X, alpha=alpha, l1_ratio=rho)
            else:
                r = fit(yy, X, alpha=alpha)
            ref = reference(yy, X, alpha=alpha, l1_ratio=rho, r=r, family=fam)["params"]
            out.append(f"{err(r.params, ref):.1e}")
        print(f"| {n:,} | " + " | ".join(out) + " |", flush=True)


def section4():
    print("\n## 4. ridge near least squares (n = 200, k = 8; max of 2 seeds; sd)")
    print("| kappa | alpha 1e-12 | alpha 1e-18 |")
    print("|---:|---:|---:|")
    for kappa in (1e8, 1e10):
        row = []
        for alpha in (1e-12, 1e-18):
            e = 0.0
            for seed in (0, 1):
                y, X = kappa_problem(kappa, seed)
                r = R.ridge(y, X, alpha=alpha)
                e = max(e, err(r.params, reference(y, X, alpha=alpha, l1_ratio=0.0, r=r)["params"]))
            row.append(f"{e:.1e}")
        print(f"| {kappa:.0e} | " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    section1()
    section2()
    section3()
    section4()
