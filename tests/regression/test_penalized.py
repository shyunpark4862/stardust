"""ridge, lasso, elastic_net, their cross-validation and the penalized GLM
(docs/api/regression.md §4, docs/design/regression.md §3.5).

Accuracy against Arb (tests/regression/ref/penalized.json: the KKT system of the solution's
active set solved exactly and certified), then scikit-learn at a tight tolerance, paths against
single fits, warm starts, standardize, cross-validation, batches, threads, NaN policies, edge
cases and errors."""

import math
import warnings

import numpy as np
import pytest

import stardust as sd

from . import refdata_penalized
from .smalldata import RNG, YB, N, X, Y, gen

R = sd.regression
GLMNET = R.penalized_glm
NAME = "penalized_glm"
REFS = refdata_penalized.load()
K = sd._core.regression


def a(x):
    return np.asarray(x, dtype=float)


def norm_err(got, ref):
    hi = np.array([v[0] for v in ref])
    lo = np.array([v[1] for v in ref])
    return np.linalg.norm((a(got) - hi) - lo) / np.linalg.norm(hi + lo)


def amax(y, X, *, l1_ratio=1.0, family=0, intercept=True, standardize=False, w=None):
    return K.alpha_max(
        "t",
        sd.asarray(y),
        sd.asarray(X),
        None if w is None else sd.asarray(w),
        l1_ratio,
        family,
        intercept,
        standardize,
    )


def data(n=120, p=6, seed=0, offset=2.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * rng.uniform(0.5, 3.0, p) + offset
    beta = np.zeros(p)
    beta[: min(p, 4)] = [1.5, -2.0, 0.5, 0.1][: min(p, 4)]
    y = X @ beta + 1.0 + rng.standard_normal(n)
    return y, X


def sk_params(m):
    return np.r_[np.ravel(m.intercept_), np.ravel(m.coef_)]


# ---- Arb references


@pytest.mark.parametrize("name", list(REFS))
def test_against_arb(name):
    p = REFS[name]
    r = refdata_penalized.fit(p)
    # the Gram's rounding (a few u at these condition numbers); the ill-conditioned fits are
    # refined with double-double residuals (G51) to the level of the final rounding
    tol = 4e-16 if "kappa" in name else 4e-15
    assert norm_err(r.params, p["ref"]["params"]) <= tol
    if p["kind"] != "ridge":
        assert r.converged
        assert abs(r.dual_gap) <= 1e-14 * (1 + abs(r.deviance if p["kind"] == "glm" else r.ssr / r.nobs))
    ref = p["ref"]
    if "ssr" in ref:
        assert r.ssr == pytest.approx(ref["ssr"][0] + ref["ssr"][1], rel=1e-11 if "kappa" in name else 1e-13)
    else:
        assert r.deviance == pytest.approx(ref["deviance"][0] + ref["deviance"][1], rel=1e-13)
    # exact zeros where the certified solution is zero
    zero = [v[0] == 0 and v[1] == 0 for v in ref["params"]]
    assert all(g == 0.0 for g, z in zip(a(r.params), zero) if z)


# ---- scikit-learn at a tight tolerance


@pytest.mark.parametrize("weighted", [False, True])
def test_lasso_enet_against_sklearn(weighted):
    lm = pytest.importorskip("sklearn.linear_model")
    y, X = data(seed=1)
    w = np.random.default_rng(2).uniform(0.2, 2.0, len(y)) if weighted else None
    if weighted:
        w[[4, 9]] = 0.0
    for alpha in (0.005, 0.05, 0.3):
        for rho in (1.0, 0.7, 0.2):
            m = lm.ElasticNet(alpha=alpha, l1_ratio=rho, tol=1e-14, max_iter=10**6).fit(X, y, sample_weight=w)
            r = (
                R.elastic_net(y, X, alpha=alpha, l1_ratio=rho, weights=w)
                if rho < 1
                else R.lasso(y, X, alpha=alpha, weights=w)
            )
            np.testing.assert_allclose(a(r.params), sk_params(m), rtol=1e-9, atol=1e-11)
            assert np.array_equal(a(r.params)[1:] == 0, m.coef_ == 0)
            assert r.rsquared == pytest.approx(m.score(X, y, sample_weight=w), rel=1e-9)
            np.testing.assert_allclose(a(r.predict(X[:5])), m.predict(X[:5]), rtol=1e-9)


def test_naive_updates_against_sklearn():
    # more than 1,024 columns: residual (naive) updates instead of the Gram's
    lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(30)
    n, p = 200, 1100
    X = rng.standard_normal((n, p)) + 1.0
    beta = np.zeros(p)
    beta[:5] = [1.0, -1.0, 2.0, 0.5, -0.3]
    y = X @ beta + rng.standard_normal(n)
    w = rng.uniform(0.5, 2.0, n)
    w[:3] = 0.0
    for alpha, rho in ((0.1, 1.0), (0.05, 0.5)):
        m = lm.ElasticNet(alpha=alpha, l1_ratio=rho, tol=1e-13, max_iter=10**6).fit(X, y, sample_weight=w)
        r = R.elastic_net(y, X, alpha=alpha, l1_ratio=rho, weights=w)
        assert r.converged
        np.testing.assert_allclose(a(r.params), sk_params(m), rtol=1e-8, atol=1e-10)
    path = R.lasso(y, X, alpha=[0.2, 0.1])
    assert np.array_equal(a(path.params)[1], a(R.lasso(y, X, alpha=0.1).params))


def test_ridge_against_sklearn():
    lm = pytest.importorskip("sklearn.linear_model")
    y, X = data(seed=3)
    w = np.random.default_rng(4).uniform(0.2, 2.0, len(y))
    w[7] = 0.0
    for alpha in (1e-4, 0.1, 10.0):
        for ww in (None, w):
            sw = len(y) if ww is None else ww.sum()
            # scikit-learn's Ridge minimizes sum w r^2 + alpha ||beta||^2: its alpha is sw times ours
            m = lm.Ridge(alpha=alpha * sw, solver="svd").fit(X, y, sample_weight=ww)
            r = R.ridge(y, X, alpha=alpha, weights=ww)
            np.testing.assert_allclose(a(r.params), sk_params(m), rtol=1e-12, atol=1e-13)
            e = (
                lm.ElasticNet(alpha=alpha, l1_ratio=0.0, tol=1e-15, max_iter=10**6).fit(X, y, sample_weight=ww)
                if alpha >= 0.1
                else None
            )
            if e is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                np.testing.assert_allclose(a(r.params), sk_params(e), rtol=1e-8)


def test_enet_path_against_sklearn():
    lm = pytest.importorskip("sklearn.linear_model")
    y, X = data(seed=5)
    grid = amax(y, X, l1_ratio=0.5) * np.geomspace(1, 1e-3, 30)
    alphas, coefs, _ = lm.enet_path(X - X.mean(0), y - y.mean(), l1_ratio=0.5, alphas=grid, tol=1e-14, max_iter=10**6)
    r = R.elastic_net(y, X, alpha=grid, l1_ratio=0.5)
    np.testing.assert_allclose(a(r.params)[:, 1:], coefs.T, rtol=1e-8, atol=1e-11)
    assert r.params.shape == (30, X.shape[1] + 1) and r.n_iter.shape == (30,)


def test_glm_against_sklearn():
    lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(6)
    n, p = 300, 5
    X = rng.standard_normal((n, p))
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ [1.0, -0.8, 0.0, 0.3, 0.0] + 0.2)))).astype(float)
    yp = rng.poisson(np.exp(X @ [0.3, -0.2, 0.0, 0.1, 0.0] + 0.5)).astype(float)
    alpha = 0.01
    C = 1 / (n * alpha)  # LogisticRegression's C sum(loss) + penalty: alpha = 1 / (C n)
    for rho, solver in ((0.0, "newton-cholesky"), (1.0, "saga"), (0.5, "saga")):
        m = lm.LogisticRegression(C=C, l1_ratio=rho, solver=solver, tol=1e-14, max_iter=10**6).fit(X, yb)
        r = GLMNET(yb, X, family="binomial", alpha=alpha, l1_ratio=rho)
        np.testing.assert_allclose(a(r.params), sk_params(m), rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(a(r.predict(X[:4])), m.predict_proba(X[:4])[:, 1], rtol=1e-8)
    m = lm.PoissonRegressor(alpha=alpha, solver="newton-cholesky", tol=1e-14, max_iter=10**4).fit(X, yp)
    r = GLMNET(yp, X, family="poisson", alpha=alpha, l1_ratio=0.0)
    np.testing.assert_allclose(a(r.params), sk_params(m), rtol=1e-10)
    np.testing.assert_allclose(a(r.fittedvalues), m.predict(X), rtol=1e-10)
    assert r.deviance_ratio == pytest.approx(m.score(X, yp), rel=1e-9)


def test_lasso_cv_against_sklearn():
    lm = pytest.importorskip("sklearn.linear_model")
    y, X = data(n=150, seed=7)
    r = R.lasso_cv(y, X, rng=sd.random.default_rng(8))
    # the same folds and grid
    fold = a(
        sd.take(sd.asarray([i % 5 for i in range(150)], dtype=sd.int64), sd.random.default_rng(8).permutation(150))
    )
    splits = [(np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]) for k in range(5)]
    m = lm.LassoCV(alphas=a(r.alphas), cv=splits, tol=1e-14, max_iter=10**6).fit(X, y)
    np.testing.assert_allclose(a(r.alphas), m.alphas_, rtol=1e-15)
    np.testing.assert_allclose(a(r.cv_error), m.mse_path_.mean(axis=1), rtol=1e-9)
    assert r.alpha_min == m.alpha_ and r.alpha == r.alpha_min
    np.testing.assert_allclose(a(r.params), sk_params(m), rtol=1e-9, atol=1e-12)
    # the default grid is scikit-learn's: alpha_max eps^(i / (L - 1))
    m2 = lm.LassoCV(cv=splits).fit(X, y)
    np.testing.assert_allclose(a(r.alphas), m2.alphas_, rtol=1e-13)


# ---- paths and warm starts


def test_path_equals_single_fits():
    y, X = data(seed=9)
    grid = amax(y, X) * np.geomspace(1, 1e-3, 25)
    for fit in (
        lambda al: R.lasso(y, X, alpha=al),
        lambda al: R.elastic_net(y, X, alpha=al, l1_ratio=0.4),
        lambda al: R.ridge(y, X, alpha=al),
    ):
        path = fit(grid)
        single = np.stack([a(fit(float(v)).params) for v in grid])
        # the polished solution depends on the active set and signs alone: the same bits
        assert np.array_equal(a(path.params), single)
        # any order of the alphas (warm starts from the largest whatever the order)
        perm = np.random.default_rng(1).permutation(25)
        assert np.array_equal(a(fit(grid[perm]).params), single[perm])
        assert path.alpha.shape == (25,)


def test_warm_start_invariance_glm():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((200, 4))
    y = (rng.uniform(size=200) < 1 / (1 + np.exp(-X @ [1.0, -1.0, 0.5, 0.0]))).astype(float)
    grid = amax(y, X, family=1) * np.geomspace(1, 1e-2, 12)
    path = GLMNET(y, X, family="binomial", alpha=grid, l1_ratio=1.0)
    single = np.stack([a(GLMNET(y, X, family="binomial", alpha=float(v), l1_ratio=1.0).params) for v in grid])
    np.testing.assert_allclose(a(path.params), single, rtol=1e-13, atol=1e-15)
    assert np.array_equal(a(path.params) == 0, single == 0)


# ---- standardize


def test_standardize_semantics():
    y, X = data(seed=11)
    w = np.random.default_rng(12).uniform(0.5, 2.0, len(y))
    mean = (w[:, None] * X).sum(0) / w.sum()
    s = np.sqrt((w[:, None] * (X - mean) ** 2).sum(0) / w.sum())  # population sd (glmnet's)
    for fit in (
        lambda Z, **k: R.lasso(y, Z, alpha=0.05, weights=w, **k),
        lambda Z, **k: R.elastic_net(y, Z, alpha=0.05, l1_ratio=0.3, weights=w, **k),
        lambda Z, **k: R.ridge(y, Z, alpha=0.05, weights=w, **k),
    ):
        r = fit(X, standardize=True)
        z = fit(X / s)  # the penalty on the standardized coefficients
        np.testing.assert_allclose(a(r.params)[1:], a(z.params)[1:] / s, rtol=1e-12)
        np.testing.assert_allclose(a(r.params)[0], a(z.params)[0], rtol=1e-12)
        np.testing.assert_allclose(a(r.fittedvalues), a(z.fittedvalues), rtol=1e-12)
    # a scale changes nothing under standardize
    r1 = R.lasso(y, X, alpha=0.05, standardize=True)
    r2 = R.lasso(y, X * 8.0, alpha=0.05, standardize=True)
    np.testing.assert_allclose(a(r2.params)[1:] * 8.0, a(r1.params)[1:], rtol=1e-12)


# ---- cross-validation


def test_cv_determinism_and_rng():
    y, X = data(seed=13)
    g = sd.random.default_rng(5)
    r1 = R.lasso_cv(y, X, rng=g)
    assert g.state["position"] > 0  # the folds come from the generator's stream
    r2 = R.lasso_cv(y, X, rng=sd.random.default_rng(5))
    assert np.array_equal(a(r1.cv_error), a(r2.cv_error)) and r1.alpha_min == r2.alpha_min
    assert np.array_equal(a(r1.params), a(r2.params))
    assert r1.alphas.shape == (100,) and r1.cv_se.shape == (100,)
    # the minimum and the 1-SE rule
    e, s, grid = a(r1.cv_error), a(r1.cv_se), a(r1.alphas)
    i = int(np.argmin(e))
    assert r1.alpha_min == grid[i]
    assert r1.alpha_1se == grid[np.nonzero(e <= e[i] + s[i])[0][0]] and r1.alpha_1se >= r1.alpha_min
    r3 = R.lasso_cv(y, X, rng=sd.random.default_rng(5), select="1se")
    assert r3.alpha == r1.alpha_1se
    np.testing.assert_allclose(a(r3.params), a(R.lasso(y, X, alpha=r1.alpha_1se).params), rtol=0, atol=0)
    # the refit is the fit at the chosen alpha
    assert np.array_equal(a(r1.params), a(R.lasso(y, X, alpha=r1.alpha_min).params))
    en = R.elastic_net_cv(y, X, rng=sd.random.default_rng(5), l1_ratio=0.5, cv=4, alphas=20)
    assert en.alphas.shape == (20,) and en.l1_ratio == 0.5


def test_ridge_cv_gcv():
    y, X = data(seed=14)
    grid = np.geomspace(10, 1e-4, 30)
    r = R.ridge_cv(y, X, cv="gcv", alphas=grid)
    path = R.ridge(y, X, alpha=grid)
    # GCV = (ssr / sw) / (1 - tr H / n)^2, tr H = df_model + 1
    n = len(y)
    gcv = a(path.ssr) / n / (1 - (a(path.df_model) + 1) / n) ** 2
    np.testing.assert_allclose(a(r.cv_error), gcv, rtol=1e-10)
    assert r.alpha_min == grid[int(np.argmin(gcv))] and math.isnan(r.alpha_1se)
    assert np.all(np.isnan(a(r.cv_se)))
    k = R.ridge_cv(y, X, rng=sd.random.default_rng(0), cv=5, alphas=grid)
    assert k.alpha_min in grid and not math.isnan(k.alpha_1se)
    # the default grid: alpha_max at l1_ratio 1e-3 (glmnet's rule for ridge)
    d = R.ridge_cv(y, X, cv="gcv")
    assert d.alphas[0] == pytest.approx(amax(y, X, l1_ratio=1e-3), rel=1e-15)


def test_cv_batch_equals_loop():
    y, X = data(seed=15)
    Y = np.stack([y, 2 * y + 1, -y])
    r = R.lasso_cv(Y, X, rng=sd.random.default_rng(3), alphas=30)
    for i in range(3):
        s = R.lasso_cv(Y[i], X, rng=sd.random.default_rng(3), alphas=30)
        assert np.array_equal(a(r.cv_error)[i], a(s.cv_error))
        assert np.array_equal(a(r.params)[i], a(s.params))
        assert a(r.alpha_min)[i] == s.alpha_min


# ---- batches, threads, edge cases


def test_batch_equals_loop():
    y, X = data(seed=16)
    Y = np.stack([y, y + 1, -y])
    Xb = np.stack([X, X * 2, X[::-1]])
    for fit in (
        lambda yy, XX: R.lasso(yy, XX, alpha=[0.2, 0.02]),
        lambda yy, XX: R.ridge(yy, XX, alpha=0.1),
        lambda yy, XX: R.elastic_net(yy, XX, alpha=0.05),
    ):
        b = fit(Y, Xb)
        for i in range(3):
            s = fit(Y[i], Xb[i])
            assert np.array_equal(a(b.params)[i], a(s.params))
            assert np.array_equal(a(b.fittedvalues)[i], a(s.fittedvalues))
    yb = (y > np.median(y)).astype(float)
    b = GLMNET(np.stack([yb, 1 - yb]), X, family="binomial", alpha=0.01)
    for i, yy in enumerate((yb, 1 - yb)):
        assert np.array_equal(a(b.params)[i], a(GLMNET(yy, X, family="binomial", alpha=0.01).params))


def test_threads_bit_identical():
    rng = np.random.default_rng(17)
    n, p = 70000, 6
    X = rng.standard_normal((n, p)) + 1.0
    y = X @ [1.0, 0.0, -0.5, 0.0, 0.2, 0.0] + rng.standard_normal(n)
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X - 1.0) @ [0.5, 0.0, -0.3, 0.0, 0.1, 0.0]))).astype(float)
    grid = amax(y, X) * np.geomspace(1, 1e-2, 10)
    old = sd.get_num_threads()
    out = []
    try:
        for th in (1, 10):
            sd.set_num_threads(th)
            out.append(
                [
                    a(R.lasso(y, X, alpha=grid).params),
                    a(R.ridge(y, X, alpha=[1.0, 0.01]).params),
                    a(R.lasso_cv(y, X, rng=sd.random.default_rng(1), alphas=8).cv_error),
                    a(GLMNET(yb, X, family="binomial", alpha=1e-3).params),
                ]
            )
    finally:
        sd.set_num_threads(old)
    for u, v in zip(*out):
        assert np.array_equal(u, v)


def test_edge_cases():
    y, X = data(seed=18)
    # ridge at alpha 0 is least squares
    o = R.ols(y, X)
    np.testing.assert_allclose(a(R.ridge(y, X, alpha=0.0).params), a(o.params), rtol=1e-12)
    np.testing.assert_allclose(a(R.elastic_net(y, X, alpha=0.0).params), a(o.params), rtol=1e-12)
    assert R.ridge(y, X, alpha=0.0).df_model == pytest.approx(X.shape[1], rel=1e-14)
    # alpha >= alpha_max: every coefficient exactly 0, the intercept the mean
    am = amax(y, X)
    for al in (am, 2 * am):
        r = R.lasso(y, X, alpha=al)
        assert np.all(a(r.params)[1:] == 0.0) and r.n_iter == 0 and r.df_model == 0
        assert a(r.params)[0] == pytest.approx(y.mean(), rel=1e-15)
    # just below: one coefficient enters
    r = R.lasso(y, X, alpha=am * (1 - 1e-6))
    assert np.count_nonzero(a(r.params)[1:]) == 1 and r.converged
    # a copied column: the lasso's solutions are not unique (any split of the coefficient with
    # one sign); the polish puts the total on the first copy and the later one at 0 (R116), the
    # fit without the copy; ridge splits it evenly
    Xd = np.c_[X, X[:, 0]]
    r = R.lasso(y, Xd, alpha=0.05)
    one = R.lasso(y, X, alpha=0.05)
    pd = a(r.params)
    assert r.converged and pd[-1] == 0.0 and r.df_model == one.df_model
    np.testing.assert_allclose(pd[:-1], a(one.params), rtol=1e-13)
    rr = R.ridge(y, Xd, alpha=0.05)
    assert a(rr.params)[1] == pytest.approx(a(rr.params)[-1], rel=1e-10)
    # a constant column has coefficient 0
    Xc = X.copy()
    Xc[:, 2] = 3.25
    for fit in (R.lasso, R.ridge):
        assert a(fit(y, Xc, alpha=0.1).params)[3] == 0.0
        assert a(fit(y, Xc, alpha=0.1, standardize=True).params)[3] == 0.0
    # zero weights: the rows' results are those without them, bit for bit
    w = np.random.default_rng(1).uniform(0.5, 2.0, len(y))
    w[[2, 30]] = 0.0
    keep = w > 0
    for fit in (
        lambda yy, XX, ww: R.lasso(yy, XX, alpha=0.05, weights=ww),
        lambda yy, XX, ww: R.ridge(yy, XX, alpha=0.05, weights=ww),
        lambda yy, XX, ww: R.lasso_cv(yy, XX, rng=sd.random.default_rng(2), weights=ww, alphas=10),
    ):
        r1, r2 = fit(y, X, w), fit(y[keep], X[keep], w[keep])
        if hasattr(r1, "cv_error"):
            continue  # the folds are drawn over all the rows
        assert np.array_equal(a(r1.params), a(r2.params))
        assert r1.nobs == keep.sum() and r1.fittedvalues.shape == (len(y),)
        np.testing.assert_allclose(a(r1.resid)[keep], a(r2.resid), rtol=0, atol=0)
    # no intercept
    r = R.lasso(y, X, alpha=0.05, intercept=False)
    assert r.params.shape == (X.shape[1],) and r.param_names == [f"x{j + 1}" for j in range(X.shape[1])]
    # max_iter: the last iterate, converged False
    yk, Xk = data(seed=19)
    Xk[:, 1] = Xk[:, 0] + 1e-6 * Xk[:, 1]
    r = R.lasso(yk, Xk, alpha=1e-6, max_iter=1)
    assert not r.converged and r.n_iter == 1 and r.dual_gap > 0


def test_glm_edge_cases():
    rng = np.random.default_rng(31)
    X = rng.standard_normal((40, 2))
    # no finite fit: every y is 0
    r = GLMNET(np.zeros(40), X, family="poisson", alpha=0.1)
    assert np.all(np.isnan(a(r.params))) and not r.converged and r.nobs == 40
    # complete separation: finite for alpha > 0 (the L1 penalty alone too), separation False
    y = (X[:, 0] > 0).astype(float)
    for rho in (1.0, 0.0):
        r = GLMNET(y, X, family="binomial", alpha=0.01, l1_ratio=rho)
        assert r.converged and np.all(np.isfinite(a(r.params))) and r.separation is False
    # alpha >= alpha_max: the intercept-only fit
    am = amax(y, X, family=1)
    r = GLMNET(y, X, family="binomial", alpha=am, l1_ratio=1.0)
    assert np.all(a(r.params)[1:] == 0.0)
    assert a(r.params)[0] == pytest.approx(math.log(y.mean() / (1 - y.mean())), rel=1e-14)


def test_alpha_zero_is_glm():
    # G50: without a penalty, glm's rules: separation by its certificates, NaN coefficients
    rng = np.random.default_rng(32)
    X = rng.standard_normal((40, 2))
    y = (X[:, 0] > 0).astype(float)
    r = GLMNET(y, X[:, :1], family="binomial", alpha=0.0)
    assert r.separation is True and np.all(np.isnan(a(r.params)))
    g = R.glm(y, X[:, :1], family="binomial")
    assert g.separation
    np.testing.assert_array_equal(a(r.fittedvalues), a(g.fittedvalues))
    np.testing.assert_array_equal(a(r.linear_predictor), a(g.linear_predictor))
    assert r.deviance == g.deviance and math.isnan(r.dual_gap)
    # a path: separation only at alpha = 0
    p = GLMNET(y, X[:, :1], family="binomial", alpha=[0.1, 0.0])
    assert a(p.separation).tolist() == [0.0, 1.0] and np.all(np.isfinite(a(p.params)[0]))
    assert np.all(np.isnan(a(p.params)[1])) and np.array_equal(a(p.fittedvalues)[1], a(g.fittedvalues))
    # without separation: glm's estimate, its deviance and iterations
    yb = (rng.uniform(size=40) < 1 / (1 + np.exp(-X @ [0.5, -0.4]))).astype(float)
    yp = rng.poisson(np.exp(0.3 * X[:, 0])).astype(float)
    w = rng.uniform(0.5, 2.0, 40)
    w[3] = 0.0
    for fam, yy in (("binomial", yb), ("poisson", yp)):
        r = GLMNET(yy, X, family=fam, alpha=0.0, weights=w)
        g = R.glm(yy, X, family=fam, var_weights=w)
        assert r.separation is False and r.converged
        np.testing.assert_allclose(a(r.params), a(g.params), rtol=1e-15, atol=0)
        assert r.deviance == pytest.approx(g.deviance, rel=1e-15) and r.n_iter == g.n_iter
        # the penalized solution approaches it
        r2 = GLMNET(yy, X, family=fam, alpha=1e-12, weights=w)
        np.testing.assert_allclose(a(r2.params), a(g.params), rtol=1e-8)
    # a batch: separation per regression
    b = GLMNET(np.stack([y, yb]), X[:, :1], family="binomial", alpha=0.0)
    assert a(b.separation).tolist() == [True, False]
    assert np.all(np.isnan(a(b.params)[0])) and np.all(np.isfinite(a(b.params)[1]))


def test_fields():
    y, X = data(seed=20)
    r = R.lasso(y, X, alpha=0.05)
    assert isinstance(r.nobs, float) and isinstance(r.df_model, float) and isinstance(r.n_iter, int)
    assert isinstance(r.converged, bool) and isinstance(r.dual_gap, float) and isinstance(r.rsquared, float)
    assert r.df_model == np.count_nonzero(a(r.params)[1:])
    assert np.array_equal(a(r.fittedvalues), a(r.predict(X)))
    np.testing.assert_allclose(a(r.resid), y - a(r.fittedvalues), rtol=0, atol=1e-13)
    assert r.ssr == pytest.approx(float(np.sum(a(r.resid) ** 2)), rel=1e-14)
    assert r.alpha == 0.05 and r.l1_ratio == 1.0 and r.standardize is False
    assert not hasattr(r, "bse") and not hasattr(r, "pvalues")
    # the elastic net's effective degrees of freedom: tr Z_A (Z_A' Z_A / n + l2 I)^-1 Z_A' / n
    e = R.elastic_net(y, X, alpha=0.05, l1_ratio=0.4)
    A = np.nonzero(a(e.params)[1:])[0]
    Z = (X - X.mean(0))[:, A]
    n = len(y)
    H = Z.T @ Z / n + 0.05 * 0.6 * np.eye(len(A))
    assert e.df_model == pytest.approx(np.trace(np.linalg.solve(H, Z.T @ Z / n)), rel=1e-12)
    rr = R.ridge(y, X, alpha=0.05)
    Zc = X - X.mean(0)
    s = np.linalg.svd(Zc / np.sqrt(n), compute_uv=False)
    assert rr.df_model == pytest.approx(np.sum(s**2 / (s**2 + 0.05)), rel=1e-12)
    assert not hasattr(rr, "n_iter")
    # a path's fields
    pth = R.lasso(y, X, alpha=[0.1, 0.01])
    assert pth.params.shape == (2, 7) and pth.fittedvalues.shape == (2, len(y)) and pth.predict(X[:3]).shape == (2, 3)
    assert pth.rsquared.shape == (2,) and pth.converged.shape == (2,)
    g = GLMNET((y > y.mean()).astype(float), X, family="binomial", alpha=0.01)
    assert 0 < g.deviance_ratio < 1 and g.family == "binomial"
    assert np.all((a(g.fittedvalues) > 0) & (a(g.fittedvalues) < 1))
    np.testing.assert_allclose(a(g.predict(X, which="linear")), a(g.linear_predictor), rtol=0, atol=0)
    assert repr(r).startswith("LassoResult(alpha=0.05")


def test_nan_policies_and_frames():
    y, X = data(seed=21)
    yn = y.copy()
    yn[5] = np.nan
    r = R.lasso(yn, X, alpha=0.05)
    assert np.all(np.isnan(a(r.params))) and math.isnan(r.nobs)
    o = R.lasso(yn, X, alpha=0.05, nan_policy="omit")
    keep = ~np.isnan(yn)
    assert np.array_equal(a(o.params), a(R.lasso(y[keep], X[keep], alpha=0.05).params))
    with pytest.raises(ValueError, match=r"lasso\(\): the input contains NaN \(nan_policy='raise'\)"):
        R.lasso(yn, X, alpha=0.05, nan_policy="raise")
    b = R.ridge(np.stack([y, yn]), X, alpha=0.1)
    assert not np.any(np.isnan(a(b.params)[0])) and np.all(np.isnan(a(b.params)[1]))
    c = R.lasso_cv(np.stack([y, yn]), X, rng=sd.random.default_rng(0), alphas=10)
    assert math.isnan(a(c.alpha_min)[1]) and not math.isnan(a(c.alpha_min)[0])
    df = sd.frame.DataFrame({"a": X[:, 0], "b": X[:, 1]})
    r = R.lasso(y, df, alpha=0.01)
    assert r.param_names == ["const", "a", "b"]


def test_errors():
    y, X = data(n=30, seed=22)
    with pytest.raises(TypeError, match=r"lasso\(\): 'alpha' must be a number or a 1-dimensional array"):
        R.lasso(y, X, alpha="x")
    with pytest.raises(ValueError, match=r"lasso\(\): 'alpha' must be finite and >= 0"):
        R.lasso(y, X, alpha=-1.0)
    with pytest.raises(ValueError, match=r"'alpha' must be finite and >= 0"):
        R.ridge(y, X, alpha=[1.0, math.nan])
    with pytest.raises(ValueError, match=r"'alpha' must not be empty"):
        R.ridge(y, X, alpha=[])
    with pytest.raises(ValueError, match=r"'alpha' must be a number or a 1-dimensional array, not 2-dimensional"):
        R.ridge(y, X, alpha=[[1.0]])
    with pytest.raises(ValueError, match=r"elastic_net\(\): 'l1_ratio' must be between 0 and 1"):
        R.elastic_net(y, X, alpha=0.1, l1_ratio=1.5)
    with pytest.raises(ValueError, match=r"'max_iter' must be >= 1"):
        R.lasso(y, X, alpha=0.1, max_iter=0)
    with pytest.raises(ValueError, match=r"'tol' must be a finite number > 0"):
        R.lasso(y, X, alpha=0.1, tol=0.0)
    with pytest.raises(TypeError, match=r"'standardize' must be True or False"):
        R.lasso(y, X, alpha=0.1, standardize=1)
    with pytest.raises(ValueError, match=r"lasso\(\): 'weights' must be >= 0"):
        R.lasso(y, X, alpha=0.1, weights=-np.ones(30))
    Xi = X.copy()
    Xi[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"ridge\(\): 'X' contains infinity"):
        R.ridge(y, Xi, alpha=0.1)
    with pytest.raises(ValueError, match=r"'y' contains infinity"):
        R.lasso(np.r_[np.inf, y[1:]], X, alpha=0.1)
    with pytest.raises(TypeError, match=r"lasso_cv\(\): 'rng' must be an sd.random.Generator"):
        R.lasso_cv(y, X, rng=1)
    with pytest.raises(TypeError, match=r"ridge_cv\(\): 'rng' must be an sd.random.Generator"):
        R.ridge_cv(y, X, cv=5)
    with pytest.raises(ValueError, match=r"'cv' must be >= 2"):
        R.lasso_cv(y, X, rng=sd.random.default_rng(0), cv=1)
    with pytest.raises(ValueError, match=r"'cv' must be <= the number of rows \(30\), not 31"):
        R.lasso_cv(y, X, rng=sd.random.default_rng(0), cv=31)
    with pytest.raises(ValueError, match=r"ridge_cv\(\): 'cv' must be an int or 'gcv', not 'x'"):
        R.ridge_cv(y, X, cv="x")
    with pytest.raises(ValueError, match=r"'select' must be 'min' or '1se', not 'x'"):
        R.lasso_cv(y, X, rng=sd.random.default_rng(0), select="x")
    with pytest.raises(ValueError, match=r"select='1se' needs folds"):
        R.ridge_cv(y, X, cv="gcv", select="1se")
    with pytest.raises(ValueError, match=r"'eps' must be > 0 and <= 1"):
        R.lasso_cv(y, X, rng=sd.random.default_rng(0), eps=0.0)
    with pytest.raises(ValueError, match=r"the default grid needs l1_ratio > 0"):
        R.elastic_net_cv(y, X, rng=sd.random.default_rng(0), l1_ratio=0.0)
    yb = (y > y.mean()).astype(float)
    with pytest.raises(ValueError, match=rf"{NAME}\(\): 'family' must be 'binomial' or 'poisson', not 'gamma'"):
        GLMNET(yb, X, family="gamma", alpha=0.1)
    with pytest.raises(ValueError, match=rf"{NAME}\(\): 'y' must be 0 or 1 for the binomial family"):
        GLMNET(yb * 0.5, X, family="binomial", alpha=0.1)
    with pytest.raises(ValueError, match=r"'y' must be >= 0 for the poisson family"):
        GLMNET(yb - 1, X, family="poisson", alpha=0.1)
    r = R.lasso(y, X, alpha=0.1)
    with pytest.raises(ValueError, match=r"predict\(\): 'X_new' must have 6 columns like the fit's 'X', not 3"):
        r.predict(np.ones((2, 3)))
    with pytest.raises(ValueError, match=r"'which' must be 'mean' or 'linear'"):
        GLMNET(yb, X, family="binomial", alpha=0.1).predict(X, which="x")


def test_ridge_cv_default_grid_and_copies():
    """G87: ridge_cv's default grid reaches eps = 1e-6 of its alpha_max (the lasso's and the
    elastic net's keep 1e-3). G86: the fitted values read copies of y and X. G85: a constant y
    has R^2 0/0, NaN."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((80, 4))
    y = X @ [3.0, -2.0, 1.0, 0.5] + rng.standard_normal(80)
    r = R.ridge_cv(y, X, cv="gcv")
    g = a(r.alphas)
    assert g[-1] / g[0] == pytest.approx(1e-6, rel=1e-12)
    assert g[-1] < r.alpha < g[0]  # a clear signal's optimum inside the grid
    lc = R.lasso_cv(y, X, rng=sd.random.default_rng(0))
    assert a(lc.alphas)[-1] / a(lc.alphas)[0] == pytest.approx(1e-3, rel=1e-12)
    Xs, ys = sd.asarray(X), sd.asarray(y)
    f = R.ridge(ys, Xs, alpha=0.5)
    want = a(f.fittedvalues)
    Xs[0, 0] = 1e6
    ys[0] = 1e6
    assert a(f.fittedvalues).tolist() == want.tolist()
    c = R.ridge(np.ones(10), np.arange(10.0), alpha=0.1)
    assert math.isnan(c.rsquared)


def test_polish_retry_at_max_iter_is_not_converged():
    """G32: the gap test passed, the KKT polish failed and the tighter sweeps reached max_iter:
    the last (unpolished) iterate with converged=False (it said True)."""
    ra = sd.random.default_rng(1)
    Za = sd.stats.Normal().sample((40, 3), rng=ra)
    Xa = sd.stack([Za[:, 0], Za[:, 0] + 0.05 * Za[:, 1], Za[:, 2]], axis=1)
    ya = 1.0 + Xa[:, 0] + Xa[:, 1] + 0.5 * Xa[:, 2] + sd.stats.Normal().sample(40, rng=ra)
    r = R.lasso(ya, Xa, alpha=0.001, tol=0.1)
    assert not r.converged and r.n_iter == 1000
    path = R.lasso(ya, Xa, alpha=sd.asarray([0.1, 0.01, 0.001]), tol=0.1)
    assert a(path.converged).tolist() == [True, True, False]
    assert R.lasso(ya, Xa, alpha=0.001).converged


def test_constant_y_null_deviance():
    """G85: the null deviance at mu = ybar itself is exactly 0 for a constant y, so
    deviance_ratio is NaN (exp(log 3) left 5e-31 and a ratio of 0)."""
    x = np.arange(8.0)
    for v in (1.0, 2.0, 3.0, 5.0, 0.7, 1e5):
        for w in (None, np.arange(1.0, 9.0)):
            p = R.penalized_glm([v] * 8, x, family="poisson", alpha=0.1, weights=w)
            assert p.null_deviance == 0.0 and math.isnan(p.deviance_ratio)
        assert R.glm([v] * 8, x, family="poisson").null_deviance == 0.0


def _conv_invariant(fit, ref, n_iter_cap=None, tol=1e-12):
    """converged implies the coefficients of the full fit to rounding."""
    full = a(ref.params)
    for r, cap in fit:
        p = a(r.params)
        if n_iter_cap:
            assert r.n_iter <= cap
        if r.converged:
            assert np.max(np.abs(p - full)) <= tol * max(1.0, np.max(np.abs(full))), (cap, p, full)


def _run(code):
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


# ---- G93: converged needs the polish; n_iter <= max_iter


def test_penalized_glm_newton_start_capped():
    """l1_ratio = 0 starts with Newton's steps: capped by max_iter, IRLS goes on from where they
    stopped (it used to return the null model, with n_iter beyond max_iter)."""
    ref = R.penalized_glm(YB, X, family="binomial", alpha=0.01, l1_ratio=0.0)
    devs = []
    for m in range(1, 8):
        r = R.penalized_glm(YB, X, family="binomial", alpha=0.01, l1_ratio=0.0, max_iter=m)
        assert r.n_iter <= m
        devs.append(r.deviance)
        if r.converged:
            assert np.max(np.abs(a(r.params) - a(ref.params))) <= 1e-12
    assert all(d1 >= d2 - 1e-12 for d1, d2 in zip(devs, devs[1:]))
    null = R.penalized_glm(YB, X, family="binomial", alpha=1e6, l1_ratio=0.0).deviance
    assert devs[0] < null  # not the null model


def test_lasso_and_pglm_converged_invariant():
    Xd = np.column_stack([X, X[:, 0] + 1e-3 * RNG.normal(size=N)])
    ref = R.elastic_net(Y, Xd, alpha=0.05, l1_ratio=0.7)
    _conv_invariant(
        [(R.elastic_net(Y, Xd, alpha=0.05, l1_ratio=0.7, max_iter=m), m) for m in range(1, 40)],
        ref,
        n_iter_cap=True,
        tol=1e-10,
    )
    ref = R.penalized_glm(YB, X, family="binomial", alpha=0.01, l1_ratio=0.5)
    _conv_invariant(
        [(R.penalized_glm(YB, X, family="binomial", alpha=0.01, l1_ratio=0.5, max_iter=m), m) for m in range(1, 10)],
        ref,
        n_iter_cap=True,
        tol=1e-10,
    )


# ---- G107: empty folds


def test_cv_skips_empty_folds():
    Xs = np.arange(10.0).reshape(5, 2) ** 1.3
    y = np.array([1.0, 2, 0, 4, 3])
    w = np.array([1.0, 1, 1, 1, 0])  # cv = n: the fold of row 4 has no test weight
    for fit in (R.lasso_cv, R.ridge_cv):
        r = fit(y, Xs, rng=sd.random.default_rng(0), weights=w, cv=5, alphas=np.array([1.0, 0.1, 0.01]))
        assert math.isfinite(r.alpha) and np.all(np.isfinite(a(r.cv_error)))
        perm = a(sd.random.default_rng(0).permutation(5)).astype(int)
        fold = np.array([i % 5 for i in range(5)])[perm]
        errs = []
        for q in range(5):
            te = w * (fold == q)
            if te.sum() == 0:
                continue
            base = R.lasso if fit is R.lasso_cv else R.ridge
            m = base(y, Xs, alpha=np.array([1.0, 0.1, 0.01]), weights=w * (fold != q))
            errs.append(((y - a(m.predict(Xs))) ** 2 * te).sum(axis=1) / te.sum())
        errs = np.array(errs)
        np.testing.assert_allclose(a(r.cv_error), errs.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(a(r.cv_se), errs.std(axis=0, ddof=1) / math.sqrt(len(errs)), rtol=1e-10)


def test_wide_lasso_all_zero_and_cv_do_not_abort():
    out = _run("""
import numpy as np, stardust as sd
rng = np.random.default_rng(9)
X = rng.normal(size=(50, 1030)); y = X[:, 0] + rng.normal(size=50)
r = sd.regression.lasso(y, X, alpha=100.0)
assert r.converged and np.count_nonzero(np.asarray(r.params.tolist())[1:]) == 0
r = sd.regression.lasso_cv(y, X, rng=sd.random.default_rng(0), alphas=5)
assert np.isfinite(r.alpha)
r = sd.regression.elastic_net_cv(y, X, rng=sd.random.default_rng(0), alphas=4)
r = sd.regression.penalized_glm((y > 0).astype(float), X, family="binomial", alpha=100.0)
print("ok")
""")
    assert out.strip() == "ok"


def test_cv_zero_weight_fold_does_not_abort():
    out = _run("""
import numpy as np, stardust as sd
X = np.arange(10.0).reshape(5, 2) ** 1.3; y = np.array([1., 2, 0, 4, 3]); w = np.array([1., 1, 1, 1, 0])
r = sd.regression.lasso_cv(y, X, rng=sd.random.default_rng(0), weights=w, cv=5, alphas=3)
print("ok", np.isfinite(r.alpha))
""")
    assert out.strip() == "ok True"


def test_penalized_glm_naive_path_matches_active_columns():
    """More than 1,024 columns: the KKT check and the final values from per-column sums and the
    active columns' Gram (no (p + 3)^2 Gram). The solution is that of the active columns."""
    rng = np.random.default_rng(5)
    n, p = 120, 1100
    Xw = rng.normal(size=(n, p))
    y = (Xw[:, 0] - Xw[:, 1] + rng.normal(size=n) > 0).astype(float)
    r = R.penalized_glm(y, Xw, family="binomial", alpha=0.08, l1_ratio=1.0)
    assert r.converged
    pa = a(r.params)
    act = np.flatnonzero(pa[1:])
    assert 0 < len(act) < 50
    sub = R.penalized_glm(y, Xw[:, act], family="binomial", alpha=0.08, l1_ratio=1.0)
    np.testing.assert_allclose(a(sub.params), np.r_[pa[0], pa[1:][act]], rtol=1e-9, atol=1e-12)
    assert r.df_model == len(act) and abs(r.dual_gap) <= 1e-12
    en = R.penalized_glm(y, Xw, family="binomial", alpha=0.08, l1_ratio=0.5)
    pe = a(en.params)
    ae = np.flatnonzero(pe[1:])
    sube = R.penalized_glm(y, Xw[:, ae], family="binomial", alpha=0.08, l1_ratio=0.5)
    np.testing.assert_allclose(en.df_model, sube.df_model, rtol=1e-9)


# ---- messages, grids, the boundary's null deviance


def test_penalized_messages():
    with pytest.raises(OverflowError, match=r"^elastic_net\(\): 'l1_ratio' is out of range$"):
        R.elastic_net(Y, X, alpha=0.1, l1_ratio=10**400)
    with pytest.raises(OverflowError, match=r"^lasso\(\): 'tol' is out of range$"):
        R.lasso(Y, X, alpha=0.1, tol=10**400)
    with pytest.raises(OverflowError, match=r"^lasso_cv\(\): 'eps' is out of range$"):
        R.lasso_cv(Y, X, rng=gen(), eps=10**400)
    with pytest.raises(OverflowError, match=r"^conf_int\(\): 'alpha' is out of range$"):
        R.ols(Y, X).conf_int(alpha=10**400)
    with pytest.raises(
        TypeError,
        match=r"^lasso_cv\(\): 'alphas' must be an int \(the grid's length\) or a "
        r"1-dimensional array of numbers, not float$",
    ):
        R.lasso_cv(Y, X, rng=gen(), alphas=1.5)
    with pytest.raises(ValueError, match=r"^ridge_cv\(\): 'rng' is only used when 'cv' is a number of folds"):
        R.ridge_cv(Y, X, cv="gcv", rng=gen())


def test_penalized_glm_boundary_null_deviance():
    for y, fam in ((np.zeros(N), "binomial"), (np.ones(N), "binomial"), (np.zeros(N), "poisson")):
        r = R.penalized_glm(y, X, family=fam, alpha=0.1)
        assert r.null_deviance == 0.0 and not r.converged and np.all(np.isnan(a(r.params)))


def test_penalized_repr_names_the_axes():
    assert repr(R.lasso(np.vstack([Y, Y]), X, alpha=0.1)) == "LassoResult(batch shape (2,), 3 parameters)"
    assert repr(R.lasso(Y, X, alpha=[0.1, 0.2])) == "LassoResult(a path of 2 alphas, 3 parameters)"
    assert (
        repr(R.ridge(np.vstack([Y, Y]), X, alpha=[0.1, 0.2]))
        == "RidgeResult(batch shape (2,) and a path of 2 alphas, 3 parameters)"
    )


def test_alphas_messages_name_the_accepted_forms():
    form = r"'alphas' must be an int \(the grid's length\) or a 1-dimensional array"
    for v in (True, "x", 1.5, None, ["a"], np.array([True])):
        with pytest.raises(TypeError, match=r"^lasso_cv\(\): " + form + r" of numbers, not "):
            R.lasso_cv(Y, X, rng=gen(), alphas=v)
    with pytest.raises(ValueError, match=r"^lasso_cv\(\): " + form + r", not 2-dimensional$"):
        R.lasso_cv(Y, X, rng=gen(), alphas=[[0.1]])


def test_lasso_alpha_zero_is_ridges():
    """G110: lasso and elastic_net at alpha = 0 give ridge(alpha=0)'s minimum-norm solution bit
    for bit (p > n too, and in a path), converged, n_iter 0, dual_gap 0; df_model the number
    of nonzero coefficients (G115)."""
    rng = np.random.default_rng(1)
    for n, p in ((30, 4), (10, 15)):
        Xw = rng.normal(size=(n, p))
        Xw = np.column_stack([Xw, Xw[:, 0]])  # an aliased column too
        y = rng.normal(size=n)
        rr = R.ridge(y, Xw, alpha=0.0)
        for fit in (lambda **k: R.lasso(y, Xw, **k), lambda **k: R.elastic_net(y, Xw, l1_ratio=0.4, **k)):
            r = fit(alpha=0.0)
            assert a(r.params).tolist() == a(rr.params).tolist()
            assert r.converged and r.n_iter == 0 and r.dual_gap == 0.0
            assert r.df_model == float(np.count_nonzero(a(rr.params)[1:]))
            path = fit(alpha=[0.3, 0.0, 0.05])
            assert a(path.params)[1].tolist() == a(rr.params).tolist() and a(path.n_iter)[1] == 0
            assert a(fit(alpha=[0.3, 0.0, 0.05]).params)[0].tolist() == a(fit(alpha=0.3).params).tolist()
        cv = R.lasso_cv(y, Xw, rng=gen(), cv=3, alphas=np.array([1.0, 0.1, 0.0]))
        if cv.alpha == 0.0:
            assert a(cv.params).tolist() == a(rr.params).tolist()
    # a NaN regression stays NaN at alpha = 0
    yn = Y.copy()
    yn[0] = np.nan
    r = R.lasso(yn, X, alpha=0.0)
    assert np.all(np.isnan(a(r.params))) and not r.converged and r.n_iter == 0


def test_lasso_df_model_counts_nonzero_at_alpha_zero():
    """G115: df_model at alpha = 0 is the number of nonzero coefficients (a column and its copy
    both count), in a path, a CV refit and a batch with a grid containing 0."""
    rng = np.random.default_rng(5)
    Xd = rng.normal(size=(40, 3))
    Xd = np.column_stack([Xd, Xd[:, 1]])
    y = Xd @ np.array([1.0, 2.0, -1.0, 0.0]) + 0.01 * rng.normal(size=40)
    rr = R.ridge(y, Xd, alpha=0.0)
    assert rr.df_model == pytest.approx(3.0)  # ridge's: the rank
    for fit in (lambda **k: R.lasso(y, Xd, **k), lambda **k: R.elastic_net(y, Xd, l1_ratio=0.3, **k)):
        assert fit(alpha=0.0).df_model == 4.0
        assert a(fit(alpha=[0.1, 0.0]).df_model)[1] == 4.0
    cv = R.lasso_cv(np.stack([y, -y]), Xd, rng=gen(), cv=4, alphas=[1.0, 0.1, 0.0])
    assert a(cv.alpha).tolist() == [0.0, 0.0] and a(cv.df_model).tolist() == [4.0, 4.0]
    assert a(cv.n_iter).tolist() == [0, 0] and a(cv.params)[0].tolist() == a(rr.params).tolist()


@pytest.mark.serial  # compares two timings: alone, so a busy neighbour cannot skew one of them
def test_lasso_alpha_zero_runs_no_descent():
    """R111: alpha = 0 goes straight to ridge's solve: its time is ridge(alpha=0)'s, not the
    coordinate descent's (X 200 x 400, where the descent took 4 ms more than the solve)."""
    import time

    rng = np.random.default_rng(2)
    Xw, y = rng.normal(size=(200, 400)), rng.normal(size=200)

    def best(f):
        f()
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    t_ridge = best(lambda: R.ridge(y, Xw, alpha=0.0))
    t_lasso = best(lambda: R.lasso(y, Xw, alpha=0.0))
    assert t_lasso < 1.3 * t_ridge + 1e-3
    assert a(R.lasso(y, Xw, alpha=0.0).params).tolist() == a(R.ridge(y, Xw, alpha=0.0).params).tolist()


def test_lasso_copy_of_an_active_column_is_exactly_zero():
    """G116: a column whose gradient sits on l1 (a copy of an active one) stays exactly 0 in
    single fits and on a warm-started path, with standardize too; df_model counts it out."""
    rng = np.random.default_rng(0)
    Xz = np.column_stack([np.arange(10.0), np.arange(10.0), rng.standard_normal(10)])
    yz = 2 * Xz[:, 0] + Xz[:, 2] + 0.01 * rng.standard_normal(10)
    path = [0.1, 1e-2, 1e-3, 1e-4]
    for std in (False, True):
        P = a(R.lasso(yz, Xz, alpha=path, standardize=std).params)
        assert np.all(P[:, 2] == 0.0) and np.all(P[:, 1] != 0.0), P
        assert a(R.lasso(yz, Xz, alpha=path, standardize=std).df_model).tolist() == [2.0] * 4
        for i, al in enumerate(path):
            r = R.lasso(yz, Xz, alpha=al, standardize=std)
            assert float(r.params[2]) == 0.0 and r.df_model == 2.0
            if not std:  # the path and the single fit agree on A and the signs: the same bits
                assert a(r.params).tolist() == P[i].tolist()
    # the penalized GLM's inner descent shares the threshold
    yb = (yz > np.median(yz)).astype(float)
    g = a(R.penalized_glm(yb, Xz, family="binomial", alpha=[0.05, 0.01], l1_ratio=1.0).params)
    assert np.all(g[:, 1] * g[:, 2] == 0.0)


# ---- an intercept-only penalized model's R^2 is 0 (RG-6)


def test_intercept_only_rsquared_is_zero():
    y = np.random.default_rng(21).normal(size=15)
    X = np.empty((15, 0))
    assert R.ols(y, X).rsquared == 0.0
    for r in (
        R.ridge(y, X, alpha=0.1),
        R.lasso(y, X, alpha=0.1),
        R.lasso_cv(y, X, rng=sd.random.default_rng(0)),
        R.ridge_cv(y, X),
    ):
        assert r.rsquared == 0.0
    # every slope exactly 0 (alpha above alpha_max)
    Xs = np.random.default_rng(22).normal(size=(15, 2))
    assert R.lasso(y, Xs, alpha=10.0).rsquared == 0.0


# ---- R116, R118: exact copies of a column in the lasso


def test_lasso_copies_converge_and_agree():
    r = sd.random.default_rng(7)
    n = 12
    Z = sd.stats.Normal().sample((n, 5), rng=r)
    c = Z[:, 0]
    X = sd.stack([Z[:, 1], c, Z[:, 2], c, Z[:, 3] + 0.95 * c, Z[:, 4]], axis=1)
    y = X[:, 0] + X[:, 1] - 0.5 * X[:, 3] + X[:, 4] + 0.3 * sd.stats.Normal().sample(n, rng=r)
    s = R.lasso(y, X, alpha=0.001)
    assert s.converged and s.n_iter < 100 and abs(s.dual_gap) < 1e-14
    assert float(s.params[4]) == 0.0  # the later copy
    rz = sd.random.default_rng(7)
    xz, zz = sd.stats.Normal().sample(30, rng=rz), sd.stats.Normal().sample(30, rng=rz)
    yz = 1.0 + 2.0 * xz - zz + 0.1 * sd.stats.Normal().sample(30, rng=rz)
    Xz = sd.stack([xz, zz, xz], axis=1)
    for std in (False, True):
        one = R.lasso(yz, Xz, alpha=0.01, standardize=std)
        path = R.lasso(yz, Xz, alpha=sd.asarray([0.1, 0.01]), standardize=std)
        assert a(one.params).tolist() == a(path.params)[1].tolist()
        assert float(one.params[3]) == 0.0 and one.df_model == 2.0


def test_standardized_path_equals_single_fits():
    """G116: a path's solutions are the single fits', bit for bit, under standardize too."""
    rg = np.random.default_rng(11)
    X = rg.normal(size=(80, 5)) * [1.0, 10.0, 0.1, 3.0, 1.0]
    y = X @ [1.0, 0.1, 5.0, 0.0, -0.5] + rg.normal(size=80)
    alphas = [0.3, 0.1, 0.03, 0.01]
    for fit in (R.lasso, lambda *v, **k: R.elastic_net(*v, l1_ratio=0.4, **k)):
        p = fit(y, X, alpha=sd.asarray(alphas), standardize=True)
        for i, al in enumerate(alphas):
            assert a(p.params)[i].tolist() == a(fit(y, X, alpha=al, standardize=True).params).tolist()


# ---- D-18, D-19: ridge_cv's and theilslopes' defaults


def test_ridge_cv_default_is_gcv():
    rg = np.random.default_rng(6)
    X = rg.normal(size=(40, 3))
    y = X @ [1.0, 0.5, 0.0] + rg.normal(size=40)
    r = R.ridge_cv(y, X)
    g = R.ridge_cv(y, X, cv="gcv")
    assert r.alpha == g.alpha and math.isnan(r.alpha_1se)
    k = R.ridge_cv(y, X, cv=5, rng=sd.random.default_rng(0))
    assert not math.isnan(k.alpha_1se)
    with pytest.raises(TypeError, match=r"ridge_cv\(\): 'rng' must be an sd.random.Generator"):
        R.ridge_cv(y, X, cv=5)
