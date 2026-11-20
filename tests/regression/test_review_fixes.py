"""The fixes of the 2026-11-20 review (RG-1 to RG-8, R116-R118, CX-1, CX-4, CX-11, CX-12) and
the user's decisions D-1 to D-4, D-18 to D-20, D-29 (docs/design/regression.md §1.1)."""

import math

import numpy as np
import pytest

import stardust as sd

R = sd.regression


def a(v):
    return np.asarray(v)


def unix_time_data(seed=30, n=100, scale=1.7e12, span=8.64e7, slope=1e-6, noise=1.0):
    """x a Unix time in ms over a day, y a trend of 86 units with noise: a large offset column
    and a large intercept that cancels it (RG-1)."""
    rg = np.random.default_rng(seed)
    x = scale + np.sort(rg.uniform(0, span, n))
    y = slope * (x - scale) + noise * rg.normal(size=n)
    return y, x


# ---- D-1 (RG-1): the perfect fit's tolerance is (k + 16) u (|y_i| + sum_j |x_ij| |beta_j|)


def test_offset_column_is_not_a_perfect_fit():
    y, x = unix_time_data()
    xc = (x - 1.7e12) / 1e7
    r, c = R.ols(y, x), R.ols(y, xc)
    assert r.ssr > 100 and 0.99 < r.rsquared < 1.0 and r.llf < 0
    assert r.ssr == pytest.approx(c.ssr, rel=1e-6)
    assert a(r.bse)[1] == pytest.approx(a(c.bse)[1] / 1e7, rel=1e-6)
    assert np.all(a(r.pvalues) > 0)
    g = R.glm(y, x, family="gaussian")
    assert g.scale == pytest.approx(r.scale, rel=1e-10) and np.all(a(g.bse) > 0)
    lr = R.linregress(x, y)
    assert lr.rvalue < 1.0 and lr.stderr > 0
    h = R.ols(y, x, cov_type="HC1")
    assert np.all(a(h.bse) > 0)
    # rlm iterates (no zero scale) and its slope is the centered fit's
    m, mc = R.rlm(y, x), R.rlm(y, xc)
    assert m.n_iter > 0 and m.scale > 0 and m.scale == pytest.approx(mc.scale, rel=1e-6)
    assert a(m.params)[1] == pytest.approx(a(mc.params)[1] / 1e7, rel=1e-6)


def test_perfect_fit_still_found():
    x = np.arange(10.0)
    r = R.ols(1 + 2 * x + 3 * x**2, np.c_[x, x**2])
    assert r.ssr == 0.0 and r.scale == 0.0
    rr = R.rlm(np.r_[1 + 2 * x[:8], 50.0, -40.0], x)  # 8 of 10 rows on a line: a zero scale
    assert rr.scale == 0.0 and rr.converged


def test_quantreg_nid_on_an_offset_column():
    """RG-3: the 'nid' refits at q +- h moved no row on such a column (the breakpoints'
    tolerance had the same flaw as the snapping one): bse NaN. Now the reparametrized fit's."""
    y, x = unix_time_data()
    xc = (x - 1.7e12) / 1e7
    for ct in ("nid", "iid", "ker"):
        q, qc = R.quantreg(y, x, cov_type=ct), R.quantreg(y, xc, cov_type=ct)
        assert q.converged
        assert a(q.bse)[1] == pytest.approx(a(qc.bse)[1] / 1e7, rel=1e-6), ct


# ---- D-2 (RG-2) and G1: an underflow of mu or 1 - mu is not the boundary


def test_cloglog_reaches_the_maximum():
    sm = pytest.importorskip("statsmodels.api")
    rg = np.random.default_rng(20)
    n = 5000
    X = rg.normal(size=(n, 2))
    eta = 0.3 + X @ [3.0, -2.0]
    y = (rg.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    g = R.glm(y, X, family="binomial", link="cloglog")
    s = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial(sm.families.links.CLogLog())).fit(
        tol=1e-13, maxiter=300
    )
    assert g.converged
    np.testing.assert_allclose(a(g.params), s.params, rtol=1e-9)
    assert float(sd.max(g.linear_predictor)) > 6.62  # rows whose 1 - mu underflows
    assert g.deviance == pytest.approx(s.deviance, rel=1e-12)


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_separation_of_many_rows(link):
    rg = np.random.default_rng(18)
    X = rg.normal(size=(1000, 3))
    y = (X[:, 0] > 0).astype(float)
    g = R.glm(y, X, family="binomial", link=link)
    assert g.separation and np.all(np.isnan(a(g.params))) and g.deviance == 0.0
    x = np.r_[-np.linspace(0.001, 3, 100), np.linspace(0.001, 3, 100)]
    g = R.glm((x > 0).astype(float), x, family="binomial", link=link)
    assert g.separation


def test_poisson_underflow_is_valid():
    """A poisson log row whose mu underflows at the maximum (eta < -745: a zero count far out)
    is a valid iterate with its exact deviance, Pearson and working residuals; its score is 0
    to the doubles, so the fit is the one without it."""
    rg = np.random.default_rng(9)
    x = rg.uniform(0, 2, 40)
    y = rg.poisson(np.exp(0.5 + x)).astype(float)
    g0 = R.glm(y, x, family="poisson")
    g = R.glm(np.r_[y, 0.0], np.r_[x, -1000.0], family="poisson")
    assert g.converged and not g.separation and float(g.linear_predictor[-1]) < -745
    np.testing.assert_allclose(a(g.params), a(g0.params), rtol=1e-13)
    assert g.deviance == pytest.approx(g0.deviance, rel=1e-13)
    for v in (g.resid_pearson, g.resid_working, g.resid_deviance):
        assert np.all(np.isfinite(a(v)))


def test_boundary_stop_is_not_convergence():
    """G1: an iteration that ended on steps halved back into the valid range (the binomial log
    link's MLE on mu = 1) is not converged; nor are NaN coefficients."""
    rg = np.random.default_rng(2)
    n = 80
    X = rg.normal(size=(n, 2))
    y = (rg.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + X @ [0.5, -0.4])))).astype(float)
    g = R.glm(y, X, family="binomial", link="log")
    assert not g.converged and float(sd.max(g.fittedvalues)) > 1 - 1e-6


# ---- D-3 (RG-4): a perfect fit's t and p are NaN


def test_perfect_fit_t_and_p_are_nan():
    x = np.arange(1.0, 11.0)
    for r in (R.ols(-3 * x, x), R.glm(-3 * x, x, family="gaussian"), R.rlm(-3 * x, x)):
        assert np.all(a(r.bse) == 0.0)
        assert np.all(np.isnan(a(r.tvalues))) and np.all(np.isnan(a(r.pvalues)))
        ci = a(r.conf_int())
        assert np.array_equal(ci[:, 0], a(r.params)) and np.array_equal(ci[:, 1], a(r.params))
    assert math.isnan(R.linregress(x, -3 * x).pvalue)
    # a constant y (G92, G94): the intercept's t too
    for r in (R.ols(np.full(10, 2.0), x), R.glm(np.full(10, 2.0), x, family="gaussian")):
        assert np.all(np.isnan(a(r.tvalues)))
    # degrees of freedom left with a fixed scale: inference as usual
    p = R.glm(np.array([1.0, 3.0, 2.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0]), family="poisson")
    assert np.all(np.isfinite(a(p.pvalues)))


# ---- G2 (RG-5): power-of-two units for glm, rlm, quantreg and the penalized fits


def _rel(u, v):
    u, v = a(u), a(v)
    return float(np.max(np.abs(u - v) / np.abs(v)))


@pytest.mark.parametrize("s", [1e-200, 1e-160, 1e160, 1e250])
def test_scale_equivariance(s):
    rg = np.random.default_rng(16)
    n = 60
    X = rg.normal(size=(n, 2))
    y = 0.3 + X @ [1.5, -1.0] + 0.5 * rg.standard_t(3, n)
    for f in (
        lambda v: R.rlm(v, X),
        lambda v: R.rlm(v, X, scale_est="huber"),
        lambda v: R.rlm(v, X, method="MM", rng=sd.random.default_rng(1)),
        lambda v: R.quantreg(v, X),
        lambda v: R.quantreg(v, X, cov_type="iid"),
        lambda v: R.glm(v, X, family="gaussian"),
    ):
        b, r = f(y), f(y * s)
        assert r.converged == b.converged and r.n_iter == b.n_iter
        assert _rel(r.params, a(b.params) * s) < 1e-13
        assert _rel(r.tvalues, b.tvalues) < 1e-12 and _rel(r.pvalues, b.pvalues) < 1e-10
    yg = np.exp(0.2 + X @ [0.3, -0.2]) * rg.gamma(5.0, 0.2, n)
    for link, fac in (("inverse", 1 / s), ("identity", s)):
        b, r = R.glm(yg, X, family="gamma", link=link), R.glm(yg * s, X, family="gamma", link=link)
        assert _rel(r.params, a(b.params) * fac) < 1e-13 and _rel(r.pvalues, b.pvalues) < 1e-10
        assert r.scale == pytest.approx(b.scale, rel=1e-13)
    b, r = R.lasso(y, X, alpha=0.05), R.lasso(y * s, X, alpha=0.05 * s)
    assert r.converged and _rel(r.params, a(b.params) * s) < 1e-13
    assert r.rsquared == pytest.approx(b.rsquared, rel=1e-13)
    b, r = R.ridge(y, X, alpha=0.05), R.ridge(y * s, X, alpha=0.05)
    assert _rel(r.params, a(b.params) * s) < 1e-13 and r.rsquared == pytest.approx(b.rsquared, rel=1e-13)


def test_column_scale_equivariance_of_the_glm():
    rg = np.random.default_rng(3)
    X = rg.normal(size=(80, 2))
    y = (rg.uniform(size=80) < 1 / (1 + np.exp(-(0.2 + X @ [1.0, -0.5])))).astype(float)
    b = R.glm(y, X, family="binomial")
    for s in (1e-200, 1e200):
        r = R.glm(y, X * s, family="binomial")
        assert _rel(r.pvalues, b.pvalues) < 1e-12
        assert _rel(a(r.params)[1:] * s, a(b.params)[1:]) < 1e-13


def test_batch_of_mixed_scales():
    rg = np.random.default_rng(4)
    X = rg.normal(size=(50, 3))
    y = 1.0 + X @ [0.5, -0.3, 0.0] + 0.3 * rg.normal(size=50)
    Y = np.stack([y, y * 1e-200, y * 1e200])
    for f in (lambda v: R.elastic_net(v, X, alpha=0.05, l1_ratio=0.3), lambda v: R.rlm(v, X)):
        r = f(Y)
        for i in range(3):
            assert np.array_equal(a(r.params)[i], a(f(Y[i]).params))


# ---- G3 (RG-6): an intercept-only penalized model's R^2 is 0


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


# ---- D-4 (CX-1): the results' arrays are read-only views


def test_result_arrays_are_read_only():
    rg = np.random.default_rng(5)
    X = rg.normal(size=(30, 2))
    y = X @ [1.0, -1.0] + rg.normal(size=30)
    fits = [
        R.ols(y, X),
        R.glm(y, X, family="gaussian"),
        R.rlm(y, X),
        R.quantreg(y, X, q=[0.3, 0.6]),
        R.lasso(y, X, alpha=sd.asarray([0.1, 0.01])),
        R.lasso_cv(y, X, rng=sd.random.default_rng(1)),
    ]
    for r in fits:
        for name in ("params", "fittedvalues", "resid", "bse", "tvalues", "pvalues", "alphas", "cv_error", "q"):
            v = getattr(r, name, None)
            if isinstance(v, sd.ndarray):
                with pytest.raises(ValueError, match="read-only"):
                    v[(0,) * v.ndim] = 1e6
        if hasattr(r, "cov_params"):
            with pytest.raises(ValueError, match="read-only"):
                r.cov_params()[0, 0] = 1e6
    o = fits[0]
    before = a(o.bse).tolist()
    p = o.params.copy()  # a copy is writable
    p[0] = 100.0
    assert a(o.bse).tolist() == before and o.predict(X[:2]).tolist() == o.fittedvalues[:2].tolist()
    b = R.linregress(np.stack([X[:, 0], X[:, 1]]), y)
    with pytest.raises(ValueError, match="read-only"):
        b.slope[0] = 1.0


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


def test_theilslopes_alpha_is_a_significance_level():
    st = pytest.importorskip("scipy.stats")
    rg = np.random.default_rng(7)
    x = rg.normal(size=30)
    y = 2 * x + rg.normal(size=30)
    d = R.theilslopes(y, x)
    assert tuple(d) == tuple(R.theilslopes(y, x, alpha=0.95)) == tuple(R.theilslopes(y, x, alpha=0.05))
    s = st.theilslopes(y, x, alpha=0.95)
    assert d.low_slope == s.low_slope and d.high_slope == s.high_slope
    with pytest.raises(OverflowError, match=r"^theilslopes\(\): 'alpha' is out of range$"):
        R.theilslopes(y, x, alpha=10**400)
    with pytest.raises(TypeError, match=r"^theilslopes\(\): 'alpha' must be an int or a float, not str$"):
        R.theilslopes(y, x, alpha="0.05")


# ---- CX-11 (G10), DC-12 (G11): the input conversion


def test_asarray_value_errors_keep_their_words():
    with pytest.raises(ValueError, match=r"^linregress\(\): inhomogeneous nested sequence"):
        R.linregress([[1.0, 2.0], [3.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match=r"^ols\(\): 0-dimensional arrays are not supported"):
        R.ols(np.array(2.0), [1.0])
    with pytest.raises(TypeError, match=r"^ols\(\): 'y' must be an sd.ndarray, a sequence of numbers"):
        R.ols(["a", "b"], [1.0, 2.0])


def test_inputs_converted_once():
    """G86 keeps a copy of what the caller can change later, and only that: a list's or a
    dtype's conversion is new memory already (DC-12)."""
    from stardust.regression._linear import _as_real_owned, _prepare

    assert _as_real_owned([1.0, 2.0], "f", "y")[2]
    assert _as_real_owned(sd.asarray([1, 2]), "f", "y")[2]  # int64 -> float64 is new
    assert not _as_real_owned(sd.asarray([1.0, 2.0]), "f", "y")[2]
    assert not _as_real_owned(np.array([1.0, 2.0]), "f", "y")[2]  # asarray views numpy's memory
    y0 = sd.asarray([1.0, 2.0, 3.0])
    y, X, *_ = _prepare("f", y0, [[1.0], [2.0], [4.0]], None, "propagate", own=True)
    assert y is not y0
    yy = [1.0, 2.0, 3.0]
    y, X, *_ = _prepare("f", yy, sd.asarray([1.0, 2.0, 4.0]), None, "propagate", own=True)
    y0[0] = 9.0
    assert float(y[0]) == 1.0


# ---- D-29 (CX-12): the functions' call-shape errors


def test_function_call_shape_messages():
    y, X = [1.0, 2.0, 3.0, 5.0], [1.0, 2.0, 3.0, 4.0]
    cases = [
        (lambda: R.ols(y, X, foo=1), r"^ols\(\): unexpected keyword argument 'foo'$"),
        (lambda: R.ols(y, X, True), r"^ols\(\): takes 2 positional arguments but 3 were given$"),
        (lambda: R.wls(y, X), r"^wls\(\): missing 1 required keyword-only argument: 'weights'$"),
        (lambda: R.glm(y, X), r"^glm\(\): missing 1 required keyword-only argument: 'family'$"),
        (lambda: R.lasso(y, X), r"^lasso\(\): missing 1 required keyword-only argument: 'alpha'$"),
        (lambda: R.rlm(y), r"^rlm\(\): missing 1 required positional argument: 'X'$"),
        (lambda: R.quantreg(y, X, qq=0.5), r"^quantreg\(\): unexpected keyword argument 'qq'"),
        (lambda: R.theilslopes(y, X, 0.9), r"^theilslopes\(\): takes from 1 to 2 positional arguments but 3"),
        (lambda: R.linregress(x=X, y=y), r"^linregress\(\): got some positional-only arguments"),
    ]
    for f, msg in cases:
        with pytest.raises(TypeError, match=msg):
            f()
    # an error from inside the function keeps its own words
    with pytest.raises(TypeError, match=r"^ols\(\): 'intercept' must be True or False"):
        R.ols(y, X, intercept=1)
