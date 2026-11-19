"""The code review of 2026-11-19 and its user decisions G93-G108 (docs/design/regression.md
§1.1): convergence with the polish, constant y, df_resid = 0 and the perfect fit, empty
inputs, read-only fields, conf_int, the GLM's pseudo R^2, models without parameters,
linregress' nan_policy, the robust, quantile and nonlinear prediction intervals, nls's
prediction inputs, a one-column y, the cross-validation's empty folds, one cluster with a
constant y; and the fixes the review found (crashes, messages, extreme scales, slopes)."""

import math
import warnings

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd
from stardust.regression._nls import _S_STATUS

R = sd.regression
K = sd._core.regression


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def gen():
    return sd.random.default_rng(3)


RNG = np.random.default_rng(20261119)
N = 60
X = RNG.normal(size=(N, 2))
Y = 1.0 + X @ [1.0, -2.0] + RNG.normal(size=N)
YB = (RNG.uniform(size=N) < 1 / (1 + np.exp(-(0.3 + X @ [1.0, -0.7])))).astype(float)
YP = RNG.poisson(np.exp(0.2 + 0.4 * X[:, 0])).astype(float)


def decay(x, a_, k, c):
    return a_ * sd.exp(-k * x) + c


def decay_jac(x, a_, k, c):
    e = sd.exp(-k * x)
    return sd.stack([e, -a_ * x * e, sd.ones(x.shape)], axis=1)


TD = np.linspace(0, 4, 40)
YD = 2.5 * np.exp(-1.3 * TD) + 0.5 + 0.01 * RNG.normal(size=40)


# ---- G93: converged needs the polish; n_iter <= max_iter

def _conv_invariant(fit, ref, n_iter_cap=None, tol=1e-12):
    """converged implies the coefficients of the full fit to rounding."""
    full = a(ref.params)
    for r, cap in fit:
        p = a(r.params)
        if n_iter_cap:
            assert r.n_iter <= cap
        if r.converged:
            assert np.max(np.abs(p - full)) <= tol * max(1.0, np.max(np.abs(full))), (cap, p, full)


def test_glm_converged_means_polished():
    for fam, link, y in (("binomial", "probit", YB), ("poisson", "sqrt", YP), ("binomial", "logit", YB)):
        ref = R.glm(y, X, family=fam, link=link)
        assert ref.converged
        fits = [(R.glm(y, X, family=fam, link=link, max_iter=m), m) for m in range(1, 12)]
        _conv_invariant(fits, ref, n_iter_cap=True)
        # some cap stops the polish after the deviance's test passed: not converged there
        assert any(not r.converged for r, _ in fits)


def test_rlm_converged_means_polished():
    y = Y.copy()
    y[:6] += 8
    for kw in (dict(), dict(norm="bisquare"), dict(scale_est="huber")):
        ref = R.rlm(y, X, **kw)
        assert ref.converged
        _conv_invariant([(R.rlm(y, X, max_iter=m, **kw), m) for m in range(1, 15)], ref, n_iter_cap=True, tol=1e-11)
    ref = R.rlm(y, X, method="MM", rng=gen())
    assert ref.converged
    _conv_invariant([(R.rlm(y, X, method="MM", rng=gen(), max_iter=m), m) for m in range(1, 10)], ref,
                    n_iter_cap=True, tol=1e-11)


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
    _conv_invariant([(R.elastic_net(Y, Xd, alpha=0.05, l1_ratio=0.7, max_iter=m), m) for m in range(1, 40)], ref,
                    n_iter_cap=True, tol=1e-10)
    ref = R.penalized_glm(YB, X, family="binomial", alpha=0.01, l1_ratio=0.5)
    _conv_invariant([(R.penalized_glm(YB, X, family="binomial", alpha=0.01, l1_ratio=0.5, max_iter=m), m)
                     for m in range(1, 10)], ref, n_iter_cap=True, tol=1e-10)


def test_nls_polish_counts_against_max_nfev():
    ref = R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], jac=decay_jac)
    assert ref.converged
    seen_cut = False
    for m in range(3, 40):
        r = R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], jac=decay_jac, max_nfev=m)
        if r.converged:
            assert np.max(np.abs(a(r.params) - a(ref.params))) <= 1e-12 * np.max(np.abs(a(ref.params)))
        seen_cut = seen_cut or (r._count_stat(_S_STATUS) > 0 and not r.converged)  # the iteration converged
    assert seen_cut


# ---- G94: constant y

def test_constant_y_exact_solution():
    yc = np.full(N, 2.0)
    g = R.glm(yc, X, family="gaussian")
    assert a(g.params).tolist() == [2.0, 0.0, 0.0] and np.all(a(g.fittedvalues) == 2.0)
    assert a(g.bse).tolist() == [0.0, 0.0, 0.0] and math.isnan(g.tvalues[1]) and math.isnan(g.pvalues[1])
    assert g.deviance == 0.0 and g.null_deviance == 0.0 and g.llf == math.inf and g.llnull == math.inf
    assert math.isnan(g.llr) and math.isnan(g.llr_pvalue) and math.isnan(g.pseudo_rsquared())
    gh = R.glm(yc, X, family="gaussian", cov_type="HC1")
    assert a(gh.bse).tolist() == [0.0, 0.0, 0.0]
    for kw in (dict(), dict(method="MM", rng=gen()), dict(scale_est="huber")):
        r = R.rlm(yc, X, **kw)
        assert a(r.params).tolist() == [2.0, 0.0, 0.0] and r.scale == 0.0 and r.converged and r.n_iter == 0
        assert math.isnan(r.tvalues[1]) and math.isnan(r.pvalues[1]) and np.all(a(r.weights) == 1.0)
    q = R.quantreg(yc, X)
    assert a(q.params).tolist() == [2.0, 0.0, 0.0] and math.isnan(q.prsquared)
    # gamma and gaussian log: a fit that happens to be exact has scale 0, llf +inf (G97)
    for fam, link in (("gamma", "log"), ("gaussian", "log"), ("gamma", "inverse")):
        r = R.glm(np.full(10, 2.0), np.arange(10.0), family=fam, link=link)
        assert r.scale == 0.0 and r.llf == math.inf and r.aic == -math.inf and math.isnan(r.llr)


def test_glm_zero_scale_llnull_minus_inf():
    """phi = 0 with a null model that does not fit exactly: llnull's limit is -inf (G97)."""
    x = np.arange(1.0, 9.0)
    r = R.glm(1.0 + 2.0 * x, x, family="gaussian", link="identity", scale="X2")
    assert r.scale == 0.0 and r.llf == math.inf
    g = R.glm(np.exp(0.5 * x), x, family="gamma", link="log")
    if g.scale == 0.0:
        assert g.llf == math.inf and g.llnull == -math.inf and g.llr == math.inf


# ---- G95: df_resid = 0, the perfect fit's tolerance

def test_df_resid_zero_is_nan():
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([1.0, 3.0, 2.0])
    Xs = np.column_stack([x, x ** 2])
    for ct in ("nonrobust", "HC0", "HC1", "HC3"):
        r = R.ols(y, Xs, cov_type=ct)
        assert r.df_resid == 0 and math.isnan(r.scale)
        assert np.all(np.isnan(a(r.bse))) and np.all(np.isnan(a(r.cov_params()))) and np.all(np.isnan(a(r.pvalues)))
        assert np.all(np.isnan(a(r.conf_int()))) and np.all(np.isnan(a(r.predict_interval(Xs))))
    g = R.glm(y, Xs, family="gaussian")
    assert math.isnan(g.scale) and np.all(np.isnan(a(g.bse)))
    p = R.glm(np.array([1.0, 3.0, 2.0]), Xs, family="poisson")  # a fixed scale: not 0/0
    assert p.scale == 1.0 and np.all(np.isfinite(a(p.bse)))
    for kw in (dict(), dict(method="MM", rng=gen()), dict(scale_est="huber")):
        r = R.rlm(y, Xs, **kw)
        assert r.scale == 0.0 and r.n_iter == 0 and r.converged and np.all(np.isnan(a(r.bse)))
        assert np.max(np.abs(a(r.fittedvalues) - y)) <= 1e-14
    n = R.nls(lambda t, b0, b1, b2: b0 + b1 * t + b2 * t * t, x, y, p0=[0.0, 0.0, 0.0])
    assert n.df_resid == 0 and math.isnan(n.scale) and np.all(np.isnan(a(n.bse)))


def test_perfect_fit_by_tolerance():
    """y = 1 + 2x + 3x^2 exactly: the QR leaves residuals of rounding size; within the
    (k + 16) u tolerance they are a perfect fit (ssr 0, bse 0, p 0, llf +inf, diagnostics NaN)."""
    x = np.arange(10.0)
    y = 1 + 2 * x + 3 * x ** 2
    Xs = np.column_stack([x, x ** 2])
    r = R.ols(y, Xs)
    assert r.ssr == 0.0 and r.scale == 0.0 and np.all(a(r.bse) == 0.0) and np.all(a(r.pvalues) == 0.0)
    assert r.llf == math.inf and r.rsquared == 1.0
    assert math.isnan(r.durbin_watson) and np.all(np.isnan(a(r.resid_studentized_internal)))
    assert np.max(np.abs(a(r.resid))) <= 1e-12  # the residuals themselves stay the computed ones
    rh = R.ols(y, Xs, cov_type="HC0")
    assert np.all(a(rh.bse) == 0.0)
    w = R.wls(y, Xs, weights=np.linspace(1, 2, 10))
    assert w.ssr == 0.0
    g = R.glm(y, Xs, family="gaussian")
    assert g.deviance == 0.0 and g.llf == math.inf
    # a fit that is not perfect stays as it was
    y2 = y + 1e-9 * np.sin(x)
    assert R.ols(y2, Xs).ssr > 0


# ---- G96: empty inputs

def _empty_ok(r):
    assert r.nobs == 0
    assert math.isnan(r.df_model)
    if hasattr(r, "df_resid"):
        assert math.isnan(r.df_resid)
    if hasattr(r, "converged"):
        assert r.converged is False and r.n_iter == 0
    assert np.all(np.isnan(a(r.params)))
    assert np.all(np.isnan(a(r.fittedvalues)))


FITS = {
    "ols": lambda y, Xx, **k: R.ols(y, Xx, **k),
    "wls": lambda y, Xx, **k: R.wls(y, Xx, weights=np.ones(len(y)), **k),
    "gls": lambda y, Xx, **k: R.gls(y, Xx, sigma=np.eye(len(y)), **k),
    "glm": lambda y, Xx, **k: R.glm(y, Xx, family="gaussian", **k),
    "ridge": lambda y, Xx, **k: R.ridge(y, Xx, alpha=0.1, **k),
    "lasso": lambda y, Xx, **k: R.lasso(y, Xx, alpha=0.1, **k),
    "lasso_cv": lambda y, Xx, **k: R.lasso_cv(y, Xx, rng=gen(), cv=2, **k),
    "ridge_cv": lambda y, Xx, **k: R.ridge_cv(y, Xx, cv="gcv", **k),
    "penalized_glm": lambda y, Xx, **k: R.penalized_glm(np.abs(y), Xx, family="poisson", alpha=0.1, **k),
    "rlm": lambda y, Xx, **k: R.rlm(y, Xx, **k),
    "rlm_MM": lambda y, Xx, **k: R.rlm(y, Xx, method="MM", rng=gen(), **k),
    "quantreg": lambda y, Xx, **k: R.quantreg(y, Xx, **k),
    "nls": lambda y, Xx, **k: R.nls(lambda t, b0, b1: b0 + b1 * t, Xx[:, 0], y, p0=[1.0, 1.0], **k),
}


@pytest.mark.parametrize("name", list(FITS))
def test_empty_input(name):
    fit = FITS[name]
    _empty_ok(fit(np.zeros(0), np.zeros((0, 2))))
    _empty_ok(fit(np.full(6, np.nan), X[:6], nan_policy="omit"))


def test_zero_weights_are_empty():
    w0 = np.zeros(N)
    for r in (R.wls(Y, X, weights=w0), R.glm(Y, X, family="gaussian", var_weights=w0),
              R.glm(YP, X, family="poisson", freq_weights=w0), R.lasso(Y, X, alpha=0.1, weights=w0),
              R.lasso_cv(Y, X, rng=gen(), cv=3, weights=w0), R.ridge(Y, X, alpha=0.1, weights=w0),
              R.penalized_glm(YP, X, family="poisson", alpha=0.1, weights=w0),
              R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], weights=np.zeros(40))):
        _empty_ok(r)
    # in a batch only that regression is NaN
    rb = R.wls(Y, X, weights=np.stack([np.ones(N), w0]))
    assert a(rb.nobs).tolist() == [N, 0.0] and np.all(np.isfinite(a(rb.params)[0]))
    assert np.all(np.isnan(a(rb.params)[1])) and math.isnan(a(rb.df_model)[1])


def test_linregress_and_slopes_too_few_points():
    for f in (lambda: R.linregress([1.0], [2.0]), lambda: R.linregress([], []),
              lambda: R.theilslopes([1.0]), lambda: R.siegelslopes([1.0, np.nan], nan_policy="omit")):
        with pytest.raises(ValueError, match=r"'x' and 'y' must have length at least 2$"):
            f()


# ---- G98, G99: read-only configuration, conf_int's keyword

def test_read_only_fields():
    cases = [(R.ols(Y, X), ("param_names", "cov_type")), (R.glm(YB, X, family="binomial"), ("family", "link")),
             (R.lasso(Y, X, alpha=0.1), ("alpha", "l1_ratio", "standardize", "param_names")),
             (R.penalized_glm(YB, X, family="binomial", alpha=0.1), ("family",)),
             (R.rlm(Y, X), ("method", "norm", "cov_type")), (R.quantreg(Y, X, q=[0.3, 0.6]), ("q", "cov_type")),
             (R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0]), ("method", "param_names"))]
    for r, names in cases:
        for nm in names:
            old = getattr(r, nm)
            with pytest.raises(AttributeError):
                setattr(r, nm, old)


def test_conf_int_keyword_only():
    for r in (R.ols(Y, X), R.glm(YB, X, family="binomial"), R.rlm(Y, X), R.quantreg(Y, X),
              R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0])):
        with pytest.raises(TypeError):
            r.conf_int(0.1)
        assert a(r.conf_int(alpha=0.1)).shape[-1] == 2


def test_quantreg_q_is_a_copy():
    q = sd.asarray([0.25, 0.75])
    r = R.quantreg(Y, X, q=q)
    q[0] = 0.5
    assert a(r.q).tolist() == [0.25, 0.75]


# ---- G100: pseudo R^2

def test_pseudo_rsquared_like_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    for fam, y, F in (("binomial", YB, sm.families.Binomial()), ("poisson", YP, sm.families.Poisson())):
        r = R.glm(y, X, family=fam)
        s = sm.GLM(y, sm.add_constant(X), family=F).fit()
        assert r.pseudo_rsquared() == pytest.approx(s.pseudo_rsquared(), rel=1e-12)
        assert r.pseudo_rsquared(kind="mcf") == pytest.approx(s.pseudo_rsquared(kind="mcf"), rel=1e-12)
    assert "Pseudo R-squ. (CS):" in R.glm(YB, X, family="binomial").summary()
    with pytest.raises(ValueError, match=r"'kind' must be 'cs' or 'mcf', not 'x'"):
        R.glm(YB, X, family="binomial").pseudo_rsquared(kind="x")


# ---- G101: no parameters

def test_no_parameters_everywhere():
    Z = np.zeros((N, 0))
    for f in (lambda: R.ols(Y, Z, intercept=False), lambda: R.wls(Y, Z, weights=np.ones(N), intercept=False),
              lambda: R.gls(Y, Z, sigma=np.eye(N), intercept=False),
              lambda: R.glm(YP, Z, family="poisson", intercept=False),
              lambda: R.ridge(Y, Z, alpha=0.1, intercept=False), lambda: R.lasso(Y, Z, alpha=0.1, intercept=False),
              lambda: R.lasso_cv(Y, Z, rng=gen(), intercept=False),
              lambda: R.penalized_glm(YP, Z, family="poisson", alpha=0.1, intercept=False),
              lambda: R.rlm(Y, Z, intercept=False), lambda: R.quantreg(Y, Z, intercept=False)):
        with pytest.raises(ValueError, match=r"the model has no parameters \(no columns in 'X' and intercept=False\)"):
            f()
    with pytest.raises(ValueError, match=r"^nls\(\): the model has no parameters \('p0' is empty\)$"):
        R.nls(lambda t: t, TD, YD, p0=[])


# ---- G102: linregress' nan_policy

def test_linregress_nan_policy():
    x = np.arange(10.0)
    y = 2 * x + np.sin(x)
    yn = y.copy()
    yn[3] = np.nan
    got = R.linregress(x, yn, nan_policy="omit")
    want = R.linregress(np.delete(x, 3), np.delete(y, 3))
    assert tuple(got) == tuple(want)
    assert all(math.isnan(v) for v in R.linregress(x, yn))
    with pytest.raises(ValueError, match=r"^linregress\(\): the input contains NaN \(nan_policy='raise'\)$"):
        R.linregress(x, yn, nan_policy="raise")
    with pytest.raises(ValueError, match=r"nan_policy='omit' takes one regression, not a batch"):
        R.linregress(np.vstack([x, x]), np.vstack([yn, y]), nan_policy="omit")
    with pytest.raises(ValueError, match=r"'nan_policy' must be 'propagate', 'omit' or 'raise', not 'x'"):
        R.linregress(x, y, nan_policy="x")
    # a constant y does not hide the NaN of x (propagate): all NaN, not the slope 0
    xn = x.copy()
    xn[2] = np.nan
    assert all(math.isnan(v) for v in R.linregress(xn, np.full(10, 3.0)))
    b = R.linregress(np.vstack([xn, x]), np.full((2, 10), 3.0))
    assert math.isnan(float(b.slope[0])) and float(b.slope[1]) == 0.0
    # the intercept of a constant -0.0 is +0.0 (G84)
    r = R.linregress(x, np.full(10, -0.0))
    assert math.copysign(1.0, r.intercept) == 1.0 and math.copysign(1.0, r.slope) == 1.0
    # inf is an error also next to a NaN under 'omit'
    yi = yn.copy()
    yi[4] = np.inf
    with pytest.raises(ValueError, match=r"'y' contains infinity"):
        R.linregress(x, yi, nan_policy="omit")


# ---- G103: prediction intervals of rlm, quantreg, nls

def test_rlm_quantreg_predict_interval():
    Xn = RNG.normal(size=(5, 2))
    for r in (R.rlm(Y, X), R.rlm(Y, X, method="MM", rng=gen()), R.quantreg(Y, X, q=0.3)):
        pi = a(r.predict_interval(Xn, alpha=0.1))
        Xf = np.column_stack([np.ones(5), Xn])
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xf, a(r.cov_params()), Xf))
        q = st.t.ppf(0.95, r.df_resid)
        mean = a(r.predict(Xn))
        np.testing.assert_allclose(pi[:, 0], mean - q * se, rtol=1e-12)
        np.testing.assert_allclose(pi[:, 1], mean + q * se, rtol=1e-12)
        with pytest.raises(TypeError):
            r.predict_interval(Xn, kind="observation")
    rq = R.quantreg(Y, X, q=[0.25, 0.75])
    assert a(rq.predict_interval(Xn)).shape == (2, 5, 2)
    rb = R.rlm(np.vstack([Y, -Y]), X)
    assert a(rb.predict_interval(Xn)).shape == (2, 5, 2)


def test_nls_predict_interval_delta_method():
    r = R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], jac=decay_jac)
    tn = np.array([0.1, 1.0, 3.5])
    pi = a(r.predict_interval(tn, alpha=0.05))
    th = a(r.params)
    J = np.column_stack([np.exp(-th[1] * tn), -th[0] * tn * np.exp(-th[1] * tn), np.ones(3)])
    se = np.sqrt(np.einsum("ij,jk,ik->i", J, a(r.cov_params()), J))
    mean = a(r.predict(tn))
    q = st.t.ppf(0.975, r.df_resid)
    np.testing.assert_allclose(pi, np.column_stack([mean - q * se, mean + q * se]), rtol=1e-12)
    # the differences' Jacobian at the solution gives the same to their accuracy
    rd = R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0])
    np.testing.assert_allclose(a(rd.predict_interval(tn)), pi, rtol=1e-8)


# ---- G104: nls's x_new

def test_nls_predict_inputs():
    df = sd.frame.DataFrame({"t": TD.tolist(), "u": (TD * 0).tolist()})

    def f2(x, a_, k, c):
        return a_ * sd.exp(-k * x[:, 0]) + c + 0 * x[:, 1]
    r = R.nls(f2, df, YD, p0=[1.0, 1.0, 0.0])
    swapped = sd.frame.DataFrame({"u": [0.0, 0.0], "t": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"'x_new' must have the columns \['t', 'u'\] in the fit's order"):
        r.predict(swapped)
    with pytest.raises(ValueError, match=r"^predict\(\): 'x_new' contains infinity$"):
        r.predict(np.array([[np.inf, 0.0]]))
    with pytest.raises(ValueError, match=r"^predict_interval\(\): 'x_new' contains infinity$"):
        r.predict_interval(np.array([[np.inf, 0.0]]))
    assert a(r.predict(np.array([[1.0, 0.0]]))).shape == (1,)


# ---- G105: a one-column y

def test_one_column_dataframe_y():
    dfy = sd.frame.DataFrame({"y": Y.tolist()})
    for fit in (lambda y: R.ols(y, X), lambda y: R.glm(y, X, family="gaussian"), lambda y: R.rlm(y, X),
                lambda y: R.quantreg(y, X), lambda y: R.lasso(y, X, alpha=0.1)):
        assert a(fit(dfy).params).tolist() == a(fit(Y).params).tolist()
    assert R.linregress(X[:, 0], dfy) == R.linregress(X[:, 0], Y)
    with pytest.raises(ValueError, match=r"^ols\(\): 'X' must have 1 row like 'y', not 60 \('y' of shape \(60, 1\) "
                                         r"is a batch of regressions with 1 row each"):
        R.ols(sd.asarray(Y.reshape(-1, 1)), X)
    with pytest.raises(ValueError, match=r"^ols\(\): 'X' must have 5 rows like 'y', not 4$"):
        R.ols(Y[:5], X[:4])


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


# ---- G108: one cluster and a constant y

def test_one_cluster_constant_y():
    r = R.ols(np.full(N, 2.0), X, cov_type="cluster", groups=np.zeros(N))
    assert np.all(np.isnan(a(r.cov_params()))) and np.all(np.isnan(a(r.bse)))
    r2 = R.ols(np.full(N, 2.0), X, cov_type="cluster", groups=np.arange(N) % 3)
    assert np.all(a(r2.bse) == 0.0)
    g = R.glm(np.full(N, 2.0), X, family="gaussian", cov_type="cluster", groups=np.zeros(N))
    assert np.all(np.isnan(a(g.bse)))


# ---- the review's fixes: crashes of wide penalized fits (run in a subprocess: they aborted)

def _run(code):
    import subprocess
    import sys
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


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


# ---- GLM: frequency weights' HC2/HC3 and HAC lags; predict's offset

def test_glm_freq_weights_hc2_hc3_are_the_repeated_data():
    f = RNG.integers(1, 4, size=N).astype(float)
    Xr, yr = np.repeat(X, f.astype(int), axis=0), np.repeat(YP, f.astype(int))
    for ct in ("HC0", "HC2", "HC3"):
        r = R.glm(YP, X, family="poisson", freq_weights=f, cov_type=ct)
        rr = R.glm(yr, Xr, family="poisson", cov_type=ct)
        np.testing.assert_allclose(a(r.bse), a(rr.bse), rtol=1e-11)


def test_glm_hac_default_lags_from_nobs():
    f = np.full(N, 50.0)
    r = R.glm(YP, X, family="poisson", freq_weights=f, cov_type="HAC")
    want = math.floor(4 * (f.sum() / 100) ** (2 / 9))
    assert f"using {want} lags" in " ".join(r.summary().split())


def test_glm_predict_offset_inf():
    r = R.glm(YP, X, family="poisson")
    for kw, nm in ((dict(offset=np.array([0.0, np.inf])), "offset"), (dict(exposure=np.array([1.0, np.inf])), "exposure")):
        with pytest.raises(ValueError, match=rf"^predict\(\): '{nm}' contains infinity$"):
            r.predict(X[:2], **kw)
        with pytest.raises(ValueError, match=rf"^predict_interval\(\): '{nm}' contains infinity$"):
            r.predict_interval(X[:2], **kw)


def test_glm_omit_keeps_errors_in_nan_rows():
    y = YP.copy()
    y[0] = np.nan
    for kw, msg in ((dict(var_weights=np.r_[-1.0, np.ones(N - 1)]), "'var_weights' must be >= 0"),
                    (dict(offset=np.r_[np.inf, np.zeros(N - 1)]), "'offset' contains infinity")):
        with pytest.raises(ValueError, match=msg):
            R.glm(y, X, family="poisson", nan_policy="omit", **kw)


# ---- penalized: messages, grids, the boundary's null deviance

def test_penalized_messages():
    with pytest.raises(OverflowError, match=r"^elastic_net\(\): 'l1_ratio' is out of range$"):
        R.elastic_net(Y, X, alpha=0.1, l1_ratio=10**400)
    with pytest.raises(OverflowError, match=r"^lasso\(\): 'tol' is out of range$"):
        R.lasso(Y, X, alpha=0.1, tol=10**400)
    with pytest.raises(OverflowError, match=r"^lasso_cv\(\): 'eps' is out of range$"):
        R.lasso_cv(Y, X, rng=gen(), eps=10**400)
    with pytest.raises(OverflowError, match=r"^conf_int\(\): 'alpha' is out of range$"):
        R.ols(Y, X).conf_int(alpha=10**400)
    with pytest.raises(TypeError, match=r"^lasso_cv\(\): 'alphas' must be an int \(the grid's length\) or a "
                                        r"1-dimensional array of numbers, not float$"):
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
    assert repr(R.ridge(np.vstack([Y, Y]), X, alpha=[0.1, 0.2])) == \
        "RidgeResult(batch shape (2,) and a path of 2 alphas, 3 parameters)"


# ---- linear: predictions, scales, messages

def test_predict_nan_under_zero_and_aliased_coefficients():
    Xd = np.column_stack([X, X[:, 0]])  # the third column is aliased
    r = R.ols(Y, Xd)
    Xn = np.array([[0.5, 0.2, np.nan], [0.5, np.nan, 0.1]])
    p = a(r.predict(Xn))
    assert math.isfinite(p[0]) and math.isnan(p[1])  # aliased: no part; in the model: NaN
    pi = a(r.predict_interval(Xn))
    assert np.all(np.isfinite(pi[0])) and np.all(np.isnan(pi[1]))
    # a coefficient exactly 0 is still in the model (the lasso's zeros too)
    las = R.lasso(Y, np.column_stack([X, RNG.normal(size=N)]), alpha=0.5)
    assert a(las.params)[3] == 0.0
    assert math.isnan(float(las.predict(np.array([[0.1, 0.2, np.nan]]))[0]))


@pytest.mark.parametrize("sy,sx", [(1e-160, 1.0), (1e-170, 1.0), (1e-300, 1.0), (1e160, 1.0), (1e300, 1.0),
                                   (1.0, 1e-170), (1.0, 1e160), (1.0, 1e-300), (1e-200, 1e-200), (1e140, 1e-140)])
def test_extreme_scales_are_equivariant(sy, sx):
    x = np.arange(8.0)
    y = np.array([1.0, 2.9, 5.2, 6.8, 9.1, 11.2, 12.8, 15.1])
    Xs = np.column_stack([x, np.sin(x)])
    for ct in ("nonrobust", "HC1"):
        b = R.ols(y, Xs, cov_type=ct)
        r = R.ols(y * sy, Xs * sx, cov_type=ct)
        np.testing.assert_allclose(a(r.tvalues), a(b.tvalues), rtol=1e-12)
        np.testing.assert_allclose(a(r.pvalues), a(b.pvalues), rtol=1e-11)
        np.testing.assert_allclose(a(r.params) / np.r_[sy, sy / sx, sy / sx], a(b.params), rtol=1e-13)
        np.testing.assert_allclose(a(r.bse) / np.r_[sy, sy / sx, sy / sx], a(b.bse), rtol=1e-12)
        assert r.rsquared == pytest.approx(b.rsquared, rel=1e-13) and r.fvalue == pytest.approx(b.fvalue, rel=1e-11)
        assert r.llf == pytest.approx(b.llf - len(y) * math.log(sy), rel=1e-12, abs=1e-9)
        ci = a(r.predict_interval(Xs[:3] * sx, kind="observation")) / sy
        np.testing.assert_allclose(ci, a(b.predict_interval(Xs[:3], kind="observation")), rtol=1e-12)
        np.testing.assert_allclose(a(r.resid_studentized_internal), a(b.resid_studentized_internal), rtol=1e-11)
    w = np.linspace(1, 2, 8)
    np.testing.assert_allclose(a(R.wls(y * sy, Xs * sx, weights=w).tvalues), a(R.wls(y, Xs, weights=w).tvalues),
                               rtol=1e-12)
    S = np.eye(8) + 0.3 * np.eye(8, k=1) + 0.3 * np.eye(8, k=-1)
    np.testing.assert_allclose(a(R.gls(y * sy, Xs * sx, sigma=S).tvalues), a(R.gls(y, Xs, sigma=S).tvalues),
                               rtol=1e-11)
    lr, lb = R.linregress(x * sx, y * sy), R.linregress(x, y)
    assert lr.pvalue == pytest.approx(lb.pvalue, rel=1e-11) and lr.rvalue == pytest.approx(lb.rvalue, rel=1e-13)


def test_linear_input_messages():
    y = Y.copy()
    y[0] = np.nan
    Xi = X.copy()
    Xi[0, 1] = np.inf
    with pytest.raises(ValueError, match=r"^ols\(\): 'X' contains infinity$"):
        R.ols(y, Xi, nan_policy="omit")
    with pytest.raises(ValueError, match=r"^wls\(\): 'weights' must be >= 0$"):
        R.wls(y, X, weights=np.r_[-1.0, np.ones(N - 1)], nan_policy="omit")
    yi = Y.copy()
    yi[1] = np.inf
    with pytest.raises(ValueError, match=r"^gls\(\): 'y' contains infinity$"):
        R.gls(yi, X, sigma=np.eye(N))
    for fn, arg in ((lambda v: R.wls(Y, X, weights=v), "weights"), (lambda v: R.gls(Y, X, sigma=v), "sigma")):
        msgs = set()
        for v in (None, "x"):
            with pytest.raises(TypeError) as e:
                fn(v)
            msgs.add(str(e.value).rsplit(", not ", 1)[0])
        assert len(msgs) == 1 and f"'{arg}' must be an sd.ndarray" in msgs.pop()
    # a long list with a bad element at the end: the type named (linear, not quadratic, work)
    with pytest.raises(TypeError, match=r"not list of str$"):
        R.ols([1.0] * 200000 + ["a"], np.arange(200001.0))


def test_predict_batch_message():
    r = R.ols(RNG.normal(size=(3, 10)), RNG.normal(size=(10, 2)))
    with pytest.raises(ValueError, match=r"^predict\(\): the batch shapes of 'X_new' \(2,\) and the fit \(3,\) do not "
                                         r"broadcast$"):
        r.predict(RNG.normal(size=(2, 4, 2)))
    with pytest.raises(ValueError, match=r"^predict_interval\(\): the batch shapes of 'X_new' \(2,\)"):
        r.predict_interval(RNG.normal(size=(2, 4, 2)))


def test_cluster_batch_labels_per_regression():
    """Labels distinct across a batch (the batch's label count far above each regression's):
    each regression's covariance is that of its own fit."""
    nb, n = 6, 40
    Xb = RNG.normal(size=(nb, n, 2))
    yb = RNG.normal(size=(nb, n))
    g = np.arange(nb * n).reshape(nb, n) // 4  # 10 groups per regression, 60 labels in all
    rb = R.ols(yb, Xb, cov_type="cluster", groups=g)
    for i in range(nb):
        ri = R.ols(yb[i], Xb[i], cov_type="cluster", groups=g[i])
        assert a(rb.bse)[i].tolist() == a(ri.bse).tolist()


# ---- robust and quantile

def test_rlm_mm_degenerate_designs():
    r = R.rlm(np.zeros(0), np.zeros((0, 2)), method="MM", rng=gen())
    assert r.nobs == 0 and not r.converged
    y = np.arange(6.0) ** 1.3
    for kw in (dict(), dict(method="MM", rng=gen())):
        r0 = R.rlm(y, np.zeros((6, 2)), intercept=False, **kw)  # rank 0
        assert r0.rank == 0 and r0.n_iter == 0 and r0.converged and np.all(np.isnan(a(r0.params)))
        assert np.all(a(r0.fittedvalues) == 0.0) and math.isfinite(r0.scale) and r0.scale > 0
    q0 = R.quantreg(y, np.zeros((6, 2)), intercept=False)
    assert q0.rank == 0 and q0.n_iter == 0 and q0.converged


def test_quantreg_ker_perfect_fit_is_nan():
    x = np.arange(12.0)
    for ct in ("ker", "nid", "iid"):
        r = R.quantreg(1.0 + 2.0 * x, x, cov_type=ct)
        assert np.all(np.isnan(a(r.bse))), ct


def test_quantreg_vertex_snaps_zero_coordinates():
    Xs = np.arange(12.0).reshape(6, 2) ** 1.2
    for q in (0.2, 0.5, 0.9):
        r = R.quantreg(np.full(6, 3.0), Xs, q=q)
        assert a(r.params).tolist() == [3.0, 0.0, 0.0]
    # a vertex with a nonzero tiny coordinate keeps it (the snap only takes what does not
    # increase the basis residuals)
    x = np.arange(1.0, 9.0)
    y = 1.0 + 1e-17 * x
    r = R.quantreg(y, x)
    assert np.max(np.abs(a(r.resid))) <= 1e-15


def test_slopes_drop_overflowing_pairs():
    y = [1e308, -1e308, 0.0, 1.0, 3.0]
    x = [1e308, -1e308, 0.0, 1.0, 2.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want_t, want_s = st.theilslopes(y, x), st.siegelslopes(y, x)
    got = R.theilslopes(y, x)
    assert tuple(got) == tuple(float(v) for v in want_t)
    assert tuple(R.siegelslopes(y, x)) == tuple(float(v) for v in want_s)
    # the pass algorithm (forced) agrees
    z = float(sd.special.ndtri(0.025))
    v = K.slopes("t", sd.asarray(y), sd.asarray(x), 0, False, z, 0, 3).tolist()
    assert v == [float(u) for u in want_t]


def test_robust_summaries_say_covariance_type():
    for r in (R.rlm(Y, X), R.quantreg(Y, X)):
        assert "Covariance Type:" in r.summary() and "Cov Type:" not in r.summary()


# ---- nls inputs

def test_nls_input_messages():
    with pytest.raises(ValueError, match=r"^nls\(\): 'bounds' must be a pair \(lower, upper\), not 3 values$"):
        R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], bounds=(0, 1, 2))
    with pytest.raises(TypeError, match=r"^nls\(\): 'bounds' must be None or a pair"):
        R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], bounds=3)
    with pytest.raises(ValueError, match=r"^nls\(\): 'jac' must return shape \(40, 3\), not a scalar$"):
        R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], jac=lambda x, *t: 1.0)
    with pytest.raises(OverflowError, match=r"^nls\(\): 'ftol' is out of range$"):
        R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], ftol=10**400)


# ---- G109: nls's perfect fit

def test_nls_perfect_fit_no_significance():
    """ssr exactly 0 with df_resid > 0: bse 0, t and p NaN (no theta / 0), conf_int [theta, theta]."""
    x = np.arange(10.0)
    r = R.nls(lambda t, b0, b1: b0 * sd.exp(b1 * t), x, np.full(10, 2.0), p0=[2.0, 0.0])
    assert r.ssr == 0.0 and r.df_resid == 8
    assert np.all(a(r.bse) == 0.0)
    assert np.all(np.isnan(a(r.tvalues))) and np.all(np.isnan(a(r.pvalues)))
    ci = a(r.conf_int())
    assert np.array_equal(ci[:, 0], a(r.params)) and np.array_equal(ci[:, 1], a(r.params))
    assert "nan" in r.summary()
    # not perfect: t as before
    rr = R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0])
    assert np.all(np.isfinite(a(rr.tvalues)))


# ---- R88-R100 (2026-11-19) and G110-G113

def test_configuration_fields_are_fresh_copies():
    """G113: a list or array field is a new copy at each access."""
    r = R.ols(Y, X)
    names = r.param_names
    names.append("zz")
    assert r.param_names == ["const", "x1", "x2"]
    q = R.quantreg(Y, X, q=[0.3, 0.6])
    qq = q.q
    qq[0] = 0.9
    assert a(q.q).tolist() == [0.3, 0.6]
    lp = R.lasso(Y, X, alpha=[0.2, 0.1])
    al = lp.alpha
    al[0] = 5.0
    assert a(lp.alpha).tolist() == [0.2, 0.1]
    lp.param_names.append("zz")
    assert len(lp.param_names) == 3


def test_method_call_shape_messages():
    """R89: the methods' call-shape errors say the method's name (sd.stats' form)."""
    cases = [R.ols(Y, X), R.glm(YB, X, family="binomial"), R.rlm(Y, X), R.quantreg(Y, X),
             R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0])]
    for r in cases:
        for call, msg in ((lambda: r.conf_int(0.1), r"^conf_int\(\): takes 0 positional arguments but 1 was given$"),
                          (lambda: r.summary(0.1), r"^summary\(\): takes 0 positional arguments but 1 was given$"),
                          (lambda: r.predict(), r"^predict\(\): missing 1 required positional argument: '[xX]_new'$"),
                          (lambda: r.predict_interval(X[:2], 0.1),
                           r"^predict_interval\(\): takes 1 positional argument but 2 were given$"),
                          (lambda: r.cov_params(1, 2), r"^cov_params\(\): takes 0 positional arguments but 2 were given$")):
            with pytest.raises(TypeError, match=msg):
                call()
        with pytest.raises(TypeError, match=r"^predict\(\): unexpected keyword argument 'foo'$"):
            r.predict(X[:2], foo=1)
    with pytest.raises(TypeError, match=r"^breusch_pagan\(\): takes 0 positional arguments but 1 was given$"):
        R.ols(Y, X).breusch_pagan(True)
    with pytest.raises(TypeError, match=r"^pseudo_rsquared\(\): takes 0 positional arguments but 1 was given$"):
        R.glm(YB, X, family="binomial").pseudo_rsquared("cs")
    for r in (R.lasso(Y, X, alpha=0.1), R.penalized_glm(YB, X, family="binomial", alpha=0.1)):
        with pytest.raises(TypeError, match=r"^predict\(\): takes 1 positional argument but 2 were given$"):
            r.predict(X[:2], X[:2])
    # the methods' own errors pass unchanged, and the signatures stay
    with pytest.raises(TypeError, match=r"^conf_int\(\): 'alpha' must be an int or a float, not str$"):
        R.ols(Y, X).conf_int(alpha="x")
    import inspect
    assert str(inspect.signature(R.ols(Y, X).conf_int)) == "(*, alpha=0.05)"


def test_extreme_scale_t_in_fitted_units():
    """R90: t from beta' / sqrt(C') in the fitted units, also when a coefficient in the data's
    units overflows or underflows."""
    h = np.array([1, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 8])
    s = np.array([52, 55, 61, 58, 66, 63, 70, 74, 73, 82.0])
    b = R.ols(s, h)
    r = R.ols(s * 1e300, h * 1e-300)  # the slope's 1e600 overflows
    np.testing.assert_allclose(a(r.tvalues), a(b.tvalues), rtol=1e-13)
    np.testing.assert_allclose(a(r.pvalues), a(b.pvalues), rtol=1e-12)
    lr, lb = R.linregress(h * 1e-300, s * 1e300), R.linregress(h, s)
    assert lr.pvalue == pytest.approx(lb.pvalue, rel=1e-12) and lr.rvalue == pytest.approx(lb.rvalue, rel=1e-14)
    x = np.arange(8.0)
    y = np.array([1.0, 2.9, 5.2, 6.8, 9.1, 11.2, 12.8, 15.1]) + 3 * np.sin(x)
    b = R.ols(y, x)
    r = R.ols(y * 1e-165, x * 1e160)  # the slope's 1e-325 underflows
    np.testing.assert_allclose(a(r.tvalues), a(b.tvalues), rtol=1e-13)
    lr, lb = R.linregress(x * 1e160, y * 1e-165), R.linregress(x, y)
    assert lr.rvalue == pytest.approx(lb.rvalue, rel=1e-14) and lr.pvalue == pytest.approx(lb.pvalue, rel=1e-12)


def test_converged_at_the_cap():
    """R94/R95: the cap stops a fit only before the test and the polish complete: a fit that
    finishes exactly at max_iter is converged with n_iter = max_iter."""
    ref = R.glm(YB, X, family="binomial")
    m = ref.n_iter
    at = R.glm(YB, X, family="binomial", max_iter=m)
    assert at.n_iter == m and at.converged and a(at.params).tolist() == a(ref.params).tolist()
    assert not R.glm(YB, X, family="binomial", max_iter=m - 1).converged


def test_alphas_messages_name_the_accepted_forms():
    form = r"'alphas' must be an int \(the grid's length\) or a 1-dimensional array"
    for v in (True, "x", 1.5, None, ["a"], np.array([True])):
        with pytest.raises(TypeError, match=r"^lasso_cv\(\): " + form + r" of numbers, not "):
            R.lasso_cv(Y, X, rng=gen(), alphas=v)
    with pytest.raises(ValueError, match=r"^lasso_cv\(\): " + form + r", not 2-dimensional$"):
        R.lasso_cv(Y, X, rng=gen(), alphas=[[0.1]])


def test_nls_predict_interval_nan_rows():
    """R100: a NaN row of x_new has a NaN interval (no differences through it), with or
    without jac; the other rows as without it."""
    tn = np.array([0.5, np.nan, 2.0])
    for jac in (None, decay_jac):
        r = R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], jac=jac)
        pi = a(r.predict_interval(tn))
        assert np.all(np.isnan(pi[1])) and np.all(np.isfinite(pi[[0, 2]]))
        np.testing.assert_allclose(pi[[0, 2]], a(r.predict_interval(tn[[0, 2]])), rtol=1e-12)
        assert np.all(np.isnan(a(r.predict_interval(np.array([np.nan])))))


def test_lasso_alpha_zero_is_ridges():
    """G110: lasso and elastic_net at alpha = 0 give ridge(alpha=0)'s minimum-norm solution bit
    for bit (p > n too, and in a path), converged, n_iter 0, dual_gap 0."""
    rng = np.random.default_rng(1)
    for n, p in ((30, 4), (10, 15)):
        Xw = rng.normal(size=(n, p))
        Xw = np.column_stack([Xw, Xw[:, 0]])  # an aliased column too
        y = rng.normal(size=n)
        rr = R.ridge(y, Xw, alpha=0.0)
        for fit in (lambda **k: R.lasso(y, Xw, **k), lambda **k: R.elastic_net(y, Xw, l1_ratio=0.4, **k)):
            r = fit(alpha=0.0)
            assert a(r.params).tolist() == a(rr.params).tolist()
            assert r.converged and r.n_iter == 0 and r.dual_gap == 0.0 and r.df_model == rr.df_model
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


@pytest.mark.parametrize("fam,link,y0,kw", [
    ("poisson", "log", 3.0, {}), ("poisson", "sqrt", 3.0, {}), ("poisson", "identity", 3.0, {}),
    ("gamma", "inverse", 2.0, {}), ("gamma", "log", 2.0, {}), ("gamma", "identity", 2.0, {}), ("gaussian", "log", 2.0, {}),
    ("binomial", "logit", 0.3, {"trials": 10.0}), ("binomial", "probit", 0.3, {"trials": 10.0}),
    ("binomial", "cloglog", 0.3, {"trials": 10.0}), ("binomial", "log", 0.3, {"trials": 10.0})])
def test_glm_constant_y_every_family(fam, link, y0, kw):
    """G111: the exact solution (g(y0), 0, ...) with mu = y0; no spurious significance: NaN for
    an estimated scale (0/0), t 0 and p 1 for a fixed one."""
    kw = {k: np.full(N, v) for k, v in kw.items()}
    r = R.glm(np.full(N, y0), X, family=fam, link=link, **kw)
    sm_link = {"logit": lambda m: math.log(m / (1 - m)), "probit": lambda m: float(sd.special.ndtri(m)),
               "cloglog": lambda m: math.log(-math.log1p(-m)), "log": math.log, "identity": lambda m: m,
               "sqrt": math.sqrt, "inverse": lambda m: 1 / m}[link]
    assert a(r.params)[1:].tolist() == [0.0, 0.0] and float(r.params[0]) == pytest.approx(sm_link(y0), rel=1e-15)
    assert np.all(a(r.fittedvalues) == y0) and r.deviance == 0.0 and r.null_deviance == 0.0
    assert r.converged and not r.separation and r.llnull == r.llf
    if fam in ("gaussian", "gamma"):
        assert r.scale == 0.0 and r.llf == math.inf and np.all(a(r.bse) == 0.0)
        assert np.all(np.isnan(a(r.pvalues)[1:])) and math.isnan(r.pseudo_rsquared(kind="mcf"))
    else:
        assert np.all(a(r.tvalues)[1:] == 0.0) and np.all(a(r.pvalues)[1:] == 1.0)
        assert r.llr == 0.0 and r.pseudo_rsquared() == 0.0 and r.pseudo_rsquared(kind="mcf") == 0.0


def test_glm_constant_y_at_the_boundary_keeps_its_rules():
    for y, fam in ((np.ones(N), "binomial"), (np.zeros(N), "binomial"), (np.zeros(N), "poisson")):
        r = R.glm(y, X, family=fam)
        assert r.separation and np.all(np.isnan(a(r.params)))


def test_mcfadden_nan_for_infinite_llf():
    """G112: McFadden is NaN where llf is infinite; Cox-Snell keeps its limit."""
    x = np.arange(1.0, 9.0)
    r = R.glm(1.0 + 2.0 * x, x, family="gaussian")
    assert r.llf == math.inf
    assert math.isnan(r.pseudo_rsquared(kind="mcf")) and r.pseudo_rsquared(kind="cs") == 1.0
