import math

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd
from tests._bits import max_rel_error

from .smalldata import RNG, TD, YB, YD, YP, N, X, Y, a, decay, gen

"""Rules every fit of sd.regression follows (docs/api/regression.md §1): a constant y, df_resid = 0
and the perfect fit's t and p, empty inputs and zero weights, no parameters, a one-column y,
read-only result fields and arrays, conf_int's keyword, the prediction intervals of rlm and
quantreg, the methods' and functions' call-shape messages, the input conversion, and
equivariance under power-of-two units across the fits."""


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


R = sd.regression


def _empty_ok(r):
    assert r.nobs == 0
    assert math.isnan(r.df_model)
    if hasattr(r, "df_resid"):
        assert math.isnan(r.df_resid)
    if hasattr(r, "converged"):
        assert r.converged is False and r.n_iter == 0
    assert np.all(np.isnan(a(r.params)))
    assert np.all(np.isnan(a(r.fittedvalues)))


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


# ---- G95: df_resid = 0, the perfect fit's tolerance


def test_df_resid_zero_is_nan():
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([1.0, 3.0, 2.0])
    Xs = np.column_stack([x, x**2])
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


# ---- G96: empty inputs


@pytest.mark.parametrize("name", list(FITS))
def test_empty_input(name):
    fit = FITS[name]
    _empty_ok(fit(np.zeros(0), np.zeros((0, 2))))
    _empty_ok(fit(np.full(6, np.nan), X[:6], nan_policy="omit"))


def test_zero_weights_are_empty():
    w0 = np.zeros(N)
    for r in (
        R.wls(Y, X, weights=w0),
        R.glm(Y, X, family="gaussian", var_weights=w0),
        R.glm(YP, X, family="poisson", freq_weights=w0),
        R.lasso(Y, X, alpha=0.1, weights=w0),
        R.lasso_cv(Y, X, rng=gen(), cv=3, weights=w0),
        R.ridge(Y, X, alpha=0.1, weights=w0),
        R.penalized_glm(YP, X, family="poisson", alpha=0.1, weights=w0),
        R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0], weights=np.zeros(40)),
    ):
        _empty_ok(r)
    # in a batch only that regression is NaN
    rb = R.wls(Y, X, weights=np.stack([np.ones(N), w0]))
    assert a(rb.nobs).tolist() == [N, 0.0] and np.all(np.isfinite(a(rb.params)[0]))
    assert np.all(np.isnan(a(rb.params)[1])) and math.isnan(a(rb.df_model)[1])


def test_linregress_and_slopes_too_few_points():
    """D-20: fewer than two points give G96's NaN result (scipy 1.18.1's NaN), not an error;
    a batch too."""
    for f in (
        lambda: R.linregress([1.0], [2.0]),
        lambda: R.linregress([], []),
        lambda: R.theilslopes([1.0]),
        lambda: R.siegelslopes([1.0, np.nan], nan_policy="omit"),
    ):
        assert all(math.isnan(v) for v in f())
    b = R.theilslopes(np.ones((3, 1)))
    assert all(v.shape == (3,) and bool(sd.all(sd.isnan(v))) for v in b)
    b = R.linregress(np.ones((2, 1)), np.ones((2, 1)))
    assert all(v.shape == (2,) and bool(sd.all(sd.isnan(v))) for v in b)


# ---- G98, G99: read-only configuration, conf_int's keyword


def test_read_only_fields():
    cases = [
        (R.ols(Y, X), ("param_names", "cov_type")),
        (R.glm(YB, X, family="binomial"), ("family", "link")),
        (R.lasso(Y, X, alpha=0.1), ("alpha", "l1_ratio", "standardize", "param_names")),
        (R.penalized_glm(YB, X, family="binomial", alpha=0.1), ("family",)),
        (R.rlm(Y, X), ("method", "norm", "cov_type")),
        (R.quantreg(Y, X, q=[0.3, 0.6]), ("q", "cov_type")),
        (R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0]), ("method", "param_names")),
    ]
    for r, names in cases:
        for nm in names:
            old = getattr(r, nm)
            with pytest.raises(AttributeError):
                setattr(r, nm, old)


def test_conf_int_keyword_only():
    for r in (
        R.ols(Y, X),
        R.glm(YB, X, family="binomial"),
        R.rlm(Y, X),
        R.quantreg(Y, X),
        R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0]),
    ):
        with pytest.raises(TypeError):
            r.conf_int(0.1)
        assert a(r.conf_int(alpha=0.1)).shape[-1] == 2


# ---- G101: no parameters


def test_no_parameters_everywhere():
    Z = np.zeros((N, 0))
    for f in (
        lambda: R.ols(Y, Z, intercept=False),
        lambda: R.wls(Y, Z, weights=np.ones(N), intercept=False),
        lambda: R.gls(Y, Z, sigma=np.eye(N), intercept=False),
        lambda: R.glm(YP, Z, family="poisson", intercept=False),
        lambda: R.ridge(Y, Z, alpha=0.1, intercept=False),
        lambda: R.lasso(Y, Z, alpha=0.1, intercept=False),
        lambda: R.lasso_cv(Y, Z, rng=gen(), intercept=False),
        lambda: R.penalized_glm(YP, Z, family="poisson", alpha=0.1, intercept=False),
        lambda: R.rlm(Y, Z, intercept=False),
        lambda: R.quantreg(Y, Z, intercept=False),
    ):
        with pytest.raises(ValueError, match=r"the model has no parameters \(no columns in 'X' and intercept=False\)"):
            f()
    with pytest.raises(ValueError, match=r"^nls\(\): the model has no parameters \('p0' is empty\)$"):
        R.nls(lambda t: t, TD, YD, p0=[])


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


# ---- G105: a one-column y


def test_one_column_dataframe_y():
    dfy = sd.frame.DataFrame({"y": Y.tolist()})
    for fit in (
        lambda y: R.ols(y, X),
        lambda y: R.glm(y, X, family="gaussian"),
        lambda y: R.rlm(y, X),
        lambda y: R.quantreg(y, X),
        lambda y: R.lasso(y, X, alpha=0.1),
    ):
        assert a(fit(dfy).params).tolist() == a(fit(Y).params).tolist()
    assert R.linregress(X[:, 0], dfy) == R.linregress(X[:, 0], Y)
    with pytest.raises(
        ValueError,
        match=r"^ols\(\): 'X' must have 1 row like 'y', not 60 \('y' of shape \(60, 1\) "
        r"is a batch of regressions with 1 row each",
    ):
        R.ols(sd.asarray(Y.reshape(-1, 1)), X)
    with pytest.raises(ValueError, match=r"^ols\(\): 'X' must have 5 rows like 'y', not 4$"):
        R.ols(Y[:5], X[:4])


def test_configuration_fields_are_fresh_copies():
    """G113: a list field is a new copy at each access; an array field is read-only (D-4)."""
    r = R.ols(Y, X)
    names = r.param_names
    names.append("zz")
    assert r.param_names == ["const", "x1", "x2"]
    q = R.quantreg(Y, X, q=[0.3, 0.6])
    with pytest.raises(ValueError, match="read-only"):
        q.q[0] = 0.9
    assert a(q.q).tolist() == [0.3, 0.6]
    lp = R.lasso(Y, X, alpha=[0.2, 0.1])
    with pytest.raises(ValueError, match="read-only"):
        lp.alpha[0] = 5.0
    assert a(lp.alpha).tolist() == [0.2, 0.1]
    lp.param_names.append("zz")
    assert len(lp.param_names) == 3


def test_configuration_field_docstrings_say_the_behaviour():
    """The site quotes these docstrings: they say what the field does, with no decision number."""
    import re

    for r in (R.ols(Y, X), R.lasso(Y, X, alpha=0.1)):
        doc = " ".join(type(r).param_names.__doc__.split())
        assert "new list at each access" in doc and not re.search(r"\b[GDOR]\d+\b", doc)


def test_method_call_shape_messages():
    """R89: the methods' call-shape errors say the method's name (sd.stats' form)."""
    cases = [
        R.ols(Y, X),
        R.glm(YB, X, family="binomial"),
        R.rlm(Y, X),
        R.quantreg(Y, X),
        R.nls(decay, TD, YD, p0=[1.0, 1.0, 0.0]),
    ]
    for r in cases:
        for call, msg in (
            (lambda: r.conf_int(0.1), r"^conf_int\(\): takes 0 positional arguments but 1 was given$"),
            (lambda: r.summary(0.1), r"^summary\(\): takes 0 positional arguments but 1 was given$"),
            (lambda: r.predict(), r"^predict\(\): missing 1 required positional argument: '[xX]_new'$"),
            (
                lambda: r.predict_interval(X[:2], 0.1),
                r"^predict_interval\(\): takes 1 positional argument but 2 were given$",
            ),
            (lambda: r.cov_params(1, 2), r"^cov_params\(\): takes 0 positional arguments but 2 were given$"),
        ):
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


# ---- power-of-two units for glm, rlm, quantreg and the penalized fits (RG-5)


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
        assert max_rel_error(r.params, a(b.params) * s) < 1e-13
        assert max_rel_error(r.tvalues, b.tvalues) < 1e-12 and max_rel_error(r.pvalues, b.pvalues) < 1e-10
    yg = np.exp(0.2 + X @ [0.3, -0.2]) * rg.gamma(5.0, 0.2, n)
    for link, fac in (("inverse", 1 / s), ("identity", s)):
        b, r = R.glm(yg, X, family="gamma", link=link), R.glm(yg * s, X, family="gamma", link=link)
        assert max_rel_error(r.params, a(b.params) * fac) < 1e-13 and max_rel_error(r.pvalues, b.pvalues) < 1e-10
        assert r.scale == pytest.approx(b.scale, rel=1e-13)
    b, r = R.lasso(y, X, alpha=0.05), R.lasso(y * s, X, alpha=0.05 * s)
    assert r.converged and max_rel_error(r.params, a(b.params) * s) < 1e-13
    assert r.rsquared == pytest.approx(b.rsquared, rel=1e-13)
    b, r = R.ridge(y, X, alpha=0.05), R.ridge(y * s, X, alpha=0.05)
    assert max_rel_error(r.params, a(b.params) * s) < 1e-13 and r.rsquared == pytest.approx(b.rsquared, rel=1e-13)


def test_column_scale_equivariance_of_the_glm():
    rg = np.random.default_rng(3)
    X = rg.normal(size=(80, 2))
    y = (rg.uniform(size=80) < 1 / (1 + np.exp(-(0.2 + X @ [1.0, -0.5])))).astype(float)
    b = R.glm(y, X, family="binomial")
    for s in (1e-200, 1e200):
        r = R.glm(y, X * s, family="binomial")
        assert max_rel_error(r.pvalues, b.pvalues) < 1e-12
        assert max_rel_error(a(r.params)[1:] * s, a(b.params)[1:]) < 1e-13


def test_batch_of_mixed_scales():
    rg = np.random.default_rng(4)
    X = rg.normal(size=(50, 3))
    y = 1.0 + X @ [0.5, -0.3, 0.0] + 0.3 * rg.normal(size=50)
    Y = np.stack([y, y * 1e-200, y * 1e200])
    for f in (lambda v: R.elastic_net(v, X, alpha=0.05, l1_ratio=0.3), lambda v: R.rlm(v, X)):
        r = f(Y)
        for i in range(3):
            assert np.array_equal(a(r.params)[i], a(f(Y[i]).params))


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


# ---- the input conversion (CX-11, DC-12; G10, G11)


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
