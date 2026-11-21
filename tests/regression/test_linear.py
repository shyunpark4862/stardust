"""G1a: ols, wls, gls (docs/api/regression.md §2, docs/design/regression.md §2-§3).

Accuracy against Arb (tests/regression/ref/linear.json, tests/regression/refdata.py), then
statsmodels and scipy, rank deficiency, NaN policies, batches, frame inputs and errors."""

import math

import numpy as np
import pytest

import stardust as sd

from . import refdata
from .smalldata import RNG, N, X, Y, unix_time_data

R = sd.regression
REFS = refdata.load()
RANDOM = ["ols_int", "ols_noint", "wls_zero", "wls_noint", "ols_cond"]
NIST = [k for k in REFS if k.startswith("nist_")]

# docs/design/regression.md §3.1, G15: the coefficient vector to 1e-14 (normwise) for kappa <= 1e11
CONTRACT = 1e-14


def fit(name, **kw):
    p = REFS[name]
    if p["weights"] is not None:
        return R.wls(p["y"], p["X"], weights=p["weights"], intercept=p["intercept"], **kw)
    return R.ols(p["y"], p["X"], intercept=p["intercept"], **kw)


def a(x):
    return np.asarray(x, dtype=float)


def norm_err(got, ref):
    hi = np.array([v[0] for v in ref])
    lo = np.array([v[1] for v in ref])
    return np.linalg.norm((a(got) - hi) - lo) / np.linalg.norm(hi + lo)


def max_rel(got, ref):
    return max(refdata.rel_err(g, r) for g, r in zip(a(got).ravel(), ref))


# ---- Arb references


@pytest.mark.parametrize("name", RANDOM)
def test_random_against_arb(name):
    p, ref, meta = REFS[name], REFS[name]["ref"], REFS[name]["meta"]
    r = fit(name)
    assert norm_err(r.params, ref["params"]) <= CONTRACT
    assert max_rel(r.params, ref["params"]) <= CONTRACT
    assert r.nobs == meta["nobs"] and r.df_resid == meta["df_resid"] and r.df_model == meta["df_model"]
    # standard errors come from R: about kappa u (docs/design/regression.md §3.1)
    tol = max(1e-13, 10 * r.condition_number * 2.0**-52)
    assert max_rel(r.bse, ref["bse"]) <= tol
    assert max_rel(r.tvalues, ref["tvalues"]) <= tol
    assert max_rel(r.pvalues, ref["pvalues"]) <= 100 * tol
    ci = a(r.conf_int(alpha=0.05))
    assert max_rel(ci[:, 0], ref["conf_int_lower"]) <= tol
    assert max_rel(ci[:, 1], ref["conf_int_upper"]) <= tol
    scalars = ["ssr", "ess", "rsquared", "rsquared_adj", "scale", "llf", "aic", "bic", "fvalue", "f_pvalue"]
    for s in scalars:
        assert refdata.rel_err(getattr(r, s), ref[s]) <= 1e-13, s
    tss = r.centered_tss if p["intercept"] else r.uncentered_tss
    assert refdata.rel_err(tss, ref["centered_tss"]) <= 1e-14  # regref's tss is the one R^2 uses
    yscale = max(abs(v) for v in p["y"])
    fv = a(r.fittedvalues)
    assert max(abs(fv[i] - refdata.value(ref["fittedvalues"][i])) for i in range(len(fv))) <= 1e-14 * yscale
    ev = a(r.resid)
    assert max(abs(ev[i] - refdata.value(ref["resid"][i])) for i in range(len(ev))) <= 1e-14 * yscale
    for ct in ("HC0", "HC1", "HC2", "HC3"):
        rh = fit(name, cov_type=ct)
        assert max_rel(rh.bse, ref["bse_" + ct]) <= tol, ct
        assert np.array_equal(a(rh.params), a(r.params))


@pytest.mark.parametrize("name", NIST)
def test_nist_against_arb(name):
    # the coefficients against Arb on the double inputs: 1e-14 on all, Filip (kappa 1.8e15,
    # beyond the contract) included since the refinement continues while it matters
    ref, meta = REFS[name]["ref"], REFS[name]["meta"]
    r = fit(name)
    assert r.rank == len(ref["params"])
    bound = CONTRACT
    assert norm_err(r.params, ref["params"]) <= bound
    if meta["exact_fit"]:
        # Wampler1: y is in the span of X even in doubles; the fit is exact (t and p NaN, D-3)
        assert r.ssr == 0.0 and np.all(a(r.bse) == 0.0)
        assert np.all(np.isnan(a(r.tvalues))) and np.all(np.isnan(a(r.pvalues)))
        return
    # standard errors come from R: about kappa u (Filip 7e-8; Wampler2's residuals are at the
    # rounding level of the fitted values, so its tiny standard errors are not relatively accurate)
    if name == "nist_Wampler2":
        assert np.all(a(r.bse) <= 1e-12)
    else:
        tol = {"nist_Filip": 1e-6, "nist_Longley": 1e-12}.get(name, 1e-13)
        assert max_rel(r.bse, ref["bse"]) <= tol


def test_nist_certified():
    # the certified values themselves (exact decimal data): the rounding of the data to doubles
    # moves the solution by about kappa u, so only the well-conditioned problems are compared
    from . import nist

    for name in ("Norris", "NoInt1", "NoInt2"):
        d = nist.load(name)
        y, X = nist.design_strings(d)
        r = R.ols(y, X, intercept=d["intercept"])
        for got, want in zip(a(r.params), d["params"]):
            assert abs(got - float(want)) <= 1e-12 * abs(float(want))
        assert abs(r.rsquared - float(d["rsquared"])) <= 1e-12


# ---- statsmodels and scipy


def _problem(n=40, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    y = X @ np.arange(1.0, k + 1) + 0.5 + rng.standard_normal(n)
    return y, X


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
@pytest.mark.parametrize("intercept", [True, False])
def test_ols_against_statsmodels(cov_type, intercept):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _problem()
    r = R.ols(y, X, intercept=intercept, cov_type=cov_type)
    # robust covariances use the t distribution (user decision 2026-11-18), statsmodels' use_t=True
    s = sm.OLS(y, sm.add_constant(X) if intercept else X).fit(cov_type=cov_type, use_t=True)
    for name in ("params", "bse", "tvalues", "pvalues", "fittedvalues", "resid"):
        np.testing.assert_allclose(a(getattr(r, name)), getattr(s, name), rtol=1e-11, err_msg=name)
    for name in (
        "nobs",
        "df_model",
        "df_resid",
        "ssr",
        "ess",
        "centered_tss",
        "uncentered_tss",
        "rsquared",
        "rsquared_adj",
        "fvalue",
        "f_pvalue",
        "llf",
        "aic",
        "bic",
        "scale",
        "condition_number",
    ):
        assert getattr(r, name) == pytest.approx(getattr(s, name), rel=1e-11), name
    np.testing.assert_allclose(a(r.conf_int(alpha=0.1)), s.conf_int(0.1), rtol=1e-11)
    np.testing.assert_allclose(a(r.cov_params()), s.cov_params(), rtol=1e-11)
    Xn = np.random.default_rng(5).standard_normal((6, 3))
    pr = s.get_prediction(sm.add_constant(Xn, has_constant="add") if intercept else Xn).summary_frame(alpha=0.1)
    np.testing.assert_allclose(a(r.predict(Xn)), pr["mean"], rtol=1e-12)
    pm = a(r.predict_interval(Xn, alpha=0.1))
    np.testing.assert_allclose(pm[:, 0], pr["mean_ci_lower"], rtol=1e-11)
    np.testing.assert_allclose(pm[:, 1], pr["mean_ci_upper"], rtol=1e-11)
    if cov_type == "nonrobust":
        po = a(r.predict_interval(Xn, alpha=0.1, kind="observation"))
        np.testing.assert_allclose(po[:, 0], pr["obs_ci_lower"], rtol=1e-11)
        np.testing.assert_allclose(po[:, 1], pr["obs_ci_upper"], rtol=1e-11)


def test_wls_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    y, X = _problem(seed=2)
    w = np.random.default_rng(3).uniform(0.1, 4.0, len(y))
    r = R.wls(y, X, weights=w)
    s = sm.WLS(y, sm.add_constant(X), weights=w).fit()
    for name in ("params", "bse", "pvalues", "fittedvalues", "resid"):
        np.testing.assert_allclose(a(getattr(r, name)), getattr(s, name), rtol=1e-11, err_msg=name)
    for name in ("ssr", "ess", "centered_tss", "rsquared", "fvalue", "llf", "aic", "bic"):
        assert getattr(r, name) == pytest.approx(getattr(s, name), rel=1e-11), name


def test_wls_zero_weights_rule():
    # R's rule (G20): zero-weight rows are out of the fit, nobs and the degrees of freedom; they
    # still get fitted values and residuals. statsmodels counts them in nobs (its df and bse differ).
    sm = pytest.importorskip("statsmodels.api")
    y, X = _problem(seed=4)
    w = np.random.default_rng(6).uniform(0.5, 2.0, len(y))
    w[[3, 11]] = 0.0
    keep = w > 0
    r = R.wls(y, X, weights=w)
    sub = sm.WLS(y[keep], sm.add_constant(X[keep]), weights=w[keep]).fit()
    full = sm.WLS(y, sm.add_constant(X), weights=w).fit()
    assert r.nobs == 38 and full.nobs == 40 and r.df_resid == 34
    for name in ("params", "bse", "pvalues"):
        np.testing.assert_allclose(a(getattr(r, name)), getattr(sub, name), rtol=1e-11)
    for name in ("ssr", "rsquared", "llf", "fvalue"):
        assert getattr(r, name) == pytest.approx(getattr(sub, name), rel=1e-11)
    np.testing.assert_allclose(a(r.params), full.params, rtol=1e-11)  # the same estimate
    assert not np.allclose(a(r.bse), full.bse, rtol=1e-3)  # other degrees of freedom
    fv = a(r.fittedvalues)
    np.testing.assert_allclose(fv, sm.add_constant(X) @ a(r.params), rtol=1e-12)
    np.testing.assert_allclose(a(r.resid), y - fv, atol=1e-13)
    # the zero rows equal a fit on the others, bit for bit
    o = R.ols(y[keep], X[keep]) if np.all(w[keep] == 1) else R.wls(y[keep], X[keep], weights=w[keep])
    assert np.array_equal(a(o.params), a(r.params))


def test_wls_unit_weights_equal_ols():
    y, X = _problem(seed=7)
    r = R.wls(y, X, weights=np.ones(len(y)))
    o = R.ols(y, X)
    assert np.array_equal(a(r.params), a(o.params)) and np.array_equal(a(r.bse), a(o.bse))


def test_gls():
    sm = pytest.importorskip("statsmodels.api")
    # n = 30: scipy 1.18.1's cholesky(lower=True), which statsmodels' GLS calls, is wrong from
    # n = 33 on in this environment (its upper triangle is not zeroed and L L^T != sigma)
    y, X = _problem(n=30, seed=8)
    idx = np.arange(30)
    S = 0.5 * np.eye(30) + 0.5 * np.exp(-np.abs(np.subtract.outer(idx, idx)) / 3)
    r = R.gls(y, X, sigma=S)
    s = sm.GLS(y, sm.add_constant(X), sigma=S).fit()
    for name in ("params", "bse", "pvalues", "fittedvalues", "resid"):
        np.testing.assert_allclose(a(getattr(r, name)), getattr(s, name), rtol=1e-10, err_msg=name)
    for name in ("ssr", "centered_tss", "uncentered_tss", "rsquared", "fvalue", "llf", "aic", "bic"):
        assert getattr(r, name) == pytest.approx(getattr(s, name), rel=1e-10), name
    # sigma = I is OLS; a diagonal sigma is WLS with weights 1/sigma_ii
    o = R.ols(y, X)
    g = R.gls(y, X, sigma=np.eye(30))
    np.testing.assert_allclose(a(g.params), a(o.params), rtol=1e-14)
    assert g.llf == pytest.approx(o.llf, rel=1e-14)
    d = np.random.default_rng(9).uniform(0.5, 2.0, 30)
    g = R.gls(y, X, sigma=np.diag(d))
    w = R.wls(y, X, weights=1 / d)
    np.testing.assert_allclose(a(g.params), a(w.params), rtol=1e-13)
    np.testing.assert_allclose(a(g.bse), a(w.bse), rtol=1e-13)
    assert g.llf == pytest.approx(w.llf, rel=1e-13) and g.rsquared == pytest.approx(w.rsquared, rel=1e-13)


def test_gls_errors():
    y, X = _problem(n=10)
    with pytest.raises(sd.linalg.LinAlgError, match=r"^gls\(\): 'sigma' is not positive definite$"):
        R.gls(y, X, sigma=-np.eye(10))
    with pytest.raises(ValueError, match=r"^gls\(\): 'sigma' must be 10x10 like the rows of 'y', not 9x9$"):
        R.gls(y, X, sigma=np.eye(9))
    S = np.eye(10)
    S[2, 2] = np.inf
    with pytest.raises(ValueError, match=r"^gls\(\): 'sigma' contains infinity$"):
        R.gls(y, X, sigma=S)
    S[2, 2] = np.nan
    assert np.all(np.isnan(a(R.gls(y, X, sigma=S).params)))
    with pytest.raises(ValueError, match="nan_policy='raise'"):
        R.gls(y, X, sigma=S, nan_policy="raise")
    o = R.gls(y, X, sigma=S, nan_policy="omit")
    keep = np.arange(10) != 2
    ref = R.gls(y[keep], X[keep], sigma=np.eye(9))
    np.testing.assert_array_equal(a(o.params), a(ref.params))


# ---- rank deficiency (G9, G21)


def test_aliased_column_like_r():
    y, X = _problem(seed=12)
    X2 = np.column_stack([X[:, 0], X[:, 1], X[:, 0] - 2 * X[:, 1], X[:, 2]])  # column 3 = c1 - 2 c2
    r = R.ols(y, X2)
    ref = R.ols(y, X)
    p = a(r.params)
    assert r.rank == 4 and r.df_model == 3 and r.df_resid == 36
    assert math.isnan(p[3]) and np.all(np.isnan(a(r.bse)[[3]])) and np.all(np.isnan(a(r.pvalues)[[3]]))
    assert np.array_equal(p[[0, 1, 2, 4]], a(ref.params))  # the reduced problem's fit, bit for bit
    assert np.array_equal(a(r.bse)[[0, 1, 2, 4]], a(ref.bse))
    assert np.array_equal(a(r.fittedvalues), a(ref.fittedvalues))
    assert r.ssr == ref.ssr and r.aic == ref.aic and r.rsquared == ref.rsquared
    cov = a(r.cov_params())
    assert np.all(np.isnan(cov[3])) and np.all(np.isnan(cov[:, 3]))
    Xn = np.random.default_rng(1).standard_normal((3, 4))
    np.testing.assert_array_equal(a(r.predict(Xn)), a(ref.predict(Xn[:, [0, 1, 3]])))
    assert np.array_equal(a(r.predict(X2)), a(r.fittedvalues))


def test_constant_column_with_intercept():
    # the front column stays: the intercept is kept, the user's constant column is aliased
    y, X = _problem(seed=13)
    r = R.ols(y, np.column_stack([np.ones(len(y)), X]))
    assert r.rank == 4 and math.isnan(float(a(r.params)[1]))
    np.testing.assert_array_equal(a(r.params)[[0, 2, 3, 4]], a(R.ols(y, X).params))
    assert r.param_names == ["const", "x1", "x2", "x3", "x4"]


def test_rtol_r_like():
    # a nearly dependent column: aliased with R's tolerance 1e-7, kept with the default
    rng = np.random.default_rng(14)
    y, X = _problem(seed=14)
    near = X[:, 0] + 1e-9 * rng.standard_normal(len(y))
    X2 = np.column_stack([X, near])
    assert R.ols(y, X2).rank == 5
    assert R.ols(y, X2, rtol=1e-7).rank == 4
    assert R.ols(y, X2, rtol=0).rank == 5


def test_many_aliased_columns():
    # past 8 aliased columns the dqrdc2 column-by-column path takes over; same choice, same fit
    y, X = _problem(seed=15)
    cols = [X[:, 0]] + [X[:, 0] * (j + 2) for j in range(12)] + [X[:, 1], X[:, 1] + X[:, 0], X[:, 2]]
    r = R.ols(y, np.column_stack(cols))
    keep = [0, 1, 14, 16]  # const, x1, x14, x16
    assert r.rank == 4
    assert np.array_equal(np.flatnonzero(~np.isnan(a(r.params))), keep)
    np.testing.assert_array_equal(a(r.params)[keep], a(R.ols(y, X).params))


def test_more_columns_than_rows():
    rng = np.random.default_rng(16)
    X = rng.standard_normal((4, 6))
    y = rng.standard_normal(4)
    r = R.ols(y, X)
    assert r.rank == 4 and r.df_resid == 0 and not math.isfinite(r.scale)
    assert np.array_equal(np.isnan(a(r.params)), [False, False, False, False, True, True, True])
    np.testing.assert_allclose(a(r.fittedvalues), y, atol=1e-14)


def test_exact_fit():
    x = np.arange(10.0)
    X = np.column_stack([x, x**2])
    r = R.ols(1 + 2 * x + 3 * x**2, X)
    assert np.array_equal(a(r.params), [1.0, 2.0, 3.0])
    assert r.ssr == 0.0 and r.scale == 0.0 and np.all(a(r.bse) == 0.0)
    # a perfect fit: t and p are NaN (D-3), the intervals [theta, theta]
    assert np.all(np.isnan(a(r.tvalues))) and np.all(np.isnan(a(r.pvalues)))
    assert np.array_equal(a(r.conf_int())[:, 0], a(r.params)) and np.array_equal(a(r.conf_int())[:, 1], a(r.params))
    assert r.rsquared == 1.0 and math.isinf(r.fvalue) and r.f_pvalue == 0.0 and r.llf == math.inf


def test_intercept_only():
    # as statsmodels: R^2 and ESS 0, no F test
    r = R.ols([1.0, 2.0, 4.0], np.zeros((3, 0)))
    assert r.params.shape == (1,) and float(r.params[0]) == pytest.approx(7 / 3, rel=1e-15)
    assert r.rsquared == 0.0 and r.ess == 0.0 and r.rsquared_adj == 0.0
    assert math.isnan(r.fvalue) and math.isnan(r.f_pvalue)


# ---- NaN policies (G16)


def test_nan_policies():
    y, X = _problem(seed=17)
    y2 = y.copy()
    y2[5] = np.nan
    r = R.ols(y2, X)
    for name in ("params", "bse", "pvalues", "fittedvalues", "resid"):
        assert np.all(np.isnan(a(getattr(r, name)))), name
    for name in ("ssr", "rsquared", "llf", "fvalue", "nobs", "df_resid", "df_model", "condition_number"):
        assert math.isnan(getattr(r, name)), name
    assert r.rank == 0
    assert np.all(np.isnan(a(r.predict(X))))
    with pytest.raises(ValueError, match=r"^ols\(\): the input contains NaN \(nan_policy='raise'\)$"):
        R.ols(y2, X, nan_policy="raise")
    X2 = X.copy()
    X2[9, 1] = np.nan
    o = R.ols(y2, X2, nan_policy="omit")
    keep = np.ones(len(y), bool)
    keep[[5, 9]] = False
    ref = R.ols(y[keep], X[keep])
    assert o.nobs == 38 and len(a(o.resid)) == 38
    assert np.array_equal(a(o.params), a(ref.params))
    w = np.ones(len(y))
    w[3] = np.nan
    assert np.all(np.isnan(a(R.wls(y, X, weights=w).params)))
    assert R.wls(y, X, weights=w, nan_policy="omit").nobs == 39
    with pytest.raises(ValueError, match=r"^ols\(\): nan_policy='omit' takes one regression, not a batch"):
        R.ols(np.stack([y, y]), X, nan_policy="omit")


# ---- batches (G14)


def _batch_equal(res, i, one):
    for name in ("params", "bse", "pvalues", "fittedvalues", "resid", "cov_params"):
        got = getattr(res, name)
        got = got() if callable(got) else got
        want = getattr(one, name)
        want = want() if callable(want) else want
        np.testing.assert_array_equal(a(got)[i], a(want), err_msg=name)
    for name in ("ssr", "rsquared", "fvalue", "f_pvalue", "llf", "aic", "nobs", "df_resid", "condition_number"):
        assert float(getattr(res, name)[i]) == getattr(one, name) or (
            math.isnan(getattr(one, name)) and math.isnan(float(getattr(res, name)[i]))
        ), name
    assert int(res.rank[i]) == one.rank


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC3"])
def test_batch_equals_loop(cov_type):
    rng = np.random.default_rng(18)
    B, n, k = 7, 25, 3
    X = rng.standard_normal((B, n, k))
    y = rng.standard_normal((B, n))
    y[4, 2] = np.nan  # one NaN regression in the batch
    X[5, :, 2] = X[5, :, 0]  # one rank-deficient one
    w = rng.uniform(0.5, 2, (B, n))
    w[1, :3] = 0
    for kind in ("ols", "wls"):
        res = R.ols(y, X, cov_type=cov_type) if kind == "ols" else R.wls(y, X, weights=w, cov_type=cov_type)
        assert a(res.params).shape == (B, k + 1) and a(res.nobs).shape == (B,)
        for i in range(B):
            one = (
                R.ols(y[i], X[i], cov_type=cov_type)
                if kind == "ols"
                else R.wls(y[i], X[i], weights=w[i], cov_type=cov_type)
            )
            _batch_equal(res, i, one)
    # broadcasting: one X for many y
    res = R.ols(y[:3], X[0])
    for i in range(3):
        _batch_equal(res, i, R.ols(y[i], X[0]))


def test_threads_bit_identical():
    rng = np.random.default_rng(19)
    Xb = rng.standard_normal((2000, 50, 3))
    yb = rng.standard_normal((2000, 50))
    X1 = rng.standard_normal((300_000, 6))
    y1 = X1 @ np.arange(6.0) + rng.standard_normal(300_000)
    old = sd.get_num_threads()
    try:
        out = []
        for t in (1, 10):
            sd.set_num_threads(t)
            rb = R.ols(yb, Xb, cov_type="HC1")
            r1 = R.wls(y1, X1, weights=np.abs(y1) + 0.5)
            out.append([a(rb.params), a(rb.bse), a(rb.ssr), a(r1.params), a(r1.bse), r1.ssr, a(r1.resid)])
        for x, z in zip(*out):
            assert np.array_equal(x, z)
    finally:
        sd.set_num_threads(old)


# ---- frames, types, names (G19)


def test_frame_inputs():
    y, X = _problem(seed=20)
    df = sd.frame.DataFrame(
        {"age": X[:, 0].tolist(), "income": X[:, 1].tolist(), "flag": [bool(v > 0) for v in X[:, 2]]}
    )
    ys = sd.frame.Series(y.tolist(), name="spend")
    r = R.ols(ys, df)
    assert r.param_names == ["const", "age", "income", "flag"]
    Xa = np.column_stack([X[:, 0], X[:, 1], (X[:, 2] > 0).astype(float)])
    np.testing.assert_array_equal(a(r.params), a(R.ols(y, Xa).params))
    assert isinstance(r.params, sd.ndarray) and isinstance(r.rsquared, float) and isinstance(r.rank, int)
    assert R.ols(y, X).param_names == ["const", "x1", "x2", "x3"]
    assert R.ols(y, X, intercept=False).param_names == ["x1", "x2", "x3"]
    assert R.ols(ys, sd.frame.Series(X[:, 0].tolist(), name="dose")).param_names == ["const", "dose"]
    # missing values become NaN and follow nan_policy
    df2 = sd.frame.DataFrame({"a": [1.0, None, 3.0, 4.0, 5.0], "b": [2.0, 1.0, 0.0, 1.0, 3.0]})
    o = R.ols([1.0, 2.0, 2.5, 4.0, 6.0], df2, nan_policy="omit")
    assert o.nobs == 4


def test_int_bool_inputs():
    y = [1, 3, 2, 5, 4, 6]
    X = [[1, True], [2, False], [3, True], [4, True], [5, False], [6, False]]
    r = R.ols(y, X)
    Xf = [[float(v) for v in row] for row in X]
    np.testing.assert_array_equal(a(r.params), a(R.ols([float(v) for v in y], Xf).params))


# ---- errors


def test_errors():
    y, X = _problem(n=10)
    with pytest.raises(ValueError, match=r"^ols\(\): 'X' must have 10 rows like 'y', not 9$"):
        R.ols(y, X[:9])
    with pytest.raises(TypeError, match=r"^ols\(\): 'X' must be real, not a complex128 array$"):
        R.ols(y, sd.asarray(X) * 1j)
    with pytest.raises(TypeError, match=r"^ols\(\): 'y' must be an sd.ndarray, a sequence of numbers"):
        R.ols("abc", X)
    with pytest.raises(OverflowError, match=r"^ols\(\): Python int too large to convert to float64$"):
        R.ols([10**400] * 10, X)
    with pytest.raises(ValueError, match=r"^ols\(\): 'y' must have at least 1 dimension$"):
        R.ols(1.0, X)
    with pytest.raises(TypeError, match=r"^ols\(\): 'intercept' must be True or False, not int$"):
        R.ols(y, X, intercept=1)
    with pytest.raises(
        ValueError,
        match=r"^ols\(\): 'cov_type' must be 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC' or 'cluster', not 'hac'$",
    ):
        R.ols(y, X, cov_type="hac")
    with pytest.raises(TypeError, match=r"^ols\(\): 'cov_type' must be .* not NoneType$"):
        R.ols(y, X, cov_type=None)
    with pytest.raises(ValueError, match=r"^ols\(\): 'rtol' must be a finite number >= 0$"):
        R.ols(y, X, rtol=-1)
    with pytest.raises(TypeError, match=r"^ols\(\): 'rtol' must be None, an int or a float, not str$"):
        R.ols(y, X, rtol="1")
    with pytest.raises(ValueError, match=r"^ols\(\): 'nan_policy' must be 'propagate', 'omit' or 'raise', not 'x'$"):
        R.ols(y, X, nan_policy="x")
    X2 = X.copy()
    X2[3, 1] = np.inf
    with pytest.raises(ValueError, match=r"^ols\(\): 'X' contains infinity$"):
        R.ols(y, X2)
    w = np.ones(10)
    w[2] = -1
    with pytest.raises(ValueError, match=r"^wls\(\): 'weights' must be >= 0$"):
        R.wls(y, X, weights=w)
    with pytest.raises(ValueError, match=r"^wls\(\): 'weights' must have 10 values like 'y', not 9$"):
        R.wls(y, X, weights=w[:9])
    yb = np.stack([y, y])
    yb[1, 0] = np.inf
    with pytest.raises(ValueError, match=r"^ols\(\): 'y' contains infinity \(at batch index \(1,\)\)$"):
        R.ols(yb, X)
    with pytest.raises(ValueError, match=r"^ols\(\): the batch shapes of 'y' \(2,\), 'X' \(3,\) do not broadcast$"):
        R.ols(np.ones((2, 10)), np.ones((3, 10, 2)))
    with pytest.raises(
        ValueError, match=r"^wls\(\): the batch shapes of 'y' \(3,\), 'X' \(\), 'weights' \(2,\) do not broadcast$"
    ):
        R.wls(np.ones((3, 10)), np.ones((10, 1)), weights=np.ones((2, 10)))
    with pytest.raises(ValueError, match=r"^wls\(\): 'weights' must have 10 values like 'y', not a scalar$"):
        R.wls(y, X, weights=2.0)
    with pytest.raises(TypeError, match=r"^ols\(\): 'y' must be .* or DataFrame, not list of str$"):
        R.ols(["a"] * 10, X)
    with pytest.raises(TypeError, match=r"^ols\(\): 'X' must have numeric columns$"):
        R.ols(y, sd.frame.DataFrame({"x": list("abcdefghij")}))
    r = R.ols(y, X)
    with pytest.raises(ValueError, match=r"^predict\(\): 'X_new' must have 3 columns like the fit's 'X', not 2$"):
        r.predict(X[:, :2])
    with pytest.raises(ValueError, match=r"^predict_interval\(\): 'kind' must be 'mean' or 'observation', not 'x'$"):
        r.predict_interval(X, kind="x")
    with pytest.raises(ValueError, match=r"^conf_int\(\): 'alpha' must be between 0 and 1 \(exclusive\)$"):
        r.conf_int(alpha=1.5)


def test_contract_edge_kappa_1e11():
    # G15: refinement until it no longer matters (at most 3 steps); one step left 1e-6 here
    ref = REFS["ols_kappa11"]["ref"]
    r = fit("ols_kappa11", rtol=0.0)
    assert r.condition_number == pytest.approx(1e11, rel=0.5)
    assert norm_err(r.params, ref["params"]) <= CONTRACT


# ---- decisions G84-G86, G90, G91 (2026-11-19)


def test_constant_y_is_zero_over_zero():
    """TSS exactly 0 (G85): R^2, adjusted R^2 and the F test are 0/0, NaN; ESS is 0. statsmodels
    gives R^2 -inf and F -4 from the rounding."""
    x = np.arange(6.0)
    for r in (
        R.ols(np.ones(6), x),
        R.wls(np.ones(6), x, weights=np.arange(1.0, 7.0)),
        R.ols(np.ones(6), x, cov_type="HC1"),
        R.gls(np.ones(6), x, sigma=np.eye(6) * 2.0),
    ):
        assert math.isnan(r.rsquared) and math.isnan(r.rsquared_adj)
        assert math.isnan(r.fvalue) and math.isnan(r.f_pvalue) and r.ess == 0.0
        assert abs(float(r.params[1])) < 1e-15
    rb = R.ols(np.vstack([np.ones(6), x * 2 + np.r_[0.1, 0, 0, 0.2, 0, 0]]), x)
    assert math.isnan(float(rb.rsquared[0])) and float(rb.ess[0]) == 0.0 and float(rb.rsquared[1]) > 0.9


def test_exact_zero_coefficients_are_positive():
    """G84: no -0.0 among the coefficients (quantreg's simplex left one)."""
    q = R.quantreg([1.0, 2, 3, 4, 5, 7], [1.0, 2, 3, 4, 5, 6])
    assert a(q.params).tolist() == [0.0, 1.0] and math.copysign(1.0, float(q.params[0])) == 1.0
    for r in (R.ols([1.0, 2, 3], [1.0, 2, 3]), R.lasso([1.0, 2, 3, 5], np.c_[[1.0, 2, 3, 4], [0.0] * 4], alpha=0.1)):
        assert all(math.copysign(1.0, v) == 1.0 for v in a(r.params).tolist() if v == 0.0)


def test_inputs_are_copied():
    """G86: changing the arrays given to the fit (an sd.ndarray, or the numpy array it views)
    afterwards changes none of the lazy fields."""
    x = sd.asarray([1.0, 2, 3, 4, 5, 6, 7, 8])
    y0 = [52.0, 55, 61, 64, 68, 70, 76, 79]
    for make in (lambda v: sd.asarray(v), lambda v: np.asarray(v, dtype=float)):
        X = make(np.c_[np.asarray(x.tolist()), np.asarray(x.tolist()) ** 2])
        y = make(y0)
        w = make(np.arange(1.0, 9.0))
        fits = [R.ols(y, X), R.wls(y, X, weights=w)]
        want = [(a(f.leverage), a(f.cooks_distance), f.breusch_pagan().lm, f.jarque_bera().statistic) for f in fits]
        X[0, 1] = 100.0
        y[1] = -5.0
        w[2] = 50.0
        for f, (lev, cook, bp, jb) in zip(fits, want):
            assert a(f.leverage).tolist() == lev.tolist() and a(f.cooks_distance).tolist() == cook.tolist()
            assert f.breusch_pagan().lm == bp and f.jarque_bera().statistic == jb
    assert (
        abs(float(R.ols(y0, np.c_[np.asarray(x.tolist()), np.asarray(x.tolist()) ** 2]).leverage[0]) - 0.70833) < 1e-5
    )


def test_predict_columns_and_inf():
    """G90: a DataFrame fit predicts from a DataFrame with the same columns in the same order
    (scikit-learn's rule), else ValueError; one side a DataFrame: by position. G91: inf in
    X_new is a ValueError, NaN gives NaN."""
    import stardust.frame as fr

    F = fr.DataFrame({"a": [1.0, 2, 3, 4, 5, 6], "b": [2.0, 1, 4, 3, 6, 5]})
    y = [3.0, 4, 8, 9, 13, 14]
    arr = np.c_[[1.0, 2, 3, 4, 5, 6], [2.0, 1, 4, 3, 6, 5]]
    good = fr.DataFrame({"a": [1.0, 2], "b": [10.0, 20]})
    for fit in (R.ols(y, F), R.glm(y, F, family="poisson"), R.rlm(y, F), R.quantreg(y, F), R.ridge(y, F, alpha=0.1)):
        want = a(fit.predict(np.c_[[1.0, 2], [10.0, 20]]))
        assert a(fit.predict(good)).tolist() == want.tolist()
        with pytest.raises(
            ValueError,
            match=r"^predict\(\): 'X_new' must have the columns \['a', 'b'\] in the "
            r"fit's order, not \['b', 'a'\]$",
        ):
            fit.predict(fr.DataFrame({"b": [10.0, 20], "a": [1.0, 2]}))
        with pytest.raises(
            ValueError,
            match=r"^predict\(\): 'X_new' must have the columns \['a', 'b'\] like the "
            r"fit's 'X', not \['a', 'c'\]$",
        ):
            fit.predict(fr.DataFrame({"a": [1.0, 2], "c": [10.0, 20]}))
        with pytest.raises(ValueError, match=r"^predict\(\): 'X_new' contains infinity$"):
            fit.predict(np.c_[[1.0, 2], [10.0, math.inf]])
        nanp = a(fit.predict(np.c_[[1.0, 2], [10.0, math.nan]]))
        assert math.isnan(nanp[1]) and nanp[0] == want[0]
    # an array fit takes a DataFrame by position
    ra = R.ols(y, arr)
    assert (
        a(ra.predict(fr.DataFrame({"b": [1.0, 2], "a": [10.0, 20]}))).tolist()
        == a(ra.predict(np.c_[[1.0, 2], [10.0, 20]])).tolist()
    )
    with pytest.raises(ValueError, match=r"^predict_interval\(\): 'X_new' contains infinity$"):
        ra.predict_interval(np.c_[[1.0], [math.inf]])
    with pytest.raises(ValueError, match=r"^predict\(\): 'X_new' must have 1 column like the fit's 'X', not 2$"):
        R.ols(y, arr[:, 0]).predict(arr)


def test_constant_y_exact_solution():
    """G92: an intercept and y exactly constant over the fit's rows: (y0, 0, ..., 0) exactly,
    fitted y0, residuals y - y0, a perfect fit (G22) with the G85 NaNs. Without the intercept
    nothing changes."""
    x = np.arange(1.0, 9.0)
    X = np.c_[x, x * x]
    fits = [
        R.ols([3.0] * 8, x),
        R.ols([0.1] * 8, X),
        R.ols([0.1] * 8, X, cov_type="HC1"),
        R.wls([0.1] * 7 + [5.0], X, weights=[1.0] * 7 + [0.0]),
        R.gls([0.1] * 8, X, sigma=np.eye(8) + 0.3),
    ]
    for r in fits:
        p = a(r.params)
        assert p[0] in (3.0, 0.1) and p[1:].tolist() == [0.0] * (len(p) - 1)
        assert np.all(a(r.fittedvalues) == p[0]) and np.all(a(r.resid)[:7] == 0.0)
        assert r.ssr == 0.0 and r.scale == 0.0 and np.all(a(r.bse) == 0.0) and r.llf == math.inf
        assert math.isnan(r.rsquared) and math.isnan(r.fvalue)
        assert np.all(np.isnan(a(r.resid_studentized_internal)[:7]))
    assert float(fits[3].resid[7]) == 4.9
    # aliased columns stay NaN; batches per regression; GLS batches too
    al = R.ols([0.1] * 8, np.c_[x, 2 * x])
    assert a(al.params)[:2].tolist() == [0.1, 0.0] and math.isnan(a(al.params)[2])
    rb = R.ols(np.vstack([[0.1] * 8, x + np.r_[0.1, 0, 0, 0, 0, 0, 0, 0.2]]), X)
    one = R.ols(x + np.r_[0.1, 0, 0, 0, 0, 0, 0, 0.2], X)
    assert a(rb.params)[0].tolist() == [0.1, 0.0, 0.0] and a(rb.params)[1].tolist() == a(one.params).tolist()
    gb = R.gls(np.vstack([[0.1] * 8, [0.2] * 8]), X, sigma=np.eye(8) * 2.0)
    assert a(gb.params).tolist() == [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    assert not a(R.ols([3.0] * 8, x, intercept=False).params).tolist() == [0.0]
    with pytest.raises(
        ValueError, match=r"^theilslopes\(\): the batch shapes of 'y' \(3,\), 'x' \(2,\) do not broadcast$"
    ):
        R.theilslopes(np.ones((3, 5)), np.ones((2, 5)))
    with pytest.raises(ValueError, match=r"^glm\(\): the batch shapes of 'y' \(3,\), 'X' \(2,\) do not broadcast$"):
        R.glm(np.ones((3, 5)), np.ones((2, 5, 1)), family="poisson")


# ---- G95: df_resid = 0, the perfect fit's tolerance


def test_perfect_fit_by_tolerance():
    """y = 1 + 2x + 3x^2 exactly: the QR leaves residuals of rounding size; within the
    (k + 16) u tolerance they are a perfect fit (ssr 0, bse 0, t and p NaN (D-3), llf +inf,
    diagnostics NaN)."""
    x = np.arange(10.0)
    y = 1 + 2 * x + 3 * x**2
    Xs = np.column_stack([x, x**2])
    r = R.ols(y, Xs)
    assert r.ssr == 0.0 and r.scale == 0.0 and np.all(a(r.bse) == 0.0) and np.all(np.isnan(a(r.pvalues)))
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


# ---- predictions, scales, messages


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


@pytest.mark.parametrize(
    "sy,sx",
    [
        (1e-160, 1.0),
        (1e-170, 1.0),
        (1e-300, 1.0),
        (1e160, 1.0),
        (1e300, 1.0),
        (1.0, 1e-170),
        (1.0, 1e160),
        (1.0, 1e-300),
        (1e-200, 1e-200),
        (1e140, 1e-140),
    ],
)
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
    np.testing.assert_allclose(
        a(R.wls(y * sy, Xs * sx, weights=w).tvalues), a(R.wls(y, Xs, weights=w).tvalues), rtol=1e-12
    )
    S = np.eye(8) + 0.3 * np.eye(8, k=1) + 0.3 * np.eye(8, k=-1)
    np.testing.assert_allclose(a(R.gls(y * sy, Xs * sx, sigma=S).tvalues), a(R.gls(y, Xs, sigma=S).tvalues), rtol=1e-11)


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
    with pytest.raises(
        ValueError,
        match=r"^predict\(\): the batch shapes of 'X_new' \(2,\) and the fit \(3,\) do not "
        r"broadcast$",
    ):
        r.predict(RNG.normal(size=(2, 4, 2)))
    with pytest.raises(ValueError, match=r"^predict_interval\(\): the batch shapes of 'X_new' \(2,\)"):
        r.predict_interval(RNG.normal(size=(2, 4, 2)))


def test_extreme_scale_t_in_fitted_units():
    """R90: t from beta' / sqrt(C') in the fitted units, also when a coefficient in the data's
    units overflows or underflows."""
    h = np.array([1, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 8])
    s = np.array([52, 55, 61, 58, 66, 63, 70, 74, 73, 82.0])
    b = R.ols(s, h)
    r = R.ols(s * 1e300, h * 1e-300)  # the slope's 1e600 overflows
    np.testing.assert_allclose(a(r.tvalues), a(b.tvalues), rtol=1e-13)
    np.testing.assert_allclose(a(r.pvalues), a(b.pvalues), rtol=1e-12)
    x = np.arange(8.0)
    y = np.array([1.0, 2.9, 5.2, 6.8, 9.1, 11.2, 12.8, 15.1]) + 3 * np.sin(x)
    b = R.ols(y, x)
    r = R.ols(y * 1e-165, x * 1e160)  # the slope's 1e-325 underflows
    np.testing.assert_allclose(a(r.tvalues), a(b.tvalues), rtol=1e-13)


def test_conf_int_ends_overflow_with_their_signs():
    """R103: conf_int's ends in the fitted units, then scaled as the parameter: an overflowing
    coefficient's interval is [inf, inf] (or [-inf, -inf]), never NaN from inf - inf."""
    s = np.array([1.1, 1.9, 3.2, 3.8, 5.1, 6.3])
    h = np.arange(1.0, 7.0)
    W = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    S = np.diag(W) + 0.1
    fits = (
        lambda y, x: R.ols(y, x),
        lambda y, x: R.ols(y, x, cov_type="HC1"),
        lambda y, x: R.wls(y, x, weights=W),
        lambda y, x: R.gls(y, x, sigma=S),
    )
    for fit in fits:
        b = a(fit(s, h).conf_int())
        for sy, sx in ((1e300, 1e-300), (-1e300, 1e-300), (1e300, -1e-300)):
            ci = a(fit(s * sy, h * sx).conf_int())
            assert not np.any(np.isnan(ci))
            sign = math.copysign(1.0, sy * sx)
            assert ci[1].tolist() == [sign * math.inf] * 2
            # the intercept's ends: the reference's, scaled exactly by sy
            np.testing.assert_allclose(ci[0], np.sort(b[0] * sy), rtol=1e-13)
        # an underflowing slope: both ends 0 (the interval is positive)
        ci = a(fit(s * 1e-300, h * 1e300).conf_int())
        assert ci[1].tolist() == [0.0, 0.0] and not np.any(np.isnan(ci))
    # a batch keeps each regression's own ends
    r = R.ols(np.stack([s * 1e300, s]), h * 1e-300)
    ci = a(r.conf_int())
    assert ci[0, 1].tolist() == [math.inf, math.inf] and np.all(np.isfinite(ci[1]))


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
