"""G1b: HAC and cluster covariances, the diagnostics and summary() (docs/api/regression.md §2,
docs/design/regression.md §3.3, G26-G29).

Accuracy against Arb (tests/regression/ref/linear.json), then statsmodels, the rules for zero
weights, aliased columns, exact fits and NaN, batches, threads and errors."""

import math
import re

import numpy as np
import pytest

import stardust as sd

from . import refdata
from .smalldata import RNG, N, X

R = sd.regression
REFS = refdata.load()
RANDOM = ["ols_int", "ols_noint", "wls_zero", "wls_noint", "ols_cond"]
# Wampler2's residuals are at the rounding level of its fitted values (docs/design/regression.md
# §3.1): nothing computed from them has relative accuracy; Wampler1 is an exact fit
NIST = [k for k in REFS if k.startswith("nist_") and k not in ("nist_Wampler1", "nist_Wampler2")]
U = 2.0**-53


def a(x):
    return np.asarray(x, dtype=float)


def fit(name, **kw):
    p = REFS[name]
    if p["weights"] is not None:
        return R.wls(p["y"], p["X"], weights=p["weights"], intercept=p["intercept"], **kw)
    return R.ols(p["y"], p["X"], intercept=p["intercept"], **kw)


def max_rel(got, ref):
    return max(refdata.rel_err(g, r) for g, r in zip(a(got).ravel(), ref))


def scaled_err(got, ref):
    """max |got - ref| / max |ref|: per-observation values, some near 0 (a residual's relative
    accuracy is lost to the rounding of params, |x||beta| / |e| times u)."""
    r = np.array([hi + lo for hi, lo in ref])
    return np.max(np.abs(a(got) - r)) / np.max(np.abs(r))


def _problem(n=60, k=3, seed=0, ar=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    e = rng.standard_normal(n) * (1 + np.abs(X[:, 0]))
    for t in range(1, n):
        e[t] += ar * e[t - 1]
    y = X @ np.arange(1.0, k + 1) + 0.5 + e
    return y, X


# ---- Arb references


@pytest.mark.parametrize("name", RANDOM + NIST)
def test_diagnostics_against_arb(name):
    ref = REFS[name]["ref"]
    r = fit(name)
    # values from residuals and the subspace of X: about kappa u (the hat values of a nearly
    # dependent design move with its column space)
    tol = max(1e-13, 10 * r.condition_number * U)
    assert scaled_err(r.leverage, ref["hat"]) <= tol
    assert scaled_err(r.resid_studentized_internal, ref["resid_studentized_internal"]) <= tol
    assert scaled_err(r.resid_studentized_external, ref["resid_studentized_external"]) <= tol
    assert scaled_err(r.cooks_distance, ref["cooks_distance"]) <= tol
    # sums over the residuals in double-double
    assert refdata.rel_err(r.durbin_watson, ref["durbin_watson"]) <= tol
    jb = r.jarque_bera()
    assert refdata.rel_err(jb.statistic, ref["jarque_bera"]) <= tol
    assert refdata.rel_err(jb.pvalue, ref["jarque_bera_pvalue"]) <= tol
    bp = r.breusch_pagan()
    for got, key in zip(bp, ("bp_lm", "bp_lm_pvalue", "bp_fvalue", "bp_f_pvalue")):
        assert refdata.rel_err(got, ref[key]) <= tol, key
    bp0 = r.breusch_pagan(studentize=False)
    assert refdata.rel_err(bp0.lm, ref["bp_lm_nonstudentized"]) <= tol
    assert refdata.rel_err(bp0.lm_pvalue, ref["bp_lm_nonstudentized_pvalue"]) <= tol
    assert bp0.fvalue == bp.fvalue and bp0.f_pvalue == bp.f_pvalue
    # HAC with the default lags floor(4 (n/100)^(2/9))
    h = fit(name, cov_type="HAC")
    assert max_rel(h.bse, ref["bse_HAC"]) <= tol


@pytest.mark.parametrize("name", RANDOM)
def test_cluster_against_arb(name):
    p, ref, meta = REFS[name], REFS[name]["ref"], REFS[name]["meta"]
    r = fit(name, cov_type="cluster", groups=p["groups"])
    tol = max(1e-13, 10 * r.condition_number * U)
    assert max_rel(r.bse, ref["bse_cluster"]) <= tol
    assert max_rel(r.pvalues, ref["pvalues_cluster"]) <= 10 * tol
    # the Wald F in Q's coordinates (no inverse of the covariance): kappa u, not kappa^2 u
    assert refdata.rel_err(r.fvalue, ref["fvalue_cluster"]) <= 10 * tol
    assert refdata.rel_err(r.f_pvalue, ref["f_pvalue_cluster"]) <= 100 * tol
    # G counts the clusters among the fit's rows (wls_zero: one cluster has only zero weights)
    assert r._df_inference() == meta["n_groups"] - 1


# ---- statsmodels


@pytest.mark.parametrize("maxlags", [None, 0, 1, 4, 200])
@pytest.mark.parametrize("intercept", [True, False])
def test_hac_against_statsmodels(maxlags, intercept):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _problem(seed=1, ar=0.6)
    n = len(y)
    r = R.ols(y, X, intercept=intercept, cov_type="HAC", maxlags=maxlags)
    L = math.floor(4 * (n / 100) ** (2 / 9)) if maxlags is None else maxlags
    # statsmodels' HAC default: use_correction=False (no small-sample factor); our p-values
    # use t(df_resid) (G23), its use_t=True
    s = sm.OLS(y, sm.add_constant(X) if intercept else X).fit(cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True)
    for name in ("bse", "tvalues", "pvalues"):
        np.testing.assert_allclose(a(getattr(r, name)), getattr(s, name), rtol=1e-12, err_msg=name)
    np.testing.assert_allclose(a(r.cov_params()), s.cov_params(), rtol=1e-12)
    np.testing.assert_allclose(a(r.conf_int(alpha=0.1)), s.conf_int(0.1), rtol=1e-12)
    assert r.fvalue == pytest.approx(s.fvalue, rel=1e-12) and r.f_pvalue == pytest.approx(s.f_pvalue, rel=1e-11)
    assert r.df_resid == s.df_resid


def test_hac_zero_lags_is_hc0():
    y, X = _problem(seed=2)
    h = R.ols(y, X, cov_type="HAC", maxlags=0)
    np.testing.assert_allclose(a(h.cov_params()), a(R.ols(y, X, cov_type="HC0").cov_params()), rtol=1e-15)


def _labels(g, kind):
    if kind == "int64":
        return sd.asarray(g)
    if kind == "list":
        return [int(v) for v in g]
    if kind == "str":
        return [f"firm-{v}" for v in g]
    if kind == "float":
        return sd.asarray(g.astype(float) * 0.5)
    return sd.frame.Series([f"firm-{v}" for v in g], name="firm")


@pytest.mark.parametrize("kind", ["int64", "list", "str", "float", "series"])
def test_cluster_against_statsmodels(kind):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _problem(seed=3)
    g = np.random.default_rng(4).integers(0, 9, len(y))
    r = R.ols(y, X, cov_type="cluster", groups=_labels(g, kind))
    # statsmodels with use_t=True: G/(G-1) (n-1)/(n-k), t and F with G - 1 degrees of freedom
    # (df_resid_inference), as Stata
    s = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": g}, use_t=True)
    assert s.df_resid_inference == 8 and r.df_resid == s.df_resid
    for name in ("bse", "tvalues", "pvalues"):
        np.testing.assert_allclose(a(getattr(r, name)), getattr(s, name), rtol=1e-12, err_msg=name)
    np.testing.assert_allclose(a(r.cov_params()), s.cov_params(), rtol=1e-12)
    np.testing.assert_allclose(a(r.conf_int()), s.conf_int(), rtol=1e-12)
    assert r.fvalue == pytest.approx(s.fvalue, rel=1e-12) and r.f_pvalue == pytest.approx(s.f_pvalue, rel=1e-11)
    # prediction intervals use t(G - 1) like conf_int; statsmodels' get_prediction takes
    # df_resid there (its own conf_int takes G - 1), so only its standard error is compared
    st = pytest.importorskip("scipy.stats")
    pr = s.get_prediction(sm.add_constant(X[:5])).summary_frame()
    want = pr["mean"] - st.t.isf(0.025, 8) * pr["mean_se"]
    np.testing.assert_allclose(a(r.predict_interval(X[:5]))[:, 0], want, rtol=1e-12)


@pytest.mark.parametrize("cov_type", ["HAC", "cluster"])
def test_wls_gls_robust_against_statsmodels(cov_type):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _problem(seed=5, ar=0.4)
    g = np.random.default_rng(6).integers(0, 7, len(y))
    kw = {"maxlags": 3} if cov_type == "HAC" else {"groups": g}
    w = np.random.default_rng(7).uniform(0.3, 3.0, len(y))
    r = R.wls(y, X, weights=w, cov_type=cov_type, **kw)
    s = sm.WLS(y, sm.add_constant(X), weights=w).fit(cov_type=cov_type, cov_kwds=kw, use_t=True)
    np.testing.assert_allclose(a(r.bse), s.bse, rtol=1e-12)
    np.testing.assert_allclose(a(r.pvalues), s.pvalues, rtol=1e-11)
    # GLS at n = 30 (scipy's cholesky in this environment, test_linear.test_gls)
    kw = {"maxlags": 2} if cov_type == "HAC" else {"groups": g[:30]}
    idx = np.arange(30)
    S = 0.5 * np.eye(30) + 0.5 * np.exp(-np.abs(np.subtract.outer(idx, idx)) / 3)
    r = R.gls(y[:30], X[:30], sigma=S, cov_type=cov_type, **kw)
    s = sm.GLS(y[:30], sm.add_constant(X[:30]), sigma=S).fit(cov_type=cov_type, cov_kwds=kw, use_t=True)
    np.testing.assert_allclose(a(r.bse), s.bse, rtol=1e-11)
    np.testing.assert_allclose(a(r.pvalues), s.pvalues, rtol=1e-10)


def test_diagnostics_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    y, X = _problem(seed=8, ar=0.3)
    for intercept in (True, False):
        r = R.ols(y, X, intercept=intercept)
        Xs = sm.add_constant(X) if intercept else X
        s = sm.OLS(y, Xs).fit()
        inf = s.get_influence()
        np.testing.assert_allclose(a(r.leverage), inf.hat_matrix_diag, rtol=1e-12)
        np.testing.assert_allclose(a(r.resid_studentized_internal), inf.resid_studentized_internal, rtol=1e-11)
        np.testing.assert_allclose(a(r.resid_studentized_external), inf.resid_studentized_external, rtol=1e-11)
        np.testing.assert_allclose(a(r.cooks_distance), inf.cooks_distance[0], rtol=1e-11)
        assert r.durbin_watson == pytest.approx(durbin_watson(s.wresid), rel=1e-13)
        jb, jbp, _, _ = jarque_bera(s.wresid)  # statsmodels centers the moments (no intercept: mean != 0)
        assert r.jarque_bera() == pytest.approx((jb, jbp), rel=1e-12)
        # the auxiliary regression always has a constant (statsmodels: the exog given)
        Z = sm.add_constant(X)
        for stud in (True, False):
            assert tuple(r.breusch_pagan(studentize=stud)) == pytest.approx(
                het_breuschpagan(s.resid, Z, robust=stud), rel=1e-12
            )
    # WLS: the transformed model's (statsmodels' wresid on a constant and wexog)
    w = np.random.default_rng(9).uniform(0.3, 3.0, len(y))
    r = R.wls(y, X, weights=w)
    s = sm.WLS(y, sm.add_constant(X), weights=w).fit()
    Z = np.column_stack([np.ones(len(y)), s.model.wexog])
    assert tuple(r.breusch_pagan()) == pytest.approx(het_breuschpagan(s.wresid, Z), rel=1e-12)
    assert r.durbin_watson == pytest.approx(durbin_watson(s.wresid), rel=1e-13)
    assert r.jarque_bera() == pytest.approx(jarque_bera(s.wresid)[:2], rel=1e-12)
    ow = sm.OLS(s.model.wendog, s.model.wexog).fit().get_influence()
    np.testing.assert_allclose(a(r.leverage), ow.hat_matrix_diag, rtol=1e-12)
    np.testing.assert_allclose(a(r.cooks_distance), ow.cooks_distance[0], rtol=1e-10)


# ---- zero weights, aliased columns, exact fits, NaN


def test_zero_weights_out_of_robust_sums():
    # G20: a WLS with zero-weight rows equals, bit for bit, the WLS without them; the HAC lags
    # pair the fit's rows in order and a cluster with only zero-weight rows is not counted.
    # (statsmodels keeps those rows as time periods with zero scores and counts the cluster.)
    y, X = _problem(seed=10)
    n = len(y)
    w = np.random.default_rng(11).uniform(0.5, 2.0, n)
    zero = [3, 4, 20]
    w[zero] = 0.0
    g = np.random.default_rng(12).integers(0, 6, n)
    g[zero] = 99
    keep = w > 0
    for ct, kw, kws in (("HAC", {"maxlags": 3}, {"maxlags": 3}), ("cluster", {"groups": g}, {"groups": g[keep]})):
        r = R.wls(y, X, weights=w, cov_type=ct, **kw)
        sub = R.wls(y[keep], X[keep], weights=w[keep], cov_type=ct, **kws)
        for name in ("bse", "pvalues", "fvalue", "f_pvalue"):
            np.testing.assert_array_equal(a(getattr(r, name)), a(getattr(sub, name)), err_msg=name)
    assert R.wls(y, X, weights=w, cov_type="cluster", groups=g)._df_inference() == 5
    # diagnostics: the weighted formulas give 0 at those rows; the sums leave them out
    r = R.wls(y, X, weights=w)
    sub = R.wls(y[keep], X[keep], weights=w[keep])
    for name in ("leverage", "resid_studentized_internal", "resid_studentized_external", "cooks_distance"):
        v = a(getattr(r, name))
        assert np.all(v[zero] == 0.0), name
        np.testing.assert_allclose(v[keep], a(getattr(sub, name)), rtol=1e-13, atol=1e-15, err_msg=name)
    assert r.durbin_watson == sub.durbin_watson and r.jarque_bera() == sub.jarque_bera()
    assert r.breusch_pagan() == pytest.approx(sub.breusch_pagan(), rel=1e-13)


def test_aliased_columns():
    # the reduced problem's values, as for the fit itself (G9)
    y, X = _problem(seed=13)
    X2 = np.column_stack([X[:, 0], X[:, 1], X[:, 0] - 2 * X[:, 1], X[:, 2]])
    g = np.random.default_rng(14).integers(0, 8, len(y))
    for ct, kw in (("HAC", {}), ("cluster", {"groups": g})):
        r, ref = R.ols(y, X2, cov_type=ct, **kw), R.ols(y, X, cov_type=ct, **kw)
        b = a(r.bse)
        assert math.isnan(b[3])
        np.testing.assert_array_equal(b[[0, 1, 2, 4]], a(ref.bse))
        assert r.fvalue == ref.fvalue
    r, ref = R.ols(y, X2), R.ols(y, X)
    for name in ("leverage", "resid_studentized_internal", "resid_studentized_external", "cooks_distance"):
        np.testing.assert_array_equal(a(getattr(r, name)), a(getattr(ref, name)), err_msg=name)
    assert r.durbin_watson == ref.durbin_watson and r.jarque_bera() == ref.jarque_bera()
    assert r.breusch_pagan() == pytest.approx(ref.breusch_pagan(), rel=1e-14)
    assert "Aliased" in r.summary() and "x3" in r.summary().split("Aliased")[1]


def test_exact_fit_and_nan():
    x = np.arange(10.0)
    X = np.column_stack([x, x**2])
    r = R.ols(1 + 2 * x + 3 * x**2, X, cov_type="HAC")
    assert np.all(a(r.bse) == 0.0)
    # 0/0: no limit, NaN (the leverage does not depend on y)
    r = R.ols(1 + 2 * x + 3 * x**2, X)
    assert np.all(np.isfinite(a(r.leverage))) and abs(float(sd.sum(r.leverage)) - 3.0) < 1e-14
    for name in ("resid_studentized_internal", "resid_studentized_external", "cooks_distance"):
        assert np.all(np.isnan(a(getattr(r, name)))), name
    assert math.isnan(r.durbin_watson) and all(math.isnan(v) for v in r.jarque_bera())
    assert all(math.isnan(v) for v in r.breusch_pagan())
    # a NaN regression: everything NaN
    y, X = _problem(seed=15)
    y[2] = np.nan
    g = np.arange(len(y)) % 5
    r = R.ols(y, X, cov_type="cluster", groups=g)
    assert np.all(np.isnan(a(r.bse))) and math.isnan(r.fvalue) and math.isnan(r.f_pvalue)
    for name in ("leverage", "resid_studentized_internal", "cooks_distance"):
        assert np.all(np.isnan(a(getattr(r, name)))), name
    assert math.isnan(r.durbin_watson) and all(math.isnan(v) for v in r.jarque_bera())
    assert all(math.isnan(v) for v in r.breusch_pagan())


def test_group_labels_and_nan_policy():
    y, X = _problem(seed=16)
    n = len(y)
    g = (np.arange(n) % 6).astype(float)
    g[7] = np.nan
    # propagate: a missing label is NaN data
    r = R.ols(y, X, cov_type="cluster", groups=g)
    assert np.all(np.isnan(a(r.params))) and r.rank == 0
    with pytest.raises(ValueError, match=r"^ols\(\): the input contains NaN \(nan_policy='raise'\)$"):
        R.ols(y, X, cov_type="cluster", groups=g, nan_policy="raise")
    keep = ~np.isnan(g)
    o = R.ols(y, X, cov_type="cluster", groups=g, nan_policy="omit")
    ref = R.ols(y[keep], X[keep], cov_type="cluster", groups=g[keep])
    np.testing.assert_array_equal(a(o.bse), a(ref.bse))
    # None in a sequence, NA in a frame Series: the same
    labels = [f"c{int(v)}" if v == v else None for v in g]
    o2 = R.ols(y, X, cov_type="cluster", groups=labels, nan_policy="omit")
    o3 = R.ols(y, X, cov_type="cluster", groups=sd.frame.Series(labels), nan_policy="omit")
    np.testing.assert_array_equal(a(o2.bse), a(ref.bse))
    np.testing.assert_array_equal(a(o3.bse), a(ref.bse))
    assert np.all(np.isnan(a(R.ols(y, X, cov_type="cluster", groups=labels).bse)))
    # only equality of labels matters
    b1 = R.ols(y, X, cov_type="cluster", groups=np.arange(n) % 4).bse
    b2 = R.ols(y, X, cov_type="cluster", groups=[["d", "a", "c", "b"][i % 4] for i in range(n)]).bse
    np.testing.assert_array_equal(a(b1), a(b2))
    # one cluster per row: HC1's covariance (n/(n-k)), with n - 1 degrees of freedom
    c = R.ols(y, X, cov_type="cluster", groups=np.arange(n))
    np.testing.assert_allclose(a(c.cov_params()), a(R.ols(y, X, cov_type="HC1").cov_params()), rtol=1e-14)
    assert c._df_inference() == n - 1
    # one cluster: no estimate (G/(G-1) is infinite and the scores sum to 0)
    one = R.ols(y, X, cov_type="cluster", groups=np.zeros(n, dtype=np.int64))
    assert np.all(np.isnan(a(one.bse))) and np.all(np.isnan(a(one.pvalues))) and math.isnan(one.fvalue)


def test_cluster_f_needs_groups():
    """G - 1 below the number of tested coefficients: the meat (rank <= G - 1) is singular and
    fvalue, f_pvalue are NaN, though rounding can let its Cholesky through (it gave 1.83e20).
    With G - 1 = q the test is defined (statsmodels' value)."""
    import statsmodels.api as sm

    h = np.arange(1.0, 9.0)
    y = np.array([51.0, 58, 57, 66, 64, 73, 75, 80])
    X = np.c_[h, h * h]
    for groups, intercept in (
        ([1, 1, 1, 1, 2, 2, 2, 2], True),
        ([1, 1, 1, 1, 2, 2, 2, 2], False),
        ([1, 1, 1, 2, 2, 2, 2, 2], True),
    ):
        for fit in (
            R.ols(y, X, cov_type="cluster", groups=groups, intercept=intercept),
            R.wls(y, X, weights=np.full(8, 2.0), cov_type="cluster", groups=groups, intercept=intercept),
        ):
            assert math.isnan(fit.fvalue) and math.isnan(fit.f_pvalue)
            assert np.all(np.isfinite(a(fit.bse)))
    g3 = np.array([1, 1, 1, 2, 2, 2, 3, 3])
    r = R.ols(y, X, cov_type="cluster", groups=g3)
    s = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds=dict(groups=g3))
    assert r.fvalue == pytest.approx(s.fvalue, rel=1e-11) and r.f_pvalue == pytest.approx(s.f_pvalue, rel=1e-10)
    # a batch: the rule per regression
    rb = R.ols(np.vstack([y, y]), X, cov_type="cluster", groups=np.vstack([[1, 1, 1, 1, 2, 2, 2, 2], g3]))
    assert math.isnan(float(rb.fvalue[0])) and float(rb.fvalue[1]) == r.fvalue


def test_many_clusters_path():
    # G x k past 2^22 doubles: the group sums go in blocks of groups (rows sorted by group)
    rng = np.random.default_rng(17)
    n, k = 300_000, 14
    X = rng.standard_normal((n, k))
    y = X @ rng.standard_normal(k) + rng.standard_normal(n)
    c = R.ols(y, X, cov_type="cluster", groups=np.arange(n)[::-1].copy())
    np.testing.assert_allclose(a(c.cov_params()), a(R.ols(y, X, cov_type="HC1").cov_params()), rtol=1e-13)
    g = rng.integers(0, n // 2, n)  # about 130,000 clusters of one to a few rows
    c = R.ols(y, X, cov_type="cluster", groups=g)
    # against the direct sums (numpy) of the same scores
    Xc = np.column_stack([np.ones(n), X])
    Q, Rm = np.linalg.qr(Xc)
    u = Q * a(c.resid)[:, None]
    _, inv = np.unique(g, return_inverse=True)
    Sg = np.zeros((inv.max() + 1, k + 1))
    np.add.at(Sg, inv, u)
    G = Sg.shape[0]
    Ri = np.linalg.inv(Rm)
    V = Ri @ (Sg.T @ Sg) @ Ri.T * G / (G - 1) * (n - 1) / (n - k - 1)
    np.testing.assert_allclose(a(c.cov_params()), V, rtol=1e-10)


# ---- batches and threads


@pytest.mark.parametrize("cov_type", ["HAC", "cluster"])
def test_batch_equals_loop(cov_type):
    rng = np.random.default_rng(18)
    B, n, k = 6, 40, 3
    X = rng.standard_normal((B, n, k))
    y = rng.standard_normal((B, n))
    y[2, 5] = np.nan
    X[3, :, 2] = X[3, :, 0]
    g1 = rng.integers(0, 7, n)  # one set of labels for every regression
    gB = rng.integers(0, 7, (B, n))  # labels per regression (an int64 array with batch axes)
    kws = [{"maxlags": 2}, {}] if cov_type == "HAC" else [{"groups": g1}, {"groups": gB}]
    for kw in kws:
        res = R.ols(y, X, cov_type=cov_type, **{k2: (sd.asarray(v) if k2 == "groups" else v) for k2, v in kw.items()})
        for i in range(B):
            one_kw = dict(kw)
            if "groups" in kw and np.ndim(kw["groups"]) == 2:
                one_kw["groups"] = kw["groups"][i]
            one = R.ols(y[i], X[i], cov_type=cov_type, **one_kw)
            for name in ("bse", "pvalues", "leverage", "resid_studentized_external", "cooks_distance"):
                np.testing.assert_array_equal(a(getattr(res, name))[i], a(getattr(one, name)), err_msg=name)
            for name in ("fvalue", "f_pvalue", "durbin_watson"):
                got, want = float(getattr(res, name)[i]), getattr(one, name)
                assert got == want or (math.isnan(got) and math.isnan(want)), name
            for got, want in zip(res.jarque_bera(), one.jarque_bera()):
                assert float(got[i]) == want or (math.isnan(want) and math.isnan(float(got[i])))
            for got, want in zip(res.breusch_pagan(), one.breusch_pagan()):
                assert float(got[i]) == want or (math.isnan(want) and math.isnan(float(got[i])))
    with pytest.raises(ValueError, match=r"^summary\(\): takes one regression, not a batch of shape \(6,\)$"):
        res.summary()


def test_threads_bit_identical():
    rng = np.random.default_rng(19)
    X = rng.standard_normal((200_000, 8))
    y = X @ np.arange(8.0) + rng.standard_normal(200_000)
    g = rng.integers(0, 500, 200_000)
    old = sd.get_num_threads()
    try:
        out = []
        for t in (1, 10):
            sd.set_num_threads(t)
            h = R.ols(y, X, cov_type="HAC")
            c = R.ols(y, X, cov_type="cluster", groups=g)
            out.append([a(h.bse), h.fvalue, a(c.bse), c.fvalue, a(h.leverage), h.durbin_watson, tuple(h.jarque_bera())])
        for x, z in zip(*out):
            assert np.array_equal(x, z) if isinstance(x, np.ndarray) else x == z
    finally:
        sd.set_num_threads(old)


# ---- errors


def test_errors():
    y, X = _problem(n=20, seed=20)
    g = np.arange(20) % 4
    with pytest.raises(ValueError, match=r"^ols\(\): 'maxlags' is only used when cov_type='HAC'$"):
        R.ols(y, X, maxlags=2)
    with pytest.raises(ValueError, match=r"^wls\(\): 'groups' is only used when cov_type='cluster'$"):
        R.wls(y, X, weights=np.ones(20), cov_type="HC1", groups=g)
    with pytest.raises(ValueError, match=r"^gls\(\): 'groups' is required when cov_type='cluster'$"):
        R.gls(y, X, sigma=np.eye(20), cov_type="cluster")
    with pytest.raises(TypeError, match=r"^ols\(\): 'maxlags' must be None or an int, not float$"):
        R.ols(y, X, cov_type="HAC", maxlags=2.0)
    with pytest.raises(TypeError, match=r"^ols\(\): 'maxlags' must be None or an int, not bool$"):
        R.ols(y, X, cov_type="HAC", maxlags=True)
    with pytest.raises(ValueError, match=r"^ols\(\): 'maxlags' must be >= 0$"):
        R.ols(y, X, cov_type="HAC", maxlags=-1)
    with pytest.raises(OverflowError, match=r"^ols\(\): 'maxlags' is out of range$"):
        R.ols(y, X, cov_type="HAC", maxlags=2**63)
    with pytest.raises(ValueError, match=r"^ols\(\): 'groups' must have 20 values like 'y', not 19$"):
        R.ols(y, X, cov_type="cluster", groups=g[:19])
    with pytest.raises(ValueError, match=r"^ols\(\): 'groups' must have 20 values like 'y', not 19$"):
        R.ols(y, X, cov_type="cluster", groups=["a"] * 19)
    with pytest.raises(
        TypeError,
        match=r"^ols\(\): 'groups' must be an sd.ndarray, a sequence of labels or a frame "
        r"Series, not DataFrame$",
    ):
        R.ols(y, X, cov_type="cluster", groups=sd.frame.DataFrame({"g": g.tolist()}))
    with pytest.raises(TypeError, match=r"^ols\(\): 'groups' must be .*, not str$"):
        R.ols(y, X, cov_type="cluster", groups="abc")
    with pytest.raises(TypeError, match=r"^ols\(\): the labels in 'groups' must be strings or numbers, not list$"):
        R.ols(y, X, cov_type="cluster", groups=[[1], "a"] * 10)
    with pytest.raises(TypeError, match=r"^ols\(\): 'groups' must be real, not a complex128 array$"):
        R.ols(y, X, cov_type="cluster", groups=sd.asarray(g) * 1j)
    with pytest.raises(ValueError, match=r"^ols\(\): nan_policy='omit' takes one regression, not a batch"):
        R.ols(y, X, cov_type="cluster", groups=np.stack([g, g]), nan_policy="omit")
    with pytest.raises(TypeError, match=r"^breusch_pagan\(\): 'studentize' must be True or False, not int$"):
        R.ols(y, X).breusch_pagan(studentize=1)


# ---- summary (G13)


def test_summary():
    y, X = _problem(seed=21)
    df = sd.frame.DataFrame(
        {"price": X[:, 0].tolist(), "advertising_spend": X[:, 1].tolist(), "rate": X[:, 2].tolist()}
    )
    r = R.ols(y, df)
    text = r.summary()
    assert isinstance(text, str)
    lines = text.split("\n")
    assert lines[0].strip() == "OLS Regression Results"
    # 78 columns, wider when a parameter name needs it (as here)
    assert len(lines[1]) == max(78, len("advertising_spend") + 1 + 66)
    assert all(len(line) <= len(lines[1]) for line in lines)
    rules = [i for i, line in enumerate(lines) if set(line) == {"="}]
    assert len(rules) == 4
    # the top block: numbers formatted from the fields
    top = "\n".join(lines[rules[0] : rules[1]])
    for label, value in (
        ("No. Observations:", "60"),
        ("Df Residuals:", "56"),
        ("Df Model:", "3"),
        ("Covariance Type:", "nonrobust"),
        ("R-squared:", f"{r.rsquared:#.3f}"),
        ("Adj. R-squared:", f"{r.rsquared_adj:#.3f}"),
        ("F-statistic:", f"{r.fvalue:#.4g}"),
        ("Prob (F-statistic):", f"{r.f_pvalue:#.3g}"),
        ("Log-Likelihood:", f"{r.llf:#.5g}"),
        ("AIC:", f"{r.aic:#.4g}"),
        ("BIC:", f"{r.bic:#.4g}"),
        ("Model:", "OLS"),
        ("Method:", "Least Squares"),
    ):
        assert re.search(re.escape(label) + r"\s+" + re.escape(value) + r"(\s|$)", top), label
    # the coefficient table: one row per parameter, in param_names order
    head = lines[rules[1] + 1].split()
    assert head == ["coef", "std", "err", "t", "P>|t|", "[0.025", "0.975]"]
    rows = lines[rules[1] + 3 : rules[2]]
    assert [row.split()[0] for row in rows] == r.param_names
    ci = a(r.conf_int())
    for j, row in enumerate(rows):
        f = row.split()[1:]
        assert float(f[0]) == pytest.approx(float(r.params[j]), abs=5e-5)
        assert float(f[1]) == pytest.approx(float(r.bse[j]), abs=5e-4)
        assert float(f[2]) == pytest.approx(float(r.tvalues[j]), abs=5e-4)
        assert float(f[3]) == pytest.approx(float(r.pvalues[j]), abs=5e-4)
        assert float(f[4]) == pytest.approx(ci[j, 0], abs=5e-4) and float(f[5]) == pytest.approx(ci[j, 1], abs=5e-4)
    bottom = "\n".join(lines[rules[2] : rules[3]])
    jb = r.jarque_bera()
    for label, value in (
        ("Durbin-Watson:", f"{r.durbin_watson:#.3f}"),
        ("Jarque-Bera (JB):", f"{jb.statistic:#.3f}"),
        ("Prob(JB):", f"{jb.pvalue:#.3g}"),
        ("Cond. No.", f"{r.condition_number:#.3g}"),
    ):
        assert re.search(re.escape(label) + r"\s+" + re.escape(value) + r"(\s|$)", bottom), label
    flat = lambda t: " ".join(t.split())  # the notes wrap at the table's width
    assert "Notes:" in text and "correctly specified" in flat(text)
    # the covariance type is noted
    t = R.ols(y, X, cov_type="HAC", maxlags=4).summary()
    assert re.search(r"Covariance Type:\s+HAC", t) and "using 4 lags" in flat(t)
    t = R.wls(y, X, weights=np.ones(60), cov_type="cluster", groups=np.arange(60) % 9).summary()
    assert t.split("\n")[0].strip() == "WLS Regression Results"
    assert "with 9 groups" in flat(t) and "8 degrees of freedom" in flat(t)
    assert "heteroscedasticity robust (HC3)" in flat(R.ols(y, X, cov_type="HC3").summary())
    # repr stays short
    assert len(repr(r).split("\n")) <= 2


def test_gls_diagnostics_are_the_whitened_problems():
    # a diagonal sigma is WLS with weights 1/sigma_ii: the same whitened problem
    y, X = _problem(n=30, seed=22, ar=0.5)
    d = np.random.default_rng(23).uniform(0.5, 2.0, 30)
    g = np.arange(30) % 5
    gl, wl = R.gls(y, X, sigma=np.diag(d)), R.wls(y, X, weights=1 / d)
    for name in ("leverage", "resid_studentized_internal", "resid_studentized_external", "cooks_distance"):
        np.testing.assert_allclose(a(getattr(gl, name)), a(getattr(wl, name)), rtol=1e-12, atol=1e-15, err_msg=name)
    assert gl.durbin_watson == pytest.approx(wl.durbin_watson, rel=1e-13)
    assert gl.jarque_bera() == pytest.approx(wl.jarque_bera(), rel=1e-12)
    assert gl.breusch_pagan() == pytest.approx(wl.breusch_pagan(), rel=1e-12)
    for ct, kw in (("HAC", {"maxlags": 3}), ("cluster", {"groups": g})):
        np.testing.assert_allclose(
            a(R.gls(y, X, sigma=np.diag(d), cov_type=ct, **kw).bse),
            a(R.wls(y, X, weights=1 / d, cov_type=ct, **kw).bse),
            rtol=1e-12,
        )


def test_inference_df_and_summary_alpha():
    # df_resid_inference: df_resid, or G - 1 under cluster (G27); summary(alpha=) like statsmodels
    rng = np.random.default_rng(11)
    X = rng.standard_normal((60, 2))
    y = X @ [1.0, -1.0] + rng.standard_normal(60)
    g = rng.integers(0, 12, 60)
    r = sd.regression.ols(y, X)
    assert r.df_resid_inference == r.df_resid == 57.0
    c = sd.regression.ols(y, X, cov_type="cluster", groups=g)
    assert c.df_resid_inference == len(set(g.tolist())) - 1
    text = r.summary(alpha=0.1)
    assert "[0.050" in text and "0.950]" in text
    lo = float(r.conf_int(alpha=0.1)[1, 0])
    assert f"{lo:.3f}" in text
    with pytest.raises(ValueError, match=r"^summary\(\): 'alpha' must be between 0 and 1"):
        r.summary(alpha=1.5)


# ---- G108: one cluster and a constant y


def test_one_cluster_constant_y():
    r = R.ols(np.full(N, 2.0), X, cov_type="cluster", groups=np.zeros(N))
    assert np.all(np.isnan(a(r.cov_params()))) and np.all(np.isnan(a(r.bse)))
    r2 = R.ols(np.full(N, 2.0), X, cov_type="cluster", groups=np.arange(N) % 3)
    assert np.all(a(r2.bse) == 0.0)
    g = R.glm(np.full(N, 2.0), X, family="gaussian", cov_type="cluster", groups=np.zeros(N))
    assert np.all(np.isnan(a(g.bse)))


# ---- predictions, scales, messages


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
