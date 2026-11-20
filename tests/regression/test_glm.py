"""glm (docs/api/regression.md §3, docs/design/regression.md §3.4).

Accuracy against Arb (tests/regression/ref/glm.json, tests/regression/refdata_glm.py), then
statsmodels for every family and link, weights, trials, offset and exposure and the covariance
types; separation, non-convergence, aliased columns, zero weights, the gaussian identity
against ols, batches, threads, NaN policies and errors."""

import math
import warnings

import numpy as np
import pytest

import stardust as sd

from . import refdata, refdata_glm
from .smalldata import RNG, YB, YP, N, X

R = sd.regression
REFS = refdata_glm.load()
NAMES = list(REFS)

SM_FAMILY = {"gaussian": "Gaussian", "binomial": "Binomial", "poisson": "Poisson", "gamma": "Gamma"}
SM_LINK = {
    "identity": "Identity",
    "log": "Log",
    "logit": "Logit",
    "probit": "Probit",
    "cloglog": "CLogLog",
    "sqrt": "Sqrt",
    "inverse": "InversePower",
}
CANONICAL = {("gaussian", "identity"), ("binomial", "logit"), ("poisson", "log"), ("gamma", "inverse")}


def a(x):
    return np.asarray(x, dtype=float)


def norm_err(got, ref):
    hi = np.array([v[0] for v in ref])
    lo = np.array([v[1] for v in ref])
    return np.linalg.norm((a(got) - hi) - lo) / np.linalg.norm(hi + lo)


def max_rel(got, ref):
    return max(refdata.rel_err(g, r) for g, r in zip(a(got).ravel(), ref))


def sm_family(family, link):
    sm = pytest.importorskip("statsmodels.api")
    return getattr(sm.families, SM_FAMILY[family])(link=getattr(sm.families.links, SM_LINK[link])())


# ---- Arb references


@pytest.mark.parametrize("name", NAMES)
def test_against_arb(name):
    p, ref, meta = REFS[name], REFS[name]["ref"], REFS[name]["meta"]
    r = refdata_glm.fit(p)
    assert r.converged and not r.separation
    assert r.nobs == meta["nobs"] and r.df_resid == meta["df_resid"] and r.rank == meta["rank"]
    assert norm_err(r.params, ref["params"]) <= 1e-14
    assert max_rel(r.params, ref["params"]) <= 1e-13
    # standard errors from R of the final solve: a few u at these condition numbers
    for name_ in ("bse", "tvalues"):
        assert max_rel(getattr(r, name_), ref[name_]) <= 1e-13, name_
    assert max_rel(r.pvalues, ref["pvalues"]) <= 1e-12
    for s in ("deviance", "pearson_chi2", "scale", "llf", "null_deviance", "llnull", "aic", "bic"):
        assert refdata.rel_err(getattr(r, s), ref[s]) <= 1e-14, s
    mu = a(r.fittedvalues)
    assert (
        max(
            abs(mu[i] - refdata.value(ref["fittedvalues"][i])) / abs(refdata.value(ref["fittedvalues"][i]))
            for i in range(len(mu))
        )
        <= 1e-14
    )
    rp = a(r.resid_pearson)
    assert np.max(np.abs(rp - [refdata.value(v) for v in ref["resid_pearson"]])) <= 1e-14 * np.max(np.abs(rp))
    for ct in ("HC0", "HC1", "HC2", "HC3", "HAC"):
        rh = refdata_glm.fit(p, cov_type=ct)
        assert max_rel(rh.bse, ref["bse_" + ct]) <= 1e-13, ct
        assert np.array_equal(a(rh.params), a(r.params))
    rc = refdata_glm.fit(p, cov_type="cluster", groups=p["groups"])
    assert max_rel(rc.bse, ref["bse_cluster"]) <= 1e-13
    assert max_rel(rc.pvalues, ref["pvalues_cluster"]) <= 1e-12
    assert rc.df_resid_inference == meta["n_groups"] - 1


# ---- statsmodels


def _data(family, link, n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = np.array(refdata_glm._response(rng, family, link, X))
    return y, X


def _compare(r, s, rtol, *, fixed):
    """r against statsmodels' result s (fit with use_t matching r's distribution). statsmodels'
    covariance comes from the weights of its next-to-last iterate, so its standard errors
    differ by about the last step (1e-10 - 1e-8 here; glm's are checked against Arb)."""
    rcov = max(rtol, 1e-7)
    for name in ("params", "fittedvalues"):
        np.testing.assert_allclose(a(getattr(r, name)), np.asarray(getattr(s, name)), rtol=rtol, err_msg=name)
    # residuals near 0: statsmodels' y log(y/mu) - (y - mu) cancels, so absolute to the largest
    for name in ("resid_pearson", "resid_deviance", "resid_response"):
        got = a(getattr(r, name))
        np.testing.assert_allclose(
            got, np.asarray(getattr(s, name)), rtol=rtol, atol=rtol * np.max(np.abs(got)), err_msg=name
        )
    for name in ("bse", "tvalues"):
        np.testing.assert_allclose(a(getattr(r, name)), np.asarray(getattr(s, name)), rtol=rcov, err_msg=name)
    np.testing.assert_allclose(a(r.pvalues), np.asarray(s.pvalues), rtol=10 * rcov)  # |t|^2 times bse's
    for name in ("deviance", "null_deviance", "pearson_chi2", "llf", "aic", "scale", "df_resid", "df_model"):
        assert getattr(r, name) == pytest.approx(getattr(s, name), rel=rtol), name
    assert r.bic == pytest.approx(s.bic_llf, rel=rtol)
    np.testing.assert_allclose(a(r.conf_int(alpha=0.1)), np.asarray(s.conf_int(0.1)), rtol=rcov)
    np.testing.assert_allclose(a(r.cov_params()), np.asarray(s.cov_params()), rtol=rcov)
    assert r._normal() == fixed


@pytest.mark.parametrize("family,link", refdata_glm.LINKS)
def test_families_against_statsmodels(family, link):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _data(family, link)
    r = R.glm(y, X, family=family, link=link)
    fixed = family in ("binomial", "poisson")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = sm.GLM(y, sm.add_constant(X), family=sm_family(family, link)).fit(tol=1e-14, maxiter=1000, use_t=not fixed)
    # statsmodels stops on an absolute change of the deviance, which settles the coefficients
    # to about sqrt(u) for a non-canonical link (here measured to Arb: glm's are u-accurate)
    rtol = 1e-11 if (family, link) in CANONICAL else 1e-6
    _compare(r, s, rtol, fixed=fixed)
    assert r.llnull == pytest.approx(s.llnull, rel=rtol) or (family, link) == ("gaussian", "identity")
    # prediction on new rows
    Xn = np.random.default_rng(9).standard_normal((5, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels' inverse power link at its bounds
        pr = s.get_prediction(sm.add_constant(Xn, has_constant="add")).summary_frame(alpha=0.1)
    np.testing.assert_allclose(a(r.predict(Xn)), pr["mean"], rtol=rtol)
    np.testing.assert_allclose(
        a(r.predict(Xn, which="linear")), np.asarray(sm.add_constant(Xn) @ s.params), rtol=max(rtol, 1e-12)
    )
    # statsmodels maps the linear predictor's bounds without ordering them (a decreasing
    # inverse link gives lower > upper); here they are in order
    pi = a(r.predict_interval(Xn, alpha=0.1))
    sp = np.sort(np.c_[pr["mean_ci_lower"], pr["mean_ci_upper"]], axis=1)
    assert np.all(pi[:, 0] <= pi[:, 1])
    np.testing.assert_allclose(pi, sp, rtol=max(rtol, 1e-7))


def test_gaussian_identity_equals_ols():
    y, X = _data("gaussian", "identity", n=80, seed=3)
    r = R.glm(y, X, family="gaussian")
    o = R.ols(y, X)
    assert np.array_equal(a(r.params), a(o.params))
    assert np.array_equal(a(r.bse), a(o.bse))
    assert np.array_equal(a(r.fittedvalues), a(o.fittedvalues))
    assert r.scale == o.scale and r.deviance == o.ssr and r.pearson_chi2 == o.ssr
    assert r.llf == pytest.approx(o.llf, rel=1e-15) and r.aic == pytest.approx(o.aic, rel=1e-15)
    assert np.array_equal(a(r.cov_params()), a(o.cov_params()))
    w = np.random.default_rng(4).uniform(0.5, 2.0, len(y))
    rw = R.glm(y, X, family="gaussian", var_weights=w)
    ow = R.wls(y, X, weights=w)
    assert np.array_equal(a(rw.params), a(ow.params)) and np.array_equal(a(rw.bse), a(ow.bse))
    for ct in ("HC0", "HC1", "HC2", "HC3", "HAC"):
        np.testing.assert_allclose(
            a(R.glm(y, X, family="gaussian", cov_type=ct).bse), a(R.ols(y, X, cov_type=ct).bse), rtol=1e-14
        )


def test_var_weights_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    y, X = _data("poisson", "log", seed=4)
    w = np.random.default_rng(1).uniform(0.3, 3.0, len(y))
    r = R.glm(y, X, family="poisson", var_weights=w)
    s = sm.GLM(y, sm.add_constant(X), family=sm_family("poisson", "log"), var_weights=w).fit(tol=1e-14)
    _compare(r, s, 1e-11, fixed=True)
    y, X = _data("gamma", "log", seed=5)
    r = R.glm(y, X, family="gamma", link="log", var_weights=w)
    s = sm.GLM(y, sm.add_constant(X), family=sm_family("gamma", "log"), var_weights=w).fit(tol=1e-14, use_t=True)
    _compare(r, s, 1e-11, fixed=False)


def test_freq_weights_are_replication():
    sm = pytest.importorskip("statsmodels.api")
    y, X = _data("binomial", "logit", n=60, seed=6)
    f = np.random.default_rng(2).integers(1, 4, len(y)).astype(float)
    r = R.glm(y, X, family="binomial", freq_weights=f)
    rep = np.repeat(np.arange(len(y)), f.astype(int))
    rr = R.glm(y[rep], X[rep], family="binomial")
    np.testing.assert_allclose(a(r.params), a(rr.params), rtol=1e-13)
    np.testing.assert_allclose(a(r.bse), a(rr.bse), rtol=1e-13)
    for name in ("nobs", "df_resid", "deviance", "llf", "aic", "bic", "null_deviance", "pearson_chi2"):
        assert getattr(r, name) == pytest.approx(getattr(rr, name), rel=1e-13), name
    np.testing.assert_allclose(
        a(R.glm(y, X, family="binomial", freq_weights=f, cov_type="HC0").bse),
        a(R.glm(y[rep], X[rep], family="binomial", cov_type="HC0").bse),
        rtol=1e-12,
    )
    s = sm.GLM(y, sm.add_constant(X), family=sm_family("binomial", "logit"), freq_weights=f).fit(tol=1e-14)
    np.testing.assert_allclose(a(r.params), s.params, rtol=1e-11)
    np.testing.assert_allclose(a(r.bse), s.bse, rtol=1e-11)
    assert r.nobs == s.model.wnobs and s.nobs == len(y)  # statsmodels' nobs counts the rows
    assert r.df_resid == s.df_resid and r.llf == pytest.approx(s.llf, rel=1e-12)


def test_trials_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(7)
    X = rng.standard_normal((80, 2))
    tr = rng.integers(1, 15, 80).astype(float)
    succ = rng.binomial(tr.astype(int), 1 / (1 + np.exp(-(0.2 + X @ [0.6, -0.3])))).astype(float)
    for link in ("logit", "probit", "cloglog"):
        r = R.glm(succ / tr, X, family="binomial", link=link, trials=tr)
        s = sm.GLM(np.c_[succ, tr - succ], sm.add_constant(X), family=sm_family("binomial", link)).fit(tol=1e-14)
        rtol = 1e-11 if link == "logit" else 1e-6
        for name in ("params", "bse", "pvalues", "fittedvalues", "resid_pearson", "resid_deviance"):
            np.testing.assert_allclose(a(getattr(r, name)), np.asarray(getattr(s, name)), rtol=rtol, err_msg=name)
        for name in ("deviance", "null_deviance", "pearson_chi2", "llf", "llnull", "aic"):
            assert getattr(r, name) == pytest.approx(getattr(s, name), rel=rtol), name
        # statsmodels' resid_response counts successes; here it is on y's scale
        np.testing.assert_allclose(a(r.resid_response) * tr, s.resid_response, rtol=rtol, atol=1e-12)


def test_offset_and_exposure_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(8)
    X = rng.standard_normal((150, 2))
    expo = rng.uniform(0.5, 5.0, 150)
    off = 0.2 * rng.standard_normal(150)
    y = rng.poisson(expo * np.exp(off + 0.2 + X @ [0.4, -0.3])).astype(float)
    r = R.glm(y, X, family="poisson", offset=off, exposure=expo)
    s = sm.GLM(y, sm.add_constant(X), family=sm_family("poisson", "log"), offset=off, exposure=expo).fit(tol=1e-14)
    _compare(r, s, 1e-10, fixed=True)
    assert r.llnull == pytest.approx(s.llnull, rel=1e-10)
    np.testing.assert_allclose(a(r.linear_predictor), np.asarray(s.predict(which="linear")), rtol=1e-12)
    Xn = rng.standard_normal((4, 2))
    e2, o2 = rng.uniform(1, 2, 4), rng.standard_normal(4) * 0.1
    np.testing.assert_allclose(
        a(r.predict(Xn, offset=o2, exposure=e2)), s.predict(sm.add_constant(Xn), offset=o2, exposure=e2), rtol=1e-12
    )
    # exposure is log(exposure) in the offset
    r2 = R.glm(y, X, family="poisson", offset=off + np.log(expo))
    np.testing.assert_allclose(a(r2.params), a(r.params), rtol=1e-14)


@pytest.mark.parametrize(
    "family,link",
    [("binomial", "logit"), ("binomial", "probit"), ("poisson", "log"), ("gamma", "log"), ("gaussian", "log")],
)
def test_robust_against_statsmodels(family, link):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _data(family, link, seed=10)
    Xc = sm.add_constant(X)
    fam = sm_family(family, link)
    rtol = 1e-10 if (family, link) in CANONICAL else 1e-6
    # HC0: statsmodels' sandwich with the observed information as bread. G39: the normal for
    # binomial and poisson (statsmodels' use_t=False), t(df_resid) for gaussian and gamma
    fixed = family in ("binomial", "poisson")
    r = R.glm(y, X, family=family, link=link, cov_type="HC0")
    s = sm.GLM(y, Xc, family=fam).fit(cov_type="HC0", use_t=not fixed, tol=1e-14)
    for name in ("bse", "pvalues"):
        np.testing.assert_allclose(a(getattr(r, name)), np.asarray(getattr(s, name)), rtol=rtol, err_msg=name)
    np.testing.assert_allclose(a(r.conf_int(alpha=0.1)), np.asarray(s.conf_int(0.1)), rtol=max(rtol, 1e-10))
    assert r.df_resid_inference == (math.inf if fixed else r.df_resid)
    # statsmodels' GLM gives HC0 for HC1-HC3 too; HC1 here is HC0 x sqrt(n/(n - k)) as ols's
    r1 = R.glm(y, X, family=family, link=link, cov_type="HC1")
    np.testing.assert_allclose(a(r1.bse), a(r.bse) * math.sqrt(r.nobs / r.df_resid), rtol=1e-13)
    r = R.glm(y, X, family=family, link=link, cov_type="HAC", maxlags=3)
    s = sm.GLM(y, Xc, family=fam).fit(cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=not fixed, tol=1e-14)
    np.testing.assert_allclose(a(r.bse), np.asarray(s.bse), rtol=rtol)
    np.testing.assert_allclose(a(r.pvalues), np.asarray(s.pvalues), rtol=rtol)
    g = np.random.default_rng(3).integers(0, 12, len(y))
    r = R.glm(y, X, family=family, link=link, cov_type="cluster", groups=g)
    s = sm.GLM(y, Xc, family=fam).fit(cov_type="cluster", cov_kwds={"groups": g}, use_t=True, tol=1e-14)
    np.testing.assert_allclose(a(r.bse), np.asarray(s.bse), rtol=rtol)
    np.testing.assert_allclose(a(r.pvalues), np.asarray(s.pvalues), rtol=rtol)
    assert r.df_resid_inference == 11


def test_inference_distribution():
    # G38: z for a fixed scale (binomial, poisson, or a given number), t(df_resid) for an
    # estimated one; G39: robust HC and HAC keep z for binomial and poisson, t for gaussian and
    # gamma and for an estimated scale; cluster t(G - 1) (G27)
    y, X = _data("poisson", "log", seed=12)
    r = R.glm(y, X, family="poisson")
    assert r.df_resid_inference == math.inf and "P>|z|" in r.summary()
    t = a(r.tvalues)
    np.testing.assert_allclose(a(r.pvalues), 2 * a(sd.stats.Normal().ccdf(sd.asarray(np.abs(t)))), rtol=1e-15)
    rq = R.glm(y, X, family="poisson", scale="X2")
    assert rq.df_resid_inference == rq.df_resid and "P>|t|" in rq.summary()
    assert rq.scale == pytest.approx(rq.pearson_chi2 / rq.df_resid, rel=1e-15)
    np.testing.assert_allclose(a(rq.bse), a(r.bse) * math.sqrt(rq.scale), rtol=1e-14)
    rd = R.glm(y, X, family="poisson", scale="dev")
    assert rd.scale == pytest.approx(rd.deviance / rd.df_resid, rel=1e-15)
    yg, Xg = _data("gamma", "log", seed=13)
    rf = R.glm(yg, Xg, family="gamma", link="log", scale=2.0)
    assert rf.scale == 2.0 and rf.df_resid_inference == math.inf
    rg = R.glm(yg, Xg, family="gamma", link="log")
    assert rg.df_resid_inference == rg.df_resid
    for ct, kw in (("HC0", {}), ("HC3", {}), ("HAC", {"maxlags": 2})):
        rh = R.glm(y, X, family="poisson", cov_type=ct, **kw)
        assert rh.df_resid_inference == math.inf and "P>|z|" in rh.summary(), ct
        np.testing.assert_allclose(
            a(rh.pvalues), 2 * a(sd.stats.Normal().ccdf(sd.asarray(np.abs(a(rh.tvalues))))), rtol=1e-15
        )
    rqh = R.glm(y, X, family="poisson", scale="X2", cov_type="HC0")
    assert rqh.df_resid_inference == rqh.df_resid and "P>|t|" in rqh.summary()
    rgh = R.glm(yg, Xg, family="gamma", link="log", cov_type="HC1")
    assert rgh.df_resid_inference == rgh.df_resid and "P>|t|" in rgh.summary()
    g = np.arange(len(y)) % 7
    rc = R.glm(y, X, family="poisson", cov_type="cluster", groups=g)
    assert rc.df_resid_inference == 6 and "P>|t|" in rc.summary()


def test_llr_and_null_model():
    sm = pytest.importorskip("statsmodels.api")
    y, X = _data("binomial", "logit", seed=14)
    r = R.glm(y, X, family="binomial")
    s = sm.GLM(y, sm.add_constant(X), family=sm_family("binomial", "logit")).fit(tol=1e-14)
    assert r.llr == pytest.approx(2 * (s.llf - s.llnull), rel=1e-11)
    assert r.llr_pvalue == pytest.approx(float(sd.stats.ChiSquared(df=2.0).ccdf(r.llr)), rel=1e-15)
    assert r.llr == pytest.approx(r.null_deviance - r.deviance, rel=1e-12)
    # no intercept: the null model is eta = offset (R's glm), with no parameter
    rn = R.glm(y, X, family="binomial", intercept=False)
    assert rn.null_deviance == pytest.approx(2 * len(y) * math.log(2), rel=1e-15)
    assert rn.df_model == 2


# ---- separation, convergence, edge cases


def test_complete_separation():
    x = np.arange(10.0)
    y = (x > 4.5).astype(float)
    for link in ("logit", "probit", "cloglog"):
        r = R.glm(y, x, family="binomial", link=link)
        assert r.separation and r.converged
        assert np.all(np.isnan(a(r.params))) and np.all(np.isnan(a(r.bse))) and np.all(np.isnan(a(r.pvalues)))
        assert np.array_equal(a(r.fittedvalues), y)
        assert r.deviance == 0.0 and r.llf == 0.0
        assert "Separation" in r.summary()


def test_quasi_complete_separation():
    # rows at x = 0 have both outcomes: the limit fits them by 1/2, the others at the boundary
    x = np.array([-3, -2, -1, 0, 0, 1, 2, 3.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    r = R.glm(y, x, family="binomial")
    assert r.separation
    np.testing.assert_allclose(a(r.fittedvalues), [0, 0, 0, 0.5, 0.5, 1, 1, 1], atol=1e-14)
    assert r.deviance == pytest.approx(4 * math.log(2), rel=1e-14)
    # a dummy whose rows all have y = 1: the rest is the fit without them
    rng = np.random.default_rng(3)
    n = 40
    x1 = rng.standard_normal(n)
    d = (np.arange(n) < 8).astype(float)
    y = (rng.uniform(size=n) < 0.5).astype(float)
    y[d == 1] = 1
    r = R.glm(y, np.c_[x1, d], family="binomial")
    rest = R.glm(y[8:], x1[8:], family="binomial")
    assert r.separation and np.all(np.isnan(a(r.params)))
    np.testing.assert_allclose(a(r.fittedvalues)[8:], a(rest.fittedvalues), rtol=1e-12)
    assert np.all(a(r.fittedvalues)[:8] == 1.0)
    assert r.deviance == pytest.approx(rest.deviance, rel=1e-13) and r.llf == pytest.approx(rest.llf, rel=1e-13)


def test_poisson_boundary():
    # all-zero counts: the intercept goes to -inf (mu = 0)
    r = R.glm(np.zeros(10), np.arange(10.0), family="poisson")
    assert r.separation and np.all(a(r.fittedvalues) == 0) and r.deviance == 0 and r.llf == 0
    # a group with only zeros
    rng = np.random.default_rng(4)
    g = (np.arange(30) < 6).astype(float)
    x = rng.standard_normal(30)
    y = rng.poisson(2, 30).astype(float)
    y[g == 1] = 0
    r = R.glm(y, np.c_[x, g], family="poisson")
    rest = R.glm(y[6:], x[6:], family="poisson")
    assert r.separation
    np.testing.assert_allclose(a(r.fittedvalues)[6:], a(rest.fittedvalues), rtol=1e-12)
    assert r.deviance == pytest.approx(rest.deviance, rel=1e-13)


def test_no_false_separation():
    # strong but overlapping data: a finite MLE with fitted values numerically 0 and 1
    x = np.linspace(-3, 3, 50)
    y = (x > 0).astype(float)
    y[24], y[26] = 1, 0
    r = R.glm(y, x, family="binomial")
    assert not r.separation and r.converged and np.all(np.isfinite(a(r.params)))
    # an identity link's boundary MLE (mu = 0 at a zero count) is not separation
    yy = np.r_[np.zeros(5), np.random.default_rng(5).poisson(3, 20)].astype(float)
    xx = np.r_[np.linspace(0, 1, 5), np.linspace(1, 3, 20)]
    rr = R.glm(yy, xx, family="poisson", link="identity")
    assert not rr.separation and np.all(np.isfinite(a(rr.params)))


def test_max_iter():
    y, X = _data("binomial", "probit", seed=15)
    full = R.glm(y, X, family="binomial", link="probit")
    r = R.glm(y, X, family="binomial", link="probit", max_iter=2)
    assert not r.converged and r.n_iter == 2 and np.all(np.isfinite(a(r.params)))
    assert not np.array_equal(a(r.params), a(full.params))
    assert full.converged and full.n_iter < 100
    assert "did not converge" in r.summary()
    r1 = R.glm(y, X, family="binomial", link="probit", max_iter=1)
    assert r1.n_iter == 1 and not r1.converged


def test_max_iter_is_not_separation():
    """Stopping early never certifies separation (G33 decides it, not max_iter) and n_iter
    never exceeds max_iter (G32). Before, the next Fisher step of an early iterate failed the
    rows' test, and the direction test compared its projection, rounding noise when the
    other rows leave no null space, to that noise's own size: the hours/passed data with
    max_iter=2 gave separation=True, n_iter 4 and a deviance (15.57) below the MLE's."""
    hours = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5]
    passed = [0.0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    full = R.glm(passed, hours, family="binomial")
    for m in range(1, 8):
        r = R.glm(passed, hours, family="binomial", max_iter=m)
        assert not r.separation and r.n_iter <= m and np.all(np.isfinite(a(r.params)))
        assert r.deviance >= full.deviance * (1 - 1e-14)
        assert r.converged or r.n_iter == m
    rng = np.random.default_rng(3)
    for _ in range(200):
        X = rng.standard_normal((30, 2))
        eta = 0.5 * rng.standard_normal() + X @ (rng.standard_normal(2) * 2.0)
        y = (rng.random(30) < 1 / (1 + np.exp(-eta))).astype(float)
        full = R.glm(y, X, family="binomial")
        for m in range(1, 8):
            r = R.glm(y, X, family="binomial", max_iter=m)
            assert r.n_iter <= m
            if not full.separation:
                assert not r.separation and np.all(np.isfinite(a(r.params)))
                assert r.deviance >= full.deviance * (1 - 1e-12)
            if not r.converged:
                assert r.n_iter == m


def test_aliased_columns():
    y, X = _data("poisson", "log", seed=16)
    X3 = np.c_[X, X[:, 0] + X[:, 1]]
    r = R.glm(y, X3, family="poisson")
    r2 = R.glm(y, X, family="poisson")
    assert r.rank == 3 and np.isnan(a(r.params)[3]) and np.isnan(a(r.bse)[3])
    np.testing.assert_allclose(a(r.params)[:3], a(r2.params), rtol=1e-13)
    np.testing.assert_allclose(a(r.bse)[:3], a(r2.bse), rtol=1e-12)
    assert r.df_resid == r2.df_resid and r.aic == pytest.approx(r2.aic, rel=1e-14)
    np.testing.assert_allclose(a(r.predict(X3[:4])), a(r2.predict(X[:4])), rtol=1e-13)
    assert "Aliased" in r.summary()


def test_zero_weights_out_of_the_fit():
    y, X = _data("gamma", "log", n=50, seed=17)
    w = np.random.default_rng(1).uniform(0.5, 2, 50)
    w[[3, 20]] = 0.0
    keep = w > 0
    r = R.glm(y, X, family="gamma", link="log", var_weights=w)
    s = R.glm(y[keep], X[keep], family="gamma", link="log", var_weights=w[keep])
    assert r.nobs == 48 and r.df_resid == 45
    for name in ("params", "bse"):
        np.testing.assert_allclose(a(getattr(r, name)), a(getattr(s, name)), rtol=1e-14)
    for name in ("deviance", "llf", "pearson_chi2", "scale", "null_deviance"):
        assert getattr(r, name) == pytest.approx(getattr(s, name), rel=1e-14)
    # the zero rows still get fitted values
    np.testing.assert_allclose(a(r.fittedvalues)[[3, 20]], a(r.predict(X[[3, 20]])), rtol=1e-14)
    f = np.ones(50)
    f[[3, 20]] = 0
    rf = R.glm(y, X, family="gamma", link="log", var_weights=np.where(keep, w, 1.0), freq_weights=f)
    np.testing.assert_allclose(a(rf.params), a(s.params), rtol=1e-14)


def test_batch_equals_loop():
    rng = np.random.default_rng(18)
    B, n = 6, 40
    X = rng.standard_normal((B, n, 2))
    Y = (rng.uniform(size=(B, n)) < 1 / (1 + np.exp(-X[..., 0]))).astype(float)
    Y[2, :] = X[2, :, 0] > 0  # one separated regression in the batch
    tr = rng.integers(1, 5, (B, n)).astype(float)
    for cov in ("nonrobust", "HC0"):
        rb = R.glm(Y, X, family="binomial", cov_type=cov, freq_weights=tr)
        assert rb.params.shape == (B, 3) and rb.separation.shape == (B,)
        for b in range(B):
            r1 = R.glm(Y[b], X[b], family="binomial", cov_type=cov, freq_weights=tr[b])
            for name in ("params", "bse", "fittedvalues", "linear_predictor", "resid_deviance"):
                assert np.array_equal(a(getattr(rb, name))[b], a(getattr(r1, name)), equal_nan=True), name
            for name in ("deviance", "llf", "null_deviance", "pearson_chi2", "nobs"):
                assert float(getattr(rb, name)[b]) == getattr(r1, name) or math.isnan(getattr(r1, name)), name
            assert int(rb.n_iter[b]) == r1.n_iter and bool(rb.separation[b]) == r1.separation
    # one X for many y
    rb = R.glm(Y, X[0], family="binomial")
    r1 = R.glm(Y[4], X[0], family="binomial")
    assert np.array_equal(a(rb.params)[4], a(r1.params))


def test_threads_bit_identical():
    rng = np.random.default_rng(19)
    n = 150_000
    X = rng.standard_normal((n, 4))
    y = rng.poisson(np.exp(0.2 + X @ [0.3, -0.2, 0.1, 0.05])).astype(float)
    g = rng.integers(0, 50, n)
    old = sd.get_num_threads()
    try:
        out = []
        for th in (1, 10):
            sd.set_num_threads(th)
            r = R.glm(y, X, family="poisson", link="sqrt", cov_type="cluster", groups=g)
            out.append((a(r.params), a(r.bse), r.deviance, r.llf, a(r.fittedvalues), r.n_iter))
        assert all(np.array_equal(u, v) if isinstance(u, np.ndarray) else u == v for u, v in zip(*out))
    finally:
        sd.set_num_threads(old)


def test_nan_policies():
    y, X = _data("poisson", "log", n=40, seed=20)
    y[5] = np.nan
    r = R.glm(y, X, family="poisson")
    assert np.all(np.isnan(a(r.params))) and r.rank == 0 and math.isnan(r.nobs) and not r.converged and r.n_iter == 0
    ro = R.glm(y, X, family="poisson", nan_policy="omit")
    keep = ~np.isnan(y)
    rk = R.glm(y[keep], X[keep], family="poisson")
    assert np.array_equal(a(ro.params), a(rk.params)) and ro.nobs == 39
    with pytest.raises(ValueError, match=r"glm\(\): the input contains NaN \(nan_policy='raise'\)"):
        R.glm(y, X, family="poisson", nan_policy="raise")
    w = np.ones(40)
    w[7] = np.nan
    ro = R.glm(np.nan_to_num(y), X, family="poisson", var_weights=w, nan_policy="omit")
    assert ro.nobs == 39
    with pytest.raises(ValueError, match="takes one regression"):
        R.glm(np.stack([y, y]), X, family="poisson", nan_policy="omit")


def test_frame_inputs_and_names():
    df = sd.frame.DataFrame({"a": [0.1, 0.5, -0.3, 1.2, 0.7, -1.0, 0.2, 0.9], "b": [1.0, 0, 1, 0, 1, 1, 0, 0]})
    y = sd.frame.Series([0.0, 1, 0, 1, 1, 0, 1, 0], name="y")
    r = R.glm(y, df, family="binomial")
    assert r.param_names == ["const", "a", "b"]
    assert "a  " in r.summary()


def test_errors():
    y, X = _data("binomial", "logit", n=20, seed=21)
    with pytest.raises(
        ValueError, match=r"glm\(\): 'family' must be 'gaussian', 'binomial', 'poisson' or 'gamma', not 'x'"
    ):
        R.glm(y, X, family="x")
    with pytest.raises(
        ValueError,
        match=r"'link' must be None, 'logit', 'probit', 'cloglog' or 'log' for the binomial family, not 'identity'",
    ):
        R.glm(y, X, family="binomial", link="identity")
    with pytest.raises(TypeError, match="family"):
        R.glm(y, X, family=None)
    with pytest.raises(ValueError, match=r"'y' must be 0 or 1 for the binomial family without 'trials'"):
        R.glm(y * 0.5, X, family="binomial")
    with pytest.raises(ValueError, match=r"'y' must be in \[0, 1\]"):
        R.glm(y * 2, X, family="binomial", trials=np.ones(20))
    with pytest.raises(ValueError, match=r"'y' must be >= 0 for the poisson family"):
        R.glm(y - 1, X, family="poisson")
    with pytest.raises(ValueError, match=r"'y' must be > 0 for the gamma family"):
        R.glm(y, X, family="gamma")
    with pytest.raises(ValueError, match=r"'trials' is only used with the binomial family"):
        R.glm(y, X, family="poisson", trials=np.ones(20))
    with pytest.raises(ValueError, match=r"'trials' must be finite and > 0"):
        R.glm(y, X, family="binomial", trials=np.zeros(20))
    with pytest.raises(ValueError, match=r"'exposure' is only used with the log link"):
        R.glm(y, X, family="binomial", exposure=np.ones(20))
    with pytest.raises(ValueError, match=r"'exposure' must be > 0"):
        R.glm(y, X, family="poisson", exposure=np.zeros(20))
    with pytest.raises(ValueError, match=r"'var_weights' must be >= 0"):
        R.glm(y, X, family="binomial", var_weights=-np.ones(20))
    with pytest.raises(ValueError, match=r"'freq_weights' must be >= 0"):
        R.glm(y, X, family="binomial", freq_weights=-np.ones(20))
    with pytest.raises(ValueError, match=r"'var_weights' must have 20 values like 'y', not 19"):
        R.glm(y, X, family="binomial", var_weights=np.ones(19))
    Xi = X.copy()
    Xi[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"glm\(\): 'X' contains infinity"):
        R.glm(y, Xi, family="binomial")
    with pytest.raises(ValueError, match=r"'offset' contains infinity"):
        R.glm(y, X, family="binomial", offset=np.r_[np.inf, np.zeros(19)])
    with pytest.raises(ValueError, match=r"'scale' must be None, 'X2', 'dev' or a number > 0, not 'x'"):
        R.glm(y, X, family="binomial", scale="x")
    with pytest.raises(ValueError, match=r"'scale' must be a finite number > 0"):
        R.glm(y, X, family="binomial", scale=-1.0)
    with pytest.raises(ValueError, match=r"'max_iter' must be >= 1"):
        R.glm(y, X, family="binomial", max_iter=0)
    with pytest.raises(TypeError, match=r"'max_iter' must be an int"):
        R.glm(y, X, family="binomial", max_iter=1.5)
    with pytest.raises(ValueError, match=r"'tol' must be a finite number > 0"):
        R.glm(y, X, family="binomial", tol=0.0)
    with pytest.raises(ValueError, match=r"'groups' is required when cov_type='cluster'"):
        R.glm(y, X, family="binomial", cov_type="cluster")
    with pytest.raises(ValueError, match=r"'maxlags' is only used when cov_type='HAC'"):
        R.glm(y, X, family="binomial", maxlags=2)
    with pytest.raises(ValueError, match="cannot find valid starting values"):
        R.glm(-np.abs(y) - 1, X, family="gaussian", link="log")
    r = R.glm(y, X, family="binomial")
    with pytest.raises(ValueError, match=r"predict\(\): 'X_new' must have 2 columns"):
        r.predict(np.ones((3, 3)))
    with pytest.raises(ValueError, match=r"'which' must be 'mean' or 'linear'"):
        r.predict(X, which="x")
    with pytest.raises(ValueError, match="summary"):
        R.glm(np.stack([y, y]), X, family="binomial").summary()


def test_summary_layout():
    y, X = _data("binomial", "logit", seed=22)
    s = R.glm(y, X, family="binomial").summary()
    lines = s.splitlines()
    assert lines[0].strip() == "Generalized Linear Model Regression Results"
    for label in (
        "Model Family:",
        "Binomial",
        "Link Function:",
        "Logit",
        "Method:",
        "IRLS",
        "No. Iterations:",
        "Deviance:",
        "Pearson chi2:",
        "Pseudo R-squ. (CS):",
        "Scale:",
        "Log-Likelihood:",
    ):
        assert label in s
    assert "[0.025" in s and "0.975]" in s
    s90 = R.glm(y, X, family="binomial").summary(alpha=0.1)
    assert "[0.050" in s90 and "0.950]" in s90


def _kernel(y, X, fam, link, chol, rtol=-1.0):
    n = len(y)
    aux = np.r_[np.ones((3, n)), np.zeros((1, n))]
    raw = sd._core.regression.glm_fit(
        "glm",
        sd.asarray(y),
        sd.asarray(X),
        sd.asarray(aux),
        None,
        0,
        fam,
        link,
        False,
        False,
        False,
        False,
        True,
        rtol,
        0,
        -1,
        100,
        1e-8,
        0,
        0.0,
        0,
        3,
        chol,
    )
    st = a(raw[6])
    return a(raw[0]), np.sqrt(np.diag(a(raw[1])) * st[5]), st


@pytest.mark.parametrize(
    "fam,link,family,lname",
    [(1, 2, "binomial", "logit"), (1, 3, "binomial", "probit"), (2, 1, "poisson", "log"), (3, 1, "gamma", "log")],
)
@pytest.mark.parametrize("kappa", [1.0, 1e5, 1e9])
def test_inner_normal_equations(fam, link, family, lname, kappa):
    # G40: the iterations before the deviance's convergence by Cholesky (QR when it is too
    # ill-conditioned: kappa(X^T W X) u > 1e-4, here kappa = 1e9) give the QR path's answer
    rng = np.random.default_rng(int(kappa) % 97 + fam)
    n, k = 300, 4
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    s = kappa ** (-np.arange(k) / (k - 1))
    X = (U * s) @ V.T * np.sqrt(n)
    beta = V @ (rng.standard_normal(k) * 0.3 / s)
    y = np.array(refdata_glm._response(rng, family, lname, np.c_[X[:, 0], X[:, 1]]))
    y = {
        "binomial": (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ beta)))).astype(float),
        "poisson": rng.poisson(np.exp(0.3 + X @ beta)).astype(float),
        "gamma": rng.gamma(3.0, np.exp(0.3 + X @ beta) / 3.0),
    }[family]
    pc, sc, stc = _kernel(y, X, fam, link, True)
    pq, sq, stq = _kernel(y, X, fam, link, False)
    # both are about kappa u from the MLE (Arb: probit at kappa 1e5 3e-14 and 7e-14), and their
    # standard errors, from R, about kappa u
    assert np.linalg.norm(pc - pq) / np.linalg.norm(pq) <= max(1e-14, 1e-2 * kappa * 2.0**-52)
    np.testing.assert_allclose(sc, sq, rtol=max(1e-12, kappa * 2.0**-52))
    assert stc[9] == 1 and stq[9] == 1 and abs(stc[8] - stq[8]) <= 1


def test_inner_normal_equations_aliasing():
    # a column G21 would drop never enters the normal equations: with a coarse rtol (R's 1e-7)
    # a nearly dependent column is aliased exactly as by the QR path
    rng = np.random.default_rng(23)
    X = rng.standard_normal((200, 3))
    X[:, 2] = X[:, 0] + X[:, 1] + 1e-9 * rng.standard_normal(200)
    y = rng.poisson(np.exp(0.2 + 0.3 * X[:, 0])).astype(float)
    pc, _, stc = _kernel(y, X, 2, 1, True, rtol=1e-7)
    pq, _, stq = _kernel(y, X, 2, 1, False, rtol=1e-7)
    assert np.isnan(pc[3]) and np.isnan(pq[3]) and stc[1] == stq[1] == 3
    np.testing.assert_allclose(pc[:3], pq[:3], rtol=1e-13)


@pytest.mark.parametrize(
    "family,link", [("binomial", "logit"), ("binomial", "probit"), ("poisson", "log"), ("gamma", "log")]
)
@pytest.mark.deep
@pytest.mark.parametrize("kappa", [1.0, 1e4, 1e8])
def test_inner_normal_equations_against_arb(family, link, kappa):
    # G40 on problems large enough for the normal equations (n k = 1200): the coefficients stay
    # u-accurate against Arb
    import sys

    sys.path.insert(0, str(refdata_glm.REF.parents[3] / "tools"))
    import regref

    rng = np.random.default_rng(int(np.log10(kappa)) + 7)
    n, k = 200, 6
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    s = kappa ** (-np.arange(k) / (k - 1))
    X = (U * s) @ V.T * np.sqrt(n)
    c = rng.standard_normal(k)
    c *= 0.8 / np.linalg.norm(U @ c)
    eta = X @ (V @ (c / s))
    y = {
        "binomial": (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float),
        "poisson": rng.poisson(np.exp(0.5 + eta)).astype(float),
        "gamma": rng.gamma(3.0, np.exp(0.5 + eta) / 3.0),
    }[family]
    r = R.glm(y, X, family=family, link=link, intercept=False)
    ref, _ = regref.glm(
        y.tolist(), X.tolist(), family=family, link=link, intercept=False, beta0=[float(v) for v in r.params.tolist()]
    )
    # canonical links u; non-canonical ones (Newton's polishing and final step through the thin
    # Q, G42) a few u
    canonical = (family, link) in CANONICAL
    assert norm_err(r.params, ref["params"]) <= (1e-15 if canonical else 5e-15)
    assert max_rel(r.bse, ref["bse"]) <= max(1e-13, 10 * kappa * 2.0**-52)


NONCANONICAL = [
    ("binomial", "probit", 0.3, 1.0),
    ("binomial", "cloglog", -0.5, 1.0),
    ("binomial", "log", -1.5, 0.4),
    ("poisson", "identity", 5.0, 2.0),
    ("poisson", "sqrt", 2.0, 0.5),
    ("gamma", "log", 0.5, 1.0),
    ("gamma", "identity", 4.0, 1.5),
    ("gaussian", "log", 0.8, 0.5),
]


@pytest.mark.deep
@pytest.mark.parametrize("family,link,base,scale", NONCANONICAL)
@pytest.mark.parametrize("kappa", [1e4, 1e6, 1e8, 1e10])
def test_noncanonical_against_arb(family, link, base, scale, kappa):
    # G42: Newton's steps through the thin Q in the polishing iterations and the final solve
    # (docs/design/regression.md §3.4): the canonical links' level up to kappa 1e8, about 10 u at
    # 1e10 (the canonical links there: 1e-15)
    import sys

    sys.path.insert(0, str(refdata_glm.REF.parents[3] / "tools"))
    import regref
    from scipy.stats import norm

    rng = np.random.default_rng(3)
    n, k = 200, 5
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    s = kappa ** (-np.arange(k) / (k - 1))
    X = (U * s) @ V.T * np.sqrt(n)
    c = rng.standard_normal(k)
    eta = base + U @ c * (scale / np.linalg.norm(U @ c)) * np.sqrt(n) / 4
    mu = {
        "probit": norm.cdf,
        "cloglog": lambda e: -np.expm1(-np.exp(e)),
        "log": np.exp,
        "identity": lambda e: e,
        "sqrt": lambda e: e * e,
    }[link](eta)
    y = {
        "binomial": lambda: (rng.uniform(size=n) < mu).astype(float),
        "poisson": lambda: rng.poisson(mu).astype(float),
        "gamma": lambda: rng.gamma(3.0, mu / 3.0),
        "gaussian": lambda: mu + 0.3 * rng.standard_normal(n),
    }[family]()
    r = R.glm(y, X, family=family, link=link)
    assert r.converged
    ref, _ = regref.glm(y.tolist(), X.tolist(), family=family, link=link, beta0=[float(v) for v in r.params.tolist()])
    assert norm_err(r.params, ref["params"]) <= (1e-14 if kappa <= 1e8 else 1e-13)


def test_slow_fisher_switches_to_newton():
    # G44: Fisher scoring with rate ~0.92 along a flat direction (gamma log, kappa 1e6): three
    # steps in a row not below half the one before switch to Newton's steps; it converges in
    # a few iterations at the Arb level. With Fisher's steps throughout it does not converge
    # in 100 iterations (statsmodels needs more than 500).
    import sys

    sys.path.insert(0, str(refdata_glm.REF.parents[3] / "tools"))
    import regref

    rng = np.random.default_rng(13)
    n, k = 200, 6
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    s = 1e6 ** (-np.arange(k) / (k - 1))
    X = (U * s) @ V.T * np.sqrt(n)
    c = rng.standard_normal(k)
    c *= 0.8 / np.linalg.norm(U @ c)
    y = rng.gamma(3.0, np.exp(0.5 + X @ (V @ (c / s))) / 3.0)
    r = R.glm(y, X, family="gamma", link="log", intercept=False)
    assert r.converged and r.n_iter <= 15
    ref, _ = regref.glm(
        y.tolist(), X.tolist(), family="gamma", link="log", intercept=False, beta0=[float(v) for v in r.params.tolist()]
    )
    assert norm_err(r.params, ref["params"]) <= 1e-15
    # without the switch the deviance flattens before the coefficients settle: Fisher's steps
    # throughout meet the deviance's criterion far from the MLE (kernel newton=0), and
    # Newton's polishing after it (newton=1) gets there in many more iterations
    aux = np.r_[np.ones((3, n)), np.zeros((1, n))]
    raws = [
        sd._core.regression.glm_fit(
            "glm",
            sd.asarray(y),
            sd.asarray(X),
            sd.asarray(aux),
            None,
            0,
            3,
            1,
            False,
            False,
            False,
            False,
            False,
            -1.0,
            0,
            -1,
            100,
            1e-8,
            0,
            0.0,
            0,
            3,
            True,
            mode,
        )
        for mode in (0, 1)
    ]
    assert norm_err(raws[0][0], ref["params"]) > 1e-8
    assert a(raws[1][6])[8] >= 2 * r.n_iter


@pytest.mark.parametrize(
    "family,link", [("binomial", "probit"), ("binomial", "cloglog"), ("gamma", "log"), ("poisson", "identity")]
)
def test_slow_switch_not_triggered_on_fast_fits(family, link):
    # the switch never fires where Fisher scoring converges at a usual rate: the fits with
    # and without it (the kernel's newton=1) are the same bits, iterations included
    y, X = _data(family, link, n=300, seed=31)
    aux = np.r_[np.ones((3, len(y))), np.zeros((1, len(y)))]
    fam = {"binomial": 1, "poisson": 2, "gamma": 3}[family]
    lk = {"probit": 3, "cloglog": 4, "log": 1, "identity": 0}[link]
    out = [
        sd._core.regression.glm_fit(
            "glm",
            sd.asarray(y),
            sd.asarray(X),
            sd.asarray(aux),
            None,
            0,
            fam,
            lk,
            False,
            False,
            False,
            False,
            True,
            -1.0,
            0,
            -1,
            100,
            1e-8,
            0,
            0.0,
            0,
            3,
            True,
            mode,
        )
        for mode in (1, 2)
    ]
    assert np.array_equal(a(out[0][0]), a(out[1][0])) and np.array_equal(a(out[0][6]), a(out[1][6]), equal_nan=True)


def test_noncanonical_batch_and_threads_bit_identical():
    # Newton's steps through the thin Q (G42, G44): batch == loop, 1 and 10 threads alike
    rng = np.random.default_rng(41)
    B, n = 4, 3000
    X = rng.standard_normal((B, n, 3))
    Y = (rng.uniform(size=(B, n)) < 1 / (1 + np.exp(-(0.2 + X @ [0.5, -0.3, 0.2])))).astype(float)
    old = sd.get_num_threads()
    try:
        res = []
        for th in (1, 10):
            sd.set_num_threads(th)
            rb = R.glm(Y, X, family="binomial", link="probit", cov_type="HC0")
            res.append((a(rb.params), a(rb.bse), a(rb.n_iter)))
            for b in range(B):
                r1 = R.glm(Y[b], X[b], family="binomial", link="probit", cov_type="HC0")
                assert np.array_equal(a(rb.params)[b], a(r1.params)) and np.array_equal(a(rb.bse)[b], a(r1.bse))
        assert all(np.array_equal(u, v) for u, v in zip(*res))
    finally:
        sd.set_num_threads(old)


def _conv_invariant(fit, ref, n_iter_cap=None, tol=1e-12):
    """converged implies the coefficients of the full fit to rounding."""
    full = a(ref.params)
    for r, cap in fit:
        p = a(r.params)
        if n_iter_cap:
            assert r.n_iter <= cap
        if r.converged:
            assert np.max(np.abs(p - full)) <= tol * max(1.0, np.max(np.abs(full))), (cap, p, full)


# ---- G93: converged needs the polish; n_iter <= max_iter


def test_glm_converged_means_polished():
    for fam, link, y in (("binomial", "probit", YB), ("poisson", "sqrt", YP), ("binomial", "logit", YB)):
        ref = R.glm(y, X, family=fam, link=link)
        assert ref.converged
        fits = [(R.glm(y, X, family=fam, link=link, max_iter=m), m) for m in range(1, 12)]
        _conv_invariant(fits, ref, n_iter_cap=True)
        # some cap stops the polish after the deviance's test passed: not converged there
        assert any(not r.converged for r, _ in fits)


# ---- G94: constant y


def test_glm_zero_scale_llnull_minus_inf():
    """phi = 0 with a null model that does not fit exactly: llnull's limit is -inf (G97)."""
    x = np.arange(1.0, 9.0)
    r = R.glm(1.0 + 2.0 * x, x, family="gaussian", link="identity", scale="X2")
    assert r.scale == 0.0 and r.llf == math.inf
    g = R.glm(np.exp(0.5 * x), x, family="gamma", link="log")
    if g.scale == 0.0:
        assert g.llf == math.inf and g.llnull == -math.inf and g.llr == math.inf


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


# ---- frequency weights' HC2/HC3 and HAC lags; predict's offset


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
    for kw, nm in (
        (dict(offset=np.array([0.0, np.inf])), "offset"),
        (dict(exposure=np.array([1.0, np.inf])), "exposure"),
    ):
        with pytest.raises(ValueError, match=rf"^predict\(\): '{nm}' contains infinity$"):
            r.predict(X[:2], **kw)
        with pytest.raises(ValueError, match=rf"^predict_interval\(\): '{nm}' contains infinity$"):
            r.predict_interval(X[:2], **kw)


def test_glm_omit_keeps_errors_in_nan_rows():
    y = YP.copy()
    y[0] = np.nan
    for kw, msg in (
        (dict(var_weights=np.r_[-1.0, np.ones(N - 1)]), "'var_weights' must be >= 0"),
        (dict(offset=np.r_[np.inf, np.zeros(N - 1)]), "'offset' contains infinity"),
    ):
        with pytest.raises(ValueError, match=msg):
            R.glm(y, X, family="poisson", nan_policy="omit", **kw)


def test_converged_at_the_cap():
    """R94/R95: the cap stops a fit only before the test and the polish complete: a fit that
    finishes exactly at max_iter is converged with n_iter = max_iter."""
    ref = R.glm(YB, X, family="binomial")
    m = ref.n_iter
    at = R.glm(YB, X, family="binomial", max_iter=m)
    assert at.n_iter == m and at.converged and a(at.params).tolist() == a(ref.params).tolist()
    assert not R.glm(YB, X, family="binomial", max_iter=m - 1).converged


@pytest.mark.parametrize(
    "fam,link,y0,kw",
    [
        ("poisson", "log", 3.0, {}),
        ("poisson", "sqrt", 3.0, {}),
        ("poisson", "identity", 3.0, {}),
        ("gamma", "inverse", 2.0, {}),
        ("gamma", "log", 2.0, {}),
        ("gamma", "identity", 2.0, {}),
        ("gaussian", "log", 2.0, {}),
        ("binomial", "logit", 0.3, {"trials": 10.0}),
        ("binomial", "probit", 0.3, {"trials": 10.0}),
        ("binomial", "cloglog", 0.3, {"trials": 10.0}),
        ("binomial", "log", 0.3, {"trials": 10.0}),
    ],
)
def test_glm_constant_y_every_family(fam, link, y0, kw):
    """G111: the exact solution (g(y0), 0, ...) with mu = y0; no spurious significance: NaN for
    an estimated scale (0/0), t 0 and p 1 for a fixed one."""
    kw = {k: np.full(N, v) for k, v in kw.items()}
    r = R.glm(np.full(N, y0), X, family=fam, link=link, **kw)
    sm_link = {
        "logit": lambda m: math.log(m / (1 - m)),
        "probit": lambda m: float(sd.special.ndtri(m)),
        "cloglog": lambda m: math.log(-math.log1p(-m)),
        "log": math.log,
        "identity": lambda m: m,
        "sqrt": math.sqrt,
        "inverse": lambda m: 1 / m,
    }[link]
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


@pytest.mark.parametrize(
    "fam,link,y0,kw",
    [
        ("poisson", "identity", 0.0, {}),
        ("poisson", "sqrt", 0.0, {}),
        ("binomial", "log", 1.0, {}),
        ("binomial", "log", 1.0, {"trials": 3.0}),
    ],
)
def test_glm_constant_y_on_the_boundary_is_exact(fam, link, y0, kw):
    """G114: y constant at the boundary of the family's range with g(y0) finite: the exact
    solution (0, ..., 0) without iterations, converged, no separation, llf 0; the inference NaN
    (V(y0) = 0)."""
    x = np.arange(1.0, 9.0)
    kw = {k: np.full(8, v) for k, v in kw.items()}
    for X_ in (x, np.column_stack([x, x])):  # an aliased column stays NaN (G9)
        r = R.glm(np.full(8, y0), X_, family=fam, link=link, **kw)
        k = a(r.params).shape[0]
        assert a(r.params)[:2].tolist() == [0.0, 0.0] and (k == 2 or math.isnan(float(r.params[2])))
        assert r.converged and r.n_iter == 0 and not r.separation
        assert r.llf == 0.0 and r.llnull == 0.0 and r.deviance == 0.0 and r.null_deviance == 0.0
        assert r.pearson_chi2 == 0.0 and np.all(a(r.fittedvalues) == y0)
        for res in (r.resid_response, r.resid_pearson, r.resid_deviance, r.resid_working):
            assert np.all(a(res) == 0.0)
        assert np.all(np.isnan(a(r.bse))) and np.all(np.isnan(a(r.pvalues))) and np.all(np.isnan(a(r.conf_int())))
        assert np.all(np.isnan(a(r.predict_interval(np.full((2, k - 1), 2.0)))))
    # a robust covariance is NaN too; an estimated scale is 0 (G97's +inf llf)
    assert np.all(np.isnan(a(R.glm(np.zeros(8), x, family="poisson", link="identity", cov_type="HC0").bse)))
    rx = R.glm(np.zeros(8), x, family="poisson", link="sqrt", scale="X2")
    assert rx.scale == 0.0 and rx.llf == math.inf
    # the log link's zeros stay separation (G33)
    assert R.glm(np.zeros(8), x, family="poisson").separation


# ---- an underflow of mu or 1 - mu is not the boundary (D-2, RG-2)


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
    """An iteration that ended on steps halved back into the valid range (the binomial log
    link's MLE on mu = 1) is not converged; nor are NaN coefficients."""
    rg = np.random.default_rng(2)
    n = 80
    X = rg.normal(size=(n, 2))
    y = (rg.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + X @ [0.5, -0.4])))).astype(float)
    g = R.glm(y, X, family="binomial", link="log")
    assert not g.converged and float(sd.max(g.fittedvalues)) > 1 - 1e-6
