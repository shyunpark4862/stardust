"""The linear fits, glm and the penalized fits against R (tests/regression/data/regression_r.json, made once by
tools/gen_regression_r.py with R 4.6.2, sandwich 3.1.3, lmtest 0.9.40 and glmnet 5.2; R is not
needed here). docs/design/regression.md §3.1-§3.5 lists where sd and R differ and why.

  - lm: coefficients, standard errors, fit statistics, residuals of zero-weight rows, aliased
    columns (R's tol = 1e-7 is sd's rtol=1e-7), the uncentered R^2 without an intercept,
    hatvalues / rstandard / rstudent / cooks.distance, sandwich's HC0-HC3, NeweyWest and
    vcovCL with lmtest::coeftest, dwtest and bptest.
  - glm: R's glm.fit at sd's solution for every family and link, weights, trials, offsets,
    no intercept and the quasi families; the null deviance, logLik and AIC conventions;
    sandwich and coeftest for the canonical links, and R's expected-information bread for the
    others.
  - glmnet: lasso, elastic net and ridge (standardized or not, weights) and the binomial and
    poisson penalized GLMs at single lambdas; the lambda_max of the default paths."""

import json
import pathlib

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

R = sd.regression
REF = json.loads((pathlib.Path(__file__).parent / "data" / "regression_r.json").read_text())
CANONICAL = {("gaussian", "identity"), ("binomial", "logit"), ("poisson", "log"), ("gamma", "inverse")}


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def r_(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def rel_to_r(got, ref):
    """max relative difference to R's value (absolute where R has 0); NaN where R has NA (aliased
    columns) must be NaN in sd too."""
    got, ref = a(got).ravel(), r_(ref).ravel()
    assert got.shape == ref.shape
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    m = ~np.isnan(ref)
    d = np.abs(got[m] - ref[m])
    return float(np.max(np.where(d == 0, 0.0, d / np.where(ref[m] == 0, 1.0, np.abs(ref[m]))), initial=0.0))


def obs_err(got, ref):
    got, ref = a(got).ravel(), r_(ref).ravel()
    assert got.shape == ref.shape
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


def finite(x):
    x = a(x)
    return x[~np.isnan(x)]


def lags(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


# ---- lm

LM = REF["lm"]


def lm_fit(p, **kw):
    kw = dict(intercept=p["intercept"], rtol=p.get("rtol"), **kw)
    if "w" in p:
        return R.wls(p["y"], p["X"], weights=p["w"], **kw)
    return R.ols(p["y"], p["X"], **kw)


@pytest.mark.parametrize("name", list(LM))
def test_lm_fit(name):
    p = LM[name]
    rr = p["R"]
    f = lm_fit(p)
    assert rel_to_r(f.params, rr["coef"]) < 1e-13
    assert rel_to_r(f.bse, rr["se"]) < 1e-13
    assert rel_to_r(finite(f.pvalues), rr["p"]) < 1e-12
    assert rel_to_r(np.sqrt(f.scale), rr["sigma"]) < 1e-14
    assert (f.df_resid, f.nobs) == (rr["df_resid"][0], rr["nobs"][0])  # zero weights are not counted (G20)
    assert rel_to_r(f.rsquared, rr["r2"]) < 1e-14  # uncentered without an intercept, even with a constant column (G25)
    assert rel_to_r(f.rsquared_adj, rr["r2adj"]) < 1e-14
    assert rel_to_r(f.fvalue, rr["fstat"]) < 1e-13
    assert obs_err(f.resid, rr["resid"]) < 1e-14  # every row, zero-weight rows too
    assert obs_err(f.fittedvalues, rr["fitted"]) < 1e-14
    assert rel_to_r(f.llf, rr["llf"]) < 1e-14
    assert f.aic + 2 == pytest.approx(rr["aic"][0], rel=1e-14)  # R counts sigma as a parameter


@pytest.mark.parametrize("name", list(LM))
def test_lm_diagnostics(name):
    p = LM[name]
    rr = p["R"]
    f = lm_fit(p)
    keep = a(p.get("w", np.ones(len(p["y"])))) > 0  # R's lm.influence drops zero-weight rows; sd gives 0 (G30)
    for got, key in (
        (f.leverage, "hat"),
        (f.resid_studentized_internal, "rstandard"),
        (f.resid_studentized_external, "rstudent"),
        (f.cooks_distance, "cooks"),
    ):
        assert obs_err(a(got)[keep], rr[key]) < 1e-13, key
        assert np.all(a(got)[~keep] == 0)
    if "dw" in rr:
        assert rel_to_r(f.durbin_watson, rr["dw"]) < 1e-14
        assert rel_to_r(f.breusch_pagan().lm, rr["bp"]) < 1e-13
        assert rel_to_r(f.breusch_pagan(studentize=False).lm, rr["bp0"]) < 1e-13


@pytest.mark.parametrize("name", list(LM))
def test_lm_robust(name):
    p = LM[name]
    rr = p["R"]
    n = len(p["y"])
    for cov in ("HC0", "HC1", "HC2", "HC3"):
        g = lm_fit(p, cov_type=cov)
        assert rel_to_r(finite(g.bse), rr["se_" + cov]) < 1e-13, cov  # sandwich leaves out aliased columns
        assert rel_to_r(finite(g.pvalues), rr["p_" + cov]) < 1e-12, cov  # t(df_resid), as coeftest (G23)
    g = lm_fit(p, cov_type="HAC", maxlags=lags(n))  # NeweyWest(prewhite=FALSE, adjust=FALSE) (G26)
    assert rel_to_r(finite(g.bse), rr["se_HAC"]) < 1e-13
    assert rel_to_r(finite(g.pvalues), rr["p_HAC"]) < 1e-12
    g = lm_fit(p, cov_type="cluster", groups=sd.asarray((np.arange(n) % 7).astype(float)))
    assert rel_to_r(finite(g.bse), rr["se_CL"]) < 1e-13  # vcovCL(type="HC1"): G/(G-1)(n-1)/(n-k) (G27)
    assert rel_to_r(finite(g.pvalues), rr["p_CL"]) < 1e-12  # coeftest(df = G - 1)


def test_lm_rtol_default_keeps_near_aliased_column():
    p = LM["near_alias"]  # R's tol = 1e-7 drops the column; sd's default max(n, k)·eps keeps it (G21)
    assert np.isnan(r_(p["R"]["coef"])[3]) and np.isnan(a(lm_fit(p).params)[3])
    assert not np.isnan(a(lm_fit(dict(p, rtol=None)).params)).any()


# ---- glm

GLM = REF["glm"]


def glm_kwargs(p, cov_type="nonrobust", **extra):
    kw = dict(
        family=p["family"],
        link=p["link"],
        intercept=p.get("intercept", True),
        scale=p.get("scale"),
        cov_type=cov_type,
        **extra,
    )
    for k in ("trials", "offset", "var_weights"):
        if k in p:
            kw[k] = p[k]
    return kw


@pytest.mark.parametrize("name", list(GLM))
def test_glm_at_solution(name):
    p = GLM[name]
    rr = p["R"]
    f = R.glm(p["y"], p["X"], **glm_kwargs(p))
    # R's Fisher step from sd's solution stays there: sd is at the maximum to rounding
    assert rel_to_r(f.params, rr["coef"]) < 1e-13
    assert rel_to_r(f.bse, rr["se"]) < 1e-13
    assert rel_to_r(f.tvalues, rr["stat"]) < 1e-13
    # z for binomial and poisson, t(df_resid) otherwise and for 'X2' (G38, G41)
    assert rel_to_r(f.pvalues, rr["p"]) < 1e-11
    assert rel_to_r(f.deviance, rr["deviance"]) < 1e-14
    assert rel_to_r(f.pearson_chi2, rr["pearson"]) < 1e-14
    assert rel_to_r(f.scale, rr["dispersion"]) < 1e-14  # Pearson chi^2 / df (G34); 'X2' is the quasi families'
    assert f.df_resid == rr["df_resid"][0]  # zero prior weights are not counted (G20)
    assert obs_err(f.fittedvalues, rr["fitted"]) < 1e-14
    assert rel_to_r(f.null_deviance, rr["null_deviance"]) < 1e-13  # no intercept: eta = offset, as R


@pytest.mark.parametrize("name", list(GLM))
def test_glm_loglik_conventions(name):
    p = GLM[name]
    rr = p["R"]
    f = R.glm(p["y"], p["X"], **glm_kwargs(p))
    fam = p["family"]
    if "r_family" in p:
        assert rr["llf"][0] is None  # quasi families have no likelihood in R
    elif fam in ("binomial", "poisson"):
        assert rel_to_r(f.llf, rr["llf"]) < 1e-14  # with log C(n, ny) for trials, sum w log f for weights
        assert f.aic == pytest.approx(rr["aic"][0], rel=1e-14)
    elif "var_weights" not in p:
        # R's logLik puts the scale at deviance / n (and its AIC counts it); sd uses the Pearson scale
        g = R.glm(p["y"], p["X"], **dict(glm_kwargs(p), scale=f.deviance / f.nobs))
        assert rel_to_r(g.llf, rr["llf"]) < 1e-14
        if p["link"] == "identity" and fam == "gaussian":
            assert rel_to_r(f.llf, rr["llf"]) < 1e-14  # the concentrated likelihood, as ols
            assert f.aic + 2 == pytest.approx(rr["aic"][0], rel=1e-14)


def _mu_eta(link, eta):
    return {
        "identity": np.ones_like(eta),
        "log": np.exp(eta),
        "logit": np.exp(-np.logaddexp(0, eta) - np.logaddexp(0, -eta)),
        "probit": st.norm.pdf(eta),
        "cloglog": np.exp(eta - np.exp(eta)),
        "sqrt": 2 * eta,
        "inverse": -1 / eta**2,
    }[link]


def _variance(family, mu):
    return {"gaussian": np.ones_like(mu), "binomial": mu * (1 - mu), "poisson": mu, "gamma": mu**2}[family]


@pytest.mark.parametrize("name", [n for n, p in GLM.items() if "se_HC0" in p["R"] or "se_HC0_expected" in p["R"]])
def test_glm_robust(name):
    p = GLM[name]
    rr = p["R"]
    n = len(p["y"])
    fixed_scale = p["family"] in ("binomial", "poisson") and p.get("scale") is None
    if "se_HC0_expected" in rr:
        # sandwich's bread is the expected information; sd's is the observed one (statsmodels'),
        # which differs for the non-canonical links. R's value is the expected-information sandwich.
        f = R.glm(p["y"], p["X"], **glm_kwargs(p))
        X = np.column_stack([np.ones(n), a(p["X"])])
        mu, eta = a(f.fittedvalues), a(f.linear_predictor)
        m = a(p.get("var_weights", np.ones(n))) * a(p.get("trials", np.ones(n)))
        d, v = _mu_eta(p["link"], eta), _variance(p["family"], mu)
        bread = np.linalg.inv(X.T @ ((m * d * d / v)[:, None] * X))
        s = X * (m * (a(p["y"]) - mu) * d / v)[:, None]
        assert rel_to_r(np.sqrt(np.diag(bread @ s.T @ s @ bread)), rr["se_HC0_expected"]) < 1e-12
        assert rel_to_r(R.glm(p["y"], p["X"], **glm_kwargs(p, "HC0")).bse, rr["se_HC0_expected"]) > 1e-3
        return
    for cov, kw in (("HC0", {}), ("HC1", {}), ("HC2", {}), ("HC3", {}), ("HAC", dict(maxlags=lags(n)))):
        g = R.glm(p["y"], p["X"], **glm_kwargs(p, cov, **kw))
        assert rel_to_r(g.bse, rr["se_" + cov]) < 1e-13, cov
        # coeftest on a glm is z for every family; sd is z only when the scale is fixed (G39)
        z = a(g.params) / a(g.bse)
        assert rel_to_r(2 * st.norm.sf(np.abs(z)), rr["p_" + cov]) < 1e-11, cov
        if fixed_scale:
            assert rel_to_r(g.pvalues, rr["p_" + cov]) < 1e-11, cov
        else:
            assert g.df_resid_inference == g.df_resid
    g = R.glm(p["y"], p["X"], **glm_kwargs(p, "cluster", groups=sd.asarray((np.arange(n) % 9).astype(float))))
    assert rel_to_r(g.bse, rr["se_CL"]) < 1e-13
    assert rel_to_r(g.pvalues, rr["p_CL"]) < 1e-11  # t(G - 1) = coeftest(df = G - 1) (G27)


# ---- glmnet

GN = REF["glmnet"]
GD = {k: np.asarray(v, dtype=float) for k, v in GN["data"].items()}


def _sd_penalized(c, lam):
    y = GD[c["y"]]
    w = GD["w"] if c.get("weights") else None
    kw = dict(standardize=c["std"], weights=w)
    l1 = c["l1"]
    if c["family"] != "gaussian":
        return R.penalized_glm(y, GD["X"], family=c["family"], alpha=lam, l1_ratio=l1, **kw)
    if l1 == 1:
        return R.lasso(y, GD["X"], alpha=lam, **kw)
    # glmnet's gaussian family standardizes y internally: its L2 term is lambda (1 - alpha) / s_y
    # with s_y the (weighted) population sd of y, while its L1 term is lambda alpha
    ww = np.ones(len(y)) if w is None else w
    s_y = np.sqrt(ww @ (y - ww @ y / ww.sum()) ** 2 / ww.sum())
    l2 = lam * (1 - l1) / s_y
    if l1 == 0:
        return R.ridge(y, GD["X"], alpha=l2, **kw)
    return R.elastic_net(y, GD["X"], alpha=lam * l1 + l2, l1_ratio=lam * l1 / (lam * l1 + l2), **kw)


@pytest.mark.parametrize("case", GN["cases"], ids=[c["name"] for c in GN["cases"]])
def test_glmnet(case):
    rr = case["R"]
    for i, lam in enumerate(case["lams"], 1):
        f = _sd_penalized(case, lam)
        ref = r_(rr[f"coef{i}"])
        got = a(f.params)
        assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-9, lam  # glmnet's thresh = 1e-20
        assert int(np.sum(got[1:] != 0)) == rr[f"df{i}"][0]  # glmnet's df: the nonzero coefficients
        if case["l1"] == 1:
            assert f.df_model == rr[f"df{i}"][0]
        dev_ratio = f.rsquared if case["family"] == "gaussian" else f.deviance_ratio
        assert rel_to_r(dev_ratio, rr[f"dev_ratio{i}"]) < 1e-9


@pytest.mark.parametrize(
    "case", [c for c in GN["cases"] if c["family"] == "gaussian" and c["l1"] == 1], ids=lambda c: c["name"]
)
def test_glmnet_lambda_max(case):
    rr = case["R"]
    y = GD[case["y"]]
    w = GD["w"] if case.get("weights") else None
    g = R.lasso_cv(y, GD["X"], rng=sd.random.default_rng(0), standardize=case["std"], weights=w)
    assert rel_to_r(a(g.alphas)[0], rr["lambda_max"]) < 1e-13
    g = R.ridge_cv(y, GD["X"], cv="gcv", standardize=case["std"], weights=w)  # alpha = max(alpha, 1e-3) (G52)
    assert rel_to_r(a(g.alphas)[0], rr["lambda_max_ridge"]) < 1e-13
