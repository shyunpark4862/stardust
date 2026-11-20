"""Tests on fitted models (docs/api/regression.md §9.1-§9.2, docs/design/regression.md §3.8).

Accuracy against Arb (tests/regression/ref/model_tests.json, tests/regression/refdata_model_tests.py: every
value from its definition), agreement with statsmodels 0.15.0 (t_test, wald_test / f_test,
compare_f_test, compare_lr_test, anova_lm, variance_inflation_factor, OLSInfluence,
GLMInfluence, het_white, het_goldfeldquandt, linear_reset), and the rules of the decisions:
NaN for aliased parameters, a perfect fit, no degree of freedom and NaN data; batches equal to
loops; the same bits for any thread count; the read-only result classes; the checks of the
comparisons; the messages. R's values are tests/regression/test_model_tests_against_r.py's."""

import math

import numpy as np
import pytest

import stardust as sd

from . import refdata_model_tests

R = sd.regression
REFS = refdata_model_tests.load() if refdata_model_tests.REF.exists() else {}
LIN = sorted(k for k, v in REFS.items() if v["kind"] == "lin")
GLMS = sorted(k for k, v in REFS.items() if v["kind"] == "glm")


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def rel(got, hl):
    """The largest relative error of got against references [hi, lo] (or lists of them); an
    absolute error where the reference is 0."""
    got = a(got).ravel()
    ref = np.asarray(hl, dtype=float).reshape(-1, 2)
    assert got.shape[0] == ref.shape[0]
    worst = 0.0
    for g, (hi, lo) in zip(got, ref):
        d = abs((g - hi) - lo)
        r = abs(hi + lo)
        worst = max(worst, d / r if r else d)
    return worst


def _lin_fit(p, cols=None, cov_type="nonrobust"):
    X = np.asarray(p["X"])
    if cols is not None:
        X = X[:, :cols]
    if p["weights"] is None:
        return R.ols(p["y"], X, intercept=p["intercept"], cov_type=cov_type)
    return R.wls(p["y"], X, weights=p["weights"], intercept=p["intercept"], cov_type=cov_type)


# measured: at most 1.2e-13 on the well-conditioned problems (every field, per element,
# p-values relative to themselves); about kappa u on cond (kappa near 1e8: 2.3e-8 for dfbetas,
# 4.4e-9 for the Type I sums of squares, 7e-10 for t)
LIN_TOL = {"ols": 2e-13, "wls_zero": 2e-13, "noint": 2e-13, "factor": 2e-13, "cond": 1e-7}


@pytest.mark.skipif(not LIN, reason="tests/regression/ref/model_tests.json is missing")
@pytest.mark.parametrize("name", LIN)
def test_linear_against_arb(name):
    p = REFS[name]
    ref = p["ref"]
    tol = LIN_TOL[name]
    r = _lin_fit(p)
    t = r.t_test(p["R"], p["q"])
    for f, key in (("effect", "t_effect"), ("sd", "t_sd"), ("tvalue", "t_tvalue"), ("pvalue", "t_pvalue")):
        assert rel(getattr(t, f), ref[key]) <= tol, f
    w = r.wald_test(p["R"], p["q"])
    assert rel(w.statistic, ref["wald_F"]) <= tol
    assert rel(w.pvalue, ref["wald_p"]) <= tol
    small = _lin_fit(p, cols=p["small"])
    c = r.compare_f_test(small)
    assert rel(c.fvalue, ref["compare_F"]) <= tol and rel(c.pvalue, ref["compare_F_p"]) <= tol
    c = r.compare_lr_test(small)
    assert rel(c.statistic, ref["compare_lr"]) <= tol and rel(c.pvalue, ref["compare_lr_p"]) <= tol
    t1 = R.anova(r, terms=p["terms"])
    assert rel(t1.sum_sq[:-1], ref["anova1_ss"]) <= tol
    assert rel(t1.statistic[:-1], ref["anova1_F"]) <= tol
    assert rel(t1.pvalue[:-1], ref["anova1_p"]) <= tol
    t3 = R.anova(r, typ=3, terms=p["terms"])
    assert rel(t3.sum_sq[:-1], ref["anova3_ss"]) <= tol
    assert rel(t3.statistic[:-1], ref["anova3_F"]) <= tol
    assert rel(t3.pvalue[:-1], ref["anova3_p"]) <= tol
    v = a(r.vif)
    assert rel(v[1:] if p["intercept"] else v, ref["vif"]) <= tol
    assert rel(r.dffits, ref["dffits"]) <= tol
    assert rel(r.dfbetas, ref["dfbetas"]) <= tol
    if name == "cond":
        # White's auxiliary design holds the squares and products of two columns 1e-7 apart:
        # its condition is near kappa^2 = 1e16, past the doubles, and the QR drops one
        # (x1^2 - 2 x1 x2 + x2^2 is (x1 - x2)^2, 1e-14 of them): one degree of freedom less
        assert r.white_test().df == 8
    else:
        wh = r.white_test()
        for f, key in (
            ("lm", "white_lm"),
            ("lm_pvalue", "white_lm_p"),
            ("fvalue", "white_F"),
            ("f_pvalue", "white_F_p"),
        ):
            assert rel(getattr(wh, f), ref[key]) <= tol, f
        wh = r.white_test(cross_terms=False)
        for f, key in (("lm", "white_nocross_lm"), ("lm_pvalue", "white_nocross_lm_p")):
            assert rel(getattr(wh, f), ref[key]) <= tol, f
    n1, drop, by = p["gq"]
    m = int(sum(1 for v in (p["weights"] or [1] * len(p["y"])) if v > 0))
    g = r.goldfeld_quandt(split=n1 if n1 is not None else m // 2, drop=drop, order_by=by)
    assert rel(g.fvalue, ref["gq_F"]) <= tol and rel(g.pvalue, ref["gq_p"]) <= tol
    rs = r.reset_test()
    assert rel(rs.fvalue, ref["reset_F"]) <= tol and rel(rs.pvalue, ref["reset_p"]) <= tol


@pytest.mark.skipif(not GLMS, reason="tests/regression/ref/model_tests.json is missing")
@pytest.mark.parametrize("name", GLMS)
def test_glm_against_arb(name):
    p = REFS[name]
    ref = p["ref"]
    tol = 1e-12
    X = np.asarray(p["X"])
    r = R.glm(p["y"], X, family=p["family"], link=p["link"])
    t = r.t_test(p["R"])
    assert rel(t.tvalue, ref["t_tvalue"]) <= tol and rel(t.pvalue, ref["t_pvalue"]) <= tol
    w = r.wald_test(p["R"])
    assert rel(w.statistic, ref["wald_stat"]) <= tol and rel(w.pvalue, ref["wald_p"]) <= tol
    small = R.glm(p["y"], X[:, : p["small"]], family=p["family"], link=p["link"])
    c = r.compare_lr_test(small)
    assert rel(c.statistic, ref["compare_lr"]) <= tol and rel(c.pvalue, ref["compare_lr_p"]) <= tol
    if "compare_F" in ref:
        c = r.compare_f_test(small)
        assert rel(c.fvalue, ref["compare_F"]) <= tol and rel(c.pvalue, ref["compare_F_p"]) <= tol
    t1 = R.anova(r, terms=p["terms"])
    assert rel(t1.sum_sq[:-1], ref["anova1_dev"]) <= tol and rel(t1.pvalue[:-1], ref["anova1_p"]) <= tol
    t3 = R.anova(r, typ=3, terms=p["terms"])
    W3 = a(t3.statistic[:-1]) * (1.0 if p["family"] in ("binomial", "poisson") else a(t3.df[:-1]))
    assert rel(W3, ref["anova3_wald"]) <= tol
    assert rel(r.leverage, ref["hat"]) <= tol
    assert rel(r.cooks_distance, ref["cooks"]) <= tol
    assert rel(r.dffits, ref["dffits"]) <= tol
    assert rel(r.dfbetas, ref["dfbetas"]) <= tol


# ---- statsmodels


def _problem(n=60, k=3, seed=0, het=True):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k))
    e = rng.standard_normal(n) * ((1 + np.abs(X[:, 0])) if het else 1.0)
    return X @ np.linspace(1, -0.5, k) + 0.3 + e, X


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC", "cluster"])
def test_t_wald_against_statsmodels(cov_type):
    sm = pytest.importorskip("statsmodels.api")
    y, X = _problem(seed=2)
    kw, skw = {}, {}
    if cov_type == "HAC":
        kw, skw = {"maxlags": 3}, {"cov_kwds": {"maxlags": 3}}
    if cov_type == "cluster":
        gr = np.arange(60) % 7
        kw, skw = {"groups": gr}, {"cov_kwds": {"groups": gr}}
    r = R.ols(y, X, cov_type=cov_type, **kw)
    s = sm.OLS(y, sm.add_constant(X)).fit(cov_type=cov_type, use_t=True, **skw)
    Rm = np.array([[0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0.0]])
    q = np.array([0.5, 0.0, 0.2])
    t, ts = r.t_test(Rm, q), s.t_test((Rm, q))
    np.testing.assert_allclose(a(t.effect), ts.effect.ravel(), rtol=1e-13)
    np.testing.assert_allclose(a(t.sd), ts.sd.ravel(), rtol=1e-12)
    np.testing.assert_allclose(a(t.tvalue), ts.tvalue.ravel(), rtol=1e-12)
    np.testing.assert_allclose(a(t.pvalue), ts.pvalue.ravel(), rtol=1e-11)
    np.testing.assert_allclose(a(t.conf_int(alpha=0.1)), ts.conf_int(alpha=0.1), rtol=1e-12)
    w, ws = r.wald_test(Rm, q), s.f_test((Rm, q))
    assert w.distribution == "F" and w.df_num == 3 and w.df_denom == ws.df_denom
    assert w.statistic == pytest.approx(float(ws.fvalue), rel=1e-12)
    assert w.pvalue == pytest.approx(float(ws.pvalue), rel=1e-11)


def test_t_wald_glm_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(5)
    X = rng.standard_normal((80, 3))
    eta = 0.2 + X @ [0.8, -0.5, 0.0]
    y = (rng.uniform(size=80) < 1 / (1 + np.exp(-eta))).astype(float)
    r = R.glm(y, X, family="binomial")
    s = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit(tol=1e-14)
    Rm = np.array([[0, 1, 1, 0], [0, 0, 0, 1.0]])
    t, ts = r.t_test(Rm), s.t_test(Rm)
    assert t.df == math.inf
    np.testing.assert_allclose(a(t.tvalue), ts.tvalue.ravel(), rtol=1e-9)
    w, ws = r.wald_test(Rm), s.wald_test(Rm, scalar=True)
    assert w.distribution == "chi2" and w.df == 2 and not hasattr(w, "df_num")
    assert w.statistic == pytest.approx(float(ws.statistic), rel=1e-9)
    yg = rng.gamma(2.0, np.exp(eta) / 2.0)
    r = R.glm(yg, X, family="gamma", link="log")
    s = sm.GLM(yg, sm.add_constant(X), family=sm.families.Gamma(sm.families.links.Log())).fit(tol=1e-14)
    w, ws = r.wald_test(Rm), s.f_test(Rm)
    assert w.distribution == "F" and w.df_denom == 76
    assert w.statistic == pytest.approx(float(ws.fvalue), rel=1e-9)


def test_t_wald_robust_quantile_nls():
    """rlm, quantreg and nls take the general tests with their own covariance and t(df)."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((60, 2))
    y = 1 + X @ [1.0, -1.0] + rng.standard_t(3, 60)
    for r in (R.rlm(y, X), R.rlm(y, X, method="MM", rng=sd.random.default_rng(1)), R.quantreg(y, X, q=0.4)):
        t = r.t_test(np.eye(3))
        np.testing.assert_allclose(a(t.tvalue), a(r.tvalues), rtol=1e-14)
        np.testing.assert_allclose(a(t.pvalue), a(r.pvalues), rtol=1e-13)
        w = r.wald_test([[0, 1, 0]])
        assert w.statistic == pytest.approx(float(r.tvalues[1]) ** 2, rel=1e-13)
        assert w.df_denom == r.df_resid
    q = R.quantreg(y, X, q=[0.25, 0.5, 0.75])
    t = q.t_test([[0, 1, 1], [0, 1, 0]])
    assert t.tvalue.shape == (3, 2)
    for i, qq in enumerate((0.25, 0.5, 0.75)):
        one = R.quantreg(y, X, q=qq).t_test([[0, 1, 1], [0, 1, 0]])
        assert np.array_equal(a(one.tvalue), a(t.tvalue)[i])
    x = np.linspace(0, 1, 40)
    yy = 2 * np.exp(0.5 * x) + 0.05 * rng.standard_normal(40)
    nl = R.nls(lambda x, b0, b1: b0 * sd.exp(b1 * x), x, yy, p0=[1.0, 1.0])
    t = nl.t_test([1.0, 0.0], 2.0)
    assert t.tvalue == pytest.approx((float(nl.params[0]) - 2.0) / float(nl.bse[0]), rel=1e-13)


def test_compare_and_anova_against_statsmodels():
    pd = pytest.importorskip("pandas")
    smf = pytest.importorskip("statsmodels.formula.api")
    from statsmodels.stats.anova import anova_lm

    y, X = _problem(seed=3)
    df = pd.DataFrame({"y": y, "a": X[:, 0], "b": X[:, 1], "c": X[:, 2]})
    big, small, mid = R.ols(y, X), R.ols(y, X[:, :1]), R.ols(y, X[:, :2])
    sb, ss, smid = smf.ols("y ~ a + b + c", df).fit(), smf.ols("y ~ a", df).fit(), smf.ols("y ~ a + b", df).fit()
    f = big.compare_f_test(small)
    fs = sb.compare_f_test(ss)
    assert tuple(f) == pytest.approx(fs, rel=1e-12)
    assert f.df_denom == 56
    lr = big.compare_lr_test(small)
    ls = sb.compare_lr_test(ss)
    assert (lr.statistic, lr.pvalue, lr.df) == pytest.approx((ls.statistic, ls.pvalue, ls.df_diff), rel=1e-12)
    tab = R.anova(small, mid, big)
    tabs = anova_lm(ss, smid, sb)
    np.testing.assert_allclose(a(tab.df_resid), tabs["df_resid"], rtol=0)
    np.testing.assert_allclose(a(tab.ssr), tabs["ssr"], rtol=1e-13)
    np.testing.assert_allclose(a(tab.ss_diff)[1:], tabs["ss_diff"][1:], rtol=1e-12)
    np.testing.assert_allclose(a(tab.statistic)[1:], tabs["F"][1:], rtol=1e-11)
    # every row's F uses the largest model's scale and its df_resid (R); statsmodels takes each
    # row's own df_resid for the p-value ([statsmodels와 다름]): the same for the last row
    assert float(tab.pvalue[2]) == pytest.approx(tabs["Pr(>F)"][2], rel=1e-10)
    import scipy.stats as st

    assert float(tab.pvalue[1]) == pytest.approx(st.f.sf(tabs["F"][1], 1, 56), rel=1e-10)
    t1, t1s = R.anova(big), anova_lm(sb)
    np.testing.assert_allclose(a(t1.sum_sq), t1s["sum_sq"], rtol=1e-12)
    np.testing.assert_allclose(a(t1.statistic)[:-1], t1s["F"][:-1], rtol=1e-11)
    t3, t3s = R.anova(big, typ=3), anova_lm(sb, typ=3)
    np.testing.assert_allclose(a(t3.sum_sq), t3s["sum_sq"][1:], rtol=1e-11)  # statsmodels has an Intercept row
    np.testing.assert_allclose(a(t3.pvalue)[:-1], t3s["PR(>F)"][1:-1], rtol=1e-10)
    for ct in ("HC0", "HC1", "HC2", "HC3"):
        t3r = R.anova(R.ols(y, X, cov_type=ct), typ=3)
        t3rs = anova_lm(sb, typ=3, robust=ct.lower())
        np.testing.assert_allclose(a(t3r.statistic)[:-1], t3rs["F"][1:-1], rtol=1e-11)
    assert t1.terms == ["x1", "x2", "x3", "Residual"]
    assert t1.terms is not t1.terms  # a new list at each read (G113)


def test_vif_dffits_dfbetas_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

    y, X = _problem(seed=4)
    X[:, 2] += 0.8 * X[:, 0]
    r = R.ols(y, X)
    ex = sm.add_constant(X)
    s = sm.OLS(y, ex).fit()
    v = a(r.vif)
    assert math.isnan(v[0])
    np.testing.assert_allclose(v[1:], [variance_inflation_factor(ex, j) for j in (1, 2, 3)], rtol=1e-13)
    inf = OLSInfluence(s)
    np.testing.assert_allclose(a(r.dffits), inf.dffits[0], rtol=1e-12)
    np.testing.assert_allclose(a(r.dfbetas), inf.dfbetas, rtol=1e-10, atol=1e-13)  # statsmodels' pinv
    # no intercept: the centered VIF, an intercept put in (statsmodels' default standardize=True)
    rn = R.ols(y, X, intercept=False)
    np.testing.assert_allclose(a(rn.vif), [variance_inflation_factor(X, j) for j in range(3)], rtol=1e-12)
    # WLS: the whitened problem (statsmodels' influence of the OLS of sqrt(w) y on sqrt(w) X)
    w = np.random.default_rng(1).uniform(0.3, 3.0, 60)
    rw = R.wls(y, X, weights=w)
    sw = np.sqrt(w)
    inf = OLSInfluence(sm.OLS(y * sw, ex * sw[:, None]).fit())
    np.testing.assert_allclose(a(rw.dffits), inf.dffits[0], rtol=1e-11)
    np.testing.assert_allclose(a(rw.wresid), y * sw - (ex * sw[:, None]) @ a(rw.params), rtol=1e-11, atol=1e-13)


def test_glm_influence_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(8)
    X = rng.standard_normal((70, 2))
    eta = 0.3 + X @ [0.7, -0.4]
    y = rng.poisson(np.exp(eta)).astype(float)
    r = R.glm(y, X, family="poisson")
    s = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson()).fit(tol=1e-14)
    inf = s.get_influence(observed=False)
    np.testing.assert_allclose(a(r.leverage), inf.hat_matrix_diag, rtol=1e-9)
    np.testing.assert_allclose(a(r.cooks_distance), inf.cooks_distance[0], rtol=1e-9)
    # a fixed scale: dfbetas is the one-step change over bse (statsmodels' GLMInfluence)
    np.testing.assert_allclose(a(r.dfbetas), inf.dfbetas, rtol=1e-8, atol=1e-12)


def test_glm_gaussian_influence_is_ols():
    y, X = _problem(seed=9)
    w = np.random.default_rng(9).uniform(0.3, 3.0, 60)
    w[7] = 0.0
    for o, g in (
        (R.ols(y, X), R.glm(y, X, family="gaussian")),
        (R.wls(y, X, weights=w), R.glm(y, X, family="gaussian", var_weights=w)),
    ):
        for name in ("leverage", "cooks_distance", "dffits", "dfbetas"):
            np.testing.assert_allclose(a(getattr(g, name)), a(getattr(o, name)), rtol=1e-11, atol=1e-14, err_msg=name)


def test_white_gq_reset_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.stats.diagnostic import het_goldfeldquandt, het_white, linear_reset

    y, X = _problem(n=61, seed=10)
    r = R.ols(y, X)
    ex = sm.add_constant(X)
    s = sm.OLS(y, ex).fit()
    assert tuple(r.white_test()) == pytest.approx(het_white(s.resid, ex), rel=1e-11)
    wt = r.white_test()
    assert wt.df == 9 and wt.df_num == 9 and wt.df_denom == 51
    g = r.goldfeld_quandt(alternative="two-sided", order_by=1)
    gs = het_goldfeldquandt(y, ex, idx=2, alternative="two-sided", result_object=False)
    assert (g.fvalue, g.pvalue) == pytest.approx(gs[:2], rel=1e-12)
    assert tuple(g)[2] == "two-sided"
    # one-sided, halves of unequal size (n odd): statsmodels takes F(df_1, df_2) for
    # s_2^2 / s_1^2, R's gqtest F(df_2, df_1) ([statsmodels와 다름])
    g = r.goldfeld_quandt(split=10)
    gs = het_goldfeldquandt(y, ex, split=10, alternative="increasing", result_object=False)
    assert g.fvalue == pytest.approx(gs[0], rel=1e-12)
    assert (g.df_num, g.df_denom) == (47.0, 6.0)
    import scipy.stats as st

    assert g.pvalue == pytest.approx(st.f.sf(g.fvalue, 47, 6), rel=1e-12)
    assert abs(g.pvalue - gs[1]) > 1e-3
    for kind, tt in (("fitted", "fitted"), ("regressor", "exog")):
        rs = r.reset_test(kind=kind)
        ls = linear_reset(s, power=3, test_type=tt, use_f=True)
        assert (rs.fvalue, rs.pvalue) == pytest.approx((float(ls.fvalue), float(ls.pvalue)), rel=1e-10)
        assert (rs.df_num, rs.df_denom) == (ls.df_num, ls.df_denom)
    rs = R.ols(y, X, cov_type="HC1").reset_test()
    ls = linear_reset(s, power=3, use_f=True, cov_type="HC1")
    assert rs.fvalue == pytest.approx(float(ls.fvalue), rel=1e-10)


def test_white_without_cross_terms_keeps_the_levels():
    """cross_terms=False: levels and squares (White's form; statsmodels' interaction_terms=False
    drops the levels: [statsmodels와 다름])."""
    y, X = _problem(seed=11)
    r = R.ols(y, X)
    e2 = a(r.resid) ** 2
    aux = R.ols(e2, np.column_stack([X, X**2]))
    w = r.white_test(cross_terms=False)
    assert w.lm == pytest.approx(60 * aux.rsquared, rel=1e-13)
    assert w.df == 6 and w.fvalue == pytest.approx(aux.fvalue, rel=1e-13)


def test_one_way_anova_is_f_oneway():
    """The one-way table from a group's dummies equals f_oneway within the accuracy contract
    (G159: the design matrix has no labels, so not the same kernel)."""
    import scipy.stats as st

    rng = np.random.default_rng(12)
    g = rng.integers(0, 4, 50)
    y = rng.standard_normal(50) + 0.4 * g
    D = np.column_stack([(g == j).astype(float) for j in (1, 2, 3)])
    t = R.anova(R.ols(y, D), terms=[[0, 1, 2]])
    ref = st.f_oneway(*[y[g == j] for j in range(4)])
    assert float(t.statistic[0]) == pytest.approx(ref.statistic, rel=1e-13)
    assert float(t.pvalue[0]) == pytest.approx(ref.pvalue, rel=1e-12)
    fo = getattr(getattr(sd, "htest", None), "f_oneway", None)
    if fo is not None:
        h = fo(*[y[g == j] for j in range(4)])
        assert float(t.statistic[0]) == pytest.approx(float(h.statistic), rel=1e-13)


# ---- the rules


def test_unit_rows_are_the_coefficients():
    y, X = _problem(seed=13)
    for r in (R.ols(y, X), R.ols(y, X, cov_type="HC2"), R.glm(y, X, family="gaussian")):
        t = r.t_test(np.eye(4))
        np.testing.assert_allclose(a(t.tvalue), a(r.tvalues), rtol=2e-15)
        np.testing.assert_allclose(a(t.sd), a(r.bse), rtol=2e-15)
        np.testing.assert_allclose(a(t.conf_int()), a(r.conf_int()), rtol=2e-15)
        w = r.wald_test(np.eye(4)[1:])
        assert w.statistic == pytest.approx(r.fvalue if hasattr(r, "fvalue") else w.statistic, rel=1e-13)
    one = R.ols(y, X).t_test([0, 1, 0, 0])
    assert isinstance(one.tvalue, float) and one.conf_int().shape == (2,)


def test_aliased_and_dependent_rows():
    y, X = _problem(seed=14)
    Xa = np.column_stack([X, X[:, 0] + X[:, 1]])
    r = R.ols(y, Xa)
    t = r.t_test(np.eye(5))
    assert np.isnan(a(t.tvalue)[4]) and np.isnan(a(t.effect)[4])
    np.testing.assert_allclose(a(t.tvalue)[:4], a(r.tvalues)[:4], rtol=2e-15)
    assert math.isnan(r.wald_test(np.eye(5)[1:]).statistic)  # a row on the aliased column: NaN
    # dependent rows are dropped (df = rank): the second row is twice the first
    w = r.wald_test([[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0]])
    w2 = r.wald_test([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
    assert w.df_num == 2 and w.statistic == pytest.approx(w2.statistic, rel=1e-14)
    v = a(r.vif)
    assert np.isnan(v[[0, 4]]).all() and np.isfinite(v[1:4]).all()
    db = a(r.dfbetas)
    assert np.isnan(db[:, 4]).all() and np.isfinite(db[:, :4]).all()
    t3 = R.anova(r, typ=3)
    assert a(t3.df).tolist() == [1, 1, 1, 0, 56] and math.isnan(float(t3.statistic[3]))
    t1 = R.anova(r)
    assert a(t1.df).tolist() == [1, 1, 1, 0, 56]


def test_degenerate_fits():
    rng = np.random.default_rng(15)
    x = np.arange(10.0)
    perfect = R.ols(2 * x + 1, x)
    t = perfect.t_test([0, 1])
    assert t.sd == 0 and math.isnan(t.tvalue) and math.isnan(t.pvalue)
    assert math.isnan(perfect.wald_test([[0, 1]]).statistic)
    assert np.isnan(a(perfect.dffits)).all() and np.isnan(a(perfect.dfbetas)).all()
    df0 = R.ols(rng.standard_normal(3), rng.standard_normal((3, 2)))
    t = df0.t_test([0, 1, 0])
    assert math.isfinite(t.effect) and math.isnan(t.sd) and math.isnan(t.pvalue)
    yn = rng.standard_normal(10)
    yn[3] = np.nan
    rn = R.ols(yn, rng.standard_normal((10, 2)))
    for v in (
        rn.t_test([0, 1, 0]).tvalue,
        rn.wald_test([[0, 1, 0]]).statistic,
        rn.white_test().lm,
        rn.goldfeld_quandt().fvalue,
        rn.reset_test().fvalue,
    ):
        assert math.isnan(v)
    assert np.isnan(a(rn.vif)).all() and np.isnan(a(rn.dfbetas)).all()
    const = R.ols(np.ones(12), rng.standard_normal((12, 2)))
    assert math.isnan(const.white_test().lm) and math.isnan(const.reset_test().fvalue)


def test_extreme_scales():
    rng = np.random.default_rng(16)
    X = rng.standard_normal((40, 2))
    y = X @ [1.0, 2.0] + rng.standard_normal(40)
    base = R.ols(y, X)
    big = R.ols(y * 1e-165, X * 1e160)
    for f in (
        lambda r: r.t_test([0, 1, -0.5]).tvalue,
        lambda r: r.wald_test([[0, 1, 0], [0, 0, 1]]).statistic,
        lambda r: r.white_test().lm,
        lambda r: r.reset_test().fvalue,
        lambda r: r.goldfeld_quandt().fvalue,
    ):
        assert f(big) == pytest.approx(f(base), rel=1e-13)
    np.testing.assert_allclose(a(big.vif)[1:], a(base.vif)[1:], rtol=1e-13)
    np.testing.assert_allclose(a(big.dfbetas), a(base.dfbetas), rtol=1e-12, atol=1e-15)


def test_batches_equal_loops():
    rng = np.random.default_rng(17)
    X = rng.standard_normal((30, 2))
    yb = rng.standard_normal((4, 30)) + X @ [1.0, 0.5]
    w = rng.uniform(0.5, 2.0, (4, 30))
    w[1, 3] = 0.0
    rb = R.wls(yb, X, weights=w)
    sb = R.wls(yb, X[:, :1], weights=w)
    Rm = [[0, 1, -1], [0, 0, 1]]
    out = {
        "t": a(rb.t_test(Rm).tvalue),
        "w": a(rb.wald_test(Rm).statistic),
        "vif": a(rb.vif),
        "dffits": a(rb.dffits),
        "dfbetas": a(rb.dfbetas),
        "white": a(rb.white_test().lm),
        "gq": a(rb.goldfeld_quandt(order_by=0).fvalue),
        "reset": a(rb.reset_test().fvalue),
        "cf": a(rb.compare_f_test(sb).fvalue),
        "lr": a(rb.compare_lr_test(sb).statistic),
        "wresid": a(rb.wresid),
    }
    for i in range(4):
        r1, s1 = R.wls(yb[i], X, weights=w[i]), R.wls(yb[i], X[:, :1], weights=w[i])
        one = {
            "t": a(r1.t_test(Rm).tvalue),
            "w": r1.wald_test(Rm).statistic,
            "vif": a(r1.vif),
            "dffits": a(r1.dffits),
            "dfbetas": a(r1.dfbetas),
            "white": r1.white_test().lm,
            "gq": r1.goldfeld_quandt(order_by=0).fvalue,
            "reset": r1.reset_test().fvalue,
            "cf": r1.compare_f_test(s1).fvalue,
            "lr": r1.compare_lr_test(s1).statistic,
            "wresid": a(r1.wresid),
        }
        for k, v in one.items():
            assert np.array_equal(np.asarray(v), out[k][i], equal_nan=True), k


def test_threads_bit_identical():
    rng = np.random.default_rng(18)
    X = rng.standard_normal((200_000, 4))
    y = X @ [1.0, 0.5, 0.0, -1.0] + rng.standard_normal(200_000) * (1 + np.abs(X[:, 0]))
    old = sd.get_num_threads()
    try:
        out = []
        for t in (1, 10):
            sd.set_num_threads(t)
            r = R.ols(y, X)
            out.append(
                [
                    a(r.dfbetas),
                    a(r.vif),
                    r.white_test().lm,
                    r.reset_test().fvalue,
                    r.wald_test([[0, 1, -1, 0, 0]]).statistic,
                    a(R.anova(r).sum_sq),
                ]
            )
        for x, z in zip(*out):
            assert np.array_equal(np.asarray(x), np.asarray(z), equal_nan=True)
    finally:
        sd.set_num_threads(old)


def test_result_classes():
    y, X = _problem(seed=19)
    r, s = R.ols(y, X), R.ols(y, X[:, :1])
    t = r.t_test([[0, 1, 0, 0], [0, 0, 1, 0]])
    assert isinstance(t, R.TTestResult)
    with pytest.raises(AttributeError):
        t.tvalue = 1
    with pytest.raises(ValueError, match="read-only"):
        t.tvalue[0] = 1.0
    ci = t.conf_int()
    ci[0, 0] = 0.0  # a new array at each call
    with pytest.raises(TypeError):
        iter(t)
    w = r.wald_test([[0, 1, 0, 0]])
    assert isinstance(w, R.WaldTestResult) and not hasattr(w, "df")
    f = r.compare_f_test(s)
    assert isinstance(f, R.CompareFTestResult) and len(tuple(f)) == 3
    lr = r.compare_lr_test(s)
    assert isinstance(lr, R.CompareLRTestResult) and tuple(lr) == (lr.statistic, lr.pvalue, lr.df)
    wh = r.white_test()
    assert isinstance(wh, R.WhiteTestResult) and len(tuple(wh)) == 4
    g = r.goldfeld_quandt()
    assert isinstance(g, R.GoldfeldQuandtResult) and tuple(g) == (g.fvalue, g.pvalue, "greater")
    rs = r.reset_test()
    assert isinstance(rs, R.ResetTestResult)
    with pytest.raises(TypeError):
        tuple(rs)
    tab = R.anova(r)
    assert isinstance(tab, R.AnovaResult) and "Type I" in tab.summary()
    with pytest.raises(AttributeError):
        _ = tab.ss_diff
    with pytest.raises(ValueError, match="read-only"):
        tab.sum_sq[0] = 1.0
    for name in ("vif", "dffits", "dfbetas", "wresid"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(r, name)[0] = 1.0
    bp = r.breusch_pagan()
    assert type(bp).__name__ == "BreuschPaganResult" and not hasattr(bp, "df")  # G157


def test_comparison_checks():
    y, X = _problem(seed=20)
    big = R.ols(y, X)
    with pytest.raises(
        TypeError,
        match="compare_f_test\\(\\): the model compared must be the same kind of result as this one \\(OLSResult\\)",
    ):
        big.compare_f_test(R.wls(y, X[:, :1], weights=np.ones(60)))
    with pytest.raises(ValueError, match="same data"):
        big.compare_f_test(R.ols(y + 1, X[:, :1]))
    with pytest.raises(ValueError, match="not nested"):
        big.compare_f_test(R.ols(y, np.column_stack([X[:, :1], X[:, 0] ** 2])))
    with pytest.raises(ValueError, match="not nested"):
        R.ols(y, X[:, :1]).compare_f_test(big)
    with pytest.raises(ValueError, match="robust covariance"):
        R.ols(y, X, cov_type="HC1").compare_lr_test(R.ols(y, X[:, :1]))
    # a column in the span of the big model's columns is nested, though not one of them
    sm2 = R.ols(y, X[:, 0] + X[:, 1])
    assert math.isfinite(big.compare_f_test(sm2).fvalue)
    rng = np.random.default_rng(21)
    yb = (rng.uniform(size=60) < 0.5).astype(float)
    gb = R.glm(yb, X, family="binomial")
    with pytest.raises(ValueError, match="compare_lr_test"):
        gb.compare_f_test(R.glm(yb, X[:, :1], family="binomial"))
    with pytest.raises(ValueError, match="same data"):
        gb.compare_lr_test(R.glm(yb, X[:, :1], family="binomial", link="probit"))


def test_errors():
    y, X = _problem(seed=22)
    r = R.ols(y, X)
    S = np.eye(60) * 0.5 + 0.5
    for call, err, msg in (
        (lambda: r.t_test([0, 1, 0]), ValueError, "t_test(): 'R' must have 4 columns like 'params', not 3"),
        (
            lambda: r.t_test([[0, 1, 0, 0]], [1, 2]),
            ValueError,
            "t_test(): 'q' must have 1 value like the rows of 'R', not 2",
        ),
        (lambda: r.wald_test([[0, np.inf, 0, 0]]), ValueError, "wald_test(): 'R' must be finite"),
        (lambda: r.t_test(np.zeros((0, 4))), ValueError, "t_test(): 'R' must have at least one row"),
        (lambda: r.goldfeld_quandt(split=0), ValueError, "goldfeld_quandt(): 'split' must be >= 1 rows"),
        (
            lambda: r.goldfeld_quandt(alternative="increasing"),
            ValueError,
            "'alternative' must be 'greater', 'less' or 'two-sided'",
        ),
        (lambda: r.goldfeld_quandt(order_by="z"), ValueError, "'order_by' names a column 'z'"),
        (lambda: R.gls(y, X, sigma=S).goldfeld_quandt(), ValueError, "goldfeld_quandt(): takes an ols or wls result"),
        (
            lambda: r.reset_test(power=3),
            TypeError,
            "reset_test(): 'power' must be a sequence of integers >= 2 like (2, 3), not int",
        ),
        (lambda: r.reset_test(power=(1, 2)), ValueError, "reset_test(): each power must be an integer >= 2, not 1"),
        (lambda: r.reset_test(power=(2, 2)), ValueError, "reset_test(): the powers must be distinct"),
        (lambda: r.reset_test(kind="exog"), ValueError, "'kind' must be 'fitted' or 'regressor'"),
        (lambda: R.anova(r, typ=2), ValueError, "anova(): 'typ' must be 1 or 3, not 2"),
        (lambda: R.anova(r, typ="I"), TypeError, "anova(): 'typ' must be 1 or 3, not str"),
        (lambda: R.anova(r, terms=[[0], [1]]), ValueError, "'terms' must put every column of 'X' in a term"),
        (lambda: R.anova(r, terms=[[0, 1], [1, 2]]), ValueError, "is in more than one term"),
        (lambda: R.anova(r, R.ols(y, X[:, :1]), typ=1), ValueError, "'typ' and 'terms' are for one model's table"),
        (
            lambda: R.anova(R.ols(y, X, cov_type="HC1")),
            ValueError,
            "Type I sums of squares do not use the robust covariance",
        ),
        (
            lambda: R.anova(R.ols(np.stack([y, y]), X)),
            ValueError,
            "anova(): takes one regression, not a batch of shape (2,)",
        ),
        (lambda: R.anova(R.rlm(y, X)), TypeError, "anova(): takes the results of ols, wls, gls or glm, not"),
        (lambda: r.t_test(), TypeError, "t_test(): missing 1 required positional argument: 'R'"),
        (lambda: r.white_test(True), TypeError, "white_test(): takes 0 positional arguments but 1 was given"),
    ):
        with pytest.raises(err) as e:
            call()
        assert msg in str(e.value), str(e.value)
