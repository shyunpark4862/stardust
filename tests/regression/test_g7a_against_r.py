"""G7a against R (tests/regression/ref/g7a_r.json, made once by tools/gen_regression_g7a_r.py
with R 4.6.2, sandwich 3.1.3, lmtest 0.9.40 and car 3.1.5; R is not needed here).
docs/design/regression.md §3.8 lists where sd and R differ and why: the GLM's Type III is the
Wald test (car's Anova(test.statistic = "Wald"); drop1 of a glm is the likelihood ratio), the
Gamma likelihood ratio (R's logLik takes the deviance / n scale), the White test is bptest with
White's auxiliary formula."""

import json
import pathlib

import numpy as np
import pytest

import stardust as sd

R = sd.regression
REF = json.loads((pathlib.Path(__file__).parent / "ref" / "g7a_r.json").read_text())


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def r_(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def rel(got, ref):
    got, ref = a(got).ravel(), r_(ref).ravel()
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    m = ~np.isnan(ref)
    d = np.abs(got[m] - ref[m])
    return float(np.max(np.where(d == 0, 0.0, d / np.where(ref[m] == 0, 1.0, np.abs(ref[m]))), initial=0.0))


def lm_fit(p, cols=None):
    X = np.asarray(p["X"])
    if cols is not None:
        X = X[:, :cols]
    icpt = p["meta"][0] == 1
    w = np.asarray(p["w"])
    if np.all(w == 1):
        return R.ols(p["y"], X, intercept=icpt)
    return R.wls(p["y"], X, weights=w, intercept=icpt)


@pytest.mark.parametrize("name", sorted(REF["lm"]))
def test_lm_tests(name):
    p = REF["lm"][name]
    ref = p["R"]
    big = lm_fit(p)
    small = lm_fit(p, cols=int(p["meta"][1]))
    Rm, q = np.asarray(p["hyp"]), np.asarray(p["q"])
    t = big.t_test(Rm, q)
    assert rel(t.effect, ref["t_effect"]) <= 1e-12
    assert rel(t.sd, ref["t_sd"]) <= 1e-12
    assert rel(t.tvalue, ref["t_tvalue"]) <= 1e-12
    assert rel(t.pvalue, ref["t_pvalue"]) <= 1e-11
    w = big.wald_test(Rm, q)
    assert rel([w.statistic, w.pvalue, w.df_num], ref["lh_F"] + ref["lh_p"] + ref["lh_df"]) <= 1e-11
    big3 = R.wls(p["y"], p["X"], weights=p["w"], intercept=p["meta"][0] == 1, cov_type="HC3")
    w = big3.wald_test(Rm, q)
    assert rel([w.statistic, w.pvalue], ref["lh_hc3_F"] + ref["lh_hc3_p"]) <= 1e-11
    c = big.compare_f_test(small)
    assert rel([c.fvalue, c.pvalue], ref["anova2_F"] + ref["anova2_p"]) <= 1e-11
    assert rel([c.fvalue, c.pvalue], ref["waldtest_F"] + ref["waldtest_p"]) <= 1e-11
    c = big.compare_lr_test(small)
    assert rel([c.statistic, c.pvalue], ref["lrtest"] + ref["lrtest_p"]) <= 1e-11
    tab = R.anova(small, big)
    assert rel(tab.ss_diff[1:], ref["anova2_ssdiff"]) <= 1e-12
    assert rel(tab.pvalue[1:], ref["anova2_p"]) <= 1e-11
    tab = R.anova(small, big, test="Chisq")
    assert rel(tab.pvalue[1:], ref["anova2_chisq_p"]) <= 1e-11
    terms = [[0], [1], [2, 3]] if name == "factor" else None
    t1 = R.anova(big, terms=terms)
    assert rel(t1.df, ref["anova1_df"]) == 0
    assert rel(t1.sum_sq, ref["anova1_ss"]) <= 1e-12
    assert rel(t1.statistic[:-1], ref["anova1_F"][:-1]) <= 1e-11
    assert rel(t1.pvalue[:-1], ref["anova1_p"][:-1]) <= 1e-11
    t3 = R.anova(big, typ=3, terms=terms)
    assert rel(t3.df[:-1], ref["drop1_df"]) == 0
    assert rel(t3.sum_sq[:-1], ref["drop1_ss"]) <= 1e-11
    assert rel(t3.statistic[:-1], ref["drop1_F"]) <= 1e-11
    assert rel(t3.pvalue[:-1], ref["drop1_p"]) <= 1e-11
    if "anova3_F" in ref:
        assert rel(t3.statistic[:-1], ref["anova3_F"]) <= 1e-11
        assert rel(t3.sum_sq[:-1], ref["anova3_ss"]) <= 1e-11
        t3h = R.anova(big3, typ=3, terms=terms)
        assert rel(t3h.statistic[:-1], ref["anova3_hc3_F"]) <= 1e-11
        assert rel(t3h.pvalue[:-1], ref["anova3_hc3_p"]) <= 1e-11
    if "vif" in ref:
        assert rel(big.vif[1:], ref["vif"]) <= 1e-12
    assert rel(big.dffits, ref["dffits"]) <= 1e-11
    assert rel(big.dfbetas, ref["dfbetas"]) <= 1e-10
    if "gq_greater" in ref:
        for alt, key in (("greater", "gq_greater"), ("less", "gq_less"), ("two-sided", "gq_two.sided")):
            g = big.goldfeld_quandt(alternative=alt)
            assert rel([g.fvalue, g.pvalue, g.df_num, g.df_denom], ref[key]) <= 1e-11, alt
        # R's fraction f of a point at 0.5: the first floor((1/2 - f/2) n) rows, the last from
        # ceiling((1/2 + f/2) n + 0.01)
        n = len(p["y"])
        n1 = int(np.floor((0.5 - 0.1) * n))
        n2 = int(np.ceil((0.5 + 0.1) * n + 0.01))
        g = big.goldfeld_quandt(split=n1, drop=n2 - 1 - n1, order_by=int(p["meta"][3]) - 1)
        assert rel([g.fvalue, g.pvalue, g.df_num, g.df_denom], ref["gq_frac_order"]) <= 1e-11
        rs = big.reset_test()
        assert rel([rs.fvalue, rs.pvalue, rs.df_num, rs.df_denom], ref["reset_fitted"]) <= 1e-10
        if "reset_regressor" in ref:
            rs = big.reset_test(kind="regressor")
            assert rel([rs.fvalue, rs.pvalue, rs.df_num, rs.df_denom], ref["reset_regressor"]) <= 1e-10
        wh = big.white_test()
        assert rel([wh.lm, wh.lm_pvalue, wh.df], ref["white"]) <= 1e-11
        wh = big.white_test(cross_terms=False)
        assert rel([wh.lm, wh.lm_pvalue, wh.df], ref["white_nocross"]) <= 1e-11


def glm_fit(p, cols=None):
    X = np.asarray(p["X"])
    if cols is not None:
        X = X[:, :cols]
    return R.glm(p["y"], X, family=p["family"], link=p["link"])


@pytest.mark.parametrize("name", sorted(REF["glm"]))
def test_glm_tests(name):
    p = REF["glm"][name]
    ref = p["R"]
    big = glm_fit(p)
    small = glm_fit(p, cols=1)
    fixed = p["family"] in ("binomial", "poisson")
    t1 = R.anova(big)
    assert rel(t1.df[:-1], ref["anova1_df"]) == 0
    # anova.glm refits the sequential models from scratch: R's IRLS stops on the deviance's
    # relative change, so these hold R's convergence error
    assert rel(t1.sum_sq[:-1], ref["anova1_dev"]) <= 1e-8
    assert rel(t1.pvalue[:-1], ref["anova1_p"]) <= 1e-7
    if not fixed:
        assert rel(t1.statistic[:-1], ref["anova1_F"]) <= 1e-8
    tab = R.anova(small, big)
    assert rel(tab.ss_diff[1:], ref["anova2_dev"]) <= 1e-9
    assert rel(tab.pvalue[1:], ref["anova2_p"]) <= 1e-8
    t3 = R.anova(big, typ=3)
    assert rel(t3.statistic[:-1] * (1 if fixed else t3.df[:-1]), ref["anova3_chisq"]) <= 1e-9
    if fixed:
        assert rel(t3.pvalue[:-1], ref["anova3_p"]) <= 1e-8
    w = big.wald_test(np.asarray(p["hyp"]))
    if fixed:
        assert rel([w.statistic, w.pvalue], ref["lh_chisq"] + ref["lh_p"]) <= 1e-9
    else:
        assert rel([w.statistic, w.pvalue], ref["lh_F"] + ref["lh_F_p"]) <= 1e-9
    c = big.compare_lr_test(small)
    if p["family"] == "gamma":
        # R's logLik of a Gamma glm takes the scale deviance / n; sd's llf (statsmodels') the
        # Pearson scale, so the likelihood ratio differs by definition ([R과 다름])
        assert rel([c.statistic], ref["lrtest"]) > 1e-3
    else:
        assert rel([c.statistic, c.pvalue], ref["lrtest"] + ref["lrtest_p"]) <= 1e-9
    assert rel(big.leverage, ref["hat"]) <= 1e-9
    assert rel(big.cooks_distance, ref["cooks"]) <= 1e-9
    # R 4.6's influence measures: the deleted Pearson scale for an estimated dispersion, the
    # model's sigma for a fixed one
    assert rel(big.dffits, ref["dffits"]) <= 1e-12
    assert rel(big.dfbetas, ref["dfbetas"]) <= 1e-11
    # drop1 of a glm is the likelihood ratio, sd's Type III the Wald test: they differ
    if fixed:
        assert rel(t3.statistic[:-1], ref["drop1_lr"]) > 1e-4
