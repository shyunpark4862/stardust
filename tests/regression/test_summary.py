"""summary() of every sd.regression result (G189-G193): the models in the regression box (a
batch one row per regression), anova's table in the box, the tests on fitted models (t_test one
row per row of R), the selections, the penalized fits (coefficients only) and the slopes; the
one-line reprs."""

import numpy as np
import pytest

import stardust as sd

R = sd.regression


def _data(seed=5, n=40):
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, 3))
    y = 1.0 + X @ np.array([1.0, -0.5, 0.0]) + r.standard_normal(n)
    return y, X


def _lines(text):
    return text.splitlines()


def _row(text, stub):
    """The cells of the table row whose stub is `stub`."""
    for line in _lines(text):
        if line.startswith(stub + " ") or line == stub:
            return line[len(stub) :].split()
    raise AssertionError(f"no row {stub!r} in\n{text}")


def test_model_summary_is_the_regression_box():
    y, X = _data()
    s = R.ols(y, X).summary()
    lines = _lines(s)
    assert lines[0].strip() == "OLS Regression Results" and lines[1] == "=" * 78
    assert "coef    std err          t      P>|t|     [0.025     0.975]" in s
    assert "Durbin-Watson:" in s and "Notes:" in s


def test_model_batch_one_row_per_regression():
    y, X = _data()
    Y = np.vstack([y, 2.0 * y, y + 1.0])
    r = R.ols(Y, X)
    s = r.summary()
    assert "Batch shape:                      (3,)" in s
    assert _row(s, "[1]")[0] == "40"  # nobs, then the fit statistics
    params = r.params
    lines = _lines(s)
    top = next(k for k, line in enumerate(lines) if line.startswith("params"))
    for i in range(3):
        one = R.ols(Y[i], X)
        assert f"{float(one.rsquared):.3f}" in _row(s, f"[{i}]")
        cells = lines[top + 2 + i].split()
        assert cells[0] == f"[{i}]"
        assert [float(v) for v in cells[1:]] == pytest.approx(params[i].tolist(), abs=5e-5)
    assert R.ols(Y, X).summary(alpha=0.1) == s  # alpha is checked, a batch shows no interval
    with pytest.raises(ValueError, match=r"^summary\(\): 'alpha' must be between 0 and 1"):
        r.summary(alpha=1.0)


def test_long_batch_is_cut():
    y, X = _data()
    Y = np.vstack([y + k for k in range(30)])
    s = R.ols(Y, X).summary()
    assert "[4]" in s and "[25]" in s and "[5] " not in s and "[24]" not in s
    assert sum(1 for line in _lines(s) if line.startswith("...")) == 2  # both tables


def test_glm_rlm_quantreg_batches():
    y, X = _data()
    Y = np.vstack([y, y[::-1]])
    g = R.glm(Y, X, family="gaussian").summary()
    assert "Model Family:                 Gaussian" in g and "converged" in g and "[1]" in g
    rl = R.rlm(Y, X).summary()
    assert "Norm:" in rl and "scale" in rl
    q = R.quantreg(Y, X, q=[0.25, 0.75]).summary()
    assert "No. Quantiles:                       2" in q and _row(q, "[1, 1]")[0] == "0.75"


def test_anova_box():
    y, X = _data()
    big, small = R.ols(y, X), R.ols(y, X[:, :1])
    s = R.anova(big).summary()
    lines = _lines(s)
    assert lines[0].strip() == "Analysis of Variance Table (Type I)"
    assert "Model:                             OLS  No. Observations:                   40" in s
    assert "Type:                                I  Df Residuals:                       36" in s
    assert "df     sum_sq    mean_sq          F     Pr(>F)" in s
    assert _row(s, "Residual")[0] == "36" and len(_row(s, "Residual")) == 3  # the test's cells blank
    assert "Type:                              III" in R.anova(big, typ=3).summary()
    m = R.anova(small, big).summary()
    assert "No. Models:                          2" in m and "df_resid        ssr    df_diff    ss_diff" in m
    assert len(_row(m, "1")) == 2  # the first model's comparison is blank
    yb = (y > 1.0).astype(float)
    c = R.anova(R.glm(yb, X, family="binomial")).summary()
    assert "Chisq   Pr(>Chi)" in c and "Test:                            Chisq" in c


def test_t_test_rows_like_statsmodels():
    y, X = _data()
    r = R.ols(y, X)
    t = r.t_test([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    s = t.summary()
    assert _lines(s)[0].strip() == "Test for Constraints"
    assert "Df:                                 36  No. Constraints:                     2" in s
    c0 = _row(s, "c0")
    assert float(c0[0]) == pytest.approx(float(r.params[1]), abs=5e-5)
    assert float(c0[2]) == pytest.approx(float(r.tvalues[1]), abs=5e-4)
    ci = t.conf_int(alpha=0.1)
    c1 = _row(t.summary(alpha=0.1), "c1")
    assert [float(v) for v in c1[4:]] == pytest.approx(ci[1].tolist(), abs=5e-4)
    assert "[0.050     0.950]" in t.summary(alpha=0.1)
    g = R.glm((y > 1.0).astype(float), X, family="binomial").t_test([0.0, 1.0, 0.0, 0.0]).summary()
    assert "P>|z|" in g and "Df:                                inf" in g
    b = R.ols(np.vstack([y, y]), X).t_test([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).summary()
    assert "[1] c1" in b and "Batch shape:" in b


def test_test_results_headers():
    y, X = _data()
    r = R.ols(y, X)
    small = R.ols(y, X[:, :1])
    w = r.wald_test([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).summary()
    assert "Wald test" in w and "Distribution:                        F" in w and "Df Num:" in w
    for text, title in (
        (r.compare_f_test(small).summary(), "F test of nested models"),
        (r.compare_lr_test(small).summary(), "Likelihood ratio test of nested models"),
        (r.white_test().summary(), "White test for heteroscedasticity"),
        (r.goldfeld_quandt().summary(), "Goldfeld-Quandt test"),
        (r.reset_test().summary(), "Ramsey RESET test"),
        (r.breusch_pagan().summary(), "Breusch-Pagan test for heteroscedasticity"),
        (r.jarque_bera().summary(), "Jarque-Bera test"),
    ):
        assert _lines(text)[0].strip() == title
        assert "No. Observations:                   40" in text
    assert "Alternative:                   greater" in r.goldfeld_quandt().summary()
    bp = R.ols(np.vstack([y, y]), X).breusch_pagan().summary()
    assert "lm  lm_pvalue     fvalue   f_pvalue" in bp and "[1]" in bp


def test_selection_summaries():
    y, X = _data(n=60)
    st = R.stepwise(y, X)
    s = st.summary()
    assert _lines(s)[0].strip() == "Stepwise Selection Results"
    assert _row(s, "0")[0] == "start" and len(_lines(s)) > 10
    assert "Selected: " + ", ".join(st.selected_names) + "." in s
    b = R.best_subset(y, X)
    t = b.summary()
    assert "size              aic  columns" in t and "Chosen Size:" in t
    assert sum(1 for line in _lines(t) if " * " in line or line.split()[:2][-1:] == ["*"]) == 1


def test_penalized_summaries():
    y, X = _data()
    s = R.ridge(y, X, alpha=0.5).summary()
    assert "Ridge Regression Results" in s and "Alpha:                             0.5" in s
    assert "std err" not in s and "standard errors" in s
    assert float(_row(s, "x1")[0]) == pytest.approx(float(R.ridge(y, X, alpha=0.5).params[1]), abs=5e-5)
    e = R.elastic_net(y, X, alpha=0.1, l1_ratio=0.3).summary()
    assert "L1 Ratio:                          0.3" in e and "Converged:" in e
    p = R.lasso(y, X, alpha=[1.0, 0.1]).summary()
    assert "No. Alphas:                          2" in p and _row(p, "[1]")[0] == "0.1"
    cv = R.lasso_cv(y, X, rng=sd.random.default_rng(1)).summary()
    assert "Alpha (min CV):" in cv and "Alpha (1 SE):" in cv
    g = R.penalized_glm((y > 1.0).astype(float), X, family="binomial", alpha=0.05).summary()
    assert "Deviance Ratio:" in g and "Model Family:                 Binomial" in g


def test_slopes_summaries():
    r = np.random.default_rng(2)
    x = np.sort(r.uniform(0, 10, 25))
    y = 2.0 - 0.5 * x + r.standard_normal(25)
    t = R.theilslopes(y, x, alpha=0.1)
    s = t.summary()
    assert "[0.050     0.950]" in s and "Method:                       separate" in s
    assert [float(v) for v in _row(s, "slope")] == pytest.approx([t.slope, t.low_slope, t.high_slope], abs=5e-4)
    assert len(_row(s, "intercept")) == 1  # Sen's interval is the slope's only
    sg = R.siegelslopes(y, x).summary()
    assert "Siegel Repeated Medians Regression Results" in sg and "[0.025" not in sg
    b = R.theilslopes(np.vstack([y, y]), x).summary()
    assert "slope  intercept     [0.025     0.975]" in b
    assert isinstance(t, tuple) and t._fields == ("slope", "intercept", "low_slope", "high_slope")


def test_reprs_show_the_fields():
    y, X = _data()
    t = R.ols(y, X).t_test([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    text = repr(t)
    assert text.startswith("TTestResult(effect=[") and "df=36" in text and "\n" not in text
    one = repr(R.ols(y, X).t_test([0.0, 1.0, 0.0, 0.0]))
    assert one.startswith("TTestResult(effect=") and "pvalue=" in one
