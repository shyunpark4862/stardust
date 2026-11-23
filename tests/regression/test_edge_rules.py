"""Edge rules of the linear models, their tests and the selections (docs/design/regression.md
G196, G197): the F test of a perfect fit, the auxiliary regressions without degrees of freedom,
anova's scalar terms and the GLM's 'Residual' mean square, the rows of weight 0, the public
element types of the selections, the batch tests' shared number of observations, and the
lasso's exact copies on a path and in a single fit (R178)."""

import math
import struct

import numpy as np
import pytest

import stardust as sd

R = sd.regression


def a(x):
    return np.asarray(x, dtype=float)


def bits(v):
    return [struct.pack("<d", float(t)) for t in a(v).reshape(-1)]


# ---- R179: a perfect fit's F is NaN, as its t and p (G118)


def test_perfect_fit_f_is_nan():
    x = [0.0, 1, 2, 3, 4, 5]
    y = [1.0, 3, 5, 7, 9, 11]
    for r in (R.ols(y, x), R.ols(y, x, cov_type="HC1"), R.wls(y, x, weights=[1.0, 2, 1, 2, 1, 2])):
        assert math.isnan(r.fvalue) and math.isnan(r.f_pvalue)
        assert np.all(np.isnan(a(r.tvalues)))
    s = R.ols(y, x).summary()
    assert "F-statistic:                       nan" in s and "Prob (F-statistic):                nan" in s
    # a batch: only the perfect lane
    b = R.ols(np.array([y, [1.0, 3, 5, 7, 9, 12]]), x)
    f, p = a(b.fvalue), a(b.f_pvalue)
    assert math.isnan(f[0]) and math.isnan(p[0]) and math.isfinite(f[1]) and 0.0 < p[1] < 1.0
    # an imperfect fit keeps its F
    r = R.ols([1.0, 3, 5, 7, 9, 12], x)
    assert r.fvalue == float(f[1]) and r.f_pvalue == float(p[1])


# ---- R192: an auxiliary regression with df_resid 0 has no LM statistic (G95)


def test_auxiliary_without_degrees_of_freedom_is_nan():
    N = sd.stats.Normal()
    rng = sd.random.default_rng(3)
    Xm = N.sample((15, 5), rng=rng)
    ym = N.sample(15, rng=rng)
    w = R.ols(ym, Xm).white_test()
    assert w.df_denom == 0.0
    assert all(math.isnan(v) for v in (w.lm, w.lm_pvalue, w.fvalue, w.f_pvalue))
    # Breusch-Pagan's auxiliary of a WLS fit has the whitened constant too: 3 columns on 3 rows
    z = R.wls([1.0, 3, 2], [1.0, 2, 3], weights=[1.0, 2, 4])
    assert z.df_resid == 1.0
    for st in (True, False):
        bp = z.breusch_pagan(studentize=st)
        assert all(math.isnan(v) for v in bp)
    # a batch: only the lane without degrees of freedom
    Xn = np.asarray(Xm.tolist())
    X2 = np.stack([Xn, Xn[:, [0, 1, 0, 1, 0]]])  # the second lane's repeated columns leave rows over
    Y2 = np.stack([np.asarray(ym.tolist())] * 2)
    wb = R.ols(Y2, X2).white_test()
    assert math.isnan(a(wb.lm)[0]) and math.isfinite(a(wb.lm)[1])
    # degrees of freedom left: the statistic as before
    r = R.ols(N.sample(40, rng=rng), N.sample((40, 2), rng=rng)).white_test()
    assert math.isfinite(r.lm) and 0.0 < r.lm_pvalue < 1.0


# ---- R190: anova's and stepwise's terms: a scalar is one column


@pytest.mark.parametrize("fit", ["anova", "stepwise"])
def test_scalar_terms(fit):
    x = sd.asarray([1.0, 2, 3, 4, 5, 6, 7, 8])
    z = sd.asarray([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.9, -0.3])
    y = x * 2 + z
    X = sd.stack([x, z], axis=-1)
    r = R.ols(y, X)

    def call(t):
        return R.anova(r, terms=t) if fit == "anova" else R.stepwise(y, X, terms=t)

    msg = rf"{fit}\(\): 'terms' must give columns as positions \(int\) or names \(str\), not "  # one shape
    for t, name in (([1.5, 0], "float"), ([True, 1], "bool"), ([b"x1", 0], "bytes"), ([[1.5], 0], "float")):
        with pytest.raises(TypeError, match=msg + name + "$"):
            call(t)
    # numpy's integers are positions, an array of positions is a term of several columns
    if fit == "anova":
        ref = a(R.anova(r, terms=[0, 1]).sum_sq).tolist()
        assert a(call([np.int64(0), np.int64(1)]).sum_sq).tolist() == ref
        assert a(call([np.array(0), (1,)]).sum_sq).tolist() == ref
        both = a(R.anova(r, terms=[[0, 1]]).sum_sq).tolist()
        assert a(call([np.array([0, 1])]).sum_sq).tolist() == both
    else:
        ref = call([0, 1]).selected
        assert call([np.int64(0), np.int64(1)]).selected == ref
        assert call([np.array([0, 1])]).selected == call([[0, 1]]).selected


# ---- R191: the 'Residual' row's mean square is the scale


def test_anova_residual_mean_square_is_the_scale():
    rng = sd.random.default_rng(5)
    x1 = sd.stats.Uniform(a=0.0, b=2.0).sample(60, rng=rng)
    x2 = sd.stats.Uniform(a=0.0, b=2.0).sample(60, rng=rng)
    cost = sd.stats.Gamma(a=4.0).sample(60, rng=rng) * sd.exp(0.5 + 0.8 * x1) / 4.0
    X = sd.stack([x1, x2], axis=-1)
    for gm in (R.glm(cost, X, family="gamma", link="log"), R.glm(sd.round(cost * 3.0), X, family="poisson")):
        for typ in (1, 3):
            t = R.anova(gm, typ=typ)
            assert float(t.mean_sq[-1]) == gm.scale
            assert float(t.df[-1]) == gm.df_resid and float(t.sum_sq[-1]) == gm.deviance
    gm = R.glm(cost, X, family="gamma", link="log")
    t = R.anova(gm, test="F")
    # F = mean_sq of the term / mean_sq of 'Residual'
    assert abs(float(t.mean_sq[0]) / float(t.mean_sq[-1]) - float(t.statistic[0])) <= 1e-15 * float(t.statistic[0])
    # the linear model: ssr / df_resid, the scale
    r = R.ols(cost, X)
    t = R.anova(r)
    assert float(t.mean_sq[-1]) == float(t.sum_sq[-1]) / float(t.df[-1])
    assert abs(float(t.mean_sq[-1]) - r.scale) <= 4e-16 * r.scale


# ---- R194: the rows of weight 0 are +0, not -0


def test_weight_zero_rows_are_positive_zero():
    xw = sd.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    yw = sd.asarray([2.3, 3.8, 6.4, 7.5, 11.2, 11.1, 15.9, 14.6])
    z = R.wls(yw, xw, weights=sd.asarray([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
    assert float(z.resid[3]) < 0.0  # the residual itself is negative
    for f in (
        z.wresid,
        z.dffits,
        z.resid_studentized_internal,
        z.resid_studentized_external,
        z.cooks_distance,
        z.leverage,
    ):
        v = float(a(f)[3])
        assert v == 0.0 and math.copysign(1.0, v) == 1.0
    assert all(math.copysign(1.0, v) == 1.0 and v == 0.0 for v in a(z.dfbetas)[3])
    # the other rows: sqrt(w) e, unchanged
    assert bits(a(z.wresid)[[0, 1, 2, 4, 5, 6, 7]]) == bits(a(z.resid)[[0, 1, 2, 4, 5, 6, 7]])


# ---- R195: the selections' element types are public (G197)


def test_selection_element_types_are_public():
    for name in ("SelectionStep", "SkippedFit", "SkippedSubset"):
        assert name in R.__all__ and hasattr(R, name)
    r = R.stepwise([1.0, 2, 3, 5, 4, 6], [[1.0], [2], [3], [4], [5], [6]])
    assert isinstance(r.steps[0], R.SelectionStep) and r.steps[0]._fields == ("action", "term", "criterion")
    assert R.SkippedFit._fields == ("step", "action", "term", "reason")
    assert R.SkippedSubset._fields == ("columns", "reason")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 3))
    y = (X[:, 0] > 0).astype(float)  # x1 separates the classes
    b = R.best_subset(y, X, model="glm", family="binomial")
    assert b.skipped and all(isinstance(s, R.SkippedSubset) for s in b.skipped)


# ---- R246: a batch test's summary has the number of observations its lanes share


def test_batch_test_summary_has_the_shared_nobs():
    rng = sd.random.default_rng(1)
    X = sd.stats.Normal().sample((40, 3), rng=rng)
    y = 1.0 + X[:, 0] + sd.stats.Normal().sample(40, rng=rng)
    one = R.ols(y, X).t_test([0, 1, 0, 0]).summary()
    b = R.ols(sd.stack([y, 2 * y]), X)
    obs = "No. Observations:                   40"
    assert obs in one
    for s in (
        b.t_test([0, 1, 0, 0]).summary(),
        b.wald_test([[0, 1, 0, 0], [0, 0, 1, 0]]).summary(),
        b.white_test().summary(),
        b.breusch_pagan().summary(),
        b.jarque_bera().summary(),
        b.compare_f_test(R.ols(sd.stack([y, 2 * y]), X[:, :1])).summary(),
    ):
        assert obs in s, s
    # lanes with different numbers of observations (weights 0) and degrees of freedom: no
    # count, and the descriptors fill both header columns
    w = sd.stack([sd.ones(40), sd.where(sd.arange(40) < 3, 0.0, 1.0)])
    t = R.wls(sd.stack([y, 2 * y]), X, weights=w).t_test([0, 1, 0, 0]).summary()
    assert "No. Observations:" not in t
    head = [line for line in t.splitlines() if line.startswith("Batch shape:")]
    assert len(head) == 1 and "No. Constraints:" in head[0]


# ---- R178: the lasso's exact copies, a path and the single fits (G116)


def _copies_data():
    ra = sd.random.default_rng(1)
    Za = sd.stats.Normal().sample((40, 3), rng=ra)
    Xa = sd.stack([Za[:, 0], Za[:, 0] + 0.05 * Za[:, 1], Za[:, 2]], axis=1)
    ya = 1.0 + Xa[:, 0] + Xa[:, 1] + 0.5 * Xa[:, 2] + sd.stats.Normal().sample(40, rng=ra)
    return ya, Xa


GRID = [0.5 * 10 ** (-3 * i / 24) for i in range(25)]


def test_lasso_copies_path_equals_single_fits():
    ya, Xa = _copies_data()
    # the descent may leave the first copy at 0 and give the later one the total: the polish
    # moves it to the first, whatever the start
    Xc = sd.stack([Xa[:, 0], Xa[:, 1], Xa[:, 2], Xa[:, 0]], axis=1)
    for X in (Xc, Xa, sd.stack([Xa[:, 0], Xa[:, 0], Xa[:, 2]], axis=1)):
        p = R.lasso(ya, X, alpha=GRID)
        for i, al in enumerate(GRID):
            s = R.lasso(ya, X, alpha=al)
            assert bits(a(p.params)[i]) == bits(s.params), (i, al)
            assert s.converged and bool(a(p.converged)[i])
    p = R.lasso(ya, Xc, alpha=GRID)
    assert np.all(a(p.params)[:, 4] == 0.0)  # the later copy
    # any order of the alphas
    rev = R.lasso(ya, Xc, alpha=GRID[::-1])
    assert bits(a(rev.params)[::-1]) == bits(p.params)


def test_lasso_copies_thread_independent():
    ya, Xa = _copies_data()
    Xc = sd.stack([Xa[:, 0], Xa[:, 1], Xa[:, 2], Xa[:, 0]], axis=1)
    Y = sd.stack([ya, 2.0 * ya, ya - Xa[:, 2]])
    old = sd.get_num_threads()
    out = []
    try:
        for th in (1, 10):
            sd.set_num_threads(th)
            out.append(bits(R.lasso(Y, Xc, alpha=GRID).params))
    finally:
        sd.set_num_threads(old)
    assert out[0] == out[1]
    # a batch's lane is the regression alone
    lane = R.lasso(Y[1], Xc, alpha=GRID)
    assert bits(a(R.lasso(Y, Xc, alpha=GRID).params)[1]) == bits(lane.params)


def test_elastic_net_copies_unchanged_rule():
    # an L2 term: the copies take equal shares (the unique solution), no merging
    ya, Xa = _copies_data()
    X = sd.stack([Xa[:, 0], Xa[:, 2], Xa[:, 0]], axis=1)
    e = R.elastic_net(ya, X, alpha=0.05, l1_ratio=0.5)
    assert e.converged and abs(float(e.params[1]) - float(e.params[3])) <= 1e-12 * abs(float(e.params[1]))
