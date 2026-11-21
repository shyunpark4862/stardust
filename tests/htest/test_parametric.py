"""sd.htest's tests of means and variances (milestone H1; docs/api/htest.md §2, docs/design/htest.md
§3.2, §4): the accuracy contract against the Arb and exact references (tests/htest/ref), batches
lane by lane, nan_policy, the degenerate inputs (H11, H20 and the H1 decisions), the result
objects, the messages, and the agreement with sd.regression and with scipy's definitions."""

import math

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

from . import problems as P
from . import refdata as R

FAMILIES = R.H1
H1 = (
    "ttest_1samp",
    "ttest_rel",
    "ttest_ind",
    "ttest_ind_from_stats",
    "f_oneway",
    "alexandergovern",
    "bartlett",
    "levene",
    "fligner",
)
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]


def call(p, **extra):
    return getattr(sd.htest, p.func)(*p.args, **{**p.kwargs, **extra})


# ---- the accuracy contract (docs/design/htest.md §4) --------------------------------------


@pytest.mark.parametrize(("fam", "pid"), CASES, ids=[pid for _, pid in CASES])
def test_contract_against_the_references(fam, pid):
    """statistic and df to 1e-14 relative, the p-value to 1e-14 max(1, kappa_p), the interval's
    ends to 1e-14 of the estimate's and the half width's sizes (the ends subtract them;
    tests/htest/refdata.py)."""
    r = call(P.PROBLEMS[pid])
    if "conf_int" in REFS[fam][pid]:
        assert r.conf_int().shape == (2,)
    for field, (e, c) in R.errors(fam, pid, r).items():
        assert c <= 1.0, (field, e)


def test_every_h1_function_has_references():
    funcs = {P.PROBLEMS[pid].func for _, pid in CASES}
    assert funcs == set(H1)


@pytest.mark.parametrize("pid", [p.id for p in P.PROBLEMS.values() if p.milestone == "H1" and "batch" in p.tags])
def test_batch_is_lane_by_lane(pid):
    """A batch (axis=-1) gives each lane's bits of the lane alone (axis=None)."""
    p = P.PROBLEMS[pid]
    r = call(p)
    kw = {k: v for k, v in p.kwargs.items() if k != "axis"}
    lanes = p.args[0].shape[0]
    assert r.statistic.shape == (lanes,)
    for i in range(lanes):
        args = [a[i] if isinstance(a, np.ndarray) and a.ndim == 2 else a for a in p.args]
        one = getattr(sd.htest, p.func)(*args, **kw)
        for f in type(one)._FIELDS:
            assert np.array_equal(float(getattr(r, f)[i]), getattr(one, f), equal_nan=True), (i, f)
        if hasattr(one, "conf_int"):
            assert np.array_equal(r.conf_int()[i], one.conf_int(), equal_nan=True)


def test_threads_do_not_change_the_bits():
    """The same bits on 1, 3 and 10 threads: a long lane (blocks on the pool) and a batch."""
    rng = np.random.default_rng(5)
    x, y, z = rng.normal(size=300_001), rng.normal(0.01, 1.1, 250_000), rng.normal(size=200_000)
    bx, by = rng.normal(size=(3000, 40)) + 1e6, rng.normal(size=(3000, 30)) + 1e6
    calls = [
        lambda: sd.htest.ttest_1samp(x, 0.001),
        lambda: sd.htest.ttest_rel(x[:250_000], y),
        lambda: sd.htest.ttest_ind(x, y, equal_var=False),
        lambda: sd.htest.ttest_ind(x, y, trim=0.1),
        lambda: sd.htest.f_oneway(x, y, z),
        lambda: sd.htest.f_oneway(x, y, z, equal_var=False),
        lambda: sd.htest.alexandergovern(x, y, z),
        lambda: sd.htest.bartlett(x, y, z),
        lambda: sd.htest.levene(x, y, z),
        lambda: sd.htest.fligner(x, y, z),
        lambda: sd.htest.ttest_ind(bx, by, axis=-1),
        lambda: sd.htest.levene(bx, by, axis=-1),
        lambda: sd.htest.fligner(bx, by, axis=-1),
    ]
    before = sd.get_num_threads()
    try:
        want = None
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            got = [np.asarray([r.statistic, r.pvalue]) for r in (c() for c in calls)]
            if want is None:
                want = got
            else:
                for g, w in zip(got, want):
                    assert np.array_equal(g, w, equal_nan=True)
    finally:
        sd.set_num_threads(before)


# ---- scipy's definitions where the data are benign -----------------------------------------


@pytest.mark.parametrize(
    ("func", "kw"),
    [
        ("ttest_ind", {}),
        ("ttest_ind", {"equal_var": False}),
        ("ttest_ind", {"trim": 0.2, "alternative": "less"}),
        ("ttest_ind", {"trim": 0.15, "equal_var": False, "alternative": "greater"}),
        ("ttest_rel", {}),
        ("ttest_1samp", {}),
    ],
)
def test_t_tests_are_scipys(func, kw):
    rng = np.random.default_rng(7)
    a, b = rng.normal(0.2, 1.0, 23), rng.normal(0.0, 1.7, 23)
    args = (a, 0.1) if func == "ttest_1samp" else (a, b)
    r, s = getattr(sd.htest, func)(*args, **kw), getattr(st, func)(*args, **kw)
    np.testing.assert_allclose([r.statistic, r.pvalue, r.df], [s.statistic, s.pvalue, s.df], rtol=1e-13)
    for alpha in (0.05, 0.2):
        ci = s.confidence_interval(1 - alpha)
        np.testing.assert_allclose(r.conf_int(alpha=alpha), [ci.low, ci.high], rtol=1e-13)


@pytest.mark.parametrize(
    ("func", "kw"),
    [
        ("f_oneway", {}),
        ("f_oneway", {"equal_var": False}),
        ("alexandergovern", {}),
        ("bartlett", {}),
        ("levene", {}),
        ("levene", {"center": "mean"}),
        ("levene", {"center": "trimmed", "proportiontocut": 0.2}),
        ("fligner", {}),
        ("fligner", {"center": "mean"}),
        ("fligner", {"center": "trimmed", "proportiontocut": 0.15}),
    ],
)
def test_k_sample_tests_are_scipys(func, kw):
    # odd sizes (an exact median), data without ties
    rng = np.random.default_rng(8)
    g = [rng.normal(0.0, 1.0, 17), rng.normal(0.3, 2.0, 21), rng.normal(-0.1, 0.7, 13)]
    r, s = getattr(sd.htest, func)(*g, **kw), getattr(st, func)(*g, **kw)
    np.testing.assert_allclose([r.statistic, r.pvalue], [s.statistic, s.pvalue], rtol=1e-13)


def test_ttest_ind_from_stats_is_ttest_ind():
    rng = np.random.default_rng(9)
    a, b = rng.normal(0.3, 1.0, 14), rng.normal(0.0, 2.0, 11)
    for ev in (True, False):
        r = sd.htest.ttest_ind(a, b, equal_var=ev)
        f = sd.htest.ttest_ind_from_stats(a.mean(), a.std(ddof=1), 14, b.mean(), b.std(ddof=1), 11, equal_var=ev)
        np.testing.assert_allclose([f.statistic, f.pvalue, f.df], [r.statistic, r.pvalue, r.df], rtol=1e-13)
        np.testing.assert_allclose(f.conf_int(alpha=0.1), r.conf_int(alpha=0.1), rtol=1e-13)


def test_ttest_ind_from_stats_broadcasts():
    r = sd.htest.ttest_ind_from_stats([0.1, 0.2, 0.3], 1.0, 10, 0.0, [[1.0], [2.0]], 12)
    assert r.statistic.shape == (2, 3) and r.conf_int().shape == (2, 3, 2)
    one = sd.htest.ttest_ind_from_stats(0.2, 1.0, 10, 0.0, 2.0, 12)
    assert float(r.statistic[1, 1]) == one.statistic and float(r.pvalue[1, 1]) == one.pvalue
    with pytest.raises(ValueError, match=r"ttest_ind_from_stats\(\): the arguments could not be broadcast together"):
        sd.htest.ttest_ind_from_stats([1.0, 2.0], 1.0, 10, [0.0, 1.0, 2.0], 1.0, 10)


# ---- the identities with other tests and with sd.regression (docs/api/htest.md §13) -------


def test_f_oneway_of_two_groups_is_the_t_test_squared():
    rng = np.random.default_rng(10)
    a, b = rng.normal(0.4, 1.0, 19) + 1e8, rng.normal(0.0, 1.0, 25) + 1e8
    f, t = sd.htest.f_oneway(a, b), sd.htest.ttest_ind(a, b)
    assert abs(f.statistic / t.statistic**2 - 1) <= 4e-15 and abs(f.pvalue / t.pvalue - 1) <= 1e-13
    assert (f.df_num, f.df_denom) == (1.0, t.df)
    w, tw = sd.htest.f_oneway(a, b, equal_var=False), sd.htest.ttest_ind(a, b, equal_var=False)
    assert abs(w.statistic / tw.statistic**2 - 1) <= 4e-15 and abs(w.df_denom / tw.df - 1) <= 4e-15
    assert abs(w.pvalue / tw.pvalue - 1) <= 1e-13


def test_f_oneway_agrees_with_the_group_dummies_ols():
    """The one-way ANOVA's F and p are the OLS of the group dummies' fvalue and f_pvalue within
    the contract (regression computes them from its QR; htest.f_oneway owns the test, §1.4)."""
    rng = np.random.default_rng(11)
    groups = [rng.normal(m, 1.0, n) + 1e3 for m, n in ((0.0, 15), (0.5, 22), (0.2, 18), (-0.3, 9))]
    y = np.concatenate(groups)
    X = np.zeros((y.size, len(groups) - 1))
    at = groups[0].size
    for j, g in enumerate(groups[1:]):
        X[at : at + g.size, j] = 1.0
        at += g.size
    o = sd.regression.ols(y, X)
    f = sd.htest.f_oneway(*groups)
    assert abs(f.statistic / o.fvalue - 1) <= 1e-13 and abs(f.pvalue / o.f_pvalue - 1) <= 1e-13
    t = sd.htest.ttest_ind(groups[1], groups[0])
    o2 = sd.regression.ols(np.concatenate(groups[:2]), np.r_[np.zeros(15), np.ones(22)])
    assert abs(t.statistic / float(o2.tvalues[1]) - 1) <= 1e-13 and abs(t.pvalue / float(o2.pvalues[1]) - 1) <= 1e-13


def test_p_values_are_sd_stats_distributions():
    """The p-value is the distribution's own method at the statistic: the bits of
    StudentT(df).ccdf, F(...).ccdf and ChiSquared(df).ccdf in Python."""
    rng = np.random.default_rng(12)
    a, b, c = rng.normal(0, 1, 30), rng.normal(0.6, 1.2, 25), rng.normal(0.1, 0.8, 20)
    t = sd.htest.ttest_ind(a, b, equal_var=False)
    assert t.pvalue == 2.0 * sd.stats.StudentT(df=t.df).ccdf(abs(t.statistic))
    assert sd.htest.ttest_ind(a, b, alternative="less").pvalue == sd.stats.StudentT(df=53.0).cdf(
        sd.htest.ttest_ind(a, b).statistic
    )
    f = sd.htest.f_oneway(a, b, c)
    assert f.pvalue == sd.stats.F(dfn=f.df_num, dfd=f.df_denom).ccdf(f.statistic)
    w = sd.htest.f_oneway(a, b, c, equal_var=False)
    assert w.pvalue == sd.stats.F(dfn=w.df_num, dfd=w.df_denom).ccdf(w.statistic)
    for fn in (sd.htest.bartlett, sd.htest.fligner, sd.htest.alexandergovern):
        r = fn(a, b, c)
        assert r.pvalue == sd.stats.ChiSquared(df=2.0).ccdf(r.statistic)
    lev = sd.htest.levene(a, b, c)
    assert lev.pvalue == sd.stats.F(dfn=2.0, dfd=72.0).ccdf(lev.statistic)


# ---- degenerate inputs (H11, H20, user decisions 2026-11-20) -------------------------------


def test_zero_standard_error_is_nan():
    """A zero standard error gives NaN t and p (scipy: inf and 0), df as usual, and the interval
    of zero width at the estimate."""
    r = sd.htest.ttest_1samp([1.0, 1.0, 1.0], 0.0)
    assert math.isnan(r.statistic) and math.isnan(r.pvalue) and r.df == 2.0
    assert r.conf_int().tolist() == [1.0, 1.0]
    r = sd.htest.ttest_ind([2.0, 2.0, 2.0], [3.0, 3.0])
    assert math.isnan(r.statistic) and math.isnan(r.pvalue) and r.df == 3.0
    assert r.conf_int().tolist() == [-1.0, -1.0]
    r = sd.htest.ttest_ind([2.0, 2.0, 2.0], [3.0, 3.0], equal_var=False)
    assert math.isnan(r.statistic) and math.isnan(r.df) and r.conf_int().tolist() == [-1.0, -1.0]
    r = sd.htest.ttest_ind_from_stats(1.0, 0.0, 5, 2.0, 0.0, 6)
    assert math.isnan(r.statistic) and math.isnan(r.pvalue) and r.df == 9.0
    # a constant sample by sd.stats skew's rule (H20): 1e8 plus rounding-sized differences
    r = sd.htest.ttest_rel(np.full(4, 1e8), np.full(4, 1e8))
    assert math.isnan(r.statistic)


def test_zero_within_group_variance_is_nan():
    for fn in (sd.htest.f_oneway, sd.htest.bartlett, sd.htest.alexandergovern, sd.htest.levene):
        r = fn([2.0, 2.0, 2.0], [3.0, 3.0], [5.0, 5.0, 5.0, 5.0])
        assert math.isnan(r.statistic) and math.isnan(r.pvalue), fn
    r = sd.htest.f_oneway([2.0, 2.0, 2.0], [3.0, 3.0], equal_var=False)
    assert math.isnan(r.statistic) and r.df_num == 1.0 and math.isnan(r.df_denom)
    # one constant group is enough for the tests that divide by each group's variance
    for fn in (sd.htest.bartlett, sd.htest.alexandergovern):
        assert math.isnan(fn([1.0, 2.0, 4.0], [3.0, 3.0, 3.0]).statistic)
    assert math.isnan(sd.htest.f_oneway([1.0, 2.0, 4.0], [3.0, 3.0, 3.0], equal_var=False).statistic)
    assert not math.isnan(sd.htest.f_oneway([1.0, 2.0, 4.0], [3.0, 3.0, 3.0]).statistic)


def test_too_few_observations():
    r = sd.htest.ttest_1samp([3.0], 0.0)  # a real scalar is a one-element sample too
    assert math.isnan(r.statistic) and r.df == 0.0 and math.isnan(r.conf_int()[0])
    s = sd.htest.ttest_1samp(3.0, 0.0)
    assert math.isnan(s.statistic) and s.df == 0.0
    r = sd.htest.ttest_1samp([], 0.0)
    assert math.isnan(r.statistic) and math.isnan(r.df)
    r = sd.htest.ttest_ind([0.7], [1.0, 2.0, 4.0])  # Student: a sample of one has variance 0 (scipy)
    s = st.ttest_ind([0.7], [1.0, 2.0, 4.0])
    assert abs(r.statistic / s.statistic - 1) < 1e-14 and r.df == 2.0
    assert math.isnan(sd.htest.ttest_ind([0.7], [1.0, 2.0, 4.0], equal_var=False).df)  # (scipy: 1.0)
    assert math.isnan(sd.htest.ttest_ind([1.0], [2.0]).statistic)
    assert math.isnan(sd.htest.f_oneway([1.0], [2.0]).statistic)
    assert math.isnan(sd.htest.f_oneway([1.0, 2.0], []).statistic)
    for fn in (sd.htest.bartlett, sd.htest.alexandergovern):
        assert math.isnan(fn([1.0, 2.0, 3.0], [4.0]).statistic)
    for fn in (sd.htest.levene, sd.htest.fligner):
        assert math.isnan(fn([1.0, 2.0, 3.0], []).statistic)
    # the degrees of freedom stay the sizes' (k - 1, N - k) where the statistic is NaN
    assert sd.htest.bartlett([1.0, 2.0, 3.0], [4.0]).df == 1.0
    lev = sd.htest.levene([1.0, 2.0, 3.0], [])
    assert (lev.df_num, lev.df_denom) == (1.0, 1.0)


def test_large_and_small_scales():
    """Every statistic is unchanged when the data are scaled by 1e300 or 1e-300 (the lanes are
    computed in a power-of-two unit; their squares would leave the doubles)."""
    rng = np.random.default_rng(13)
    a, b, c = rng.normal(0, 1, 20), rng.normal(0.5, 2, 15), rng.normal(1, 1, 12)
    tests = [
        lambda s: sd.htest.ttest_1samp(a * s, 0.1 * s),
        lambda s: sd.htest.ttest_rel(a[:15] * s, b * s),
        lambda s: sd.htest.ttest_ind(a * s, b * s),
        lambda s: sd.htest.ttest_ind(a * s, b * s, trim=0.1),
        lambda s: sd.htest.f_oneway(a * s, b * s, c * s),
        lambda s: sd.htest.f_oneway(a * s, b * s, c * s, equal_var=False),
        lambda s: sd.htest.alexandergovern(a * s, b * s, c * s),
        lambda s: sd.htest.bartlett(a * s, b * s, c * s),
        lambda s: sd.htest.levene(a * s, b * s, c * s),
        lambda s: sd.htest.fligner(a * s, b * s, c * s),
    ]
    for t in tests:
        base = t(1.0)
        for s in (2.0**990, 2.0**-990, 1e300, 1e-300):
            r = t(s)
            assert abs(r.statistic / base.statistic - 1) <= 1e-14 and abs(r.pvalue / base.pvalue - 1) <= 1e-13
            if hasattr(r, "conf_int"):
                want = np.asarray(base.conf_int())
                np.testing.assert_allclose(
                    np.asarray(r.conf_int()) / s, want, rtol=0, atol=1e-14 * float(np.max(np.abs(want)))
                )


def test_infinite_values_give_nan():
    for r in (
        sd.htest.ttest_1samp([1.0, math.inf, 2.0], 0.0),
        sd.htest.ttest_ind([1.0, 2.0, 3.0], [1.0, -math.inf]),
        sd.htest.f_oneway([1.0, math.inf, 2.0], [1.0, 3.0]),
        sd.htest.bartlett([1.0, math.inf, 2.0], [1.0, 3.0]),
    ):
        assert math.isnan(r.statistic) and math.isnan(r.pvalue)


# ---- nan_policy ---------------------------------------------------------------------------


def test_nan_policy():
    rng = np.random.default_rng(14)
    a, b, c = rng.normal(0, 1, 20), rng.normal(0.5, 2, 15), rng.normal(1, 1, 12)
    an, bn = a.copy(), b.copy()
    an[[3, 11]] = np.nan
    bn[5] = np.nan
    for fn, args, clean in [
        (sd.htest.ttest_1samp, (an, 0.2), (an[~np.isnan(an)], 0.2)),
        (sd.htest.ttest_ind, (an, bn), (an[~np.isnan(an)], bn[~np.isnan(bn)])),
        (sd.htest.f_oneway, (an, bn, c), (an[~np.isnan(an)], bn[~np.isnan(bn)], c)),
        (sd.htest.bartlett, (an, bn, c), (an[~np.isnan(an)], bn[~np.isnan(bn)], c)),
        (sd.htest.levene, (an, bn, c), (an[~np.isnan(an)], bn[~np.isnan(bn)], c)),
        (sd.htest.fligner, (an, bn, c), (an[~np.isnan(an)], bn[~np.isnan(bn)], c)),
        (sd.htest.alexandergovern, (an, bn, c), (an[~np.isnan(an)], bn[~np.isnan(bn)], c)),
    ]:
        r = fn(*args)
        assert all(math.isnan(getattr(r, f)) for f in type(r)._FIELDS), fn
        assert fn(*args, nan_policy="omit").statistic == fn(*clean).statistic, fn
        with pytest.raises(ValueError, match=r"\(\): the input contains NaN \(nan_policy='raise'\)"):
            fn(*args, nan_policy="raise")
    # paired: the pairs that hold a NaN are left out
    b20 = rng.normal(0, 1, 20)
    keep = ~np.isnan(an)
    assert sd.htest.ttest_rel(an, b20, nan_policy="omit").statistic == sd.htest.ttest_rel(a[keep], b20[keep]).statistic
    t = sd.htest.ttest_ind(an, bn, trim=0.1, nan_policy="omit")
    assert t.statistic == sd.htest.ttest_ind(an[keep], bn[~np.isnan(bn)], trim=0.1).statistic
    # a batch: each lane drops its own NaNs, the other lanes compute
    x = rng.normal(size=(3, 12))
    x[1, 4] = np.nan
    r = sd.htest.ttest_1samp(x, 0.0, axis=-1)
    assert math.isnan(r.statistic[1]) and not math.isnan(r.statistic[0]) and math.isnan(r.df[1])
    r = sd.htest.ttest_1samp(x, 0.0, axis=-1, nan_policy="omit")
    assert float(r.df[1]) == 10.0 and float(r.statistic[1]) == sd.htest.ttest_1samp(np.delete(x[1], 4), 0.0).statistic


# ---- axis, keepdims, popmean ----------------------------------------------------------------


def test_axis_keepdims_and_popmean():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(4, 9, 5))
    r = sd.htest.ttest_1samp(x, 0.0, axis=1)
    assert r.statistic.shape == (4, 5) and r.conf_int().shape == (4, 5, 2)
    assert float(r.statistic[2, 3]) == sd.htest.ttest_1samp(x[2, :, 3], 0.0).statistic
    k = sd.htest.ttest_1samp(x, 0.0, axis=1, keepdims=True)
    assert k.statistic.shape == (4, 1, 5) and k.df.shape == (4, 1, 5)
    t = sd.htest.ttest_1samp(x, 0.0, axis=(0, 2))
    assert t.statistic.shape == (9,) and float(t.df[0]) == 19.0
    assert isinstance(sd.htest.ttest_1samp(x, 0.0).statistic, float)
    # popmean per lane: the result's shape, or scipy's with the sample axis kept
    mu = rng.normal(size=(4, 5))
    r = sd.htest.ttest_1samp(x, mu, axis=1)
    assert float(r.statistic[1, 2]) == sd.htest.ttest_1samp(x[1, :, 2], float(mu[1, 2])).statistic
    r2 = sd.htest.ttest_1samp(x, mu[:, None, :], axis=1)
    assert np.array_equal(r.statistic, r2.statistic)
    with pytest.raises(
        ValueError, match=r"ttest_1samp\(\): 'popmean' cannot be broadcast to the result's shape \(4, 5\)"
    ):
        sd.htest.ttest_1samp(x, np.zeros(3), axis=1)
    # samples of different lengths broadcast outside the sample axis
    a, b = rng.normal(size=(3, 1, 8)), rng.normal(size=(1, 2, 11))
    r = sd.htest.f_oneway(a, b, axis=-1)
    assert r.statistic.shape == (3, 2) and float(r.statistic[2, 1]) == sd.htest.f_oneway(a[2, 0], b[0, 1]).statistic


# ---- result objects (H6, H7, H16, H19, H21) -------------------------------------------------


def test_results():
    rng = np.random.default_rng(16)
    a, b, c = rng.normal(0, 1, 20), rng.normal(0.5, 2, 15), rng.normal(1, 1, 12)
    t = sd.htest.ttest_ind(a, b)
    stat, p = t
    assert (stat, p) == (t.statistic, t.pvalue) and len(t) == 2 and t[1] == t.pvalue
    assert type(t).__name__ == "TtestIndResult" and isinstance(t.df, float)
    assert repr(t).startswith("TtestIndResult(statistic=") and "df=33" in repr(t)
    for fn, name, n in [
        (sd.htest.ttest_1samp(a, 0.0), "Ttest1sampResult", 2),
        (sd.htest.ttest_rel(a[:15], b), "TtestRelResult", 2),
        (sd.htest.ttest_ind_from_stats(0.0, 1.0, 5, 1.0, 1.0, 6), "TtestIndFromStatsResult", 2),
        (sd.htest.f_oneway(a, b, c), "FOnewayResult", 2),
        (sd.htest.bartlett(a, b, c), "BartlettResult", 2),
        (sd.htest.levene(a, b, c), "LeveneResult", 2),
        (sd.htest.fligner(a, b, c), "FlignerResult", 2),
    ]:
        assert type(fn).__name__ == name and len(fn) == n
    f = sd.htest.f_oneway(a, b, c)
    assert (f.df_num, f.df_denom) == (2.0, 44.0)
    ag = sd.htest.alexandergovern(a, b, c)
    with pytest.raises(TypeError):
        tuple(ag)  # scipy's result does not unpack (H7)
    with pytest.raises(AttributeError):
        t.statistic = 1.0
    # the degrees of freedom of every test (H29), outside the unpacking (H7)
    lev = sd.htest.levene(a, b, c)
    assert (lev.df_num, lev.df_denom) == (2.0, 44.0) and len(lev) == 2 and tuple(lev) == (lev.statistic, lev.pvalue)
    for fn in (sd.htest.bartlett, sd.htest.fligner, sd.htest.alexandergovern):
        assert fn(a, b, c).df == 2.0 and isinstance(fn(a, b).df, float)
    assert len(sd.htest.bartlett(a, b, c)) == 2 and len(sd.htest.fligner(a, b, c)) == 2
    bl = sd.htest.levene(rng.normal(size=(3, 8)), rng.normal(size=(3, 9)), axis=-1, nan_policy="omit")
    assert bl.df_denom.tolist() == [15.0, 15.0, 15.0]
    # batch fields are read-only arrays, given without a copy (H21); conf_int is a new array
    bt = sd.htest.ttest_ind(rng.normal(size=(3, 8)), rng.normal(size=(3, 9)), axis=-1)
    assert bt.statistic is bt.statistic
    with pytest.raises(ValueError, match="read-only"):
        bt.df[0] = 1.0
    ci = bt.conf_int()
    ci[0, 0] = 5.0
    assert float(bt.conf_int()[0, 0]) != 5.0
    assert repr(bt) == "TtestIndResult(batch shape (3,))"
    # one-sided intervals end at -+inf (scipy's)
    lo, hi = sd.htest.ttest_ind(a, b, alternative="less").conf_int().tolist()
    assert lo == -math.inf and math.isfinite(hi)
    lo, hi = sd.htest.ttest_ind(a, b, alternative="greater").conf_int(alpha=0.1).tolist()
    assert hi == math.inf and math.isfinite(lo)
    with pytest.raises(ValueError, match=r"conf_int\(\): 'alpha' must be between 0 and 1 \(exclusive\)"):
        t.conf_int(alpha=1.0)


def test_scalar_inputs_are_one_element_samples():
    """A real scalar is a sample of one (user decision 2026-11-20, sd.stats' rule)."""
    r = sd.htest.f_oneway(1.0, [2.0, 3.0, 4.5])
    s = sd.htest.f_oneway([1.0], [2.0, 3.0, 4.5])
    assert r.statistic == s.statistic and r.df_denom == 2.0


# ---- messages (API.md §2.1) -------------------------------------------------------------


def test_messages():
    a, b = [1.0, 2.0, 4.0], [2.0, 5.0, 7.0, 8.0]
    with pytest.raises(TypeError, match=r"ttest_ind\(\): unexpected keyword argument 'equal_variance'"):
        sd.htest.ttest_ind(a, b, equal_variance=True)
    with pytest.raises(TypeError, match=r"ttest_ind\(\): takes 2 positional arguments but 3 were given"):
        sd.htest.ttest_ind(a, b, 0)
    with pytest.raises(
        ValueError, match=r"ttest_ind\(\): 'alternative' must be 'two-sided', 'less' or 'greater', not 'larger'"
    ):
        sd.htest.ttest_ind(a, b, alternative="larger")
    with pytest.raises(ValueError, match=r"ttest_ind\(\): 'trim' must be >= 0 and < 0.5"):
        sd.htest.ttest_ind(a, b, trim=0.5)
    with pytest.raises(TypeError, match=r"ttest_ind\(\): 'trim' must be an int or a float, not str"):
        sd.htest.ttest_ind(a, b, trim="0.1")
    with pytest.raises(TypeError, match=r"ttest_ind\(\): 'equal_var' must be True or False, not int"):
        sd.htest.ttest_ind(a, b, equal_var=1)
    with pytest.raises(
        TypeError, match=r"ttest_ind\(\): 'method' must be None, a PermutationMethod or a MonteCarloMethod, not str"
    ):
        sd.htest.ttest_ind(a, b, method="permutation")
    with pytest.raises(TypeError, match=r"ttest_1samp\(\): 'popmean' must be an int or a float, not bool"):
        sd.htest.ttest_1samp(a, True)
    with pytest.raises(ValueError, match=r"ttest_rel\(\): the paired samples must agree in size, not 3 and 4"):
        sd.htest.ttest_rel(a, b)
    with pytest.raises(TypeError, match=r"f_oneway\(\): at least two samples are required, not 1"):
        sd.htest.f_oneway(a)
    with pytest.raises(TypeError, match=r"bartlett\(\): at least two samples are required, not 0"):
        sd.htest.bartlett()
    with pytest.raises(TypeError, match=r"levene\(\): 'samples\[1\]' must be an sd.ndarray, a sequence of numbers"):
        sd.htest.levene(a, {"x": 1})
    with pytest.raises(ValueError, match=r"levene\(\): 'center' must be 'median', 'mean' or 'trimmed', not 'trim'"):
        sd.htest.levene(a, b, center="trim")
    with pytest.raises(ValueError, match=r"fligner\(\): 'proportiontocut' must be >= 0 and < 0.5"):
        sd.htest.fligner(a, b, center="trimmed", proportiontocut=0.5)
    with pytest.raises(TypeError, match=r"ttest_1samp\(\): 'a' must be real, not a complex128 array"):
        sd.htest.ttest_1samp(np.array([1j, 2.0]), 0.0)
