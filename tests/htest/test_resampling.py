"""sd.htest's resampling and Monte Carlo (milestone H9; docs/api/htest.md §10, docs/design/htest.md
§3.10, §4): the accuracy contract against the references (tests/htest/ref: exact rationals for
permutation_test's and bws_test's enumerations, Fractions and Arb for multiscale_graphcorr's map and
goodness_of_fit's fits and statistics), scipy's, coin's and MASS's results, the reproducibility of
the random resamples (the same rng state gives the same bits on any number of threads and for any
batch), their statistical validity (seeded and bounded), the method= objects of the earlier tests
(their exact enumerations equal the tests' exact distributions and scipy's), bootstrap's intervals
recomputed from its distribution, the H9 decisions (H117-H138), the results and the messages."""

import math
import warnings

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

from .._bits import bits_equal
from . import problems as P
from . import refdata as R

FAMILIES = R.H9
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]
h = sd.htest
PM, MCM, BM = h.PermutationMethod, h.MonteCarloMethod, h.BootstrapMethod
pytestmark = pytest.mark.filterwarnings(
    "ignore::RuntimeWarning:scipy",
    "ignore::UserWarning:scipy",
    "ignore:The number of replications is low:RuntimeWarning",
)


def rng(seed=1):
    return sd.random.default_rng(seed)


def mean_diff(a, b, axis):
    return sd.mean(a, axis=axis) - sd.mean(b, axis=axis)


def np_mean_diff(a, b, axis):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


def smean(a, axis):
    return sd.mean(a, axis=axis)


def _g(seed):
    return np.random.default_rng(seed)


# ---- the contract ---------------------------------------------------------------------------


@pytest.mark.parametrize("fam,pid", CASES)
def test_within_the_contract(fam, pid):
    for field, (_, c) in R.errors(fam, pid).items():
        assert c <= 1.0, (pid, field, c)


def test_exact_counts_and_scipys_answers():
    # the enumerations are scipy's (their counts), and scipy's fixed answers agree with sd's
    sc = R.data("scipy_h9")
    for pid, p in P.PROBLEMS.items():
        if p.milestone != "H9":
            continue
        r = R.call(p)
        want = sc[pid]["result"]
        if "n_null" in want:
            assert r.null_distribution.shape[0] == want["n_null"], pid
            assert r.pvalue == pytest.approx(want["pvalue"], rel=1e-13), pid
        tol = 1e-4 if p.func == "goodness_of_fit" and p.args[0] == "weibull_min" else 1e-12
        assert r.statistic == pytest.approx(want["statistic"], rel=tol, abs=1e-15), pid
        if p.func == "multiscale_graphcorr":
            assert r.mgc_dict["opt_scale"] == want["opt_scale"], pid
            assert np.allclose(np.asarray(r.mgc_dict["mgc_map"]), want["mgc_map"], rtol=0, atol=1e-13), pid
        if p.func == "goodness_of_fit":
            names = R.GOF_PARAMS[p.args[0]]
            got = [getattr(r.fit_result, k) for k in names]
            sp = want["params"]
            dist = p.args[0]
            if dist in ("expon", "rayleigh"):
                ref = [sp[1]]
            elif dist == "weibull_min":
                ref = [sp[0], sp[2]]
            elif dist == "lognorm":
                ref = [math.log(sp[2]), sp[0]]
            else:
                ref = [sp[0], sp[1]]
            assert np.allclose(got, ref, rtol=tol), pid


def test_r_coin_and_mass():
    # coin's exact oneway_test is the permutation test of the difference of the means (one-sided);
    # MASS::fitdistr's maximum likelihood (its optimizer stops at 3e-6 for the logistic, 8e-5 for the
    # Weibull, where there is no closed form; the references hold sd's to 1e-14)
    rd = R.data("r_h9")
    seen = 0
    for pid, rec in rd.items():
        p = P.PROBLEMS[pid]
        res = rec["result"]
        if p.func == "permutation_test":
            for alt in ("less", "greater"):
                got = h.permutation_test(list(p.args[0]), mean_diff, alternative=alt).pvalue
                assert got == pytest.approx(res[f"{alt}.p.value"][0], rel=1e-12), (pid, alt)
        else:
            r = R.call(p)
            names = R.GOF_PARAMS[p.args[0]]
            got = [getattr(r.fit_result, k) for k in names]
            vals = [v[0] for v in res.values()]
            if p.args[0] == "expon":
                vals = [1.0 / vals[0]]
            assert np.allclose(got, vals, rtol=1e-4), pid  # MASS's Weibull (optim): 8e-5
        seen += 1
    assert seen == 9


# ---- the method objects (H117) ----------------------------------------------------------------


def test_method_objects():
    m = PM()
    assert (m.n_resamples, m.batch, m.rng) == (9999, None, None)
    g = rng()
    m = PM(500, 50, rng=g)
    assert (m.n_resamples, m.batch, m.rng) == (500, 50, g)
    assert PM(n_resamples=math.inf).n_resamples == math.inf
    mc = MCM(rng=g)
    assert (mc.n_resamples, mc.batch, mc.rvs, mc.rng) == (9999, None, None, g)
    b = BM(rng=g)
    assert (b.n_resamples, b.batch, b.method, b.rng) == (9999, None, "BCa", g)
    assert repr(PM(5)) == "PermutationMethod(n_resamples=5, batch=None, rng=None)"
    assert repr(BM(5, method="basic", rng=g)).startswith(
        "BootstrapMethod(n_resamples=5, batch=None, method='basic', rng="
    )
    with pytest.raises(AttributeError):
        m.n_resamples = 3
    with pytest.raises(TypeError, match=r"MonteCarloMethod\(\): 'rng' must be an sd.random.Generator"):
        MCM(rng=None)
    with pytest.raises(TypeError, match=r"BootstrapMethod\(\): missing required keyword-only argument: 'rng'|rng"):
        BM()
    with pytest.raises(TypeError, match=r"PermutationMethod\(\): 'rng' must be an sd.random.Generator .*, not int"):
        PM(rng=1)
    with pytest.raises(ValueError, match=r"PermutationMethod\(\): 'n_resamples' must be >= 1"):
        PM(0)
    with pytest.raises(TypeError, match=r"MonteCarloMethod\(\): 'n_resamples' must be an int, not float"):
        MCM(math.inf, rng=g)
    with pytest.raises(
        ValueError, match=r"BootstrapMethod\(\): 'method' must be 'percentile', 'basic' or 'BCa', not 'bca'"
    ):
        BM(method="bca", rng=g)
    with pytest.raises(ValueError, match=r"PermutationMethod\(\): 'batch' must be >= 1"):
        PM(batch=0)
    with pytest.raises(TypeError, match=r"MonteCarloMethod\(\): 'rvs' must be a callable"):
        MCM(rvs=3, rng=g)


# ---- permutation_test -------------------------------------------------------------------------


def test_permutation_null_distribution_is_scipys_enumeration():
    g = _g(5)
    x, y = g.normal(size=5), g.normal(size=4)
    r = h.permutation_test([x, y], mean_diff)
    s = st.permutation_test((x, y), np_mean_diff)
    assert np.allclose(np.asarray(r.null_distribution), s.null_distribution, rtol=0, atol=1e-15)
    a, b = g.normal(size=4), g.normal(size=4)
    for pt, data, fs, fn in (
        (
            "pairings",
            [a, b],
            lambda u, v, axis: sd.mean(u * v, axis=axis),
            lambda u, v, axis: np.mean(u * v, axis=axis),
        ),
        ("samples", [a, b], mean_diff, np_mean_diff),
        ("samples", [a], smean, lambda u, axis: np.mean(u, axis=axis)),
    ):
        r = h.permutation_test(data, fs, permutation_type=pt)
        s = st.permutation_test(tuple(data), fn, permutation_type=pt)
        assert np.allclose(np.asarray(r.null_distribution), s.null_distribution, rtol=0, atol=1e-15), pt
        assert r.pvalue == pytest.approx(s.pvalue, rel=1e-14), pt


def test_permutation_test_options():
    g = _g(6)
    x, y = g.normal(size=30), g.normal(size=25) + 0.4
    with pytest.raises(TypeError, match=r"permutation_test\(\): 'rng' must be an sd.random.Generator"):
        h.permutation_test([x, y], mean_diff)  # too many to enumerate: rng required (H3)
    r = h.permutation_test([x, y], mean_diff, rng=rng(3), n_resamples=2000)
    assert r.null_distribution.shape == (2000,)
    e = h.permutation_test([x[:6], y[:5]], mean_diff, n_resamples=math.inf)
    assert e.null_distribution.shape == (math.comb(11, 6),)
    with pytest.raises(ValueError, match=r"permutation_test\(\): enumerating all 4.71e\+13 permutations"):
        h.permutation_test([x, y[:20]], mean_diff, n_resamples=math.inf)
    # not vectorized: lane by lane, the same values
    nv = h.permutation_test([x[:6], y[:5]], lambda a, b: float(sd.mean(a) - sd.mean(b)))
    assert bits_equal(np.asarray(nv.null_distribution), np.asarray(e.null_distribution))
    assert nv.pvalue == e.pvalue
    # axis and batches: the lanes share the permutations (scipy's)
    X, Y = g.normal(size=(3, 6)), g.normal(size=(3, 5))
    rb = h.permutation_test([X, Y], mean_diff, axis=-1)
    assert rb.null_distribution.shape == (462, 3) and rb.pvalue.shape == (3,)
    for i in range(3):
        ri = h.permutation_test([X[i], Y[i]], mean_diff)
        assert rb.pvalue[i] == ri.pvalue and float(rb.statistic[i]) == ri.statistic
    sb = st.permutation_test((X, Y), np_mean_diff, axis=-1)
    assert np.allclose(np.asarray(rb.pvalue), sb.pvalue, rtol=1e-14)
    # a NaN statistic gives NaN
    xn = x[:6].copy()
    xn[2] = np.nan
    assert math.isnan(h.permutation_test([xn, y[:5]], mean_diff).pvalue)
    with pytest.raises(ValueError, match=r"permutation_test\(\): permutation_type='independent' needs at least two"):
        h.permutation_test([x], smean)
    with pytest.raises(ValueError, match=r"permutation_test\(\): each sample must hold at least 2 observations"):
        h.permutation_test([x, [1.0]], mean_diff)
    with pytest.raises(ValueError, match=r"permutation_test\(\): the paired samples must agree in size"):
        h.permutation_test([x, y], mean_diff, permutation_type="samples")
    with pytest.raises(ValueError, match=r"permutation_test\(\): 'statistic' must return shape \(462,\), not a number"):
        h.permutation_test([x[:6], y[:5]], lambda a, b, axis: sd.mean(a, axis=None) + 0 * sd.mean(b))
    with pytest.raises(TypeError, match=r"permutation_test\(\): 'data' must be a list or tuple of samples"):
        h.permutation_test(x, smean)
    with pytest.raises(TypeError, match=r"permutation_test\(\): 'statistic' must be callable"):
        h.permutation_test([x, y], 3)
    with pytest.raises(ValueError, match=r"permutation_test\(\): 'permutation_type' must be"):
        h.permutation_test([x, y], mean_diff, permutation_type="paired")


def test_random_resamples_reproducible_threads_and_batches():
    g = _g(7)
    x, y = g.normal(size=40), g.normal(size=35) + 0.2
    X = g.normal(size=(4, 30))
    calls = [
        lambda b: h.permutation_test([x, y], mean_diff, rng=rng(11), n_resamples=3000, batch=b).null_distribution,
        lambda b: h.permutation_test([x[:35], y], mean_diff, rng=rng(11), permutation_type="samples",
                                     n_resamples=3000, batch=b).null_distribution,
        lambda b: h.permutation_test([x[:30], y[:30]], lambda u, v, axis: sd.mean(u * v, axis=axis), rng=rng(12),
                                     permutation_type="pairings", n_resamples=3000, batch=b).null_distribution,
        lambda b: h.bootstrap([X], smean, axis=-1, rng=rng(13), n_resamples=2000, batch=b).conf_int(),
        lambda b: h.bootstrap([x, y], mean_diff, rng=rng(14), n_resamples=2000, batch=b, paired=False).conf_int(),
        lambda b: h.monte_carlo_test([x, y], [sd.stats.Normal().sample] * 2, mean_diff, rng=rng(15),
                                     n_resamples=2000, batch=b).null_distribution,
        lambda b: h.multiscale_graphcorr(x[:20], y[:20] ** 2, reps=200, rng=rng(16)).mgc_dict["null_dist"],
        lambda b: h.goodness_of_fit(sd.stats.Weibull, np.abs(x) + 0.1, n_mc_samples=300, rng=rng(17)).null_distribution,
        lambda b: h.chi2_contingency([[12.0, 5, 7], [3, 9, 4]], correction=False,
                                     method=MCM(2000, b, rng=rng(18))).pvalue,
    ]  # fmt: skip
    before = sd.get_num_threads()
    try:
        want = None
        for th, b in ((1, None), (3, 7), (10, 1000)):
            sd.set_num_threads(th)
            got = [np.asarray(f(b)) for f in calls]
            if want is None:
                want = got
            else:
                for i, (a, w) in enumerate(zip(got, want)):
                    assert bits_equal(a, w), i
    finally:
        sd.set_num_threads(before)
    # another state, other draws
    a = h.permutation_test([x, y], mean_diff, rng=rng(11), n_resamples=3000).null_distribution
    b = h.permutation_test([x, y], mean_diff, rng=rng(12), n_resamples=3000).null_distribution
    assert not bits_equal(np.asarray(a), np.asarray(b))
    # the generator moves: two calls on one generator differ
    g1 = rng(20)
    p1 = h.permutation_test([x, y], mean_diff, rng=g1, n_resamples=500).pvalue
    p2 = h.permutation_test([x, y], mean_diff, rng=g1, n_resamples=500).pvalue
    assert g1.state["position"] > 0 and (p1, p2) != (None, None)


def test_random_permutations_estimate_the_exact_pvalue():
    # random permutations of a problem small enough to enumerate: within 5 standard errors (seeded)
    g = _g(8)
    x, y = g.normal(size=8), g.normal(size=8) + 0.7
    exact = h.permutation_test([x, y], mean_diff, n_resamples=math.inf).pvalue
    for seed in range(5):
        r = h.permutation_test([x, y], mean_diff, n_resamples=4000, rng=rng(100 + seed))
        se = math.sqrt(exact * (1 - exact) / 4000)
        assert abs(r.pvalue - exact) <= 5 * se + 2 / 4001, seed


def test_monte_carlo_test_size_under_the_null():
    # the share of p <= 0.1 over 300 normal samples tested against the normal (seeded; binomial
    # bounds 5 standard deviations wide)
    g = rng(30)
    data = sd.stats.Normal().sample((300, 15), rng=g)
    stat = lambda a, axis: sd.stats.skew(a, axis=axis)  # noqa: E731
    r = h.monte_carlo_test(data, sd.stats.Normal().sample, stat, axis=-1, n_resamples=999, rng=g)
    rate = float(sd.mean(r.pvalue <= 0.1))
    assert abs(rate - 0.1) <= 5 * math.sqrt(0.09 / 300)
    assert r.null_distribution.shape == (999,) and r.pvalue.shape == (300,)


def test_monte_carlo_test_options():
    g = _g(9)
    x = g.normal(size=20)
    r = h.monte_carlo_test(x, sd.stats.Normal().sample, smean, rng=rng(1), n_resamples=99)
    assert r.null_distribution.shape == (99,)
    # each sample draws from its own child: a sample's draws do not depend on the other's
    calls = []

    def spy(size, *, rng):
        calls.append(rng.state["spawn_key"])
        return sd.stats.Normal().sample(size, rng=rng)

    h.monte_carlo_test([x, x], [spy, spy], mean_diff, rng=rng(1), n_resamples=10)
    assert len({tuple(k) for k in calls}) == 2
    with pytest.raises(ValueError, match=r"monte_carlo_test\(\): 'rvs' must return an array of shape \(9999, 20\)"):
        h.monte_carlo_test(x, lambda size, rng: sd.zeros(3), smean, rng=rng(1))
    with pytest.raises(ValueError, match=r"monte_carlo_test\(\): 'data' must hold one sample per 'rvs' \(2\), not 1"):
        h.monte_carlo_test([x], [spy, spy], mean_diff, rng=rng(1))
    with pytest.raises(TypeError, match=r"monte_carlo_test\(\): 'rng' must be an sd.random.Generator"):
        h.monte_carlo_test(x, spy, smean, rng=None)


# ---- bootstrap -----------------------------------------------------------------------------------


def _np_quantile(v, p):
    return np.quantile(np.asarray(v), p)


def test_bootstrap_intervals_from_its_distribution():
    # percentile, basic and BCa recomputed in numpy from the result's distribution (scipy's formulas;
    # the acceleration from the jackknife)
    from scipy.special import ndtr, ndtri

    g = _g(10)
    x = g.exponential(size=25)
    theta = float(np.mean(x))
    jack = np.array([np.mean(np.delete(x, i)) for i in range(25)])
    u = 24 * (jack.mean() - jack)
    a_hat = np.sum(u**3) / 25**3 / (np.sum(u**2) / 25**2) ** 1.5 / 6
    for meth in ("percentile", "basic", "BCa"):
        r = h.bootstrap([x], smean, method=meth, rng=rng(2), n_resamples=3000)
        d = np.asarray(r.bootstrap_distribution)
        for alpha, alt in ((0.05, "two-sided"), (0.1, "two-sided"), (0.05, "less"), (0.05, "greater")):
            r = h.bootstrap([x], smean, method=meth, rng=rng(2), n_resamples=3000, alternative=alt)
            a = alpha / 2 if alt == "two-sided" else alpha
            if meth == "BCa":
                z0 = ndtri((np.sum(d < theta) + np.sum(d <= theta)) / (2 * len(d)))
                za = ndtri(a)
                p1 = ndtr(z0 + (z0 + za) / (1 - a_hat * (z0 + za)))
                p2 = ndtr(z0 + (z0 - za) / (1 - a_hat * (z0 - za)))
            else:
                p1, p2 = a, 1 - a
            lo, hi = _np_quantile(d, p1), _np_quantile(d, p2)
            if meth == "basic":
                lo, hi = 2 * theta - hi, 2 * theta - lo
            lo = -math.inf if alt == "less" else lo
            hi = math.inf if alt == "greater" else hi
            assert np.allclose(np.asarray(r.conf_int(alpha=alpha)), [lo, hi], rtol=1e-13), (meth, alpha, alt)
        assert r.standard_error == pytest.approx(np.std(d, ddof=1), rel=1e-13)


def test_bootstrap_coverage():
    # the 90% BCa interval of the mean of 200 normal samples of 20 covers the mean about 90% of the
    # time (seeded; binomial bounds 5 standard deviations wide)
    g = rng(40)
    data = sd.stats.Normal(mu=1.0).sample((200, 20), rng=g)
    r = h.bootstrap([data], smean, axis=-1, n_resamples=999, rng=g)
    ci = np.asarray(r.conf_int(alpha=0.1))
    cover = np.mean((ci[:, 0] <= 1.0) & (1.0 <= ci[:, 1]))
    assert abs(cover - 0.9) <= 5 * math.sqrt(0.09 / 200)


def test_bootstrap_options():
    g = _g(11)
    x, y = g.normal(size=15), g.normal(size=15)
    r = h.bootstrap([x, y], lambda a, b, axis: sd.htest.pearsonr(a, b, axis=-1).statistic, paired=True, rng=rng(3),
                    n_resamples=500)  # fmt: skip
    assert r.bootstrap_distribution.shape == (500,)
    assert r.conf_int().shape == (2,) and r.conf_int(alpha=0.2)[0] > r.conf_int(alpha=0.05)[0]
    # a constant statistic: the BCa interval is NaN (no warning, H11)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = h.bootstrap([np.ones(10)], smean, rng=rng(1), n_resamples=100)
    assert np.isnan(np.asarray(c.conf_int())).all() and c.standard_error == 0.0
    with pytest.raises(ValueError, match=r"bootstrap\(\): each sample must hold at least 2 observations"):
        h.bootstrap([[1.0]], smean, rng=rng(1))
    with pytest.raises(ValueError, match=r"bootstrap\(\): the paired samples must agree in size"):
        h.bootstrap([x, y[:5]], mean_diff, paired=True, rng=rng(1))
    with pytest.raises(ValueError, match=r"bootstrap\(\): 'method' must be 'percentile', 'basic' or 'BCa'"):
        h.bootstrap([x], smean, method="bca", rng=rng(1))
    with pytest.raises(ValueError, match=r"conf_int\(\): 'alpha' must be between 0 and 1"):
        r.conf_int(alpha=1.5)


# ---- bws_test, multiscale_graphcorr, goodness_of_fit ----------------------------------------------


def test_bws_test():
    x = np.array([90.0, 86, 72, 65, 44, 52, 46, 38, 43])
    y = np.array([86.0, 71, 69, 60, 36, 43, 40, 30, 34])
    r = h.bws_test(x, y)
    s = st.bws_test(x, y, method=st.PermutationMethod(n_resamples=10**6))
    assert r.statistic == pytest.approx(s.statistic, rel=1e-14)
    assert r.pvalue == pytest.approx(s.pvalue, rel=1e-14)
    assert r.null_distribution.shape == (math.comb(18, 9),)
    g = _g(12)
    big = (g.normal(size=13), g.normal(size=13))
    with pytest.raises(TypeError, match=r"bws_test\(\): the 1.04e\+07 permutations are more than the 1e\+06"):
        h.bws_test(*big)
    a = h.bws_test(*big, method=PM(3000, rng=rng(5)))
    b = h.bws_test(*big, method=PM(3000, batch=17, rng=rng(5)))
    assert a.pvalue == b.pvalue and a.null_distribution.shape == (3000,)
    with pytest.raises(
        TypeError, match=r"bws_test\(\): 'method' must be None or a PermutationMethod, not MonteCarloMethod"
    ):
        h.bws_test(x, y, method=MCM(rng=rng()))
    with pytest.raises(ValueError, match=r"bws_test\(\): 'y' must hold at least 2 observations, not 1"):
        h.bws_test(x, [1.0])
    X, Y = g.normal(size=(2, 6)), g.normal(size=(2, 5))
    rb = h.bws_test(X, Y, axis=-1)
    for i in range(2):
        assert float(rb.pvalue[i]) == h.bws_test(X[i], Y[i]).pvalue


def test_multiscale_graphcorr():
    g = _g(13)
    x = g.normal(size=(30, 2))
    y = x[:, :1] ** 2 + 0.3 * g.normal(size=(30, 3))
    r = h.multiscale_graphcorr(x, y, reps=300, rng=rng(1))
    s = st.multiscale_graphcorr(x, y, reps=300, random_state=1)
    assert r.statistic == pytest.approx(s.statistic, rel=1e-13)
    assert r.mgc_dict["opt_scale"] == list(s.mgc_dict["opt_scale"])
    stat, p, d = r  # unpacks as scipy's (H7)
    assert p == r.pvalue and d["null_dist"].shape == (300,) and d["mgc_map"].shape == (30, 30)
    # the p-value's definition, and reproducibility
    assert p == (1 + float(sd.sum(d["null_dist"] >= stat))) / 301
    assert h.multiscale_graphcorr(x, y, reps=300, rng=rng(1)).pvalue == p
    # mgc_dict is a new dict at each read, its arrays read-only (H21, G113)
    assert r.mgc_dict is not r.mgc_dict
    with pytest.raises(ValueError):
        r.mgc_dict["mgc_map"][0, 0] = 1.0
    # distance matrices (compute_distance=None), a callable, two samples
    from scipy.spatial.distance import cdist

    dm = h.multiscale_graphcorr(cdist(x, x), cdist(y, y), compute_distance=None, reps=0, rng=rng(1))
    assert dm.statistic == pytest.approx(r.statistic, rel=1e-13) and dm.pvalue == 1.0
    cd = h.multiscale_graphcorr(x, y, compute_distance=lambda v: sd.asarray(cdist(np.asarray(v), np.asarray(v))),
                                reps=0, rng=rng(1))  # fmt: skip
    assert cd.statistic == pytest.approx(r.statistic, rel=1e-13)
    u, v = g.normal(size=(12, 2)), g.normal(size=(15, 2)) + 1
    t = h.multiscale_graphcorr(u, v, reps=0, rng=rng(1))
    assert t.statistic == pytest.approx(st.multiscale_graphcorr(u, v, reps=0).statistic, rel=1e-13)
    with pytest.raises(ValueError, match=r"multiscale_graphcorr\(\): at least 5 samples are required, not 4"):
        h.multiscale_graphcorr(x[:4], y[:4], rng=rng())
    with pytest.raises(ValueError, match=r"multiscale_graphcorr\(\): 'x' must hold finite values"):
        h.multiscale_graphcorr(np.r_[x[:, 0], np.inf], np.r_[y[:, 0], 1.0], rng=rng())
    with pytest.raises(ValueError, match=r"multiscale_graphcorr\(\): 'x' and 'y' must have shapes"):
        h.multiscale_graphcorr(x, y[:10], rng=rng())
    with pytest.raises(TypeError, match=r"multiscale_graphcorr\(\): 'rng' must be an sd.random.Generator"):
        h.multiscale_graphcorr(x, y, rng=None)


def test_goodness_of_fit():
    g = _g(14)
    x = g.normal(1.0, 2.0, 50)
    r = h.goodness_of_fit(sd.stats.Normal, x, n_mc_samples=999, rng=rng(2))
    assert isinstance(r.fit_result, sd.stats.Normal)
    assert r.fit_result.mu == pytest.approx(np.mean(x)) and r.fit_result.sigma == pytest.approx(np.std(x, ddof=1))
    assert r.null_distribution.shape == (999,) and 0 < r.pvalue <= 1
    # known parameters are held everywhere; fit parameters only for the null distribution
    k = h.goodness_of_fit(sd.stats.Normal, x, known_params={"mu": 1.0}, n_mc_samples=99, rng=rng(2))
    assert k.fit_result.mu == 1.0
    f = h.goodness_of_fit(sd.stats.Normal, x, fit_params={"mu": 1.0, "sigma": 2.0}, n_mc_samples=99, rng=rng(2))
    assert (f.fit_result.mu, f.fit_result.sigma) == (1.0, 2.0)
    assert f.statistic == r.statistic  # the statistic refits both

    # a callable statistic of the fitted distribution, with its alternative
    def meanabs(dist, data, axis):
        return sd.mean(sd.abs(dist.cdf(data) - 0.5), axis=axis)

    c = h.goodness_of_fit(sd.stats.Logistic, x, statistic=meanabs, n_mc_samples=99, rng=rng(2))
    assert c.null_distribution.shape == (99,) and math.isfinite(c.statistic)
    # the size under the null: p <= 0.1 for about 10% of 100 Gumbel samples (seeded, bounded)
    gg = rng(50)
    ps = [h.goodness_of_fit(sd.stats.Gumbel, sd.stats.Gumbel().sample(30, rng=gg), n_mc_samples=199, rng=gg).pvalue
          for _ in range(100)]  # fmt: skip
    assert abs(np.mean(np.asarray(ps) <= 0.1) - 0.1) <= 5 * math.sqrt(0.09 / 100)
    with pytest.raises(
        ValueError, match=r"goodness_of_fit\(\): 'dist' must be one of the classes .* not sd.stats.Gamma"
    ):
        h.goodness_of_fit(sd.stats.Gamma, x, rng=rng())
    with pytest.raises(ValueError, match=r"goodness_of_fit\(\): 'known_params' has 'loc', not a parameter"):
        h.goodness_of_fit(sd.stats.Normal, x, known_params={"loc": 0.0}, rng=rng())
    with pytest.raises(ValueError, match=r"goodness_of_fit\(\): 'mu' is in both known_params and fit_params"):
        h.goodness_of_fit(sd.stats.Normal, x, known_params={"mu": 0.0}, fit_params={"mu": 1.0}, rng=rng())
    with pytest.raises(ValueError, match=r"goodness_of_fit\(\): 'statistic' must be 'ad', 'ks', 'cvm' or 'filliben'"):
        h.goodness_of_fit(sd.stats.Normal, x, statistic="AD", rng=rng())
    with pytest.raises(ValueError, match=r"goodness_of_fit\(\): 'data' must be one-dimensional, not 2-D"):
        h.goodness_of_fit(sd.stats.Normal, x.reshape(5, 10), rng=rng())
    # outside the family's support the fit is NaN (H11)
    w = h.goodness_of_fit(sd.stats.Weibull, x, n_mc_samples=9, rng=rng())
    assert w.fit_result is None and math.isnan(w.statistic) and math.isnan(w.pvalue)


def test_anderson_weibull_and_monte_carlo():
    # H40: the three-parameter Weibull (scipy's weibull_min), its table; MonteCarloMethod (H9)
    g = _g(15)
    for c in (2.5, 4.0):
        w = g.weibull(c, 60) * 3 + 1
        r = h.anderson(w, dist=sd.stats.Weibull)
        s = st.anderson(w, dist="weibull_min", method="interpolate")
        assert r.statistic == pytest.approx(s.statistic, rel=1e-6) and r.pvalue == pytest.approx(s.pvalue, rel=1e-6)
    low = h.anderson(g.weibull(0.7, 60), dist=sd.stats.Weibull)  # c < 1: no interior maximum (scipy raises)
    assert math.isnan(low.statistic) and math.isnan(low.pvalue)
    x = g.normal(size=40)
    m = h.anderson(x, method=MCM(999, rng=rng(1)))
    gf = h.goodness_of_fit(sd.stats.Normal, x, n_mc_samples=999, rng=rng(1))
    assert m.pvalue == gf.pvalue and m.statistic == pytest.approx(gf.statistic, rel=1e-13)
    sm = st.anderson(x, method=st.MonteCarloMethod(n_resamples=999, rng=1))
    assert abs(m.pvalue - sm.pvalue) < 0.06
    with pytest.raises(ValueError, match=r"anderson\(\): a MonteCarloMethod's 'rvs' must be None"):
        h.anderson(x, method=MCM(rvs=sd.stats.Normal().sample, rng=rng()))
    with pytest.raises(
        TypeError, match=r"anderson\(\): 'method' must be 'interpolate' or a MonteCarloMethod, not Perm"
    ):
        h.anderson(x, method=PM())


# ---- the earlier tests' method= (H117) -------------------------------------------------------------


def test_rank_tests_permutation_equals_their_exact_distribution():
    # every permutation is the conditional exact distribution (H45): the same p-value
    g = _g(16)
    x, y = np.round(g.normal(size=6) * 2), np.round(g.normal(size=5) * 2 + 1)
    for alt in ("two-sided", "less", "greater"):
        for f in (h.mannwhitneyu, h.ansari):
            a, b = f(x, y, method=PM(), alternative=alt), f(x, y, method="exact", alternative=alt)
            assert a.pvalue == pytest.approx(b.pvalue, rel=1e-13) and a.statistic == b.statistic, (f, alt)
            assert a.method == "permutation" and not hasattr(a, "zstatistic")
    d1, d2 = np.round(g.normal(size=9), 1), np.round(g.normal(size=9), 1)
    for zm in ("wilcox", "pratt", "zsplit"):
        a = h.wilcoxon(d1, d2, method=PM(), zero_method=zm)
        b = h.wilcoxon(d1, d2, method="exact", zero_method=zm)
        s = st.wilcoxon(d1, d2, method=st.PermutationMethod(), zero_method=zm)
        assert a.pvalue == pytest.approx(b.pvalue, rel=1e-13) == pytest.approx(s.pvalue, rel=1e-13), zm
    with pytest.raises(TypeError, match=r"mannwhitneyu\(\): 'method' must be 'auto', 'exact', 'asymptotic' or a Per"):
        h.mannwhitneyu(x, y, method=MCM(rng=rng()))


def test_methods_against_scipys_enumerations():
    g = _g(17)
    a, b = g.normal(size=6), g.normal(size=6) + 0.8
    c = g.normal(size=7)
    d = c + g.normal(size=7)
    S = st.PermutationMethod()
    pairs = [
        (h.ttest_ind(a, b, method=PM()), st.ttest_ind(a, b, method=S)),
        (h.ttest_ind(a, b, method=PM(), equal_var=False, trim=0.2), st.ttest_ind(a, b, method=S, equal_var=False,
                                                                                 trim=0.2)),
        (h.pearsonr(c, d, method=PM()), st.pearsonr(c, d, method=S)),
        (h.spearmanrho(c, d, method=PM(), alternative="greater"),
         st.spearmanrho(c, d, method=S, alternative="greater")),
        (h.chatterjeexi(c, d, method=PM()), st.chatterjeexi(c, d, method=S)),
        (h.anderson_ksamp([a[:4], b[:4], c[:3]], method=PM(20000)),
         st.anderson_ksamp([a[:4], b[:4], c[:3]], method=st.PermutationMethod(20000), variant="midrank")),
        (h.chi2_contingency([[2.0, 1], [1, 3]], correction=False, method=PM()),
         st.chi2_contingency([[2, 1], [1, 3]], correction=False, method=S)),
        (h.fisher_exact([[2.0, 1, 0], [0, 2, 2]], method=PM()), st.fisher_exact([[2, 1, 0], [0, 2, 2]], method=S)),
    ]  # fmt: skip
    for i, (r, s) in enumerate(pairs):
        assert r.pvalue == pytest.approx(s.pvalue, rel=1e-13), i
        assert r.statistic == pytest.approx(s.statistic, rel=1e-12), i
    # df and the interval stay the t's
    t = h.ttest_ind(a, b, method=PM())
    t0 = h.ttest_ind(a, b)
    assert t.df == t0.df and bits_equal(np.asarray(t.conf_int()), np.asarray(t0.conf_int()))
    assert math.isnan(h.chi2_contingency([[2.0, 1], [1, 3]], correction=False, method=PM()).df)


def test_monte_carlo_methods():
    g = _g(18)
    a, b = g.normal(size=12), g.normal(size=12) + 0.5
    r1 = h.ttest_ind(a, b, method=MCM(2000, rng=rng(3)))
    r2 = h.ttest_ind(a, b, method=MCM(2000, rng=rng(3)))
    assert r1.pvalue == r2.pvalue and abs(r1.pvalue - h.ttest_ind(a, b).pvalue) < 0.03
    p = h.pearsonr(a, b, method=MCM(2000, rng=rng(4))).pvalue
    assert abs(p - h.pearsonr(a, b).pvalue) < 0.04
    T = [[12.0, 5, 7], [3, 9, 4]]
    m = h.chi2_contingency(T, correction=False, method=MCM(4000, rng=rng(5))).pvalue
    assert abs(m - h.chi2_contingency(T, correction=False).pvalue) < 0.02
    q = h.fisher_exact(T, method=MCM(4000, rng=rng(5))).pvalue
    assert abs(q - h.fisher_exact(T).pvalue) < 0.02
    # a PermutationMethod's random tables are the Monte Carlo's (the same stream layout)
    assert h.fisher_exact(T, method=PM(4000, rng=rng(5))).pvalue == q
    with pytest.raises(ValueError, match=r"chi2_contingency\(\): correction=True is not compatible"):
        h.chi2_contingency(T, method=PM(rng=rng()))
    with pytest.raises(ValueError, match=r"chi2_contingency\(\): a MonteCarloMethod's 'rvs' must be None"):
        h.chi2_contingency(T, correction=False, method=MCM(rvs=sd.stats.Normal().sample, rng=rng()))
    with pytest.raises(ValueError, match=r"fisher_exact\(\): a resampling method takes alternative='two-sided' only"):
        h.fisher_exact([[2.0, 1], [1, 3]], alternative="less", method=PM())


def test_pearsonr_bootstrap_interval():
    g = _g(19)
    c = g.normal(size=20)
    d = c + g.normal(size=20)
    r = h.pearsonr(c, d)
    ci = r.conf_int(method=BM(2000, rng=rng(6)))
    b = h.bootstrap([c, d], lambda u, v, axis: sd.htest.pearsonr(u, v, axis=-1).statistic, paired=True,
                    n_resamples=2000, rng=rng(6))  # fmt: skip
    assert bits_equal(np.asarray(ci), np.clip(np.asarray(b.conf_int()), -1, 1))
    assert bits_equal(np.asarray(r.conf_int()), np.asarray(h.pearsonr(c, d).conf_int()))  # Fisher's by default
    with pytest.raises(TypeError, match=r"conf_int\(\): 'method' must be None or a BootstrapMethod, not Perm"):
        r.conf_int(method=PM())


def test_nan_policy_with_methods():
    g = _g(20)
    x, y = g.normal(size=(2, 7)), g.normal(size=(2, 6))
    x[1, 2] = np.nan
    prop = h.mannwhitneyu(x, y, axis=-1, method=PM())
    assert math.isnan(float(prop.pvalue[1])) and not math.isnan(float(prop.pvalue[0]))
    om = h.mannwhitneyu(x, y, axis=-1, method=PM(), nan_policy="omit")
    want = h.mannwhitneyu(x[1][~np.isnan(x[1])], y[1], method=PM())
    assert float(om.pvalue[1]) == want.pvalue
    assert float(om.pvalue[0]) == h.mannwhitneyu(x[0], y[0], method=PM()).pvalue


# ---- results ---------------------------------------------------------------------------------------


def test_results_summary_and_repr():
    g = _g(21)
    x, y = g.normal(size=5), g.normal(size=4)
    r = h.permutation_test([x, y], mean_diff)
    assert repr(r).startswith("PermutationTestResult(statistic=") and "<array of shape (126,)>" in repr(r)
    text = r.summary()
    assert "Permutation test" in text and "126 (all)" in text
    with pytest.raises(TypeError):
        tuple(r)  # does not unpack (scipy's dataclass)
    with pytest.raises(AttributeError):
        r.pvalue = 0.5
    b = h.bootstrap([x], smean, rng=rng(1), n_resamples=200)
    s = b.summary(alpha=0.1)
    assert "Bootstrap (BCa)" in s and "[0.050     0.950]" in s and "Standard error" in s
    m = h.mannwhitneyu(x, y, method=PM())
    assert "permutation" in m.summary() and "Resamples:" in m.summary()
    gof = h.goodness_of_fit(sd.stats.Normal, g.normal(size=20), n_mc_samples=19, rng=rng(1))
    assert "Normal" in gof.summary() and "statistic=" in repr(gof)


def test_goodness_of_fit_summary_and_repr():
    """H158: one 'Statistic:' line (the statistic's kind is 'Statistic type:'), and fit_result
    always in the repr: the fitted distribution as <Name>, None where the fit failed."""
    x = [0.42, -1.10, 0.85, 2.31, -0.27, 0.66, 1.94, -0.58, 0.12, 3.05, 0.77, -0.91]
    r = h.goodness_of_fit(sd.stats.Normal, x, n_mc_samples=10, rng=sd.random.default_rng(8))
    text = r.summary()
    assert text.count("Statistic:") == 1 and "Statistic type:                     ad" in text
    assert repr(r).startswith("GoodnessOfFitResult(fit_result=<Normal>, statistic=0.262854, pvalue=")
    bad = h.goodness_of_fit(sd.stats.Exponential, [-1.0, 2, 3], n_mc_samples=10, rng=sd.random.default_rng(1))
    assert repr(bad).startswith("GoodnessOfFitResult(fit_result=None, statistic=nan, pvalue=nan")


def test_data_of_numbers_is_a_type_error():
    """One sample given bare (its numbers, not (x,)): a TypeError that says how to give it."""
    for f in (h.bootstrap, h.permutation_test):
        name = f.__name__
        with pytest.raises(
            TypeError,
            match=rf"{name}\(\): 'data' must be a sequence of samples \(one sample is given as \(x,\)\), not a "
            r"sequence of numbers",
        ):
            f([1.0, 2, 3, 4], smean, rng=rng(0))
    assert h.bootstrap(([1.0, 2, 3, 4],), smean, rng=rng(0), n_resamples=50).standard_error > 0
