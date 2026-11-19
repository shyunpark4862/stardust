"""X.sample(shape, rng=rng) (T7, docs/api/stats.md §2.5, docs/design/stats.md §4): the samples
follow their distribution (KS against the distribution's own cdf for the continuous ones, chi
square against its cdf steps for the discrete ones, at fixed seeds), depend on the generator's
state only (any thread count, split calls), and keep the shape, NaN and argument rules."""

import zlib

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

S = sd.stats
N = 20000
P_MIN = 1e-4  # fixed seeds: a pass or a fail is the same on every run

CONTINUOUS = [
    ("Normal", dict(mu=1.0, sigma=2.0)),
    ("LogNormal", dict(mu=0.5, sigma=1.5)),
    ("Exponential", dict(scale=2.0)),
    ("Uniform", dict(a=-1.0, b=3.0)),
    ("Laplace", dict(mu=0.5, sigma=2.0)),
    ("Cauchy", dict(mu=0.0, sigma=1.0)),
    ("Logistic", dict(mu=0.0, sigma=1.0)),
    ("Gumbel", dict(mu=1.0, sigma=2.0)),
    ("GEV", dict(c=0.3)),
    ("GEV", dict(c=-0.4)),
    ("GEV", dict(c=0.0)),
    ("Weibull", dict(c=0.7)),
    ("Pareto", dict(b=2.5)),
    ("Rayleigh", dict(scale=1.5)),
    ("Gamma", dict(a=3.0, scale=2.0)),
    ("Gamma", dict(a=0.3)),
    ("Gamma", dict(a=1e6)),
    ("ChiSquared", dict(df=0.5)),
    ("ChiSquared", dict(df=7.0)),
    ("InverseGamma", dict(a=0.5)),
    ("InverseGamma", dict(a=4.0, scale=3.0)),
    ("Beta", dict(a=0.2, b=0.3)),
    ("Beta", dict(a=2.0, b=5.0)),
    ("Beta", dict(a=0.5, b=3.0)),
    ("StudentT", dict(df=0.5)),
    ("StudentT", dict(df=5.0)),
    ("StudentT", dict(df=1e4)),
    ("F", dict(dfn=0.7, dfd=3.0)),
    ("F", dict(dfn=10.0, dfd=20.0)),
    ("NoncentralChiSquared", dict(df=0.5, nc=3.0)),
    ("NoncentralChiSquared", dict(df=4.0, nc=50.0)),
    ("NoncentralChiSquared", dict(df=4.0, nc=0.0)),
    ("NoncentralT", dict(df=3.0, nc=-2.0)),
    ("NoncentralT", dict(df=30.0, nc=5.0)),
    ("NoncentralF", dict(dfn=3.0, dfd=8.0, nc=4.0)),
]
DISCRETE = [
    ("Bernoulli", dict(p=0.3)),
    ("DiscreteUniform", dict(low=-3.0, high=7.0)),
    ("Geometric", dict(p=0.2)),
    ("Geometric", dict(p=1e-6)),
    ("Poisson", dict(mu=0.7)),
    ("Poisson", dict(mu=9.9)),
    ("Poisson", dict(mu=10.0)),
    ("Poisson", dict(mu=1234.5)),
    ("Poisson", dict(mu=1e12)),
    ("Binomial", dict(n=20.0, p=0.3)),
    ("Binomial", dict(n=20.0, p=0.9)),
    ("Binomial", dict(n=1e4, p=0.3)),
    ("Binomial", dict(n=1e9, p=0.7)),
    ("Binomial", dict(n=1e6, p=1e-5)),
    ("NegativeBinomial", dict(n=5.0, p=0.3)),
    ("NegativeBinomial", dict(n=0.3, p=0.01)),
    ("Hypergeometric", dict(total=1000.0, successes=300.0, draws=100.0)),
    ("Hypergeometric", dict(total=50.0, successes=45.0, draws=40.0)),
    ("Hypergeometric", dict(total=1e7, successes=4e6, draws=5e5)),
    ("BetaBinomial", dict(n=20.0, a=2.0, b=3.0)),
    ("BetaBinomial", dict(n=100.0, a=0.3, b=0.2)),
    ("LogSeries", dict(p=0.9)),
    ("LogSeries", dict(p=0.999)),
    ("LogSeries", dict(p=1e-3)),
    ("Zipf", dict(a=2.0)),
    ("Zipf", dict(a=1.3)),
    ("Zipf", dict(a=6.0)),
]
ALL_NAMES = sorted({n for n, _ in CONTINUOUS + DISCRETE})


def _ids(cases):
    return [f"{n}-{'-'.join(f'{k}={v:g}' for k, v in kw.items())}" for n, kw in cases]


@pytest.mark.parametrize(("name", "kw"), CONTINUOUS, ids=_ids(CONTINUOUS))
def test_continuous_follow_their_cdf(name, kw):
    D = getattr(S, name)(**kw)
    x = np.asarray(D.sample(N, rng=sd.random.default_rng(zlib.crc32(name.encode()))))
    assert x.shape == (N,) and np.all(np.isfinite(x))
    p = st.kstest(x, lambda v: np.asarray(D.cdf(sd.asarray(v)))).pvalue
    assert p > P_MIN, (name, kw, p)


@pytest.mark.parametrize(("name", "kw"), DISCRETE, ids=_ids(DISCRETE))
def test_discrete_follow_their_pmf(name, kw):
    D = getattr(S, name)(**kw)
    x = np.asarray(D.sample(N, rng=sd.random.default_rng(zlib.crc32(name.encode()))))
    finite = x[np.isfinite(x)]
    assert np.all(finite == np.floor(finite))
    lo, hi = D.support()
    assert np.all((x >= lo) & (x <= hi))
    # chi square on the steps of the cdf between sample quantiles (pooled where the expected
    # count is below 5)
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 0.999, 300)).round())
    probs = np.diff(np.concatenate([[0.0], np.asarray(D.cdf(sd.asarray(edges))), [1.0]]))
    counts = np.histogram(x, bins=np.concatenate([[-np.inf], edges + 0.5, [np.inf]]))[0]
    keep = probs * N >= 5
    expected, observed = probs[keep] * N, counts[keep]
    rest_e, rest_o = N - expected.sum(), N - observed.sum()
    chi = ((observed - expected) ** 2 / expected).sum()
    dof = keep.sum() - 1
    if rest_e >= 5:
        chi += (rest_o - rest_e) ** 2 / rest_e
        dof += 1
    p = st.chi2.sf(chi, max(dof, 1))
    assert p > P_MIN, (name, kw, p)


def test_shapes_below_the_smallest_double():
    # Gamma(a = 1e-3): 47.5% of the mass is below 5e-324 and rounds to 0; the rest follows
    # the conditional cdf
    D = S.Gamma(a=1e-3)
    x = np.asarray(D.sample(200000, rng=sd.random.default_rng(7)))
    c0 = float(D.cdf(5e-324))
    assert abs((x == 0).mean() - c0) < 4 * np.sqrt(c0 * (1 - c0) / x.size)
    pos = x[x > 0]
    assert st.kstest(pos, lambda v: (np.asarray(D.cdf(sd.asarray(v))) - c0) / (1 - c0)).pvalue > P_MIN


def test_zipf_beyond_float64():
    # a = 1.01: P(X > DBL_MAX) = 8.2e-4, those samples are inf (as the icdf beyond float64, T25)
    D = S.Zipf(a=1.01)
    x = np.asarray(D.sample(1000000, rng=sd.random.default_rng(3)))
    p = float(D.ccdf(np.finfo(np.float64).max))
    assert abs(np.isinf(x).mean() - p) < 4 * np.sqrt(p / x.size)


def _one(name):
    return getattr(S, name)(**next(kw for n, kw in CONTINUOUS + DISCRETE if n == name))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_same_bits_on_any_thread_count(name):
    D = _one(name)
    before = sd.get_num_threads()
    try:
        outs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            outs.append(np.asarray(D.sample(50000, rng=sd.random.default_rng(11))).view(np.uint64))
    finally:
        sd.set_num_threads(before)
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2])


@pytest.mark.parametrize("name", ALL_NAMES)
def test_split_calls_continue_the_stream(name):
    D = _one(name)
    whole = np.asarray(D.sample(3000, rng=sd.random.default_rng(5)))
    rng = sd.random.default_rng(5)
    parts = np.concatenate([np.asarray(D.sample(1000, rng=rng)), np.asarray(D.sample(2000, rng=rng))])
    assert np.array_equal(whole.view(np.uint64), parts.view(np.uint64))
    assert rng.state["position"] == 3000


@pytest.mark.parametrize("name", ALL_NAMES)
def test_array_parameters_match_scalar_ones(name):
    # parameters broadcast to (3,): element (i, j) is stream element 3 i + j, the same as a
    # scalar call's element 3 i + j
    kw = next(kw for n, kw in CONTINUOUS + DISCRETE if n == name)
    scalar = np.asarray(getattr(S, name)(**kw).sample(12, rng=sd.random.default_rng(9)))
    arr = getattr(S, name)(**{k: sd.asarray([v, v, v]) for k, v in kw.items()})
    got = np.asarray(arr.sample(4, rng=sd.random.default_rng(9)))
    assert got.shape == (4, 3)
    assert np.array_equal(got.reshape(-1).view(np.uint64), scalar.view(np.uint64))


@pytest.mark.parametrize("name", [n for n, kw in CONTINUOUS + DISCRETE if len(kw) >= 2])
@pytest.mark.parametrize("which", ["first", "last"])
def test_scalar_beside_array_parameters(name, which):
    # one parameter an array, the others Python scalars: the same bits as all of them arrays
    # (a scalar first parameter used to make the kernel see one parameter set)
    kw = next(kw for n, kw in CONTINUOUS + DISCRETE if n == name)
    keys = list(kw)
    arr_key = keys[-1] if which == "first" else keys[0]
    v0 = kw[arr_key]  # three different values, so that reading only the first one shows
    vals = [v0, v0 + 1, v0 + 2] if float(v0).is_integer() else [v0, v0 * 0.999, v0 * 0.998]
    mixed = {k: (sd.asarray(vals) if k == arr_key else v) for k, v in kw.items()}
    full = {k: sd.asarray(vals if k == arr_key else [v, v, v]) for k, v in kw.items()}
    a = np.asarray(getattr(S, name)(**mixed).sample(4, rng=sd.random.default_rng(9)))
    b = np.asarray(getattr(S, name)(**full).sample(4, rng=sd.random.default_rng(9)))
    assert a.shape == b.shape == (4, 3)
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))


def test_scalar_beside_array_parameters_values():
    # the reported cases: each column follows its own parameters
    rng = sd.random.default_rng(3)
    x = np.asarray(S.Normal(mu=0.0, sigma=sd.asarray([1e-9, 1000.0])).sample(4000, rng=rng))
    assert np.std(x[:, 0]) < 1e-8 and 900 < np.std(x[:, 1]) < 1100
    b = np.asarray(S.Binomial(n=10, p=sd.asarray([0.01, 0.99])).sample(4000, rng=rng))
    assert b[:, 0].mean() < 0.5 and b[:, 1].mean() > 9.5
    v = np.asarray(S.Normal(sigma=sd.asarray([1.0, -1.0])).sample(3, rng=rng))
    assert np.all(np.isfinite(v[:, 0])) and np.all(np.isnan(v[:, 1]))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_one_float_is_the_first_element(name):
    D = _one(name)
    one = D.sample(rng=sd.random.default_rng(13))
    assert isinstance(one, float)
    first = float(np.asarray(D.sample(4, rng=sd.random.default_rng(13)))[0])
    assert np.array([one]).view(np.uint64)[0] == np.array([first]).view(np.uint64)[0]


def test_shapes():
    rng = sd.random.default_rng(1)
    assert isinstance(S.Normal().sample(rng=rng), float)
    assert S.Normal().sample(5, rng=rng).shape == (5,)
    assert S.Normal().sample((2, 3), rng=rng).shape == (2, 3)
    assert S.Normal().sample((0,), rng=rng).shape == (0,)
    X = S.Normal(mu=sd.asarray([0.0, 10.0, 20.0]), sigma=sd.asarray([[1.0], [2.0]]))
    assert X.sample(rng=rng).shape == (2, 3)
    assert X.sample((4,), rng=rng).shape == (4, 2, 3)
    # the parameters shape the trailing axes: the third column is near 20
    x = np.asarray(S.Normal(mu=sd.asarray([0.0, 10.0, 20.0])).sample(1000, rng=rng))
    assert np.allclose(x.mean(axis=0), [0, 10, 20], atol=0.2)
    assert S.Binomial(n=10.0, p=sd.asarray([0.1, 0.5])).sample(3, rng=rng).dtype == sd.float64


def test_invalid_parameters_give_nan():
    rng = sd.random.default_rng(1)
    assert np.isnan(S.Normal(sigma=-1.0).sample(rng=rng))
    x = np.asarray(S.Gamma(a=sd.asarray([1.0, -1.0, float("nan"), 2.0])).sample(rng=rng))
    assert np.isfinite(x[[0, 3]]).all() and np.isnan(x[[1, 2]]).all()
    assert np.isnan(np.asarray(S.Binomial(n=2.5, p=0.5).sample(3, rng=rng))).all()


def test_degenerate_parameters():
    rng = sd.random.default_rng(1)
    assert np.all(np.asarray(S.Binomial(n=7.0, p=1.0).sample(10, rng=rng)) == 7)
    assert np.all(np.asarray(S.Binomial(n=7.0, p=0.0).sample(10, rng=rng)) == 0)
    assert np.all(np.asarray(S.Poisson(mu=0.0).sample(10, rng=rng)) == 0)
    assert np.all(np.asarray(S.Geometric(p=1.0).sample(10, rng=rng)) == 1)
    assert np.all(np.asarray(S.NegativeBinomial(n=3.0, p=1.0).sample(10, rng=rng)) == 0)
    assert np.all(np.asarray(S.Hypergeometric(total=10.0, successes=10.0, draws=4.0).sample(10, rng=rng)) == 4)
    assert np.all(np.asarray(S.DiscreteUniform(low=3.0, high=4.0).sample(10, rng=rng)) == 3)


def test_rng_is_required_and_checked():
    with pytest.raises(TypeError):
        S.Normal().sample(3)
    with pytest.raises(TypeError, match="sd.random.Generator"):
        S.Normal().sample(3, rng=1)
    with pytest.raises(TypeError, match="sd.random.Generator"):
        S.Normal().sample(3, rng=np.random.default_rng(1))
    with pytest.raises(TypeError):
        S.Normal().sample(2.5, rng=sd.random.default_rng(1))
    with pytest.raises(ValueError):
        S.Normal().sample(-1, rng=sd.random.default_rng(1))


def test_samples_advance_the_generator():
    rng = sd.random.default_rng(2)
    a = np.asarray(S.Normal().sample(100, rng=rng))
    b = np.asarray(S.Normal().sample(100, rng=rng))
    assert not np.array_equal(a, b)
    assert rng.state["position"] == 200
    # the same position in another generator's stream gives the same values
    r2 = sd.random.default_rng(2)
    r2.random(100)
    assert np.array_equal(np.asarray(S.Normal().sample(100, rng=r2)), b)


def test_rejections_first_test_keeps_the_exact_decision():
    """Poisson (PTRS) and binomial (BTRS) decide in plain double where a margin allows and by the
    double-double log pmf otherwise: their samples follow the pmf across the switch points and
    for huge parameters (where the margin always defers)."""
    r = sd.random.default_rng(31)
    for D, lo in (
        (S.Poisson(mu=10.0), 0),
        (S.Poisson(mu=250.5), 150),
        (S.Binomial(n=40.0, p=0.25), 0),
        (S.Binomial(n=3000.0, p=0.97), 2850),
        (S.NegativeBinomial(n=7.0, p=0.2), 0),
    ):
        x = np.asarray(D.sample(400000, rng=r))
        ks = np.arange(lo, lo + 120.0)
        pm = np.asarray(D.pmf(sd.asarray(ks)))
        keep = pm * len(x) >= 20.0
        obs = np.array([(x == k).sum() for k in ks[keep]])
        chi2 = ((obs - pm[keep] * len(x)) ** 2 / (pm[keep] * len(x))).sum()
        assert st.chi2.sf(chi2, keep.sum()) > 1e-6, D
    for D, m, s in ((S.Poisson(mu=1e15), 1e15, 10**7.5), (S.Binomial(n=1e15, p=0.5), 5e14, 2.5e14**0.5)):
        x = np.asarray(D.sample(20000, rng=r))
        assert abs(x.mean() - m) < 6 * s / 20000**0.5 and 0.97 < x.std() / s < 1.03


def test_reworked_samplers_follow_their_distributions():
    """Gamma and beta below shape 1 (powers of uniforms as exponentials), the noncentral
    chi-square as a gamma and a shifted normal square, and the noncentral F on top of it."""
    r = sd.random.default_rng(32)
    cases = [
        S.Gamma(a=0.5),
        S.Gamma(a=0.07, scale=3.0),
        S.Gamma(a=0.01),
        S.Beta(a=0.5, b=0.5),
        S.Beta(a=0.07, b=0.9),
        S.Beta(a=0.05, b=0.9),
        S.Beta(a=0.3, b=4.0),
        S.NoncentralChiSquared(df=5.0, nc=2.0),
        S.NoncentralChiSquared(df=1.0, nc=3.0),
        S.NoncentralChiSquared(df=0.4, nc=3.0),
        S.NoncentralChiSquared(df=7.5, nc=0.0),
        S.NoncentralF(dfn=5.0, dfd=7.0, nc=2.0),
        S.NoncentralF(dfn=0.5, dfd=1.5, nc=1.0),
        S.ChiSquared(df=0.3),
        S.InverseGamma(a=0.4),
        S.NegativeBinomial(n=0.5, p=0.4),
        S.StudentT(df=0.8),
    ]
    for D in cases:
        x = np.asarray(D.sample(200000, rng=r))
        if isinstance(D, S.NegativeBinomial):
            ks = np.arange(0.0, 12.0)
            obs = np.array([(x == k).sum() for k in ks])
            ex = np.asarray(D.pmf(sd.asarray(ks))) * len(x)
            assert st.chi2.sf(((obs - ex) ** 2 / ex).sum(), len(ks)) > 1e-6
            continue
        u = np.asarray(D.cdf(sd.asarray(x)))
        assert st.kstest(u, "uniform").pvalue > 1e-6, D


def test_extreme_parameters_return_in_bounded_time():
    import time

    rng = sd.random.default_rng(11)
    t0 = time.perf_counter()
    # Zipf: 2^(a-1) overflows (inf / inf in the acceptance test: never accepted before)
    assert np.all(np.asarray(S.Zipf(a=1e300).sample(1000, rng=rng)) == 1.0)
    assert np.all(np.asarray(S.Zipf(a=2000.0).sample(1000, rng=rng)) == 1.0)
    # hypergeometric: the walk from the mode is bounded (a standard deviation above 65536,
    # or a population beyond 2^53: NaN); below, samples as before
    for total, succ, draws in (
        (1e17, 5e16, 5e16),
        (1e100, 5e99, 5e99),
        (1e300, 1e299, 1e299),
        (1e15, 5e14, 5e14),
        (2.0**53, 9007199.0, 2.0**52),
    ):
        v = np.asarray(S.Hypergeometric(total=total, successes=succ, draws=draws).sample(2000, rng=rng))
        assert np.isnan(v).all(), (total, succ, draws)
    v = np.asarray(S.Hypergeometric(total=1e15, successes=1e6, draws=5e14).sample(2000, rng=rng))
    assert abs(v.mean() - 5e5) < 100 and abs(v.std() - 500.0) < 50
    # beta with subnormal shapes: both logs overflow (the rejection compared NaN forever)
    v = np.asarray(S.Beta(a=5e-324, b=5e-324).sample(4000, rng=rng))
    assert set(np.unique(v)) <= {0.0, 0.5, 1.0} and 0.4 < v.mean() < 0.6
    v = np.asarray(S.Beta(a=1e-310, b=3e-310).sample(4000, rng=rng))
    assert set(np.unique(v)) <= {0.0, 0.5, 1.0} and 0.2 < v.mean() < 0.3  # P(1) = a / (a + b)
    v = np.asarray(S.BetaBinomial(n=7.0, a=5e-324, b=5e-324).sample(4000, rng=rng))
    assert set(np.unique(v)) <= {0.0, 7.0} and 0.4 < (v == 7.0).mean() < 0.6
    # more integers than a word indexes: NaN (uniform on [0, 2^64) before)
    assert np.isnan(np.asarray(S.DiscreteUniform(low=0.0, high=1e100).sample(100, rng=rng))).all()
    assert np.isnan(np.asarray(S.DiscreteUniform(low=-1e300, high=1e300).sample(100, rng=rng))).all()
    v = np.asarray(S.DiscreteUniform(low=0.0, high=2.0**60).sample(4000, rng=rng))
    assert 0.0 <= v.min() and v.max() < 2.0**60 and 0.45 < v.mean() / 2.0**60 < 0.55
    # F with a subnormal dfn: dfd / dfn overflowed (NaN in 99% of the samples); since T49 NaN on
    # purpose (as its values), and just above 2^-1021 the ratio holds
    v = np.asarray(S.F(dfn=1e-310, dfd=0.5).sample(2000, rng=rng))
    assert np.isnan(v).all()
    v = np.asarray(S.F(dfn=2.0**-1021, dfd=0.5).sample(2000, rng=rng))
    assert not np.isnan(v).any() and (v >= 0.0).all()
    # (the noncentral F with a df below 2^-1021 is NaN, as its values: user decision 2026-11-19, T45)
    v = np.asarray(S.NoncentralF(dfn=5e-324, dfd=0.5, nc=1.0).sample(2000, rng=rng))
    assert np.isnan(v).all()
    v = np.asarray(S.NoncentralF(dfn=2.0**-1021, dfd=0.5, nc=1.0).sample(2000, rng=rng))
    assert not np.isnan(v).any() and (v >= 0.0).all()
    # binomial by inversion with a count beyond 2^53
    v = np.asarray(S.Binomial(n=1e300, p=1e-300).sample(200000, rng=rng))
    assert np.isfinite(v).all() and abs(v.mean() - 1.0) < 0.02
    assert time.perf_counter() - t0 < 5.0
