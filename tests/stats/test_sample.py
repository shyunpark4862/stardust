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
    ("Normal", dict(mu=1.0, sigma=2.0)), ("LogNormal", dict(mu=0.5, sigma=1.5)), ("Exponential", dict(scale=2.0)),
    ("Uniform", dict(a=-1.0, b=3.0)), ("Laplace", dict(mu=0.5, sigma=2.0)), ("Cauchy", dict(mu=0.0, sigma=1.0)),
    ("Logistic", dict(mu=0.0, sigma=1.0)), ("Gumbel", dict(mu=1.0, sigma=2.0)), ("GEV", dict(c=0.3)),
    ("GEV", dict(c=-0.4)), ("GEV", dict(c=0.0)), ("Weibull", dict(c=0.7)), ("Pareto", dict(b=2.5)),
    ("Rayleigh", dict(scale=1.5)), ("Gamma", dict(a=3.0, scale=2.0)), ("Gamma", dict(a=0.3)), ("Gamma", dict(a=1e6)),
    ("ChiSquared", dict(df=0.5)), ("ChiSquared", dict(df=7.0)), ("InverseGamma", dict(a=0.5)),
    ("InverseGamma", dict(a=4.0, scale=3.0)), ("Beta", dict(a=0.2, b=0.3)), ("Beta", dict(a=2.0, b=5.0)), ("Beta", dict(a=0.5, b=3.0)),
    ("StudentT", dict(df=0.5)), ("StudentT", dict(df=5.0)), ("StudentT", dict(df=1e4)), ("F", dict(dfn=0.7, dfd=3.0)),
    ("F", dict(dfn=10.0, dfd=20.0)), ("NoncentralChiSquared", dict(df=0.5, nc=3.0)),
    ("NoncentralChiSquared", dict(df=4.0, nc=50.0)), ("NoncentralChiSquared", dict(df=4.0, nc=0.0)),
    ("NoncentralT", dict(df=3.0, nc=-2.0)), ("NoncentralT", dict(df=30.0, nc=5.0)),
    ("NoncentralF", dict(dfn=3.0, dfd=8.0, nc=4.0)),
]
DISCRETE = [
    ("Bernoulli", dict(p=0.3)), ("DiscreteUniform", dict(low=-3.0, high=7.0)), ("Geometric", dict(p=0.2)),
    ("Geometric", dict(p=1e-6)), ("Poisson", dict(mu=0.7)), ("Poisson", dict(mu=9.9)), ("Poisson", dict(mu=10.0)),
    ("Poisson", dict(mu=1234.5)), ("Poisson", dict(mu=1e12)), ("Binomial", dict(n=20.0, p=0.3)),
    ("Binomial", dict(n=20.0, p=0.9)), ("Binomial", dict(n=1e4, p=0.3)), ("Binomial", dict(n=1e9, p=0.7)),
    ("Binomial", dict(n=1e6, p=1e-5)), ("NegativeBinomial", dict(n=5.0, p=0.3)),
    ("NegativeBinomial", dict(n=0.3, p=0.01)), ("Hypergeometric", dict(total=1000.0, successes=300.0, draws=100.0)),
    ("Hypergeometric", dict(total=50.0, successes=45.0, draws=40.0)),
    ("Hypergeometric", dict(total=1e7, successes=4e6, draws=5e5)), ("BetaBinomial", dict(n=20.0, a=2.0, b=3.0)),
    ("BetaBinomial", dict(n=100.0, a=0.3, b=0.2)), ("LogSeries", dict(p=0.9)), ("LogSeries", dict(p=0.999)),
    ("LogSeries", dict(p=1e-3)), ("Zipf", dict(a=2.0)), ("Zipf", dict(a=1.3)), ("Zipf", dict(a=6.0)),
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
