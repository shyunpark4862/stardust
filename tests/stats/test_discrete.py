"""The discrete distributions' behaviour (docs/api/stats.md §2, docs/design/stats.md T25):
pmf/logpmf, the cdfs at non-integers, the quantile conventions, modes with ties, degenerate
parameters, and agreement with scipy in the body of each distribution."""

import inspect
import math

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

from . import refdata

S = sd.stats

# (class, parameters, scipy frozen distribution, support)
CASES = [
    (S.Binomial, {"n": 20, "p": 0.3}, st.binom(20, 0.3), (0.0, 20.0)),
    (S.Poisson, {"mu": 4.5}, st.poisson(4.5), (0.0, math.inf)),
    (S.Geometric, {"p": 0.2}, st.geom(0.2), (1.0, math.inf)),
    (S.NegativeBinomial, {"n": 2.5, "p": 0.3}, st.nbinom(2.5, 0.3), (0.0, math.inf)),
    (S.Hypergeometric, {"total": 50, "successes": 12, "draws": 20}, st.hypergeom(50, 12, 20), (0.0, 12.0)),
    (S.Bernoulli, {"p": 0.3}, st.bernoulli(0.3), (0.0, 1.0)),
    (S.DiscreteUniform, {"low": 2, "high": 9}, st.randint(2, 9), (2.0, 8.0)),
    (S.BetaBinomial, {"n": 20, "a": 2.5, "b": 1.7}, st.betabinom(20, 2.5, 1.7), (0.0, 20.0)),
    (S.LogSeries, {"p": 0.7}, st.logser(0.7), (1.0, math.inf)),
    (S.Zipf, {"a": 6.5}, st.zipf(6.5), (1.0, math.inf)),
]
IDS = [c[0].__name__ for c in CASES]


@pytest.fixture
def arb():
    """python-flint's arb at 256 bits: the references computed in a test are Arb balls."""
    flint = pytest.importorskip("flint")
    old = flint.ctx.prec
    flint.ctx.prec = 256
    yield flint.arb
    flint.ctx.prec = old


def arb_error(got, ref, mixed=False):
    """An upper bound of |got - ref| / |ref| (mixed: / max(1, |ref|)) for a float and an arb."""
    one = type(ref)(1)
    return float((abs(type(ref)(float(got)) - ref) / (max(one, abs(ref)) if mixed else abs(ref))).upper())


INPUT = ["pmf", "logpmf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf"]
PARAM = ["mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy"]


def test_exports():
    for c in CASES:
        assert issubclass(c[0], S.DiscreteDistribution)
        assert c[0].__name__ in S.__all__
    assert not hasattr(S.Binomial(n=3, p=0.5), "pdf")  # pmf in place of pdf (T9)


@pytest.mark.parametrize("cls, params, _, __", CASES, ids=IDS)
def test_keyword_only_and_required(cls, params, _, __):
    sig = inspect.signature(cls)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())
    assert all(p.default is inspect.Parameter.empty for p in sig.parameters.values())
    with pytest.raises(TypeError):
        cls(*params.values())


@pytest.mark.parametrize("cls, params, _, __", CASES, ids=IDS)
def test_scalars_are_floats(cls, params, _, __):
    X = cls(**params)
    for m in INPUT:
        assert type(getattr(X, m)(0.3 if not m.startswith("ilog") else -0.3)) is float, m
    for m in PARAM:
        assert type(getattr(X, m)()) is float, m


@pytest.mark.parametrize("cls, params, ref, support", CASES, ids=IDS)
def test_support_and_quantile_conventions(cls, params, ref, support):
    X = cls(**params)
    lo, hi = support
    assert X.support() == support
    assert (X.icdf(0.0), X.icdf(1.0), X.iccdf(0.0), X.iccdf(1.0)) == (lo, hi, hi, lo)
    assert (X.ilogcdf(0.0), X.ilogcdf(-math.inf), X.ilogccdf(0.0), X.ilogccdf(-math.inf)) == (hi, lo, lo, hi)
    assert math.isnan(X.icdf(1.5)) and math.isnan(X.ilogcdf(0.5)) and math.isnan(X.pmf(math.nan))
    assert X.pmf(lo - 1.0) == 0.0 and X.logpmf(lo - 1.0) == -math.inf
    assert (X.cdf(lo - 0.5), X.ccdf(lo - 0.5)) == (0.0, 1.0)
    # a non-integer: pmf 0, the cdfs those at floor(x)
    k = lo + 1.0
    assert X.pmf(k + 0.5) == 0.0 and X.logpmf(k + 0.5) == -math.inf
    assert X.cdf(k + 0.5) == X.cdf(k) and X.ccdf(k + 0.5) == X.ccdf(k) and X.logcdf(k + 0.7) == X.logcdf(k)
    # icdf(p) is the smallest k with cdf(k) >= p; iccdf(p) the smallest with ccdf(k) <= p
    for p in (0.05, 0.3, 0.5, 0.77, 0.99):
        q = X.icdf(p)
        assert X.cdf(q) >= p and (q == lo or X.cdf(q - 1.0) < p)
        r = X.iccdf(p)
        assert X.ccdf(r) <= p and (r == lo or X.ccdf(r - 1.0) > p)
    assert X.median() == X.icdf(0.5)  # T25


@pytest.mark.parametrize("cls, params, ref, support", CASES, ids=IDS)
def test_body_agrees_with_scipy(cls, params, ref, support):
    X = cls(**params)
    p = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    k = ref.ppf(p)
    for ours, theirs in (
        ("pmf", "pmf"),
        ("logpmf", "logpmf"),
        ("cdf", "cdf"),
        ("logcdf", "logcdf"),
        ("ccdf", "sf"),
        ("logccdf", "logsf"),
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(X, ours)(sd.asarray(k))), getattr(ref, theirs)(k), rtol=1e-12, err_msg=ours
        )
    assert np.asarray(X.icdf(sd.asarray(p))).tolist() == ref.ppf(p).tolist()
    assert np.asarray(X.iccdf(sd.asarray(p))).tolist() == ref.isf(p).tolist()
    assert np.asarray(X.ilogcdf(sd.asarray(np.log(p)))).tolist() == ref.ppf(p).tolist()
    m, v, s, kk = (float(t) for t in ref.stats(moments="mvsk"))
    np.testing.assert_allclose([X.mean(), X.variance(), X.skewness(), X.kurtosis()], [m, v, s, kk], rtol=1e-11)
    assert X.median() == ref.median()
    if cls is not S.Zipf:  # scipy's zipf entropy is a sum that does not converge (7 digits)
        assert math.isclose(X.entropy(), ref.entropy(), rel_tol=1e-12)


def test_modes_with_ties_are_nan():
    # two equal modes (T25 with T21): (n + 1) p an integer, an integer mu, a uniform
    assert math.isnan(S.Binomial(n=9, p=0.5).mode())
    assert S.Binomial(n=10, p=0.5).mode() == 5.0
    assert math.isnan(S.Poisson(mu=3.0).mode()) and S.Poisson(mu=3.5).mode() == 3.0 and S.Poisson(mu=0.5).mode() == 0.0
    assert math.isnan(S.NegativeBinomial(n=3.0, p=0.5).mode())  # (n - 1) q / p = 2
    assert math.isnan(S.DiscreteUniform(low=0, high=6).mode()) and S.DiscreteUniform(low=4, high=5).mode() == 4.0
    assert math.isnan(S.Bernoulli(p=0.5).mode()) and S.Bernoulli(p=0.6).mode() == 1.0
    assert math.isnan(S.BetaBinomial(n=10, a=1.0, b=1.0).mode())  # uniform on 0..10
    assert math.isnan(S.Hypergeometric(total=10, successes=5, draws=3).mode())  # (n + 1)(K + 1) / (N + 2) = 2
    assert S.Geometric(p=0.3).mode() == 1.0 and S.Zipf(a=2.0).mode() == 1.0 and S.LogSeries(p=0.9).mode() == 1.0


def test_degenerate_parameters():
    # point masses: p = 0 or 1, n = 0, mu = 0
    for X, at in (
        (S.Binomial(n=7, p=0.0), 0.0),
        (S.Binomial(n=7, p=1.0), 7.0),
        (S.Binomial(n=0, p=0.4), 0.0),
        (S.Poisson(mu=0.0), 0.0),
        (S.Bernoulli(p=1.0), 1.0),
        (S.Geometric(p=1.0), 1.0),
        (S.NegativeBinomial(n=2.0, p=1.0), 0.0),
        (S.NegativeBinomial(n=0.0, p=0.5), 0.0),
    ):
        assert X.pmf(at) == 1.0 and X.logpmf(at) == 0.0 and X.cdf(at) == 1.0 and X.ccdf(at - 1.0) == 1.0
        assert X.mean() == at and X.variance() == 0.0 and X.mode() == at and X.entropy() == 0.0
        assert math.isnan(X.skewness()) and math.isnan(X.kurtosis())  # undefined for a point mass


def test_parameters_outside_the_domain():
    for X in (
        S.Binomial(n=5.5, p=0.3),
        S.Binomial(n=5, p=1.5),
        S.Poisson(mu=-1.0),
        S.Geometric(p=0.0),
        S.NegativeBinomial(n=-1.0, p=0.5),
        S.Hypergeometric(total=10, successes=11, draws=3),
        S.DiscreteUniform(low=3, high=3),
        S.DiscreteUniform(low=0.5, high=3),
        S.BetaBinomial(n=4, a=0.0, b=1.0),
        S.LogSeries(p=1.0),
        S.Zipf(a=1.0),
    ):
        assert math.isnan(X.pmf(1.0)) and math.isnan(X.cdf(1.0)) and math.isnan(X.mean()) and math.isnan(X.icdf(0.5))


def test_infinite_quantiles_and_moments():
    # a quantile beyond float64 is inf; moments that diverge are inf (the rule of §2)
    assert S.Zipf(a=1.001).iccdf(1e-10) == math.inf
    assert S.Zipf(a=1.5).mean() == math.inf and math.isnan(S.Zipf(a=2.0).variance())
    assert S.Zipf(a=2.5).variance() == math.inf and S.Zipf(a=3.5).skewness() == math.inf
    assert S.Zipf(a=4.5).kurtosis() == math.inf and math.isnan(S.Zipf(a=3.0).kurtosis())


# ---- the chunk kernels: memos, fast paths, search starts (stats/discrete.hpp) ----

# wide parameter sets (nearly every k in an array differs: the fast paths and the searches'
# normal starts) beside CASES (small supports: the memos)
WIDE = [
    (S.Binomial, {"n": 10**9, "p": 0.3}),
    (S.Poisson, {"mu": 3e8}),
    (S.Geometric, {"p": 1e-6}),
    (S.NegativeBinomial, {"n": 2.5e6, "p": 1e-3}),
    (S.Hypergeometric, {"total": 5 * 10**6, "successes": 12 * 10**5, "draws": 2 * 10**6}),
    (S.BetaBinomial, {"n": 3000, "a": 2.5, "b": 1.7}),
    (S.LogSeries, {"p": 1 - 1e-5}),
]
BOTH = [(c[0], c[1]) for c in CASES] + WIDE
BOTH_IDS = IDS + ["wide-" + c[0].__name__ for c in WIDE]


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64).tolist()


def _points(X, n, seed=0):
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0, 1, n - 8), [0.0, 1.0, 1e-300, 1e-17, 0.5, 1 - 1e-16, 1e-5, 0.999999]])
    k = np.asarray(X.icdf(sd.asarray(p)))
    k[~np.isfinite(k)] = 1e300
    x = np.concatenate([k, k[:16] + 0.5, [-1.0, math.nan, math.inf, 0.0, 1.0, 13.0, 12.0]])
    return x, p


@pytest.mark.parametrize("cls, params", BOTH, ids=BOTH_IDS)
def test_array_scalar_and_parameter_arrays_give_the_same_bits(cls, params):
    # f(x)[i] == f(x[i]) through the memos and the fast paths; a broadcast scalar parameter and
    # an array of equal values; the same array twice (the second time out of the memo)
    X = cls(**params)
    n = 300 if cls in (S.Hypergeometric, S.BetaBinomial) else 1500
    x, p = _points(X, n)
    XA = cls(**{name: sd.asarray(np.full(x.size, float(v))) for name, v in params.items()})
    PA = cls(**{name: sd.asarray(np.full(p.size, float(v))) for name, v in params.items()})
    for m in INPUT:
        inverse = m.startswith("i")
        with np.errstate(divide="ignore"):
            arg = (np.log(p) if "log" in m else p) if inverse else x
            got = np.asarray(getattr(X, m)(sd.asarray(arg)))
        assert _bits(getattr(X, m)(sd.asarray(arg))) == _bits(got), m
        assert _bits(getattr(PA if inverse else XA, m)(sd.asarray(arg))) == _bits(got), m
        for i in range(0, arg.size, 7 if arg.size < 500 else 23):
            assert _bits([getattr(X, m)(float(arg[i]))]) == _bits(got[i : i + 1]), (m, i, arg[i])


@pytest.mark.parametrize("cls, params", BOTH, ids=BOTH_IDS)
def test_threads_give_the_same_bits(cls, params):
    X = cls(**params)
    slow = cls in (S.Hypergeometric, S.BetaBinomial) and params in [w[1] for w in WIDE]
    x, p = _points(X, 400 if slow else 20000, seed=1)
    before = sd.get_num_threads()
    try:
        for m in INPUT:
            with np.errstate(divide="ignore"):
                arg = (np.log(p) if "log" in m else p) if m.startswith("i") else x
            runs = []
            for t in (1, 3, 10):
                sd.set_num_threads(t)
                with np.errstate(divide="ignore"):
                    runs.append(_bits(getattr(X, m)(sd.asarray(arg))))
            assert runs[0] == runs[1] == runs[2], m
    finally:
        sd.set_num_threads(before)


@pytest.mark.parametrize("cls, params", WIDE, ids=[c[0].__name__ for c in WIDE])
def test_wide_quantiles_are_the_smallest_k(cls, params):
    # the searches from the normal (beta, closed-form) starts: the definition of T25
    X = cls(**params)
    p = np.array([1e-300, 1e-30, 1e-6, 0.01, 0.2, 0.5, 0.5 + 1e-9, 0.8, 0.999, 1 - 1e-12])
    lo = X.support()[0]
    q = np.asarray(X.icdf(sd.asarray(p)))
    r = np.asarray(X.iccdf(sd.asarray(p)))
    lq = np.asarray(X.ilogcdf(sd.asarray(np.log(p))))
    assert lq.tolist() == q.tolist()
    for pi, qi, ri in zip(p, q, r):  # compared on the side below 1/2, as the searches do (1 - p is exact)
        if pi <= 0.5:
            assert X.cdf(qi) >= pi and (qi == lo or X.cdf(qi - 1.0) < pi), pi
            assert X.ccdf(ri) <= pi and (ri == lo or X.ccdf(ri - 1.0) > pi), pi
        else:
            assert X.ccdf(qi) <= 1.0 - pi and (qi == lo or X.ccdf(qi - 1.0) > 1.0 - pi), pi
            assert X.cdf(ri) >= 1.0 - pi and (ri == lo or X.cdf(ri - 1.0) < 1.0 - pi), pi


def test_saddle_point_fast_paths_against_arb(arb):
    # pmf and logpmf in float64 around the mean of wide distributions (the fast paths), at
    # their limits (|t| = 1/8, counts of 13) and beyond (the double-double definitions)
    def lg(v):
        return arb(v).lgamma()

    def lchoose(n, k):
        return lg(n + 1) - lg(k + 1) - lg(n - k + 1)

    def binom(k, n, p):
        return lchoose(n, k) + k * arb(p).log() + (n - k) * (-arb(p)).log1p()

    def nbinom(k, n, p):
        n = arb(n)
        return (k + n).lgamma() - n.lgamma() - lg(k + 1) + n * arb(p).log() + k * (-arb(p)).log1p()

    def poisson(k, mu):
        return k * arb(mu).log() - arb(mu) - lg(k + 1)

    def hyper(k, N, K, n):
        return lchoose(K, k) + lchoose(N - K, n - k) - lchoose(N, n)

    def geom(k, p):
        return (k - 1) * (-arb(p)).log1p() + arb(p).log()

    def logser(k, p):
        p = arb(p)
        return k * p.log() - arb(k).log() - (-(-p).log1p()).log()

    rng = np.random.default_rng(5)
    cases = []
    for _ in range(40):
        n = float(math.floor(10 ** rng.uniform(1.5, 15.5)))
        p = float(rng.uniform(0.001, 0.999))
        z = rng.uniform(-6, 6, 6)
        sdv = math.sqrt(n * p * (1 - p))
        ks = np.clip(np.floor(n * p + z * sdv), 0, n).tolist() + [
            math.floor(n * p * 1.124),
            math.floor(n * p * 0.876),
            13.0,
            12.0,
            n - 13.0,
        ]
        cases += [(S.Binomial, {"n": n, "p": p}, k, binom) for k in ks if 0 <= k <= n]
        r = float(10 ** rng.uniform(0.5, 12))
        mean, sdn = r * (1 - p) / p, math.sqrt(r * (1 - p)) / p
        ks = np.clip(np.floor(mean + z * sdn), 0, None).tolist() + [13.0, 12.0]
        cases += [(S.NegativeBinomial, {"n": r, "p": p}, k, nbinom) for k in ks]
        mu = float(10 ** rng.uniform(0, 15))
        ks = np.clip(np.floor(mu + z * math.sqrt(mu)), 0, None).tolist() + [13.0, 12.0, math.floor(mu * 1.124)]
        cases += [(S.Poisson, {"mu": mu}, k, poisson) for k in ks]
        N = float(math.floor(10 ** rng.uniform(2.2, 15)))
        K, d = float(math.floor(N * rng.uniform(0.05, 0.95))), float(math.floor(N * rng.uniform(0.05, 0.95)))
        H = S.Hypergeometric(total=N, successes=K, draws=d)
        lo, hi = H.support()
        ks = np.clip(np.floor(H.mean() + z * H.standard_deviation()), lo, hi).tolist()
        cases += [(S.Hypergeometric, {"total": N, "successes": K, "draws": d}, k, hyper) for k in ks]
        pg = float(10 ** rng.uniform(-9, -0.01))
        cases += [(S.Geometric, {"p": pg}, float(math.ceil(e / pg)), geom) for e in (0.01, 1.0, 30.0, 600.0)]
        pl = float(1 - 10 ** rng.uniform(-9, -0.01))
        cases += [(S.LogSeries, {"p": pl}, float(math.ceil(e / (1 - pl))), logser) for e in (0.01, 1.0, 30.0, 600.0)]
    worst = 0.0
    for cls, params, k, ref in cases:
        X = cls(**params)
        args = (
            int(k),
            *(
                int(v) if cls is S.Hypergeometric or (cls is S.Binomial and name == "n") else v
                for name, v in params.items()
            ),
        )
        t = ref(*args)
        got_l = np.asarray(X.logpmf(sd.asarray(np.full(9, k))))[0]  # an array: the chunk kernel
        got = np.asarray(X.pmf(sd.asarray(np.full(9, k))))[0]
        el = arb_error(got_l, t, mixed=True)
        ep = arb_error(got, t.exp()) if got > 1e-300 else 0.0
        assert el < 3e-15 and ep < 5e-15, (cls.__name__, params, k, el, ep)
        worst = max(worst, el, ep)
    assert worst < 5e-15


def test_sum_cdfs_against_arb(arb):
    # the tail sums of wide hypergeometric and beta-binomial cdfs (thousands of terms)
    def lg(v):
        return arb(v).lgamma()

    def total(first, ratio, k, lo):
        t = s = first.exp()
        j = k
        while j > lo:
            t *= ratio(j)
            j -= 1
            s += t
            if t < s * arb(10) ** -30:
                break
        return s

    N, K, n = 676696, 129144, 356741
    X = S.Hypergeometric(total=N, successes=K, draws=n)
    for k in (68000, 68103, 68300, 67500):
        first = (
            lg(K + 1)
            - lg(k + 1)
            - lg(K - k + 1)
            + lg(N - K + 1)
            - lg(n - k + 1)
            - lg(N - K - n + k + 1)
            - (lg(N + 1) - lg(n + 1) - lg(N - n + 1))
        )
        t = total(first, lambda j: arb(j) * (N - K - n + j) / ((K - j + 1) * (n - j + 1)), k, 0)
        assert arb_error(X.cdf(float(k)), t) < 5e-15, k
        assert arb_error(X.ccdf(float(k)), 1 - t) < 5e-15, k
    nn, a, b = 260831, 145.90266206, 4.94719612
    Y = S.BetaBinomial(n=nn, a=a, b=b)
    am, bm = arb(a), arb(b)
    for k in (249929, 252000, 240000):
        first = (
            lg(nn + 1)
            - lg(k + 1)
            - lg(nn - k + 1)
            + (k + am).lgamma()
            + (nn - k + bm).lgamma()
            - (nn + am + bm).lgamma()
            - (am.lgamma() + bm.lgamma() - (am + bm).lgamma())
        )
        t = total(first, lambda j: arb(j) * (nn - j + bm) / ((nn - j + 1) * (j - 1 + am)), k, 0)
        assert arb_error(Y.cdf(float(k)), t) < 5e-15, k


# Arb (python-flint, 160 bits): the terms from k outwards by their exact ratios, times the pmf
# from Arb's log-gammas. (total, successes, draws, k, P(X <= k), P(X >= k))
HYPERGEOMETRIC_LONG_TAILS = [
    (4 * 10**11 + 10, 2 * 10**11 + 7, 2 * 10**11 + 3, 10**11, 0.4999949537349561836, 0.5000075693975654849),
    (4 * 10**13 + 10, 2 * 10**13 + 7, 2 * 10**13 + 3, 10**13, 0.4999994953734955962, 0.5000007569397566055),
    (10**13, 4 * 10**12, 5 * 10**12, 2000002300000, 0.9985075459215661553, 0.001492460349185472659),
    (4 * 10**14, 2 * 10**14, 2 * 10**14, 99999850000000, 4.906728662803633161e-198, 1.0),
    (44 * 10**14 + 10, 22 * 10**14 + 7, 22 * 10**14 + 3, 11 * 10**14, 0.4999999518857506509, 0.5000000721713740237),
]


@pytest.mark.parametrize("N, K, n, k, lower, upper", HYPERGEOMETRIC_LONG_TAILS)
def test_hypergeometric_tails_of_more_than_2_20_terms(N, K, n, k, lower, upper):
    """A standard deviation of 1.6e5 to 1.7e7: the tails are not added term by term (the integral
    of the pmf with its Euler-Maclaurin correction, csrc/stats/discrete.hpp smooth_tail_sum)."""
    import time

    X = S.Hypergeometric(total=N, successes=K, draws=n)
    start = time.perf_counter()
    got = (X.cdf(k), X.ccdf(k - 1), X.logcdf(k), X.logccdf(k - 1), X.ccdf(k), X.cdf(k - 1))
    assert time.perf_counter() - start < 0.1
    assert got[0] == pytest.approx(lower, rel=5e-15, abs=0.0)
    assert got[1] == pytest.approx(upper, rel=5e-15, abs=0.0)
    assert got[2] == pytest.approx(math.log(lower), rel=5e-15, abs=5e-15)
    assert got[3] == pytest.approx(math.log(upper), rel=5e-15, abs=5e-15)
    pmf = X.pmf(k)
    assert got[4] == pytest.approx(1.0 - lower if lower < 0.9 else upper - pmf, rel=1e-12)
    assert got[5] == pytest.approx(1.0 - upper if upper < 0.9 else lower - pmf, rel=1e-12)
    # the quantiles search the same tails: a probability between those of k - 1 and k
    if lower < 0.5:
        assert X.icdf(0.5 * (got[5] + got[0])) == k
    else:
        assert X.iccdf(0.5 * (got[1] + got[4])) == k
    x = sd.asarray([float(k), float(k - 1)])
    assert X.cdf(x).tolist() == [got[0], got[5]]


@pytest.mark.parametrize("dist", refdata.TAIL_DISTRIBUTIONS)
def test_long_tail_sums_against_arb(dist):
    """The cdfs where a side of the pmf has 1e4 to 1e9 terms (tests/stats/ref/tails, tools/
    stats_ref.py): bells of a standard deviation up to 5e5, and the beta-binomial's power-law
    ends, U and J shapes and the uniform up to n = 1e8 (to 2^34 near the ends of the support). A
    long smooth run of terms is an integral with its Euler-Maclaurin correction (csrc/stats/
    discrete.hpp smooth_tail_sum), the ends of the support and short runs are added term by term:
    within 5e-15 (of max(1, |log|) for the logs; 45 subnormal steps below the normal range), and a
    call takes well under a millisecond."""
    import time

    cls = getattr(S, refdata.CLASSES[dist])
    names = inspect.signature(cls).parameters
    worst, points = 0.0, 0
    start = time.perf_counter()
    for params, k, ref in refdata.tail_groups(dist):
        X = cls(**dict(zip(names, params)))
        for m, (hi, lo) in ref.items():
            got = np.asarray(getattr(X, m)(sd.asarray(k)))
            diff = np.abs((got - hi) - lo)
            if m.startswith("log"):
                with np.errstate(invalid="ignore"):
                    err = np.where(np.isinf(hi), np.where(got == hi, 0.0, np.inf), diff / np.maximum(1.0, np.abs(hi)))
            else:
                tiny = np.abs(hi) < 2.0**-1022
                with np.errstate(invalid="ignore", divide="ignore"):
                    err = np.where(tiny, diff / refdata.SUBNORMAL_ABS * 5e-15, diff / np.abs(hi))
                err = np.where(diff == 0.0, 0.0, err)
            assert not np.isnan(got).any(), (dist, params, m, k[np.isnan(got)])
            bad = err > 5e-15
            assert not bad.any(), (dist, params, m, k[bad], got[bad], hi[bad], err[bad])
            worst, points = max(worst, float(err.max())), points + k.size
            one = float(k[k.size // 2])  # a scalar call is the array's element
            assert getattr(X, m)(one) == got[k.size // 2]
    assert points >= 400 and time.perf_counter() - start < 20.0


# Arb, as above. Sums of 1e5 to 1e6 terms that fall slowly: the terms left when the sum stops
# (below 2^-60 of it each, 1 / (1 - ratio) of them) were 3e-14 of these values.
HYPERGEOMETRIC_SLOW_TAILS = [
    (932424782597, 258537903787, 243342061157, 67471677254, False, 3.096733521509940357e-7),
    (74155550401079, 2838515285389, 1259652720278, 48217817259, True, 9.941752266199140963e-7),
    (1656105621075497, 3283630072363, 1628503789544210, 3228904018848, True, 3.023157958842238832e-8),
]


@pytest.mark.parametrize("N, K, n, k, upper, value", HYPERGEOMETRIC_SLOW_TAILS)
def test_hypergeometric_slowly_falling_tail_sums(N, K, n, k, upper, value):
    X = S.Hypergeometric(total=N, successes=K, draws=n)
    got = X.ccdf(k - 1) if upper else X.cdf(k)
    assert got == pytest.approx(value, rel=5e-15, abs=0.0)


def test_calls_far_outside_the_guaranteed_region_return_at_once():
    """Every sum and search is bounded (csrc/stats/discrete.hpp): a call whose sum would pass
    its bound returns NaN (docs/api/stats.md §2.3) instead of running on."""
    import time

    B = S.BetaBinomial(n=1e12, a=2.5, b=1.7)
    H = S.Hypergeometric(total=1e15, successes=1e14, draws=1e14)
    W = S.BetaBinomial(n=3e6, a=2.5, b=1.7)  # tail sums of more than 2^20 terms in the middle
    calls = [
        lambda: S.Binomial(n=1e15, p=0.3).entropy(),
        lambda: S.NegativeBinomial(n=1e100, p=0.3).entropy(),
        lambda: S.NegativeBinomial(n=2.5, p=1e-9).entropy(),
        lambda: B.pmf(1e10),
        lambda: B.cdf(1e10),
        lambda: B.ccdf(1e10),
        lambda: B.logcdf(1e10),
        lambda: B.logccdf(1e10),
        lambda: B.icdf(0.01),
        lambda: B.icdf(0.5),
        lambda: B.iccdf(0.99),
        lambda: B.ilogcdf(-1.0),
        lambda: B.median(),
        lambda: B.entropy(),
        lambda: H.entropy(),
        lambda: H.kurtosis(),
        lambda: S.Hypergeometric(total=1e100, successes=1e99, draws=1e99).pmf(1e98),
        lambda: S.Binomial(n=1e100, p=0.3).cdf(3e99),
        lambda: S.Binomial(n=1e100, p=0.3).icdf(0.5),
    ]
    start = time.perf_counter()
    values = [f() for f in calls]
    assert time.perf_counter() - start < 10.0  # ~0.5 s in all
    assert all(math.isnan(v) for v in values), values
    # the far tails of the same distributions end in a few terms and are values
    assert 0.0 < W.cdf(10.0) < 1e-9 and W.icdf(1e-12) < 1e3 and 14.0 < W.entropy() < 15.0  # 3e6 terms: inside the bound
    assert H.logcdf(1e10) < -1e12 and H.cdf(1e10) == 0.0 and H.ccdf(1e10) == 1.0
    # a hypergeometric tail of more than 2^20 terms is a value too (the smooth sum), at once
    start = time.perf_counter()
    assert H.cdf(1e13) == pytest.approx(0.5, abs=1e-7) and H.icdf(0.5) == 1e13 and H.iccdf(0.5) == 1e13
    # and so is a beta-binomial's (Arb: P(X <= 1503807) = 0.33413417803759825271)
    assert W.cdf(1503807.0) == pytest.approx(0.33413417803759825271, rel=5e-15, abs=0.0)
    m = W.icdf(0.5)
    assert W.cdf(m - 1.0) < 0.5 <= W.cdf(m) and W.median() == m
    assert time.perf_counter() - start < 1.0
    # outside the support nothing is summed
    assert (
        B.cdf(-1.0) == 0.0 and B.cdf(1e12) == 1.0 and B.pmf(2e12) == 0.0 and B.icdf(0.0) == 0.0 and B.icdf(1.0) == 1e12
    )


def test_logseries_entropy_near_p_one_against_arb(arb):
    """Below -log p = 2^-10 the entropy is the polylogarithm's expansion at p = 1, not the sum."""
    flint = pytest.importorskip("flint")
    for p in (1 - 2.0**-9, 1 - 2.0**-11, 1 - 1e-6, 1 - 1e-9, 1 - 1e-12, 1 - 2.0**-53):
        pa = arb(p)
        L = -(-pa).log1p()
        # sum p^k log(k) / k = -d/ds Li_s(p) at s = 1 (Arb's polylogarithm as a power series in s)
        t = -flint.acb_series.polylog(flint.acb_series([1, 1], prec=2), flint.acb(pa)).coeffs()[1].real
        h = L.log() - pa.log() * pa / ((1 - pa) * L) + t / L
        assert arb_error(S.LogSeries(p=p).entropy(), h) < 2e-15, p


def test_a_k_beyond_the_exponents_range_against_arb(arb):
    """Where k log q passes the float64 range the log pmf is -inf (not inf - inf), and the log
    of an upper tail too far for the incomplete functions is that of its first term."""
    flint = pytest.importorskip("flint")
    for X in (S.Geometric(p=1 - 1e-9), S.LogSeries(p=1e-8), S.Poisson(mu=1e-3), S.NegativeBinomial(n=0.1, p=0.999999)):
        assert X.pmf(1e308) == 0.0 and X.logpmf(1e308) == -math.inf and X.logccdf(1e308) == -math.inf
        assert X.cdf(1e308) == 1.0 and X.ccdf(1e308) == 0.0 and X.logcdf(1e308) == 0.0
    x = sd.asarray([1.0, 1e308, 3.0, 1e300] * 8)
    for m in ("pmf", "logpmf", "cdf", "logcdf"):  # the chunk kernels give the scalar definitions' values
        G = S.Geometric(p=1 - 1e-9)
        assert np.array_equal(np.asarray(getattr(G, m)(x))[:4], [getattr(G, m)(v) for v in (1.0, 1e308, 3.0, 1e300)]), m
    # a tiny p: k log q is in range at k = 1e305 and the value is not the limit
    assert S.Geometric(p=1e-300).ccdf(1e301) == pytest.approx(math.exp(-10.0), rel=1e-14)
    flint.ctx.prec = 2400  # log Gamma(k + n) - log Gamma(k + 2) at k = 1e300 cancels 1000 bits
    k, n, p = arb(10) ** 300, arb(2.5), arb(0.3)
    # log P(X > k) = log pmf(k + 1) - log p to n / (p k)
    nb = (k + 1 + n).lgamma() - n.lgamma() - (k + 2).lgamma() + n * p.log() + (k + 1) * (-p).log1p() - p.log()
    assert arb_error(S.NegativeBinomial(n=2.5, p=0.3).logccdf(1e300), nb) < 1e-15
    assert arb_error(S.NegativeBinomial(n=2.5, p=0.3).logpmf(1e300), nb) < 1e-15
    mu = arb(4.5)
    po = (k + 1) * mu.log() - mu - (k + 2).lgamma()
    assert arb_error(S.Poisson(mu=4.5).logccdf(1e300), po) < 1e-15


def test_modes_next_to_an_integer_product():
    # floor((n + 1) p) of the exact product: 10 * 0.3 is 2.9999999999999998, which rounds to 3
    assert S.Binomial(n=9, p=0.3).mode() == 2.0
    assert S.Binomial(n=1e12, p=0.99999999).mode() == 999999990000.0
    # floor((n - 1) q / p): 99 * 0.9999 / 1e-4 is just below 989901
    assert S.NegativeBinomial(n=100, p=1e-4).mode() == 989900.0
    assert S.NegativeBinomial(n=1e4, p=1e-4).mode() == 99980000.0


def test_moments_at_extreme_parameters():
    # standardized moments whose denominators leave the normal range, factor by factor
    assert S.LogSeries(p=1e-300).skewness() == pytest.approx(math.sqrt(2e300), rel=1e-14)
    assert S.LogSeries(p=1e-300).kurtosis() == pytest.approx(2e300, rel=1e-14)
    assert S.Zipf(a=1000.0).skewness() == pytest.approx(2.0**500, rel=1e-13)
    assert S.NegativeBinomial(n=1e-300, p=1e-300).variance() == pytest.approx(1e300, rel=1e-15)
    # the mean of a range that cancels, and a width beyond float64 (no pmf: outside the domain)
    assert S.DiscreteUniform(low=-1e300, high=1e300).mean() == -0.5
    assert math.isnan(S.DiscreteUniform(low=-1.7e308, high=1.7e308).pmf(0.0))
    assert math.isnan(S.DiscreteUniform(low=-math.inf, high=3).pmf(0.0)) and math.isnan(
        S.DiscreteUniform(low=0, high=math.inf).cdf(1.0)
    )
    # counts past 2^53 (binomial, hypergeometric) or a beta-binomial n past 2^36: no pmf between the ends
    assert math.isnan(S.Binomial(n=1e100, p=0.3).logpmf(3e99)) and S.Binomial(n=1e100, p=0.3).pmf(2e100) == 0.0
    assert S.Binomial(n=1e100, p=0.3).logpmf(0.0) == pytest.approx(1e100 * math.log1p(-0.3), rel=1e-15)
    assert math.isnan(S.BetaBinomial(n=2.0**35, a=2.5, b=1.7).pmf(5.0))  # n + a + b past 2^35
    assert S.BetaBinomial(n=2.0**35 - 8, a=2.5, b=1.7).pmf(5.0) > 0.0
    assert math.isnan(S.BetaBinomial(n=1, a=0.1, b=1e100).pmf(1.0)) and math.isnan(
        S.BetaBinomial(n=5, a=1e-300, b=1.0).mean()
    )
    # the larger end of a pmf that is not unimodal: n for a > b, a tie for a = b
    assert math.isnan(S.BetaBinomial(n=1, a=0.001, b=0.001).mode()) and S.BetaBinomial(n=7, a=0.5, b=0.25).mode() == 7.0
    assert S.BetaBinomial(n=7, a=0.25, b=0.5).mode() == 0.0 and S.BetaBinomial(n=7, a=1.5, b=0.5).mode() == 7.0
    assert math.isnan(S.Hypergeometric(total=2.0**24 + 2, successes=5e6, draws=7e6).kurtosis())


A = sd.asarray
inf = math.inf
sp = sd.special


def test_binomial_mode_at_2_52():
    assert S.Binomial(n=2.0**52, p=0.5).mode() == 2.0**51
    assert math.isnan(S.Binomial(n=2.0**52 + 2, p=0.5).mode())


# ---- the family's support; NegativeBinomial(n=0)


def test_support_is_the_familys():
    cases = [
        (S.Binomial(n=10, p=0.0), (0.0, 10.0)),
        (S.Binomial(n=10, p=1.0), (0.0, 10.0)),
        (S.Bernoulli(p=0.0), (0.0, 1.0)),
        (S.Poisson(mu=0.0), (0.0, inf)),
        (S.Geometric(p=1.0), (1.0, inf)),
        (S.NegativeBinomial(n=3.0, p=1.0), (0.0, inf)),
        (S.NegativeBinomial(n=0.0, p=0.4), (0.0, inf)),
    ]
    for X, sup in cases:
        assert X.support() == sup
        assert X.icdf(1.0) == sup[1] and X.iccdf(0.0) == sup[1]
        assert X.icdf(0.0) == sup[0] and X.iccdf(1.0) == sup[0]
    # inside (0, 1) the quantile is the point mass
    assert S.Poisson(mu=0.0).icdf(0.5) == 0.0 and S.Geometric(p=1.0).iccdf(0.3) == 1.0
    # as scipy's support and ppf(1)
    assert S.Poisson(mu=0.0).icdf(1.0) == inf


def test_negative_binomial_n0_is_the_point_mass():
    X = S.NegativeBinomial(n=0.0, p=0.5)
    assert (X.pmf(0.0), X.pmf(1.0), X.logpmf(0.0), X.cdf(2.0), X.ccdf(2.0)) == (1.0, 0.0, 0.0, 1.0, 0.0)
    assert X.cdf(2.0) == sp.nbdtr(2, 0.0, 0.5)
    assert (X.mean(), X.variance(), X.median(), X.mode(), X.entropy()) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert math.isnan(X.skewness()) and math.isnan(X.kurtosis())
    assert list(np.asarray(X.sample(5, rng=sd.random.default_rng(1)))) == [0.0] * 5
    assert float(S.NegativeBinomial(n=A([0.0, 2.0]), p=0.5).pmf(A([0.0, 0.0]))[0]) == 1.0
    # p = 0 and an infinite mean are not distributions: NaN (special's values are the limits)
    assert math.isnan(S.NegativeBinomial(n=3.0, p=0.0).cdf(2.0)) and sp.nbdtr(2, 3, 0.0) == 0.0
    assert math.isnan(S.Poisson(mu=inf).cdf(2.0)) and sp.pdtr(2, inf) == 0.0


def test_point_masses_have_zero_entropy():
    for a, b in ((1.0, 1.0), (0.5, 0.5), (2.0, 3.0)):
        assert S.BetaBinomial(n=0.0, a=a, b=b).entropy() == 0.0
    assert S.Dirichlet(alpha=A([2.0])).entropy() == 0.0
    assert S.Dirichlet(alpha=A([[2.0], [0.5]])).entropy().tolist() == [0.0, 0.0]
    assert math.isnan(S.Dirichlet(alpha=A([-1.0])).entropy())


# ---- D-13: discrete quantiles round-trip --------------------------------------------------


def test_discrete_icdf_of_cdf():
    X = S.Poisson(mu=2.2919601461531838)
    assert X.icdf(X.cdf(2.0)) == 2.0
    rng = np.random.default_rng(7)
    for _ in range(20):
        X = S.Binomial(n=float(rng.integers(1, 100)), p=float(rng.random()))
        k = sd.arange(0.0, 30.0)
        c = np.asarray(X.cdf(k))
        back = np.asarray(X.icdf(sd.asarray(c)))
        for i in range(30):
            if 0.0 < c[i] <= 0.9:  # where the small side is not near the cdf's rounding
                first = i
                while first > 0 and c[first - 1] == c[i]:
                    first -= 1
                assert back[i] == first, (X, i, c[i], back[i])
    # near 1 the contract on the small side 1 - p keeps the exact quantile
    P = S.Poisson(mu=0.0025)
    assert P.icdf(0.9999999999999992) == 5.0
