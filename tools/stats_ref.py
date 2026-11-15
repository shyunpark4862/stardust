"""Reference values for sd.stats' distributions from mpmath (docs/design/stats.md §9).

    .venv/bin/python tools/stats_ref.py                     # every distribution, on all cores
    .venv/bin/python tools/stats_ref.py normal gamma        # only these
    .venv/bin/python tools/stats_ref.py --jobs=8 normal     # 8 processes instead of all cores
    .venv/bin/python tools/stats_ref.py normal:cdf,icdf     # only these methods
    .venv/bin/python tools/stats_ref.py --masks-only gamma  # rewrite the guaranteed masks only
    .venv/bin/python tools/stats_ref.py --intervals zipf    # add the discrete inverses' [kmin, kmax]

writes tests/stats/ref/<distribution>_<method>.npz in tests/special/refdata.py's format:
x0..x{n-1} are the native function's inputs (params..., then the method's input, if any),
hi + lo the reference as a double-double, `guaranteed` the guaranteed region, `scale` the
bound's multiplier (max(1, kappa) for the inverses, whose contract is
|dx / x| <= 1e-14 max(1, kappa); the T20 mixed bound for logpdf and entropy is applied by
the test, not stored); for the discrete inverses `kmin`, `kmax` the answers the backward
error contract allows (BACKWARD). The generator is seeded: running the tool again gives the same file.

The mpmath definitions are written from the textbook formulas, never from our code; the
inverses solve log F(x) = log p by Newton from our own value (a start only: the reference
is the root at 30 digits, whatever the start).
"""

import multiprocessing
import signal
import sys
from pathlib import Path

import mpmath
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import special_ref as sr  # noqa: E402  (shared helpers: input sets, double-double, evaluation)

OUT = ROOT / "tests" / "stats" / "ref"
POINT_SECONDS = 30

grid, loguniform, around = sr.grid, sr.loguniform, sr.around
mpf = mpmath.mpf

INPUT_METHODS = ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf")
PARAM_METHODS = ("mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy")
INVERSE = {"icdf", "iccdf", "ilogcdf", "ilogccdf"}


# ---------------------------------------------------------------- the distributions
#
# Each is a class of static mpmath functions of mpf arguments (x first, then the
# parameters in native order), with `params` and the input sets:
#   param_sets(rng)        -> (tuple of parameter arrays, guaranteed mask)    for the moments
#   x_sets(rng)            -> (params..., x), guaranteed                        for pdf..logccdf
#   p_sets(rng)            -> (params..., p), guaranteed                        for icdf, iccdf
#   logp_sets(rng)         -> (params..., logp), guaranteed                     for ilogcdf, ilogccdf
# cdf and ccdf must both be accurate (the logs take the smaller one; never 1 - F).


def _p_values(rng, n):
    """Probabilities for icdf and iccdf: uniform, both tails down to 1e-300 and the
    subnormals, and around 1/2."""
    k = n // 8
    p = np.concatenate([rng.uniform(0, 1, 2 * k), loguniform(rng, 1e-300, 0.5, 2 * k),
                        1 - loguniform(rng, 2**-53, 0.5, 2 * k), loguniform(rng, 1e-320, 1e-300, k),
                        0.5 + rng.uniform(-1e-6, 1e-6, n - 7 * k)])
    return p


def _logp_values(rng, n):
    return -np.concatenate([loguniform(rng, 1e-300, 1, n // 2), loguniform(rng, 1, 1e10, n - n // 2)])


class Distribution:
    """Input sets from _params(rng, n) -> parameter arrays and _x(rng, params) -> x."""

    positive = False  # support in (0, inf): the inverses' Newton runs in log x

    @classmethod
    def param_sets(cls, rng):
        ps = cls._params(rng, 2500)
        return ps, np.ones(ps[0].size, dtype=bool)

    @classmethod
    def x_sets(cls, rng):
        ps = cls._params(rng, 6000)
        x = cls._x(rng, ps)
        return (*ps, x), np.isfinite(x)

    @classmethod
    def p_sets(cls, rng):
        ps = cls._params(rng, 4000)
        p = _p_values(rng, ps[0].size)
        return (*ps, p), (p > 0) & (p < 1)

    @classmethod
    def logp_sets(cls, rng):
        ps = cls._params(rng, 4000)
        lp = _logp_values(rng, ps[0].size)
        return (*ps, lp), lp < 0


def _location_scale(rng, n):
    """mu and sigma for a location-scale family: 0 and 1 often, then wide."""
    q = n // 4
    mu = np.concatenate([np.zeros(q), rng.uniform(-10, 10, q), loguniform(rng, 1e-3, 1e6, n - 2 * q, signed=True)])
    sigma = np.concatenate([np.ones(q), loguniform(rng, 1e-3, 1e3, n - q)])
    rng.shuffle(mu)
    return mu, sigma


def _z(rng, n, far):
    """Standardized points: the body, then out to +-far."""
    t = n // 3
    return np.concatenate([rng.uniform(-8, 8, t), loguniform(rng, 1e-8, 40, t, signed=True),
                           loguniform(rng, 40, far, n - 2 * t, signed=True)])


def _normal_quantile(p):
    """z with P(Z <= z) = p (mpf p in (0, 1)), Newton on the log of the smaller side from
    sd.special.ndtri_exp's value (a start only)."""
    import stardust as sd

    upper = p > mpf(1) / 2
    q = 1 - p if upper else p  # the small side
    lq = mpmath.log(q)
    z = -float(sd.special.ndtri_exp(float(lq)))  # the upper-side z of q, as a start
    z = mpf(z) if upper else -mpf(z)  # P(Z <= z) = p
    for _ in range(200):
        side = mpmath.ncdf(-z) if upper else mpmath.ncdf(z)
        slope = (-1 if upper else 1) * mpmath.npdf(z) / side
        dz = (mpmath.log(side) - lq) / slope
        z -= dz
        if abs(dz) <= (abs(z) + mpf(10) ** -300) * mpf(10) ** -40:
            return z
    raise RuntimeError("no convergence")


class Normal(Distribution):
    params = ("mu", "sigma")

    @staticmethod
    def pdf(x, mu, s):
        return mpmath.npdf(x, mu, s)

    @staticmethod
    def cdf(x, mu, s):
        return mpmath.ncdf(x, mu, s)

    @staticmethod
    def ccdf(x, mu, s):
        return mpmath.ncdf(-x, -mu, s)

    mean = staticmethod(lambda mu, s: mu)
    variance = staticmethod(lambda mu, s: s * s)
    standard_deviation = staticmethod(lambda mu, s: s)
    skewness = staticmethod(lambda mu, s: mpf(0))
    kurtosis = staticmethod(lambda mu, s: mpf(0))
    median = staticmethod(lambda mu, s: mu)
    mode = staticmethod(lambda mu, s: mu)
    entropy = staticmethod(lambda mu, s: (1 + mpmath.log(2 * mpmath.pi)) / 2 + mpmath.log(s))
    icdf_mp = staticmethod(lambda p, mu, s: mu + s * _normal_quantile(p))
    iccdf_mp = staticmethod(lambda q, mu, s: mu - s * _normal_quantile(q))

    @staticmethod
    def _params(rng, n):
        return _location_scale(rng, n)

    @staticmethod
    def _x(rng, ps):
        mu, sigma = ps
        return mu + sigma * _z(rng, mu.size, 1e4)

    @classmethod
    def param_sets(cls, rng):
        mu, sigma = _location_scale(rng, 2000)
        sigma = np.concatenate([sigma, loguniform(rng, 1e-300, 1e300, 500), around([1 / np.sqrt(2 * np.pi * np.e)], 8)])
        mu = np.concatenate([mu, loguniform(rng, 1e-300, 1e300, 500, signed=True), np.zeros(17)])
        return (mu, sigma), np.ones(mu.size, dtype=bool)


class Exponential(Distribution):
    params = ("scale",)
    positive = True

    pdf = staticmethod(lambda x, s: mpmath.exp(-x / s) / s if x >= 0 else mpf(0))
    cdf = staticmethod(lambda x, s: -mpmath.expm1(-x / s) if x > 0 else mpf(0))
    ccdf = staticmethod(lambda x, s: mpmath.exp(-x / s) if x > 0 else mpf(1))
    mean = staticmethod(lambda s: s)
    variance = staticmethod(lambda s: s * s)
    standard_deviation = staticmethod(lambda s: s)
    skewness = staticmethod(lambda s: mpf(2))
    kurtosis = staticmethod(lambda s: mpf(6))
    median = staticmethod(lambda s: s * mpmath.log(2))
    mode = staticmethod(lambda s: mpf(0))
    entropy = staticmethod(lambda s: 1 + mpmath.log(s))
    icdf_mp = staticmethod(lambda p, s: -s * mpmath.log1p(-p))
    iccdf_mp = staticmethod(lambda q, s: -s * mpmath.log(q))

    @staticmethod
    def _params(rng, n):
        s = np.concatenate([np.ones(n // 4), loguniform(rng, 1e-3, 1e3, n - n // 4 - 20), around([np.exp(-1.0)], 9)[:20]])
        return (s,)

    @staticmethod
    def _x(rng, ps):
        (s,) = ps
        n = s.size
        y = np.concatenate([loguniform(rng, 1e-300, 1e3, n // 2), rng.uniform(0, 50, n - n // 2)])
        return s * y


class Uniform(Distribution):
    params = ("a", "b")

    pdf = staticmethod(lambda x, a, b: 1 / (b - a) if a <= x <= b else mpf(0))
    cdf = staticmethod(lambda x, a, b: mpf(0) if x <= a else (mpf(1) if x >= b else (x - a) / (b - a)))
    ccdf = staticmethod(lambda x, a, b: mpf(1) if x <= a else (mpf(0) if x >= b else (b - x) / (b - a)))
    mean = staticmethod(lambda a, b: (a + b) / 2)
    variance = staticmethod(lambda a, b: (b - a) ** 2 / 12)
    standard_deviation = staticmethod(lambda a, b: (b - a) / mpmath.sqrt(12))
    skewness = staticmethod(lambda a, b: mpf(0))
    kurtosis = staticmethod(lambda a, b: mpf(-6) / 5)
    median = staticmethod(lambda a, b: (a + b) / 2)
    mode = staticmethod(lambda a, b: mpmath.nan)
    entropy = staticmethod(lambda a, b: mpmath.log(b - a))
    icdf_mp = staticmethod(lambda p, a, b: a + p * (b - a))
    iccdf_mp = staticmethod(lambda q, a, b: b - q * (b - a))

    @staticmethod
    def _params(rng, n):
        q = n // 4
        a = np.concatenate([np.zeros(q), rng.uniform(-100, 100, q), loguniform(rng, 1e-3, 1e6, n - 2 * q, signed=True)])
        w = np.concatenate([np.ones(q), loguniform(rng, 1e-3, 1e6, n - q)])
        rng.shuffle(a)
        return a, a + w

    @staticmethod
    def _x(rng, ps):
        a, b = ps
        u = np.concatenate([rng.uniform(-0.1, 1.1, a.size // 2), loguniform(rng, 1e-12, 1, a.size - a.size // 2)])
        return a + (b - a) * u


class Laplace(Distribution):
    params = ("mu", "sigma")

    @staticmethod
    def pdf(x, mu, s):
        return mpmath.exp(-abs(x - mu) / s) / (2 * s)

    @staticmethod
    def cdf(x, mu, s):
        z = (x - mu) / s
        return mpmath.exp(z) / 2 if z < 0 else 1 - mpmath.exp(-z) / 2

    @staticmethod
    def ccdf(x, mu, s):
        z = (x - mu) / s
        return mpmath.exp(-z) / 2 if z > 0 else 1 - mpmath.exp(z) / 2

    mean = staticmethod(lambda mu, s: mu)
    variance = staticmethod(lambda mu, s: 2 * s * s)
    standard_deviation = staticmethod(lambda mu, s: mpmath.sqrt(2) * s)
    skewness = staticmethod(lambda mu, s: mpf(0))
    kurtosis = staticmethod(lambda mu, s: mpf(3))
    median = staticmethod(lambda mu, s: mu)
    mode = staticmethod(lambda mu, s: mu)
    entropy = staticmethod(lambda mu, s: 1 + mpmath.log(2 * s))
    icdf_mp = staticmethod(lambda p, mu, s: mu + s * mpmath.log(2 * p) if p <= mpf(1) / 2 else mu - s * mpmath.log(2 * (1 - p)))
    iccdf_mp = staticmethod(lambda q, mu, s: mu - s * mpmath.log(2 * q) if q <= mpf(1) / 2 else mu + s * mpmath.log(2 * (1 - q)))

    @staticmethod
    def _params(rng, n):
        return _location_scale(rng, n)

    @staticmethod
    def _x(rng, ps):
        mu, sigma = ps
        return mu + sigma * _z(rng, mu.size, 1e4)


class Cauchy(Distribution):
    params = ("mu", "sigma")

    @staticmethod
    def pdf(x, mu, s):
        z = (x - mu) / s
        return 1 / (mpmath.pi * s * (1 + z * z))

    @staticmethod
    def cdf(x, mu, s):
        return mpmath.atan2(1, -(x - mu) / s) / mpmath.pi

    @staticmethod
    def ccdf(x, mu, s):
        return mpmath.atan2(1, (x - mu) / s) / mpmath.pi

    mean = variance = standard_deviation = skewness = kurtosis = staticmethod(lambda mu, s: mpmath.nan)
    median = staticmethod(lambda mu, s: mu)
    mode = staticmethod(lambda mu, s: mu)
    entropy = staticmethod(lambda mu, s: mpmath.log(4 * mpmath.pi * s))
    icdf_mp = staticmethod(lambda p, mu, s: mu - s / mpmath.tan(mpmath.pi * p))
    iccdf_mp = staticmethod(lambda q, mu, s: mu + s / mpmath.tan(mpmath.pi * q))

    @staticmethod
    def _params(rng, n):
        return _location_scale(rng, n)

    @staticmethod
    def _x(rng, ps):
        mu, sigma = ps
        return mu + sigma * _z(rng, mu.size, 1e300)


class LogNormal(Distribution):
    params = ("mu", "sigma")
    positive = True

    pdf = staticmethod(lambda x, mu, s: mpmath.npdf(mpmath.log(x), mu, s) / x if x > 0 else mpf(0))
    cdf = staticmethod(lambda x, mu, s: mpmath.ncdf(mpmath.log(x), mu, s) if x > 0 else mpf(0))
    ccdf = staticmethod(lambda x, mu, s: mpmath.ncdf(-mpmath.log(x), -mu, s) if x > 0 else mpf(1))
    mean = staticmethod(lambda mu, s: mpmath.exp(mu + s * s / 2))
    variance = staticmethod(lambda mu, s: mpmath.expm1(s * s) * mpmath.exp(2 * mu + s * s))
    standard_deviation = staticmethod(lambda mu, s: mpmath.sqrt(mpmath.expm1(s * s) * mpmath.exp(2 * mu + s * s)))
    skewness = staticmethod(lambda mu, s: (mpmath.exp(s * s) + 2) * mpmath.sqrt(mpmath.expm1(s * s)))
    kurtosis = staticmethod(lambda mu, s: mpmath.exp(4 * s * s) + 2 * mpmath.exp(3 * s * s) + 3 * mpmath.exp(2 * s * s) - 6)
    median = staticmethod(lambda mu, s: mpmath.exp(mu))
    mode = staticmethod(lambda mu, s: mpmath.exp(mu - s * s))
    entropy = staticmethod(lambda mu, s: mu + mpf(1) / 2 + mpmath.log(s * mpmath.sqrt(2 * mpmath.pi)))
    icdf_mp = staticmethod(lambda p, mu, s: mpmath.exp(mu + s * _normal_quantile(p)))
    iccdf_mp = staticmethod(lambda q, mu, s: mpmath.exp(mu - s * _normal_quantile(q)))

    @staticmethod
    def _params(rng, n):
        q = n // 4
        mu = np.concatenate([np.zeros(q), rng.uniform(-10, 10, n - 2 * q), rng.uniform(-300, 300, q)])
        sigma = np.concatenate([np.ones(q), loguniform(rng, 1e-2, 10, n - q)])
        rng.shuffle(mu)
        return mu, sigma

    @staticmethod
    def _x(rng, ps):
        mu, sigma = ps
        e = mu + sigma * _z(rng, mu.size, 1e4)
        e = np.where(np.abs(e) < 700, e, np.sign(e) * rng.uniform(0, 700, e.size))
        return np.exp(e)


def _weibull_gk(c, k):
    return mpmath.gamma(1 + mpf(k) / c)


class Weibull(Distribution):
    params = ("c", "scale")
    positive = True

    @staticmethod
    def pdf(x, c, s):
        if x < 0:
            return mpf(0)
        y = x / s
        return c / s * y ** (c - 1) * mpmath.exp(-(y ** c))

    cdf = staticmethod(lambda x, c, s: -mpmath.expm1(-((x / s) ** c)) if x > 0 else mpf(0))
    ccdf = staticmethod(lambda x, c, s: mpmath.exp(-((x / s) ** c)) if x > 0 else mpf(1))
    mean = staticmethod(lambda c, s: s * _weibull_gk(c, 1))
    variance = staticmethod(lambda c, s: s * s * (_weibull_gk(c, 2) - _weibull_gk(c, 1) ** 2))
    standard_deviation = staticmethod(lambda c, s: s * mpmath.sqrt(_weibull_gk(c, 2) - _weibull_gk(c, 1) ** 2))

    @staticmethod
    def skewness(c, s):
        g1, g2, g3 = (_weibull_gk(c, k) for k in (1, 2, 3))
        return (g3 - 3 * g1 * g2 + 2 * g1**3) / (g2 - g1**2) ** mpf(1.5)

    @staticmethod
    def kurtosis(c, s):
        g1, g2, g3, g4 = (_weibull_gk(c, k) for k in (1, 2, 3, 4))
        return (g4 - 4 * g1 * g3 + 6 * g1**2 * g2 - 3 * g1**4) / (g2 - g1**2) ** 2 - 3

    median = staticmethod(lambda c, s: s * mpmath.log(2) ** (1 / c))
    mode = staticmethod(lambda c, s: s * ((c - 1) / c) ** (1 / c) if c > 1 else mpf(0))
    entropy = staticmethod(lambda c, s: mpmath.euler * (1 - 1 / c) + mpmath.log(s / c) + 1)
    icdf_mp = staticmethod(lambda p, c, s: s * (-mpmath.log1p(-p)) ** (1 / c))
    iccdf_mp = staticmethod(lambda q, c, s: s * (-mpmath.log(q)) ** (1 / c))

    @staticmethod
    def _params(rng, n):
        q = n // 4
        c = np.concatenate([np.ones(q // 2), loguniform(rng, 0.1, 50, n - q), loguniform(rng, 1e-2, 1e3, q - q // 2)])
        s = np.concatenate([np.ones(q), loguniform(rng, 1e-3, 1e3, n - q)])
        rng.shuffle(c)
        return c, s

    @staticmethod
    def _x(rng, ps):
        c, s = ps
        t = loguniform(rng, 1e-300, 700, c.size)
        return s * t ** (1 / c)

    @classmethod
    def param_sets(cls, rng):
        (c, s), g = super().param_sets(rng)
        zeros = np.concatenate([around([3.6023494], 6), around([2.2520], 6), around([5.7734], 6)])  # near the skewness and kurtosis zeros
        return (np.concatenate([c, zeros, loguniform(rng, 1e3, 1e8, 50)]),
                np.concatenate([s, np.ones(zeros.size + 50)])), np.ones(c.size + zeros.size + 50, dtype=bool)


def _quantile_points(rng, n, ppf, isf, extra):
    """x spread over a distribution: ppf/isf (scipy, a sampler only) of p from the body to
    1e-300 in both tails, plus `extra(rng, k)` points beyond."""
    import warnings
    k = n // 5
    p = np.concatenate([rng.uniform(0.01, 0.99, k), loguniform(rng, 1e-300, 0.01, k)])
    q = loguniform(rng, 1e-300, 0.01, k)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        lo = ppf(p, np.arange(p.size))
        hi = isf(q, np.arange(p.size, p.size + q.size))
    x = np.concatenate([lo, hi, extra(rng, n - lo.size - hi.size)])
    bad = ~np.isfinite(x)  # scipy's quantiles fail in places (NaN, inf): replaced by extra points
    x[bad] = extra(rng, int(bad.sum()))
    return x


def _gamma_entropy(a):
    return a + mpmath.loggamma(a) + (1 - a) * mpmath.digamma(a)


class Gamma(Distribution):
    params = ("a", "scale")
    positive = True

    @staticmethod
    def pdf(x, a, s):
        if x <= 0:
            return mpmath.inf if (x == 0 and a < 1) else (1 / s if (x == 0 and a == 1) else mpf(0))
        y = x / s
        return mpmath.exp((a - 1) * mpmath.log(y) - y - mpmath.loggamma(a)) / s

    cdf = staticmethod(lambda x, a, s: mpmath.gammainc(a, 0, x / s, regularized=True) if x > 0 else mpf(0))
    ccdf = staticmethod(lambda x, a, s: mpmath.gammainc(a, x / s, mpmath.inf, regularized=True) if x > 0 else mpf(1))
    mean = staticmethod(lambda a, s: a * s)
    variance = staticmethod(lambda a, s: a * s * s)
    standard_deviation = staticmethod(lambda a, s: mpmath.sqrt(a) * s)
    skewness = staticmethod(lambda a, s: 2 / mpmath.sqrt(a))
    kurtosis = staticmethod(lambda a, s: 6 / a)
    mode = staticmethod(lambda a, s: (a - 1) * s if a > 1 else mpf(0))
    entropy = staticmethod(lambda a, s: _gamma_entropy(a) + mpmath.log(s))

    @staticmethod
    def _params(rng, n):
        q = n // 4
        a = np.concatenate([np.ones(q // 4), loguniform(rng, 1e-2, 1e4, n - q), loguniform(rng, 1e-5, 1e7, q - q // 4)])
        s = np.concatenate([np.ones(q * 2), loguniform(rng, 1e-3, 1e3, n - 2 * q)])
        rng.shuffle(a)
        return a, s

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        a, s = ps
        y = _quantile_points(rng, a.size, lambda p, i: st.gamma.ppf(p, a[i]), lambda q, i: st.gamma.isf(q, a[i]),
                             lambda rng, k: loguniform(rng, 1e-300, 1e5, k))
        return y * s


class ChiSquared(Distribution):
    params = ("df",)
    positive = True

    pdf = staticmethod(lambda x, df: Gamma.pdf(x, df / 2, mpf(2)))
    cdf = staticmethod(lambda x, df: Gamma.cdf(x, df / 2, mpf(2)))
    ccdf = staticmethod(lambda x, df: Gamma.ccdf(x, df / 2, mpf(2)))
    mean = staticmethod(lambda df: df)
    variance = staticmethod(lambda df: 2 * df)
    standard_deviation = staticmethod(lambda df: mpmath.sqrt(2 * df))
    skewness = staticmethod(lambda df: mpmath.sqrt(8 / df))
    kurtosis = staticmethod(lambda df: 12 / df)
    mode = staticmethod(lambda df: df - 2 if df > 2 else mpf(0))
    entropy = staticmethod(lambda df: _gamma_entropy(df / 2) + mpmath.log(2))

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([loguniform(rng, 1e-2, 1e5, n - 50), np.arange(1.0, 51.0)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        (df,) = ps
        return _quantile_points(rng, df.size, lambda p, i: st.chi2.ppf(p, df[i]), lambda q, i: st.chi2.isf(q, df[i]),
                                lambda rng, k: loguniform(rng, 1e-300, 1e5, k))


class Beta(Distribution):
    params = ("a", "b")
    unit = True

    @staticmethod
    def pdf(x, a, b):
        if x <= 0 or x >= 1:
            if x == 0:
                return mpmath.inf if a < 1 else (b if a == 1 else mpf(0))
            if x == 1:
                return mpmath.inf if b < 1 else (a if b == 1 else mpf(0))
            return mpf(0)
        return mpmath.exp((a - 1) * mpmath.log(x) + (b - 1) * mpmath.log1p(-x) - mpmath.log(mpmath.beta(a, b)))

    cdf = staticmethod(lambda x, a, b: mpmath.betainc(a, b, 0, x, regularized=True) if 0 < x < 1 else mpf(0 if x <= 0 else 1))
    # the upper tail as I_{1-x}(b, a): mpmath's betainc(a, b, x, 1) is 1 - I_x and loses a small
    # tail to cancellation. Below x = 1e-30, 1 - x is not representable at the working
    # precision; the upper tail is then ~1 and 1 - I_x is exact enough.
    @staticmethod
    def ccdf(x, a, b):
        if not 0 < x < 1:
            return mpf(1 if x <= 0 else 0)
        if x < mpf(10) ** -30:
            return 1 - mpmath.betainc(a, b, 0, x, regularized=True)
        return mpmath.betainc(b, a, 0, 1 - x, regularized=True)
    mean = staticmethod(lambda a, b: a / (a + b))
    variance = staticmethod(lambda a, b: a * b / ((a + b) ** 2 * (a + b + 1)))
    standard_deviation = staticmethod(lambda a, b: mpmath.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
    skewness = staticmethod(lambda a, b: 2 * (b - a) * mpmath.sqrt(a + b + 1) / ((a + b + 2) * mpmath.sqrt(a * b)))
    kurtosis = staticmethod(lambda a, b: 6 * ((a - b) ** 2 * (a + b + 1) - a * b * (a + b + 2)) / (a * b * (a + b + 2) * (a + b + 3)))

    @staticmethod
    def mode(a, b):
        if a > 1 and b > 1:
            return (a - 1) / (a + b - 2)
        if a <= 1 and b <= 1:
            return mpf(1) if (a == 1 and b < 1) else (mpf(0) if (b == 1 and a < 1) else mpmath.nan)
        return mpf(0) if a <= 1 else mpf(1)

    @staticmethod
    def entropy(a, b):
        c = a + b
        return (mpmath.log(mpmath.beta(a, b)) - (a - 1) * mpmath.digamma(a) - (b - 1) * mpmath.digamma(b)
                + (c - 2) * mpmath.digamma(c))

    @staticmethod
    def _params(rng, n):
        q = n // 4
        a = np.concatenate([np.ones(q // 4), loguniform(rng, 1e-2, 1e4, n - q // 4)])
        b = np.concatenate([loguniform(rng, 1e-2, 1e4, n - q // 4), np.ones(q // 4)])
        rng.shuffle(a)
        return a, b

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        a, b = ps
        return _quantile_points(rng, a.size, lambda p, i: st.beta.ppf(p, a[i], b[i]),
                                lambda q, i: st.beta.isf(q, a[i], b[i]),
                                lambda rng, k: np.concatenate([loguniform(rng, 1e-300, 0.5, k // 2),
                                                               1 - loguniform(rng, 2**-53, 0.5, k - k // 2)]))


class StudentT(Distribution):
    params = ("df",)

    @staticmethod
    def pdf(t, df):
        return mpmath.exp(mpmath.loggamma((df + 1) / 2) - mpmath.loggamma(df / 2) - mpmath.log(df * mpmath.pi) / 2
                          - (df + 1) / 2 * mpmath.log1p(t * t / df))

    @staticmethod
    def _lower(t, df):  # P(T <= t) for t <= 0
        x = df / (df + t * t)
        return mpmath.betainc(df / 2, mpf(1) / 2, 0, x, regularized=True) / 2

    cdf = staticmethod(lambda t, df: StudentT._lower(t, df) if t <= 0 else 1 - StudentT._lower(-t, df))
    ccdf = staticmethod(lambda t, df: StudentT._lower(-t, df) if t >= 0 else 1 - StudentT._lower(t, df))
    mean = staticmethod(lambda df: mpf(0) if df > 1 else mpmath.nan)
    variance = staticmethod(lambda df: df / (df - 2) if df > 2 else (mpmath.inf if df > 1 else mpmath.nan))
    standard_deviation = staticmethod(lambda df: mpmath.sqrt(df / (df - 2)) if df > 2 else (mpmath.inf if df > 1 else mpmath.nan))
    skewness = staticmethod(lambda df: mpf(0) if df > 3 else mpmath.nan)
    kurtosis = staticmethod(lambda df: 6 / (df - 4) if df > 4 else (mpmath.inf if df > 2 else mpmath.nan))
    median = staticmethod(lambda df: mpf(0))
    mode = staticmethod(lambda df: mpf(0))

    @staticmethod
    def entropy(df):
        return ((df + 1) / 2 * (mpmath.digamma((df + 1) / 2) - mpmath.digamma(df / 2))
                + mpmath.log(mpmath.sqrt(df) * mpmath.beta(df / 2, mpf(1) / 2)))

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([loguniform(rng, 1e-1, 1e6, n - 60), loguniform(rng, 1e-2, 1e-1, 20),
                                np.array([1.0, 2.0, 3.0, 4.0] * 10)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        (df,) = ps
        return _quantile_points(rng, df.size, lambda p, i: st.t.ppf(p, df[i]), lambda q, i: st.t.isf(q, df[i]),
                                lambda rng, k: loguniform(rng, 1e-300, 1e300, k, signed=True))

    @classmethod
    def param_sets(cls, rng):
        (df,), g = super().param_sets(rng)
        edges = np.concatenate([around([1.0, 2.0, 3.0, 4.0], 3), loguniform(rng, 1e6, 1e15, 30)])
        return (np.concatenate([df, edges]),), np.ones(df.size + edges.size, dtype=bool)


class F(Distribution):
    params = ("dfn", "dfd")
    positive = True

    @staticmethod
    def pdf(x, m, n):
        if x <= 0:
            return (mpmath.inf if m < 2 else (mpf(1) if m == 2 else mpf(0))) if x == 0 else mpf(0)
        a, b = m / 2, n / 2
        return mpmath.exp(a * mpmath.log(m * x) + b * mpmath.log(n) - (a + b) * mpmath.log(m * x + n) - mpmath.log(x)
                          - mpmath.log(mpmath.beta(a, b)))

    # Each side directly from its own argument (w = m x / (m x + n) for the cdf, v = 1 - w
    # for the ccdf), except where that argument is within 1e-30 of 1 and so not
    # representable at the working precision (v = 1 - 3e-125): then that side is ~1 and
    # 1 minus the other is exact enough.
    @staticmethod
    def cdf(x, m, n):
        if x <= 0:
            return mpf(0)
        v = n / (m * x + n)
        if v >= mpf(10) ** -30:
            return mpmath.betainc(m / 2, n / 2, 0, m * x / (m * x + n), regularized=True)
        return 1 - mpmath.betainc(n / 2, m / 2, 0, v, regularized=True)

    @staticmethod
    def ccdf(x, m, n):
        if x <= 0:
            return mpf(1)
        w = m * x / (m * x + n)
        if w >= mpf(10) ** -30:
            return mpmath.betainc(n / 2, m / 2, 0, n / (m * x + n), regularized=True)
        return 1 - mpmath.betainc(m / 2, n / 2, 0, w, regularized=True)

    mean = staticmethod(lambda m, n: n / (n - 2) if n > 2 else mpmath.inf)
    variance = staticmethod(lambda m, n: 2 * n * n * (m + n - 2) / (m * (n - 2) ** 2 * (n - 4)) if n > 4 else
                            (mpmath.inf if n > 2 else mpmath.nan))
    standard_deviation = staticmethod(lambda m, n: mpmath.sqrt(F.variance(m, n)) if n > 2 else mpmath.nan)
    skewness = staticmethod(lambda m, n: (2 * m + n - 2) * mpmath.sqrt(8 * (n - 4)) / ((n - 6) * mpmath.sqrt(m * (m + n - 2)))
                            if n > 6 else (mpmath.inf if n > 4 else mpmath.nan))
    kurtosis = staticmethod(lambda m, n: 12 * (m * (5 * n - 22) * (m + n - 2) + (n - 4) * (n - 2) ** 2)
                            / (m * (n - 6) * (n - 8) * (m + n - 2)) if n > 8 else (mpmath.inf if n > 6 else mpmath.nan))
    mode = staticmethod(lambda m, n: (m - 2) / m * n / (n + 2) if m > 2 else mpf(0))

    @staticmethod
    def entropy(m, n):
        a, b = m / 2, n / 2
        return (mpmath.log(n / m) + mpmath.log(mpmath.beta(a, b)) + (1 - a) * mpmath.digamma(a)
                - (1 + b) * mpmath.digamma(b) + (a + b) * mpmath.digamma(a + b))

    @staticmethod
    def _params(rng, n):
        m = loguniform(rng, 1e-1, 1e5, n)
        d = loguniform(rng, 1e-1, 1e5, n)
        return m, d

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        m, d = ps
        x = _quantile_points(rng, m.size, lambda p, i: st.f.ppf(p, m[i], d[i]), lambda q, i: st.f.isf(q, m[i], d[i]),
                             lambda rng, k: loguniform(rng, 1e-300, 1e300, k))
        return np.where(np.isfinite(x) & (x > 0), x, loguniform(rng, 1e-300, 1e300, x.size))

    @classmethod
    def param_sets(cls, rng):
        (m, d), g = super().param_sets(rng)
        edges = np.concatenate([around([2.0, 4.0, 6.0, 8.0], 3), loguniform(rng, 1e5, 1e12, 30)])
        return (np.concatenate([m, loguniform(rng, 1e-1, 1e3, edges.size)]), np.concatenate([d, edges])), \
            np.ones(m.size + edges.size, dtype=bool)


# ---------------------------------------------------------------- discrete (T5)
#
# A discrete class sets `discrete = True`, defines pmf, cdf and ccdf at integers (the
# cdfs at floor(x)), the moments and `support(params...)` -> (lo, hi). The inverses'
# references are exact integers found by search on the mpmath cdf (stats_ref's
# discrete_inverse), the mode by scanning the pmf (a tie -> NaN, T25).


def _floor(x):
    return mpmath.floor(x)


def _is_int(k):
    return mpmath.isfinite(k) and mpmath.floor(k) == k


def _quantile_ints(rng, n, ppf, isf, extra):
    """Integers spread over a discrete distribution (scipy's ppf/isf as a sampler), a few
    non-integers (k + 1/2: the cdfs at floor(x)), and extra points."""
    import warnings
    k = n // 5
    p = np.concatenate([rng.uniform(0.01, 0.99, k), loguniform(rng, 1e-300, 0.01, k)])
    q = loguniform(rng, 1e-300, 0.01, k)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        lo = ppf(p, np.arange(p.size))
        hi = isf(q, np.arange(p.size, p.size + q.size))
    x = np.concatenate([lo, hi, extra(rng, n - lo.size - hi.size)])
    bad = ~np.isfinite(x)
    x[bad] = extra(rng, int(bad.sum()))
    half = rng.random(x.size) < 0.05
    x[half] += 0.5
    return x


def _entropy_sum(D, p, lo, hi, centre, spread, ratio=None):
    """-sum pmf log pmf over the integers within `spread` of `centre` (the rest negligible);
    with `ratio` (pmf(k + 1) / pmf(k)) the pmf by that recurrence from the first point."""
    a = max(lo, mpmath.floor(centre - spread))
    b = min(hi, mpmath.ceil(centre + spread))
    h = mpf(0)
    k = a
    v = D.pmf(k, *p)
    while k <= b:
        if ratio is None:
            v = D.pmf(k, *p)
        if v > 0:
            h -= v * mpmath.log(v)
        if ratio is not None:
            v *= ratio(k)
        k += 1
    return h


def _mp_mode(D, p, lo, hi, guess):
    """The argmax of the pmf near `guess` (a unimodal pmf), NaN for a tie."""
    k = min(max(lo, mpmath.floor(guess)), hi)
    while k > lo and D.pmf(k - 1, *p) > D.pmf(k, *p):
        k -= 1
    while k < hi and D.pmf(k + 1, *p) > D.pmf(k, *p):
        k += 1
    v = D.pmf(k, *p)
    tol = v * mpf(10) ** -35
    for j in (k - 1, k + 1):
        if lo <= j <= hi and abs(D.pmf(j, *p) - v) <= tol:
            return mpmath.nan
    return k


class Discrete(Distribution):
    discrete = True

    @classmethod
    def x_sets(cls, rng):
        ps = cls._params(rng, 6000)
        x = cls._x(rng, ps)
        return (*ps, x), np.isfinite(x)


class Binomial(Discrete):
    params = ("n", "p")

    @staticmethod
    def support(n, p):
        return mpf(0), n

    @staticmethod
    def pmf(k, n, p):
        if not _is_int(k) or k < 0 or k > n:
            return mpf(0)
        return mpmath.binomial(n, k) * p**k * (1 - p) ** (n - k)

    @staticmethod
    def cdf(x, n, p):
        k = _floor(x)
        if k < 0:
            return mpf(0)
        if k >= n:
            return mpf(1)
        return mpmath.betainc(n - k, k + 1, 0, 1 - p, regularized=True)

    @staticmethod
    def ccdf(x, n, p):
        k = _floor(x)
        if k < 0:
            return mpf(1)
        if k >= n:
            return mpf(0)
        return mpmath.betainc(k + 1, n - k, 0, p, regularized=True)

    mean = staticmethod(lambda n, p: n * p)
    variance = staticmethod(lambda n, p: n * p * (1 - p))
    standard_deviation = staticmethod(lambda n, p: mpmath.sqrt(n * p * (1 - p)))
    skewness = staticmethod(lambda n, p: (1 - 2 * p) / mpmath.sqrt(n * p * (1 - p)))
    kurtosis = staticmethod(lambda n, p: (1 - 6 * p * (1 - p)) / (n * p * (1 - p)))
    mode = staticmethod(lambda n, p: _mp_mode(Binomial, (n, p), 0, n, (n + 1) * p))
    entropy = staticmethod(lambda n, p: _entropy_sum(Binomial, (n, p), 0, n, n * p, 40 * mpmath.sqrt(n * p * (1 - p)) + 60))

    @staticmethod
    def _params(rng, n):
        q = n // 4
        nn = np.concatenate([np.floor(loguniform(rng, 1, 1e4, n - q)), np.floor(loguniform(rng, 1e4, 1e6, q))])
        p = np.concatenate([rng.uniform(0, 1, n // 2), loguniform(rng, 1e-8, 0.5, n // 4), 1 - loguniform(rng, 1e-8, 0.5, n - n // 2 - n // 4)])
        rng.shuffle(nn)
        return nn, p

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        nn, p = ps
        return _quantile_ints(rng, nn.size, lambda q, i: st.binom.ppf(q, nn[i], p[i]), lambda q, i: st.binom.isf(q, nn[i], p[i]),
                              lambda rng, k: np.floor(rng.uniform(0, 1, k) * nn[:k]))

    @classmethod
    def param_sets(cls, rng):
        nn, p = cls._params(rng, 800)
        p = np.concatenate([p, [0.5, 0.25, 0.2113248654051871, 0.7886751345948129, 0.0, 1.0]])
        nn = np.concatenate([nn, [9.0, 7.0, 30.0, 30.0, 10.0, 10.0]])
        return (nn, p), np.ones(nn.size, dtype=bool)


class Bernoulli(Discrete):
    params = ("p",)
    support = staticmethod(lambda p: (mpf(0), mpf(1)))
    # 1 - p exactly (a p of 1e-300 vanishes in 1 - p at any working precision)
    pmf = staticmethod(lambda k, p: (mpmath.fsub(1, p, exact=True) if k == 0 else (p if k == 1 else mpf(0))))
    cdf = staticmethod(lambda x, p: mpf(0) if x < 0 else (mpmath.fsub(1, p, exact=True) if x < 1 else mpf(1)))
    ccdf = staticmethod(lambda x, p: mpf(1) if x < 0 else (p if x < 1 else mpf(0)))
    mean = staticmethod(lambda p: p)
    variance = staticmethod(lambda p: p * (1 - p))
    standard_deviation = staticmethod(lambda p: mpmath.sqrt(p * (1 - p)))
    skewness = staticmethod(lambda p: (1 - 2 * p) / mpmath.sqrt(p * (1 - p)))
    kurtosis = staticmethod(lambda p: (1 - 6 * p * (1 - p)) / (p * (1 - p)))
    mode = staticmethod(lambda p: mpf(0) if p < mpf(1) / 2 else (mpf(1) if p > mpf(1) / 2 else mpmath.nan))
    entropy = staticmethod(lambda p: -(p * mpmath.log(p) + (1 - p) * mpmath.log1p(-p)) if 0 < p < 1 else mpf(0))

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([rng.uniform(0, 1, n // 2), loguniform(rng, 1e-300, 0.5, n // 4),
                                1 - loguniform(rng, 2**-53, 0.5, n - n // 2 - n // 4)]),)

    @staticmethod
    def _x(rng, ps):
        return rng.choice([-1.0, 0.0, 0.5, 1.0, 2.0], ps[0].size)


class Poisson(Discrete):
    params = ("mu",)
    support = staticmethod(lambda mu: (mpf(0), mpmath.inf))

    @staticmethod
    def pmf(k, mu):
        if not _is_int(k) or k < 0:
            return mpf(0)
        return mpmath.exp(k * mpmath.log(mu) - mu - mpmath.loggamma(k + 1))

    cdf = staticmethod(lambda x, mu: mpf(0) if x < 0 else mpmath.gammainc(_floor(x) + 1, mu, mpmath.inf, regularized=True))
    ccdf = staticmethod(lambda x, mu: mpf(1) if x < 0 else mpmath.gammainc(_floor(x) + 1, 0, mu, regularized=True))
    mean = variance = staticmethod(lambda mu: mu)
    standard_deviation = staticmethod(lambda mu: mpmath.sqrt(mu))
    skewness = staticmethod(lambda mu: 1 / mpmath.sqrt(mu))
    kurtosis = staticmethod(lambda mu: 1 / mu)
    mode = staticmethod(lambda mu: _mp_mode(Poisson, (mu,), 0, mpmath.inf, mu))
    entropy = staticmethod(lambda mu: _entropy_sum(Poisson, (mu,), 0, mpmath.inf, mu, 40 * mpmath.sqrt(mu) + 60))

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([loguniform(rng, 1e-3, 1e4, n - n // 8), loguniform(rng, 1e4, 1e6, n // 8)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        (mu,) = ps
        return _quantile_ints(rng, mu.size, lambda q, i: st.poisson.ppf(q, mu[i]), lambda q, i: st.poisson.isf(q, mu[i]),
                              lambda rng, k: np.floor(loguniform(rng, 1, 1e6, k)))

    @classmethod
    def param_sets(cls, rng):
        (mu,) = cls._params(rng, 600)
        return (np.concatenate([mu, [1.0, 2.0, 3.0, 10.0, 0.5]]),), np.ones(mu.size + 5, dtype=bool)


class Geometric(Discrete):
    params = ("p",)
    support = staticmethod(lambda p: (mpf(1), mpmath.inf))
    pmf = staticmethod(lambda k, p: (1 - p) ** (k - 1) * p if _is_int(k) and k >= 1 else mpf(0))
    cdf = staticmethod(lambda x, p: mpf(0) if x < 1 else -mpmath.expm1(_floor(x) * mpmath.log1p(-p)))
    ccdf = staticmethod(lambda x, p: mpf(1) if x < 1 else mpmath.exp(_floor(x) * mpmath.log1p(-p)))
    mean = staticmethod(lambda p: 1 / p)
    variance = staticmethod(lambda p: (1 - p) / p**2)
    standard_deviation = staticmethod(lambda p: mpmath.sqrt(1 - p) / p)
    skewness = staticmethod(lambda p: (2 - p) / mpmath.sqrt(1 - p))
    kurtosis = staticmethod(lambda p: 6 + p**2 / (1 - p))
    mode = staticmethod(lambda p: mpf(1))
    entropy = staticmethod(lambda p: -((1 - p) * mpmath.log(1 - p) + p * mpmath.log(p)) / p)

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([rng.uniform(0.001, 0.999, n // 2), loguniform(rng, 1e-9, 0.5, n // 4),
                                1 - loguniform(rng, 1e-9, 0.5, n - n // 2 - n // 4)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        (p,) = ps
        return _quantile_ints(rng, p.size, lambda q, i: st.geom.ppf(q, p[i]), lambda q, i: st.geom.isf(q, p[i]),
                              lambda rng, k: np.floor(loguniform(rng, 1, 1e12, k)))


class NegativeBinomial(Discrete):
    params = ("n", "p")
    support = staticmethod(lambda n, p: (mpf(0), mpmath.inf))

    @staticmethod
    def pmf(k, n, p):
        if not _is_int(k) or k < 0:
            return mpf(0)
        return mpmath.exp(mpmath.loggamma(k + n) - mpmath.loggamma(n) - mpmath.loggamma(k + 1) + n * mpmath.log(p)
                          + k * mpmath.log1p(-p))

    cdf = staticmethod(lambda x, n, p: mpf(0) if x < 0 else mpmath.betainc(n, _floor(x) + 1, 0, p, regularized=True))
    ccdf = staticmethod(lambda x, n, p: mpf(1) if x < 0 else mpmath.betainc(_floor(x) + 1, n, 0, 1 - p, regularized=True))
    mean = staticmethod(lambda n, p: n * (1 - p) / p)
    variance = staticmethod(lambda n, p: n * (1 - p) / p**2)
    standard_deviation = staticmethod(lambda n, p: mpmath.sqrt(n * (1 - p)) / p)
    skewness = staticmethod(lambda n, p: (2 - p) / mpmath.sqrt(n * (1 - p)))
    kurtosis = staticmethod(lambda n, p: 6 / n + p**2 / (n * (1 - p)))
    mode = staticmethod(lambda n, p: _mp_mode(NegativeBinomial, (n, p), 0, mpmath.inf, (n - 1) * (1 - p) / p if n > 1 else 0))
    # the range: 40 sd and, for a small n (mode 0, the pmf ~ (1 - p)^k), 60 / p more for
    # the exponential tail
    entropy = staticmethod(lambda n, p: _entropy_sum(NegativeBinomial, (n, p), 0, mpmath.inf, n * (1 - p) / p,
                                                     (40 * mpmath.sqrt(n * (1 - p)) + 60) / p + 60,
                                                     lambda k: (k + n) * (1 - p) / (k + 1)))

    @staticmethod
    def _params(rng, n):
        nn = np.concatenate([loguniform(rng, 0.1, 1e4, n - n // 8), np.floor(loguniform(rng, 1, 100, n // 8))])
        p = np.concatenate([rng.uniform(0.01, 0.99, n // 2), loguniform(rng, 1e-4, 0.5, n // 4), 1 - loguniform(rng, 1e-6, 0.5, n - n // 2 - n // 4)])
        rng.shuffle(nn)
        return nn, p

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        nn, p = ps
        return _quantile_ints(rng, nn.size, lambda q, i: st.nbinom.ppf(q, nn[i], p[i]), lambda q, i: st.nbinom.isf(q, nn[i], p[i]),
                              lambda rng, k: np.floor(loguniform(rng, 1, 1e8, k)))

    @classmethod
    def param_sets(cls, rng):
        nn, p = cls._params(rng, 600)
        return (np.concatenate([nn, [3.0, 5.0]]), np.concatenate([p, [0.5, 0.2]])), np.ones(nn.size + 2, dtype=bool)


class Hypergeometric(Discrete):
    params = ("total", "successes", "draws")

    @staticmethod
    def support(N, K, n):
        return max(mpf(0), n - (N - K)), min(n, K)

    @staticmethod
    def pmf(k, N, K, n):
        lo, hi = Hypergeometric.support(N, K, n)
        if not _is_int(k) or k < lo or k > hi:
            return mpf(0)
        return mpmath.binomial(K, k) * mpmath.binomial(N - K, n - k) / mpmath.binomial(N, n)

    @staticmethod
    def cdf(x, N, K, n):
        lo, hi = Hypergeometric.support(N, K, n)
        k = _floor(x)
        if k < lo:
            return mpf(0)
        if k >= hi:
            return mpf(1)
        return mpmath.fsum(Hypergeometric.pmf(j, N, K, n) for j in mpmath.arange(lo, k + 1))

    @staticmethod
    def ccdf(x, N, K, n):
        lo, hi = Hypergeometric.support(N, K, n)
        k = _floor(x)
        if k < lo:
            return mpf(1)
        if k >= hi:
            return mpf(0)
        return mpmath.fsum(Hypergeometric.pmf(j, N, K, n) for j in mpmath.arange(k + 1, hi + 1))

    @staticmethod
    def _central(N, K, n, m):
        lo, hi = Hypergeometric.support(N, K, n)
        mu = n * K / N
        return mpmath.fsum((j - mu) ** m * Hypergeometric.pmf(j, N, K, n) for j in mpmath.arange(lo, hi + 1))

    mean = staticmethod(lambda N, K, n: n * K / N)
    variance = staticmethod(lambda N, K, n: Hypergeometric._central(N, K, n, 2))
    standard_deviation = staticmethod(lambda N, K, n: mpmath.sqrt(Hypergeometric._central(N, K, n, 2)))
    skewness = staticmethod(lambda N, K, n: Hypergeometric._central(N, K, n, 3) / Hypergeometric._central(N, K, n, 2) ** mpf(1.5))
    kurtosis = staticmethod(lambda N, K, n: Hypergeometric._central(N, K, n, 4) / Hypergeometric._central(N, K, n, 2) ** 2 - 3)
    mode = staticmethod(lambda N, K, n: _mp_mode(Hypergeometric, (N, K, n), *Hypergeometric.support(N, K, n), n * K / N))

    @staticmethod
    def entropy(N, K, n):
        lo, hi = Hypergeometric.support(N, K, n)
        return _entropy_sum(Hypergeometric, (N, K, n), lo, hi, (lo + hi) / 2, (hi - lo) / 2 + 1)

    @staticmethod
    def _params(rng, n):
        N = np.floor(loguniform(rng, 2, 3000, n))
        K = np.floor(rng.uniform(0, 1, n) * (N + 1))
        d = np.floor(rng.uniform(0, 1, n) * (N + 1))
        return N, K, d

    @staticmethod
    def _x(rng, ps):
        N, K, d = ps
        lo, hi = np.maximum(0, d - (N - K)), np.minimum(d, K)
        x = np.floor(lo + rng.uniform(-0.05, 1.05, N.size) * (hi - lo + 1))
        half = rng.random(x.size) < 0.05
        return x + 0.5 * half

    @classmethod
    def param_sets(cls, rng):
        N, K, d = cls._params(rng, 500)
        return (N, K, d), np.ones(N.size, dtype=bool)


class DiscreteUniform(Discrete):
    params = ("low", "high")
    support = staticmethod(lambda lo, hi: (lo, hi - 1))
    pmf = staticmethod(lambda k, lo, hi: 1 / (hi - lo) if _is_int(k) and lo <= k < hi else mpf(0))
    cdf = staticmethod(lambda x, lo, hi: mpf(0) if _floor(x) < lo else (mpf(1) if _floor(x) >= hi - 1 else (_floor(x) - lo + 1) / (hi - lo)))
    ccdf = staticmethod(lambda x, lo, hi: mpf(1) if _floor(x) < lo else (mpf(0) if _floor(x) >= hi - 1 else (hi - 1 - _floor(x)) / (hi - lo)))
    mean = staticmethod(lambda lo, hi: (lo + hi - 1) / 2)
    variance = staticmethod(lambda lo, hi: ((hi - lo) ** 2 - 1) / 12)
    standard_deviation = staticmethod(lambda lo, hi: mpmath.sqrt(((hi - lo) ** 2 - 1) / 12))
    skewness = staticmethod(lambda lo, hi: mpf(0))
    kurtosis = staticmethod(lambda lo, hi: -6 * ((hi - lo) ** 2 + 1) / (5 * ((hi - lo) ** 2 - 1)))
    mode = staticmethod(lambda lo, hi: mpmath.nan)
    entropy = staticmethod(lambda lo, hi: mpmath.log(hi - lo))

    @staticmethod
    def _params(rng, n):
        lo = np.floor(np.concatenate([rng.uniform(-100, 100, n // 2), loguniform(rng, 1, 1e12, n - n // 2, signed=True)]))
        w = np.floor(loguniform(rng, 2, 1e9, n))
        return lo, lo + w

    @staticmethod
    def _x(rng, ps):
        lo, hi = ps
        x = np.floor(lo + rng.uniform(-0.05, 1.05, lo.size) * (hi - lo))
        return x + 0.5 * (rng.random(x.size) < 0.05)


class BetaBinomial(Discrete):
    params = ("n", "a", "b")
    support = staticmethod(lambda n, a, b: (mpf(0), n))

    @staticmethod
    def pmf(k, n, a, b):
        if not _is_int(k) or k < 0 or k > n:
            return mpf(0)
        return mpmath.binomial(n, k) * mpmath.beta(k + a, n - k + b) / mpmath.beta(a, b)

    cdf = staticmethod(lambda x, n, a, b: mpf(0) if _floor(x) < 0 else (mpf(1) if _floor(x) >= n else
                       mpmath.fsum(BetaBinomial.pmf(j, n, a, b) for j in mpmath.arange(0, _floor(x) + 1))))
    ccdf = staticmethod(lambda x, n, a, b: mpf(1) if _floor(x) < 0 else (mpf(0) if _floor(x) >= n else
                        mpmath.fsum(BetaBinomial.pmf(j, n, a, b) for j in mpmath.arange(_floor(x) + 1, n + 1))))

    @staticmethod
    def _central(n, a, b, m):
        mu = n * a / (a + b)
        return mpmath.fsum((j - mu) ** m * BetaBinomial.pmf(j, n, a, b) for j in mpmath.arange(0, n + 1))

    mean = staticmethod(lambda n, a, b: n * a / (a + b))
    variance = staticmethod(lambda n, a, b: n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1)))
    standard_deviation = staticmethod(lambda n, a, b: mpmath.sqrt(n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1))))
    skewness = staticmethod(lambda n, a, b: BetaBinomial._central(n, a, b, 3) / BetaBinomial._central(n, a, b, 2) ** mpf(1.5))
    kurtosis = staticmethod(lambda n, a, b: BetaBinomial._central(n, a, b, 4) / BetaBinomial._central(n, a, b, 2) ** 2 - 3)

    @staticmethod
    def mode(n, a, b):
        vals = [BetaBinomial.pmf(j, n, a, b) for j in mpmath.arange(0, n + 1)]
        m = max(vals)
        idx = [i for i, v in enumerate(vals) if abs(v - m) <= m * mpf(10) ** -35]
        return mpf(idx[0]) if len(idx) == 1 else mpmath.nan

    entropy = staticmethod(lambda n, a, b: _entropy_sum(BetaBinomial, (n, a, b), 0, n, n / 2, n / 2 + 1))

    @staticmethod
    def _params(rng, n):
        nn = np.floor(loguniform(rng, 1, 2000, n))
        a = loguniform(rng, 0.1, 100, n)
        b = loguniform(rng, 0.1, 100, n)
        return nn, a, b

    @staticmethod
    def _x(rng, ps):
        nn, a, b = ps
        x = np.floor(rng.uniform(-0.05, 1.05, nn.size) * (nn + 1))
        return x + 0.5 * (rng.random(x.size) < 0.05)

    @classmethod
    def param_sets(cls, rng):
        nn, a, b = cls._params(rng, 400)
        return (np.concatenate([nn, [10.0, 10.0, 5.0]]), np.concatenate([a, [1.0, 0.5, 2.0]]),
                np.concatenate([b, [1.0, 0.5, 3.0]])), np.ones(nn.size + 3, dtype=bool)


def _logser_l(p):
    return -mpmath.log1p(-p)


class LogSeries(Discrete):
    params = ("p",)
    support = staticmethod(lambda p: (mpf(1), mpmath.inf))
    pmf = staticmethod(lambda k, p: p**k / (k * _logser_l(p)) if _is_int(k) and k >= 1 else mpf(0))

    @staticmethod
    def ccdf(x, p):
        k = _floor(x)
        if k < 1:
            return mpf(1)
        return p ** (k + 1) * mpmath.lerchphi(p, 1, k + 1) / _logser_l(p)

    @staticmethod
    def cdf(x, p):
        k = _floor(x)
        if k < 1:
            return mpf(0)
        if k <= 2000:
            return mpmath.fsum(p**j / j for j in mpmath.arange(1, k + 1)) / _logser_l(p)
        return 1 - LogSeries.ccdf(x, p)

    @staticmethod
    def _raw(p, r):
        L, w = _logser_l(p), 1 - p
        A = [mpf(1), mpf(1), 1 + p, 1 + 4 * p + p * p][r - 1]
        return p * A / (L * w**r)

    mean = staticmethod(lambda p: LogSeries._raw(p, 1))
    variance = staticmethod(lambda p: LogSeries._raw(p, 2) - LogSeries._raw(p, 1) ** 2)
    standard_deviation = staticmethod(lambda p: mpmath.sqrt(LogSeries._raw(p, 2) - LogSeries._raw(p, 1) ** 2))

    @staticmethod
    def _central(p, m):
        e = [LogSeries._raw(p, r) for r in (1, 2, 3, 4)]
        mu = e[0]
        if m == 2:
            return e[1] - mu**2
        if m == 3:
            return e[2] - 3 * mu * e[1] + 2 * mu**3
        return e[3] - 4 * mu * e[2] + 6 * mu**2 * e[1] - 3 * mu**4

    skewness = staticmethod(lambda p: LogSeries._central(p, 3) / LogSeries._central(p, 2) ** mpf(1.5))
    kurtosis = staticmethod(lambda p: LogSeries._central(p, 4) / LogSeries._central(p, 2) ** 2 - 3)
    mode = staticmethod(lambda p: mpf(1))

    @staticmethod
    def entropy(p):
        L = _logser_l(p)
        return mpmath.log(L) - mpmath.log(p) * LogSeries._raw(p, 1) + mpmath.nsum(lambda k: p**k * mpmath.log(k) / (k * L), [2, mpmath.inf])

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([rng.uniform(0.001, 0.999, n // 2), loguniform(rng, 1e-8, 0.5, n // 4),
                                1 - loguniform(rng, 1e-6, 0.5, n - n // 2 - n // 4)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        (p,) = ps
        import stardust as sd  # scipy's logser.ppf is slow in the far tail: our own icdf as the sampler

        return _quantile_ints(rng, p.size, lambda q, i: np.asarray(sd.stats.LogSeries(p=sd.asarray(p[i])).icdf(sd.asarray(q))),
                              lambda q, i: np.asarray(sd.stats.LogSeries(p=sd.asarray(p[i])).iccdf(sd.asarray(q))),
                              lambda rng, k: np.floor(loguniform(rng, 1, 1e7, k)))

    @classmethod
    def param_sets(cls, rng):
        (p,) = cls._params(rng, 600)
        return (p,), np.ones(p.size, dtype=bool)


class Zipf(Discrete):
    params = ("a",)
    support = staticmethod(lambda a: (mpf(1), mpmath.inf))
    pmf = staticmethod(lambda k, a: k ** (-a) / mpmath.zeta(a) if _is_int(k) and k >= 1 else mpf(0))
    ccdf = staticmethod(lambda x, a: mpf(1) if _floor(x) < 1 else mpmath.zeta(a, _floor(x) + 1) / mpmath.zeta(a))

    @staticmethod
    def cdf(x, a):
        k = _floor(x)
        if k < 1:
            return mpf(0)
        return (mpmath.zeta(a) - mpmath.zeta(a, k + 1)) / mpmath.zeta(a)

    @staticmethod
    def _raw(a, r):
        return mpmath.zeta(a - r) / mpmath.zeta(a)

    mean = staticmethod(lambda a: Zipf._raw(a, 1) if a > 2 else mpmath.inf)

    @staticmethod
    def _central(a, m):
        e = [Zipf._raw(a, r) for r in (1, 2, 3, 4)[:m]]
        mu = e[0]
        if m == 2:
            return e[1] - mu**2
        if m == 3:
            return e[2] - 3 * mu * e[1] + 2 * mu**3
        return e[3] - 4 * mu * e[2] + 6 * mu**2 * e[1] - 3 * mu**4

    variance = staticmethod(lambda a: Zipf._central(a, 2) if a > 3 else (mpmath.inf if a > 2 else mpmath.nan))
    standard_deviation = staticmethod(lambda a: mpmath.sqrt(Zipf._central(a, 2)) if a > 3 else (mpmath.inf if a > 2 else mpmath.nan))
    skewness = staticmethod(lambda a: Zipf._central(a, 3) / Zipf._central(a, 2) ** mpf(1.5) if a > 4 else (mpmath.inf if a > 3 else mpmath.nan))
    kurtosis = staticmethod(lambda a: Zipf._central(a, 4) / Zipf._central(a, 2) ** 2 - 3 if a > 5 else (mpmath.inf if a > 3 else mpmath.nan))
    mode = staticmethod(lambda a: mpf(1))
    entropy = staticmethod(lambda a: mpmath.log(mpmath.zeta(a)) - a * mpmath.zeta(a, 1, 1) / mpmath.zeta(a))

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([1 + loguniform(rng, 1e-3, 1, n // 2), loguniform(rng, 2, 60, n - n // 2)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st
        (a,) = ps
        import stardust as sd  # scipy's zipf.ppf runs out of memory for a near 1 and a tiny q

        return _quantile_ints(rng, a.size, lambda q, i: np.asarray(sd.stats.Zipf(a=sd.asarray(a[i])).icdf(sd.asarray(q))),
                              lambda q, i: np.asarray(sd.stats.Zipf(a=sd.asarray(a[i])).iccdf(sd.asarray(q))),
                              lambda rng, k: np.floor(loguniform(rng, 1, 1e12, k)))

    @classmethod
    def param_sets(cls, rng):
        (a,) = cls._params(rng, 600)
        return (a,), np.ones(a.size, dtype=bool)


DISTRIBUTIONS = {"normal": Normal, "exponential": Exponential, "uniform": Uniform, "laplace": Laplace,
                 "cauchy": Cauchy, "lognormal": LogNormal, "weibull": Weibull, "gamma": Gamma,
                 "chisquared": ChiSquared, "beta": Beta, "studentt": StudentT, "f": F,
                 "binomial": Binomial, "bernoulli": Bernoulli, "poisson": Poisson, "geometric": Geometric,
                 "negativebinomial": NegativeBinomial, "hypergeometric": Hypergeometric,
                 "discreteuniform": DiscreteUniform, "betabinomial": BetaBinomial, "logseries": LogSeries, "zipf": Zipf}
DISCRETE_INPUT_METHODS = ("pmf", "logpmf") + INPUT_METHODS[2:]


# ---------------------------------------------------------------- method definitions


def _log_cdf(D, x, p, upper):
    """log F(x) (upper: log(1 - F)): the log of the smaller side, log1p(-other) for the larger."""
    lo, hi = (D.ccdf, D.cdf) if upper else (D.cdf, D.ccdf)
    v = lo(x, *p)
    if v <= 0.5:
        return mpmath.log(v)
    return mpmath.log1p(-hi(x, *p))


def method_value(D, m, args):
    """The reference value of a non-inverse method at mpf args (params..., x)."""
    if m in PARAM_METHODS:
        return getattr(D, m)(*args)
    *p, x = args
    if m in ("pdf", "pmf"):
        return D.pmf(x, *p) if m == "pmf" else D.pdf(x, *p)
    if m == "logpmf":
        v = D.pmf(x, *p)
        return mpmath.log(v) if v > 0 else -mpmath.inf
    if m == "logpdf":
        return D.logpdf(x, *p) if hasattr(D, "logpdf") else mpmath.log(D.pdf(x, *p))
    if m == "cdf":
        return D.cdf(x, *p)
    if m == "ccdf":
        return D.ccdf(x, *p)
    if m == "logcdf":
        return _log_cdf(D, x, p, False)
    if m == "logccdf":
        return _log_cdf(D, x, p, True)
    raise KeyError(m)


def _newton(D, p, target, upper, x0):
    """x with log F(x) = target (upper: log(1 - F)), Newton from x0; in log x for a positive
    support."""
    side = D.ccdf if upper else D.cdf
    sign = -1 if upper else 1
    tol = mpf(10) ** -40
    if getattr(D, "unit", False):  # support (0, 1): Newton in u = logit x, dx/du = x (1 - x)
        u = mpmath.log(mpf(x0) / (1 - mpf(x0)))
        for _ in range(200):
            x = 1 / (1 + mpmath.exp(-u))
            F, f = side(x, *p), D.pdf(x, *p)
            du = (mpmath.log(F) - target) / (sign * f * x * (1 - x) / F)
            u -= du
            if abs(du) <= tol * max(1, abs(u)):
                return 1 / (1 + mpmath.exp(-u))
        raise RuntimeError("no convergence")
    if getattr(D, "positive", False):
        u = mpmath.log(mpf(x0))
        for _ in range(200):
            x = mpmath.exp(u)
            F, f = side(x, *p), D.pdf(x, *p)
            du = (mpmath.log(F) - target) / (sign * f * x / F)
            u -= du
            if abs(du) <= tol:
                return mpmath.exp(u)
        raise RuntimeError("no convergence")
    x = mpf(x0)
    for _ in range(200):
        F, f = side(x, *p), D.pdf(x, *p)
        dx = (mpmath.log(F) - target) / (sign * f / F)
        x -= dx
        if abs(dx) <= (abs(x) + mpf(10) ** -300) * tol:
            return x
    raise RuntimeError("no convergence")


def _closed_inverse(D, m, p, t):
    """x from the distribution's closed-form mpmath inverses (icdf_mp, iccdf_mp), or None."""
    if not hasattr(D, "icdf_mp"):
        return None
    lower = lambda v: D.icdf_mp(v, *p) if v <= mpf(1) / 2 else D.iccdf_mp(1 - v, *p)
    upper = lambda v: D.iccdf_mp(v, *p) if v <= mpf(1) / 2 else D.icdf_mp(1 - v, *p)
    if m == "icdf":
        return lower(t)
    if m == "iccdf":
        return upper(t)
    small = t < -mpmath.log(2)  # e^t is the small side
    if m == "ilogcdf":
        return D.icdf_mp(mpmath.exp(t), *p) if small else D.iccdf_mp(-mpmath.expm1(t), *p)
    return D.iccdf_mp(mpmath.exp(t), *p) if small else D.icdf_mp(-mpmath.expm1(t), *p)


def _beyond_range(D, m, p, t, x0):
    """Our root is 0 or +-inf: accept it as the reference when the true root lies beyond the
    float64 range (it then rounds to x0), checked on the side solved: the target log
    probability against the side's log at the range's edge (half the smallest subnormal
    for 0, the largest float64 for +-inf)."""
    if np.isnan(x0):
        raise ValueError("NaN start")
    if m in ("icdf", "iccdf"):
        upper, target = m == "iccdf", mpmath.log(t)
        if t > mpf(1) / 2:
            upper, target = not upper, mpmath.log(1 - t)
    else:
        upper, target = m == "ilogccdf", t
        if t >= -mpmath.log(2):
            upper, target = not upper, mpmath.log(-mpmath.expm1(t))
    if x0 == 0:
        edge = mpf(2) ** -1075  # half the smallest subnormal: a root below rounds to 0 (2.0**-1075 is 0.0)
    elif getattr(D, "unit", False) and x0 == 1:
        edge = 1 - mpf(2) ** -54  # half an ULP below 1: a root above rounds to 1
    else:
        edge = mpf(np.finfo(np.float64).max) * (1 if x0 > 0 else -1)
    side = (D.ccdf if upper else D.cdf)(edge, *p)
    # the lower side increases with x, the upper side decreases: the root is past the low
    # edge (0 or -inf) when the side there is already above the target (lower) or still
    # below it (upper), and past the high edge the other way round
    below = x0 <= 0
    above_target = side > 0 and mpmath.log(side) > target
    if above_target != (below == (not upper)):
        raise ValueError("root inside the range")
    return mpf(x0)


def inverse_value(D, m, args, x0):
    """(x, kappa) of an inverse method at mpf args (params..., p or logp). kappa is the
    relative condition number in the method's own input: |p / (x f)| for icdf and iccdf,
    |logp p / (x f)| (p = e^logp) for ilogcdf and ilogccdf."""
    *p, t = args
    x = _closed_inverse(D, m, p, t)
    if x is None and (x0 == 0 or not np.isfinite(x0) or (getattr(D, "unit", False) and x0 == 1)):
        return _beyond_range(D, m, p, t, x0), mpf(1)
    if x is None:
        if m in ("icdf", "iccdf"):
            upper, target = m == "iccdf", t
            if t > mpf(1) / 2:  # solve on the other side, where 1 - t is exact
                upper, target = not upper, 1 - t
            x = _newton(D, p, mpmath.log(target), upper, x0)
        else:
            upper = m == "ilogccdf"
            if t < -mpmath.log(2):
                x = _newton(D, p, t, upper, x0)
            else:  # the larger side: solve for the other one at log(1 - e^t)
                x = _newton(D, p, mpmath.log(-mpmath.expm1(t)), not upper, x0)
    f = D.pdf(x, *p)
    if x == 0 or f == 0 or mpmath.isinf(x):
        return x, mpf(1)
    if m in ("icdf", "iccdf"):
        return x, t / (f * abs(x))
    return x, abs(t) * mpmath.exp(t) / (f * abs(x))


def _smallest_int(lo, hi, k0, pred):
    """The smallest integer k in [lo, hi] with pred(k) (pred monotone, true at hi)."""
    k = k0 if (k0 is not None and mpmath.isfinite(k0)) else lo
    k = min(max(lo, mpmath.floor(k)), hi)
    if pred(k):
        if k == lo or not pred(k - 1):
            return k
        b, step = k, mpf(1)
        while True:
            c = max(lo, b - step)
            if not pred(c):
                a = c
                break
            b = c
            if c == lo:
                return lo
            step *= 2
    else:
        a, step = k, mpf(1)
        while True:
            c = a + step
            if c >= hi:
                b = hi
                break
            if c > mpf(2) ** 1024:  # beyond float64: the reference is inf (our search stops there too)
                return mpmath.inf
            if pred(c):
                b = c
                break
            a = c
            step *= 2
    while b - a > 1:
        mid = mpmath.floor((a + b) / 2)
        if pred(mid):
            b = mid
        else:
            a = mid
    return b


def discrete_inverse(D, m, p, t, k0):
    """The exact quantile of a discrete distribution (T25): icdf = smallest k with
    cdf(k) >= t, iccdf = smallest k with ccdf(k) <= t, the log forms with t = log p; each
    compared on the side whose probability is small (its complement exact in mpmath)."""
    lo, hi = D.support(*p)
    half = mpf(1) / 2
    if m == "icdf":
        pred = (lambda k: D.ccdf(k, *p) <= 1 - t) if t > half else (lambda k: D.cdf(k, *p) >= t)
    elif m == "iccdf":
        pred = (lambda k: D.cdf(k, *p) >= 1 - t) if t > half else (lambda k: D.ccdf(k, *p) <= t)
    elif m == "ilogcdf":
        q = -mpmath.expm1(t)
        pred = (lambda k: D.ccdf(k, *p) <= q) if t > -mpmath.log(2) else (lambda k: D.cdf(k, *p) >= mpmath.exp(t))
    else:
        q = -mpmath.expm1(t)
        pred = (lambda k: D.cdf(k, *p) >= q) if t > -mpmath.log(2) else (lambda k: D.ccdf(k, *p) <= mpmath.exp(t))
    return _smallest_int(lo, hi, None if k0 is None else mpf(k0), pred)


# The discrete inverses' contract is a backward error (user decision 2026-11-16): the
# answer is right when it is the exact quantile of some target within relative 1e-14 of the
# given one: for icdf and iccdf on the small side (p or 1 - p, whichever is below 1/2), for
# the log forms relative in t itself (as the continuous inverses' kappa; t = -1e5 has an ULP
# of 1.5e-11, so no bound finer than relative is meaningful there). That covers a target within the cdf's last bits of a step (the
# neighbour is as good) and a quantile so large that the steps are finer than the cdf's
# resolution (Zipf near a = 1). Each point stores the interval [kmin, kmax] of such answers.
BACKWARD = mpf(10) ** -14


def _perturbed(m, t):
    """The two ends of the targets within BACKWARD of t (method m's own input)."""
    d = BACKWARD
    if m in ("icdf", "iccdf"):
        if t <= mpf(1) / 2:
            return t * (1 - d), t * (1 + d)
        q = 1 - t
        return 1 - q * (1 + d), 1 - q * (1 - d)
    return t * (1 + d), t * (1 - d)  # t < 0: relative in t, the input's own resolution


def discrete_interval(D, m, p, t, k):
    """[kmin, kmax]: the exact quantiles at the ends of the perturbed targets (each method is
    monotone in its input, so the answers in between lie between)."""
    ends = [discrete_inverse(D, m, p, u, None if not mpmath.isfinite(k) else k) for u in _perturbed(m, t)]
    return min(ends), max(ends)


def _interval_point(task):
    dist, m, i, args, k = task
    D = DISTRIBUTIONS[dist]
    old = signal.signal(signal.SIGALRM, sr._alarm)
    signal.alarm(POINT_SECONDS)
    try:
        mm, a = ("icdf", list(args) + [0.5]) if m == "median" else (m, list(args))
        dps = 2 * sr.DPS + 20
        lo_, hi_ = _at(dps, lambda: discrete_interval(D, mm, [mpf(v) for v in a[:-1]], mpf(a[-1]), mpf(k)))
        return task, (float(lo_), float(hi_))
    except Exception:
        return task, None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def add_intervals(targets, jobs):
    """Add kmin and kmax to existing discrete inverse files (the exact k is kept); a point
    whose interval cannot be settled is dropped from the file."""
    import time

    tasks, files = [], {}
    for dist, m in targets:
        if not (getattr(DISTRIBUTIONS[dist], "discrete", False) and (m in INVERSE or m == "median")):
            continue
        path = OUT / f"{dist}_{m}.npz"
        with np.load(path) as f:
            data = dict(f)
        xs = [data[f"x{k}"] for k in range(int(data["nin"]))]
        files[(dist, m)] = (path, data, {})
        tasks += [(dist, m, i, tuple(float(x[i]) for x in xs), float(data["hi"][i])) for i in range(xs[0].size)]
    start = time.monotonic()
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        for n, (task, r) in enumerate(pool.imap_unordered(_interval_point, tasks, chunksize=16), 1):
            files[(task[0], task[1])][2][task[2]] = r
            if n % max(1, len(tasks) // 20) == 0:
                print(f"  intervals {n}/{len(tasks)}, {(time.monotonic() - start) / 60:.1f} min", flush=True)
    for (dist, m), (path, data, res) in files.items():
        n = data["hi"].size
        kmin, kmax, keep = np.empty(n), np.empty(n), np.zeros(n, dtype=bool)
        for i, r in res.items():
            if r is not None:
                kmin[i], kmax[i] = r
                keep[i] = True
        for key in ("hi", "lo", "scale", "guaranteed") + tuple(f"x{k}" for k in range(int(data["nin"]))):
            data[key] = data[key][keep]
        data["kmin"], data["kmax"] = kmin[keep], kmax[keep]
        np.savez_compressed(path, **data)
        print(f"{dist}_{m}: {int(keep.sum())} intervals" + (f", {n - int(keep.sum())} dropped" if n > keep.sum() else ""),
              flush=True)


# ---------------------------------------------------------------- building


def _at(dps, fn):
    with mpmath.workdps(dps):
        return fn()


def evaluate(fn, args):
    """fn() at increasing precision until two agree to 30 digits (sr.evaluate's rule)."""
    big = max([abs(a) for a in args if np.isfinite(a)] + [1.0])
    dps = sr.DPS + int(np.log10(big)) + 10
    v = _at(dps, fn)
    while dps < 2000:
        w = _at(2 * dps, fn)
        if mpmath.isnan(v) or mpmath.isnan(w) or mpmath.isinf(w) or v == w or abs(v - w) <= abs(w) * mpf(10) ** -30:
            return w
        dps *= 2
        v = w
    return v


def _our(dist, m, args):
    import stardust as sd

    return float(getattr(sd.stats._distribution._k, f"{dist}_{m}")(*args))


def _point(task):
    dist, m, i, args = task
    D = DISTRIBUTIONS[dist]
    old = signal.signal(signal.SIGALRM, sr._alarm)
    signal.alarm(POINT_SECONDS)
    try:
        sc = 1.0
        if getattr(D, "discrete", False) and (m in INVERSE or m == "median"):
            x0 = _our(dist, m, args)
            mm, a = ("icdf", list(args) + [0.5]) if m == "median" else (m, list(args))
            vals = []
            for dps in (sr.DPS + 10, 2 * sr.DPS + 20):
                vals.append(_at(dps, lambda: discrete_inverse(D, mm, [mpf(v) for v in a[:-1]], mpf(a[-1]),
                                                              x0 if np.isfinite(x0) else None)))
            if vals[0] != vals[1]:
                return i, None
            hi_, lo_ = sr.double_double(vals[1])
            kmin, kmax = _at(2 * sr.DPS + 20, lambda: discrete_interval(D, mm, [mpf(v) for v in a[:-1]], mpf(a[-1]),
                                                                        vals[1]))
            return i, (hi_, lo_, 1.0, float(kmin), float(kmax))
        if m == "median" and not hasattr(D, "median"):
            x0 = _our(dist, m, args)
            dps = sr.DPS + 10
            v, k = _at(dps, lambda: inverse_value(D, "icdf", [mpf(a) for a in args] + [mpf(1) / 2], x0))
            w, k = _at(2 * dps, lambda: inverse_value(D, "icdf", [mpf(a) for a in args] + [mpf(1) / 2], x0))
            hi, lo = sr.double_double(w)
            return i, (hi, lo, max(1.0, float(k)))
        if m in INVERSE:
            x0 = _our(dist, m, args)
            dps = sr.DPS + 10
            v, k = _at(dps, lambda: inverse_value(D, m, [mpf(a) for a in args], x0))
            while dps < 1000:
                w, k = _at(2 * dps, lambda: inverse_value(D, m, [mpf(a) for a in args], x0))
                if v == w or abs(v - w) <= abs(w) * mpf(10) ** -30:
                    break
                dps *= 2
                v = w
            v = w
            sc = max(1.0, float(k))
        else:
            v = evaluate(lambda: method_value(D, m, [mpf(a) for a in args]), args)
        hi, lo = sr.double_double(v)
        return i, (hi, lo, sc)
    except Exception:  # a point mpmath cannot settle (timeout, no convergence, huge exponents) is dropped
        return i, None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# The guaranteed region in the parameters: the ranges over which sd.special's functions
# underneath are measured and guaranteed (docs/api/special.md §3, §13). Points outside
# stay in the files and are held to the documented loose bound. The discrete distributions
# are guaranteed over the parameter ranges their sets sample (user decision 2026-11-16;
# docs/api/stats.md §2.3), so every point of theirs is in the region.
PARAM_RANGE = {
    "gamma": lambda a, s: a <= 1e6,
    "chisquared": lambda df: df <= 2e6,
    "beta": lambda a, b: (a <= 1e5) & (b <= 1e5),
    "studentt": lambda df: df <= 1e6,
    "f": lambda m, n: (m <= 1e5) & (n <= 1e5),
}


# Corners left out of the guaranteed region inside the parameter range (user decision
# 2026-11-15): t's inverses for df >= 1e5 with the probability below 2^-1000 (where they
# solve on the log side; csrc/stats/dist/beta.hpp kDeepP), where the beta side's BGRAT
# branch has no log form (measured 1.4e-14).
TINY_P = 2.0**-1000


def corner(dist, m, xs):
    if dist == "studentt" and m in ("icdf", "iccdf"):
        return (xs[0] >= 1e5) & (xs[-1] < TINY_P)
    if dist == "studentt" and m in ("ilogcdf", "ilogccdf"):
        return (xs[0] >= 1e5) & (xs[-1] < np.log(TINY_P))
    return np.zeros(xs[0].size, dtype=bool)


def region(dist, D, xs, guaranteed, m=None):
    """The input set's own mask, narrowed to the distribution's parameter range and
    without the documented corners."""
    g = np.asarray(guaranteed, dtype=bool)
    if dist in PARAM_RANGE:
        g = g & PARAM_RANGE[dist](*xs[:len(D.params)])
    if m is not None:
        g = g & ~corner(dist, m, xs)
    return g


_INPUT_CACHE = {}


def inputs(D, m, rng):
    """The input set of method m (the same set for every method of a kind, made once: the
    generator is reseeded per file, so the sets are identical anyway)."""
    if m in PARAM_METHODS:
        kind, make = "param", D.param_sets
    elif m in ("icdf", "iccdf"):
        kind, make = "p", D.p_sets
    elif m in ("ilogcdf", "ilogccdf"):
        kind, make = "logp", D.logp_sets
    else:
        kind, make = "x", D.x_sets
    key = (D, kind)
    if key not in _INPUT_CACHE:
        _INPUT_CACHE[key] = make(rng)
    return _INPUT_CACHE[key]


def _tasks(dist, m):
    """The inputs of one file and its point tasks."""
    D = DISTRIBUTIONS[dist]
    rng = np.random.default_rng(20261115)
    xs, guaranteed = inputs(D, m, rng)
    xs = tuple(np.asarray(x, dtype=np.float64) for x in xs)
    guaranteed = region(dist, D, xs, guaranteed, m)
    return xs, guaranteed, [(dist, m, i, tuple(float(x[i]) for x in xs)) for i in range(xs[0].size)]


def _save(dist, m, xs, guaranteed, results):
    n = xs[0].size
    hi, lo, scale = np.empty(n), np.empty(n), np.ones(n)
    kmin, kmax = np.empty(n), np.empty(n)  # the discrete inverses' intervals (BACKWARD)
    keep = np.zeros(n, dtype=bool)
    interval = False
    for i, r in results.items():
        if r is not None:
            hi[i], lo[i], scale[i] = r[:3]
            if len(r) == 5:
                kmin[i], kmax[i] = r[3:]
                interval = True
            keep[i] = True
    extra = {"kmin": kmin[keep], "kmax": kmax[keep]} if interval else {}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{dist}_{m}.npz", nin=len(xs), hi=hi[keep], lo=lo[keep], scale=scale[keep],
                        guaranteed=np.asarray(guaranteed, dtype=bool)[keep],
                        **{f"x{k}": x[keep] for k, x in enumerate(xs)}, **extra, mpmath=mpmath.__version__, dps=sr.DPS)
    return int(keep.sum()), int(n - keep.sum())


def _point_task(task):
    """(task, _point(task)): imap_unordered returns results as they finish, so each carries its task."""
    return task, _point(task)


def build_all(targets, jobs):
    """Every (distribution, method) file through one process pool: the points of all files
    are interleaved and handed out one at a time (imap_unordered, chunksize 1), so a slow
    point (mpmath's betainc near the mean of large parameters) holds one worker, not a
    whole file, and no core waits for the slowest point of a file before the next file
    starts. Each file is written as soon as its last point arrives."""
    files, pending, tasks = {}, {}, []
    for dist, m in targets:
        xs, g, t = _tasks(dist, m)
        files[(dist, m)] = (xs, g, {})
        pending[(dist, m)] = len(t)
        tasks += t
    order = np.random.default_rng(0).permutation(len(tasks))  # mix slow and fast files
    tasks = [tasks[i] for i in order]

    def done(task, r):
        key = (task[0], task[1])
        xs, g, res = files[key]
        res[r[0]] = r[1]
        pending[key] -= 1
        if pending[key] == 0:
            kept, dropped = _save(key[0], key[1], xs, g, res)
            print(f"{key[0]}_{key[1]}: {kept} points" + (f", {dropped} dropped" if dropped else ""), flush=True)

    import time

    start, total, step = time.monotonic(), len(tasks), max(1, len(tasks) // 20)
    count = 0

    def progress():
        nonlocal count
        count += 1
        if count % step == 0:
            el = time.monotonic() - start
            print(f"  progress {count}/{total} points ({100 * count // total}%), {el / 60:.1f} min, "
                  f"~{el / count * (total - count) / 60:.1f} min left", flush=True)

    if jobs > 1:
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            for task, r in pool.imap_unordered(_point_task, tasks, chunksize=1):  # in completion order
                done(task, r)
                progress()
    else:
        for task in tasks:
            done(task, _point(task))


def remask(targets):
    """Rewrite the guaranteed masks of existing files from region() (no mpmath)."""
    for dist, m in targets:
        path = OUT / f"{dist}_{m}.npz"
        if not path.exists():
            continue
        with np.load(path) as f:
            data = dict(f)
        D = DISTRIBUTIONS[dist]
        xs = tuple(data[f"x{k}"] for k in range(int(data["nin"])))
        v = xs[-1]  # the input sets' own regions (x_sets, p_sets, logp_sets, param_sets)
        if m in PARAM_METHODS:
            own = np.ones(v.size, dtype=bool)
        elif m in ("icdf", "iccdf"):
            own = (v > 0) & (v < 1)
        elif m in ("ilogcdf", "ilogccdf"):
            own = v < 0
        else:
            own = np.isfinite(v)
        data["guaranteed"] = region(dist, D, xs, own, m)
        np.savez_compressed(path, **data)
        print(f"{dist}_{m}: {int(data['guaranteed'].sum())}/{xs[0].size} guaranteed", flush=True)


def main(argv):
    import os

    jobs = os.cpu_count() or 1  # all cores by default (M4 Pro: 10 performance + 4 efficiency)
    masks_only = "--masks-only" in argv
    intervals = "--intervals" in argv
    argv = [a for a in argv if a not in ("--masks-only", "--intervals")]
    if argv and argv[0].startswith("--jobs="):
        jobs, argv = int(argv[0].split("=", 1)[1]), argv[1:]
    targets = []
    for a in argv or list(DISTRIBUTIONS):
        dist, _, ms = a.partition(":")
        base = DISCRETE_INPUT_METHODS if getattr(DISTRIBUTIONS[dist], "discrete", False) else INPUT_METHODS
        methods = ms.split(",") if ms else list(base + PARAM_METHODS)
        targets += [(dist, m) for m in methods]
    if masks_only:
        remask(targets)
    elif intervals:
        add_intervals(targets, jobs)
    else:
        build_all(targets, jobs)


if __name__ == "__main__":
    main(sys.argv[1:])
