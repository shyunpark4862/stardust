"""Reference values for sd.stats' distributions from mpmath (docs/design/stats.md §9).

    .venv/bin/python tools/stats_ref.py                     # every distribution, on all cores
    .venv/bin/python tools/stats_ref.py normal gamma        # only these
    .venv/bin/python tools/stats_ref.py --jobs=8 normal     # 8 processes instead of all cores
    .venv/bin/python tools/stats_ref.py normal:cdf,icdf     # only these methods
    .venv/bin/python tools/stats_ref.py --masks-only gamma  # rewrite the guaranteed masks only

writes tests/stats/ref/<distribution>_<method>.npz in tests/special/refdata.py's format:
x0..x{n-1} are the native function's inputs (params..., then the method's input, if any),
hi + lo the reference as a double-double, `guaranteed` the guaranteed region, `scale` the
bound's multiplier (max(1, kappa) for the inverses, whose contract is
|dx / x| <= 1e-14 max(1, kappa); the T20 mixed bound for logpdf and entropy is applied by
the test, not stored). The generator is seeded: running the tool again gives the same file.

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


DISTRIBUTIONS = {"normal": Normal, "exponential": Exponential, "uniform": Uniform, "laplace": Laplace,
                 "cauchy": Cauchy, "lognormal": LogNormal, "weibull": Weibull, "gamma": Gamma,
                 "chisquared": ChiSquared, "beta": Beta, "studentt": StudentT, "f": F}


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
    if m == "pdf":
        return D.pdf(x, *p)
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
# stay in the files and are held to the documented loose bound.
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


def inputs(D, m, rng):
    if m in PARAM_METHODS:
        return D.param_sets(rng)
    if m in ("icdf", "iccdf"):
        return D.p_sets(rng)
    if m in ("ilogcdf", "ilogccdf"):
        return D.logp_sets(rng)
    return D.x_sets(rng)


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
    keep = np.zeros(n, dtype=bool)
    for i, r in results.items():
        if r is not None:
            hi[i], lo[i], scale[i] = r
            keep[i] = True
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{dist}_{m}.npz", nin=len(xs), hi=hi[keep], lo=lo[keep], scale=scale[keep],
                        guaranteed=np.asarray(guaranteed, dtype=bool)[keep],
                        **{f"x{k}": x[keep] for k, x in enumerate(xs)}, mpmath=mpmath.__version__, dps=sr.DPS)
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
    argv = [a for a in argv if a != "--masks-only"]
    if argv and argv[0].startswith("--jobs="):
        jobs, argv = int(argv[0].split("=", 1)[1]), argv[1:]
    targets = []
    for a in argv or list(DISTRIBUTIONS):
        dist, _, ms = a.partition(":")
        methods = ms.split(",") if ms else list(INPUT_METHODS + PARAM_METHODS)
        targets += [(dist, m) for m in methods]
    if masks_only:
        remask(targets)
    else:
        build_all(targets, jobs)


if __name__ == "__main__":
    main(sys.argv[1:])
