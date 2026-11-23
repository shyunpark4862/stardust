"""Reference values for sd.stats' distributions from Arb (python-flint; docs/design/stats.md §9),
and the multivariate ones (tools/mvref.py, tools/mvcdfref.py).

    .venv/bin/python tools/stats_ref.py                       # every distribution, on 8 processes
    .venv/bin/python tools/stats_ref.py normal gamma          # only these
    .venv/bin/python tools/stats_ref.py --jobs 4 normal       # 4 processes (at most 10)
    .venv/bin/python tools/stats_ref.py normal:cdf,icdf       # only these methods
    .venv/bin/python tools/stats_ref.py --masks-only gamma    # rewrite the guaranteed masks only
    .venv/bin/python tools/stats_ref.py --intervals zipf      # add the discrete inverses' [kmin, kmax]
    .venv/bin/python tools/stats_ref.py --resample zipf       # draw new input sets (else an existing
                                                              # file's inputs are reused, frozen)
    .venv/bin/python tools/stats_ref.py mvnormal bvn          # multivariate families and cdf sets
    .venv/bin/python tools/stats_ref.py hypergeometric_tails betabinomial_tails
                                                              # the long tail sums (tests/stats/ref/tails/;
                                                              # about 25 minutes on one process)

writes tests/stats/ref/<distribution>_<kind>.npz (layout: tests/reffile.py; read by
tests/stats/refdata.py): one input set per distribution and kind -- x (pdf, logpdf, cdf, ...,
pmf), p (icdf, iccdf), logp (ilogcdf, ilogccdf), param (the moments) -- holding the native
function's inputs x0..x{n-1} (params..., then the method's input, if any) and, per method of
that kind, hi + lo the reference as a double-double, `guaranteed` the guaranteed region and
`scale` the bound's multiplier (max(1, kappa) for the inverses, whose contract is
|dx / x| <= 1e-14 max(1, kappa); the mixed bound for logpdf and entropy is applied by the test,
not stored); for the discrete inverses `kmin`, `kmax` the answers the backward error contract
allows (BACKWARD). The generator is seeded; an existing file's inputs are reused (only the
references are recomputed, and a method's points dropped before are tried again) unless
--resample is given. The multivariate families and cdf sets go to tests/stats/ref/mv/. An
unknown name is a usage error (exit status 2); the console output also goes to
build/logs/stats_ref/.

Every reference is an Arb ball that provably contains the true value, settled to 2^-110
relative (tools/arbref.py: no "two precisions agree" heuristic). The definitions are the
textbook formulas, never our code; a sum we cut ourselves (Poisson mixtures, entropy sums)
adds a bound on what it leaves out to the ball. The continuous inverses are roots of
log F(x) = log p certified by a sign change at x (1 -+ 2^-110) (Newton from our own value is
a start only); the discrete ones are exact integers, each comparison of the cdf with its
target decided by a ball that excludes equality, or exactly in rational arithmetic where
equality can happen. A point whose value does not settle by 8192 bits, or whose series,
integral or iteration does not end within its cap, is dropped and recorded (the method's
dropped field); no wall clock decides it, so any machine drops the same points.
"""

import math
import multiprocessing
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import acb, acb_series, arb, arb_series, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import arbref as ar  # noqa: E402
from arbref import NotSettled, add_error, precision, rigorous, to_dd  # noqa: E402
from special_ref import around, grid, loguniform  # noqa: E402,F401  (pure-numpy input helpers)

OUT = ROOT / "tests" / "stats" / "ref"
TOOL = "tools/stats_ref.py"

INPUT_METHODS = ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf")
PARAM_METHODS = ("mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy")
INVERSE = {"icdf", "iccdf", "ilogcdf", "ilogccdf"}

NAN, INF = math.nan, math.inf


# ---------------------------------------------------------------- Arb helpers
#
# Every function below takes and returns arb balls at the current precision (ctx.prec);
# a NaN or an infinity that is the exact answer (an undefined or infinite moment, a density
# pole) is a Python float.


def PI():
    return arb.pi()


def EULER():
    return arb.const_euler()


def LN2():
    return arb.const_log2()


def _a(v):
    return v if isinstance(v, arb) else arb(v)


def _log(v):
    """log v, with log 0 = -inf and the floats inf, nan passed through."""
    if isinstance(v, float):
        return math.log(v) if v > 0 else (-INF if v == 0 else NAN)
    if v.is_zero():
        return -INF
    return v.log()


def _sqrt(v):
    return math.sqrt(v) if isinstance(v, float) else v.sqrt()


def _eps():
    """2^-(prec + 4): a truncation bound below this fraction of a value costs nothing."""
    return arb(2) ** (-ctx.prec - 4)


def _ncdf(z):
    """P(Z <= z)."""
    return (-z / arb(2).sqrt()).erfc() / 2


def _npdf(z):
    return (-z * z / 2).exp() / (2 * PI()).sqrt()


def _nquantile(p):
    """z with P(Z <= z) = p, for p in (0, 1)."""
    return -arb(2).sqrt() * (2 * p).erfcinv()


def _gamma_q(a, x):
    """Q(a, x), the regularized upper incomplete gamma (Arb's own; reliable throughout)."""
    if x.is_zero():
        return arb(1)
    return x.gamma_upper(a, 1)


def _gamma_p(a, x):
    """P(a, x) = 1 - Q(a, x): as 1 - Q where P is not small (Arb's own P loses all its
    digits for large a near the transition, e.g. P(1e4, 1e4) at 128 bits), else by the
    series P = x^a e^-x / Gamma(a + 1) sum_k x^k / ((a + 1)...(a + k)) with its tail bounded:
    once x < a + k + 1 the terms fall by at least r = x / (a + k + 1) each."""
    if x.is_zero():
        return arb(0)
    q = x.gamma_upper(a, 1)
    if x > a + 1 or (q.is_finite() and q < 0.75):
        return 1 - q
    s, t, k, eps = arb(1), arb(1), 0, _eps()
    while True:
        k += 1
        t = t * x / (a + k)
        s += t
        if k % 8 == 0 and x < a + k + 1:
            r = x / (a + k + 1)
            tail = t * r / (1 - r)
            if tail < s * eps:
                break
    s = add_error(s, tail)
    return s * (a * x.log() - x - (a + 1).lgamma()).exp()


def _ibeta(x, a, b, y=None):
    """I_x(a, b), regularized; y = 1 - x (computed by the caller where 1 - x would round).
    Arb's beta_lower returns NaN where the value is 1 to within e^-2000 (I_0.9(0.05, 5e4)):
    there 1 - I_y(b, a), and the narrower of the two where the first is not sharp."""
    if y is None:
        y = 1 - x
    if x.is_zero():
        return arb(0)
    if y.is_zero():
        return arb(1)
    v = x.beta_lower(a, b, 1)
    if v.is_finite() and v.rel_accuracy_bits() > ctx.prec - 32:
        return v
    w = 1 - y.beta_lower(b, a, 1)
    return ar.intersect(v, w)


def _lbeta(a, b):
    return a.lgamma() + b.lgamma() - (a + b).lgamma()


def _int(v):
    """An exact integer arb (or a Python number) as a Python int."""
    if isinstance(v, (int, float)):
        return int(v)
    return int(v.unique_fmpz())


# ---------------------------------------------------------------- the distributions
#
# Each is a class of static Arb functions of arb arguments (x first, then the parameters in
# native order), with `params` and the input sets:
#   param_sets(rng)        -> (tuple of parameter arrays, guaranteed mask)    for the moments
#   x_sets(rng)            -> (params..., x), guaranteed                        for pdf..logccdf
#   p_sets(rng)            -> (params..., p), guaranteed                        for icdf, iccdf
#   logp_sets(rng)         -> (params..., logp), guaranteed                     for ilogcdf, ilogccdf
# cdf and ccdf must both be accurate (the logs take the smaller one; never 1 - F of a small F).


def _p_values(rng, n):
    """Probabilities for icdf and iccdf: uniform, both tails down to 1e-300 and the
    subnormals, and around 1/2."""
    k = n // 8
    p = np.concatenate(
        [
            rng.uniform(0, 1, 2 * k),
            loguniform(rng, 1e-300, 0.5, 2 * k),
            1 - loguniform(rng, 2**-53, 0.5, 2 * k),
            loguniform(rng, 1e-320, 1e-300, k),
            0.5 + rng.uniform(-1e-6, 1e-6, n - 7 * k),
        ]
    )
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
    return np.concatenate(
        [
            rng.uniform(-8, 8, t),
            loguniform(rng, 1e-8, 40, t, signed=True),
            loguniform(rng, 40, far, n - 2 * t, signed=True),
        ]
    )


class Normal(Distribution):
    params = ("mu", "sigma")

    pdf = staticmethod(lambda x, mu, s: _npdf((x - mu) / s) / s)
    cdf = staticmethod(lambda x, mu, s: _ncdf((x - mu) / s))
    ccdf = staticmethod(lambda x, mu, s: _ncdf((mu - x) / s))
    mean = staticmethod(lambda mu, s: mu)
    variance = staticmethod(lambda mu, s: s * s)
    standard_deviation = staticmethod(lambda mu, s: s)
    skewness = staticmethod(lambda mu, s: arb(0))
    kurtosis = staticmethod(lambda mu, s: arb(0))
    median = staticmethod(lambda mu, s: mu)
    mode = staticmethod(lambda mu, s: mu)
    entropy = staticmethod(lambda mu, s: (1 + (2 * PI()).log()) / 2 + s.log())
    icdf_mp = staticmethod(lambda p, mu, s: mu + s * _nquantile(p))
    iccdf_mp = staticmethod(lambda q, mu, s: mu - s * _nquantile(q))

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

    pdf = staticmethod(lambda x, s: (-x / s).exp() / s if x >= 0 else arb(0))
    cdf = staticmethod(lambda x, s: -(-x / s).expm1() if x > 0 else arb(0))
    ccdf = staticmethod(lambda x, s: (-x / s).exp() if x > 0 else arb(1))
    mean = staticmethod(lambda s: s)
    variance = staticmethod(lambda s: s * s)
    standard_deviation = staticmethod(lambda s: s)
    skewness = staticmethod(lambda s: arb(2))
    kurtosis = staticmethod(lambda s: arb(6))
    median = staticmethod(lambda s: s * LN2())
    mode = staticmethod(lambda s: arb(0))
    entropy = staticmethod(lambda s: 1 + s.log())
    icdf_mp = staticmethod(lambda p, s: -s * (-p).log1p())
    iccdf_mp = staticmethod(lambda q, s: -s * q.log())

    @staticmethod
    def _params(rng, n):
        s = np.concatenate(
            [np.ones(n // 4), loguniform(rng, 1e-3, 1e3, n - n // 4 - 20), around([np.exp(-1.0)], 9)[:20]]
        )
        return (s,)

    @staticmethod
    def _x(rng, ps):
        (s,) = ps
        n = s.size
        y = np.concatenate([loguniform(rng, 1e-300, 1e3, n // 2), rng.uniform(0, 50, n - n // 2)])
        return s * y


class Uniform(Distribution):
    params = ("a", "b")

    pdf = staticmethod(lambda x, a, b: 1 / (b - a) if a <= x <= b else arb(0))
    cdf = staticmethod(lambda x, a, b: arb(0) if x <= a else (arb(1) if x >= b else (x - a) / (b - a)))
    ccdf = staticmethod(lambda x, a, b: arb(1) if x <= a else (arb(0) if x >= b else (b - x) / (b - a)))
    mean = staticmethod(lambda a, b: (a + b) / 2)
    variance = staticmethod(lambda a, b: (b - a) ** 2 / 12)
    standard_deviation = staticmethod(lambda a, b: (b - a) / arb(12).sqrt())
    skewness = staticmethod(lambda a, b: arb(0))
    kurtosis = staticmethod(lambda a, b: arb(-6) / 5)
    median = staticmethod(lambda a, b: (a + b) / 2)
    mode = staticmethod(lambda a, b: NAN)
    entropy = staticmethod(lambda a, b: (b - a).log())
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
        return (-abs(x - mu) / s).exp() / (2 * s)

    @staticmethod
    def cdf(x, mu, s):
        z = (x - mu) / s
        return z.exp() / 2 if x < mu else 1 - (-z).exp() / 2

    @staticmethod
    def ccdf(x, mu, s):
        z = (x - mu) / s
        return (-z).exp() / 2 if x > mu else 1 - z.exp() / 2

    mean = staticmethod(lambda mu, s: mu)
    variance = staticmethod(lambda mu, s: 2 * s * s)
    standard_deviation = staticmethod(lambda mu, s: arb(2).sqrt() * s)
    skewness = staticmethod(lambda mu, s: arb(0))
    kurtosis = staticmethod(lambda mu, s: arb(3))
    median = staticmethod(lambda mu, s: mu)
    mode = staticmethod(lambda mu, s: mu)
    entropy = staticmethod(lambda mu, s: 1 + (2 * s).log())
    icdf_mp = staticmethod(lambda p, mu, s: mu + s * (2 * p).log() if p <= 0.5 else mu - s * (2 * (1 - p)).log())
    iccdf_mp = staticmethod(lambda q, mu, s: mu - s * (2 * q).log() if q <= 0.5 else mu + s * (2 * (1 - q)).log())

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
        return 1 / (PI() * s * (1 + z * z))

    @staticmethod
    def cdf(x, mu, s):
        return arb.atan2(arb(1), -(x - mu) / s) / PI()

    @staticmethod
    def ccdf(x, mu, s):
        return arb.atan2(arb(1), (x - mu) / s) / PI()

    mean = variance = standard_deviation = skewness = kurtosis = staticmethod(lambda mu, s: NAN)
    median = staticmethod(lambda mu, s: mu)
    mode = staticmethod(lambda mu, s: mu)
    entropy = staticmethod(lambda mu, s: (4 * PI() * s).log())
    icdf_mp = staticmethod(lambda p, mu, s: mu - s / (PI() * p).tan())
    iccdf_mp = staticmethod(lambda q, mu, s: mu + s / (PI() * q).tan())

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

    pdf = staticmethod(lambda x, mu, s: _npdf((x.log() - mu) / s) / (s * x) if x > 0 else arb(0))
    cdf = staticmethod(lambda x, mu, s: _ncdf((x.log() - mu) / s) if x > 0 else arb(0))
    ccdf = staticmethod(lambda x, mu, s: _ncdf((mu - x.log()) / s) if x > 0 else arb(1))
    mean = staticmethod(lambda mu, s: (mu + s * s / 2).exp())
    variance = staticmethod(lambda mu, s: (s * s).expm1() * (2 * mu + s * s).exp())
    standard_deviation = staticmethod(lambda mu, s: ((s * s).expm1() * (2 * mu + s * s).exp()).sqrt())
    skewness = staticmethod(lambda mu, s: ((s * s).exp() + 2) * (s * s).expm1().sqrt())
    kurtosis = staticmethod(lambda mu, s: (4 * s * s).exp() + 2 * (3 * s * s).exp() + 3 * (2 * s * s).exp() - 6)
    median = staticmethod(lambda mu, s: mu.exp())
    mode = staticmethod(lambda mu, s: (mu - s * s).exp())
    entropy = staticmethod(lambda mu, s: mu + arb(0.5) + (s * (2 * PI()).sqrt()).log())
    icdf_mp = staticmethod(lambda p, mu, s: (mu + s * _nquantile(p)).exp())
    iccdf_mp = staticmethod(lambda q, mu, s: (mu - s * _nquantile(q)).exp())

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
    return (1 + arb(k) / c).gamma()


class Weibull(Distribution):
    params = ("c", "scale")
    positive = True

    @staticmethod
    def pdf(x, c, s):
        if x < 0:
            return arb(0)
        if x == 0:  # the density's limit at 0
            return INF if c < 1 else (1 / s if c == 1 else arb(0))
        y = x / s
        return c / s * ((c - 1) * y.log() - (c * y.log()).exp()).exp()

    cdf = staticmethod(lambda x, c, s: -(-((c * (x / s).log()).exp())).expm1() if x > 0 else arb(0))
    ccdf = staticmethod(lambda x, c, s: (-((c * (x / s).log()).exp())).exp() if x > 0 else arb(1))
    mean = staticmethod(lambda c, s: s * _weibull_gk(c, 1))
    variance = staticmethod(lambda c, s: s * s * (_weibull_gk(c, 2) - _weibull_gk(c, 1) ** 2))
    standard_deviation = staticmethod(lambda c, s: s * (_weibull_gk(c, 2) - _weibull_gk(c, 1) ** 2).sqrt())

    @staticmethod
    def skewness(c, s):
        g1, g2, g3 = (_weibull_gk(c, k) for k in (1, 2, 3))
        return (g3 - 3 * g1 * g2 + 2 * g1**3) / (g2 - g1**2) ** arb(1.5)

    @staticmethod
    def kurtosis(c, s):
        g1, g2, g3, g4 = (_weibull_gk(c, k) for k in (1, 2, 3, 4))
        return (g4 - 4 * g1 * g3 + 6 * g1**2 * g2 - 3 * g1**4) / (g2 - g1**2) ** 2 - 3

    median = staticmethod(lambda c, s: s * (LN2().log() / c).exp())
    mode = staticmethod(lambda c, s: s * (((c - 1) / c).log() / c).exp() if c > 1 else arb(0))
    entropy = staticmethod(lambda c, s: EULER() * (1 - 1 / c) + (s / c).log() + 1)
    icdf_mp = staticmethod(lambda p, c, s: s * ((-(-p).log1p()).log() / c).exp())
    iccdf_mp = staticmethod(lambda q, c, s: s * ((-q.log()).log() / c).exp())

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
        zeros = np.concatenate(
            [around([3.6023494], 6), around([2.2520], 6), around([5.7734], 6)]
        )  # near the skewness and kurtosis zeros
        return (
            np.concatenate([c, zeros, loguniform(rng, 1e3, 1e8, 50)]),
            np.concatenate([s, np.ones(zeros.size + 50)]),
        ), np.ones(c.size + zeros.size + 50, dtype=bool)


# ---- extreme value and lifetime (docs/design/stats.md)


class Logistic(Distribution):
    params = ("mu", "sigma")

    @staticmethod
    def pdf(x, mu, s):
        e = (-abs((x - mu) / s)).exp()
        return e / (s * (1 + e) ** 2)

    cdf = staticmethod(lambda x, mu, s: 1 / (1 + (-(x - mu) / s).exp()))
    ccdf = staticmethod(lambda x, mu, s: 1 / (1 + ((x - mu) / s).exp()))
    mean = staticmethod(lambda mu, s: mu)
    variance = staticmethod(lambda mu, s: s * s * PI() ** 2 / 3)
    standard_deviation = staticmethod(lambda mu, s: s * PI() / arb(3).sqrt())
    skewness = staticmethod(lambda mu, s: arb(0))
    kurtosis = staticmethod(lambda mu, s: arb(6) / 5)
    median = staticmethod(lambda mu, s: mu)
    mode = staticmethod(lambda mu, s: mu)
    entropy = staticmethod(lambda mu, s: s.log() + 2)
    icdf_mp = staticmethod(lambda p, mu, s: mu + s * (p.log() - (-p).log1p()))
    iccdf_mp = staticmethod(lambda q, mu, s: mu - s * (q.log() - (-q).log1p()))

    @staticmethod
    def _params(rng, n):
        return _location_scale(rng, n)

    @staticmethod
    def _x(rng, ps):
        mu, sigma = ps
        return mu + sigma * _z(rng, mu.size, 1e5)


class Gumbel(Distribution):
    params = ("mu", "sigma")

    @staticmethod
    def pdf(x, mu, s):
        z = (x - mu) / s
        return (-z - (-z).exp()).exp() / s

    cdf = staticmethod(lambda x, mu, s: (-(-(x - mu) / s).exp()).exp())
    ccdf = staticmethod(lambda x, mu, s: -(-(-(x - mu) / s).exp()).expm1())
    mean = staticmethod(lambda mu, s: mu + EULER() * s)
    variance = staticmethod(lambda mu, s: s * s * PI() ** 2 / 6)
    standard_deviation = staticmethod(lambda mu, s: s * PI() / arb(6).sqrt())
    skewness = staticmethod(lambda mu, s: 12 * arb(6).sqrt() * arb(3).zeta() / PI() ** 3)
    kurtosis = staticmethod(lambda mu, s: arb(12) / 5)
    median = staticmethod(lambda mu, s: mu - s * LN2().log())
    mode = staticmethod(lambda mu, s: mu)
    entropy = staticmethod(lambda mu, s: s.log() + EULER() + 1)
    icdf_mp = staticmethod(lambda p, mu, s: mu - s * (-p.log()).log())
    iccdf_mp = staticmethod(lambda q, mu, s: mu - s * (-(-q).log1p()).log())

    @staticmethod
    def _params(rng, n):
        return _location_scale(rng, n)

    @staticmethod
    def _x(rng, ps):
        mu, sigma = ps
        return mu + sigma * _z(rng, mu.size, 700)


def _gev_g(c, k):
    return (1 + k * c).gamma()


class GEV(Distribution):
    """scipy's genextreme: P(X <= x) = exp(-(1 - c z)^(1/c)), the Gumbel at c = 0."""

    params = ("c", "mu", "sigma")

    @staticmethod
    def _t(x, c, mu, s):
        """t = (1 - c z)^(1/c), or None outside the support."""
        z = (x - mu) / s
        if c == 0:
            return (-z).exp()
        y = 1 - c * z
        if not y > 0:
            if y <= 0:
                return None
            raise NotSettled("GEV: support edge")
        return (y.log() / c).exp()

    @staticmethod
    def pdf(x, c, mu, s):
        t = GEV._t(x, c, mu, s)
        if t is None:
            return arb(0)
        return ((1 - c) * t.log() - t).exp() / s  # (1 - c z)^(1/c - 1) = t^(1 - c)

    @staticmethod
    def cdf(x, c, mu, s):
        t = GEV._t(x, c, mu, s)
        if t is None:
            return arb(1) if c > 0 else arb(0)
        return (-t).exp()

    @staticmethod
    def ccdf(x, c, mu, s):
        t = GEV._t(x, c, mu, s)
        if t is None:
            return arb(0) if c > 0 else arb(1)
        return -(-t).expm1()

    @staticmethod
    def mean(c, mu, s):
        if c == 0:
            return mu + s * EULER()
        return mu + s * (1 - _gev_g(c, 1)) / c if c > -1 else INF

    @staticmethod
    def variance(c, mu, s):
        if c == 0:
            return s * s * PI() ** 2 / 6
        if not 2 * c > -1:
            return INF if c > -1 else NAN
        return s * s * (_gev_g(c, 2) - _gev_g(c, 1) ** 2) / c**2

    @staticmethod
    def standard_deviation(c, mu, s):
        v = GEV.variance(c, mu, s)
        return _sqrt(v)

    @staticmethod
    def skewness(c, mu, s):
        if c == 0:
            return Gumbel.skewness(mu, s)
        if not 3 * c > -1:
            return INF if 2 * c > -1 else NAN
        g1, g2, g3 = (_gev_g(c, k) for k in (1, 2, 3))
        return -c.sgn() * (g3 - 3 * g1 * g2 + 2 * g1**3) / (g2 - g1**2) ** arb(1.5)

    @staticmethod
    def kurtosis(c, mu, s):
        if c == 0:
            return arb(12) / 5
        if not 4 * c > -1:
            return INF if 2 * c > -1 else NAN
        g1, g2, g3, g4 = (_gev_g(c, k) for k in (1, 2, 3, 4))
        return (g4 - 4 * g1 * g3 + 6 * g1**2 * g2 - 3 * g1**4) / (g2 - g1**2) ** 2 - 3

    @staticmethod
    def _z_of_t(t, c):
        return -t.log() if c == 0 else -(c * t.log()).expm1() / c

    median = staticmethod(lambda c, mu, s: mu + s * GEV._z_of_t(LN2(), c))
    mode = staticmethod(lambda c, mu, s: mu + s / c if c >= 1 else mu + s * GEV._z_of_t(1 - c, c))
    entropy = staticmethod(lambda c, mu, s: s.log() + EULER() * (1 - c) + 1)
    icdf_mp = staticmethod(lambda p, c, mu, s: mu + s * GEV._z_of_t(-p.log(), c))
    iccdf_mp = staticmethod(lambda q, c, mu, s: mu + s * GEV._z_of_t(-(-q).log1p(), c))

    @staticmethod
    def _params(rng, n):
        k = n // 6
        c = np.concatenate(
            [
                np.zeros(k // 2),
                loguniform(rng, 1e-12, 1e-3, k, signed=True),
                rng.uniform(-2, 2, 2 * k),
                loguniform(rng, 1e-3, 20, n - 3 * k - k // 2, signed=True),
            ]
        )
        rng.shuffle(c)
        mu, sigma = _location_scale(rng, n)
        return c, mu, sigma

    @staticmethod
    def _x(rng, ps):
        c, mu, sigma = ps
        n = c.size
        t = loguniform(rng, 1e-300, 700, n)
        with np.errstate(all="ignore"):
            z = np.where(c == 0, -np.log(t), -np.expm1(c * np.log(t)) / np.where(c == 0, 1.0, c))
        z[: n // 20] = rng.uniform(-50, 50, n // 20)  # some beyond the support's end
        return mu + sigma * z

    @classmethod
    def param_sets(cls, rng):
        (c, mu, s), g = super().param_sets(rng)
        # near the zeros of the skewness (c = 0.2776) and of the excess kurtosis (0.1732, 0.4440):
        # W = Y^c is the Weibull of shape 1/c
        zeros = np.concatenate([around([1 / 3.6023494], 6), around([1 / 5.7734], 6), around([1 / 2.2520], 6)])
        k = zeros.size
        return (
            np.concatenate([c, zeros]),
            np.concatenate([mu, np.zeros(k)]),
            np.concatenate([s, np.ones(k)]),
        ), np.ones(c.size + k, dtype=bool)


class Pareto(Distribution):
    params = ("b", "scale")
    positive = True

    pdf = staticmethod(lambda x, b, s: b / s * (-(b + 1) * (x / s).log()).exp() if x >= s else arb(0))
    cdf = staticmethod(lambda x, b, s: -(-b * (x / s).log()).expm1() if x > s else arb(0))
    ccdf = staticmethod(lambda x, b, s: (-b * (x / s).log()).exp() if x > s else arb(1))
    mean = staticmethod(lambda b, s: s * b / (b - 1) if b > 1 else INF)
    variance = staticmethod(lambda b, s: s * s * b / ((b - 1) ** 2 * (b - 2)) if b > 2 else (INF if b > 1 else NAN))
    standard_deviation = staticmethod(lambda b, s: _sqrt(Pareto.variance(b, s)) if b > 1 else NAN)
    skewness = staticmethod(
        lambda b, s: 2 * (1 + b) / (b - 3) * ((b - 2) / b).sqrt() if b > 3 else (INF if b > 2 else NAN)
    )
    kurtosis = staticmethod(
        lambda b, s: 6 * (b**3 + b**2 - 6 * b - 2) / (b * (b - 3) * (b - 4)) if b > 4 else (INF if b > 2 else NAN)
    )
    median = staticmethod(lambda b, s: s * (LN2() / b).exp())
    mode = staticmethod(lambda b, s: s)
    entropy = staticmethod(lambda b, s: (s / b).log() + 1 / b + 1)
    icdf_mp = staticmethod(lambda p, b, s: s * (-(-p).log1p() / b).exp())
    iccdf_mp = staticmethod(lambda q, b, s: s * (-q.log() / b).exp())

    @staticmethod
    def _params(rng, n):
        q = n // 4
        b = np.concatenate([np.ones(q // 2), loguniform(rng, 0.05, 100, n - q // 2)])
        s = np.concatenate([np.ones(q), loguniform(rng, 1e-3, 1e3, n - q)])
        rng.shuffle(b)
        return b, s

    @staticmethod
    def _x(rng, ps):
        b, s = ps
        L = loguniform(rng, 1e-300, 700, b.size)
        x = s * np.exp(np.minimum(L / b, 700.0))
        x[: b.size // 40] = s[: b.size // 40] * rng.uniform(0.1, 1.0, b.size // 40)  # below the support
        return x


class Rayleigh(Distribution):
    params = ("scale",)
    positive = True

    pdf = staticmethod(lambda x, s: x / s**2 * (-(x**2) / (2 * s**2)).exp() if x > 0 else arb(0))
    cdf = staticmethod(lambda x, s: -(-(x**2) / (2 * s**2)).expm1() if x > 0 else arb(0))
    ccdf = staticmethod(lambda x, s: (-(x**2) / (2 * s**2)).exp() if x > 0 else arb(1))
    mean = staticmethod(lambda s: s * (PI() / 2).sqrt())
    variance = staticmethod(lambda s: s * s * (4 - PI()) / 2)
    standard_deviation = staticmethod(lambda s: s * ((4 - PI()) / 2).sqrt())
    skewness = staticmethod(lambda s: 2 * PI().sqrt() * (PI() - 3) / (4 - PI()) ** arb(1.5))
    kurtosis = staticmethod(lambda s: -(6 * PI() ** 2 - 24 * PI() + 16) / (4 - PI()) ** 2)
    median = staticmethod(lambda s: s * (2 * LN2()).sqrt())
    mode = staticmethod(lambda s: s)
    entropy = staticmethod(lambda s: 1 + (s / arb(2).sqrt()).log() + EULER() / 2)
    icdf_mp = staticmethod(lambda p, s: s * (-2 * (-p).log1p()).sqrt())
    iccdf_mp = staticmethod(lambda q, s: s * (-2 * q.log()).sqrt())

    @staticmethod
    def _params(rng, n):
        q = n // 4
        s = np.concatenate([np.ones(q), loguniform(rng, 1e-3, 1e3, n - q)])
        rng.shuffle(s)
        return (s,)

    @staticmethod
    def _x(rng, ps):
        (s,) = ps
        t = loguniform(rng, 1e-300, 1e5, s.size)
        return s * np.sqrt(2 * t)


class InverseGamma(Distribution):
    start = staticmethod(lambda a, s: s / a)
    """X = scale / Y, Y ~ gamma(a, 1)."""

    params = ("a", "scale")
    positive = True

    @staticmethod
    def pdf(x, a, s):
        if x <= 0:
            return arb(0)
        return (a * s.log() - (a + 1) * x.log() - s / x - a.lgamma()).exp()

    cdf = staticmethod(lambda x, a, s: _gamma_q(a, s / x) if x > 0 else arb(0))
    ccdf = staticmethod(lambda x, a, s: _gamma_p(a, s / x) if x > 0 else arb(1))
    mean = staticmethod(lambda a, s: s / (a - 1) if a > 1 else INF)
    variance = staticmethod(lambda a, s: s * s / ((a - 1) ** 2 * (a - 2)) if a > 2 else (INF if a > 1 else NAN))
    standard_deviation = staticmethod(lambda a, s: _sqrt(InverseGamma.variance(a, s)) if a > 1 else NAN)
    skewness = staticmethod(lambda a, s: 4 * (a - 2).sqrt() / (a - 3) if a > 3 else (INF if a > 2 else NAN))
    kurtosis = staticmethod(lambda a, s: 6 * (5 * a - 11) / ((a - 3) * (a - 4)) if a > 4 else (INF if a > 2 else NAN))
    mode = staticmethod(lambda a, s: s / (a + 1))
    entropy = staticmethod(lambda a, s: a + s.log() + a.lgamma() - (1 + a) * a.digamma())

    @staticmethod
    def _params(rng, n):
        return Gamma._params(rng, n)

    @staticmethod
    def _x(rng, ps):
        a, s = ps
        y = Gamma._x(rng, (a, np.ones(a.size)))
        with np.errstate(divide="ignore"):
            return s / y


# ---- noncentral
#
# The cdfs are Poisson mixtures of incomplete gamma or beta functions, both tails summed
# directly as sums of positive terms (1 - the other tail would lose a tiny value), each
# family of incomplete functions by its stable recurrence: the lower functions (decreasing
# in the shape) backward from J past the weights' bulk, the upper ones (increasing)
# forward from j0 before it. What the cut leaves out is bounded and added to the ball:
# the values lie in [0, 1] and are monotone in j, and the weights' tails are geometric.


def _own_quantiles(cls_name, params, p):
    """x = icdf(p) by sd.stats itself (scipy's ppf stalls in the far tails): input points only."""
    import stardust as sd

    X = getattr(sd.stats, cls_name)(**{k: sd.asarray(v) for k, v in params.items()})
    return np.asarray(X.icdf(sd.asarray(p)))


def _nc_probs(rng, n):
    k = n // 4
    return np.concatenate(
        [rng.uniform(0, 1, 2 * k), loguniform(rng, 1e-300, 0.5, k), 1 - loguniform(rng, 1e-16, 0.5, n - 3 * k)]
    )


class _Weights:
    """Mixture weights w_j (j >= 0) with w_{j+1} / w_j = mu / (j + c) (c = 1: Poisson,
    e^-mu mu^j / j!; c = 3/2: mu^j / Gamma(j + 3/2), the noncentral t's odd family), and
    bounds on their sums below j0 and above J (geometric: the ratio is below
    (j0 - 2 + c) / mu going down and mu / (J + 1 + c) going up)."""

    def __init__(self, mu, c):
        self.mu, self.c = mu, arb(c)

    def at(self, j):
        if self.c == 1:
            return (-self.mu + j * self.mu.log() - arb(j + 1).lgamma()).exp()
        return (j * self.mu.log() - (j + self.c).lgamma()).exp()

    def step(self, j):
        """w_{j+1} / w_j."""
        return self.mu / (j + self.c)

    def tails(self, j0, J):
        mu, c = self.mu, self.c
        left = arb(0)
        if j0 > 0:
            r = (j0 - 2 + c) / mu  # w_{j-1} / w_j = (j - 1 + c) / mu <= r for j <= j0 - 1
            if not r < 1:
                return None
            left = self.at(j0 - 1) / (1 - r)
        r = mu / (J + 1 + c)
        if not r < 1:
            return None
        return left.upper(), (self.at(J + 1) / (1 - r)).upper()


def _mixture(W, mf, values, lower, vratio=None):
    """sum_j w_j V_j, V_j in [0, 1] decreasing in j (lower) or increasing; values(j0, J)
    yields V_j in its stable order (lower: j = J down to j0; upper: j0 up to J), streamed
    (no lists: a deep tail sums a million terms). The range [j0, J] around the weights' bulk
    widens until the bounds on the rest, (sum_{j < j0} w_j) (1 or V_j0) +
    (sum_{j > J} w_j) (V_J or 1), are below the sum's last bits. For an increasing family
    vratio(j) >= V_{j+1} / V_j (non-increasing in j) gives the sharper end
    w_J V_J r / (1 - r), r = (w_{J+1} / w_J) vratio(J) < 1: the terms fall geometrically
    from J on (deep upper tails, where V_J is tiny)."""
    width = 10 * math.sqrt(mf) + 10
    while True:
        j0, J = max(0, math.floor(mf - width)), math.ceil(mf + width)
        if J > 10**6:
            raise NotSettled("mixture: too many terms")
        tails = W.tails(j0, J)
        if tails is None:
            width *= 2
            continue
        S = arb(0)
        if lower:
            w = W.at(J)
            for i, v in enumerate(values(j0, J)):  # j = J - i
                if i:
                    w = w / W.step(J - i)
                else:
                    vJ = v
                S += w * v
            vj0 = v
        else:
            w = W.at(j0)
            for i, v in enumerate(values(j0, J)):  # j = j0 + i
                if i:
                    w = w * W.step(j0 + i - 1)
                else:
                    vj0 = v
                S += w * v
            vJ, wJ = v, w
        left, right = tails
        if not lower and vratio is not None:
            r = W.step(J) * vratio(J)
            if r < 1:
                alt = (wJ * vJ * r / (1 - r)).upper()
                right = alt if alt < right else right
        err = left + right * vJ if lower else left * vj0 + right
        if err < abs(S) * _eps():
            return add_error(S, err)
        if j0 == 0 and width > 1e7:
            raise NotSettled("mixture")
        width *= 2


def _lower_values(first, T, down):
    """L_J, L_(J-1), ..., L_j0 of a decreasing family with L_{j-1} = L_j + T_{j-1}:
    first(J) = L_J, T(J - 1) direct, T_{m-1} = T_m down(m)."""

    def values(j0, J):
        L = first(J)
        yield L
        t = T(J - 1) if J > j0 else None
        for j in range(J, j0, -1):
            L = L + t
            yield L
            if j - 1 > j0:
                t = t * down(j - 1)

    return values


def _upper_values(first, T, up):
    """U_j0, ..., U_J of an increasing family with U_{j+1} = U_j + T_j: first(j0) = U_j0,
    T(j0) direct, T_{j+1} = T_j up(j)."""

    def values(j0, J):
        U = first(j0)
        yield U
        t = T(j0)
        for j in range(j0, J):
            U = U + t
            yield U
            t = t * up(j)

    return values


def _ncx2_pair(df, nc, x, upper):
    """P(X <= x) (or P(X > x) when upper): sum_j Poisson(nc/2)_j P or Q(df/2 + j, x/2)."""
    a, y, mu = df / 2, x / 2, nc / 2
    T = lambda j: ((a + j) * y.log() - y - (a + j + 1).lgamma()).exp()  # P(a+j) - P(a+j+1)
    W = _Weights(mu, 1)
    if upper:
        # deep in the upper tail the terms w_j Q(a+j, y) peak near j (a + j) = mu y: past
        # 3e5 terms the point is given up at once (every precision would time out on it)
        jstar = math.sqrt(_fl(mu) * _fl(y))
        if jstar + 10 * math.sqrt(jstar) > 3e5:
            raise NotSettled("ncx2: too deep in the upper tail")
        vals = _upper_values(lambda j: _gamma_q(a + j, y), T, lambda j: y / (a + j + 1))
    else:
        vals = _lower_values(lambda J: _gamma_p(a + J, y), T, lambda m: (a + m) / y)
    # Q(s+1, y) / Q(s, y) = 1 + T(s) / Q(s, y) <= 1 + y / s, as Q(s, y) >= y^(s-1) e^-y / Gamma(s)
    return _mixture(W, _fl(mu), vals, not upper, vratio=lambda j: 1 + y / (a + j))


def _ncf_pair(dfn, dfd, nc, f, upper):
    """P(X <= f) (or the upper tail): sum_j Poisson(nc/2)_j I_w(dfn/2 + j, dfd/2), or I_v(dfd/2, dfn/2 + j)."""
    a, b, mu = dfn / 2, dfd / 2, nc / 2
    s = dfn * f + dfd
    w, v = dfn * f / s, dfd / s
    lw, lv = w.log(), v.log()
    T = lambda j: ((a + j) * lw + b * lv - (a + j).log() - _lbeta(a + j, b)).exp()  # I(a+j) - I(a+j+1)
    W = _Weights(mu, 1)
    if upper:
        vals = _upper_values(lambda j: _ibeta(v, b, a + j, w), T, lambda j: w * (a + j + b) / (a + j + 1))
    else:
        vals = _lower_values(lambda J: _ibeta(w, a + J, b, v), T, lambda m: (a + m) / (w * (a + m - 1 + b)))
    # I_v(b, c+1) / I_v(b, c) = 1 + T(c) / I_v(b, c) <= 1 + b / c: I_v(b, c) is at least the first
    # term v^b w^c / (b B(b, c)) of its series
    return _mixture(W, _fl(mu), vals, not upper, vratio=lambda j: 1 + b / (a + j))


def _ncx2_logpdf(df, nc, x):
    """log of e^(-(x+nc)/2) (x/nc)^(df/4-1/2) I_(df/2-1)(sqrt(nc x)) / 2 (Arb's scaled Bessel I)."""
    z = (nc * x).sqrt()
    return -LN2() - (x + nc) / 2 + (df / 4 - arb(0.5)) * (x / nc).log() + z.bessel_i(df / 2 - 1, scaled=True).log() + z


def _hyp1f1(al, be, z):
    """1F1(al; be; z) for al, be > 0 (z an arb or acb): Arb's, or where that is not finite
    (large al, be and z: NaN) the series sum_k (al)_k / (be)_k z^k / k!, its tail past term
    k bounded geometrically: every later ratio (al + j) |z| / ((be + j)(j + 1)) is at most
    R = max(1, (al + k) / (be + k)) |z| / (k + 1)."""
    if isinstance(z, arb):
        v = z.hypgeom_1f1(al, be)
        if v.is_finite() and v.rel_accuracy_bits() > ctx.prec - 32:
            return v
    if not z.is_finite():
        return z * 0 + arb(NAN)
    s, t, k = z * 0 + 1, z * 0 + 1, 0
    eps, az = _eps(), abs(z)
    k_conv = None
    while True:
        t = t * (al + k) * z / ((be + k) * (k + 1))
        k += 1
        s += t
        if k % 8 == 0:
            q = (al + k) / (be + k)
            R = (q if q > 1 else arb(1)) * az / (k + 1)
            if R < 0.5:
                k_conv = k_conv or k
                tail = abs(t) * R / (1 - R)
                # below the last bits, or (a wide box: |s| may hold 0) a valid, wider enclosure
                if tail < abs(s) * eps or k > 2 * k_conv + 256:
                    break
        if k > 10**6:
            raise NotSettled("1F1 series")
    if isinstance(s, arb):
        return add_error(s, tail)
    return s + acb(arb(0, tail.upper()), arb(0, tail.upper()))


def _ncf_logpdf(dfn, dfd, nc, x):
    """log of e^(-nc/2) (central F density) 1F1((dfn+dfd)/2; dfn/2; nc dfn x / (2 (dfn x + dfd)))."""
    a, b = dfn / 2, dfd / 2
    central = a * (dfn / dfd).log() + (a - 1) * x.log() - (a + b) * (dfn * x / dfd).log1p() - _lbeta(a, b)
    return -nc / 2 + central + _hyp1f1(a + b, a, nc * dfn * x / (2 * (dfn * x + dfd))).log()


class NoncentralChiSquared(Distribution):
    start = staticmethod(lambda df, nc: df + nc)  # a Newton start where our root is 0 or inf
    # Chernoff at lambda = 1/4: P(X > x) <= e^(-x/4) E e^(X/4) = e^(-x/4) 2^(df/2) e^(nc/2)
    log_ccdf_bound = staticmethod(lambda x, df, nc: -x / 4 + df / 2 * LN2() + nc / 2)
    params = ("df", "nc")
    positive = True

    @staticmethod
    def logpdf(x, df, nc):
        if nc == 0:
            return _log(_gamma_pdf(x, df / 2, arb(2)))
        if not x > 0:
            return -INF if (x < 0 or df > 2) else (INF if df < 2 else -LN2() - nc / 2)
        return _ncx2_logpdf(df, nc, x)

    @staticmethod
    def pdf(x, df, nc):
        if nc == 0:
            return _gamma_pdf(x, df / 2, arb(2))
        if not x > 0:
            return arb(0) if (x < 0 or df > 2) else (INF if df < 2 else (-nc / 2).exp() / 2)
        return _ncx2_logpdf(df, nc, x).exp()

    cdf = staticmethod(
        lambda x, df, nc: (ChiSquared.cdf(x, df) if nc == 0 else _ncx2_pair(df, nc, x, False)) if x > 0 else arb(0)
    )
    ccdf = staticmethod(
        lambda x, df, nc: (ChiSquared.ccdf(x, df) if nc == 0 else _ncx2_pair(df, nc, x, True)) if x > 0 else arb(1)
    )
    mean = staticmethod(lambda df, nc: df + nc)
    variance = staticmethod(lambda df, nc: 2 * (df + 2 * nc))
    standard_deviation = staticmethod(lambda df, nc: (2 * (df + 2 * nc)).sqrt())
    skewness = staticmethod(lambda df, nc: arb(8).sqrt() * (df + 3 * nc) / (df + 2 * nc) ** arb(1.5))
    kurtosis = staticmethod(lambda df, nc: 12 * (df + 4 * nc) / (df + 2 * nc) ** 2)

    @staticmethod
    def mode(df, nc):
        """f' = (f_(df-2) - f_df) / 2: the root of log f_(df-2) - log f_df, which is
        -log(x/nc)/2 + log I_(v-1)(z) - log I_v(z), z = sqrt(nc x), v = df/2 - 1 (0 for
        df <= 2, where the density falls from 0)."""
        if nc == 0:
            return ChiSquared.mode(df)
        if not df > 2:
            return arb(0)
        v = df / 2 - 1

        def g(x):
            z = (nc * x).sqrt()
            return -(x / nc).log() / 2 + z.bessel_i(v - 1, scaled=True).log() - z.bessel_i(v, scaled=True).log()

        x0 = _our("noncentralchisquared", "mode", (_fl(df), _fl(nc)))
        return _solve(g, x0 if x0 > 0 else _fl(df + nc - 2), increasing=False, positive=True)

    @staticmethod
    def entropy(df, nc):
        if nc == 0:
            return ChiSquared.entropy(df)
        return _ncx2_entropy(df, nc)

    @staticmethod
    def _params(rng, n):
        k = n // 3
        df = np.concatenate(
            [loguniform(rng, 1e-2, 1e4, k), loguniform(rng, 1e-2, 10, k), loguniform(rng, 1e-2, 1e4, n - 2 * k)]
        )
        nc = np.concatenate(
            [loguniform(rng, 1e-3, 1e4, k), loguniform(rng, 1e-3, 50, k), loguniform(rng, 1e-3, 1e4, n - 2 * k)]
        )
        nc[: n // 50] = 0.0
        rng.shuffle(nc)
        return df, nc

    @staticmethod
    def _x(rng, ps):
        df, nc = ps
        return _own_quantiles("NoncentralChiSquared", {"df": df, "nc": nc}, _nc_probs(rng, df.size))

    @classmethod
    def param_sets(cls, rng):
        ps = cls._params(rng, 600)
        return ps, np.ones(ps[0].size, dtype=bool)


def _solve(g, x0, increasing, positive):
    """A root of g near x0 (Newton on the midpoints with a difference quotient, in log x
    for a positive root), certified to 2^-110 relative by the sign change of g
    (arbref.certify_root), at doubling precision."""
    prec = ar.START_PREC
    u = arb(x0).log() if positive else arb(x0)
    while prec <= ar.MAX_PREC:
        with precision(prec):
            try:
                h = arb(2) ** (-prec // 3)
                for _ in range(100):
                    x = (u.exp() if positive else u).mid()
                    x2 = ((u + h).exp() if positive else u + h).mid()
                    gx, g2 = g(x).mid(), g(x2).mid()
                    d = (g2 - gx) / h
                    if not (gx.is_finite() and d.is_finite()) or d.is_zero():
                        raise NotSettled("solve: no slope")
                    du = gx / d
                    u = (u - du).mid()
                    if abs(du) <= arb(2) ** (-prec + 16) * max(arb(1), abs(u)):
                        break
                x = (u.exp() if positive else u).mid()
                return ar.certify_root(g, x, 2.0**-ar.TARGET_BITS, increasing)
            except NotSettled, ZeroDivisionError, ValueError, ArithmeticError:
                pass
        prec *= 2
    raise NotSettled("root")


# ---- the noncentral t: T = (Z + d) / S, S = sqrt(V / df), V ~ chi-square(df)
#
# Its density and its tail away from d are integrals over S of positive integrands, taken
# in v = log s (where they are smooth and fall off on both sides) by acb.integral, which
# bounds its own error; the two ends cut off are bounded in closed form.


def _xexp(v):
    """exp v for an integrand's box: Arb's exp of a ball of radius r is e^m (1 +- (e^r - 1)),
    which takes in negative values (and so 1 + e^v in 0) for r > log 2; for a wide real part
    this encloses e^v by its values at the real part's ends instead."""
    if isinstance(v, acb):
        re, im = v.real, v.imag
        if re.rad() > 0.25:
            e = acb(arb(re.lower()).exp().union(arb(re.upper()).exp()))
            return e if im.is_zero() else e * acb(0, im).exp()
        return v.exp()
    if v.rad() > 0.25:
        return arb(v.lower()).exp().union(arb(v.upper()).exp())
    return v.exp()


def _integral(fn, a, b, analytic=False, **kw):
    """acb.integral of fn(v) over [a, b]; with analytic, fn(v, an) takes the integrator's
    flag (to pass to the logs it takes: the integrand must check its own analyticity), else
    fn(v) is entire. The integrator's own caps (its evaluation and depth limits) bound the
    work; a box the integrand cannot bound gives NaN there (the integrator subdivides). An
    error that is not about the box stops the integral and is raised after."""
    failed = []

    def cb(v, an):
        if failed:
            return acb(NAN)
        try:
            if v.rad() > 2.0**-40:  # a bound over a whole box (the integrator's enclosures): at 64
                with precision(64):  # bits, as tight there, where Arb's Bessel I takes 0.2 s at 128
                    return fn(v, an) if analytic else fn(v)
            return fn(v, an) if analytic else fn(v)
        except NotSettled, ArithmeticError, ValueError:  # not on this box (too wide): subdivide
            return acb(NAN)
        except Exception as e:  # a bug in the integrand: the integrator would swallow it
            failed.append(e)
            return acb(NAN)

    r = acb.integral(cb, a, b, **kw)
    if failed:
        raise failed[0]
    return r


def _nct_lc(df):
    """log of f_S's constant: f_S(s) = e^lc s^(df-1) e^(-df s^2 / 2)."""
    return LN2() + (df / 2) * (df / 2).log() - (df / 2).lgamma()


def _phi_max(t, d, s_lo, s_hi):
    """An upper bound on e^(-(t s - d)^2 / 2) over s in [s_lo, s_hi] (s_hi None: infinity):
    t s - d runs over an interval; e^(-m^2/2), m its distance from 0."""
    a = t * s_lo - d
    if t.is_zero():
        lo_, hi_ = -d, -d
    elif t > 0:
        lo_, hi_ = a, (t * s_hi - d if s_hi is not None else None)
    else:
        lo_, hi_ = (t * s_hi - d if s_hi is not None else None), a
    if lo_ is not None and lo_ > 0:
        m = lo_.lower()
    elif hi_ is not None and hi_ < 0:
        m = (-hi_).lower()
    else:
        return arb(1)
    return (-(arb(m) ** 2) / 2).exp().upper()


def _nct_integral(df, d, t, k=None):
    """M_k(t) = int_0^inf s^k phi(t s - d) f_S(s) ds (k given), or for k None the tail
    int_0^inf Phi(-(t s - d)) f_S(s) ds (t > 0, d < 0: P(T > t), the side away from d)."""
    lc = _nct_lc(df)
    if k is not None:
        c0 = lc - (2 * PI()).log() / 2
        e = df + k  # the power of s in the integrand in v (s^k f_S(s) ds = ... s^(df + k) dv)

        def h(v):  # log of the integrand in v (arb or acb)
            s = _xexp(v)
            return c0 + e * v - df * s * s / 2 - (t * s - d) ** 2 / 2

        def integrand(v, hm):
            return _xexp(h(v) - hm)
    else:
        e = df
        a = -d

        def h(v):  # (real v only: the cut placement)
            s = v.exp()
            z = (a + t * s) / arb(2).sqrt()
            return lc + e * v - df * s * s / 2 + (z.erfc() / 2).log()

        def integrand(v, hm):  # no log of erfc: on a complex box its phase can wrap round 0
            s = _xexp(v)
            return _xexp(lc + e * v - df * s * s / 2 - hm) * ((a + t * s) / arb(2).sqrt()).erfc() / 2

    # the peak: in v the integrand is unimodal; start from the stationary point of the
    # Gaussian approximation and its curvature (low precision: these only place the cuts)
    with precision(64):
        tt, dd, dff = arb(t.mid()), arb(d.mid()), arb(df.mid())
        A2, B1 = tt * tt + dff, -dd * tt
        s0 = (
            2 * (dff + (k or 0)) / (B1 + (B1 * B1 + 4 * A2 * (dff + (k or 0))).sqrt())
            if B1 >= 0
            else (-B1 + (B1 * B1 + 4 * A2 * (dff + (k or 0))).sqrt()) / (2 * A2)
        )
        if k is None:
            aa = -dd
            s0 = 2 * dff / (aa * tt + (aa * aa * tt * tt + 4 * dff * (tt * tt + dff)).sqrt())
        curv = 2 * (tt * s0) ** 2 + abs(dd * tt * s0) + 2 * dff * s0 * s0
        v0 = float(s0.log().mid())
        sig = float((1 / curv.sqrt()).mid())
        hmax = h(arb(v0)).mid()  # an exact point: a scale only
    drop = arb(ctx.prec) * LN2() + 40
    eps = _eps()

    def bound_left(vl):
        sl = arb(vl).exp()
        if k is not None:
            return (c0 + e * vl).exp() / e * _phi_max(t, d, arb(0), sl)
        return _ncdf(d) * (lc + e * vl).exp() / e  # Phi(-a) P(S < s_lo), a = -d

    def bound_right(vh):
        sh = arb(vh).exp()
        u = df * sh * sh / 2
        if k is not None:  # int_{s_hi}^inf s^(df+k-1) e^(-df s^2/2) ds = (1/2)(2/df)^(e/2) Gamma(e/2, u)
            return c0.exp() * (2 / df) ** (e / 2) / 2 * u.gamma_upper(e / 2) * _phi_max(t, d, sh, None)
        return _ncdf(d - t * sh) * _gamma_q(df / 2, u)  # Phi(-(a + t s_hi)) P(S > s_hi)

    step_l = step_r = max(sig, 1e-6)
    vl, vh = v0 - step_l, v0 + step_r
    while True:
        while h(arb(vl)) - hmax > -drop:
            step_l *= 2
            vl = v0 - step_l
        while h(arb(vh)) - hmax > -drop:
            step_r *= 2
            vh = v0 + step_r
        tol = arb(2) ** (-ctx.prec - 8) * (vh - vl)
        hm = hmax
        I = _integral(lambda v: integrand(v, hm), arb(vl), arb(vh), rel_tol=arb(2) ** (-ctx.prec - 4), abs_tol=tol).real
        if not I.is_finite():
            raise NotSettled("nct integral")
        I = I * hmax.exp()
        bl, br = bound_left(vl), bound_right(vh)
        lo_ok, hi_ok = bl < I * eps, br < I * eps
        if lo_ok and hi_ok:
            return add_error(I, bl + br)
        if not lo_ok:
            step_l *= 2
            vl = v0 - step_l
        if not hi_ok:
            step_r *= 2
            vh = v0 + step_r
        if step_l > 1e5 or step_r > 1e5:
            raise NotSettled("nct integral")


def _nct_upper_sum(df, d, t):
    """P(T > t) for t > 0, d >= 0: (1/2) [sum_j p_j I_y(b, j + 1/2) + sum_j q_j I_y(b, j + 1)],
    y = df / (df + t^2), b = df/2, p_j the Poisson(d^2/2) weights and
    q_j = d e^(-d^2/2) (d^2/2)^j / (sqrt 2 Gamma(j + 3/2)) (both families increasing in j)."""
    b, z = df / 2, t * t
    x, y = z / (df + z), df / (df + z)
    mu = d * d / 2
    lx, ly = x.log(), y.log()
    half = arb(0.5)

    def fam(c0):
        T = lambda j: ((j + c0) * lx + b * ly - (j + c0).log() - _lbeta(j + c0, b)).exp()  # I_x(c) - I_x(c+1)
        return _upper_values(lambda j: _ibeta(y, b, j + c0, x), T, lambda j: x * (j + c0 + b) / (j + c0 + 1))

    if mu.is_zero():
        return _ibeta(y, b, half, x) / 2
    sp = _mixture(_Weights(mu, 1), _fl(mu), fam(half), False, vratio=lambda j: 1 + b / (j + half))
    sq = _mixture(_Weights(mu, 1.5), _fl(mu), fam(arb(1)), False, vratio=lambda j: 1 + b / (j + 1))
    return (sp + d * (-mu).exp() / arb(2).sqrt() * sq) / 2


def _nct_lower_sum(df, d, t):
    """P(T <= t) for t > 0: Phi(-d) + (1/2) [sum_j p_j I_x(j + 1/2, b) + sum_j q_j I_x(j + 1, b)]
    (the families decreasing in j; for d < 0 the q sum enters negative, against a total of
    at least Phi(-d) > 1/2)."""
    b, z = df / 2, t * t
    x, y = z / (df + z), df / (df + z)
    mu = d * d / 2
    lx, ly = x.log(), y.log()
    half = arb(0.5)

    def fam(c0):
        T = lambda j: ((j + c0) * lx + b * ly - (j + c0).log() - _lbeta(j + c0, b)).exp()
        return _lower_values(lambda J: _ibeta(x, J + c0, b, y), T, lambda m: (m + c0) / (x * (m + c0 - 1 + b)))

    phi = _ncdf(-d)
    if mu.is_zero():
        return phi + _ibeta(x, half, b, y) / 2
    sp = _mixture(_Weights(mu, 1), _fl(mu), fam(half), True)
    sq = _mixture(_Weights(mu, 1.5), _fl(mu), fam(arb(1)), True)
    return phi + (sp + d * (-mu).exp() / arb(2).sqrt() * sq) / 2


def _nct_pair(df, d, t, upper):
    if t.is_zero():
        return _ncdf(d) if upper else _ncdf(-d)
    if t < 0:  # P(T <= t; d) = P(T > -t; -d)
        return _nct_pair(df, -d, -t, not upper)
    if upper:
        return _nct_upper_sum(df, d, t) if d >= 0 else _nct_integral(df, d, t)
    return _nct_lower_sum(df, d, t)


class NoncentralT(Distribution):
    start = staticmethod(lambda df, nc: nc if nc != 0 else 1.0)
    params = ("df", "nc")

    @staticmethod
    def logpdf(t, df, d):
        if d == 0:
            return _t_pdf(t, df).log()
        return _nct_integral(df, d, t, 1).log()

    pdf = staticmethod(lambda t, df, d: _t_pdf(t, df) if d == 0 else _nct_integral(df, d, t, 1))
    cdf = staticmethod(lambda t, df, d: StudentT.cdf(t, df) if d == 0 else _nct_pair(df, d, t, False))
    ccdf = staticmethod(lambda t, df, d: StudentT.ccdf(t, df) if d == 0 else _nct_pair(df, d, t, True))
    mean = staticmethod(lambda df, d: _nct_raw(df, d, 1) if df > 1 else NAN)

    @staticmethod
    def variance(df, d):
        if not df > 2:
            return INF if df > 1 else NAN
        return _nct_raw(df, d, 2) - _nct_raw(df, d, 1) ** 2

    @staticmethod
    def standard_deviation(df, d):
        return _sqrt(NoncentralT.variance(df, d))

    @staticmethod
    def skewness(df, d):
        if not df > 3:
            return NAN
        m1, m2, m3 = (_nct_raw(df, d, r) for r in (1, 2, 3))
        v = m2 - m1 * m1
        return (m3 - 3 * m1 * m2 + 2 * m1**3) / v ** arb(1.5)

    @staticmethod
    def kurtosis(df, d):
        if not df > 4:
            return INF if df > 2 else NAN
        m1, m2, m3, m4 = (_nct_raw(df, d, r) for r in (1, 2, 3, 4))
        v = m2 - m1 * m1
        return (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1**4) / v**2 - 3

    @staticmethod
    def mode(df, d):
        """f'(t) = -(t M_3 - d M_2): the root of t M_3 / M_2 - d (increasing through it)."""
        if d == 0:
            return arb(0)
        t0 = _our("noncentralt", "mode", (_fl(df), _fl(d)))
        g = lambda t: t * _nct_integral(df, d, t, 3) / _nct_integral(df, d, t, 2) - d
        return _solve(g, t0, increasing=True, positive=False)

    @staticmethod
    def entropy(df, d):
        if d == 0:
            return StudentT.entropy(df)
        return _nct_entropy(df, d)

    @staticmethod
    def _params(rng, n):
        k = n // 3
        df = np.concatenate(
            [loguniform(rng, 1e-1, 1e5, k), loguniform(rng, 1e-1, 10, k), loguniform(rng, 1, 1e5, n - 2 * k)]
        )
        d = np.concatenate(
            [
                loguniform(rng, 1e-3, 100, k, signed=True),
                loguniform(rng, 1e-3, 10, k, signed=True),
                loguniform(rng, 1e-3, 100, n - 2 * k, signed=True),
            ]
        )
        d[: n // 50] = 0.0
        rng.shuffle(d)
        return df, d

    @staticmethod
    def _x(rng, ps):
        df, d = ps
        return _own_quantiles("NoncentralT", {"df": df, "nc": d}, _nc_probs(rng, df.size))

    @classmethod
    def param_sets(cls, rng):
        ps = cls._params(rng, 300)
        return ps, np.ones(ps[0].size, dtype=bool)


def _nct_raw(df, d, r):
    zr = [arb(1), d, d * d + 1, d**3 + 3 * d, d**4 + 6 * d * d + 3][r]
    return (df / 2) ** (arb(r) / 2) * (((df - r) / 2).lgamma() - (df / 2).lgamma()).exp() * zr


def _ncf_raw(dfn, dfd, nc, r):
    k, l = dfn, nc
    x = [
        arb(1),
        k + l,
        (k + l) ** 2 + 2 * (k + 2 * l),
        (k + l) ** 3 + 6 * (k + l) * (k + 2 * l) + 8 * (k + 3 * l),
        (k + l) ** 4 + 12 * (k + l) ** 2 * (k + 2 * l) + 4 * (11 * k * k + 44 * k * l + 36 * l * l) + 48 * (k + 4 * l),
    ][r]
    inv = ((dfd / 2 - r).lgamma() - (dfd / 2).lgamma()).exp() / arb(2) ** r
    return (dfd / dfn) ** r * x * inv


class NoncentralF(Distribution):
    start = staticmethod(lambda dfn, dfd, nc: (dfn + nc) / dfn)
    params = ("dfn", "dfd", "nc")
    positive = True

    @staticmethod
    def logpdf(x, dfn, dfd, nc):
        if nc == 0:
            return _log(_f_pdf(x, dfn, dfd))
        if not x > 0:
            return -INF if (x < 0 or dfn > 2) else (INF if dfn < 2 else (-nc / 2).log())
        return _ncf_logpdf(dfn, dfd, nc, x)

    @staticmethod
    def pdf(x, dfn, dfd, nc):
        if nc == 0:
            return _f_pdf(x, dfn, dfd)
        if not x > 0:
            return arb(0) if (x < 0 or dfn > 2) else (INF if dfn < 2 else (-nc / 2).exp())
        return _ncf_logpdf(dfn, dfd, nc, x).exp()

    cdf = staticmethod(
        lambda x, dfn, dfd, nc: (
            (F.cdf(x, dfn, dfd) if nc == 0 else _ncf_pair(dfn, dfd, nc, x, False)) if x > 0 else arb(0)
        )
    )
    ccdf = staticmethod(
        lambda x, dfn, dfd, nc: (
            (F.ccdf(x, dfn, dfd) if nc == 0 else _ncf_pair(dfn, dfd, nc, x, True)) if x > 0 else arb(1)
        )
    )
    mean = staticmethod(lambda dfn, dfd, nc: _ncf_raw(dfn, dfd, nc, 1) if dfd > 2 else INF)

    @staticmethod
    def variance(dfn, dfd, nc):
        if not dfd > 4:
            return INF if dfd > 2 else NAN
        return _ncf_raw(dfn, dfd, nc, 2) - _ncf_raw(dfn, dfd, nc, 1) ** 2

    @staticmethod
    def standard_deviation(dfn, dfd, nc):
        return _sqrt(NoncentralF.variance(dfn, dfd, nc))

    @staticmethod
    def skewness(dfn, dfd, nc):
        if not dfd > 6:
            return INF if dfd > 4 else NAN
        m1, m2, m3 = (_ncf_raw(dfn, dfd, nc, r) for r in (1, 2, 3))
        v = m2 - m1 * m1
        return (m3 - 3 * m1 * m2 + 2 * m1**3) / v ** arb(1.5)

    @staticmethod
    def kurtosis(dfn, dfd, nc):
        if not dfd > 8:
            return INF if dfd > 4 else NAN
        m1, m2, m3, m4 = (_ncf_raw(dfn, dfd, nc, r) for r in (1, 2, 3, 4))
        v = m2 - m1 * m1
        return (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1**4) / v**2 - 3

    @staticmethod
    def mode(dfn, dfd, nc):
        """The root of d/dx log f = (a-1)/x - (a+b) dfn / (dfd + dfn x)
        + (al/be) 1F1(al+1; be+1; z) / 1F1(al; be; z) dz/dx (al = a + b, be = a), falling
        through it; 0 for dfn <= 2, where the density falls from 0."""
        if nc == 0:
            return F.mode(dfn, dfd)
        if not dfn > 2:
            return arb(0)
        a, b = dfn / 2, dfd / 2
        al, be = a + b, a

        def g(x):
            s = dfn * x + dfd
            z = nc * dfn * x / (2 * s)
            dz = nc * dfn * dfd / (2 * s * s)
            return (a - 1) / x - al * dfn / s + al / be * _hyp1f1(al + 1, be + 1, z) / _hyp1f1(al, be, z) * dz

        x0 = _our("noncentralf", "mode", (_fl(dfn), _fl(dfd), _fl(nc)))
        if not x0 > 0:
            x0 = _fl((dfn - 2) / dfn * dfd / (dfd + 2) * (1 + nc / dfn))
        return _solve(g, x0, increasing=False, positive=True)

    @staticmethod
    def entropy(dfn, dfd, nc):
        if nc == 0:
            return F.entropy(dfn, dfd)
        return _ncf_entropy(dfn, dfd, nc)

    @staticmethod
    def _params(rng, n):
        k = n // 3
        dfn = np.concatenate(
            [loguniform(rng, 1e-2, 1e4, k), loguniform(rng, 1e-2, 10, k), loguniform(rng, 1e-1, 1e4, n - 2 * k)]
        )
        dfd = np.concatenate(
            [loguniform(rng, 1e-2, 1e4, k), loguniform(rng, 1e-2, 10, k), loguniform(rng, 3, 1e4, n - 2 * k)]
        )
        nc = np.concatenate(
            [loguniform(rng, 1e-3, 1e4, k), loguniform(rng, 1e-3, 50, k), loguniform(rng, 1e-3, 1e4, n - 2 * k)]
        )
        nc[: n // 50] = 0.0
        rng.shuffle(nc)
        return dfn, dfd, nc

    @staticmethod
    def _x(rng, ps):
        dfn, dfd, nc = ps
        return _own_quantiles("NoncentralF", {"dfn": dfn, "dfd": dfd, "nc": nc}, _nc_probs(rng, dfn.size))

    @classmethod
    def param_sets(cls, rng):
        ps = cls._params(rng, 600)
        return ps, np.ones(ps[0].size, dtype=bool)


# ---- the noncentral entropies: -int f log f, in u = log x for the two positive ones
#
# H = -int F(u) log f(e^u) du, F(u) = f(e^u) e^u, by acb.integral (the integrand is
# analytic near the real axis; the logs check it) over [u_lo, u_hi], plus bounds on the two
# ends left out, from bounds on f there of the forms K x^p:
#   x in (0, X0], X0 <= 1, K_lo x^(s-1) <= f <= K_hi x^(s-1):
#     int_0^X0 f |log f| dx <= K_hi X0^s / s (A + B (log(1/X0) + 1/s)),
#     A = max |log K|, B = |s - 1|  (added to the ball both ways: f may exceed 1 there);
#   beyond X1 per distribution (f < 1 there, so -f log f >= 0: added one way, as [0, bound]).
# The window widens until both bounds are below the integral's last bits.


def _entropy_lower_tail(khi, klo, s, x0):
    A = abs(khi.log()) if abs(khi.log()) > abs(klo.log()) else abs(klo.log())
    B = abs(s - 1)
    return khi * x0**s / s * (A + B * (-x0.log() + 1 / s))


def _entropy_window(logF, u_lo, u_hi, lower_bound, upper_bound, upper_step=None):
    """-int F(u) (logF(u) - u) du over [u_lo, u_hi] plus the end bounds (lower_bound(u_lo)
    symmetric, upper_bound(u_hi) one-sided), widening each end (u_lo down by doubling its
    distance from the middle; u_hi up the same way, or by upper_step for a tail that falls
    exponentially in x) until its bound is below the last bits."""
    mid = (u_lo + u_hi) / 2
    while True:
        I = _integral(
            lambda u, an: _ent_integrand(logF, u, an),
            arb(u_lo),
            arb(u_hi),
            analytic=True,
            rel_tol=arb(2) ** (-ctx.prec - 4),
            abs_tol=arb(2) ** (-ctx.prec - 20),
        ).real
        if not I.is_finite():
            raise NotSettled("entropy integral")
        bl, bu = lower_bound(arb(u_lo).exp()), upper_bound(arb(u_hi).exp())
        tol = abs(I) * _eps()
        ok_l, ok_u = bl.is_finite() and bl < tol, bu.is_finite() and bu < tol
        if ok_l and ok_u:
            return I + arb(0, bl.upper()) + arb(bu.upper() / 2, bu.upper() / 2)
        if not ok_l:
            u_lo = mid - 2 * (mid - u_lo)
        if not ok_u:
            u_hi = u_hi + upper_step if upper_step else mid + 2 * (u_hi - mid)
        if u_hi - u_lo > 1e7:
            raise NotSettled("entropy window")


def _ent_integrand(logF, u, an):
    lf = logF(u, an)
    return -_xexp(lf) * (lf - u)


def _ncx2_entropy(df, nc):
    """f(x) = e^(-nc/2) g_df(x) 0F1(; s; nc x / 4) (s = df/2, g_df the central density):
    on (0, X0] f lies between e^(-nc/2 - X0/2) c x^(s-1) and e^(-nc/2) 0F1(; s; nc X0/4) c x^(s-1)
    (c = 1 / (2^s Gamma(s)), 0F1 rising from 1); beyond X1 >= max(4 nc, 1), using
    0F1(; s; y) <= 1 + (y/s) e^(2 sqrt y) ((s)_k >= s (k-1)!) and
    (sqrt x - sqrt nc)^2 >= 2 kap x, kap = (1 - sqrt(nc/X1))^2 / 2 >= 1/8:
    f <= c x^(s-1) (1 + nc x / (4 s)) e^(-kap x) (falling for x >= (s + 1) / kap, where X1
    must lie), and f >= e^(-nc/2) c x^(s-1) e^(-x/2) gives -log f <= A1 + B1 x
    (A1 = nc/2 - log c, B1 = 1/2 + |s - 1|): the bound is
    c sum_j k_j kap^-(s+j) Gamma(s + j, kap X1) of the expanded product."""
    s = df / 2
    v = s - 1
    lnc = nc.log()
    c = -(s * LN2() + s.lgamma())  # log c

    def logF(u, an):  # log(f(e^u) e^u) by the Bessel form, z = sqrt(nc e^u) entire in u
        x = _xexp(u)
        z = _xexp((lnc + u) / 2)
        return -LN2() - (x + nc) / 2 + v / 2 * (u - lnc) + z.bessel_i(v, scaled=True).log(analytic=an) + z + u

    def lower(x0):
        y = nc * x0 / 4  # 0F1(; s; y) = Gamma(s) y^((1-s)/2) I_(s-1)(2 sqrt y)
        f01 = s.gamma() * y ** ((1 - s) / 2) * (2 * y.sqrt()).bessel_i(v) if y > 0 else arb(1)
        khi = (-nc / 2 + c).exp() * f01
        klo = (-nc / 2 - x0 / 2 + c).exp()
        return _entropy_lower_tail(khi, klo, s, x0)

    def upper(x1):
        cc = c.exp()
        c1 = nc / (4 * s)
        A1 = nc / 2 - c
        A1 = A1 if A1 > 0 else arb(0)
        B1 = arb(0.5) + abs(s - 1)
        if not (x1 >= 4 * nc and x1 >= 1):
            return arb(INF)
        kap = ((1 - (nc / x1).sqrt()) ** 2 / 2).lower()
        kap = arb(kap)
        if not (x1 >= (s + 1) / kap and cc * x1 ** (s - 1) * (1 + c1 * x1) * (-kap * x1).exp() < 1):
            return arb(INF)  # the bound must fall beyond X1, and f < 1 there
        z = kap * x1
        G = lambda a: z.gamma_upper(a) / kap**a
        return cc * (A1 * G(s) + (A1 * c1 + B1) * G(s + 1) + B1 * c1 * G(s + 2))

    m, sd_ = _fl(df + nc), math.sqrt(_fl(2 * (df + 2 * nc)))
    u_hi = math.log(max(m + 15 * sd_ + 30, 4 * _fl(nc), 2 * _fl(s) + 2, 1.0))
    u_lo = min(0.0, math.log(m - 10 * sd_) if m > 10 * sd_ else 0.0) - (ctx.prec * 0.7 + 40) / _fl(s)
    return _entropy_window(logF, u_lo, u_hi, lower, upper, upper_step=math.log(1.5))


def _ncf_entropy(dfn, dfd, nc):
    """f(x) = e^(-nc/2) f_F(x) 1F1(a+b; a; (nc/2) w), w = dfn x / (dfn x + dfd) in [0, 1),
    1F1 rising in w from 1 to M = 1F1(a+b; a; nc/2). On (0, X0]: f between
    e^(-nc/2) r^a (1 + r X0)^(-(a+b)) / B x^(a-1) and e^(-nc/2) r^a 1F1(.. w(X0)) / B x^(a-1)
    (r = dfn/dfd, B = B(a, b)); beyond X1 >= 1/r: K'_lo x^(-b-1) <= f <= K'_hi x^(-b-1),
    K'_hi = e^(-nc/2) M r^(-b) / B, K'_lo = 2^(-(a+b)) e^(-nc/2) r^(-b) / B, so
    int_X1^inf f (-log f) <= K'_hi X1^(-b) (A/b + (b+1)(log X1 / b + 1/b^2)), A = max(0, -log K'_lo)."""
    a, b = dfn / 2, dfd / 2
    r = dfn / dfd
    lr = r.log()
    c0 = -nc / 2 + a * lr - _lbeta(a, b)

    def logF(u, an):
        x = _xexp(u)
        w = 1 / (1 + _xexp(-(u + lr)))
        h = _hyp1f1(a + b, a, nc / 2 * w)
        return c0 + a * u - (a + b) * (r * x + 1).log(analytic=an) + h.log(analytic=an)

    def lower(x0):
        w0 = r * x0 / (1 + r * x0)
        khi = (c0 + _hyp1f1(a + b, a, nc / 2 * w0).log()).exp()
        klo = (c0 - (a + b) * (1 + r * x0).log()).exp()
        return _entropy_lower_tail(khi, klo, a, x0)

    def upper(x1):
        if not r * x1 >= 1:
            return arb(INF)
        base = -nc / 2 - b * lr - _lbeta(a, b)
        khi = (base + _hyp1f1(a + b, a, nc / 2).log()).exp()
        klo = (base - (a + b) * LN2()).exp()
        if not khi * x1 ** (-b - 1) < 1:
            return arb(INF)
        A = -klo.log()
        A = A if A > 0 else arb(0)
        return khi * x1 ** (-b) * (A / b + (b + 1) * (x1.log() / b + 1 / b**2))

    m = _fl((dfn + nc) / dfn)
    u_hi = max(math.log(m) + 10, -_fl(lr) + 1) + (ctx.prec * 0.7 + 40) / _fl(b)
    u_lo = min(0.0, math.log(m) - 10) - (ctx.prec * 0.7 + 40) / _fl(a)
    return _entropy_window(logF, u_lo, u_hi, lower, upper)


# The noncentral t's entropy: H = H+(d) + H+(-d), H+(d) = -int_0^inf f log f dt (the density
# at -t is the one at t with -d). On t > 0, s = u / t turns the density into
# f(t) = C t^-(df+1) G(1/t^2), C = e^lc / sqrt(2 pi), G(e2) = int_0^inf u^df e^(-(u-d)^2/2 - df u^2 e2/2) du,
# G falling in e2 >= 0 and entire in it; in w = log t, F(w) = f(e^w) e^w = C e^(-df w) G(e^(-2w)),
# and H+ = -int F (log F - w) dw (_entropy_window). The ends:
#   t <= t0 = e^w_lo: f <= E[S] / sqrt(2 pi) and f >= (1/2) phi(|d| + 2) P(1/2 <= S <= 2) (t0 <= 1),
#     so the piece is at most t0 max f |log f| (both ways);
#   t >= T = e^w_hi: K_lo t^-(df+1) <= f <= K_hi t^-(df+1), K_lo = C G(1/T^2), K_hi = C G(0), so
#     int_T^inf f (-log f) <= K_hi T^-df (A/df + (df+1)(log T/df + 1/df^2)), A = max(0, -log K_lo)
#     (f < 1 there).


def _nct_G(df, d, e2):
    """G(e2) = int_0^inf u^df e^(-(u-d)^2/2 - df u^2 e2/2) du (e2 an arb, or an acb box with
    Re e2 >= 0, else NaN), by acb.integral in y = log u around the peak (the log integrand
    (df+1) y - (u-d)^2/2 - df e2 u^2/2 is unimodal in y for real e2), plus the ends:
    u < U0: <= e^((df+1) y_lo) / (df+1) sup e^(-(u-d)^2/2); u > U1 > |d|, with
    (u - d)^2 >= u^2 (1 - |d|/U1)^2: <= (1/2) k^(-(df+1)/2) Gamma((df+1)/2, k U1^2),
    k = (1 - |d|/U1)^2 / 2 + df rho / 2, rho = min Re e2 >= 0."""
    rho = e2.real.lower() if isinstance(e2, acb) else e2.lower()
    if rho < 0:
        return acb(NAN) if isinstance(e2, acb) else arb(NAN)
    rho = arb(rho)
    e = df + 1

    def h(y):
        u = _xexp(y)
        return e * y - (u - d) ** 2 / 2 - df * e2 * u * u / 2

    with precision(64):
        dd, dff = arb(d.mid()), arb(df.mid())
        A = 1 + dff * arb((e2.real if isinstance(e2, acb) else e2).mid())
        s0 = (dd + (dd * dd + 4 * A * (dff + 1)).sqrt()) / (2 * A)
        y0 = float(s0.log().mid())
        sig = float((1 / (2 * A * s0 * s0 - dd * s0).sqrt()).mid())
        e2r = arb((e2.real if isinstance(e2, acb) else e2).mid())
        hr = lambda y: e * y - (arb(y).exp() - d) ** 2 / 2 - df * e2r * arb(y).exp() ** 2 / 2  # h on the real axis
        hmax = hr(y0).mid()
    drop = arb(ctx.prec) * LN2() + 40
    eps = _eps()

    def bound_left(yl):
        u0 = arb(yl).exp()
        lo_, hi_ = -d, u0 - d
        m = lo_.lower() if lo_ > 0 else ((-hi_).lower() if hi_ < 0 else 0)
        return (e * yl).exp() / e * (-(arb(m) ** 2) / 2).exp()

    def bound_right(yh):
        u1 = arb(yh).exp()
        if not u1 > abs(d):
            return arb(INF)
        k = (1 - abs(d) / u1) ** 2 / 2 + df * rho / 2
        return k ** (-e / 2) / 2 * (k * u1 * u1).gamma_upper(e / 2)

    step_l = step_r = max(sig, 1e-6)
    yl, yh = y0 - step_l, y0 + step_r
    with precision(64):
        while hr(yl) - hmax > -drop:
            step_l *= 2
            yl = y0 - step_l
        while hr(yh) - hmax > -drop:
            step_r *= 2
            yh = y0 + step_r
    tries = 0
    while True:
        tol = arb(2) ** (-ctx.prec - 8) * (yh - yl)
        hm = hmax
        I = _integral(lambda y: _xexp(h(y) - hm), arb(yl), arb(yh), rel_tol=arb(2) ** (-ctx.prec - 4), abs_tol=tol)
        if not I.is_finite():
            return I
        I = I * hmax.exp()
        if isinstance(e2, arb):
            I = I.real
        mag = abs(I).lower() if isinstance(I, arb) else abs(I.real).lower()
        bl, br = bound_left(yl), bound_right(yh)
        ok_l, ok_r = bl < mag * eps, br.is_finite() and br < mag * eps
        tries += 1
        # tight ends, or (a box of the outer integral, whose enclosure only has to be valid)
        # finite ones after a few widenings: the ball is right either way, only wider
        if (ok_l and ok_r) or (tries > 4 and bl.is_finite() and br.is_finite()):
            err = (bl + br).upper()
            return add_error(I, err) if isinstance(I, arb) else I + acb(arb(0, err), arb(0, err))
        if not ok_l:
            step_l *= 2
            yl = y0 - step_l
        if not ok_r:
            step_r *= 2
            yh = y0 + step_r
        if step_l > 1e4 or step_r > 1e4:
            raise NotSettled("nct G")


def _nct_entropy_half(df, d):
    """H+(d) = -int_0^inf f log f dt (see above)."""
    lc = _nct_lc(df)
    lC = lc - (2 * PI()).log() / 2

    def logF(w, an):
        G = _nct_G(df, d, _xexp(-2 * w))
        return lC - df * w + G.log(analytic=an)

    ES = (2 / df).sqrt() * (((df + 1) / 2).lgamma() - (df / 2).lgamma()).exp()
    fmax = ES / (2 * PI()).sqrt()
    pmid = _gamma_p(df / 2, 2 * df) - _gamma_p(df / 2, df / 8)  # P(1/2 <= S <= 2), S^2 = V/df
    fmin = _npdf(abs(d) + 2) / 2 * pmid
    L = abs(fmax.log()) if abs(fmax.log()) > abs(fmin.log()) else abs(fmin.log())

    def lower(t0):  # int_0^t0 f |log f| dt <= t0 fmax L (t0 <= 1)
        if not t0 <= 1:
            return arb(INF)
        return t0 * fmax * L

    G0 = _nct_G(df, d, arb(0))

    def upper(T):
        khi = (lC + G0.log()).exp()
        klo = (lC + _nct_G(df, d, 1 / (T * T)).log()).exp()
        if not khi * T ** (-df - 1) < 1:
            return arb(INF)
        A = -klo.log()
        A = A if A > 0 else arb(0)
        return khi * T ** (-df) * (A / df + (df + 1) * (T.log() / df + 1 / df**2))

    c = abs(_fl(d)) + 1
    w_hi = math.log(c + 10) + (ctx.prec * 0.7 + 40) / _fl(df)
    w_lo = math.log(1 / (10 * c)) - (ctx.prec * 0.7 + 40)
    return _entropy_window(logF, w_lo, min(w_hi, 1e4), lower, upper)


def _nct_entropy(df, d):
    return _nct_entropy_half(df, d) + _nct_entropy_half(df, -d)


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
    return a + a.lgamma() + (1 - a) * a.digamma()


def _gamma_pdf(x, a, s):
    if x <= 0:
        return INF if (x == 0 and a < 1) else (1 / s if (x == 0 and a == 1) else arb(0))
    y = x / s
    return ((a - 1) * y.log() - y - a.lgamma()).exp() / s


class Gamma(Distribution):
    start = staticmethod(lambda a, s: a * s)
    params = ("a", "scale")
    positive = True

    pdf = staticmethod(_gamma_pdf)
    cdf = staticmethod(lambda x, a, s: _gamma_p(a, x / s) if x > 0 else arb(0))
    ccdf = staticmethod(lambda x, a, s: _gamma_q(a, x / s) if x > 0 else arb(1))
    mean = staticmethod(lambda a, s: a * s)
    variance = staticmethod(lambda a, s: a * s * s)
    standard_deviation = staticmethod(lambda a, s: a.sqrt() * s)
    skewness = staticmethod(lambda a, s: 2 / a.sqrt())
    kurtosis = staticmethod(lambda a, s: 6 / a)
    mode = staticmethod(lambda a, s: (a - 1) * s if a > 1 else arb(0))
    entropy = staticmethod(lambda a, s: _gamma_entropy(a) + s.log())

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
        y = _quantile_points(
            rng,
            a.size,
            lambda p, i: st.gamma.ppf(p, a[i]),
            lambda q, i: st.gamma.isf(q, a[i]),
            lambda rng, k: loguniform(rng, 1e-300, 1e5, k),
        )
        return y * s


class ChiSquared(Distribution):
    start = staticmethod(lambda df: df)
    params = ("df",)
    positive = True

    pdf = staticmethod(lambda x, df: _gamma_pdf(x, df / 2, arb(2)))
    cdf = staticmethod(lambda x, df: Gamma.cdf(x, df / 2, arb(2)))
    ccdf = staticmethod(lambda x, df: Gamma.ccdf(x, df / 2, arb(2)))
    mean = staticmethod(lambda df: df)
    variance = staticmethod(lambda df: 2 * df)
    standard_deviation = staticmethod(lambda df: (2 * df).sqrt())
    skewness = staticmethod(lambda df: (8 / df).sqrt())
    kurtosis = staticmethod(lambda df: 12 / df)
    mode = staticmethod(lambda df: df - 2 if df > 2 else arb(0))
    entropy = staticmethod(lambda df: _gamma_entropy(df / 2) + LN2())

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([loguniform(rng, 1e-2, 1e5, n - 50), np.arange(1.0, 51.0)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st

        (df,) = ps
        return _quantile_points(
            rng,
            df.size,
            lambda p, i: st.chi2.ppf(p, df[i]),
            lambda q, i: st.chi2.isf(q, df[i]),
            lambda rng, k: loguniform(rng, 1e-300, 1e5, k),
        )


class Beta(Distribution):
    start = staticmethod(lambda a, b: a / (a + b))
    params = ("a", "b")
    unit = True

    @staticmethod
    def pdf(x, a, b):
        if x <= 0 or x >= 1:
            if x == 0:
                return INF if a < 1 else (b if a == 1 else arb(0))
            if x == 1:
                return INF if b < 1 else (a if b == 1 else arb(0))
            return arb(0)
        return ((a - 1) * x.log() + (b - 1) * (-x).log1p() - _lbeta(a, b)).exp()

    cdf = staticmethod(lambda x, a, b: _ibeta(x, a, b) if 0 < x < 1 else arb(0 if x <= 0 else 1))
    ccdf = staticmethod(lambda x, a, b: _ibeta(1 - x, b, a, x) if 0 < x < 1 else arb(1 if x <= 0 else 0))
    mean = staticmethod(lambda a, b: a / (a + b))
    variance = staticmethod(lambda a, b: a * b / ((a + b) ** 2 * (a + b + 1)))
    standard_deviation = staticmethod(lambda a, b: (a * b / ((a + b) ** 2 * (a + b + 1))).sqrt())
    skewness = staticmethod(lambda a, b: 2 * (b - a) * (a + b + 1).sqrt() / ((a + b + 2) * (a * b).sqrt()))
    kurtosis = staticmethod(
        lambda a, b: 6 * ((a - b) ** 2 * (a + b + 1) - a * b * (a + b + 2)) / (a * b * (a + b + 2) * (a + b + 3))
    )

    @staticmethod
    def mode(a, b):
        if a > 1 and b > 1:
            return (a - 1) / (a + b - 2)
        if a <= 1 and b <= 1:
            return arb(1) if (a == 1 and b < 1) else (arb(0) if (b == 1 and a < 1) else NAN)
        return arb(0) if a <= 1 else arb(1)

    @staticmethod
    def entropy(a, b):
        c = a + b
        return _lbeta(a, b) - (a - 1) * a.digamma() - (b - 1) * b.digamma() + (c - 2) * c.digamma()

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
        return _quantile_points(
            rng,
            a.size,
            lambda p, i: st.beta.ppf(p, a[i], b[i]),
            lambda q, i: st.beta.isf(q, a[i], b[i]),
            lambda rng, k: np.concatenate(
                [loguniform(rng, 1e-300, 0.5, k // 2), 1 - loguniform(rng, 2**-53, 0.5, k - k // 2)]
            ),
        )


def _t_pdf(t, df):
    return (
        ((df + 1) / 2).lgamma() - (df / 2).lgamma() - (df * PI()).log() / 2 - (df + 1) / 2 * (t * t / df).log1p()
    ).exp()


def _t_lower(t, df):
    """P(T <= t) for t <= 0: I_y(df/2, 1/2) / 2, y = df / (df + t^2); near t = 0 (y near 1)
    as (1 - I_x(1/2, df/2)) / 2, x = t^2 / (df + t^2), where I_x is not near 1."""
    t2 = t * t
    den = df + t2
    if t2 < df:
        u = _ibeta(t2 / den, arb(0.5), df / 2, df / den)
        if u < 0.75:
            return (1 - u) / 2
    return _ibeta(df / den, df / 2, arb(0.5), t2 / den) / 2


class StudentT(Distribution):
    start = staticmethod(lambda df: 1.0)
    params = ("df",)

    pdf = staticmethod(lambda t, df: _t_pdf(t, df))
    cdf = staticmethod(lambda t, df: _t_lower(t, df) if t <= 0 else 1 - _t_lower(-t, df))
    ccdf = staticmethod(lambda t, df: _t_lower(-t, df) if t >= 0 else 1 - _t_lower(t, df))
    mean = staticmethod(lambda df: arb(0) if df > 1 else NAN)
    variance = staticmethod(lambda df: df / (df - 2) if df > 2 else (INF if df > 1 else NAN))
    standard_deviation = staticmethod(lambda df: (df / (df - 2)).sqrt() if df > 2 else (INF if df > 1 else NAN))
    skewness = staticmethod(lambda df: arb(0) if df > 3 else NAN)
    kurtosis = staticmethod(lambda df: 6 / (df - 4) if df > 4 else (INF if df > 2 else NAN))
    median = staticmethod(lambda df: arb(0))
    mode = staticmethod(lambda df: arb(0))

    @staticmethod
    def entropy(df):
        h = arb(0.5)
        return (df + 1) / 2 * (((df + 1) / 2).digamma() - (df / 2).digamma()) + df.log() / 2 + _lbeta(df / 2, h)

    @staticmethod
    def _params(rng, n):
        return (
            np.concatenate(
                [
                    loguniform(rng, 1e-1, 1e6, n - 60),
                    loguniform(rng, 1e-2, 1e-1, 20),
                    np.array([1.0, 2.0, 3.0, 4.0] * 10),
                ]
            ),
        )

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st

        (df,) = ps
        return _quantile_points(
            rng,
            df.size,
            lambda p, i: st.t.ppf(p, df[i]),
            lambda q, i: st.t.isf(q, df[i]),
            lambda rng, k: loguniform(rng, 1e-300, 1e300, k, signed=True),
        )

    @classmethod
    def param_sets(cls, rng):
        (df,), g = super().param_sets(rng)
        edges = np.concatenate([around([1.0, 2.0, 3.0, 4.0], 3), loguniform(rng, 1e6, 1e15, 30)])
        return (np.concatenate([df, edges]),), np.ones(df.size + edges.size, dtype=bool)


def _f_pdf(x, m, n):
    if x <= 0:
        return (INF if m < 2 else (arb(1) if m == 2 else arb(0))) if x == 0 else arb(0)
    a, b = m / 2, n / 2
    return (a * (m * x).log() + b * n.log() - (a + b) * (m * x + n).log() - x.log() - _lbeta(a, b)).exp()


class F(Distribution):
    start = staticmethod(lambda m, n: 1.0)
    params = ("dfn", "dfd")
    positive = True

    pdf = staticmethod(_f_pdf)

    # Each side directly from its own argument (w = m x / (m x + n) for the cdf, v = 1 - w
    # for the ccdf, both computed apart so neither rounds near 1).
    @staticmethod
    def cdf(x, m, n):
        if x <= 0:
            return arb(0)
        s = m * x + n
        return _ibeta(m * x / s, m / 2, n / 2, n / s)

    @staticmethod
    def ccdf(x, m, n):
        if x <= 0:
            return arb(1)
        s = m * x + n
        return _ibeta(n / s, n / 2, m / 2, m * x / s)

    mean = staticmethod(lambda m, n: n / (n - 2) if n > 2 else INF)
    variance = staticmethod(
        lambda m, n: 2 * n * n * (m + n - 2) / (m * (n - 2) ** 2 * (n - 4)) if n > 4 else (INF if n > 2 else NAN)
    )
    standard_deviation = staticmethod(lambda m, n: _sqrt(F.variance(m, n)) if n > 2 else NAN)
    skewness = staticmethod(
        lambda m, n: (
            (2 * m + n - 2) * (8 * (n - 4)).sqrt() / ((n - 6) * (m * (m + n - 2)).sqrt())
            if n > 6
            else (INF if n > 4 else NAN)
        )
    )
    kurtosis = staticmethod(
        lambda m, n: (
            12 * (m * (5 * n - 22) * (m + n - 2) + (n - 4) * (n - 2) ** 2) / (m * (n - 6) * (n - 8) * (m + n - 2))
            if n > 8
            else (INF if n > 4 else NAN)
        )
    )
    mode = staticmethod(lambda m, n: (m - 2) / m * n / (n + 2) if m > 2 else arb(0))

    @staticmethod
    def entropy(m, n):
        a, b = m / 2, n / 2
        return (
            (n / m).log() + _lbeta(a, b) + (1 - a) * a.digamma() - (1 + b) * b.digamma() + (a + b) * (a + b).digamma()
        )

    @staticmethod
    def _params(rng, n):
        m = loguniform(rng, 1e-1, 1e5, n)
        d = loguniform(rng, 1e-1, 1e5, n)
        return m, d

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st

        m, d = ps
        x = _quantile_points(
            rng,
            m.size,
            lambda p, i: st.f.ppf(p, m[i], d[i]),
            lambda q, i: st.f.isf(q, m[i], d[i]),
            lambda rng, k: loguniform(rng, 1e-300, 1e300, k),
        )
        return np.where(np.isfinite(x) & (x > 0), x, loguniform(rng, 1e-300, 1e300, x.size))

    @classmethod
    def param_sets(cls, rng):
        (m, d), g = super().param_sets(rng)
        edges = np.concatenate([around([2.0, 4.0, 6.0, 8.0], 3), loguniform(rng, 1e5, 1e12, 30)])
        return (np.concatenate([m, loguniform(rng, 1e-1, 1e3, edges.size)]), np.concatenate([d, edges])), np.ones(
            m.size + edges.size, dtype=bool
        )


# ---------------------------------------------------------------- discrete
#
# A discrete class sets `discrete = True`, defines pmf, cdf and ccdf at integers (the
# cdfs at floor(x)) of arb arguments, the moments, and `support(params...)` -> (lo, hi) as
# Python ints (hi may be inf) of the float parameters. The inverses' references are exact
# integers found by search on the Arb cdf (discrete_inverse); the mode is found from the
# exact ratio pmf(k) / pmf(k - 1) of the float parameters (`ratio_exact`, a Fraction), so a
# tie (-> NaN, docs/design/stats.md §2.3) is decided exactly. `cdf_exact(k, params...)` (a Fraction, or None) is
# the rational cdf where a cdf can equal its target exactly (a median of 1/2).


def _arbq(q):
    """A Fraction as an arb (rounded at the current precision)."""
    return arb(q.numerator) / q.denominator


def _fl(v):
    """The float value of an exact arb parameter."""
    return float(v.mid()) if isinstance(v, arb) else float(v)


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


def _entropy_tail(pk, rho):
    """A bound on sum_{j >= 1} -q_j log q_j, q_j = pk rho^j: the tail of an entropy sum past
    a point of probability pk when every further pmf ratio is at most rho (< 1). Each pmf
    there is below q_j, and -q log q increases in q up to 1/e, so the sum of the actual
    terms is below sum q_j (-log pk - j log rho) = pk rho / (1 - rho) (-log pk)
    + pk rho (-log rho) / (1 - rho)^2. None when rho >= 1 or pk rho > 1/e."""
    if pk.is_zero():
        return arb(0)
    rho = _a(rho)
    if not (rho < 1 and pk * rho < arb(-1).exp()):
        return None
    if rho.is_zero():
        return arb(0)
    one = 1 - rho
    return (pk * rho / one * (-pk.log()) + pk * rho * (-rho.log()) / one**2).upper()


def _entropy_sum(pmf, ratio, lo, hi, centre, spread, rho_right=None, rho_left=None):
    """-sum pmf log pmf over the integers within `spread` of `centre` (pmf(a) direct, then
    pmf(k) = pmf(k - 1) ratio(k)), plus bounds on the two tails cut off (_entropy_tail with
    rho_right(b) >= every pmf(k + 1) / pmf(k), k >= b, and rho_left(a) >= every
    pmf(k - 1) / pmf(k), k <= a; either may be None for a tail it cannot bound): the range
    doubles until those bounds are below the sum's last bits."""
    while True:
        a = int(max(lo, math.floor(centre - spread)))
        b = int(min(hi, math.ceil(centre + spread)))
        pa = pmf(a)
        pk, h = pa, arb(0)
        for k in range(a, b + 1):
            if k > a:
                pk = pk * ratio(k)
            if not pk.is_zero():
                h -= pk * pk.log()
        err, ok = arb(0), True
        if b < hi:
            e = _entropy_tail(pk, rho_right(b)) if rho_right else None
            ok, err = (ok and e is not None), err + (e if e is not None else 0)
        if a > lo:
            e = _entropy_tail(pa, rho_left(a)) if rho_left else None
            ok, err = (ok and e is not None), err + (e if e is not None else 0)
        if ok and (err.is_zero() or err < abs(h) * _eps()):
            return add_error(h, err)
        if a <= lo and b >= hi:
            raise NotSettled("entropy tail")
        spread *= 2


def _mode_by_ratio(ratio, lo, hi, guess):
    """The argmax of a unimodal pmf near `guess` from the exact ratios
    ratio(k) = pmf(k) / pmf(k - 1); NaN for a tie."""
    k = int(min(max(lo, math.floor(guess)), hi))
    while k > lo and ratio(k) < 1:
        k -= 1
    while k < hi and ratio(k + 1) > 1:
        k += 1
    if (k > lo and ratio(k) == 1) or (k < hi and ratio(k + 1) == 1):
        return NAN
    return arb(k)


class Discrete(Distribution):
    discrete = True

    @staticmethod
    def cdf_exact(k, *p):
        return None

    @classmethod
    def x_sets(cls, rng):
        ps = cls._params(rng, 6000)
        x = cls._x(rng, ps)
        return (*ps, x), np.isfinite(x)


def _binom_frac(n, k, p):
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


class Binomial(Discrete):
    params = ("n", "p")

    @staticmethod
    def support(n, p):
        return 0, int(n)

    @staticmethod
    def pmf(k, n, p):
        if not k.is_integer() or k < 0 or k > n:
            return arb(0)
        if p == 0 or p == 1:
            return arb(1) if k == (0 if p == 0 else n) else arb(0)
        return ((n + 1).lgamma() - (k + 1).lgamma() - (n - k + 1).lgamma() + k * p.log() + (n - k) * (-p).log1p()).exp()

    @staticmethod
    def cdf(x, n, p):
        k = x.floor()
        if k < 0:
            return arb(0)
        if k >= n:
            return arb(1)
        return _ibeta(1 - p, n - k, k + 1, p)

    @staticmethod
    def ccdf(x, n, p):
        k = x.floor()
        if k < 0:
            return arb(1)
        if k >= n:
            return arb(0)
        return _ibeta(p, k + 1, n - k, 1 - p)

    @staticmethod
    def cdf_exact(k, n, p):
        n = int(n)
        if n > 3000:
            return None
        P = Fraction(p)
        return sum((_binom_frac(n, j, P) for j in range(0, min(k, n) + 1)), Fraction(0))

    @staticmethod
    def ratio_exact(k, n, p):
        P = Fraction(p)
        return (int(n) - k + 1) * P / (k * (1 - P))

    mean = staticmethod(lambda n, p: n * p)
    variance = staticmethod(lambda n, p: n * p * (1 - p))
    standard_deviation = staticmethod(lambda n, p: (n * p * (1 - p)).sqrt())
    skewness = staticmethod(lambda n, p: (1 - 2 * p) / (n * p * (1 - p)).sqrt())
    kurtosis = staticmethod(lambda n, p: (1 - 6 * p * (1 - p)) / (n * p * (1 - p)))

    @staticmethod
    def mode(n, p):
        nf, pf = _fl(n), _fl(p)
        if pf == 1:
            return n
        return _mode_by_ratio(lambda k: Binomial.ratio_exact(k, nf, pf), 0, int(nf), (nf + 1) * pf)

    @staticmethod
    def entropy(n, p):
        nf, pf = _fl(n), _fl(p)
        if pf == 0 or pf == 1:
            return arb(0)
        r = lambda k: (n - k + 1) * p / (k * (1 - p))
        return _entropy_sum(
            lambda k: Binomial.pmf(arb(k), n, p),
            r,
            0,
            int(nf),
            nf * pf,
            16 * math.sqrt(nf * pf * (1 - pf)) + 40,
            lambda b: r(b + 1),
            lambda a: 1 / r(a),
        )

    @staticmethod
    def _params(rng, n):
        q = n // 4
        nn = np.concatenate([np.floor(loguniform(rng, 1, 1e4, n - q)), np.floor(loguniform(rng, 1e4, 1e6, q))])
        p = np.concatenate(
            [
                rng.uniform(0, 1, n // 2),
                loguniform(rng, 1e-8, 0.5, n // 4),
                1 - loguniform(rng, 1e-8, 0.5, n - n // 2 - n // 4),
            ]
        )
        rng.shuffle(nn)
        return nn, p

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st

        nn, p = ps
        return _quantile_ints(
            rng,
            nn.size,
            lambda q, i: st.binom.ppf(q, nn[i], p[i]),
            lambda q, i: st.binom.isf(q, nn[i], p[i]),
            lambda rng, k: np.floor(rng.uniform(0, 1, k) * nn[:k]),
        )

    @classmethod
    def param_sets(cls, rng):
        nn, p = cls._params(rng, 800)
        p = np.concatenate([p, [0.5, 0.25, 0.2113248654051871, 0.7886751345948129, 0.0, 1.0]])
        nn = np.concatenate([nn, [9.0, 7.0, 30.0, 30.0, 10.0, 10.0]])
        return (nn, p), np.ones(nn.size, dtype=bool)


class Bernoulli(Discrete):
    params = ("p",)
    support = staticmethod(lambda p: (0, 1))
    pmf = staticmethod(lambda k, p: 1 - p if k == 0 else (p if k == 1 else arb(0)))
    logpmf = staticmethod(lambda k, p: (-p).log1p() if k == 0 else (p.log() if k == 1 else -INF))
    cdf = staticmethod(lambda x, p: arb(0) if x < 0 else (1 - p if x < 1 else arb(1)))
    ccdf = staticmethod(lambda x, p: arb(1) if x < 0 else (p if x < 1 else arb(0)))
    cdf_exact = staticmethod(lambda k, p: Fraction(0) if k < 0 else (1 - Fraction(p) if k < 1 else Fraction(1)))
    mean = staticmethod(lambda p: p)
    variance = staticmethod(lambda p: p * (1 - p))
    standard_deviation = staticmethod(lambda p: (p * (1 - p)).sqrt())
    skewness = staticmethod(lambda p: (1 - 2 * p) / (p * (1 - p)).sqrt())
    kurtosis = staticmethod(lambda p: (1 - 6 * p * (1 - p)) / (p * (1 - p)))
    mode = staticmethod(lambda p: arb(0) if p < 0.5 else (arb(1) if p > 0.5 else NAN))
    entropy = staticmethod(lambda p: -(p * p.log() + (1 - p) * (-p).log1p()) if 0 < p < 1 else arb(0))

    @staticmethod
    def _params(rng, n):
        return (
            np.concatenate(
                [
                    rng.uniform(0, 1, n // 2),
                    loguniform(rng, 1e-300, 0.5, n // 4),
                    1 - loguniform(rng, 2**-53, 0.5, n - n // 2 - n // 4),
                ]
            ),
        )

    @staticmethod
    def _x(rng, ps):
        return rng.choice([-1.0, 0.0, 0.5, 1.0, 2.0], ps[0].size)


class Poisson(Discrete):
    params = ("mu",)
    support = staticmethod(lambda mu: (0, INF))

    @staticmethod
    def pmf(k, mu):
        if not k.is_integer() or k < 0:
            return arb(0)
        return (k * mu.log() - mu - (k + 1).lgamma()).exp()

    cdf = staticmethod(lambda x, mu: arb(0) if x < 0 else _gamma_q(x.floor() + 1, mu))
    ccdf = staticmethod(lambda x, mu: arb(1) if x < 0 else _gamma_p(x.floor() + 1, mu))
    mean = variance = staticmethod(lambda mu: mu)
    standard_deviation = staticmethod(lambda mu: mu.sqrt())
    skewness = staticmethod(lambda mu: 1 / mu.sqrt())
    kurtosis = staticmethod(lambda mu: 1 / mu)
    ratio_exact = staticmethod(lambda k, mu: Fraction(mu) / k)
    mode = staticmethod(lambda mu: _mode_by_ratio(lambda k: Fraction(_fl(mu)) / k, 0, INF, _fl(mu)))

    @staticmethod
    def entropy(mu):
        m = _fl(mu)
        return _entropy_sum(
            lambda k: Poisson.pmf(arb(k), mu),
            lambda k: mu / k,
            0,
            INF,
            m,
            16 * math.sqrt(m) + 40,
            lambda b: mu / (b + 1),
            lambda a: a / mu,
        )

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([loguniform(rng, 1e-3, 1e4, n - n // 8), loguniform(rng, 1e4, 1e6, n // 8)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st

        (mu,) = ps
        return _quantile_ints(
            rng,
            mu.size,
            lambda q, i: st.poisson.ppf(q, mu[i]),
            lambda q, i: st.poisson.isf(q, mu[i]),
            lambda rng, k: np.floor(loguniform(rng, 1, 1e6, k)),
        )

    @classmethod
    def param_sets(cls, rng):
        (mu,) = cls._params(rng, 600)
        return (np.concatenate([mu, [1.0, 2.0, 3.0, 10.0, 0.5]]),), np.ones(mu.size + 5, dtype=bool)


class Geometric(Discrete):
    params = ("p",)
    support = staticmethod(lambda p: (1, INF))
    pmf = staticmethod(lambda k, p: ((k - 1) * (-p).log1p()).exp() * p if k.is_integer() and k >= 1 else arb(0))
    cdf = staticmethod(lambda x, p: arb(0) if x < 1 else -(x.floor() * (-p).log1p()).expm1())
    ccdf = staticmethod(lambda x, p: arb(1) if x < 1 else (x.floor() * (-p).log1p()).exp())
    cdf_exact = staticmethod(lambda k, p: (Fraction(0) if k < 1 else 1 - (1 - Fraction(p)) ** k) if k <= 4000 else None)
    mean = staticmethod(lambda p: 1 / p)
    variance = staticmethod(lambda p: (1 - p) / p**2)
    standard_deviation = staticmethod(lambda p: (1 - p).sqrt() / p)
    skewness = staticmethod(lambda p: (2 - p) / (1 - p).sqrt())
    kurtosis = staticmethod(lambda p: 6 + p**2 / (1 - p))
    mode = staticmethod(lambda p: arb(1))
    entropy = staticmethod(lambda p: -((1 - p) * (-p).log1p() + p * p.log()) / p)

    @staticmethod
    def _params(rng, n):
        return (
            np.concatenate(
                [
                    rng.uniform(0.001, 0.999, n // 2),
                    loguniform(rng, 1e-9, 0.5, n // 4),
                    1 - loguniform(rng, 1e-9, 0.5, n - n // 2 - n // 4),
                ]
            ),
        )

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st

        (p,) = ps
        return _quantile_ints(
            rng,
            p.size,
            lambda q, i: st.geom.ppf(q, p[i]),
            lambda q, i: st.geom.isf(q, p[i]),
            lambda rng, k: np.floor(loguniform(rng, 1, 1e12, k)),
        )


class NegativeBinomial(Discrete):
    params = ("n", "p")
    support = staticmethod(lambda n, p: (0, INF))

    @staticmethod
    def pmf(k, n, p):
        if not k.is_integer() or k < 0:
            return arb(0)
        return ((k + n).lgamma() - n.lgamma() - (k + 1).lgamma() + n * p.log() + k * (-p).log1p()).exp()

    cdf = staticmethod(lambda x, n, p: arb(0) if x < 0 else _ibeta(p, n, x.floor() + 1, 1 - p))
    ccdf = staticmethod(lambda x, n, p: arb(1) if x < 0 else _ibeta(1 - p, x.floor() + 1, n, p))

    @staticmethod
    def cdf_exact(k, n, p):
        if n != int(n) or k > 4000:
            return None
        n, P = int(n), Fraction(p)
        return (
            sum((math.comb(j + n - 1, j) * P**n * (1 - P) ** j for j in range(0, k + 1)), Fraction(0))
            if k >= 0
            else Fraction(0)
        )

    ratio_exact = staticmethod(lambda k, n, p: (k - 1 + Fraction(n)) * (1 - Fraction(p)) / k)
    mean = staticmethod(lambda n, p: n * (1 - p) / p)
    variance = staticmethod(lambda n, p: n * (1 - p) / p**2)
    standard_deviation = staticmethod(lambda n, p: (n * (1 - p)).sqrt() / p)
    skewness = staticmethod(lambda n, p: (2 - p) / (n * (1 - p)).sqrt())
    kurtosis = staticmethod(lambda n, p: 6 / n + p**2 / (n * (1 - p)))

    @staticmethod
    def mode(n, p):
        nf, pf = _fl(n), _fl(p)
        return _mode_by_ratio(
            lambda k: NegativeBinomial.ratio_exact(k, nf, pf), 0, INF, (nf - 1) * (1 - pf) / pf if nf > 1 else 0
        )

    @staticmethod
    def entropy(n, p):
        # the range: 16 sd and, for a small n (mode 0, the pmf ~ (1 - p)^k), 40 / p more for
        # the exponential tail
        nf, pf = _fl(n), _fl(p)
        r = lambda k: (k - 1 + n) * (1 - p) / k
        right = (
            (lambda b: r(b + 1)) if nf >= 1 else (lambda b: 1 - p)
        )  # the ratios fall to 1 - p (n >= 1) or rise to it
        left = (lambda a: 1 / r(a)) if nf >= 1 else None
        return _entropy_sum(
            lambda k: NegativeBinomial.pmf(arb(k), n, p),
            r,
            0,
            INF,
            nf * (1 - pf) / pf,
            (16 * math.sqrt(nf * (1 - pf)) + 40) / pf + 40,
            right,
            left,
        )

    @staticmethod
    def _params(rng, n):
        nn = np.concatenate([loguniform(rng, 0.1, 1e4, n - n // 8), np.floor(loguniform(rng, 1, 100, n // 8))])
        p = np.concatenate(
            [
                rng.uniform(0.01, 0.99, n // 2),
                loguniform(rng, 1e-4, 0.5, n // 4),
                1 - loguniform(rng, 1e-6, 0.5, n - n // 2 - n // 4),
            ]
        )
        rng.shuffle(nn)
        return nn, p

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st

        nn, p = ps
        return _quantile_ints(
            rng,
            nn.size,
            lambda q, i: st.nbinom.ppf(q, nn[i], p[i]),
            lambda q, i: st.nbinom.isf(q, nn[i], p[i]),
            lambda rng, k: np.floor(loguniform(rng, 1, 1e8, k)),
        )

    @classmethod
    def param_sets(cls, rng):
        nn, p = cls._params(rng, 600)
        return (np.concatenate([nn, [3.0, 5.0]]), np.concatenate([p, [0.5, 0.2]])), np.ones(nn.size + 2, dtype=bool)


class Hypergeometric(Discrete):
    """Every value is a rational of the integer parameters: computed exactly."""

    params = ("total", "successes", "draws")

    @staticmethod
    def support(N, K, n):
        N, K, n = int(N), int(K), int(n)
        return max(0, n - (N - K)), min(n, K)

    @staticmethod
    def _counts(N, K, n):
        """C(K, j) C(N - K, n - j) for j over the support, and C(N, n)."""
        N, K, n = (_int(v) for v in (N, K, n))
        lo, hi = Hypergeometric.support(N, K, n)
        return lo, [math.comb(K, j) * math.comb(N - K, n - j) for j in range(lo, hi + 1)], math.comb(N, n)

    @staticmethod
    def pmf(k, N, K, n):
        lo, hi = Hypergeometric.support(_fl(N), _fl(K), _fl(n))
        if not k.is_integer() or k < lo or k > hi:
            return arb(0)
        k, N, K, n = (_int(v) for v in (k, N, K, n))
        return _arbq(Fraction(math.comb(K, k) * math.comb(N - K, n - k), math.comb(N, n)))

    @staticmethod
    def cdf_exact(k, N, K, n):
        lo, c, tot = Hypergeometric._counts(N, K, n)
        return Fraction(sum(c[: max(0, min(len(c), k - lo + 1))]), tot)

    @staticmethod
    def cdf(x, N, K, n):
        return _arbq(Hypergeometric.cdf_exact(_int(x.floor()), *(_fl(v) for v in (N, K, n))))

    @staticmethod
    def ccdf(x, N, K, n):
        return _arbq(1 - Hypergeometric.cdf_exact(_int(x.floor()), *(_fl(v) for v in (N, K, n))))

    @staticmethod
    def _central(N, K, n, m):
        """The m-th central moment, exactly."""
        lo, c, tot = Hypergeometric._counts(N, K, n)
        mu = Fraction(_int(n) * _int(K), _int(N))
        return sum(((lo + i - mu) ** m * ci for i, ci in enumerate(c)), Fraction(0)) / tot

    @staticmethod
    def _standardized(N, K, n, m):
        c2 = Hypergeometric._central(N, K, n, 2)
        if c2 == 0:  # one point: the ratio is undefined
            raise NotSettled("degenerate")
        cm = Hypergeometric._central(N, K, n, m)
        if m % 2 == 0:  # rational: exact (an excess kurtosis of exactly 0 would never settle)
            return cm / c2 ** (m // 2)
        return _arbq(cm) / _arbq(c2) ** (arb(m) / 2)

    ratio_exact = staticmethod(
        lambda k, N, K, n: Fraction((int(K) - k + 1) * (int(n) - k + 1), k * (int(N) - int(K) - int(n) + k))
    )
    mean = staticmethod(lambda N, K, n: n * K / N)
    variance = staticmethod(lambda N, K, n: _arbq(Hypergeometric._central(N, K, n, 2)))
    standard_deviation = staticmethod(lambda N, K, n: _arbq(Hypergeometric._central(N, K, n, 2)).sqrt())
    skewness = staticmethod(lambda N, K, n: Hypergeometric._standardized(N, K, n, 3))
    kurtosis = staticmethod(lambda N, K, n: _arbq(Hypergeometric._standardized(N, K, n, 4) - 3))

    @staticmethod
    def mode(N, K, n):
        f = [_fl(v) for v in (N, K, n)]
        lo, hi = Hypergeometric.support(*f)
        return _mode_by_ratio(lambda k: Hypergeometric.ratio_exact(k, *f), lo, hi, f[2] * f[1] / f[0])

    @staticmethod
    def entropy(N, K, n):
        lo, c, tot = Hypergeometric._counts(N, K, n)
        h = arb(0)
        for ci in c:
            if ci:
                pk = _arbq(Fraction(ci, tot))
                h -= pk * pk.log()
        return h

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
    support = staticmethod(lambda lo, hi: (int(lo), int(hi) - 1))
    pmf = staticmethod(lambda k, lo, hi: 1 / (hi - lo) if k.is_integer() and lo <= k < hi else arb(0))
    cdf = staticmethod(
        lambda x, lo, hi: (
            arb(0) if x.floor() < lo else (arb(1) if x.floor() >= hi - 1 else (x.floor() - lo + 1) / (hi - lo))
        )
    )
    ccdf = staticmethod(
        lambda x, lo, hi: (
            arb(1) if x.floor() < lo else (arb(0) if x.floor() >= hi - 1 else (hi - 1 - x.floor()) / (hi - lo))
        )
    )
    cdf_exact = staticmethod(
        lambda k, lo, hi: Fraction(min(max(k - int(lo) + 1, 0), int(hi) - int(lo)), int(hi) - int(lo))
    )
    mean = staticmethod(lambda lo, hi: (lo + hi - 1) / 2)
    variance = staticmethod(lambda lo, hi: ((hi - lo) ** 2 - 1) / 12)
    standard_deviation = staticmethod(lambda lo, hi: (((hi - lo) ** 2 - 1) / 12).sqrt())
    skewness = staticmethod(lambda lo, hi: arb(0))
    kurtosis = staticmethod(lambda lo, hi: -6 * ((hi - lo) ** 2 + 1) / (5 * ((hi - lo) ** 2 - 1)))
    mode = staticmethod(lambda lo, hi: NAN)
    entropy = staticmethod(lambda lo, hi: (hi - lo).log())

    @staticmethod
    def _params(rng, n):
        lo = np.floor(
            np.concatenate([rng.uniform(-100, 100, n // 2), loguniform(rng, 1, 1e12, n - n // 2, signed=True)])
        )
        w = np.floor(loguniform(rng, 2, 1e9, n))
        return lo, lo + w

    @staticmethod
    def _x(rng, ps):
        lo, hi = ps
        x = np.floor(lo + rng.uniform(-0.05, 1.05, lo.size) * (hi - lo))
        return x + 0.5 * (rng.random(x.size) < 0.05)


class BetaBinomial(Discrete):
    params = ("n", "a", "b")
    support = staticmethod(lambda n, a, b: (0, int(n)))

    @staticmethod
    def pmf(k, n, a, b):
        if not k.is_integer() or k < 0 or k > n:
            return arb(0)
        return (
            (n + 1).lgamma() - (k + 1).lgamma() - (n - k + 1).lgamma() + _lbeta(k + a, n - k + b) - _lbeta(a, b)
        ).exp()

    @staticmethod
    def _ratio(j, n, a, b):
        """pmf(j) / pmf(j - 1)."""
        return (n - j + 1) * (j - 1 + a) / (j * (n - j + b))

    @staticmethod
    def _sum(j0, j1, n, a, b):
        """pmf(j0) + ... + pmf(j1), by the ratio from pmf(j0)."""
        pk = BetaBinomial.pmf(arb(j0), n, a, b)
        s = pk
        for j in range(j0 + 1, j1 + 1):
            pk = pk * BetaBinomial._ratio(j, n, a, b)
            s += pk
        return s

    @staticmethod
    def cdf(x, n, a, b):
        k = _int(x.floor())
        return arb(0) if k < 0 else (arb(1) if k >= n else BetaBinomial._sum(0, k, n, a, b))

    @staticmethod
    def ccdf(x, n, a, b):
        k = _int(x.floor())
        return arb(1) if k < 0 else (arb(0) if k >= n else BetaBinomial._sum(k + 1, _int(n), n, a, b))

    @staticmethod
    def ratio_exact(k, n, a, b):
        A, B = Fraction(a), Fraction(b)
        return (int(n) - k + 1) * (k - 1 + A) / (k * (int(n) - k + B))

    @staticmethod
    def cdf_exact(k, n, a, b):
        n = int(n)
        if n > 200:
            return None
        A, B = Fraction(a), Fraction(b)
        p0 = Fraction(1)
        for i in range(n):  # pmf(0) = (b)_n / (a + b)_n
            p0 *= (B + i) / (A + B + i)
        s, pk = p0, p0
        for j in range(1, min(k, n) + 1):
            pk *= BetaBinomial.ratio_exact(j, n, a, b)
            s += pk
        return s if k >= 0 else Fraction(0)

    @staticmethod
    def _central(n, a, b, m):
        mu = n * a / (a + b)
        pk = BetaBinomial.pmf(arb(0), n, a, b)
        s = (0 - mu) ** m * pk
        for j in range(1, _int(n) + 1):
            pk = pk * BetaBinomial._ratio(j, n, a, b)
            s += (j - mu) ** m * pk
        return s

    mean = staticmethod(lambda n, a, b: n * a / (a + b))
    variance = staticmethod(lambda n, a, b: n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1)))
    standard_deviation = staticmethod(lambda n, a, b: (n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1))).sqrt())
    # a = b is symmetric about n / 2: the skewness is exactly 0
    skewness = staticmethod(
        lambda n, a, b: (
            arb(0) if a == b else BetaBinomial._central(n, a, b, 3) / BetaBinomial._central(n, a, b, 2) ** arb(1.5)
        )
    )
    kurtosis = staticmethod(
        lambda n, a, b: BetaBinomial._central(n, a, b, 4) / BetaBinomial._central(n, a, b, 2) ** 2 - 3
    )

    @staticmethod
    def mode(n, a, b):
        """The global argmax (the pmf may be U-shaped): the largest ball, and any other whose
        ball reaches it compared exactly (the product of the exact ratios between them)."""
        nf, af, bf = _int(n), _fl(a), _fl(b)
        vals, pk = [BetaBinomial.pmf(arb(0), n, a, b)], None
        pk = vals[0]
        for j in range(1, nf + 1):
            pk = pk * BetaBinomial._ratio(j, n, a, b)
            vals.append(pk)
        top = max(range(nf + 1), key=lambda j: vals[j].mid())
        cand = [j for j in range(nf + 1) if j != top and vals[j].upper() >= vals[top].lower()]
        best, tie = top, False
        for j in sorted(cand):
            lo_, hi_ = min(best, j), max(best, j)
            r = Fraction(1)
            for i in range(lo_ + 1, hi_ + 1):
                r *= BetaBinomial.ratio_exact(i, nf, af, bf)  # pmf(hi_) / pmf(lo_)
            if r == 1:
                tie = True
            elif (r > 1) == (j > best):
                best, tie = j, False
        return NAN if tie else arb(best)

    @staticmethod
    def entropy(n, a, b):
        pk = BetaBinomial.pmf(arb(0), n, a, b)
        h = -pk * pk.log()
        for j in range(1, _int(n) + 1):
            pk = pk * BetaBinomial._ratio(j, n, a, b)
            h -= pk * pk.log()
        return h

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
        return (
            np.concatenate([nn, [10.0, 10.0, 5.0]]),
            np.concatenate([a, [1.0, 0.5, 2.0]]),
            np.concatenate([b, [1.0, 0.5, 3.0]]),
        ), np.ones(nn.size + 3, dtype=bool)


def _logser_l(p):
    return -(-p).log1p()


class LogSeries(Discrete):
    params = ("p",)
    support = staticmethod(lambda p: (1, INF))
    pmf = staticmethod(lambda k, p: (k * p.log()).exp() / (k * _logser_l(p)) if k.is_integer() and k >= 1 else arb(0))

    @staticmethod
    def ccdf(x, p):
        """p^(k+1) Phi(p, 1, k + 1) / L (Arb's Lerch transcendent)."""
        k = x.floor()
        if k < 1:
            return arb(1)
        phi = acb(p).lerch_phi(acb(1), acb(k + 1)).real
        return ((k + 1) * p.log()).exp() * phi / _logser_l(p)

    @staticmethod
    def cdf(x, p):
        """1 - ccdf: the cdf is at least pmf(1) = p / L (above 0.07 for every p < 1 - 1e-16),
        so the difference loses a few bits at most."""
        if x.floor() < 1:
            return arb(0)
        return 1 - LogSeries.ccdf(x, p)

    @staticmethod
    def _raw(p, r):
        L, w = _logser_l(p), 1 - p
        A = [arb(1), arb(1), 1 + p, 1 + 4 * p + p * p][r - 1]
        return p * A / (L * w**r)

    mean = staticmethod(lambda p: LogSeries._raw(p, 1))
    variance = staticmethod(lambda p: LogSeries._raw(p, 2) - LogSeries._raw(p, 1) ** 2)
    standard_deviation = staticmethod(lambda p: (LogSeries._raw(p, 2) - LogSeries._raw(p, 1) ** 2).sqrt())

    @staticmethod
    def _central(p, m):
        e = [LogSeries._raw(p, r) for r in (1, 2, 3, 4)]
        mu = e[0]
        if m == 2:
            return e[1] - mu**2
        if m == 3:
            return e[2] - 3 * mu * e[1] + 2 * mu**3
        return e[3] - 4 * mu * e[2] + 6 * mu**2 * e[1] - 3 * mu**4

    skewness = staticmethod(lambda p: LogSeries._central(p, 3) / LogSeries._central(p, 2) ** arb(1.5))
    kurtosis = staticmethod(lambda p: LogSeries._central(p, 4) / LogSeries._central(p, 2) ** 2 - 3)
    mode = staticmethod(lambda p: arb(1))

    @staticmethod
    def entropy(p):
        """log L - log(p) E[K] + (1/L) sum_k p^k log(k) / k, the sum being -d/ds Li_s(p) at
        s = 1 (Arb's polylog as a power series in s)."""
        L = _logser_l(p)
        dli = acb_series.polylog(acb_series([1, 1], prec=2), acb(p)).coeffs()[1].real
        return L.log() - p.log() * LogSeries._raw(p, 1) - dli / L

    @staticmethod
    def _params(rng, n):
        return (
            np.concatenate(
                [
                    rng.uniform(0.001, 0.999, n // 2),
                    loguniform(rng, 1e-8, 0.5, n // 4),
                    1 - loguniform(rng, 1e-6, 0.5, n - n // 2 - n // 4),
                ]
            ),
        )

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st  # noqa: F401

        (p,) = ps
        import stardust as sd  # scipy's logser.ppf is slow in the far tail: our own icdf as the sampler

        return _quantile_ints(
            rng,
            p.size,
            lambda q, i: np.asarray(sd.stats.LogSeries(p=sd.asarray(p[i])).icdf(sd.asarray(q))),
            lambda q, i: np.asarray(sd.stats.LogSeries(p=sd.asarray(p[i])).iccdf(sd.asarray(q))),
            lambda rng, k: np.floor(loguniform(rng, 1, 1e7, k)),
        )

    @classmethod
    def param_sets(cls, rng):
        (p,) = cls._params(rng, 600)
        return (p,), np.ones(p.size, dtype=bool)


class Zipf(Discrete):
    params = ("a",)
    support = staticmethod(lambda a: (1, INF))
    pmf = staticmethod(lambda k, a: (-a * k.log()).exp() / a.zeta() if k.is_integer() and k >= 1 else arb(0))
    ccdf = staticmethod(lambda x, a: arb(1) if x.floor() < 1 else a.zeta(x.floor() + 1) / a.zeta())
    # 1 - ccdf: the cdf is at least pmf(1) = 1 / zeta(a) (above 1e-3 for a > 1.001)
    cdf = staticmethod(lambda x, a: arb(0) if x.floor() < 1 else 1 - a.zeta(x.floor() + 1) / a.zeta())

    @staticmethod
    def _raw(a, r):
        return (a - r).zeta() / a.zeta()

    mean = staticmethod(lambda a: Zipf._raw(a, 1) if a > 2 else INF)

    @staticmethod
    def _central(a, m):
        e = [Zipf._raw(a, r) for r in (1, 2, 3, 4)[:m]]
        mu = e[0]
        if m == 2:
            return e[1] - mu**2
        if m == 3:
            return e[2] - 3 * mu * e[1] + 2 * mu**3
        return e[3] - 4 * mu * e[2] + 6 * mu**2 * e[1] - 3 * mu**4

    variance = staticmethod(lambda a: Zipf._central(a, 2) if a > 3 else (INF if a > 2 else NAN))
    standard_deviation = staticmethod(lambda a: Zipf._central(a, 2).sqrt() if a > 3 else (INF if a > 2 else NAN))
    skewness = staticmethod(
        lambda a: Zipf._central(a, 3) / Zipf._central(a, 2) ** arb(1.5) if a > 4 else (INF if a > 3 else NAN)
    )
    kurtosis = staticmethod(
        lambda a: Zipf._central(a, 4) / Zipf._central(a, 2) ** 2 - 3 if a > 5 else (INF if a > 3 else NAN)
    )
    mode = staticmethod(lambda a: arb(1))

    @staticmethod
    def entropy(a):
        """log zeta(a) - a zeta'(a) / zeta(a) (Arb's zeta as a power series in s)."""
        z0, z1 = arb_series([a, 1], prec=2).zeta().coeffs()[:2]
        return z0.log() - a * z1 / z0

    @staticmethod
    def _params(rng, n):
        return (np.concatenate([1 + loguniform(rng, 1e-3, 1, n // 2), loguniform(rng, 2, 60, n - n // 2)]),)

    @staticmethod
    def _x(rng, ps):
        import scipy.stats as st  # noqa: F401

        (a,) = ps
        import stardust as sd  # scipy's zipf.ppf runs out of memory for a near 1 and a tiny q

        return _quantile_ints(
            rng,
            a.size,
            lambda q, i: np.asarray(sd.stats.Zipf(a=sd.asarray(a[i])).icdf(sd.asarray(q))),
            lambda q, i: np.asarray(sd.stats.Zipf(a=sd.asarray(a[i])).iccdf(sd.asarray(q))),
            lambda rng, k: np.floor(loguniform(rng, 1, 1e12, k)),
        )

    @classmethod
    def param_sets(cls, rng):
        (a,) = cls._params(rng, 600)
        return (a,), np.ones(a.size, dtype=bool)


DISTRIBUTIONS = {
    "normal": Normal,
    "exponential": Exponential,
    "uniform": Uniform,
    "laplace": Laplace,
    "cauchy": Cauchy,
    "lognormal": LogNormal,
    "weibull": Weibull,
    "gamma": Gamma,
    "chisquared": ChiSquared,
    "beta": Beta,
    "studentt": StudentT,
    "f": F,
    "binomial": Binomial,
    "bernoulli": Bernoulli,
    "poisson": Poisson,
    "geometric": Geometric,
    "negativebinomial": NegativeBinomial,
    "hypergeometric": Hypergeometric,
    "discreteuniform": DiscreteUniform,
    "betabinomial": BetaBinomial,
    "logseries": LogSeries,
    "zipf": Zipf,
    "logistic": Logistic,
    "gumbel": Gumbel,
    "gev": GEV,
    "pareto": Pareto,
    "rayleigh": Rayleigh,
    "inversegamma": InverseGamma,
    "noncentralchisquared": NoncentralChiSquared,
    "noncentralt": NoncentralT,
    "noncentralf": NoncentralF,
}
DISCRETE_INPUT_METHODS = ("pmf", "logpmf") + INPUT_METHODS[2:]


# ---------------------------------------------------------------- method definitions


def _log_cdf(D, x, p, upper):
    """log F(x) (upper: log(1 - F)): the log of the smaller side, log1p(-other) for the larger."""
    lo, hi = (D.ccdf, D.cdf) if upper else (D.cdf, D.ccdf)
    v = lo(x, *p)
    if isinstance(v, float) or v <= 0.5:
        return _log(v)
    return (-hi(x, *p)).log1p()


def method_value(D, m, args):
    """The reference value of a non-inverse method at arb args (params..., x)."""
    if m in PARAM_METHODS:
        return getattr(D, m)(*args)
    *p, x = args
    if not x.is_finite():  # x = +-inf: beyond every support
        up = x > 0
        return {
            "pdf": arb(0),
            "pmf": arb(0),
            "logpdf": -INF,
            "logpmf": -INF,
            "cdf": arb(int(up)),
            "ccdf": arb(int(not up)),
            "logcdf": arb(0) if up else -INF,
            "logccdf": -INF if up else arb(0),
        }[m]
    if m in ("pdf", "pmf"):
        return D.pmf(x, *p) if m == "pmf" else D.pdf(x, *p)
    if m == "logpmf":
        return D.logpmf(x, *p) if hasattr(D, "logpmf") else _log(D.pmf(x, *p))
    if m == "logpdf":
        return D.logpdf(x, *p) if hasattr(D, "logpdf") else _log(D.pdf(x, *p))
    if m == "cdf":
        return D.cdf(x, *p)
    if m == "ccdf":
        return D.ccdf(x, *p)
    if m == "logcdf":
        return _log_cdf(D, x, p, False)
    if m == "logccdf":
        return _log_cdf(D, x, p, True)
    raise KeyError(m)


def _closed_inverse(D, m, p, t):
    """x from the distribution's closed-form inverses (icdf_mp, iccdf_mp; each is called on
    the small side, p <= 1/2), or None."""
    if not hasattr(D, "icdf_mp"):
        return None
    lower = lambda v: D.icdf_mp(v, *p) if v <= 0.5 else D.iccdf_mp(1 - v, *p)
    upper = lambda v: D.iccdf_mp(v, *p) if v <= 0.5 else D.icdf_mp(1 - v, *p)
    if m == "icdf":
        return lower(t)
    if m == "iccdf":
        return upper(t)
    small = t < -LN2()  # e^t is the small side
    if m == "ilogcdf":
        return D.icdf_mp(t.exp(), *p) if small else D.iccdf_mp(-t.expm1(), *p)
    return D.iccdf_mp(t.exp(), *p) if small else D.icdf_mp(-t.expm1(), *p)


def _side(m, t):
    """(upper, log target): the side solved for method m at its input t (float) and the log
    of its target as a function of the precision: the smaller probability's side, where the
    target is exact (1 - t for t > 1/2; log(1 - e^t) for t > -log 2)."""
    if m in ("icdf", "iccdf"):
        upper = m == "iccdf"
        if t > 0.5:
            return not upper, lambda: (1 - arb(t)).log()
        return upper, lambda: arb(t).log()
    upper = m == "ilogccdf"
    if t > -math.log(2):
        return not upper, lambda: (-arb(t).expm1()).log()
    return upper, lambda: arb(t)


def _newton_root(D, p, upper, logtarget, x0):
    """The root x of log side(x) = log target (side the cdf, or the ccdf when upper), as an
    arb point certified to 2^-110 relative by the sign change of log side(x) - log target
    at x (1 -+ 2^-110) (arbref.certify_root): Newton on the midpoints from our own x0, in
    u = log x for a positive support and u = logit x for (0, 1), at doubling precision."""
    side = D.ccdf if upper else D.cdf
    sign = -1 if upper else 1
    unit, positive = getattr(D, "unit", False), getattr(D, "positive", False)
    if unit:
        to_x = lambda u: 1 / (1 + (-u).exp())
        dxdu = lambda x: x * (1 - x)
        u = (arb(x0) / (1 - arb(x0))).log()
    elif positive:
        to_x = lambda u: u.exp()
        dxdu = lambda x: x
        u = arb(x0).log()
    else:
        to_x = lambda u: u
        dxdu = lambda x: arb(1)
        u = arb(x0)
    g = lambda x: _log(side(x, *p)) - logtarget()
    prec = ar.START_PREC
    while prec <= ar.MAX_PREC:
        with precision(prec):
            try:
                for _ in range(200):
                    x = to_x(u).mid()
                    F = side(x, *p)
                    f = D.pdf(x, *p)
                    gx = (_log(F) - logtarget()).mid()
                    d = (sign * f * dxdu(x) / F).mid()
                    if not (gx.is_finite() and d.is_finite()) or d.is_zero():
                        raise NotSettled("newton: no slope")
                    du = gx / d
                    # damped far from the root (a start from the centre, where our root is 0 or inf)
                    cap = arb(2) if (unit or positive) else max(arb(1), abs(u))
                    if abs(du) > cap:
                        du = cap * du.sgn()
                    u = (u - du).mid()
                    if abs(du) <= arb(2) ** (-prec + 12) * (abs(u) if not (unit or positive) else max(arb(1), abs(u))):
                        break
                x = to_x(u).mid()
                return ar.certify_root(g, x, 2.0**-ar.TARGET_BITS, increasing=not upper)
            except NotSettled, ZeroDivisionError, ValueError, ArithmeticError:
                pass
        prec *= 2
    raise NotSettled("root")


def _beyond_range(D, m, p, t, x0):
    """Our root is 0 or +-inf: accept it as the reference when the true root lies beyond the
    float64 range (it then rounds to x0), checked on the side solved: the target log
    probability against the side's log at the range's edge (half the smallest subnormal
    for 0, the largest float64 for +-inf)."""
    if np.isnan(x0):
        raise ValueError("NaN start")
    upper, logtarget = _side(m, t)
    if x0 == 0:
        edge = lambda: arb(2) ** -1075  # half the smallest subnormal: a root below rounds to 0
    elif getattr(D, "unit", False) and x0 == 1:
        edge = lambda: 1 - arb(2) ** -54  # half an ULP below 1: a root above rounds to 1
    else:
        edge = lambda: arb(float(np.finfo(np.float64).max)) * (1 if x0 > 0 else -1)
    if upper and x0 > 0 and hasattr(D, "log_ccdf_bound"):  # a cheap bound decides it first
        p_ = [arb(v) for v in p]
        if D.log_ccdf_bound(edge(), *p_) < logtarget():
            raise ValueError("root inside the range")
    # the lower side increases with x, the upper side decreases: the root is past the low
    # edge (0 or -inf) when the side there is already above the target (lower) or still
    # below it (upper), and past the high edge the other way round
    side = D.ccdf if upper else D.cdf
    s = ar.sign(lambda: _log(side(_a(edge()), *[arb(v) for v in p])) - logtarget())
    below = x0 <= 0
    if (s > 0) != (below == (not upper)):
        raise ValueError("root inside the range")
    return x0


def inverse_value(D, m, args, x0):
    """(x, kappa) of an inverse method at float args (params..., p or logp): x an arb point
    or a float. kappa is the relative condition number in the method's own input:
    |p / (x f)| for icdf and iccdf, |logp p / (x f)| (p = e^logp) for ilogcdf and ilogccdf."""
    *pf, t = args
    if hasattr(D, "icdf_mp"):
        x = rigorous(lambda: _closed_inverse(D, m, [arb(v) for v in pf], arb(t)))
    elif x0 == 0 or not np.isfinite(x0) or (getattr(D, "unit", False) and x0 == 1):
        try:
            return _beyond_range(D, m, pf, t, x0), 1.0
        except ValueError, NotSettled, ArithmeticError:
            # our 0 or inf is wrong (the root is inside the range), or the side cannot be
            # evaluated at the range's edge: the certified root from the distribution's centre
            if np.isnan(x0) or not hasattr(D, "start"):
                raise
            upper, logtarget = _side(m, t)
            x = _newton_root(D, [arb(v) for v in pf], upper, logtarget, D.start(*pf))
    else:
        upper, logtarget = _side(m, t)
        x = _newton_root(D, [arb(v) for v in pf], upper, logtarget, x0)
    return x, _kappa(D, m, pf, t, x)


def _kappa(D, m, pf, t, x):
    """The condition number at the root x (a float of a few correct digits: the bound's
    scale), at doubling precision where x sits within the working precision of a support
    edge. A root within 2^-2048 of a support edge (GEV's ilogccdf at t = -7e9: x is
    mu + s/c to e^(-c 7e9)) gets 1: the density there cannot be evaluated, and at those
    edges kappa -> 0 (|t| e^t / (x f) ~ s q^c / x for q = e^t, f ~ q^(1 - c) / s)."""
    prec = ar.START_PREC
    while prec <= 2048:
        with precision(prec):
            try:
                p = [arb(v) for v in pf]
                xx = x if isinstance(x, arb) else arb(x)
                if hasattr(D, "icdf_mp") and isinstance(x, arb):  # the closed form again, at this precision
                    xx = _closed_inverse(D, m, p, arb(t))
                if not xx.is_finite() or xx.is_zero():
                    return 1.0
                f = D.pdf(xx, *p)
                if isinstance(f, float) or f.is_zero():
                    return 1.0
                k = arb(t) / (f * abs(xx)) if m in ("icdf", "iccdf") else abs(arb(t)) * arb(t).exp() / (f * abs(xx))
                if k.is_finite() and k.rel_accuracy_bits() > 20:
                    return abs(float(k.mid()))
            except NotSettled, ZeroDivisionError, ValueError:
                pass
        prec *= 2
    return 1.0


# The discrete inverses: each comparison of the cdf (or ccdf) at an integer with the target
# is decided by arbref.sign (a ball that excludes equality, at doubling precision), or,
# where the two may be exactly equal (a median at a cdf of exactly 1/2), exactly by the
# distribution's rational cdf_exact.


def _smallest_int(lo, hi, k0, pred):
    """The smallest integer k in [lo, hi] with pred(k) (pred monotone, true at hi)."""
    k = k0 if (k0 is not None and math.isfinite(k0)) else lo
    k = int(min(max(lo, math.floor(k)), hi))
    if pred(k):
        if k == lo or not pred(k - 1):
            return k
        b, step = k, 1
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
        a, step = k, 1
        while True:
            c = a + step
            if c >= hi:
                b = hi
                break
            if c > 2**1024:  # beyond float64: the reference is inf (our search stops there too)
                return INF
            if pred(c):
                b = c
                break
            a = c
            step *= 2
    while b - a > 1:
        mid = (a + b) // 2
        if pred(mid):
            b = mid
        else:
            a = mid
    return b


def _pred(D, pf, upper, target, exact_target):
    """pred(k): cdf(k) >= target (upper: ccdf(k) <= target), target() an arb at the current
    precision and exact_target a Fraction (or None where it is irrational)."""
    side = D.ccdf if upper else D.cdf

    def pred(k):
        def diff():
            p = [arb(v) for v in pf]
            v = side(arb(k), *p) - target()
            return -v if upper else v

        def exact():
            if exact_target is None:
                return None
            c = D.cdf_exact(k, *pf)
            if c is None:
                return None
            v = (1 - c) - exact_target if upper else c - exact_target
            return -v if upper else v

        return ar.sign(diff, exact=exact, exact_after=512) >= 0

    return pred


_D14 = Fraction(1, 10**14)


def discrete_inverse(D, m, pf, t, k0, pert=0):
    """The exact quantile of a discrete distribution (docs/design/stats.md §2.3): icdf = smallest k with
    cdf(k) >= p, iccdf = smallest k with ccdf(k) <= p, the log forms with p = e^t; each
    compared on the side whose probability is small. pert = -1 or +1 moves the target by
    BACKWARD (relative, see _perturbed), exactly."""
    lo, hi = D.support(*pf)
    d = pert * _D14

    def q_exact(v):  # the target t (1 + d), or for t > 1/2 the side 1 - t (1 + d) of icdf/iccdf
        return Fraction(v) * (1 + d)

    if m in ("icdf", "iccdf"):
        small = t <= 0.5
        if small:
            tgt, ex = (lambda: arb(t) * (1 + _arbq(d))), q_exact(t)
        else:
            q = Fraction(1) - Fraction(t)  # exact: 1 - t of a double in (1/2, 1)
            tgt, ex = (lambda: _arbq(q) * (1 - _arbq(d))), q * (1 - d)
        # small side: icdf -> cdf(k) >= t; iccdf -> ccdf(k) <= t; else the other side at 1 - t
        upper = (m == "iccdf") == small
        return _smallest_int(lo, hi, k0, _pred(D, pf, upper, tgt, ex))
    T = arb(t)
    tt = lambda: T * (1 - _arbq(d))  # relative in t: t (1 - d) moves the target p = e^t up for d > 0
    if t > -math.log(2):
        q = lambda: -tt().expm1()  # 1 - e^t, the small side
        upper = m == "ilogcdf"
    else:
        q = lambda: tt().exp()
        upper = m == "ilogccdf"
    return _smallest_int(lo, hi, k0, _pred(D, pf, upper, q, None))


# The discrete inverses' contract is a backward error (user decision 2026-11-16): the answer
# is right when it is the exact quantile of some target within relative 1e-14 of the given
# one: for icdf and iccdf on the small side (p or 1 - p, whichever is below 1/2), for the log
# forms relative in t itself (as the continuous inverses' kappa; t = -1e5 has an ULP of
# 1.5e-11, so no bound finer than relative is meaningful there). That covers a target within
# the cdf's last bits of a step (the neighbour is as good) and a quantile so large that the
# steps are finer than the cdf's resolution (Zipf near a = 1). Each point stores the interval
# [kmin, kmax] of such answers.
BACKWARD = 1e-14


def discrete_interval(D, m, pf, t, k):
    """[kmin, kmax]: the exact quantiles at the ends of the perturbed targets (each method is
    monotone in its input, so the answers in between lie between)."""
    k0 = k if math.isfinite(k) else None
    ends = [discrete_inverse(D, m, pf, t, k0, pert) for pert in (-1, 1)]
    return min(ends), max(ends)


def _discrete_point(D, dist, m, args):
    x0 = _our(dist, m, args)
    mm, a = ("icdf", list(args) + [0.5]) if m == "median" else (m, list(args))
    *pf, t = a
    k = discrete_inverse(D, mm, pf, t, x0 if np.isfinite(x0) else None)
    kmin, kmax = discrete_interval(D, mm, pf, t, k)
    hi_ = float(k)
    lo_ = float(k - int(hi_)) if math.isfinite(hi_) else 0.0
    return hi_, lo_, 1.0, float(kmin), float(kmax)


# ---------------------------------------------------------------- files
# A file holds one input set of a distribution (its kind: x, p, logp or param) and the methods of
# that kind (tests/reffile.py). Writing a method keeps the file's other methods.

KINDS = ("x", "p", "logp", "param")


def kind(m):
    if m in PARAM_METHODS:
        return "param"
    if m in ("icdf", "iccdf"):
        return "p"
    if m in ("ilogcdf", "ilogccdf"):
        return "logp"
    return "x"


def file_path(dist, m):
    return OUT / f"{dist}_{kind(m)}.npz"


def _read(dist, m):
    from tests import reffile

    path = file_path(dist, m)
    return reffile.read(path) if path.exists() else None


def _write(dist, m, xs, methods, n_requested):
    """The (distribution, kind) file with these inputs and methods (the dict of tests/reffile.write)."""
    from tests import reffile

    order = [mm for mm in (INPUT_METHODS + DISCRETE_INPUT_METHODS[:2] + PARAM_METHODS) if mm in methods]
    reffile.write(
        file_path(dist, m),
        **ar.provenance(),
        tool=TOOL,
        n_requested=n_requested,
        inputs={f"x{k}": np.asarray(x, dtype=np.float64) for k, x in enumerate(xs)},
        methods={mm: methods[mm] for mm in dict.fromkeys(order)},
    )


def _method_dict(mm):
    out = {"over": mm.over, "index": mm.index, "dropped": mm.dropped, "hi": mm.hi, "lo": mm.lo}
    out.update(mm.fields)
    return out


def _same_inputs(r, xs):
    if r is None or len(r.inputs) != len(xs):
        return False
    return all(
        np.array_equal(np.asarray(a).view(np.int64), np.asarray(b).view(np.int64))
        for a, b in zip(r.inputs.values(), xs)
    )


# ---------------------------------------------------------------- building


def _our(dist, m, args):
    import stardust as sd

    return float(getattr(sd.stats._distribution._k, f"{dist}_{m}")(*args))


# A point is dropped when its value does not settle by its precision cap (arbref.MAX_PREC, or the
# distribution's max_prec), a series, an integral or an iteration runs past its cap, or Arb rejects
# it: NotSettled is an ArithmeticError. Anything else is a bug in this tool and stops the build.
_DROPS = (ArithmeticError, ValueError, ZeroDivisionError, OverflowError)


def _point(task):
    dist, m, i, args = task
    D = DISTRIBUTIONS[dist]
    try:
        if getattr(D, "discrete", False) and (m in INVERSE or m == "median"):
            return i, _discrete_point(D, dist, m, args)
        sc = 1.0
        if m in INVERSE or (m == "median" and not hasattr(D, "median")):
            mm, a = ("icdf", list(args) + [0.5]) if m == "median" else (m, list(args))
            x, k = inverse_value(D, mm, a, _our(dist, m, args))
            sc = max(1.0, k)
            v = x
        else:
            v = rigorous(
                lambda: method_value(D, m, [arb(a) for a in args]), max_prec=getattr(D, "max_prec", ar.MAX_PREC)
            )
        hi, lo = to_dd(v)
        return i, (hi, lo, sc)
    except _DROPS:  # a point Arb cannot settle within the caps (no root, an infinite ball) is dropped
        return i, None


def _interval_point(task):
    dist, m, i, args, k = task
    D = DISTRIBUTIONS[dist]
    try:
        mm, a = ("icdf", list(args) + [0.5]) if m == "median" else (m, list(args))
        kk = int(k) if math.isfinite(k) else k
        lo_, hi_ = discrete_interval(D, mm, a[:-1], a[-1], kk)
        return task, (float(lo_), float(hi_))
    except _DROPS:
        return task, None


def add_intervals(targets, jobs):
    """Add kmin and kmax to the discrete inverses (the exact k is kept); a point whose interval
    cannot be settled is dropped from the method (and recorded as dropped)."""
    tasks, files = [], {}
    for dist, m in targets:
        if not (getattr(DISTRIBUTIONS[dist], "discrete", False) and (m in INVERSE or m == "median")):
            continue
        r = _read(dist, m)
        if r is None or m not in r.methods:
            continue
        mm = r.methods[m]
        xs = r.args(m)
        files[(dist, m)] = (r, {})
        tasks += [(dist, m, j, tuple(float(x[j]) for x in xs), float(mm.hi[j])) for j in range(mm.index.size)]
    import stardust  # noqa: F401  (inherited by the workers, as in build_all)

    start = time.monotonic()
    with multiprocessing.get_context("fork").Pool(jobs, maxtasksperchild=500) as pool:
        for n, (task, r) in enumerate(pool.imap_unordered(_interval_point, tasks, chunksize=16), 1):
            files[(task[0], task[1])][1][task[2]] = r
            if n % max(1, len(tasks) // 20) == 0:
                print(f"  intervals {n}/{len(tasks)}, {(time.monotonic() - start) / 60:.1f} min", flush=True)
    for (dist, m), (r, res) in files.items():
        mm = r.methods[m]
        n = mm.index.size
        kmin, kmax, keep = np.empty(n), np.empty(n), np.zeros(n, dtype=bool)
        for j, v in res.items():
            if v is not None:
                kmin[j], kmax[j] = v
                keep[j] = True
        d = _method_dict(mm)
        for key in ("hi", "lo", "scale", "guaranteed"):
            d[key] = d[key][keep]
        d["dropped"] = np.union1d(mm.dropped, mm.index[~keep]).astype(np.int64)
        d["index"] = mm.index[keep]
        d["kmin"], d["kmax"] = kmin[keep], kmax[keep]
        methods = {k: _method_dict(v) for k, v in r.methods.items()}
        methods[m] = d
        _write(dist, m, tuple(r.inputs.values()), methods, r.meta["n_requested"])
        dropped = f", {n - int(keep.sum())} dropped" if n > keep.sum() else ""
        print(f"{dist}_{m}: {int(keep.sum())} intervals{dropped}", flush=True)


# The guaranteed region in the parameters: the ranges over which sd.special's functions
# underneath are measured and guaranteed (docs/api/special.md §3, §13). Points outside
# stay in the files and are held to the documented loose bound. The discrete distributions
# are guaranteed over the parameter ranges their sets sample (user decision 2026-11-16;
# docs/api/stats.md §2.3), so every point of theirs is in the region.
PARAM_RANGE = {
    "gamma": lambda a, s: a <= 1e6,
    "inversegamma": lambda a, s: a <= 1e6,
    "chisquared": lambda df: df <= 2e6,
    "beta": lambda a, b: (a <= 1e5) & (b <= 1e5),
    "studentt": lambda df: df <= 1e6,
    "f": lambda m, n: (m <= 1e5) & (n <= 1e5),
}


# Corners left out of the guaranteed region inside the parameter range (user decision
# 2026-11-15): t's inverses for df >= 1e5 with the probability below 2^-1000 (where they
# solve on the log side; csrc/stats/dist/beta.hpp kDeepP), measured 1.4e-14.
TINY_P = 2.0**-1000


# The noncentral distributions are guaranteed where sd.special's noncentral functions are
# measured (docs/api/special.md §14; user decision 2026-11-16): their parameter ranges,
# probabilities down to 1e-300 (x points whose smaller tail, by our own cdf, is below it are
# outside).
NC_RANGE = {
    "noncentralchisquared": lambda df, nc: (df >= 1e-2) & (df <= 1e4) & (nc <= 1e4),
    "noncentralt": lambda df, d: (df >= 1e-1) & (df <= 1e5) & (np.abs(d) <= 100),
    "noncentralf": lambda dfn, dfd, nc: (dfn >= 1e-2) & (dfn <= 1e4) & (dfd >= 1e-2) & (dfd <= 1e4) & (nc <= 1e4),
}
NC_TINY = 1e-300


def _tails(dist, xs):
    """min(cdf, ccdf) at the x points from the file's own Arb references (NaN where either is
    missing), for the noncentral distributions' guaranteed region."""
    r = _read(dist, "cdf")
    t = np.full(xs[0].size, np.nan)
    if not _same_inputs(r, xs) or "cdf" not in r.methods or "ccdf" not in r.methods:
        return t
    c, cc = np.full(t.size, np.nan), np.full(t.size, np.nan)
    c[r.methods["cdf"].index] = r.methods["cdf"].hi
    cc[r.methods["ccdf"].index] = r.methods["ccdf"].hi
    return np.minimum(c, cc)


def _nc_outside(dist, m, xs):
    npar = len(DISTRIBUTIONS[dist].params)
    out = ~NC_RANGE[dist](*xs[:npar])
    if m == "entropy":  # quadrature: a measured, loose bound (docs/design/stats.md)
        return np.ones(xs[0].size, dtype=bool)
    if m in PARAM_METHODS:
        return out
    v = xs[-1]
    if m in ("icdf", "iccdf"):
        return out | (v < NC_TINY) | (1 - v < NC_TINY)
    if m in ("ilogcdf", "ilogccdf"):
        return out | (v < np.log(NC_TINY)) | (-np.expm1(v) < NC_TINY)
    t = _tails(dist, xs)  # the reference, not the code measured, decides; unknown: outside
    with np.errstate(invalid="ignore"):
        return out | ~(t >= NC_TINY)


def refdata_class(dist):
    return {"noncentralchisquared": "NoncentralChiSquared", "noncentralt": "NoncentralT", "noncentralf": "NoncentralF"}[
        dist
    ]


def corner(dist, m, xs):
    if dist in NC_RANGE:
        return _nc_outside(dist, m, xs)
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
        g = g & PARAM_RANGE[dist](*xs[: len(D.params)])
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


RESAMPLE = False  # --resample: draw new input sets instead of reusing the existing files' x arrays


def _own_mask(m, xs):
    """The input set's own region (x_sets, p_sets, logp_sets, param_sets) of a file's inputs."""
    v = xs[-1]
    if m in PARAM_METHODS:
        return np.ones(v.size, dtype=bool)
    if m in ("icdf", "iccdf"):
        return (v > 0) & (v < 1)
    if m in ("ilogcdf", "ilogccdf"):
        return v < 0
    return np.isfinite(v)


def file_inputs(dist, m):
    """(xs, guaranteed) of a method: the existing file's inputs (frozen: the samplers, scipy's and
    our own quantiles, change between versions, and the references should be redone on the same
    points; a point the method dropped before is tried again), or with --resample (or no file
    yet) the seeded input set."""
    D = DISTRIBUTIONS[dist]
    r = _read(dist, m)
    if r is not None and not RESAMPLE:
        xs = tuple(np.asarray(x, dtype=np.float64) for x in r.inputs.values())
        return xs, region(dist, D, xs, _own_mask(m, xs), m)
    rng = np.random.default_rng(20261115)
    xs, guaranteed = inputs(D, m, rng)
    xs = tuple(np.asarray(x, dtype=np.float64) for x in xs)
    return xs, region(dist, D, xs, guaranteed, m)


def _save(dist, m, xs, guaranteed, results):
    """Write method m into its (distribution, kind) file: the references at the points that
    settled, the others recorded as dropped. The file's other methods are kept when its inputs
    are these; when the inputs changed (--resample) the others no longer apply and are left out."""
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
    d = {"over": "point", "index": np.flatnonzero(keep), "dropped": np.flatnonzero(~keep), "hi": hi[keep],
         "lo": lo[keep], "guaranteed": np.asarray(guaranteed, dtype=bool)[keep], "scale": scale[keep]}  # fmt: skip
    if interval:
        d["kmin"], d["kmax"] = kmin[keep], kmax[keep]
    old = _read(dist, m)
    methods = {}
    if _same_inputs(old, xs):
        methods = {k: _method_dict(v) for k, v in old.methods.items()}
    elif old is not None:
        print(f"{dist}_{kind(m)}: new inputs; the methods not rebuilt now are left out: "
              f"{', '.join(k for k in old.methods if k != m)}", flush=True)  # fmt: skip
    methods[m] = d
    _write(dist, m, xs, methods, n)
    return int(keep.sum()), int(n - keep.sum())


_FILES = []  # (dist, m, xs) of the methods being built: the workers inherit them (fork) and read the inputs there


def _point_task(task):
    """((k, i), result) for point i of method k: imap_unordered returns results as they finish."""
    k, i = task
    dist, m, xs = _FILES[k]
    return task, _point((dist, m, i, tuple(float(x[i]) for x in xs)))


def build_all(targets, jobs):
    """Every (distribution, method) file through one process pool: the points of all files
    are interleaved and handed out one at a time (imap_unordered, chunksize 1), so a slow
    point holds one worker, not a whole file, and no core waits for the slowest point of a
    file before the next file starts. Each file is written as soon as its last point arrives.
    The parent keeps the inputs as numpy arrays and hands out (file, point) index pairs, and
    freezes its objects before forking (gc.freeze): a worker's collector would otherwise walk
    every object it inherits and copy the parent's pages into each of the ten workers."""
    import gc
    import time

    import stardust  # noqa: F401  (imported once here: the forked workers inherit it, and its
    # import, an editable build's check, would otherwise count against each worker's first point)

    _FILES.clear()
    files, pending = {}, {}
    for dist, m in targets:
        xs, g = file_inputs(dist, m)
        files[len(_FILES)] = (xs, g, {})
        pending[len(_FILES)] = xs[0].size
        _FILES.append((dist, m, xs))
    _INPUT_CACHE.clear()
    sizes = np.array([pending[k] for k in range(len(_FILES))])
    fk = np.repeat(np.arange(len(_FILES)), sizes)
    fi = np.concatenate([np.arange(n) for n in sizes])
    order = np.random.default_rng(0).permutation(fk.size)  # mix slow and fast files
    fk, fi = fk[order], fi[order]

    def done(task, r):
        k, i = task
        xs, g, res = files[k]
        res[i] = r[1]
        pending[k] -= 1
        if pending[k] == 0:
            dist, m, _ = _FILES[k]
            kept, dropped = _save(dist, m, xs, g, res)
            files[k] = None
            print(f"{dist}_{m}: {kept} points" + (f", {dropped} dropped" if dropped else ""), flush=True)

    start, total, step = time.monotonic(), int(fk.size), max(1, int(fk.size) // 20)
    count = 0

    def progress():
        nonlocal count
        count += 1
        if count % step == 0:
            el = time.monotonic() - start
            print(
                f"  progress {count}/{total} points ({100 * count // total}%), {el / 60:.1f} min, "
                f"~{el / count * (total - count) / 60:.1f} min left",
                flush=True,
            )

    tasks = ((int(k), int(i)) for k, i in zip(fk, fi))
    if jobs > 1:
        # workers are replaced every 500 points: whatever FLINT caches (constants, Bernoulli
        # numbers) stays bounded
        gc.collect()
        gc.freeze()
        with multiprocessing.get_context("fork").Pool(jobs, maxtasksperchild=500) as pool:
            for task, r in pool.imap_unordered(_point_task, tasks, chunksize=1):  # in completion order
                done(task, r)
                progress()
    else:
        for task in tasks:
            done(*_point_task(task))


def remask(targets):
    """Rewrite the guaranteed masks of existing methods from region() (no Arb); the count of
    points whose mask changed, per method."""
    changed = {}
    for dist, m in targets:
        r = _read(dist, m)
        if r is None or m not in r.methods:
            continue
        D = DISTRIBUTIONS[dist]
        xs = tuple(r.inputs.values())
        mm = r.methods[m]
        g = region(dist, D, xs, _own_mask(m, xs), m)[mm.index]
        changed[f"{dist}_{m}"] = int((g != mm.fields["guaranteed"]).sum())
        methods = {k: _method_dict(v) for k, v in r.methods.items()}
        methods[m]["guaranteed"] = g
        _write(dist, m, xs, methods, r.meta["n_requested"])
        print(f"{dist}_{m}: {int(g.sum())}/{g.size} guaranteed ({changed[f'{dist}_{m}']} changed)", flush=True)
    return changed


# ---------------------------------------------------------------- long tail sums
#
# tests/stats/ref/tails/<distribution>.npz: the cdfs of the hypergeometric and the beta-binomial
# where a side of the pmf has 1e4 to 1e9 terms (sd sums those as an integral with an
# Euler-Maclaurin correction, csrc/stats/discrete.hpp smooth_tail_sum; the main input sets
# stop near the guaranteed region, whose sums are short). The definitions are the term-by-term
# sums: the pmf at one point from Arb's log-gammas, its neighbours by the exact ratio, each side
# of k summed apart (no cancellation). A side stops past the mode once the geometric bound on
# what is left is below 2^-140 of the sum where the pmf is log-concave (the hypergeometric, a
# beta-binomial with both shapes at least 1: the ratios fall on outwards), and the bound goes
# into the ball. A beta-binomial side otherwise runs to its end of the support: for n up to 1e8
# both sides of every k, past that (n = 1e9, 2^34) only points within 3e7 of an end, the side
# toward that end summed and the other side 1 minus it (no cancellation: the side summed is the
# smaller or not below 1e-30, and the precision is 256 bits). The log of a side above 1/2 is
# log1p of minus the other. The inputs are fixed lists (no generator), one group per parameter
# set.

TAILS_OUT = OUT / "tails"
TAILS = ("hypergeometric_tails", "betabinomial_tails")
TAIL_METHODS = ("cdf", "ccdf", "logcdf", "logccdf")
TAILS_PREC = 256


def _hyper_tail_points():
    """(N, K, n, k): standard deviations 2e3 to 5e5, balanced and skewed margins, k from 30
    standard deviations below the mode to 12 above."""
    out = []
    shapes = ((0.5, 0.5), (0.1, 0.7), (0.98, 0.03), (0.3, 0.001))
    zs = (-30.0, -6.0, -1.0, 0.0, 0.5, 4.0, 12.0)
    for sd, nshapes, nz in ((2e3, 4, 7), (1e4, 4, 7), (5e4, 4, 7), (2e5, 2, 7), (5e5, 2, 4)):
        for fk, fn in shapes[:nshapes]:
            N = int(sd * sd / (fk * (1 - fk) * fn * (1 - fn)))
            K, n = int(N * fk), int(N * fn)
            mode = (n + 1) * (K + 1) // (N + 2)
            s = math.sqrt(n * (K / N) * (1 - K / N) * (1 - n / N))
            for z in zs[:nz] if nz == 7 else zs[1:5]:
                out.append((N, K, n, mode + int(z * s)))
    return out


def _betabinom_tail_points():
    """(n, a, b, ks, how): bells, power-law ends, U and J shapes, the uniform; k near both ends,
    around the mean and between. how: "all" (both sides of every k by passes over the whole
    support), "ends" (k within 3e7 of an end) or "bell" (log-concave, k around the mean, each
    side from k outwards)."""
    shapes = (
        (2.5, 1.7), (0.5, 0.5), (0.3, 1.2), (0.7, 3.0), (50.0, 80.0), (1.0, 1.0), (1000.0, 2000.0), (1.0001, 1.5),
        (0.01, 0.02), (3.0, 0.6), (200.0, 1.5),
    )  # fmt: skip
    out = []
    for n, nshapes in ((30_000, 11), (300_000, 11), (3_000_000, 11), (10_000_000, 4)):
        for a, b in shapes[:nshapes]:
            mean = n * a / (a + b)
            ks = {5, 3000, 5000, n // 40, int(mean / 2), int(mean), int(mean) + 1, int((mean + n) / 2)}
            ks |= {n - n // 40, n - 5000, n - 3000, n - 5, n // 3, (2 * n) // 3}
            out.append((n, a, b, sorted(k for k in ks if 0 <= k < n), "all"))
    n = 100_000_000
    for a, b in ((2.5, 1.7), (0.7, 3.0), (0.5, 0.5)):
        mean = n * a / (a + b)
        ks = {5, 3000, 2_000_000, int(mean / 2), int(mean), int((mean + n) / 2), n // 3, n - 2_000_000, n - 3000}
        out.append((n, a, b, sorted(ks), "all"))
    for n, shapes_far in ((1_000_000_000, ((2.5, 1.7), (0.7, 3.0), (0.5, 0.5))), (2**34, ((1.0001, 1.5),))):
        for a, b in shapes_far:
            ks = [5, 3000, 100_000, 3_000_000, 30_000_000]
            out.append((n, a, b, ks + [n - 1 - k for k in reversed(ks)], "ends"))
    n, a, b = 1_000_000_000, 1e5, 1e5
    sd = math.sqrt(n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1)))
    out.append((n, a, b, [int(n / 2 + z * sd) for z in (-6.0, -1.0, 0.5, 3.0)], "bell"))
    return out


def _hyper_tail(task):
    """(P(X <= k), P(X > k)) of one hypergeometric point."""
    N, K, n, k = task
    M = N - K - n
    lo, hi = max(0, -M), min(n, K)
    mode = (n + 1) * (K + 1) // (N + 2)
    with precision(TAILS_PREC):
        eps = arb(2) ** -140

        def lg(v):
            return (arb(v) + 1).lgamma()

        def choose(u, v):
            return lg(u) - lg(v) - lg(u - v)

        def side(j, down):  # sum of pmf(i) / pmf(j) over i = j, j -+ 1, ...
            t = s = arb(1)
            while (j > lo) if down else (j < hi):
                r = (
                    arb(j * (M + j)) / ((K - j + 1) * (n - j + 1))
                    if down
                    else arb((K - j) * (n - j)) / ((j + 1) * (M + j + 1))
                )
                t = t * r
                j += -1 if down else 1
                s += t
                if (j & 1023) == 0 and ((j < mode) if down else (j > mode)) and r < 1:
                    rest = ar.geometric_tail(t, r)
                    if rest < eps * s:
                        return add_error(s, rest)
            return s

        def at(j):
            return (choose(K, j) + choose(N - K, n - j) - choose(N, n)).exp()

        lower = at(k) * side(k, True) if k >= lo else arb(0)
        upper = at(k + 1) * side(k + 1, False) if k < hi else arb(0)
        return lower, upper


def _betabinom_tail(task):
    """[(P(X <= k), P(X > k)) for k in ks] of one beta-binomial parameter set (how: see
    _betabinom_tail_points)."""
    n, a, b, ks, how = task
    with precision(TAILS_PREC):
        A, B = arb(a), arb(b)
        eps = arb(2) ** -140

        def up(j):  # pmf(j + 1) / pmf(j)
            return (n - j) * (A + j) / ((j + 1) * (B + (n - j - 1)))

        def down(j):  # pmf(j - 1) / pmf(j)
            return j * (B + (n - j)) / ((n - j + 1) * (A + (j - 1)))

        def log_pmf(j):
            lg = lambda v: arb(v).lgamma()  # noqa: E731
            return (
                lg(arb(n) + 1) - lg(arb(j) + 1) - lg(arb(n - j) + 1) + lg(A + j) + lg(B + (n - j)) - lg(A + B + n)
                + lg(A + B) - lg(A) - lg(B)
            )  # fmt: skip

        if how == "bell":  # each side from k outwards, the geometric bound past the mode
            mode = (n * (a - 1) + 1 - b) / (a + b - 2)

            def side(j, step):
                t = s = arb(1)
                while 0 < j < n if step < 0 else j < n:
                    r = down(j) if step < 0 else up(j)
                    t, j = t * r, j + step
                    s += t
                    if (j & 1023) == 0 and ((j < mode) if step < 0 else (j > mode)) and r < 1:
                        rest = ar.geometric_tail(t, r)
                        if rest < eps * s:
                            return add_error(s, rest)
                return s

            return [(log_pmf(k).exp() * side(k, -1), log_pmf(k + 1).exp() * side(k + 1, 1)) for k in ks]
        p0, pn = log_pmf(0).exp(), log_pmf(n).exp()
        lows = [k for k in ks if how == "all" or k < n // 2]
        highs = [k for k in ks if how == "all" or k >= n // 2]
        lower, upper = {}, {}
        if lows:
            want, t, s = set(lows), arb(1), arb(1)  # pmf(j) / pmf(0) and its sum from 0
            for j in range(max(lows) + 1):
                if j in want:
                    lower[j] = p0 * s
                t = t * up(j)
                s += t
        if highs:
            want, t, s = set(highs), arb(1), arb(1)  # pmf(j) / pmf(n) and its sum from n
            for j in range(n, min(highs), -1):
                if j - 1 in want:
                    upper[j - 1] = pn * s
                t = t * down(j)
                s += t
        out = []
        for k in ks:
            lo_, up_ = lower.get(k), upper.get(k)
            out.append((lo_ if lo_ is not None else 1 - up_, up_ if up_ is not None else 1 - lo_))
        return out


def _tail_job(job):
    """[{method: (hi, lo)}] of one task's points (balls do not pickle: the double-doubles leave
    the worker)."""
    name, task = job
    pairs = [_hyper_tail(task)] if name == "hypergeometric_tails" else _betabinom_tail(task)
    rows = []
    with precision(TAILS_PREC):
        for lower, upper in pairs:
            row = {}
            for m, (v, other) in {"cdf": (lower, upper), "ccdf": (upper, lower)}.items():
                # (a side above 1/2: log1p of minus the other, its log is near 0)
                lv = (-other).log1p() if v > 0.5 else (v.log() if v > 0 else -INF)
                for mm, w in ((m, v), ("log" + m, lv)):
                    if not isinstance(w, float) and not ar.settled(w):
                        raise NotSettled((name, task, mm))
                    row[mm] = to_dd(w)
            rows.append(row)
    return rows


def build_tails(name, jobs):
    """Write tests/stats/ref/tails/<distribution>.npz."""
    from tests import reffile

    t0 = time.time()
    if name == "hypergeometric_tails":
        tasks = _hyper_tail_points()
        groups = sorted({t[:3] for t in tasks}, key=[t[:3] for t in tasks].index)
        points = [(t[:3], t[3]) for t in tasks]
    else:
        tasks = _betabinom_tail_points()
        groups = [t[:3] for t in tasks]
        points = [(t[:3], k) for t in tasks for k in t[3]]
    with multiprocessing.get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool:
        raw = pool.map(_tail_job, [(name, t) for t in tasks], chunksize=1)
    cols = {m: ([], []) for m in TAIL_METHODS}
    for row in (row for rows in raw for row in rows):
        for m in TAIL_METHODS:
            cols[m][0].append(row[m][0])
            cols[m][1].append(row[m][1])
    index = np.arange(len(points), dtype=np.int64)
    none = np.zeros(0, dtype=np.int64)
    x = [[float(g[i]) for g, _ in points] for i in range(3)] + [[float(k) for _, k in points]]
    TAILS_OUT.mkdir(parents=True, exist_ok=True)
    reffile.write(
        TAILS_OUT / f"{name.partition('_')[0]}.npz",
        **ar.provenance(extra="term-by-term sums of the pmf by its exact ratio"),
        tool=TOOL,
        n_requested=len(points),
        inputs={f"x{i}": np.asarray(v, dtype=np.float64) for i, v in enumerate(x)},
        group=[groups.index(g) for g, _ in points],
        ngroups=len(groups),
        methods={
            m: {"over": "point", "index": index, "dropped": none, "hi": np.asarray(h), "lo": np.asarray(l)}
            for m, (h, l) in cols.items()
        },
    )
    print(f"{name}: {len(points)} points in {len(groups)} groups, {time.time() - t0:.0f} s")


def main(argv=None):
    import argparse

    from bench import cli

    global RESAMPLE
    import mvcdfref
    import mvref

    mv_names = list(mvref.FAMILIES) + list(mvcdfref.SETS)
    ap = argparse.ArgumentParser(prog="tools/stats_ref.py", description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="distribution[:method,...], or a multivariate family or cdf set (default: every distribution)",
    )
    cli.add_jobs(ap)
    ap.add_argument("--masks-only", action="store_true", help="rewrite the guaranteed masks only (no Arb)")
    ap.add_argument("--intervals", action="store_true", help="add the discrete inverses' [kmin, kmax]")
    ap.add_argument("--resample", action="store_true", help="draw new input sets instead of reusing the files'")
    a = ap.parse_args(argv)
    RESAMPLE = a.resample
    targets, mv = [], []
    for name in a.names or list(DISTRIBUTIONS):
        if name in mv_names or name in TAILS:
            mv.append(name)
            continue
        dist, _, ms = name.partition(":")
        if dist not in DISTRIBUTIONS:
            ap.error(f"unknown distribution {dist!r}")
        base = DISCRETE_INPUT_METHODS if getattr(DISTRIBUTIONS[dist], "discrete", False) else INPUT_METHODS
        methods = ms.split(",") if ms else list(base + PARAM_METHODS)
        bad = [m for m in methods if m not in base + PARAM_METHODS]
        if bad:
            ap.error(f"unknown method {bad[0]!r} of {dist}")
        targets += [(dist, m) for m in methods]
    with ar.logged("stats_ref"):
        if a.masks_only:
            remask(targets)
        elif a.intervals:
            add_intervals(targets, a.jobs)
        else:
            if targets:
                build_all(targets, a.jobs)
                # the noncentral distributions' masks follow their final cdf and ccdf references
                remask([(d, m) for d, m in targets if d in NC_RANGE and kind(m) == "x"])
            for name in mv:
                if name in TAILS:
                    build_tails(name, a.jobs)
                else:
                    (mvref.write if name in mvref.FAMILIES else mvcdfref.build)(name, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
