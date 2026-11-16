"""Reference values for sd.special from Arb (python-flint; docs/design/special.md §6).

    .venv/bin/python tools/special_ref.py            # every function in FUNCTIONS and COMPLEX
    .venv/bin/python tools/special_ref.py NAME ...   # only these
    .venv/bin/python tools/special_ref.py --jobs=8 NAME ...   # points in 8 processes (default 10)

writes tests/special/ref/<name>.npz (format: tests/special/refdata.py). The files are
committed; tests and bench/special/accuracy.py read them and never call Arb.

Each reference is an Arb ball [mid +- rad] that contains the true value, evaluated at
doubling precision until rad is below 2^-110 of the value (or 2^-1132 absolutely, below the
normal doubles), so hi + lo is the value rounded (tools/arbref.py). Arb's functions bound
their own errors; the sums written here (the incomplete gamma and beta series, the
noncentral Poisson mixtures) add a bound on their truncated tail (arbref.add_error); the
inverses are certified by the sign change of F(x) - p across x (1 -+ 2^-110). A point that
does not settle by 8192 bits, or within POINT_SECONDS, is dropped.

Each function's input set (docs/design/special.md §6) is
  - a grid over its guaranteed region,
  - log-uniform random points (both signs where the domain has them), inside the
    guaranteed region and in the loose region around it,
  - edge points: a few ULP either side of every region boundary, poles and zeros,
    0, and the domain ends.
The generator is seeded: running the tool again gives the same file.
"""

import math
import sys
from contextlib import contextmanager
from pathlib import Path

import arbref
import flint
import numpy as np
from arbref import (
    NotSettled,
    add_error,
    geometric_tail,
    precision,
    rigorous,
    solve_log_settled,
    solve_settled,
    to_dd,
)
from flint import acb, arb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "special" / "ref"
DPS = 50  # the mpmath tools' digits; kept because tools/stats_ref.py may still read it (Arb: arbref.TARGET_BITS)
FLOOR = arbref.DOUBLE_TINY  # values below the normal doubles need only an absolute 2^-1132
MODULUS_BITS = arbref.TARGET_BITS + 10  # a complex part far below |f| need only 2^-120 |f| (S8: |error| / |f|)
TRUNC_BITS = arbref.TARGET_BITS + 12  # the series here stop when their tail bound is below 2^-max(122, prec) of the sum
MAX_TERMS = 4_000_000  # a series that has not converged by then gives up (NotSettled)


def _trunc():
    """Where the series written here stop: their tail bound below 2^-max(122, prec) of the sum. It
    follows the precision, so a value that 1 - (a sum) cancels settles as the precision doubles."""
    return arb(2) ** -max(TRUNC_BITS, arbref.ctx.prec)


def _sqrt_pi():
    return arb.const_sqrt_pi()


def _f(v):
    """A float of an arb or acb modulus (scales and starts; not references)."""
    return float(abs(v).mid())


def _exact(fn):
    """fn() at a precision where its arb operations (a negation, x2, 1 - y: each rounds to the
    current precision in python-flint) are exact."""
    with precision(4 * arbref.MAX_PREC):
        return fn()


def _kappa(fn):
    """kappa = |p / (x F'(x))| of an inverse: fn() known to 2^-24 (a float; NaN if it never settles)."""
    try:
        return _f(rigorous(fn, target=24))
    except NotSettled:
        return float("nan")


def _scale_of(fn):
    """max(1, fn()) for a bound's multiplier (scale): fn() (an arb) known to 2^-24 (an inf
    where it overflows a double)."""
    return max(1.0, _f(rigorous(fn, target=24)))


# ---------------------------------------------------------------- input sets


def grid(lo, hi, n):
    """n evenly spaced float64 points in [lo, hi] (inputs: numpy in, numpy out)."""
    return np.linspace(lo, hi, n)


def loguniform(rng, lo, hi, n, signed=False):
    """n float64 points with log|x| uniform in [log lo, log hi] (lo > 0); random signs when signed."""
    x = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    if signed:
        x *= rng.choice([-1.0, 1.0], n)
    return x


def around(points, k=4):
    """Each point and the k float64 neighbours on either side (a float64 array)."""
    out = []
    for p in points:
        v = float(p)
        out.append(v)
        up = down = v
        for _ in range(k):
            up, down = np.nextafter(up, np.inf), np.nextafter(down, -np.inf)
            out += [float(up), float(down)]
    return np.array(out)


# ---------------------------------------------------------------- the functions
#
# name: (nin, function of arb inputs -> arb (at the current precision; a float for an exact
# special value), input-set builder rng -> (inputs, guaranteed))


def _cf_tan_inputs(rng):
    # Guaranteed |x| <= 1; loose 1 < |x| < pi/2, where the fraction needs more terms and
    # tan's own condition number x / (sin x cos x) grows without bound.
    g = np.concatenate([grid(-1, 1, 2001), loguniform(rng, 1e-300, 1, 4000, signed=True),
                        around([0.0, 1.0, -1.0, 5e-324, -5e-324])])
    loose = np.concatenate([rng.uniform(1, 1.5707963267948966, 2000) * rng.choice([-1.0, 1.0], 2000),
                            around([1.5707963267948966, -1.5707963267948966])])
    x = np.concatenate([g, loose])
    return (x,), np.abs(x) <= 1.0


# ---- S1: logistic and information (docs/api/special.md §5). Guaranteed region: the whole
# domain (user decision 2026-11-13), so every finite input is "guaranteed".

def signed_wide(rng, n):
    """Finite reals of every magnitude, both signs."""
    return loguniform(rng, 1e-320, 1e308, n, signed=True)


def _expit_inputs(rng):
    x = np.concatenate([grid(-40, 40, 4001), signed_wide(rng, 3000), rng.uniform(-745.2, -700, 1000),
                        around([0.0, 36.7, -36.7, 709.78, -709.78, -745.13, -745.14, 5e-324, -5e-324])])
    return (x,), np.isfinite(x)


def _logit_inputs(rng):
    near1 = 1.0 - loguniform(rng, 2**-53, 0.5, 2000)
    p = np.concatenate([rng.uniform(0, 1, 3000), loguniform(rng, 1e-320, 0.5, 2000), near1,
                        0.5 + rng.uniform(-1e-6, 1e-6, 500), around([0.25, 0.5, 0.75, 5e-324]),
                        around([1.0 - 2**-53], 2)])
    p = p[(p > 0) & (p < 1)]
    return (p,), np.ones(p.size, dtype=bool)


def _xlogy_inputs(rng):
    n = 4000
    x = loguniform(rng, 1e-300, 1e300, n, signed=True)
    y = np.concatenate([loguniform(rng, 1e-320, 1e308, n - 1000), 1.0 + rng.uniform(-1e-8, 1e-8, 1000)])
    return (x, y), np.ones(n, dtype=bool)


def _xlog1py_inputs(rng):
    n = 4000
    x = loguniform(rng, 1e-300, 1e300, n, signed=True)
    y = np.concatenate([loguniform(rng, 1e-320, 1e308, 1500), -loguniform(rng, 1e-320, 0.5, 1500),
                        -1.0 + loguniform(rng, 2**-53, 0.5, 1000)])
    return (x, y), np.ones(n, dtype=bool)


def _entr_inputs(rng):
    x = np.concatenate([loguniform(rng, 5e-324, 1e308, 4000), 1.0 + rng.uniform(-1e-8, 1e-8, 1000),
                        around([1.0, np.exp(-1.0), 5e-324])])
    x = x[x >= 0]  # x < 0 is the -inf convention, checked in tests/special/test_logistic.py
    return (x,), np.ones(x.size, dtype=bool)


def _rel_entr_inputs(rng):
    n = 2000
    x1, y1 = loguniform(rng, 1e-300, 1e300, n), loguniform(rng, 1e-300, 1e300, n)  # ratios of every size
    x2 = loguniform(rng, 1e-300, 1e300, n)
    y2 = x2 * (1.0 + loguniform(rng, 1e-15, 0.9, n, signed=True))  # x / y near 1
    x3 = loguniform(rng, 1e-300, 1e-100, 500)
    y3 = loguniform(rng, 1e100, 1e300, 500)  # x / y underflows
    x = np.concatenate([x1, x2, x3, y3])
    y = np.concatenate([y1, y2, y3, x3])
    return (x, y), np.ones(x.size, dtype=bool)


# ---- S2: the error function and the normal distribution (docs/api/special.md §2).
# Guaranteed region: the whole domain (user decision 2026-11-13).

SQRT2 = np.sqrt(2.0)


def _erf_like_inputs(rng, lo, hi, extra, edges):
    x = np.concatenate([grid(lo, hi, 4001), signed_wide(rng, 3000), extra(rng), around(edges)])
    return (x,), np.isfinite(x)


def _erf_inputs(rng):
    return _erf_like_inputs(rng, -6, 6, lambda r: r.uniform(-1, 1, 500), [0.0, 1.0, -1.0, 8.0, -8.0, 5e-324, -5e-324])


def _erfc_inputs(rng):
    return _erf_like_inputs(rng, -10, 30, lambda r: r.uniform(26.5, 27.3, 1000),
                            [0.0, 1.0, -1.0, 8.0, -8.0, 26.55, 27.23, 5e-324])


def _erfcx_inputs(rng):
    x = np.concatenate([grid(-26.6, 30, 4001), loguniform(rng, 1e-300, 1e308, 2000), -loguniform(rng, 1e-300, 26.6, 1500),
                        around([0.0, 1.0, -1.0, 8.0, 1e8, -26.6])])
    return (x,), np.isfinite(x)


def _ndtr_inputs(rng):
    return _erf_like_inputs(rng, -40, 10, lambda r: r.uniform(-38.5, -37.0, 1000),
                            [0.0, SQRT2, -SQRT2, 8 * SQRT2, -8 * SQRT2, -37.5, -38.47])


def _log_ndtr_inputs(rng):
    return _erf_like_inputs(rng, -50, 40, lambda r: r.uniform(37.0, 38.5, 1000),
                            [0.0, SQRT2, -SQRT2, 8 * SQRT2, -8 * SQRT2, 37.5, 38.47])


def _ndtri_inputs(rng):
    tail = np.exp(-25.0)  # where AS241 switches its tail rational (s = 5)
    p = np.concatenate([rng.uniform(0, 1, 3000), loguniform(rng, 5e-324, 0.5, 2500), 1 - loguniform(rng, 2**-53, 0.5, 2000),
                        around([0.075, 0.925, 0.5, tail, 1 - tail, 5e-324, 2.5e-317])])
    p = p[(p > 0) & (p < 1)]
    return (p,), np.ones(p.size, dtype=bool)


def _erfinv_inputs(rng):
    tail = 1 - 2 * np.exp(-25.0)
    y = np.concatenate([rng.uniform(-1, 1, 3000), loguniform(rng, 5e-324, 0.5, 2500, signed=True),
                        (1 - loguniform(rng, 2**-53, 0.5, 2000)) * rng.choice([-1.0, 1.0], 2000),
                        around([0.85, -0.85, tail, -tail, 0.0, 5e-324, -5e-324])])
    y = y[np.abs(y) < 1]
    return (y,), np.ones(y.size, dtype=bool)


def _erfcinv_inputs(rng):
    tail = 2 * np.exp(-25.0)
    z = np.concatenate([rng.uniform(0, 2, 3000), loguniform(rng, 5e-324, 1, 2500), 2 - loguniform(rng, 2**-52, 1, 1500),
                        1 + loguniform(rng, 1e-16, 0.5, 500, signed=True), around([0.15, 1.85, 1.0, tail, 2 - tail, 5e-324])])
    z = z[(z > 0) & (z < 2)]
    return (z,), np.ones(z.size, dtype=bool)


def _a(v):
    return v if isinstance(v, (arb, acb)) else arb(v)


def _expit(x):
    if x >= 0:
        return 1 / (1 + (-x).exp())
    e = x.exp()
    return e / (1 + e)


def _log_expit(x):
    if x >= 0:
        return -(-x).exp().log1p()
    return x - x.exp().log1p()


def _ref_erfcx(x):
    if x > 1:  # erfcx(x) = U(1/2, 1/2, x^2) / sqrt(pi): no exp(x^2) to cancel for a huge x
        return (x * x).hypgeom_u(0.5, 0.5) / _sqrt_pi()
    return (x * x).exp() * x.erfc()


def _ndtr(x):
    """Phi(x) (arb in, arb out)."""
    x = _a(x)
    return (-x / arb(2).sqrt()).erfc() / 2


def _npdf(x):
    return (-x * x / 2).exp() / (2 * arb.pi()).sqrt()


def _log_ncdf(x):
    """log Phi(x) (arb in, arb out). x > 0: log1p(-Phi(-x)) keeps a Phi of 1 - 1e-300; x < -2:
    -x^2/2 + log U(1/2, 1/2, x^2/2) - log(2 sqrt(pi)), since Phi(x) = e^(-x^2/2) U(1/2, 1/2, x^2/2) /
    (2 sqrt pi) for x < 0 (no e^(-x^2/2) to underflow or to lose its digits at x = -1e150).
    The branch is chosen on the midpoint; each formula holds on its whole side with margin."""
    x = _a(x)
    m = x.mid()
    if m > 0:
        return (-_ndtr(-x)).log1p()
    if m > -2:
        return _ndtr(x).log()
    h = x * x / 2
    return -h + h.hypgeom_u(0.5, 0.5).log() - (2 * _sqrt_pi()).log()


def _ncdf_slope(x):
    """d/dx log Phi(x) = phi / Phi; sqrt(2) / U(1/2, 1/2, x^2/2) for x < -2 (the same identity)."""
    x = _a(x)
    if x.mid() < -2:
        return arb(2).sqrt() / (x * x / 2).hypgeom_u(0.5, 0.5)
    return _npdf(x) / _ndtr(x)


def _ref_log_ndtr(x):
    return _log_ncdf(x)


def _ref_ndtri(p):
    return -arb(2).sqrt() * (2 * p).erfcinv()  # Arb's erfcinv is a rigorous ball (no root to certify)


# ---- S3: gamma, beta and zeta (docs/api/special.md §3, §7). Guaranteed (relative 1e-14,
# user decision 2026-11-14): gamma, rgamma, Riemann zeta everywhere; gammaln, digamma,
# polygamma for x > 0; beta for a, b > 0; Hurwitz zeta for q > 0; binom for n >= 0. The
# rest -- and all of betaln -- is held to |error| <= 1e-14 max(1, |value|) (test_reference).

def _noninteger(x):
    return x[x != np.floor(x)]


def _near_poles(rng, n, reach):
    """-k + d for k = 0..reach, d both sides and of every size down to 1e-15."""
    k = rng.integers(0, reach, n).astype(float)
    d = loguniform(rng, 1e-15, 0.5, n) * rng.choice([-1.0, 1.0], n)
    return _noninteger(-k + d)


def _gamma_inputs(rng):
    x = np.concatenate([loguniform(rng, 1e-300, 171.6, 2500), grid(0.01, 171.6, 1500), np.arange(1.0, 24.0),
                        _near_poles(rng, 1500, 180), _noninteger(rng.uniform(-184.5, 0, 1500)),
                        around([0.5, 1.0, 2.0, 2.5, 13.0, 23.0, 171.6, 5e-324])])
    x = x[(x != 0) & (x <= 171.62)]
    return (x,), np.ones(x.size, dtype=bool)


def _rgamma_inputs(rng):
    (x,), g = _gamma_inputs(rng)
    x = np.concatenate([x, rng.uniform(171.6, 178.5, 500), -loguniform(rng, 5e-324, 1e-10, 200)])
    return (x,), np.ones(x.size, dtype=bool)


def _gammaln_inputs(rng):
    x = np.concatenate([loguniform(rng, 5e-324, 1e305, 2500), 1 + loguniform(rng, 1e-17, 0.5, 800, signed=True),
                        2 + loguniform(rng, 1e-16, 0.5, 800, signed=True), grid(0.01, 20, 800),
                        _near_poles(rng, 1000, 1000), -_noninteger(loguniform(rng, 1e-300, 1e15, 800)),
                        around([-2.4570247382208006, -2.7476826467274127, 0.5, 1.5, 2.5, 13.0])])
    x = x[x != 0]
    return (x,), x > 0


def _digamma_inputs(rng):
    root = 1.4616321449683623
    x = np.concatenate([loguniform(rng, 1e-300, 1e300, 2500), root + loguniform(rng, 1e-17, 0.3, 800, signed=True),
                        grid(0.01, 20, 800), _near_poles(rng, 1000, 1000), -_noninteger(loguniform(rng, 1e-3, 1e15, 800)),
                        around([root, 1.0, 2.0, 10.0])])
    x = x[x != 0]
    return (x,), x > 0


def _polygamma_inputs(rng):
    n = np.concatenate([rng.integers(1, 31, 3000), np.full(300, 50), np.full(300, 100), np.full(300, 170),
                        rng.integers(1, 6, 1000)]).astype(float)
    x = np.concatenate([loguniform(rng, 1e-3, 1e6, 3900), _near_poles(rng, 1000, 30)[:1000]])
    m = min(n.size, x.size)
    n, x = n[:m], x[:m]
    return (n, x), x > 0


def _beta_inputs(rng):
    m = 2000
    a = np.concatenate([loguniform(rng, 1e-5, 3000, m), loguniform(rng, 1e4, 1e15, m), loguniform(rng, 1e-300, 1e-5, 500),
                        _noninteger(rng.uniform(-30, 30, 1000))])
    b = np.concatenate([loguniform(rng, 1e-5, 3000, m), loguniform(rng, 1e-3, 60, m), loguniform(rng, 1e-3, 10, 500),
                        _noninteger(rng.uniform(-30, 30, 1000))])
    k = min(a.size, b.size)
    a, b = a[:k], b[:k]
    return (a, b), (a > 0) & (b > 0)


def _betaln_inputs(rng):
    m = 1500
    a = np.concatenate([loguniform(rng, 1e-300, 1e300, m), loguniform(rng, 1e-3, 50, m), 1 + loguniform(rng, 1e-12, 1, m)])
    b = np.concatenate([loguniform(rng, 1e-300, 1e300, m), loguniform(rng, 1e-3, 50, m), 1 + loguniform(rng, 1e-12, 1, m)])
    return (a, b), np.zeros(a.size, dtype=bool)  # betaln: the mixed bound everywhere


def _binom_inputs(rng):
    m = 2500
    n = np.concatenate([rng.uniform(0, 1000, m), rng.uniform(0, 50, m), rng.integers(-40, 60, 800).astype(float),
                        rng.uniform(-30, 0, 800)])
    k = np.concatenate([rng.uniform(0, 1, m) * (n[:m] + 1), rng.uniform(0, 80, m), rng.integers(0, 30, 800).astype(float),
                        rng.uniform(0, 20, 800)])
    return (n, k), n >= 0


def _zeta_inputs(rng):
    x = np.concatenate([1 + loguniform(rng, 1e-15, 1e3, 2500), rng.uniform(0, 1, 1000), -loguniform(rng, 1e-15, 260, 2500),
                        around([0.0, 1.0, 2.0, -1.0, -2.0, -171.0, -250.5]), -2 * rng.integers(1, 130, 50).astype(float)])
    x = x[x != 1]
    return (x,), np.ones(x.size, dtype=bool)


def _hurwitz_inputs(rng):
    m = 3500
    x = np.concatenate([1 + loguniform(rng, 1e-12, 100, m), rng.integers(2, 12, 800).astype(float)])
    q = np.concatenate([loguniform(rng, 1e-3, 1e10, m), _noninteger(rng.uniform(-20, 0, 1000))[:800]])
    k = min(x.size, q.size)
    return (x[:k], q[:k]), q[:k] > 0


def _ref_gammaln(x):
    if x > 0:
        return x.lgamma()
    return arb.pi().log() - abs(x.sin_pi()).log() - (1 - x).lgamma()  # Gamma(x) Gamma(1-x) = pi / sin(pi x)


def _ref_polygamma(n, x):
    return acb(x).polygamma(acb(int(float(n)))).real


def _ref_binom(n, k):
    if k == k.floor() and k >= 0:  # the falling factorial (S3 user decision)
        r = arb(1)
        for i in range(1, int(float(k)) + 1):
            r *= (n - k + i) / i
        return r
    return (n + 1).gamma() * (k + 1).rgamma() * (n - k + 1).rgamma()


def _ref_beta(a, b):
    return a.gamma() * b.gamma() * (a + b).rgamma()


def _lbeta(a, b):
    """log B(a, b) for a, b > 0."""
    return a.lgamma() + b.lgamma() - (a + b).lgamma()


def _ref_betaln(a, b):
    if a > 0 and b > 0:
        return _lbeta(a, b)
    return abs(_ref_beta(a, b)).log()


def _ref_zeta_hurwitz(x, q):
    if q > 0:
        return x.zeta(q)
    return acb(x).zeta(acb(q)).real  # q < 0 (the set pairs it with integer x): real, beyond arb's zeta


def _zero_at_zero(f):
    return lambda *a: 0.0 if a[0] == 0 else f(*a)


# S4: the incomplete gamma and beta functions and their inverses. Guaranteed everywhere
# (S4 user decision); the sets cover a <= 1e6 (gamma) and a, b <= 1e5 (beta). The inverses
# store `scale` = max(1, kappa), kappa = |F / (x F')|: their contract is |dx / x| <= 1e-14 scale.


def _gammainc_inputs(rng):
    m = 600
    a = np.concatenate([loguniform(rng, 1e-3, 1e3, m), loguniform(rng, 1e-3, 1.0, m // 2), loguniform(rng, 20, 1e6, m),
                        loguniform(rng, 1e3, 1e6, m // 2), loguniform(rng, 1e-300, 1e-3, m // 2), loguniform(rng, 1, 50, m // 2)])
    x = np.concatenate([a[:m] * loguniform(rng, 1e-3, 1e3, m), loguniform(rng, 1e-3, 1.1, m // 2),
                        a[m + m // 2:2 * m + m // 2] * (1 + rng.uniform(-0.4, 0.4, m)),
                        a[2 * m + m // 2:3 * m] * (1 + rng.normal(0, 1, m // 2) * 5 / np.sqrt(a[2 * m + m // 2:3 * m])),
                        loguniform(rng, 1e-300, 10, m // 2), loguniform(rng, 1e-300, 1e-3, m // 2)])
    # region boundaries: x = 0.5, 1.1; a = 20; |x - a| = 0.4 a; the scipy errors of S4
    ea = np.array([0.3, 0.3, 0.8, 0.8, 21.0, 21.0, 100.0, 100.0, 1e-300, 5e-324, 1.0, 0.5, 2.0])
    ex = np.array([0.5, 0.5000000000000001, 1.1, 1.0999999999999999, 21 * 1.4, 21 * 0.6, 140.0, 60.0, 5e-324, 0.5,
                   5e-324, 1e-320, 1e-300])
    return (np.concatenate([a, ea]), np.concatenate([x, ex])), np.ones(a.size + ea.size, dtype=bool)


def _gamma_p_series(a, x):
    """P(a, x) = x^a e^-x / Gamma(a+1) sum_k t_k, t_k = x^k / ((a+1)...(a+k)) (a, x > 0). Past
    k = x - a the ratio x / (a+k+1) of the terms only falls, so the tail after t_k is at most
    t_k r / (1 - r), r = x / (a+k+1)."""
    tol = _trunc()
    t = s = arb(1)
    k = 0
    while True:
        k += 1
        t = t * x / (a + k)
        s += t
        if k % 8 == 0:
            r = x / (a + k + 1)
            if r < 1:
                tail = geometric_tail(t, r)
                if tail <= tol * s:
                    break
        if k > MAX_TERMS:
            raise NotSettled("P series")
    return (a * x.log() - x - (a + 1).lgamma()).exp() * add_error(s, tail)


def _gamma_q_series(a, x):
    """Q(a, x) for x > a: integrating by parts k times,
        Gamma(a, x) = x^(a-1) e^-x (sum_{j<k} c_j + c_k R_k),  c_j = (a-1)...(a-j) / x^j,
    with 0 <= R_k = Gamma(a-k, x) / (x^(a-k-1) e^-x) <= max(1, x / (x - a + k + 1)) (Gamma(s, x)
    <= x^(s-1) e^-x x / (x - s + 1) for s >= 1, <= x^(s-1) e^-x for s < 1). The terms fall while
    |a - k - 1| < x; NotSettled if they turn before the remainder is small enough (x too
    small for the expansion: the caller takes 1 - P)."""
    tol = _trunc()
    s, c, k = arb(0), arb(1), 0
    while True:
        rem = abs(c) * (1 + x / (x - a + k + 1))  # >= |c_k| max(1, x / (x - a + k + 1))
        if k > 0 and rem <= tol * abs(s):
            break
        if abs(a - k - 1) >= x:
            raise NotSettled("Q expansion turns")
        s += c
        k += 1
        c = c * (a - k) / x
        if k > MAX_TERMS:
            raise NotSettled("Q expansion")
    return ((a - 1) * x.log() - x - a.lgamma()).exp() * add_error(s, rem)


def _gammainc(a, x, upper=False):
    """P(a, x), or Q(a, x) when upper (a > 0, x >= 0; arb in, arb out). Q from its expansion
    above the mean (x > a + 3 sqrt(a) + 1) where that converges, else P from its series; the
    other one as 1 minus it (a cancellation widens the ball, and the precision doubles past it).
    Arb's own gamma_lower does not settle for a >~ 1000 (its radius stops shrinking)."""
    a, x = _a(a), _a(x)
    if x == 0:
        return arb(1) if upper else arb(0)
    am, xm = float(a.mid()), float(x.mid())
    if xm > am + 3 * math.sqrt(am) + 1:
        try:
            q = _gamma_q_series(a, x)
            return q if upper else 1 - q
        except NotSettled:
            pass
    p = _gamma_p_series(a, x)
    return 1 - p if upper else p


def _ref_gammainc(a, x):
    return _gammainc(a, x, False)


def _ref_gammaincc(a, x):
    return _gammainc(a, x, True)


def _beta_sets(rng, m):
    a = np.concatenate([loguniform(rng, 1e-2, 1e3, m), loguniform(rng, 1e2, 1e5, m), loguniform(rng, 1e-3, 1, m),
                        loguniform(rng, 1e-3, 10, m), loguniform(rng, 1e-2, 1e2, m)])
    b = np.concatenate([loguniform(rng, 1e-2, 1e3, m), loguniform(rng, 1e2, 1e5, m), loguniform(rng, 1e-3, 1, m),
                        loguniform(rng, 1e2, 1e5, m), loguniform(rng, 1e-2, 1e2, m)])
    return a, b


def _betainc_inputs(rng):
    m = 400
    a, b = _beta_sets(rng, m)
    mean = a[m:2 * m] / (a[m:2 * m] + b[m:2 * m])
    sd_ = np.sqrt(a[m:2 * m] * b[m:2 * m] / ((a[m:2 * m] + b[m:2 * m]) ** 2 * (a[m:2 * m] + b[m:2 * m] + 1)))
    x = np.concatenate([rng.uniform(0, 1, m), np.clip(mean + rng.normal(0, 3, m) * sd_, 1e-300, 1 - 1e-16),
                        rng.uniform(0, 1, m), loguniform(rng, 1e-8, 1e-1, m), 1 - loguniform(rng, 1e-16, 1e-2, m)])
    ea = np.array([2.0, 0.5, 3.0, 1e-300])
    eb = np.array([1e300, 0.5, 5.0, 1e-300])
    ex = np.array([5e-324, 0.5, 0.375, 0.5])
    a, b, x = np.concatenate([a, ea]), np.concatenate([b, eb]), np.concatenate([x, ex])
    keep = (x > 0) & (x < 1)
    return (a[keep], b[keep], x[keep]), np.ones(keep.sum(), dtype=bool)


def _beta_series(a, b, x, y):
    """I_x(a, b) = x^a y^b / (a B(a, b)) sum_k t_k, t_{k+1} = t_k x (a+b+k) / (a+1+k), y = 1 - x
    (2F1(a+b, 1; a+1; x)). Beyond term k the ratios are at most r = x (a+b+k) / (a+1+k) when
    b >= 1 (they fall toward x), r = x when b < 1 (they rise toward it); the tail is at most
    t_k r / (1 - r)."""
    tol = _trunc()
    t = s = arb(1)
    k = 0
    while True:
        t = t * x * (a + b + k) / (a + 1 + k)
        k += 1
        s += t
        if k % 8 == 0:
            r = x * (a + b + k) / (a + 1 + k) if b >= 1 else x
            if r < 1:
                tail = geometric_tail(t, r)
                if tail <= tol * s:
                    break
        if k > MAX_TERMS:
            raise NotSettled("beta series")
    return (a * x.log() + b * y.log() - a.log() - _lbeta(a, b)).exp() * add_error(s, tail)


def _beta_cost(a, b, x):
    """About how many terms _beta_series(a, b, x) takes: up to its largest term, then ~85 / (1 - x)."""
    if x >= 1:
        return math.inf
    return max(0.0, (x * (a + b) - a - 1) / (1 - x)) + 85 / (1 - x)


def _ibeta(a, b, x, y):
    """I_x(a, b), y = 1 - x given (arb in, arb out; a, b > 0): from its own series, or as
    1 - I_y(b, a) when that series is much shorter (by _beta_cost; a cancellation there widens
    the ball and the precision doubles past it). (Arb's beta_lower does not settle for
    a, b ~ 1e4 near the mean.)"""
    a, b, x, y = _a(a), _a(b), _a(x), _a(y)
    if x == 0 or y == 0:
        return arb(0) if x == 0 else arb(1)
    af, bf, xf, yf = (float(v.mid()) for v in (a, b, x, y))
    if _beta_cost(af, bf, xf) <= 4 * _beta_cost(bf, af, yf) + 200:
        return _beta_series(a, b, x, y)
    return 1 - _beta_series(b, a, y, x)


def _ibeta_pair(a, b, x, y):
    """(I_x(a, b), 1 - I_x(a, b)) with y = 1 - x given (arb in, arb out; a, b > 0): each
    evaluated as _ibeta (1 minus the other would lose a tiny value)."""
    return _ibeta(a, b, x, y), _ibeta(b, a, y, x)


def _ref_betainc(a, b, x):
    return _ibeta(a, b, x, 1 - x)  # 1 - x is exact at the doubled precisions; a ball before


def _ref_betaincc(a, b, x):
    return _ibeta(b, a, 1 - x, x)


def _gammaincinv_inputs(rng):
    m = 500
    a = np.concatenate([loguniform(rng, 1e-3, 1e3, 2 * m), loguniform(rng, 1e3, 1e6, m), loguniform(rng, 1e-300, 1e-3, m)])
    p = np.concatenate([loguniform(rng, 1e-300, 0.999, m), rng.uniform(0, 1, m), loguniform(rng, 1e-300, 0.999, m),
                        rng.uniform(0, 1, m)])
    ea, ep = np.array([5e-324, 1e-300, 2.0, 1.0]), np.array([0.5, 0.5, 1e-320, 0.5])
    a, p = np.concatenate([a, ea]), np.concatenate([p, ep])
    keep = (p > 0) & (p < 1)
    return (a[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


def _betaincinv_inputs(rng):
    m = 300
    a, b = _beta_sets(rng, m)
    p = np.concatenate([loguniform(rng, 1e-300, 0.999, 3 * m), rng.uniform(0, 1, 2 * m)])
    rng.shuffle(p)
    ea, eb, ep = np.array([2.0, 0.01]), np.array([3.0, 36.0]), np.array([5e-324, 3e-169])
    a, b, p = np.concatenate([a, ea]), np.concatenate([b, eb]), np.concatenate([p, ep])
    keep = (p > 0) & (p < 1)
    return (a[keep], b[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


def _sign(fn):
    """The sign of fn() (an arb), at doubling precision until it is certain."""
    p = arbref.START_PREC
    while p <= arbref.MAX_PREC:
        with precision(p):
            v = fn()
        if v > 0:
            return 1
        if v < 0:
            return -1
        if v == 0:
            return 0
        p *= 2
    raise NotSettled("sign")




def _TINY():
    return arb(2) ** -1075  # a root below it rounds to 0


def _gamma_inverse(upper):
    """gammaincinv (P(a, x) = p) or gammainccinv (Q(a, x) = p): the root of log F(x) = log t, F
    the one of P, Q whose value there is <= 1/2 (t = p or 1 - p, exact), certified by a sign
    change (arbref.solve_log_settled). Returns (x, kappa), kappa = |p / (x F'(x))|."""
    name = "gammainccinv" if upper else "gammaincinv"

    def f(a, p):
        x0 = _start2(name, a, p)
        use_p = (p <= 0.5) != upper  # the function whose value at the root is <= 1/2
        t = p if p <= 0.5 else 1 - p  # exact (Sterbenz)
        F = lambda x: _gammainc(a, x, not use_p)
        dP = lambda x: ((a - 1) * x.log() - x - a.lgamma()).exp()
        if x0 == 0:  # ours underflowed: the reference is 0 if the root is below 2^-1075
            if _sign(lambda: (F(_TINY()) - t) * (1 if use_p else -1)) >= 0:
                return 0.0, 1.0
            x0 = 2.0**-1074
        x = solve_log_settled(lambda x: F(x).log() - t.log(), lambda x: (1 if use_p else -1) * x * dP(x) / F(x),
                              x0, increasing=use_p)
        return x, _kappa(lambda: p / (x * dP(x)))
    return f


def _beta_inverse(upper):
    """betaincinv (I_x(a, b) = p) or betainccinv (1 - I = p). The unknown is x below 1/2 and
    y = 1 - x above (each exact near its own end), and the equation is the one of I = ti,
    1 - I = tc whose value is <= 1/2 (I_x(a, b) or I_y(b, a), each from its own series), so
    log F - log t keeps a far tail; the root is certified by a sign change. Ours at 0 or 1 is
    checked: the reference is 0 (1) only if the root is below 2^-1075 (above 1 - 2^-54).
    Returns (x, kappa), kappa = |p / (x I'(x))|."""
    name = "betainccinv" if upper else "betaincinv"

    def f(a, b, p):
        x0 = _start2(name, a, b, p)
        _no_start(x0)
        ti, tc = (1 - p, p) if upper else (p, 1 - p)  # I and 1 - I at the root, exact
        dens = lambda x, y: ((a - 1) * x.log() + (b - 1) * y.log() - _lbeta(a, b)).exp()  # dI/dx
        if x0 == 0:
            if _sign(lambda: _ibeta(a, b, _TINY(), 1 - _TINY()) - ti) >= 0:
                return 0.0, 1.0
            x0 = 2.0**-1074
        if x0 == 1:
            e = arb(2.0**-54)
            if _sign(lambda: _ibeta(b, a, e, 1 - e) - tc) > 0:  # I(1 - 2^-54) < ti: the root rounds to 1
                return 1.0, 1.0
            x0 = 1 - 2.0**-53
        use_i = ti <= 0.5  # solve I = ti (else 1 - I = tc)
        t = ti if use_i else tc
        if x0 <= 0.5:  # in x: I increases
            F = (lambda x: _ibeta(a, b, x, 1 - x)) if use_i else (lambda x: _ibeta(b, a, 1 - x, x))
            sgn = 1 if use_i else -1
            x = solve_log_settled(lambda x: F(x).log() - t.log(), lambda x: sgn * x * dens(x, 1 - x) / F(x), x0,
                                  increasing=use_i, hi=arb(1))
        else:  # in y = 1 - x: 1 - I increases
            G = (lambda y: _ibeta(a, b, 1 - y, y)) if use_i else (lambda y: _ibeta(b, a, y, 1 - y))
            sgn = -1 if use_i else 1
            y = solve_log_settled(lambda y: G(y).log() - t.log(), lambda y: sgn * y * dens(1 - y, y) / G(y), 1 - x0,
                                  increasing=not use_i, hi=arb(1))
            x = _exact(lambda: 1 - y)
        return x, _kappa(lambda: p / (x * dens(x, 1 - x)))
    return f


def _start2(name, *v):
    """Our own value (a float) at float inputs (arbs are converted): the inverses' Newton start."""
    import stardust as sd
    return float(getattr(sd.special, name)(*(float(t) for t in v)))


def _expi_inputs(rng):
    m = 1000
    x = np.concatenate([loguniform(rng, 1e-300, 716, m), 0.3725074107813666 + rng.uniform(-0.08, 0.08, m // 2),
                        0.3725074107813666 + loguniform(rng, 1e-17, 1e-3, m // 4, signed=True), rng.uniform(0.45, 40, m // 2),
                        -loguniform(rng, 1e-300, 745, m), around([0.3, 0.45, 40.0, 709.78, 716.0, -709.0, -738.0])])
    return (x,), np.ones(x.size, dtype=bool)


def _expn_inputs(rng):
    m = 900
    n = np.concatenate([rng.integers(0, 10, m), rng.integers(10, 100_000, m), rng.integers(1, 30, m),
                        np.array([1, 5, 50, 2**31, 2**40])]).astype(float)
    x = np.concatenate([loguniform(rng, 1e-10, 740, m), loguniform(rng, 1e-3, 740, m), rng.uniform(0.5, 2, m),
                        np.array([712.0, 725.0, 725.0, 1.0, 1.0])])
    return (n, x), np.ones(n.size, dtype=bool)


def _ellip_inputs(rng):
    m = 1000
    x = np.concatenate([rng.uniform(0, 1, m), 1 - loguniform(rng, 1e-16, 1e-2, m), -loguniform(rng, 1e-300, 1e300, m),
                        around([0.5, 1 - 2.0**-34, 0.0, -1.0])])
    x = x[x < 1]
    return (x,), np.ones(x.size, dtype=bool)


# S6: Bessel and Airy (S6 user decisions). The oscillatory functions store `scale` =
# max(1, M / |f|) where the envelope bound applies (x >= 0.99 z1, z1 the function's first
# positive zero; every x for a negative non-integer order; x < 0 for Airy), so the test's
# relative bound times scale is |error| <= 1e-14 M. iv with a negative non-integer order:
# scale = (|I_a| + |(2/pi) sin(a pi) K_a|) / |I_-a|.


def _bessel01_inputs(lo, hi, signed=False):
    def f(rng):
        m = 1200
        x = np.concatenate([loguniform(rng, 1e-300, hi, m), loguniform(rng, lo, hi, m)])
        if signed:
            x *= rng.choice([-1.0, 1.0], x.size)
        return (x,), np.ones(x.size, dtype=bool)
    return f


def _jy01_inputs(rng):
    m = 900
    x = np.concatenate([loguniform(rng, 1e-300, 10, m), loguniform(rng, 1, 1e4, m), loguniform(rng, 1e4, 1e300, m // 3)])
    return (x,), np.ones(x.size, dtype=bool)


def _first_zero(kind, nu):
    """The first positive zero of J_nu (kind 'j') or Y_nu ('y'), nu >= 0, to ~1e-13 (it only
    places the scale's boundary x = 0.99 z1). Both functions keep one sign on (0, z1), and
    z1 > nu: scanned from nu in steps of 0.1 (below the gap between the first zeros of J and
    Y, which interlace), then bisected; each sign certain (_sign: a large order needs more
    than 80 bits)."""
    f = (lambda x: arb(x).bessel_j(nu)) if kind == "j" else (lambda x: arb(x).bessel_y(nu))
    s0 = 1 if kind == "j" else -1  # the sign on (0, z1)
    inside = lambda x: _sign(lambda: f(x)) == s0
    lo = max(float(nu), 0.01)
    hi = lo + 0.1
    while inside(hi):
        lo, hi = hi, hi + 0.1
    for _ in range(50):
        mid = (lo + hi) / 2
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _envelope_scale(kind, n):
    """scale(args, value) for J (kind 'j') or Y ('y') of order n (None: the first argument)."""
    def scale(args, v):
        nu = float(args[0]) if n is None else float(n)
        x = float(args[-1])
        if x <= 0:
            return 1.0
        if nu >= 0 and (x < 0.99 * nu or x < 0.99 * _first_zero(kind, nu)):  # z1 > nu
            return 1.0
        if v == 0:
            return float("inf")
        return _scale_of(lambda: (arb(x).bessel_j(nu) ** 2 + arb(x).bessel_y(nu) ** 2).sqrt() / abs(v))
    return scale


def _iv_scale(args, v):
    nu, x = float(args[0]), float(args[1])
    if nu >= 0 or nu == math.floor(nu) or x <= 0:
        return 1.0
    if v == 0:
        return float("inf")
    a = -nu
    return _scale_of(lambda: (abs(arb(x).bessel_i(a)) + abs(2 / arb.pi() * arb(a).sin_pi() * arb(x).bessel_k(a))) / abs(v))


def _order_inputs(lo_x, hi_x, negative=True):
    def f(rng):
        m = 500
        nu = np.concatenate([rng.uniform(-50 if negative else 0, 50, 3 * m), rng.uniform(-1000 if negative else 0, 1000, m // 2),
                             rng.integers(-20 if negative else 0, 21, m // 2).astype(float)])
        x = np.concatenate([loguniform(rng, lo_x, hi_x, 3 * m), loguniform(rng, 30, hi_x, m // 2), loguniform(rng, lo_x, hi_x, m // 2)])
        return (nu, x), np.ones(nu.size, dtype=bool)
    return f


def _airy_inputs(rng):
    m = 700
    x = np.concatenate([rng.uniform(-2, 1, m), loguniform(rng, 1, 110, m), -rng.uniform(2, 9.7, m),
                        -loguniform(rng, 9.7, 1e12, m), loguniform(rng, 1e-300, 1e-3, m // 4, signed=True)])
    return (x,), np.ones(x.size, dtype=bool)


def _airy_part(i):
    return lambda x: x.airy()[i]


def _airy_scale(i):
    def scale(args, v):
        x = float(args[0])
        if x >= 0:
            return 1.0
        if v == 0:
            return float("inf")

        def ratio():
            ai, aip, bi, bip = arb(x).airy()
            return ((ai * ai + bi * bi).sqrt() if i % 2 == 0 else (aip * aip + bip * bip).sqrt()) / abs(v)
        return _scale_of(ratio)
    return scale


# name -> scale(args, value), for the functions whose bound is not plain relative
SCALES = {
    "j0": _envelope_scale("j", 0), "j1": _envelope_scale("j", 1), "y0": _envelope_scale("y", 0),
    "y1": _envelope_scale("y", 1), "jv": _envelope_scale("j", None), "yv": _envelope_scale("y", None),
    "iv": _iv_scale, "airy_ai": _airy_scale(0), "airy_aip": _airy_scale(1), "airy_bi": _airy_scale(2),
    "airy_bip": _airy_scale(3),
}


# S7 (user decisions): lambertw everywhere; hyp1f1, hyp2f1 guaranteed only where a form with
# positive terms exists (the flag per point), the rest is their loose region.


def _lambertw_inputs(branch):
    def f(rng):
        m = 1500
        if branch == 0:
            x = np.concatenate([loguniform(rng, 1e-300, 1e300, m), -loguniform(rng, 1e-300, 0.367879441, m // 2),
                                -np.exp(-1) + loguniform(rng, 1e-16, 0.05, m // 2)])
        else:
            x = np.concatenate([-loguniform(rng, 1e-300, 0.367879441, m), -np.exp(-1) + loguniform(rng, 1e-16, 0.05, m // 2)])
        return (x,), np.ones(x.size, dtype=bool)
    return f


def _hyp1f1_inputs(rng):
    m = 400
    a = np.concatenate([rng.uniform(0, 50, m), rng.uniform(0, 50, m), rng.uniform(0, 5, m), rng.uniform(0, 20, m),
                        rng.uniform(-50, 0, m // 2), rng.uniform(-50, 50, m // 2)])
    b = np.concatenate([rng.uniform(0.01, 50, m), a[m:2 * m] + rng.uniform(0, 30, m), a[2 * m:3 * m] + rng.uniform(0, 5, m),
                        a[3 * m:4 * m] + rng.uniform(0, 20, m), rng.uniform(0.1, 50, m // 2), rng.uniform(-50, 0, m // 2)])
    x = np.concatenate([loguniform(rng, 1e-3, 700, m), -loguniform(rng, 1e-3, 700, m), rng.uniform(-10, 10, m),
                        -loguniform(rng, 700, 1e5, m), rng.uniform(-100, 100, m // 2), rng.uniform(-100, 100, m // 2)])
    g = (a >= 0) & (b > 0) & ((x >= 0) | (b >= a))
    return (a, b, x), g


def _hyp2f1_inputs(rng):
    m = 500
    a = np.concatenate([rng.uniform(0, 20, m), rng.uniform(0, 10, m // 2), rng.uniform(0, 10, m // 2), rng.uniform(-20, 20, m)])
    b = np.concatenate([rng.uniform(0, 20, m), rng.uniform(0, 10, m // 2), rng.uniform(0, 10, m // 2), rng.uniform(-20, 20, m)])
    c = np.concatenate([np.minimum(a[:m], b[:m]) + rng.uniform(0.01, 20, m), rng.uniform(0.01, 10, m // 2),
                        rng.uniform(0.01, 10, m // 2), rng.uniform(-20, 20, m)])
    z = np.concatenate([rng.uniform(-1, 0.9, m), 1 - loguniform(rng, 1e-12, 0.1, m // 2), -loguniform(rng, 1, 1e8, m // 2),
                        rng.uniform(-10, 0.999, m)])
    g = (a >= 0) & (b >= 0) & (c > 0) & (c >= np.minimum(a, b)) & (z >= -1) & (z <= 0.9)
    return (a, b, c, z), g


# T1: the distribution functions (docs/api/special.md §13). Measured over the whole domain
# first (the guaranteed regions are chosen from these measurements, T1 decision). The
# inverses store scale = max(1, kappa), kappa = |p / (x F'(x))|, as in S4.


def _t_tail(df, t):  # P(T <= -|t|)
    z = t * t
    return _ibeta(df / 2, arb(0.5), df / (df + z), z / (df + z)) / 2


def _ref_stdtr(df, t):
    if t == 0:
        return arb(0.5)
    tail = _t_tail(df, t)
    return tail if t < 0 else 1 - tail


def _stdtr_inputs(rng):
    m = 500
    df = np.concatenate([loguniform(rng, 1e-2, 1e6, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 10, 1e6, m)])
    t = np.concatenate([loguniform(rng, 1e-4, 1e4, m, signed=True), loguniform(rng, 1e-3, 1e8, m, signed=True),
                        rng.normal(0, 1, m) * 6])
    edf, et = np.array([1.0, 1.0, 3.0, 1e-2, 1e6, 0.5]), np.array([1e150, -1e200, 5e-324, 1e-300, 30.0, -1e10])
    return (np.concatenate([df, edf]), np.concatenate([t, et])), np.ones(3 * m + edf.size, dtype=bool)


def _t_pdf(df, t):
    return ((df + 1) / 2).lgamma().exp() * (df / 2).rgamma() / (df * arb.pi()).sqrt() * (-(df + 1) / 2 * (t * t / df).log1p()).exp()


def _no_start(x0, positive=False):
    if not np.isfinite(x0) or (positive and not x0 > 0):
        raise NotSettled("no start")


def _ref_stdtrit(df, p):
    """Newton in log u, u = |t|, on the log of the tail P(T <= -u) (exact in far tails)."""
    t0 = _start2("stdtrit", df, p)
    _no_start(t0)
    if t0 == 0:
        raise NotSettled("no start")
    lower = p < 0.5
    tail_p = p if lower else 1 - p
    u = solve_log_settled(lambda u: _t_tail(df, u).log() - tail_p.log(), lambda u: -u * _t_pdf(df, u) / _t_tail(df, u),
                          abs(t0), increasing=False)
    t = _exact(lambda: -u) if lower else u
    return t, _kappa(lambda: p / (t * _t_pdf(df, t)))


def _stdtrit_inputs(rng):
    m = 500
    df = np.concatenate([loguniform(rng, 1e-2, 1e6, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 10, 1e6, m)])
    p = np.concatenate([rng.uniform(0, 1, m), loguniform(rng, 1e-300, 0.5, m), 1 - loguniform(rng, 1e-16, 0.5, m)])
    edf, ep = np.array([3.0, 0.06385, 1.0]), np.array([5e-324, 0.26009166350995649, 0.4999999])
    df, p = np.concatenate([df, edf]), np.concatenate([p, ep])
    keep = (p > 0) & (p < 1) & (p != 0.5)
    return (df[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


def _chdtr_inputs(rng):
    m = 600
    v = np.concatenate([loguniform(rng, 1e-2, 1e5, m), loguniform(rng, 1e-2, 10, m)])
    x = np.concatenate([v[:m] * loguniform(rng, 1e-2, 1e2, m), loguniform(rng, 1e-6, 50, m)])
    return (v, x), np.ones(v.size, dtype=bool)


def _ref_chdtri(v, p):
    y, k = _gamma_inverse(True)(v / 2, p)
    return (_exact(lambda: 2 * y) if isinstance(y, arb) else 2 * y), k


def _chdtri_inputs(rng):
    m = 600
    v = np.concatenate([loguniform(rng, 1e-2, 1e5, m), loguniform(rng, 1e-2, 10, m)])
    p = np.concatenate([rng.uniform(0, 1, m), loguniform(rng, 1e-300, 0.999, m)])
    keep = (p > 0) & (p < 1)
    return (v[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


def _ref_fdtr(dfn, dfd, x):
    s = dfn * x + dfd
    return _ibeta(dfn / 2, dfd / 2, dfn * x / s, dfd / s)


def _ref_fdtrc(dfn, dfd, x):
    s = dfn * x + dfd
    return _ibeta(dfd / 2, dfn / 2, dfd / s, dfn * x / s)


def _fdtr_inputs(rng):
    m = 500
    dfn = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 10, 1e4, m)])
    dfd = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 10, 1e4, m), loguniform(rng, 1e-2, 10, m)])
    x = np.concatenate([loguniform(rng, 1e-4, 1e4, m), loguniform(rng, 1e-6, 1e6, m), rng.uniform(0.2, 3, m)])
    return (dfn, dfd, x), np.ones(dfn.size, dtype=bool)


def _ref_fdtri(dfn, dfd, p):
    """The x with fdtr(dfn, dfd, x) = p, solved for x itself (not through w = dfn x / (dfn x + dfd),
    which rounds to 1 for a huge x): Newton in log x on the log of the smaller tail."""
    x0 = _start2("fdtri", dfn, dfd, p)
    _no_start(x0, True)
    a, b = dfn / 2, dfd / 2

    def density(x):  # d fdtr / dx
        s = dfn * x + dfd
        return ((a - 1) * (dfn * x / s).log() + (b - 1) * (dfd / s).log() - _lbeta(a, b)).exp() * dfn * dfd / (s * s)

    lower = p <= 0.5
    tail = (lambda x: _ref_fdtr(dfn, dfd, x)) if lower else (lambda x: _ref_fdtrc(dfn, dfd, x))
    target = p if lower else 1 - p
    x = solve_log_settled(lambda x: tail(x).log() - target.log(), lambda x: (1 if lower else -1) * x * density(x) / tail(x),
                          x0, increasing=lower)
    return x, _kappa(lambda: p / (x * density(x)))


def _fdtri_inputs(rng):
    m = 400
    dfn = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 10, 1e4, m)])
    dfd = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 10, 1e4, m), loguniform(rng, 1e-2, 10, m)])
    p = np.concatenate([rng.uniform(0, 1, m), loguniform(rng, 1e-100, 0.999, m), rng.uniform(0, 1, m)])
    keep = (p > 0) & (p < 1)
    return (dfn[keep], dfd[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


def _pdtr_inputs(rng):
    m = 700
    k = np.concatenate([np.floor(loguniform(rng, 1, 1e4, m)) - 1, rng.integers(0, 30, m).astype(float)])
    mu = np.concatenate([(k[:m] + 1) * loguniform(rng, 0.1, 10, m), loguniform(rng, 1e-3, 60, m)])
    return (k, mu), np.ones(k.size, dtype=bool)


def _bdtr_inputs(rng, real_n=False):
    # (named _binom_inputs until 2026-11-16, which shadowed S3's binom set: binom.npz is that set)
    m = 600
    n = np.concatenate([np.floor(loguniform(rng, 1, 1e4, m)), rng.integers(1, 40, m).astype(float)])
    k = np.floor(rng.uniform(0, 1, 2 * m) * n)
    if real_n:  # nbdtr: any real n > 0, k unbounded by n
        n = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 50, m)])
        k = np.floor(loguniform(rng, 1, 1e4, 2 * m)) - 1
    p = np.concatenate([rng.uniform(0, 1, m), loguniform(rng, 1e-8, 1, m)])
    keep = (p > 0) & (p < 1) & (k >= 0) & ((k < n) | real_n)
    return (k[keep], n[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


def _ref_ndtri_exp(y):
    x0 = _start2("ndtri_exp", y)
    _no_start(x0)
    x = solve_settled(lambda x: _log_ncdf(x) - y, _ncdf_slope, x0, increasing=True)
    return x, _kappa(lambda: y / (x * _ncdf_slope(x)))


def _ndtri_exp_inputs(rng):
    m = 800
    y = np.concatenate([-loguniform(rng, 1e-300, 1e300, m), -rng.uniform(0, 3, m), -loguniform(rng, 1, 1e4, m),
                        -around([0.6931471805599453, 700.0, 1e17])])
    return (y,), np.ones(y.size, dtype=bool)


# T2: the noncentral distributions. Each value is a Poisson mixture of incomplete gamma or
# beta functions; both tails are summed directly as sums of positive terms (1 - the other
# tail would lose a tiny value), each family of incomplete functions by its stable
# recurrence: the lower functions (decreasing in the shape) backward from a start J past
# the mode, the upper ones (increasing) forward from 0. J grows until the bound on the
# terms beyond (geometric: the Poisson weights fall by mu / (J + 1) or faster, the lower
# functions fall, the upper ones are <= 1) is below 2^-max(122, prec) of the sum; the bound is added
# to the ball (arbref.add_error). All take and return arbs.

_MIX_WIDTH_MAX = 2560


def _pois_weight(mu, j):
    """e^-mu mu^j / j! (mu > 0)."""
    return (-mu + j * mu.log() - arb(j + 1).lgamma()).exp()


def _mix_lower(mu, start, term, ratio_down):
    """sum_j w_j L_j, w_j = Poisson(mu) weights and L_j decreasing in j with
    L_{j-1} = L_j + term(j-1); start(J) = L_J, term(j) direct, ratio_down(j) = term(j-1) / term(j).
    Beyond J: w_i L_i <= w_J L_J (mu / (J+1))^(i-J)."""
    if mu == 0:
        return start(0)
    tol, muf, width = _trunc(), float(mu.mid()), 40
    while True:
        J = int(muf + width * math.sqrt(muf + 1) + width)
        L, t, w = start(J), term(J - 1) if J > 0 else arb(0), _pois_weight(mu, J)
        S = w * L
        last = S
        for j in range(J, 0, -1):
            L += t  # L_{j-1}
            w *= j / mu  # w_{j-1}
            if j - 1 > 0:
                t *= ratio_down(j - 1)  # term(j-2)
            S += w * L
        r = mu / (J + 1)
        if r < 1:
            tail = geometric_tail(last, r)
            if tail <= tol * S:
                return add_error(S, tail)
        width *= 2
        if width > _MIX_WIDTH_MAX:
            raise NotSettled("mixture (lower)")


def _mix_upper(mu, first, term0, ratio_up):
    """sum_j w_j U_j with U_j increasing in j, U_{j+1} = U_j + term(j) <= 1;
    first = U_0, term0 = term(0), ratio_up(j) = term(j+1) / term(j). Beyond j > mu:
    sum_{i>j} w_i U_i <= sum_{i>j} w_i <= w_j r / (1 - r), r = mu / (j+1)."""
    if mu == 0:
        return first
    tol, muf = _trunc(), float(mu.mid())
    U, t, w = first, term0, (-mu).exp()
    S = w * U
    j = 0
    while True:
        U += t
        t *= ratio_up(j)
        j += 1
        w *= mu / j
        S += w * U
        if j > muf:
            tail = geometric_tail(w, mu / (j + 1))
            if tail <= tol * S:
                return add_error(S, tail)
        if j > 10 * muf + 10000:
            raise NotSettled("mixture (upper)")


def _ncx2_pair(df, nc, x, upper):
    """chndtr (upper False) or chndtrc: sum_j Poisson(nc/2) P or Q(df/2 + j, x/2) (arb in, arb out)."""
    df, nc, x = _a(df), _a(nc), _a(x)
    a, y, mu = df / 2, x / 2, nc / 2
    T = lambda s: (s * y.log() - y - (s + 1).lgamma()).exp()  # P(s) - P(s+1)
    if upper:
        return _mix_upper(mu, _gammainc(a, y, True), T(a), lambda j: y / (a + j + 1))
    return _mix_lower(mu, lambda J: _gammainc(a + J, y), lambda j: T(a + j), lambda j: (a + j) / y)


def _ncf_pair(dfn, dfd, nc, f, upper):
    """ncfdtr (upper False) or ncfdtrc: sum_j Poisson(nc/2) I_w(dfn/2 + j, dfd/2) or its complement
    (arb in, arb out)."""
    dfn, dfd, nc, f = _a(dfn), _a(dfd), _a(nc), _a(f)
    a, b, mu = dfn / 2, dfd / 2, nc / 2
    s = dfn * f + dfd
    w, v = dfn * f / s, dfd / s
    lw, lv = w.log(), v.log()
    T = lambda c: (c * lw + b * lv - c.log() - _lbeta(c, b)).exp()  # I(c) - I(c+1)
    if upper:
        return _mix_upper(mu, _ibeta(b, a, v, w), T(a), lambda j: w * (a + j + b) / (a + j + 1))
    return _mix_lower(mu, lambda J: _ibeta(a + J, b, w, v), lambda j: T(a + j), lambda j: (a + j) / ((a + j - 1 + b) * w))


@contextmanager
def _alarm_deferred():
    """SIGALRM held while Arb calls back into Python (acb.integral): an exception raised inside
    the callback does not stop the integration, so the handler only records the alarm (the
    integrand then returns NaN and the integration ends at once); the original handler runs after."""
    import signal

    fired = []
    old = signal.signal(signal.SIGALRM, lambda *a: fired.append(a))
    try:
        yield fired
    finally:
        signal.signal(signal.SIGALRM, old)
        if fired and callable(old):
            old(*fired[0])


def _nct_opposite(df, a, t):
    """P(T > t) for t > 0 and d = -a < 0, the tail away from the noncentrality, from the
    definition T = (Z + d) / S, S = sqrt(V / df): int_0^inf Phi(-(a + t s)) f_S(s) ds, in
    v = log s (the series' two families cancel here). The integrand
        e^(lc + df v - df e^(2v) / 2) Phi(-(a + t e^v))
    is entire in v; acb.integral bounds its error, with breakpoints around the peak, between
    ends where the parts outside are below e^-200 of the estimated value: those parts are at
    most Phi(-a) P(S < s_lo) and Phi(-(a + t s_hi)) P(S > s_hi) (df S^2 is chi-square(df),
    Phi(-(a + t s)) falls in s), added to the ball."""
    lc = arb(2).log() + (df / 2) * (df / 2).log() - (df / 2).lgamma()
    af, tf, dff, lcf = (float(v.mid()) for v in (a, t, df, lc))

    def gf(v):  # log of the integrand (a float: it only places the pieces)
        with precision(64):
            s = arb(v).exp()
            return float((_log_ncdf(-(arb(af) + arb(tf) * s)) + lcf + dff * v - dff * s * s / 2).mid())

    # the stationary point of -z^2/2 - (df/2)(s^2 - 2 log s): (t^2 + df) s^2 + a t s - df = 0
    # (divided through by t, so a t of 1e177 does not overflow)
    s0 = 2 * dff / tf / (af + math.sqrt(af * af + 4 * dff * (1 + dff / tf / tf)))
    v0 = math.log(s0)
    u0 = tf * s0
    sig = 1 / math.sqrt(u0 * u0 + (af + u0) * u0 + 2 * dff * s0 * s0)
    gmax = gf(v0)
    est = gmax + math.log(sig * 2.5066)  # log of the value, roughly (Laplace)

    def left(v):  # log of the bound on the part below v
        with precision(64):
            return float((_log_ncdf(-arb(af)) + _gammainc(arb(dff) / 2, arb(dff) * arb(2 * v).exp() / 2).log()).mid())

    def right(v):  # log of the bound on the part above v
        with precision(64):
            s = arb(v).exp()
            return float((_log_ncdf(-(arb(af) + arb(tf) * s)) + _gammainc(arb(dff) / 2, arb(dff) * s * s / 2, True).log()).mid())

    def end(sign, bound):  # the integrand falls off monotonically either side
        step = sig
        while gf(v0 + sign * step) - gmax > -400 or bound(v0 + sign * step) > est - 200:
            step *= 2
        return v0 + sign * step

    lo, hi = end(-1, left), end(1, right)
    pts = [lo] + [v0 + k * sig for k in (-20, -6, -2, 0, 2, 6, 20) if lo < v0 + k * sig < hi] + [hi]
    G, r2 = arb(gmax), arb(2).sqrt()

    with _alarm_deferred() as fired:
        def integrand(v, _analytic):
            if fired:
                return acb(float("nan"))
            s = v.exp()
            return ((a + t * s) / r2).erfc() / 2 * (lc - G + df * v - df * s * s / 2).exp()

        total = sum((acb.integral(integrand, arb(pts[i]), arb(pts[i + 1])) for i in range(len(pts) - 1)), acb(0))
    body = total.real * G.exp()
    s_lo, s_hi = arb(lo).exp(), arb(hi).exp()
    outside = (_ndtr(-a) * _gammainc(df / 2, df * s_lo * s_lo / 2)
               + _ndtr(-(a + t * s_hi)) * _gammainc(df / 2, df * s_hi * s_hi / 2, True))
    return add_error(body + outside / 2, outside / 2)


def _nct_upper_pos(df, d, t):
    """P(T > t) for t > 0, any d. For d < 0 the series' q family cancels its p family: the
    series where the ball still settles (a small |d| or t), else the integral."""
    if d < 0:
        try:
            v = _nct_upper_series(df, d, t)
            if v.rel_accuracy_bits() >= arbref.TARGET_BITS + 4:
                return v
        except NotSettled:
            pass
        return _nct_opposite(df, -d, t)
    return _nct_upper_series(df, d, t)


def _nct_upper_series(df, d, t):
    """P(T > t) for t > 0: (1/2) sum_j [p_j I_y(df/2, j + 1/2) + q_j I_y(df/2, j + 1)],
    y = df / (df + t^2), p_j Poisson(d^2/2), q_j = d e^(-d^2/2) (d^2/2)^j / (sqrt 2 Gamma(j + 3/2))."""
    b, z = df / 2, t * t
    x, y = z / (df + z), df / (df + z)
    mu = d * d / 2
    lx, ly = x.log(), y.log()
    T = lambda c: (c * lx + b * ly - c.log() - _lbeta(c, b)).exp()  # I_x(c, b) - I_x(c+1, b)
    up = lambda c: _ibeta(b, c, y, x)
    half = arb(0.5)
    sp = _mix_upper(mu, up(half), T(half), lambda j: x * (j + half + b) / (j + half + 1)) if mu != 0 else up(half)
    if mu == 0:
        return sp / 2
    # the q family: sum_j q_j U_j = d e^-mu / sqrt2 sum_j wq_j U_j, wq_j = mu^j / Gamma(j + 3/2),
    # U_j = I_y(b, j + 1) <= 1; beyond j > mu the weights fall by mu / (j + 3/2) or faster
    tol, muf = _trunc(), float(mu.mid())
    U, tt = up(arb(1)), T(arb(1))
    lead = d * (-mu).exp() / arb(2).sqrt()
    wq = arb(1.5).rgamma()
    S, j = wq * U, 0
    while True:
        U += tt
        tt *= x * (j + 1 + b) / (j + 2)
        j += 1
        wq *= mu / (j + half)
        S += wq * U
        if j > muf:
            tail = geometric_tail(wq, mu / (j + 1.5))
            if tail <= tol * S:
                break
        if j > 10 * muf + 10000:
            raise NotSettled("nct q family")
    return (sp + lead * add_error(S, tail)) / 2


def _nct_lower_pos(df, d, t):
    """P(T <= t) for t > 0: Phi(-d) + (1/2) sum_j [p_j I_x(j + 1/2, df/2) + q_j I_x(j + 1, df/2)]."""
    b, z = df / 2, t * t
    x, y = z / (df + z), df / (df + z)
    mu = d * d / 2
    lx, ly = x.log(), y.log()
    T = lambda c: (c * lx + b * ly - c.log() - _lbeta(c, b)).exp()
    low = lambda c: _ibeta(c, b, x, y)
    half = arb(0.5)
    phi = _ndtr(-d)
    if mu == 0:
        return phi + low(half) / 2
    sp = _mix_lower(mu, lambda J: low(J + half), lambda j: T(j + half), lambda j: (j + half) / ((j + half - 1 + b) * x))
    # q family: weights mu^j / Gamma(j + 3/2), backward from J (as _mix_lower); beyond J
    # wq_i L_i <= wq_J L_J (mu / (J + 3/2))^(i - J)
    tol, muf, width = _trunc(), float(mu.mid()), 40
    while True:
        J = int(muf + width * math.sqrt(muf + 1) + width)
        L, tt = low(arb(J + 1)), T(arb(J))
        wq = (J * mu.log() - arb(J + 1.5).lgamma()).exp()
        S = wq * L
        last = S
        for j in range(J, 0, -1):
            L += tt  # I_x(j, b)
            tt *= j / ((j - 1 + b) * x)  # T(j - 1)
            wq *= (j + half) / mu
            S += wq * L
        r = mu / (J + 1.5)
        if r < 1:
            tail = geometric_tail(last, r)
            if tail <= tol * S:
                break
        width *= 2
        if width > _MIX_WIDTH_MAX:
            raise NotSettled("nct q family (lower)")
    return phi + (sp + d * (-mu).exp() / arb(2).sqrt() * add_error(S, tail)) / 2


def _ref_nct(df, d, t, upper):
    """nctdtr (upper False) or nctdtrc (arb in, arb out)."""
    df, d, t = _a(df), _a(d), _a(t)
    if t == 0:
        return _ndtr(d) if upper else _ndtr(-d)
    if t.mid() > 0:
        return _nct_upper_pos(df, d, t) if upper else _nct_lower_pos(df, d, t)
    return _nct_lower_pos(df, -d, -t) if upper else _nct_upper_pos(df, -d, -t)


_mp_nct = _ref_nct  # the mpmath tool's name (tools/stats_ref.py)


def _chndtr_inputs(rng):
    m = 400
    df = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 1e-2, 1e4, m)])
    nc = np.concatenate([loguniform(rng, 1e-3, 1e4, m), loguniform(rng, 1e-3, 50, m), loguniform(rng, 1e-3, 1e4, m)])
    mean, sdv = df + nc, np.sqrt(2 * (df + 2 * nc))
    x = np.concatenate([mean[:m] * loguniform(rng, 1e-3, 1e2, m), loguniform(rng, 1e-4, 300, m),
                        np.abs(mean[2 * m:] + rng.normal(0, 4, m) * sdv[2 * m:]) + 1e-3])
    return (df, nc, x), np.ones(df.size, dtype=bool)


def _nctdtr_inputs(rng):
    m = 400
    df = np.concatenate([loguniform(rng, 1e-1, 1e5, m), loguniform(rng, 1e-1, 10, m), loguniform(rng, 1, 1e5, m)])
    d = np.concatenate([loguniform(rng, 1e-3, 100, m, signed=True), loguniform(rng, 1e-3, 10, m, signed=True),
                        loguniform(rng, 1e-3, 100, m, signed=True)])
    t = np.concatenate([loguniform(rng, 1e-3, 1e4, m, signed=True), loguniform(rng, 1e-3, 1e3, m, signed=True),
                        d[2 * m:] + rng.normal(0, 4, m) * (1 + np.abs(d[2 * m:]) / np.sqrt(df[2 * m:]))])
    return (df, d, t), np.ones(df.size, dtype=bool)


def _ncfdtr_inputs(rng):
    m = 400
    dfn = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 1e-1, 1e4, m)])
    dfd = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 3, 1e4, m)])
    nc = np.concatenate([loguniform(rng, 1e-3, 1e4, m), loguniform(rng, 1e-3, 50, m), loguniform(rng, 1e-3, 1e4, m)])
    mean = (dfn + nc) / dfn * np.where(dfd > 2.5, dfd / (dfd - 2), 1.0)
    f = np.concatenate([mean[:m] * loguniform(rng, 1e-3, 1e2, m), loguniform(rng, 1e-4, 1e3, m),
                        mean[2 * m:] * np.exp(rng.normal(0, 0.5, m))])
    return (dfn, dfd, nc, f), np.ones(dfn.size, dtype=bool)


def _ncx2_logpdf(df, nc, x):
    """log of the noncentral chi-square density, e^(-(x+nc)/2) (x/nc)^(df/4-1/2) I_(df/2-1)(sqrt(nc x)) / 2
    (arb in, arb out; I scaled by e^-z, so no huge exponential)."""
    df, nc, x = _a(df), _a(nc), _a(x)
    z = (nc * x).sqrt()
    return -arb(2).log() - (x + nc) / 2 + (df / 4 - arb(0.5)) * (x / nc).log() + z.bessel_i(df / 2 - 1, scaled=True).log() + z


def _ref_chndtrix(df, nc, p):
    x0 = _start2("chndtrix", df, nc, p)
    _no_start(x0, True)
    lower = p <= 0.5
    tail = lambda x: _ncx2_pair(df, nc, x, not lower)
    target = p if lower else 1 - p
    dens = lambda x: _ncx2_logpdf(df, nc, x).exp()
    x = solve_log_settled(lambda x: tail(x).log() - target.log(), lambda x: (1 if lower else -1) * x * dens(x) / tail(x),
                          x0, increasing=lower)
    return x, _kappa(lambda: p / (x * dens(x)))


def _nct_pdf(df, d, t):
    """d/dt nctdtr (arb in, arb out): (df / t) (F(df + 2, d; t sqrt(1 + 2/df)) - F(df, d; t)), from
    whichever tail is small."""
    df, d, t = _a(df), _a(d), _a(t)
    if t == 0:
        return (((df + 1) / 2).lgamma() - (df / 2).lgamma() - d * d / 2).exp() / (arb.pi() * df).sqrt()
    t2 = t * (1 + 2 / df).sqrt()
    if _ref_nct(df, d, t, False).mid() <= 0.5:
        return df / t * (_ref_nct(df + 2, d, t2, False) - _ref_nct(df, d, t, False))
    return df / t * (_ref_nct(df, d, t, True) - _ref_nct(df + 2, d, t2, True))


def _ref_nctdtrit(df, d, p):
    t0 = _start2("nctdtrit", df, d, p)
    _no_start(t0)
    lower = p <= 0.5
    target = p if lower else 1 - p
    tail = lambda t: _ref_nct(df, d, t, not lower)
    sgn = 1 if lower else -1
    t = solve_settled(lambda t: tail(t).log() - target.log(), lambda t: sgn * _nct_pdf(df, d, t) / tail(t), t0,
                      increasing=lower)
    return t, (_kappa(lambda: p / (t * _nct_pdf(df, d, t))) if t != 0 else 1.0)


def _ncf_logpdf(dfn, dfd, nc, x):
    """log density (arb in, arb out): e^(-nc/2) (central F density) 1F1((dfn+dfd)/2; dfn/2; nc dfn x / (2 (dfn x + dfd)))."""
    dfn, dfd, nc, x = _a(dfn), _a(dfd), _a(nc), _a(x)
    a, b = dfn / 2, dfd / 2
    central = a * (dfn / dfd).log() + (a - 1) * x.log() - (a + b) * (dfn * x / dfd).log1p() - _lbeta(a, b)
    return -nc / 2 + central + _hyp1f1_series(a + b, a, nc * dfn * x / (2 * (dfn * x + dfd))).log()


def _hyp1f1_series(a, b, z):
    """1F1(a; b; z) for a, b > 0, z >= 0, all terms positive (Arb's hypgeom_1f1 returns NaN for
    a, b ~ 1e3, z ~ 2e3). The ratio of the terms after term k, (a+k) z / ((b+k)(k+1)), is at most
    r = z / (k+1) max(1, (a+k) / (b+k)) from there on (both factors fall when a >= b; the first
    stays below 1 when a < b)."""
    if z == 0:
        return arb(1)
    tol = _trunc()
    t = s = arb(1)
    k = 0
    while True:
        t = t * (a + k) * z / ((b + k) * (k + 1))
        k += 1
        s += t
        if k % 8 == 0:
            r = z / (k + 1) * ((a + k) / (b + k) if a >= b else 1)
            if r < 1:
                tail = geometric_tail(t, r)
                if tail <= tol * s:
                    return add_error(s, tail)
        if k > MAX_TERMS:
            raise NotSettled("1F1 series")


def _ref_ncfdtri(dfn, dfd, nc, p):
    x0 = _start2("ncfdtri", dfn, dfd, nc, p)
    _no_start(x0, True)
    lower = p <= 0.5
    tail = lambda x: _ncf_pair(dfn, dfd, nc, x, not lower)
    target = p if lower else 1 - p
    dens = lambda x: _ncf_logpdf(dfn, dfd, nc, x).exp()
    x = solve_log_settled(lambda x: tail(x).log() - target.log(), lambda x: (1 if lower else -1) * x * dens(x) / tail(x),
                          x0, increasing=lower)
    return x, _kappa(lambda: p / (x * dens(x)))


def _probs(rng, m):
    return np.concatenate([rng.uniform(0, 1, m), loguniform(rng, 1e-300, 0.5, m), 1 - loguniform(rng, 1e-16, 0.5, m)])


def _chndtrix_inputs(rng):
    m = 400
    df = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 1e-2, 1e4, m)])
    nc = np.concatenate([loguniform(rng, 1e-3, 1e4, m), loguniform(rng, 1e-3, 50, m), loguniform(rng, 1e-3, 1e4, m)])
    p = _probs(rng, m)
    keep = (p > 0) & (p < 1)
    return (df[keep], nc[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


def _nctdtrit_inputs(rng):
    m = 400
    df = np.concatenate([loguniform(rng, 1e-1, 1e5, m), loguniform(rng, 1e-1, 10, m), loguniform(rng, 1, 1e5, m)])
    d = np.concatenate([loguniform(rng, 1e-3, 100, m, signed=True), loguniform(rng, 1e-3, 10, m, signed=True),
                        loguniform(rng, 1e-3, 100, m, signed=True)])
    p = _probs(rng, m)
    keep = (p > 0) & (p < 1)
    return (df[keep], d[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


def _ncfdtri_inputs(rng):
    m = 400
    dfn = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 1e-1, 1e4, m)])
    dfd = np.concatenate([loguniform(rng, 1e-2, 1e4, m), loguniform(rng, 1e-2, 10, m), loguniform(rng, 3, 1e4, m)])
    nc = np.concatenate([loguniform(rng, 1e-3, 1e4, m), loguniform(rng, 1e-3, 50, m), loguniform(rng, 1e-3, 1e4, m)])
    p = _probs(rng, m)
    keep = (p > 0) & (p < 1)
    return (dfn[keep], dfd[keep], nc[keep], p[keep]), np.ones(keep.sum(), dtype=bool)


INVERSES = {"gammaincinv", "gammainccinv", "betaincinv", "betainccinv", "stdtrit", "chdtri", "fdtri", "ndtri_exp",
            "chndtrix", "nctdtrit", "ncfdtri"}


FUNCTIONS = {
    # the engine's test function (csrc/special/ops/testing.hpp): checks the pipeline end to end
    "_cf_tan": (1, lambda x: x.tan(), _cf_tan_inputs),
    # S1
    "expit": (1, _expit, _expit_inputs),
    "logit": (1, lambda p: (p / (1 - p)).log(), _logit_inputs),
    "log_expit": (1, _log_expit, _expit_inputs),
    "xlogy": (2, _zero_at_zero(lambda x, y: x * y.log()), _xlogy_inputs),
    "xlog1py": (2, _zero_at_zero(lambda x, y: x * y.log1p()), _xlog1py_inputs),
    "entr": (1, _zero_at_zero(lambda x: -x * x.log()), _entr_inputs),
    "rel_entr": (2, _zero_at_zero(lambda x, y: x * (x / y).log()), _rel_entr_inputs),
    # S2
    "erf": (1, lambda x: x.erf(), _erf_inputs),
    "erfc": (1, lambda x: x.erfc(), _erfc_inputs),
    "erfcx": (1, _ref_erfcx, _erfcx_inputs),
    "ndtr": (1, _ndtr, _ndtr_inputs),
    "log_ndtr": (1, _ref_log_ndtr, _log_ndtr_inputs),
    "ndtri": (1, _ref_ndtri, _ndtri_inputs),
    "erfinv": (1, lambda y: y.erfinv(), _erfinv_inputs),
    "erfcinv": (1, lambda z: z.erfcinv(), _erfcinv_inputs),
    # S3
    "gamma": (1, lambda x: x.gamma(), _gamma_inputs),
    "rgamma": (1, lambda x: x.rgamma(), _rgamma_inputs),
    "gammaln": (1, _ref_gammaln, _gammaln_inputs),
    "digamma": (1, lambda x: x.digamma(), _digamma_inputs),
    "polygamma": (2, _ref_polygamma, _polygamma_inputs),
    "beta": (2, _ref_beta, _beta_inputs),
    "betaln": (2, _ref_betaln, _betaln_inputs),
    "binom": (2, _ref_binom, _binom_inputs),
    "zeta": (1, lambda x: x.zeta(), _zeta_inputs),
    "zeta_hurwitz": (2, _ref_zeta_hurwitz, _hurwitz_inputs),
    # S4
    "gammainc": (2, _ref_gammainc, _gammainc_inputs),
    "gammaincc": (2, _ref_gammaincc, _gammainc_inputs),
    "gammaincinv": (2, _gamma_inverse(False), _gammaincinv_inputs),
    "gammainccinv": (2, _gamma_inverse(True), _gammaincinv_inputs),
    "betainc": (3, _ref_betainc, _betainc_inputs),
    "betaincc": (3, _ref_betaincc, _betainc_inputs),
    "betaincinv": (3, _beta_inverse(False), _betaincinv_inputs),
    "betainccinv": (3, _beta_inverse(True), _betaincinv_inputs),
    # T1
    "stdtr": (2, _ref_stdtr, _stdtr_inputs),
    "stdtrit": (2, _ref_stdtrit, _stdtrit_inputs),
    "chdtr": (2, lambda v, x: _gammainc(v / 2, x / 2), _chdtr_inputs),
    "chdtrc": (2, lambda v, x: _gammainc(v / 2, x / 2, True), _chdtr_inputs),
    "chdtri": (2, _ref_chdtri, _chdtri_inputs),
    "fdtr": (3, _ref_fdtr, _fdtr_inputs),
    "fdtrc": (3, _ref_fdtrc, _fdtr_inputs),
    "fdtri": (3, _ref_fdtri, _fdtri_inputs),
    "pdtr": (2, lambda k, m: _gammainc(k + 1, m, True), _pdtr_inputs),
    "pdtrc": (2, lambda k, m: _gammainc(k + 1, m), _pdtr_inputs),
    "bdtr": (3, lambda k, n, p: _ibeta(n - k, k + 1, 1 - p, p), _bdtr_inputs),
    "bdtrc": (3, lambda k, n, p: _ibeta(k + 1, n - k, p, 1 - p), _bdtr_inputs),
    "nbdtr": (3, lambda k, n, p: _ibeta(n, k + 1, p, 1 - p), lambda rng: _bdtr_inputs(rng, True)),
    "nbdtrc": (3, lambda k, n, p: _ibeta(k + 1, n, 1 - p, p), lambda rng: _bdtr_inputs(rng, True)),
    "ndtri_exp": (1, _ref_ndtri_exp, _ndtri_exp_inputs),
    # T2 (x last: chndtr(df, nc, x), T2 decision)
    "chndtr": (3, lambda df, nc, x: _ncx2_pair(df, nc, x, False), _chndtr_inputs),
    "chndtrc": (3, lambda df, nc, x: _ncx2_pair(df, nc, x, True), _chndtr_inputs),
    "nctdtr": (3, lambda df, d, t: _ref_nct(df, d, t, False), _nctdtr_inputs),
    "nctdtrc": (3, lambda df, d, t: _ref_nct(df, d, t, True), _nctdtr_inputs),
    "ncfdtr": (4, lambda dfn, dfd, nc, f: _ncf_pair(dfn, dfd, nc, f, False), _ncfdtr_inputs),
    "ncfdtrc": (4, lambda dfn, dfd, nc, f: _ncf_pair(dfn, dfd, nc, f, True), _ncfdtr_inputs),
    "chndtrix": (3, _ref_chndtrix, _chndtrix_inputs),
    "nctdtrit": (3, _ref_nctdtrit, _nctdtrit_inputs),
    "ncfdtri": (4, _ref_ncfdtri, _ncfdtri_inputs),
    # S5
    "expi": (1, lambda x: x.ei(), _expi_inputs),
    "expn": (2, lambda n, x: x.expint(n), _expn_inputs),
    "ellipk": (1, lambda m: acb(m).elliptic_k().real, _ellip_inputs),
    "ellipe": (1, lambda m: acb(m).elliptic_e().real, _ellip_inputs),
    # S6 (I_n(-x) = (-1)^n I_n(x); Arb's scaled I is e^-z I(z), so the scaled ones are taken at |x|)
    "i0": (1, lambda x: abs(x).bessel_i(0), _bessel01_inputs(1e-3, 713.9, True)),
    "i1": (1, lambda x: x.sgn() * abs(x).bessel_i(1), _bessel01_inputs(1e-3, 713.9, True)),
    "i0e": (1, lambda x: abs(x).bessel_i(0, scaled=True), _bessel01_inputs(1e-3, 1e300, True)),
    "i1e": (1, lambda x: x.sgn() * abs(x).bessel_i(1, scaled=True), _bessel01_inputs(1e-3, 1e300, True)),
    "k0": (1, lambda x: x.bessel_k(0), _bessel01_inputs(1e-3, 745)),
    "k1": (1, lambda x: x.bessel_k(1), _bessel01_inputs(1e-3, 745)),
    "k0e": (1, lambda x: x.bessel_k(0, scaled=True), _bessel01_inputs(1e-3, 1e300)),
    "k1e": (1, lambda x: x.bessel_k(1, scaled=True), _bessel01_inputs(1e-3, 1e300)),
    "j0": (1, lambda x: x.bessel_j(0), _jy01_inputs),
    "j1": (1, lambda x: x.bessel_j(1), _jy01_inputs),
    "y0": (1, lambda x: x.bessel_y(0), _jy01_inputs),
    "y1": (1, lambda x: x.bessel_y(1), _jy01_inputs),
    "jv": (2, lambda v, x: x.bessel_j(v), _order_inputs(1e-3, 1e5)),
    "yv": (2, lambda v, x: x.bessel_y(v), _order_inputs(1e-3, 1e5)),
    "iv": (2, lambda v, x: x.bessel_i(v), _order_inputs(1e-3, 700)),
    "kv": (2, lambda v, x: x.bessel_k(v), _order_inputs(1e-3, 700)),
    "ive": (2, lambda v, x: x.bessel_i(v, scaled=True), _order_inputs(1e-3, 1e5, False)),
    "kve": (2, lambda v, x: x.bessel_k(v, scaled=True), _order_inputs(1e-3, 1e5)),
    "airy_ai": (1, _airy_part(0), _airy_inputs),
    "airy_aip": (1, _airy_part(1), _airy_inputs),
    "airy_bi": (1, _airy_part(2), _airy_inputs),
    "airy_bip": (1, _airy_part(3), _airy_inputs),
    # S7
    "lambertw": (1, lambda x: x.lambertw(0), _lambertw_inputs(0)),
    "lambertw_m1": (1, lambda x: x.lambertw(-1), _lambertw_inputs(-1)),
    "hyp1f1": (3, lambda a, b, x: x.hypgeom_1f1(a, b), _hyp1f1_inputs),
    "hyp2f1": (4, lambda a, b, c, z: z.hypgeom_2f1(a, b, c), _hyp2f1_inputs),
}


# ---- S8: complex arguments (docs/api/special.md §12). The relative error is |error| / |f|
# (S8 user decision); `scale` carries the envelope where zeros make it unreachable:
#   wofz  Im z < 0:  (2 |exp(-z^2)| + |w(-z)|) / |w(z)|
#   erfc  Re z < 0:  (2 + |erfc(-z)|) / |erfc(z)|
#   erf   |z| >= 2:  (1 + |erfc(+-z)|) / |erf(z)|  (the sign making Re >= 0)
#   erfcx Re z < 0:  (2 |exp(z^2)| + |erfcx(-z)|) / |erfcx(z)|
#   loggamma Re z <= 0: max(1, 1/|f|)  (|error| <= 1e-14 max(1, |f|))
# On a branch cut (the negative real axis), Im z = -0 is the limit from below, taken as
# conj(f(conj z)) (loggamma) or conj(W_{-k}(conj z)) (lambertw); Arb's value on the cut is
# the limit from above, as mpmath's was.


def _w_ref(z):
    """Faddeeva w(z) = e^(-z^2) erfc(-iz); for Im z > 0 and |z| > 8, U(1/2, 1/2, -z^2) / sqrt(pi)
    (erfc(c) = e^(-c^2) U(1/2, 1/2, c^2) / sqrt(pi) for Re c > 0, c = -iz): no e^(-z^2) whose
    phase at |z| = 1e150 would need 1000 bits."""
    if z.imag > 0 and abs(z) > 8:
        return (-(z * z)).hypgeom_u(0.5, 0.5) / _sqrt_pi()
    return (-(z * z)).exp() * (acb(0, -1) * z).erfc()


def _erfcx_ref(z):
    """e^(z^2) erfc(z); for Re z > 0 and |z| > 8, U(1/2, 1/2, z^2) / sqrt(pi) (as _w_ref)."""
    if z.real > 0 and abs(z) > 8:
        return (z * z).hypgeom_u(0.5, 0.5) / _sqrt_pi()
    return (z * z).exp() * z.erfc()


def _cpoints(rng, n, lo, hi, arg_lo=-np.pi, arg_hi=np.pi):
    """n points with log|z| uniform in [log lo, log hi] and arg uniform in [arg_lo, arg_hi]."""
    return loguniform(rng, lo, hi, n) * np.exp(1j * rng.uniform(arg_lo, arg_hi, n))


def _near(points, scales=(1e-14, 1e-10, 1e-6, 1e-3)):
    """Points at the given distances around each point, in four directions."""
    out = []
    for p in points:
        for d in scales:
            out += [p + d, p - d, p + 1j * d, p - 1j * d]
    return np.array(out)


def _axes(rng, n, lo, hi):
    """Points on both axes, with both signs and both signed zeros."""
    x = loguniform(rng, lo, hi, n, signed=True)
    return np.concatenate([x + 0j, x - 0j, 1j * x, complex(-0.0, 1.0) * x])


# The first zeros of erfc and erf (to 4 digits, refined by findroot): erfc's in the second
# quadrant and their conjugates, erf's in the first and their reflections.
_ERFC_ZEROS = [-1.3548 + 1.9915j, -2.1769 + 2.6911j, -2.7756 + 3.2382j]
_ERF_ZEROS = [1.4506 + 1.8806j, 2.2447 + 2.6165j, 2.8340 + 3.1756j]


def _refine(f, df, guesses):
    """Newton from each guess on f (acb), at 128 bits; each zero rounded to complex128."""
    out = []
    with precision(128):
        for g in guesses:
            z = acb(g.real, g.imag)
            for _ in range(30):
                z = (z - f(z) / df(z)).mid()
            out.append(complex(float(z.real.mid()), float(z.imag.mid())))
    return out


def _derf(z):
    return 2 / _sqrt_pi() * (-(z * z)).exp()


def _erfc_zeros():
    zs = _refine(lambda z: z.erfc(), lambda z: -_derf(z), _ERFC_ZEROS)
    return zs + [z.conjugate() for z in zs]


def _erf_zeros():
    zs = _refine(lambda z: z.erf(), _derf, _ERF_ZEROS)
    return zs + [z.conjugate() for z in zs] + [-z for z in zs] + [-z.conjugate() for z in zs]


def _wofz_inputs(rng):
    z = np.concatenate([
        rng.uniform(-7, 7, 2000) + 1j * rng.uniform(0, 4, 2000),  # the grid (both signs of Re z)
        rng.uniform(-40, 40, 1000) + 1j * rng.uniform(0, 15, 1000),  # the continued fraction
        rng.uniform(-40, 40, 500) + 1j * loguniform(rng, 1e-300, 1, 500),  # near the real axis
        _cpoints(rng, 1500, 1e-300, 1e300, 0, np.pi),  # the upper half plane, every size
        rng.uniform(-8, 8, 1500) + 1j * rng.uniform(-6, 0, 1500),  # the lower half plane
        _near([1j * z for z in _erfc_zeros()]),  # zeros of w: w(z) = 0 where erfc(-iz) = 0
        _axes(rng, 200, 1e-300, 1e3),
        np.array([7 + 4j, 7 + 4.0000001j, 7.0000001 + 2j, 3e8 / 2 + 3e8 / 2 * 1j, 1e-300 + 1e-300j]),
    ])
    return (z,), np.isfinite(z)


def _wofz_scale(z, v):
    if z.imag >= 0:
        return 1.0
    return _scale_of(lambda: (2 * abs((-(z * z)).exp()) + abs(_w_ref(-z))) / abs(v))


def _erf_family_inputs(zeros):
    def build(rng):
        z = np.concatenate([
            _cpoints(rng, 2000, 1e-300, 30), _cpoints(rng, 500, 30, 1e150),
            rng.uniform(-4, 4, 1500) + 1j * rng.uniform(-4, 4, 1500),
            rng.uniform(-30, 30, 500) + 1j * rng.uniform(-30, 30, 500),
            _near(zeros()), _axes(rng, 200, 1e-300, 1e3),
            np.array([0.5 + 0j, 0.3 + 0.4j, 0.30000001 + 0.4j, 2 + 0j, 1.2 + 1.6j, 1.2000001 + 1.6j]),
        ])
        return (z,), np.isfinite(z)
    return build


def _erf_scale(z, v):
    if _f(z) < 2:
        return 1.0
    zp = z if z.real >= 0 else -z
    return _scale_of(lambda: (1 + abs(zp.erfc())) / abs(v))


def _erfc_scale(z, v):
    if z.real >= 0:
        return 1.0
    return _scale_of(lambda: (2 + abs((-z).erfc())) / abs(v))


def _erfcx_scale(z, v):
    if z.real >= 0:
        return 1.0
    return _scale_of(lambda: (2 * abs((z * z).exp()) + abs(_erfcx_ref(-z))) / abs(v))


def _loggamma_inputs(rng):
    neg = -loguniform(rng, 1e-3, 300, 400)
    z = np.concatenate([
        rng.uniform(0, 8, 2000) + 1j * rng.uniform(-8, 8, 2000),  # the right box
        rng.uniform(0.1, 4.2, 1500) + 1j * rng.uniform(-2.5, 2.5, 1500),  # around the zeros
        rng.uniform(-40, 0.1, 1500) + 1j * rng.uniform(-8, 8, 1500),  # the left box (reflection)
        _cpoints(rng, 1500, 1e-300, 1e300),
        _near([1 + 0j, 2 + 0j]), _near([-1 + 0j, -2 + 0j, -10 + 0j], (1e-10, 1e-3)),
        neg + 0j, neg - 0j,  # the cut: both sides
        np.array([0.1 + 2j, 0.1 - 2j, 4 + 2j, 4.0000001 + 0.5j, 1.5 + 0j, 7 + 7j, 7.0000001 + 1j]),
    ])
    z = z[~((z.imag == 0) & (z.real <= 0) & (z.real == np.floor(z.real)))]  # the poles are tests
    return (z,), np.isfinite(z)


def _loggamma_scale(z, v):
    if z.real > 0:
        return 1.0
    return _scale_of(lambda: 1 / abs(v))


def _lambertw_ref(k):
    return lambda z: z.lambertw(k)


def _lambertw_c_inputs(rng):
    em = -np.exp(-1.0)
    near_bp = em + loguniform(rng, 1e-17, 0.5, 1000) * np.exp(1j * rng.uniform(-np.pi, np.pi, 1000))
    cut = -loguniform(rng, 1e-300, 1e300, 300)
    z = np.concatenate([
        _cpoints(rng, 2000, 1e-300, 1e300), rng.uniform(-3, 3, 1500) + 1j * rng.uniform(-3, 3, 1500),
        near_bp, cut + 0j, cut - 0j, _near([em + 0j, 0j, -1 + 0j], (1e-12, 1e-6, 1e-2)),
        np.array([complex(-0.36787944117144233, 0.0), complex(-0.36787944117144233, -0.0)]),
    ])
    z = z[z != 0]
    return (z,), np.isfinite(z)


# name: (function of acb z -> acb, input builder, scale or None, cut): on the negative real
# axis with Im z = -0 the value is conj(cut(x)) (None: no cut there)
COMPLEX = {
    "wofz": (_w_ref, _wofz_inputs, _wofz_scale, None),
    "erf_complex": (lambda z: z.erf(), _erf_family_inputs(_erf_zeros), _erf_scale, None),
    "erfc_complex": (lambda z: z.erfc(), _erf_family_inputs(_erfc_zeros), _erfc_scale, None),
    "erfcx_complex": (_erfcx_ref, _erf_family_inputs(lambda: [-z for z in _erfc_zeros()]), _erfcx_scale, None),
    "loggamma_complex": (lambda z: z.lgamma(), _loggamma_inputs, _loggamma_scale, lambda z: z.lgamma()),
    "lambertw_c0": (_lambertw_ref(0), _lambertw_c_inputs, None, _lambertw_ref(0)),
    "lambertw_cm1": (_lambertw_ref(-1), _lambertw_c_inputs, None, _lambertw_ref(1)),
    "lambertw_c1": (_lambertw_ref(1), _lambertw_c_inputs, None, _lambertw_ref(-1)),
    "lambertw_c4": (_lambertw_ref(4), _lambertw_c_inputs, None, _lambertw_ref(-4)),
}


def build_reduce():
    """tests/special/ref/reduce_cases.npz: arrays for logsumexp, softmax and log_softmax with
    their references (S5). Each case is a 1-d array; `start` holds the offsets into the
    concatenation `x`. Per case: logsumexp (hi, lo); per element: softmax and log_softmax (hi, lo).
    Each case's values are settled together (rigorous over the whole list)."""
    rng = np.random.default_rng(20261114)
    cases = []
    for k in range(120):
        n = int(rng.choice([1, 2, 3, 10, 100, 1000, 5000]))
        scale = float(rng.choice([1e-3, 1, 10, 300, 1e5]))
        x = rng.normal(0, scale, n) + float(rng.choice([0.0, -700.0, 700.0, 1e10]))
        cases.append(x)
    cases.append(np.array([0.0, -745.0, -1e300]))  # subnormal and zero softmax values
    cases.append(np.full(1000, 3.0))
    start = np.cumsum([0] + [c.size for c in cases])
    allx = np.concatenate(cases)
    lse_hi, lse_lo = np.empty(len(cases)), np.empty(len(cases))
    sm_hi, sm_lo = np.empty(allx.size), np.empty(allx.size)
    ls_hi, ls_lo = np.empty(allx.size), np.empty(allx.size)
    for c, x in enumerate(cases):
        xs = [float(v) for v in x]
        m = max(xs)

        def values(prec, xs=xs, m=m):
            with precision(prec):
                s_ = arb(0)
                for v in xs:
                    s_ += (arb(v) - m).exp()
                lse = m + s_.log()
                return [lse] + [(arb(v) - lse).exp() for v in xs] + [arb(v) - lse for v in xs]

        p = arbref.START_PREC
        while True:
            vals = values(p)
            if all(arbref.settled(v, floor=FLOOR) for v in vals):
                break
            p *= 2
            if p > arbref.MAX_PREC:
                raise NotSettled("reduce case")
        n = len(xs)
        lse_hi[c], lse_lo[c] = to_dd(vals[0])
        for i in range(n):
            sm_hi[start[c] + i], sm_lo[start[c] + i] = to_dd(vals[1 + i])
            ls_hi[start[c] + i], ls_lo[start[c] + i] = to_dd(vals[1 + n + i])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "reduce_cases.npz", x=allx, start=start, lse_hi=lse_hi, lse_lo=lse_lo,
                        softmax_hi=sm_hi, softmax_lo=sm_lo, log_softmax_hi=ls_hi, log_softmax_lo=ls_lo,
                        **_PROVENANCE)
    return len(cases)


# ---------------------------------------------------------------- writing

_PROVENANCE = {"arb": f"python-flint {flint.__version__}", "bits": arbref.TARGET_BITS}


def double_double(v):
    """(hi, lo) of a settled arb (or a float): arbref.to_dd, hi the double nearest the value."""
    return to_dd(v)


def evaluate(f, args):
    """f(*args) (args floats, passed as exact arbs) as a settled ball (arbref.rigorous with the
    double-double floor): raises NotSettled if it never settles."""
    return rigorous(lambda: f(*(arb(a) for a in args)), floor=FLOOR)


def evaluate_complex(f, z):
    """f(z) (z a Python complex) as a settled acb: each part to 2^-110 of itself, or of |f(z)|
    times 2^-10 (S8 measures |error| / |f|), or 2^-1132 absolutely."""
    return rigorous(lambda: f(acb(z.real, z.imag)), floor=FLOOR, modulus_bits=MODULUS_BITS)


class _Timeout(Exception):
    pass


def _alarm(*_):
    """SIGALRM handler: raises _Timeout (between Arb calls; a point over POINT_SECONDS is dropped)."""
    raise _Timeout


def evaluate_inverse(f, args):
    """(x, kappa) of an inverse: f certifies its own root (arbref.solve_settled / solve_log_settled)."""
    with precision(arbref.START_PREC):
        return f(*(arb(a) for a in args))


# A point that takes longer than this is dropped (and counted). The alarm fires between Arb
# calls; each call at the largest precision returns within seconds.
POINT_SECONDS = 30
_DROPS = (_Timeout, ArithmeticError, ValueError, RuntimeError)  # NotSettled is an ArithmeticError


def _point(task):
    """One reference point (in a worker): (index, hi, lo, scale) or (index, None) if dropped.
    Complex functions: hi and lo are complex (the parts' double-doubles)."""
    import signal

    name, i, args = task
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(POINT_SECONDS)
    try:
        sc = 1.0
        if name in COMPLEX:
            f, _, scale_of, cut = COMPLEX[name]
            zi = args[0]
            if cut is not None and zi.imag == 0 and math.copysign(1.0, zi.imag) < 0 and zi.real < 0:
                w = evaluate_complex(cut, complex(zi.real, 0.0))
                v = _exact(w.conjugate)  # the limit from below
            else:
                v = evaluate_complex(f, zi)
            if scale_of is not None and v.is_finite() and v != 0:
                with precision(arbref.START_PREC):
                    sc = scale_of(acb(zi.real, zi.imag), v)
            rh, rl = to_dd(v.real)
            ih, il = to_dd(v.imag)
            return i, (complex(rh, ih), complex(rl, il), sc)
        _, f, _ = FUNCTIONS[name]
        if name in INVERSES:
            v, k = evaluate_inverse(f, args)
            if math.isnan(k):
                raise NotSettled("kappa")
            sc = max(1.0, float(k)) if to_dd(v)[0] != 0 else 1.0  # a root that rounds to 0: no kappa
        else:
            v = evaluate(f, args)
            if name in SCALES and (isinstance(v, float) or v.is_finite()):
                sc = SCALES[name](args, arb(v) if isinstance(v, float) else v)
        return i, (*to_dd(v), sc)
    except _DROPS:
        return i, None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


MAX_TASKS_PER_CHILD = 500  # workers are replaced after this many points (memory stays bounded)


def _run(tasks, jobs):
    import stardust  # noqa: F401  (the inverses' start; loaded once here, not in every worker)

    if jobs > 1:
        import multiprocessing

        with multiprocessing.get_context("fork").Pool(jobs, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
            return list(pool.imap_unordered(_point, tasks, chunksize=1))
    return list(map(_point, tasks))


def build_complex(name, jobs=1):
    _, inputs, _, _ = COMPLEX[name]
    rng = np.random.default_rng(20261114)
    (z,), guaranteed = inputs(rng)
    z = np.asarray(z, dtype=np.complex128)
    n = z.size
    hi, lo, scale = np.empty(n, dtype=np.complex128), np.empty(n, dtype=np.complex128), np.ones(n)
    keep = np.ones(n, dtype=bool)
    for i, r in _run([(name, i, (complex(z[i]),)) for i in range(n)], jobs):
        if r is None:
            keep[i] = False
        else:
            hi[i], lo[i], scale[i] = r
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", nin=1, hi=hi[keep], lo=lo[keep], scale=scale[keep],
                        guaranteed=np.asarray(guaranteed, dtype=bool)[keep], x0=z[keep], **_PROVENANCE)
    return int(keep.sum()), int((~keep).sum())


def build(name, jobs=1):
    nin, f, inputs = FUNCTIONS[name]
    rng = np.random.default_rng(20261113)
    xs, guaranteed = inputs(rng)
    xs = tuple(np.asarray(x, dtype=np.float64) for x in xs)
    n = xs[0].size
    hi, lo, scale = np.empty(n), np.empty(n), np.ones(n)
    keep = np.ones(n, dtype=bool)
    tasks = [(name, i, tuple(float(x[i]) for x in xs)) for i in range(n)]
    for i, r in _run(tasks, jobs):
        if r is None:
            keep[i] = False
        else:
            hi[i], lo[i], scale[i] = r
    xs = tuple(x[keep] for x in xs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", nin=nin, hi=hi[keep], lo=lo[keep], scale=scale[keep],
                        guaranteed=np.asarray(guaranteed, dtype=bool)[keep],
                        **{f"x{k}": x for k, x in enumerate(xs)}, **_PROVENANCE)
    return int(keep.sum()), int((~keep).sum())


def main(argv):
    jobs = 10
    if argv and argv[0].startswith("--jobs="):
        jobs, argv = int(argv[0].split("=", 1)[1]), argv[1:]
    names = argv or list(FUNCTIONS) + list(COMPLEX) + ["reduce_cases"]
    for name in names:
        if name == "reduce_cases":
            print(f"reduce_cases: {build_reduce()} arrays -> tests/special/ref/reduce_cases.npz", flush=True)
            continue
        n, dropped = build_complex(name, jobs) if name in COMPLEX else build(name, jobs)
        note = f" ({dropped} dropped: not settled, or over {POINT_SECONDS} s)" if dropped else ""
        print(f"{name}: {n} points -> {(OUT / name).with_suffix('.npz').relative_to(ROOT)}{note}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
