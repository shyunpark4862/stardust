"""Reference values for sd.special from mpmath (docs/design/special.md §6).

    .venv/bin/python tools/special_ref.py            # every function in FUNCTIONS
    .venv/bin/python tools/special_ref.py NAME ...   # only these

writes tests/special/ref/<name>.npz (format: tests/special/refdata.py). The files are
committed; tests and bench/special/accuracy.py read them and never call mpmath.

Each function's input set (docs/design/special.md §6) is
  - a grid over its guaranteed region,
  - log-uniform random points (both signs where the domain has them), inside the
    guaranteed region and in the loose region around it,
  - edge points: a few ULP either side of every region boundary, poles and zeros,
    0, and the domain ends.
The generator is seeded: running the tool again gives the same file.
"""

import sys
from pathlib import Path

import mpmath
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "special" / "ref"
DPS = 50  # significant decimal digits: far beyond float64's 16, so hi + lo is exact to rounding


# ---------------------------------------------------------------- input sets


def grid(lo, hi, n):
    return np.linspace(lo, hi, n)


def loguniform(rng, lo, hi, n, signed=False):
    """n points with log|x| uniform in [log lo, log hi] (lo > 0); random signs when signed."""
    x = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    if signed:
        x *= rng.choice([-1.0, 1.0], n)
    return x


def around(points, k=4):
    """Each point and the k float64 neighbours on either side."""
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
# name: (nin, mpmath function of mpf inputs, input-set builder rng -> (inputs, guaranteed))


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


def _mp_erfc(x):
    return mpmath.mpf(0) if x > 1e10 else (mpmath.mpf(2) if x < -1e10 else mpmath.erfc(x))


def _mp_erfcx(x):
    if x > 1e8:  # asymptotic series; mpmath's erfc overflows a float internally for huge x
        return (1 / (x * mpmath.sqrt(mpmath.pi))) * (1 - 1 / (2 * x * x) + 3 / (4 * x**4) - 15 / (8 * x**6))
    return mpmath.exp(x * x) * mpmath.erfc(x)


def _mp_ndtr(x):
    return mpmath.mpf(0) if x < -1e10 else (mpmath.mpf(1) if x > 1e10 else mpmath.ncdf(x))


def _mp_log_ndtr(x):
    if x >= 0:
        return mpmath.log1p(-_mp_ndtr(-x))
    if x < -1e10:  # log(phi(x) / |x|) + log(1 - 1/x^2 + ...)
        return -x * x / 2 - mpmath.log(-x) - mpmath.log(2 * mpmath.pi) / 2 + mpmath.log1p(-1 / (x * x))
    return mpmath.log(mpmath.ncdf(x))


def _newton_log(g, dg, x0):
    """Solve g(x) = 0 by Newton's method from x0 (g a log-space residual, so deep tails converge)."""
    x = mpmath.mpf(x0)
    for _ in range(100):
        dx = g(x) / dg(x)
        x -= dx
        if abs(dx) <= abs(x) * mpmath.mpf(10) ** -45 or dx == 0:
            return x
    raise RuntimeError("no convergence")


def _start(name, v):
    import stardust as sd  # our own value is a good start; Newton makes the reference independent of it
    return getattr(sd.special, name)(float(v))


def _mp_ndtri(p):
    x0 = _start("ndtri", p)
    if p > 0.5:  # solve on the upper tail, where 1 - p is exact
        q = 1 - p
        return -_newton_log(lambda x: mpmath.log(mpmath.ncdf(x)) - mpmath.log(q),
                            lambda x: mpmath.npdf(x) / mpmath.ncdf(x), -x0)
    return _newton_log(lambda x: mpmath.log(mpmath.ncdf(x)) - mpmath.log(p), lambda x: mpmath.npdf(x) / mpmath.ncdf(x), x0)


def _derfc(x):
    return -2 / mpmath.sqrt(mpmath.pi) * mpmath.exp(-x * x)


def _mp_erfinv(y):
    if y == 0:
        return mpmath.mpf(0)
    s, a = (1, y) if y > 0 else (-1, -y)
    x0 = abs(_start("erfinv", a))
    if a < 0.5:
        r = _newton_log(lambda x: mpmath.log(mpmath.erf(x)) - mpmath.log(a), lambda x: -_derfc(x) / mpmath.erf(x), x0)
    else:
        c = 1 - a
        r = _newton_log(lambda x: mpmath.log(mpmath.erfc(x)) - mpmath.log(c), lambda x: _derfc(x) / mpmath.erfc(x), x0)
    return s * r


def _mp_erfcinv(z):
    if z == 1:
        return mpmath.mpf(0)
    s, c = (1, z) if z < 1 else (-1, 2 - z)
    x0 = abs(_start("erfcinv", c))
    return s * _newton_log(lambda x: mpmath.log(mpmath.erfc(x)) - mpmath.log(c), lambda x: _derfc(x) / mpmath.erfc(x), x0)


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


def _mp_polygamma(n, x):
    return mpmath.psi(int(n), x)


def _mp_binom(n, k):
    if k == mpmath.floor(k) and k >= 0:  # the falling factorial (S3 user decision)
        r = mpmath.mpf(1)
        for i in range(1, int(k) + 1):
            r *= (n - k + i) / i
        return r
    return mpmath.binomial(n, k)


def _mp_beta(a, b):
    return mpmath.beta(a, b)


def _mp_zeta(x):
    return mpmath.zeta(x)


def _zero_at_zero(f):
    return lambda *a: mpmath.mpf(0) if a[0] == 0 else f(*a)


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


def _mp_gammainc(a, x):
    try:
        return mpmath.gammainc(a, 0, x, regularized=True)
    except mpmath.libmp.NoConvergence:  # large a: P = x^a e^-x / Gamma(a+1) 1F1(1; a+1; x), more terms
        return mpmath.exp(a * mpmath.log(x) - x - mpmath.loggamma(a + 1)) * mpmath.hyp1f1(1, a + 1, x, maxterms=10**7)


def _mp_gammaincc(a, x):
    try:
        return mpmath.gammainc(a, x, mpmath.inf, regularized=True)
    except mpmath.libmp.NoConvergence:
        with mpmath.workdps(2 * mpmath.mp.dps):
            return 1 - _mp_gammainc(a, x)


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


def _mp_betainc(a, b, x):
    return mpmath.betainc(a, b, 0, x, regularized=True)


def _mp_betaincc(a, b, x):
    return mpmath.betainc(b, a, 0, 1 - x, regularized=True)  # 1 - x is exact in mpmath


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


def _solve_log(logF, dlogF, x0, hi=None):
    """The root of log F(x) = 0 by Newton's method in s = log x, from x0 > 0 (our own value: the
    reference does not depend on it beyond the start), kept below `hi`. dlogF(x) = x F'(x) / F(x)."""
    x = mpmath.mpf(x0)
    for _ in range(200):
        xn = x * mpmath.exp(-logF(x) / dlogF(x))
        if hi is not None and xn >= hi:
            xn = (x + hi) / 2
        if abs(xn - x) <= abs(xn) * mpmath.mpf(10) ** -40:
            return xn
        x = xn
    raise RuntimeError("no convergence")


def _gamma_inverse(upper):
    """gammaincinv (P(a, x) = p) or gammainccinv (Q(a, x) = p): solves the smaller of p, 1 - p
    against its own function. Returns (x, kappa), kappa = |p / (x F'(x))|."""
    name = "gammainccinv" if upper else "gammaincinv"

    def f(a, p):
        x0 = _start2(name, a, p)
        lg = mpmath.loggamma(a)
        dP = lambda x: mpmath.exp((a - 1) * mpmath.log(x) - x - lg)
        P = lambda x: mpmath.gammainc(a, 0, x, regularized=True)
        Q = lambda x: mpmath.gammainc(a, x, mpmath.inf, regularized=True)
        use_p = (p <= 0.5) != upper  # the function whose value at the root is <= 1/2
        t = p if p <= 0.5 else 1 - p
        F = P if use_p else Q
        tiny = mpmath.mpf(2) ** -1075
        if x0 == 0:  # ours underflowed: the reference is 0 if the root is below 2^-1075
            if (P(tiny) >= t) if use_p else (Q(tiny) <= t):
                return mpmath.mpf(0), 1.0
            x0 = 2.0**-1074
        x = _solve_log(lambda x: mpmath.log(F(x)) - mpmath.log(t),
                       lambda x: (1 if use_p else -1) * x * dP(x) / F(x), x0)
        return x, abs(p / (x * dP(x)))
    return f


def _beta_inverse(upper):
    """betaincinv (I_x(a, b) = p) or betainccinv (1 - I = p). At mpmath's precision 1 - p is
    exact, so the function solved is the one mpmath evaluates accurately at the root: I_x(a, b)
    in x when the root is below 1/2, I_y(b, a) = 1 - I in y = 1 - x above (mpmath's betainc
    near an argument of 1 - 1e-28 loses digits). Returns (x, kappa), kappa = |p / (x F'(x))|."""
    name = "betainccinv" if upper else "betaincinv"

    def f(a, b, p):
        x0 = _start2(name, a, b, p)
        if x0 == 0 or x0 == 1:
            return mpmath.mpf(x0), 1.0
        lb = mpmath.log(mpmath.beta(a, b))
        dI = lambda x: mpmath.exp((a - 1) * mpmath.log(x) + (b - 1) * mpmath.log1p(-x) - lb)
        target_i = 1 - p if upper else p
        if x0 <= 0.5:
            F = lambda x: mpmath.betainc(a, b, 0, x, regularized=True)
            x = _solve_log(lambda x: mpmath.log(F(x)) - mpmath.log(target_i), lambda x: x * dI(x) / F(x), x0, 1)
        else:
            G = lambda y: mpmath.betainc(b, a, 0, y, regularized=True)
            y = _solve_log(lambda y: mpmath.log(G(y)) - mpmath.log(1 - target_i), lambda y: y * dI(1 - y) / G(y),
                           1 - mpmath.mpf(x0), 1)
            x = 1 - y
        return x, abs(p / (x * dI(x)))
    return f


def _start2(name, *v):
    import stardust as sd
    return float(getattr(sd.special, name)(*(float(t) for t in v)))


# S5: exponential and elliptic integrals, guaranteed everywhere (S5 user decision; expn's
# sets reach n = 1e5). The reductions have their own file (build_reduce).


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


def _envelope_scale(kind, n):
    """scale(args, value) for J (kind 'j') or Y ('y') of order n (None: the first argument)."""
    def scale(args, v):
        nu = mpmath.mpf(args[0]) if n is None else mpmath.mpf(n)
        x = mpmath.mpf(args[-1])
        if x <= 0:
            return 1.0
        M = mpmath.sqrt(mpmath.besselj(nu, x) ** 2 + mpmath.bessely(nu, x) ** 2)
        if nu >= 0:
            z1 = mpmath.besseljzero(nu, 1) if kind == "j" else mpmath.besselyzero(nu, 1)
            if x < mpmath.mpf("0.99") * z1:
                return 1.0
        return max(1.0, float(M / abs(v))) if v != 0 else float("inf")
    return scale


def _iv_scale(args, v):
    nu, x = mpmath.mpf(args[0]), mpmath.mpf(args[1])
    if nu >= 0 or nu == mpmath.floor(nu) or x <= 0:
        return 1.0
    a = -nu
    size = abs(mpmath.besseli(a, x)) + abs(2 / mpmath.pi * mpmath.sin(a * mpmath.pi) * mpmath.besselk(a, x))
    return max(1.0, float(size / abs(v))) if v != 0 else float("inf")


def _order_inputs(lo_x, hi_x, negative=True):
    def f(rng):
        m = 500
        nu = np.concatenate([rng.uniform(-50 if negative else 0, 50, 3 * m), rng.uniform(-1000 if negative else 0, 1000, m // 2),
                             rng.integers(-20 if negative else 0, 21, m // 2).astype(float)])
        x = np.concatenate([loguniform(rng, lo_x, hi_x, 3 * m), loguniform(rng, 30, hi_x, m // 2), loguniform(rng, lo_x, hi_x, m // 2)])
        return (nu, x), np.ones(nu.size, dtype=bool)
    return f


def _wide(f):
    """f(v, x) with 0.9 |v| + 20 more digits for |v| > 20: mpmath forms K_v, Y_v (and J_v for
    v < 0) through the reflection (I_-v - I_v) / sin(v pi), cancelling ~0.43 |v| digits, and
    returns a wrong value at its default precision without noticing (besselk(806.7, 626.3) =
    -4.3e66; the value is 2.2e-70). Checked against 2x the digits."""
    def g(v, x):
        extra = int(0.9 * abs(v)) + 20 if abs(v) > 20 else 0
        with mpmath.workdps(mpmath.mp.dps + extra):
            return +f(v, x)
    return g


def _airy_inputs(rng):
    m = 700
    x = np.concatenate([rng.uniform(-2, 1, m), loguniform(rng, 1, 110, m), -rng.uniform(2, 9.7, m),
                        -loguniform(rng, 9.7, 1e12, m), loguniform(rng, 1e-300, 1e-3, m // 4, signed=True)])
    return (x,), np.ones(x.size, dtype=bool)


def _airy_part(i):
    fs = [mpmath.airyai, lambda x: mpmath.airyai(x, 1), mpmath.airybi, lambda x: mpmath.airybi(x, 1)]
    return fs[i]


def _airy_scale(i):
    def scale(args, v):
        x = mpmath.mpf(args[0])
        if x >= 0:
            return 1.0
        if i % 2 == 0:
            env = mpmath.sqrt(mpmath.airyai(x) ** 2 + mpmath.airybi(x) ** 2)
        else:
            env = mpmath.sqrt(mpmath.airyai(x, 1) ** 2 + mpmath.airybi(x, 1) ** 2)
        return max(1.0, float(env / abs(v))) if v != 0 else float("inf")
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


INVERSES = {"gammaincinv", "gammainccinv", "betaincinv", "betainccinv"}


FUNCTIONS = {
    # the engine's test function (csrc/special/ops/testing.hpp): checks the pipeline end to end
    "_cf_tan": (1, mpmath.tan, _cf_tan_inputs),
    # S1
    "expit": (1, lambda x: 1 / (1 + mpmath.exp(-x)), _expit_inputs),
    "logit": (1, lambda p: mpmath.log(p / (1 - p)), _logit_inputs),
    "log_expit": (1, lambda x: -mpmath.log1p(mpmath.exp(-x)), _expit_inputs),
    "xlogy": (2, _zero_at_zero(lambda x, y: x * mpmath.log(y)), _xlogy_inputs),
    "xlog1py": (2, _zero_at_zero(lambda x, y: x * mpmath.log1p(y)), _xlog1py_inputs),
    "entr": (1, _zero_at_zero(lambda x: -x * mpmath.log(x)), _entr_inputs),
    "rel_entr": (2, _zero_at_zero(lambda x, y: x * mpmath.log(x / y)), _rel_entr_inputs),
    # S2
    "erf": (1, lambda x: mpmath.erf(x), _erf_inputs),
    "erfc": (1, _mp_erfc, _erfc_inputs),
    "erfcx": (1, _mp_erfcx, _erfcx_inputs),
    "ndtr": (1, _mp_ndtr, _ndtr_inputs),
    "log_ndtr": (1, _mp_log_ndtr, _log_ndtr_inputs),
    "ndtri": (1, _mp_ndtri, _ndtri_inputs),
    "erfinv": (1, _mp_erfinv, _erfinv_inputs),
    "erfcinv": (1, _mp_erfcinv, _erfcinv_inputs),
    # S3
    "gamma": (1, mpmath.gamma, _gamma_inputs),
    "rgamma": (1, mpmath.rgamma, _rgamma_inputs),
    "gammaln": (1, lambda x: mpmath.log(abs(mpmath.gamma(x))), _gammaln_inputs),
    "digamma": (1, mpmath.digamma, _digamma_inputs),
    "polygamma": (2, _mp_polygamma, _polygamma_inputs),
    "beta": (2, _mp_beta, _beta_inputs),
    "betaln": (2, lambda a, b: mpmath.log(abs(mpmath.beta(a, b))), _betaln_inputs),
    "binom": (2, _mp_binom, _binom_inputs),
    "zeta": (1, _mp_zeta, _zeta_inputs),
    "zeta_hurwitz": (2, lambda x, q: mpmath.zeta(x, q), _hurwitz_inputs),
    # S4
    "gammainc": (2, _mp_gammainc, _gammainc_inputs),
    "gammaincc": (2, _mp_gammaincc, _gammainc_inputs),
    "gammaincinv": (2, _gamma_inverse(False), _gammaincinv_inputs),
    "gammainccinv": (2, _gamma_inverse(True), _gammaincinv_inputs),
    "betainc": (3, _mp_betainc, _betainc_inputs),
    "betaincc": (3, _mp_betaincc, _betainc_inputs),
    "betaincinv": (3, _beta_inverse(False), _betaincinv_inputs),
    "betainccinv": (3, _beta_inverse(True), _betaincinv_inputs),
    # S5
    "expi": (1, mpmath.ei, _expi_inputs),
    "expn": (2, lambda n, x: mpmath.expint(int(n), x), _expn_inputs),
    "ellipk": (1, mpmath.ellipk, _ellip_inputs),
    "ellipe": (1, mpmath.ellipe, _ellip_inputs),
    # S6
    "i0": (1, lambda x: mpmath.besseli(0, x), _bessel01_inputs(1e-3, 713.9, True)),
    "i1": (1, lambda x: mpmath.besseli(1, x), _bessel01_inputs(1e-3, 713.9, True)),
    "i0e": (1, lambda x: mpmath.besseli(0, x) * mpmath.exp(-abs(x)), _bessel01_inputs(1e-3, 1e300, True)),
    "i1e": (1, lambda x: mpmath.besseli(1, x) * mpmath.exp(-abs(x)), _bessel01_inputs(1e-3, 1e300, True)),
    "k0": (1, lambda x: mpmath.besselk(0, x), _bessel01_inputs(1e-3, 745)),
    "k1": (1, lambda x: mpmath.besselk(1, x), _bessel01_inputs(1e-3, 745)),
    "k0e": (1, lambda x: mpmath.besselk(0, x) * mpmath.exp(x), _bessel01_inputs(1e-3, 1e300)),
    "k1e": (1, lambda x: mpmath.besselk(1, x) * mpmath.exp(x), _bessel01_inputs(1e-3, 1e300)),
    "j0": (1, lambda x: mpmath.besselj(0, x), _jy01_inputs),
    "j1": (1, lambda x: mpmath.besselj(1, x), _jy01_inputs),
    "y0": (1, lambda x: mpmath.bessely(0, x), _jy01_inputs),
    "y1": (1, lambda x: mpmath.bessely(1, x), _jy01_inputs),
    "jv": (2, _wide(mpmath.besselj), _order_inputs(1e-3, 1e5)),
    "yv": (2, _wide(mpmath.bessely), _order_inputs(1e-3, 1e5)),
    "iv": (2, mpmath.besseli, _order_inputs(1e-3, 700)),
    "kv": (2, _wide(mpmath.besselk), _order_inputs(1e-3, 700)),
    "ive": (2, lambda v, x: mpmath.besseli(v, x) * mpmath.exp(-x), _order_inputs(1e-3, 1e5, False)),
    "kve": (2, _wide(lambda v, x: mpmath.besselk(v, x) * mpmath.exp(x)), _order_inputs(1e-3, 1e5)),
    "airy_ai": (1, _airy_part(0), _airy_inputs),
    "airy_aip": (1, _airy_part(1), _airy_inputs),
    "airy_bi": (1, _airy_part(2), _airy_inputs),
    "airy_bip": (1, _airy_part(3), _airy_inputs),
    # S7
    "lambertw": (1, lambda x: mpmath.lambertw(x, 0), _lambertw_inputs(0)),
    "lambertw_m1": (1, lambda x: mpmath.lambertw(x, -1), _lambertw_inputs(-1)),
    "hyp1f1": (3, mpmath.hyp1f1, _hyp1f1_inputs),
    "hyp2f1": (4, mpmath.hyp2f1, _hyp2f1_inputs),
}


# ---- S8: complex arguments (docs/api/special.md §12). The relative error is |error| / |f|
# (S8 user decision); `scale` carries the envelope where zeros make it unreachable:
#   wofz  Im z < 0:  (2 |exp(-z^2)| + |w(-z)|) / |w(z)|
#   erfc  Re z < 0:  (2 + |erfc(-z)|) / |erfc(z)|
#   erf   |z| >= 2:  (1 + |erfc(+-z)|) / |erf(z)|  (the sign making Re >= 0)
#   erfcx Re z < 0:  (2 |exp(z^2)| + |erfcx(-z)|) / |erfcx(z)|
#   loggamma Re z <= 0: max(1, 1/|f|)  (|error| <= 1e-14 max(1, |f|))
# mpmath has no signed zeros: on a branch cut, Im z = -0 is the limit from below, taken
# as conj(f(conj z)) (loggamma) or conj(W_{-k}(conj z)) (lambertw).


def _w_mp(z):
    return mpmath.exp(-z * z) * mpmath.erfc(-1j * z)


def _erfcx_mp(z):
    return mpmath.exp(z * z) * mpmath.erfc(z)


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


def _refine(f, guesses):
    out = []
    for g in guesses:
        with mpmath.workdps(30):
            z = complex(mpmath.findroot(f, mpmath.mpc(g.real, g.imag)))
        out.append(z)
    return out


def _erfc_zeros():
    zs = _refine(mpmath.erfc, _ERFC_ZEROS)
    return zs + [z.conjugate() for z in zs]


def _erf_zeros():
    zs = _refine(mpmath.erf, _ERF_ZEROS)
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
    return max(1.0, float((2 * abs(mpmath.exp(-z * z)) + abs(_w_mp(-z))) / abs(v)))


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
    if abs(z) < 2:
        return 1.0
    zp = z if z.real >= 0 else -z
    return max(1.0, float((1 + abs(mpmath.erfc(zp))) / abs(v)))


def _erfc_scale(z, v):
    if z.real >= 0:
        return 1.0
    return max(1.0, float((2 + abs(mpmath.erfc(-z))) / abs(v)))


def _erfcx_scale(z, v):
    if z.real >= 0:
        return 1.0
    return max(1.0, float((2 * abs(mpmath.exp(z * z)) + abs(_erfcx_mp(-z))) / abs(v)))


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
    return max(1.0, float(1 / abs(v)))


def _lambertw_mp(k):
    return lambda z: mpmath.lambertw(z, k)


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


# name: (mpmath function of mpc z, input builder, scale or None, cut): on the negative real
# axis with Im z = -0 the value is conj(cut(x)) (None: no cut there)
COMPLEX = {
    "wofz": (_w_mp, _wofz_inputs, _wofz_scale, None),
    "erf_complex": (mpmath.erf, _erf_family_inputs(_erf_zeros), _erf_scale, None),
    "erfc_complex": (mpmath.erfc, _erf_family_inputs(_erfc_zeros), _erfc_scale, None),
    "erfcx_complex": (_erfcx_mp, _erf_family_inputs(lambda: [-z for z in _erfc_zeros()]), _erfcx_scale, None),
    "loggamma_complex": (mpmath.loggamma, _loggamma_inputs, _loggamma_scale, mpmath.loggamma),
    "lambertw_c0": (_lambertw_mp(0), _lambertw_c_inputs, None, _lambertw_mp(0)),
    "lambertw_cm1": (_lambertw_mp(-1), _lambertw_c_inputs, None, _lambertw_mp(1)),
    "lambertw_c1": (_lambertw_mp(1), _lambertw_c_inputs, None, _lambertw_mp(-1)),
    "lambertw_c4": (_lambertw_mp(4), _lambertw_c_inputs, None, _lambertw_mp(-4)),
}


def evaluate_complex(f, z):
    """f(z) at enough digits for exp(-z^2)'s phase (2 log10|z| more), checked against twice the
    digits until both parts agree to 30 digits."""
    big = max(1.0, abs(z))
    dps = DPS + 2 * int(np.log10(big)) + 10
    def at(d):
        with mpmath.workdps(d):
            return +f(mpmath.mpc(z.real, z.imag))
    v = at(dps)
    while dps < 2000:
        w = at(2 * dps)
        tol = abs(w) * mpmath.mpf(10) ** -30
        if not (mpmath.isfinite(w.real) and mpmath.isfinite(w.imag)) or abs(v - w) <= tol:
            return w
        dps *= 2
        v = w
    return v


def build_complex(name):
    import signal

    f, inputs, scale_of, cut = COMPLEX[name]
    rng = np.random.default_rng(20261114)
    (z,), guaranteed = inputs(rng)
    z = np.asarray(z, dtype=np.complex128)
    n = z.size
    hi, lo, scale = np.empty(n, dtype=np.complex128), np.empty(n, dtype=np.complex128), np.ones(n)
    keep = np.ones(n, dtype=bool)
    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        for i in range(n):
            signal.alarm(POINT_SECONDS)
            try:
                zi = complex(z[i])
                if cut is not None and zi.imag == 0 and np.signbit(zi.imag) and zi.real < 0:
                    v = mpmath.conj(evaluate_complex(cut, zi))  # the limit from below
                else:
                    v = evaluate_complex(f, zi)
                if scale_of is not None and mpmath.isfinite(v.real) and mpmath.isfinite(v.imag) and v != 0:
                    with mpmath.workdps(40):
                        scale[i] = scale_of(mpmath.mpc(z[i].real, z[i].imag), v)
            except (_Timeout, mpmath.libmp.NoConvergence, RuntimeError, ZeroDivisionError, ValueError):
                keep[i] = False
                continue
            finally:
                signal.alarm(0)
            rh, rl = double_double(mpmath.re(v))
            ih, il = double_double(mpmath.im(v))
            hi[i], lo[i] = complex(rh, ih), complex(rl, il)
    finally:
        signal.signal(signal.SIGALRM, old)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", nin=1, hi=hi[keep], lo=lo[keep], scale=scale[keep],
                        guaranteed=np.asarray(guaranteed, dtype=bool)[keep], x0=z[keep],
                        mpmath=mpmath.__version__, dps=DPS)
    return int(keep.sum()), int((~keep).sum())


def build_reduce():
    """tests/special/ref/reduce_cases.npz: arrays for logsumexp, softmax and log_softmax with
    their references (S5). Each case is a 1-d array; `start` holds the offsets into the
    concatenation `x`. Per case: logsumexp (hi, lo); per element: softmax and log_softmax (hi, lo)."""
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
        with mpmath.workdps(80):
            xs = [mpmath.mpf(float(v)) for v in x]
            m = max(xs)
            s = mpmath.fsum(mpmath.exp(v - m) for v in xs)
            lse = m + mpmath.log(s)
            lse_hi[c], lse_lo[c] = double_double(lse)
            for i, v in enumerate(xs):
                sm_hi[start[c] + i], sm_lo[start[c] + i] = double_double(mpmath.exp(v - lse))
                ls_hi[start[c] + i], ls_lo[start[c] + i] = double_double(v - lse)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "reduce_cases.npz", x=allx, start=start, lse_hi=lse_hi, lse_lo=lse_lo,
                        softmax_hi=sm_hi, softmax_lo=sm_lo, log_softmax_hi=ls_hi, log_softmax_lo=ls_lo,
                        mpmath=mpmath.__version__)
    return len(cases)


# ---------------------------------------------------------------- writing


def double_double(v):
    """(hi, lo) with hi + lo ~= v, hi = v rounded to float64."""
    if mpmath.isnan(v):
        return np.nan, 0.0
    if mpmath.isinf(v):
        return (np.inf if v > 0 else -np.inf), 0.0
    hi = float(v)
    if np.isinf(hi):  # beyond float64: the reference overflows
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def _at(f, args, dps):
    with mpmath.workdps(dps):
        try:
            return +f(*(mpmath.mpf(a) for a in args))
        except (ValueError, ZeroDivisionError):
            return mpmath.nan


def evaluate(f, args):
    """f(args) at DPS digits, checked against twice the digits and doubled until the two agree
    to 30 digits: mpmath loses digits silently in places (a + b with a/b = 1e290 in beta,
    Hurwitz zeta at large q), and the reference must not."""
    # start above the largest argument's decimal exponent: a sum a + b with a = 1e281 needs
    # 281 more digits to keep b, and two precisions that both drop it would agree
    big = max([abs(a) for a in args if np.isfinite(a)] + [1.0])
    dps = DPS + int(np.log10(big)) + 10
    v = _at(f, args, dps)
    while dps < 4000:
        w = _at(f, args, 2 * dps)
        if isinstance(v, mpmath.mpc) or isinstance(w, mpmath.mpc):
            v = w
            break
        if mpmath.isnan(v) or mpmath.isnan(w) or mpmath.isinf(w) or v == w or abs(v - w) <= abs(w) * mpmath.mpf(10) ** -30:
            v = w
            break
        dps *= 2
        v = w
    return v


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def evaluate_inverse(f, args):
    """(x, kappa) of an inverse, at two precisions doubled until x agrees to 30 digits."""
    dps = DPS + 10
    with mpmath.workdps(dps):
        v, k = f(*(mpmath.mpf(a) for a in args))
    while dps < 1000:
        with mpmath.workdps(2 * dps):
            w, k = f(*(mpmath.mpf(a) for a in args))
        if v == w or abs(v - w) <= abs(w) * mpmath.mpf(10) ** -30:
            return w, k
        dps *= 2
        v = w
    return v, k


# mpmath's betainc near the mean of a, b ~ 1e5 can take minutes per point: a point that
# takes longer than this is dropped (and counted).
POINT_SECONDS = 30


def build(name):
    import signal

    nin, f, inputs = FUNCTIONS[name]
    rng = np.random.default_rng(20261113)
    xs, guaranteed = inputs(rng)
    xs = tuple(np.asarray(x, dtype=np.float64) for x in xs)
    n = xs[0].size
    hi, lo, scale = np.empty(n), np.empty(n), np.ones(n)
    keep = np.ones(n, dtype=bool)
    old = signal.signal(signal.SIGALRM, _alarm)
    try:
        for i in range(n):
            args = tuple(float(x[i]) for x in xs)
            signal.alarm(POINT_SECONDS)
            try:
                if name in INVERSES:
                    v, k = evaluate_inverse(f, args)
                    scale[i] = max(1.0, float(k))
                else:
                    v = evaluate(f, args)
                    if name in SCALES and mpmath.isfinite(v):
                        with mpmath.workdps(40):
                            scale[i] = SCALES[name](args, v)
            except (_Timeout, mpmath.libmp.NoConvergence, RuntimeError, ZeroDivisionError):
                keep[i] = False
                continue
            finally:
                signal.alarm(0)
            if isinstance(v, mpmath.mpc):  # a real function outside its real domain
                v = v.real if v.imag == 0 else mpmath.nan
            hi[i], lo[i] = double_double(v)
    finally:
        signal.signal(signal.SIGALRM, old)
    xs = tuple(x[keep] for x in xs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", nin=nin, hi=hi[keep], lo=lo[keep], scale=scale[keep],
                        guaranteed=np.asarray(guaranteed, dtype=bool)[keep],
                        **{f"x{k}": x for k, x in enumerate(xs)}, mpmath=mpmath.__version__, dps=DPS)
    return int(keep.sum()), int((~keep).sum())


def main(argv):
    names = argv or list(FUNCTIONS) + list(COMPLEX) + ["reduce_cases"]
    for name in names:
        if name in COMPLEX:
            n, dropped = build_complex(name)
            note = f" ({dropped} dropped)" if dropped else ""
            print(f"{name}: {n} points -> tests/special/ref/{name}.npz{note}", flush=True)
            continue
        if name == "reduce_cases":
            print(f"reduce_cases: {build_reduce()} arrays -> tests/special/ref/reduce_cases.npz", flush=True)
            continue
        n, dropped = build(name)
        note = f" ({dropped} dropped: over {POINT_SECONDS} s, or mpmath did not converge)" if dropped else ""
        print(f"{name}: {n} points -> {(OUT / name).with_suffix('.npz').relative_to(ROOT)}{note}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
