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
}


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


def build(name):
    nin, f, inputs = FUNCTIONS[name]
    rng = np.random.default_rng(20261113)
    xs, guaranteed = inputs(rng)
    xs = tuple(np.asarray(x, dtype=np.float64) for x in xs)
    n = xs[0].size
    hi, lo = np.empty(n), np.empty(n)
    for i in range(n):
        v = evaluate(f, tuple(float(x[i]) for x in xs))
        if isinstance(v, mpmath.mpc):  # a real function outside its real domain
            v = v.real if v.imag == 0 else mpmath.nan
        hi[i], lo[i] = double_double(v)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", nin=nin, hi=hi, lo=lo, guaranteed=np.asarray(guaranteed, dtype=bool),
                        **{f"x{k}": x for k, x in enumerate(xs)}, mpmath=mpmath.__version__, dps=DPS)
    return n


def main(argv):
    names = argv or list(FUNCTIONS)
    for name in names:
        n = build(name)
        print(f"{name}: {n} points -> {(OUT / name).with_suffix('.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
