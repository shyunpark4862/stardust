"""The Gauss-Legendre nodes and weights of fixed_quad, correctly rounded (docs/design/integrate.md
§3.7, decision C42). Not public.

n <= 100 reads the Arb-made table (csrc/integrate/gauss_table.hpp, tools/gen_gauss_legendre.py).
Past it, csrc/integrate/gauss.cpp gives each nonnegative node and weight as a double-double
(Newton's method, about 106 bits) and this module rounds it. The rounding is the correct one
unless the true value may lie on the other side of a rounding boundary, which happens only when
the double-double is within its error bound of the boundary (C54: _tol_x and _tol_w below, at
least 2^10 times the errors measured against Arb for n = 101 ... 3000; the chance of such a value
is about n^2 2^-56 per node and n^2 2^-48 per weight). Then the value is settled by exact
arithmetic on integers:

  - a node: x rounds to c exactly when P_n changes sign between the two midpoints around c,
    c - ulp/2 and c + ulp/2, dyadic rationals at which P_n's sign is exact (_legendre_exact:
    the three-term recurrence scaled to integers);
  - a weight: w = 1/g(r), g(x) = (1 - x^2) P_n'(x)^2 / 2, at the node r enclosed by an interval
    [a, b] with a sign change; g on [a, b] lies within g(a) +- L (b - a) with
    L = 3/4 (n (n + 1))^2 (|g'| <= |x| P_n'^2 + n (n + 1) |P_n P_n'| and Markov's
    |P_n'| <= n (n + 1) / 2 on [-1, 1]), and the interval is halved until both ends of the
    enclosure of w round to the same double.
"""

import math
from fractions import Fraction

from .. import _core as _sd

_k = _sd.integrate
TABLE_MAX = _k.GAUSS_TABLE_MAX

# The double-double's error bounds (C54): a node within _tol_x(n) ulp(x), a weight within
# _tol_w(n) |w|; the measured errors were at most 2^-51 ulp and 2^-90.5 |w| (n = 101 ... 3000).
_SCALE = 1.0  # the tests raise it to send every value through the exact path


def _tol_x(n):
    return _SCALE * n * n * 2.0**-57


def _tol_w(n):
    return _SCALE * n * n * 2.0**-100


_cache = {}
_CACHE_MAX = 8  # the most recent n past the table (each holds 2n doubles)


def _ambiguous(hi, lo, tol):
    """Whether hi + lo +- tol may round to a double other than hi."""
    gap = (math.nextafter(hi, math.inf) - hi) if lo >= 0 else (hi - math.nextafter(hi, -math.inf))
    return abs(lo) + tol >= gap / 2


def _legendre_exact(n, x):
    """(P_n(x), P_{n-1}(x)) exactly at the dyadic rational x: the recurrence
    p_{k+1} = (2k + 1) M p_k - k^2 4^E p_{k-1} on p_k = k! 2^(E k) P_k(x), x = M / 2^E."""
    m, d = x.numerator, x.denominator
    e = d.bit_length() - 1
    four = 1 << (2 * e)
    q, p = 1, m
    for k in range(1, n):
        q, p = p, (2 * k + 1) * m * p - k * k * four * q
    fact = math.factorial(n - 1)
    return Fraction(p, fact * n << (e * n)), Fraction(q, fact << (e * (n - 1)))


def _sign(n, x):
    p, _ = _legendre_exact(n, x)
    return (p > 0) - (p < 0)


def _rounding_interval(c):
    """The exact interval of the reals that round to the double c > 0."""
    lo = (Fraction(c) + Fraction(math.nextafter(c, -math.inf))) / 2
    hi = (Fraction(c) + Fraction(math.nextafter(c, math.inf))) / 2
    return lo, hi


def _node_exact(n, i, c):
    """The correctly rounded i-th nonnegative node (ascending) near the double c, and an exact
    interval [a, b] holding the node."""
    r = (n + 1) // 2 - i  # the r-th largest zero: P_n has the sign (-1)^(r-1) just right of it
    right = 1 if r % 2 == 1 else -1
    for _ in range(64):
        a, b = _rounding_interval(c)
        sa, sb = _sign(n, a), _sign(n, b)
        if sa != sb:
            return c, a, b
        c = math.nextafter(c, -math.inf) if sa == right else math.nextafter(c, math.inf)
    raise ArithmeticError(f"fixed_quad: no node of P_{n} near {c!r}")


def _weight_exact(n, a, b):
    """The correctly rounded weight of the node in [a, b] (P_n changes sign on it)."""
    lip = Fraction(3, 4) * (n * (n + 1)) ** 2
    sa = _sign(n, a)
    while True:
        if a == 0:  # the middle node of an odd n
            g = Fraction(n * n) * _legendre_exact(n, a)[1] ** 2 / 2
            return float(1 / g)
        p, q = _legendre_exact(n, a)
        g = n * n * (a * p - q) ** 2 / (2 * (1 - a * a))
        spread = lip * (b - a)
        if g > spread:
            lo, hi = float(1 / (g + spread)), float(1 / (g - spread))
            if lo == hi:
                return lo
        mid = (a + b) / 2
        if _sign(n, mid) == sa:
            a = mid
        else:
            b = mid


def _settled(n):
    """The nonnegative nodes and weights of n > TABLE_MAX points, correctly rounded."""
    xh, xl, wh, wl = (v.tolist() for v in _k.legendre_dd(n))
    tx, tw = _tol_x(n), _tol_w(n)
    for i in range(len(xh)):
        exact_x = n % 2 == 1 and i == 0  # 0 itself
        if not exact_x and _ambiguous(xh[i], xl[i], tx * math.ulp(xh[i])):
            xh[i], a, b = _node_exact(n, i, xh[i])
        else:
            a = b = None
        if _ambiguous(wh[i], wl[i], tw * wh[i]):
            if a is None:
                if exact_x:
                    a = b = Fraction(0)
                else:
                    _, a, b = _node_exact(n, i, xh[i])
            wh[i] = _weight_exact(n, a, b)
    return xh, wh


def legendre(n):
    """(x, w): the n nodes ascending and their weights, float64 sd.ndarrays (cached)."""
    got = _cache.get(n)
    if got is not None:
        return got
    if n <= TABLE_MAX:
        xp, wp = (v.tolist() for v in _k.legendre_table(n))
    else:
        xp, wp = _settled(n)
    skip = 1 if n % 2 == 1 else 0  # 0 has no mirror
    x = [-v for v in reversed(xp[skip:])] + xp
    w = list(reversed(wp[skip:])) + wp
    out = (_sd.asarray(x), _sd.asarray(w))
    if n > TABLE_MAX:
        if len([k for k in _cache if k > TABLE_MAX]) >= _CACHE_MAX:
            del _cache[next(k for k in _cache if k > TABLE_MAX)]
    _cache[n] = out
    return out
