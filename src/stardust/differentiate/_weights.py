"""The weights of derivative's difference formulas, correctly rounded (decision C28 in
docs/design/integrate.md §1.1).

scipy's _derivative_weights solves the Vandermonde system of the stencil's nodes with
np.linalg.solve (LAPACK) and so rounds its weights several times: at order 8 and step factor 2,
8 of the 9 central weights differ from the exact ones, by up to 2.7e-14 relative. Here every
weight is the exact weight rounded once to the nearest double. The nodes are scipy's (the
evaluation points are not affected):

  central   -1/c^(n-1), ..., -1/c, -1, 0, 1, 1/c, ..., 1/c^(n-1)   (2n + 1 nodes)
  one-sided 0, 1, 1/d, ..., 1/d^(2n-1) with d = sqrt(c)            (2n + 1 nodes)

for n = (order + 1) // 2 and the step factor c. The weight of node t_i is the derivative at 0
of its Lagrange basis polynomial, L_i'(0) = prod_{j != i, t_j != 0} (-t_j) / prod_{j != i}
(t_i - t_j) (and -sum_{j != i} 1 / t_j at t_i = 0): the exact solution of scipy's system.

c is a double, so it is a rational and the central weights are rationals (computed with
fractions.Fraction; the centre's is exactly 0 and the others are exactly antisymmetric, which
scipy imposes after its solve). The one-sided nodes lie in Q(sqrt(c)): the weights are computed
exactly as a + b sqrt(c) with rational a and b, then rounded to the nearest double by enclosing
sqrt(c) between integers at increasing precision until both ends round alike (sqrt(c) is
irrational unless c is the square of a rational, when it is computed as a rational). Arb is not
a dependency of the library, so this is the exact computation; tests/differentiate check every
weight against Arb.
"""

import functools
import math
from fractions import Fraction


def _round(v):
    """A rational as the nearest double (Python's int/int true division rounds correctly),
    +-inf beyond the doubles."""
    try:
        return float(v)
    except OverflowError:
        return math.inf if v > 0 else -math.inf


class _Quad:
    """a + b sqrt(c) with rational a, b and a rational c that is not a rational square."""

    __slots__ = ("a", "b", "c")

    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def __add__(self, o):
        return _Quad(self.a + o.a, self.b + o.b, self.c)

    def __sub__(self, o):
        return _Quad(self.a - o.a, self.b - o.b, self.c)

    def __neg__(self):
        return _Quad(-self.a, -self.b, self.c)

    def __eq__(self, o):  # by value (the zero test of _lagrange)
        return self.a == o.a and self.b == o.b

    __hash__ = None

    def __mul__(self, o):
        return _Quad(self.a * o.a + self.b * o.b * self.c, self.a * o.b + self.b * o.a, self.c)

    def inverse(self):
        norm = self.a * self.a - self.b * self.b * self.c  # nonzero: sqrt(c) is irrational
        return _Quad(self.a / norm, -self.b / norm, self.c)

    def __truediv__(self, o):
        return self * o.inverse()

    def rounded(self):
        """The nearest double to a + b sqrt(c)."""
        if self.b == 0:
            return _round(self.a)
        # sqrt(c) = sqrt(p q) / q for c = p / q; S <= sqrt(p q) 2^K < S + 1
        p, q = self.c.numerator, self.c.denominator
        k = 64
        while True:
            s = math.isqrt((p * q) << (2 * k))
            lo = self.a + self.b * Fraction(s, q << k)
            hi = self.a + self.b * Fraction(s + 1, q << k)
            r0, r1 = _round(lo), _round(hi)
            if r0 == r1:  # the value lies between them, so it rounds alike (an irrational is no tie)
                return r0
            k *= 2


def _rational_sqrt(c):
    """sqrt(c) as a Fraction when c is the square of a rational, else None."""
    p, q = c.numerator, c.denominator
    rp, rq = math.isqrt(p), math.isqrt(q)
    if rp * rp == p and rq * rq == q:
        return Fraction(rp, rq)
    return None


def _lagrange(nodes, zero, one):
    """The weights L_i'(0) of the nodes (one of them 0) in their field (zero and one its
    elements)."""
    out = []
    for i, ti in enumerate(nodes):
        den = one
        for j, tj in enumerate(nodes):
            if j != i:
                den = den * (ti - tj)
        if ti == zero:
            s = zero
            for j, tj in enumerate(nodes):
                if j != i:
                    s = s - one / tj
            out.append(s)
            continue
        num = one
        for j, tj in enumerate(nodes):
            if j != i and tj != zero:
                num = num * (-tj)
        out.append(num / den)
    return out


@functools.lru_cache(maxsize=64)
def central_weights(n, c):
    """The 2n + 1 central weights (doubles) for n terms and the step factor c (a finite double
    > 0, not 1), in the order of scipy's stencil (above)."""
    cf = Fraction(c)
    central = [-(cf ** -(k - 1)) for k in range(n, 0, -1)] + [Fraction(0)] + [cf ** -(k - 1) for k in range(1, n + 1)]
    return tuple(_round(w) for w in _lagrange(central, Fraction(0), Fraction(1)))


@functools.lru_cache(maxsize=64)
def one_sided_weights(n, c):
    """The 2n + 1 one-sided weights (doubles) for n terms and the step factor c, in the order of
    scipy's stencil (above). Computed only when an element takes a side: in Q(sqrt(c)) they cost
    about 5 times the central ones (the largest n = 9, c = 2: 2.0 ms of 2.4 ms)."""
    cf = Fraction(c)
    root = _rational_sqrt(cf)
    if root is not None:
        side = [Fraction(0)] + [root**-k for k in range(2 * n)]
        return tuple(_round(w) for w in _lagrange(side, Fraction(0), Fraction(1)))
    zero, one = _Quad(Fraction(0), Fraction(0), cf), _Quad(Fraction(1), Fraction(0), cf)

    def inv_root_power(k):  # d^-k = c^(-k/2): rational for even k, sqrt(c) c^(-(k+1)/2) for odd
        if k % 2 == 0:
            return _Quad(cf ** -(k // 2), Fraction(0), cf)
        return _Quad(Fraction(0), cf ** -((k + 1) // 2), cf)

    side = [zero] + [inv_root_power(k) for k in range(2 * n)]
    return tuple(w.rounded() for w in _lagrange(side, zero, one))


def weights(n, c):
    """(central, one_sided): the two tuples of 2n + 1 doubles (central_weights,
    one_sided_weights; each kept once computed)."""
    return central_weights(n, c), one_sided_weights(n, c)
