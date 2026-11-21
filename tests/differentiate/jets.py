"""Second-order forward-mode jets over Arb balls: the exact gradient and Hessian of an expression
at a point, for the Hessian references of tools/differentiate_ref.py.

ArbJet is tests/optimize/ad.py's Dual2 with arb values: a value, its gradient and its Hessian
with respect to n variables (numpy object arrays of arb). JET is the namespace M of the
optimize problems (tests/optimize/mgh.py's functions take one) whose functions apply to jets and
to plain numbers. Only the operations the problems use are defined.
"""

from __future__ import annotations

import numpy as np


def _arb(v):
    from flint import arb

    return v if isinstance(v, arb) else arb(v)


def _zeros(*shape):
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(*shape):
        out[idx] = _arb(0)
    return out


class ArbJet:
    __slots__ = ("v", "g", "h")

    def __init__(self, v, g, h):
        self.v, self.g, self.h = _arb(v), g, h

    @staticmethod
    def variables(x):
        n = len(x)
        out = []
        for i in range(n):
            g = _zeros(n)
            g[i] = _arb(1)
            out.append(ArbJet(_arb(x[i]), g, _zeros(n, n)))
        return out

    def _lift(self, c):
        n = self.g.shape[0]
        return ArbJet(c, _zeros(n), _zeros(n, n))

    def _chain(self, f0, f1, f2):
        return ArbJet(f0, f1 * self.g, f1 * self.h + f2 * np.outer(self.g, self.g))

    def __add__(self, o):
        if isinstance(o, ArbJet):
            return ArbJet(self.v + o.v, self.g + o.g, self.h + o.h)
        return ArbJet(self.v + _arb(o), self.g, self.h)

    __radd__ = __add__

    def __neg__(self):
        return ArbJet(-self.v, -self.g, -self.h)

    def __pos__(self):
        return self

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, ArbJet):
            gg = np.outer(self.g, o.g)
            return ArbJet(self.v * o.v, self.v * o.g + o.v * self.g, self.v * o.h + o.v * self.h + gg + gg.T)
        o = _arb(o)
        return ArbJet(self.v * o, self.g * o, self.h * o)

    __rmul__ = __mul__

    def _recip(self):
        v = self.v
        return self._chain(1 / v, -1 / (v * v), 2 / (v * v * v))

    def __truediv__(self, o):
        if isinstance(o, ArbJet):
            return self * o._recip()
        return self * (1 / _arb(o))

    def __rtruediv__(self, o):
        return self._recip() * o

    def __pow__(self, p):
        if isinstance(p, ArbJet):
            return (p * self.log()).exp()
        v = self.v
        if p == 0:
            return self._lift(_arb(1))
        if float(p).is_integer():
            k = int(p)
            f1 = k * v ** (k - 1)
            f2 = k * (k - 1) * v ** (k - 2) if k != 1 else _arb(0)
            return self._chain(v**k, f1, f2)
        p = _arb(p)
        return self._chain(v**p, p * v ** (p - 1), p * (p - 1) * v ** (p - 2))

    def __rpow__(self, c):
        return (self * _arb(c).log()).exp()

    def __abs__(self):
        return -self if self.v < 0 else self

    def _val(self, o):
        return o.v if isinstance(o, ArbJet) else _arb(o)

    def __lt__(self, o):
        return self.v < self._val(o)

    def __le__(self, o):
        return self.v <= self._val(o)

    def __gt__(self, o):
        return self.v > self._val(o)

    def __ge__(self, o):
        return self.v >= self._val(o)

    def __eq__(self, o):
        return self.v == self._val(o)

    __hash__ = None

    def exp(self):
        e = self.v.exp()
        return self._chain(e, e, e)

    def log(self):
        v = self.v
        return self._chain(v.log(), 1 / v, -1 / (v * v))

    def sin(self):
        s, c = self.v.sin(), self.v.cos()
        return self._chain(s, c, -s)

    def cos(self):
        s, c = self.v.sin(), self.v.cos()
        return self._chain(c, -s, -c)

    def atan(self):
        v = self.v
        d = 1 + v * v
        return self._chain(v.atan(), 1 / d, -2 * v / (d * d))

    def sqrt(self):
        s = self.v.sqrt()
        return self._chain(s, 1 / (2 * s), -1 / (4 * s * self.v))


class _Jet:
    """ArbJet where an argument is one, Arb balls elsewhere."""

    @staticmethod
    def _f(name, x):
        return getattr(x, name)() if isinstance(x, ArbJet) else getattr(_arb(x), name)()

    def exp(self, x):
        return self._f("exp", x)

    def log(self, x):
        return self._f("log", x)

    def sin(self, x):
        return self._f("sin", x)

    def cos(self, x):
        return self._f("cos", x)

    def atan(self, x):
        return self._f("atan", x)

    def sqrt(self, x):
        return self._f("sqrt", x)

    @staticmethod
    def copysign(a, b):
        bv = b.v if isinstance(b, ArbJet) else _arb(b)
        return abs(a) if bv >= 0 else -abs(a)

    @staticmethod
    def const(v):
        return _arb(v)


JET = _Jet()


def hessian_sum_squares(fun, x):
    """The Hessian of f = sum F_i^2 at x (doubles) for fun(x, M) -> list of residuals: an n x n
    object array of arb."""
    xs = ArbJet.variables(x)
    F = fun(xs, JET)
    n = len(x)
    total = ArbJet(_arb(0), _zeros(n), _zeros(n, n))
    for r in F:
        r = r if isinstance(r, ArbJet) else total._lift(_arb(r))
        total = total + r * r
    return total.h
