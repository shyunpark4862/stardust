"""Number types for the optimize test problems (docs/design/optimize.md §7).

Every problem in tests/optimize/mgh.py, systems.py and functions.py is written once, for a list
of scalars and a namespace `M` of elementary functions, so that the same code runs

  - in double precision (FLOAT: Python floats and the math module, in the operation order of
    the Fortran originals where there is one, so that values agree with them to the bit),
  - in Arb balls (ARB: python-flint), for derivative references at high precision,
  - in second-order forward-mode dual numbers (AD: Dual2), which give the exact gradient and
    Hessian of any expression in double precision (no differences).

Only the operations the problems use are defined: + - * / with numbers, integer and real
powers, abs, comparisons (on the value), exp, log, sin, cos, atan, sqrt, copysign.
"""

import math

import numpy as np


def _ieee(fn, x):
    """math's function with IEEE results where math raises: overflow to inf, domain errors to
    NaN (as Fortran and numpy give them), as a numpy float64 so that the arithmetic after it is
    IEEE too (division by zero gives inf, not ZeroDivisionError)."""
    try:
        return np.float64(fn(x))
    except OverflowError:
        return np.float64(math.inf)
    except ValueError:
        if fn is math.log and x == 0:
            return np.float64(-math.inf)
        return np.float64(math.nan)


class _Float:
    """Doubles: math's functions (libm, as the Fortran originals use) with IEEE special values.
    The problems' callers pass numpy float64 scalars, so that every operation is IEEE."""

    @staticmethod
    def exp(x):
        return _ieee(math.exp, x)

    @staticmethod
    def log(x):
        return _ieee(math.log, x)

    @staticmethod
    def sin(x):
        return _ieee(math.sin, x)

    @staticmethod
    def cos(x):
        return _ieee(math.cos, x)

    @staticmethod
    def atan(x):
        return _ieee(math.atan, x)

    @staticmethod
    def sqrt(x):
        return _ieee(math.sqrt, x)

    @staticmethod
    def copysign(a, b):
        return np.float64(math.copysign(a, b))

    @staticmethod
    def const(v):
        return np.float64(v)


FLOAT = _Float()


class _Arb:
    """Arb balls (python-flint) at the context's precision; doubles enter exactly."""

    @staticmethod
    def _a(x):
        from flint import arb
        return x if isinstance(x, arb) else arb(x)

    def exp(self, x):
        return self._a(x).exp()

    def log(self, x):
        return self._a(x).log()

    def sin(self, x):
        return self._a(x).sin()

    def cos(self, x):
        return self._a(x).cos()

    def atan(self, x):
        return self._a(x).atan()

    def sqrt(self, x):
        return self._a(x).sqrt()

    def copysign(self, a, b):
        return self._a(abs(a) if b >= 0 else -abs(a))

    def const(self, v):
        return self._a(v)


ARB = _Arb()


class Dual2:
    """A value with its gradient and Hessian with respect to n variables (numpy arrays)."""

    __slots__ = ("v", "g", "h")

    def __init__(self, v, g, h):
        self.v, self.g, self.h = np.float64(v), g, h  # numpy scalars: IEEE arithmetic

    @staticmethod
    def variables(x):
        n = len(x)
        eye = np.eye(n)
        return [Dual2(x[i], eye[i].copy(), np.zeros((n, n))) for i in range(n)]

    def _lift(self, c):
        n = self.g.shape[0]
        return Dual2(c, np.zeros(n), np.zeros((n, n)))

    def _chain(self, f0, f1, f2):
        """f(self) from f, f', f'' at the value."""
        return Dual2(f0, f1 * self.g, f1 * self.h + f2 * np.outer(self.g, self.g))

    def __add__(self, o):
        if isinstance(o, Dual2):
            return Dual2(self.v + o.v, self.g + o.g, self.h + o.h)
        return Dual2(self.v + o, self.g, self.h)

    __radd__ = __add__

    def __neg__(self):
        return Dual2(-self.v, -self.g, -self.h)

    def __pos__(self):
        return self

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Dual2):
            gg = np.outer(self.g, o.g)
            return Dual2(self.v * o.v, self.v * o.g + o.v * self.g,
                         self.v * o.h + o.v * self.h + gg + gg.T)
        return Dual2(self.v * o, self.g * o, self.h * o)

    __rmul__ = __mul__

    def _recip(self):
        v = self.v
        return self._chain(1.0 / v, -1.0 / (v * v), 2.0 / (v * v * v))

    def __truediv__(self, o):
        if isinstance(o, Dual2):
            return self * o._recip()
        return self * (1.0 / o)

    def __rtruediv__(self, o):
        return self._recip() * o

    def __pow__(self, p):
        if isinstance(p, Dual2):
            return (p * self.log()).exp()
        v = self.v
        if p == 0:
            return self._lift(1.0)
        if float(p).is_integer():
            k = int(p)
            f0 = v ** k
            f1 = k * v ** (k - 1)
            f2 = k * (k - 1) * v ** (k - 2) if k != 1 else 0.0
            return self._chain(f0, f1, f2)
        return self._chain(v ** p, p * v ** (p - 1), p * (p - 1) * v ** (p - 2))

    def __rpow__(self, c):
        return (self * _ieee(math.log, c)).exp()

    def __abs__(self):
        return -self if self.v < 0 else self

    def __lt__(self, o):
        return self.v < (o.v if isinstance(o, Dual2) else o)

    def __le__(self, o):
        return self.v <= (o.v if isinstance(o, Dual2) else o)

    def __gt__(self, o):
        return self.v > (o.v if isinstance(o, Dual2) else o)

    def __ge__(self, o):
        return self.v >= (o.v if isinstance(o, Dual2) else o)

    def __eq__(self, o):
        return self.v == (o.v if isinstance(o, Dual2) else o)

    __hash__ = None

    def exp(self):
        e = _ieee(math.exp, self.v)
        return self._chain(e, e, e)

    def log(self):
        v = self.v
        return self._chain(_ieee(math.log, v), 1.0 / v, -1.0 / (v * v))

    def sin(self):
        s, c = _ieee(math.sin, self.v), _ieee(math.cos, self.v)
        return self._chain(s, c, -s)

    def cos(self):
        s, c = _ieee(math.sin, self.v), _ieee(math.cos, self.v)
        return self._chain(c, -s, -c)

    def atan(self):
        v = self.v
        d = 1.0 + v * v
        return self._chain(_ieee(math.atan, v), 1.0 / d, -2.0 * v / (d * d))

    def sqrt(self):
        s = _ieee(math.sqrt, self.v)
        return self._chain(s, 0.5 / s, -0.25 / (s * self.v))


class _AD:
    """Dual2 where an argument is one, doubles elsewhere."""

    @staticmethod
    def _f(name, x):
        return getattr(x, name)() if isinstance(x, Dual2) else _ieee(getattr(math, name), x)

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
        return math.copysign(a, b.v if isinstance(b, Dual2) else b)

    @staticmethod
    def const(v):
        return float(v)


AD = _AD()


def value_grad_hess(f, x):
    """(f(x), its gradient, its Hessian) for f(list of numbers, M) -> number, by Dual2."""
    r = f(Dual2.variables([float(v) for v in x]), AD)
    if not isinstance(r, Dual2):  # a constant function
        n = len(x)
        return float(r), np.zeros(n), np.zeros((n, n))
    return r.v, r.g, r.h


def jacobian_ad(F, x):
    """The Jacobian of F(list, M) -> list of numbers by Dual2 (first order only is used)."""
    n = len(x)
    rows = []
    for r in F(Dual2.variables([float(v) for v in x]), AD):
        rows.append(r.g.copy() if isinstance(r, Dual2) else np.zeros(n))
    return np.array(rows)


def arb_jacobian(F, x, prec=320, step_bits=80):
    """The Jacobian of F(list, M) -> list at the double point x by central differences in Arb
    balls at `prec` bits with step 2^-step_bits (the truncation error is about 2^-160 times the
    third derivative: far below double precision). Returns a numpy array of the midpoints."""
    from flint import arb, ctx

    old = ctx.prec
    ctx.prec = prec
    try:
        h = arb(2) ** (-step_bits)
        n = len(x)
        cols = []
        for j in range(n):
            xp = [arb(float(v)) for v in x]
            xm = [arb(float(v)) for v in x]
            xp[j] += h
            xm[j] -= h
            fp, fm = F(xp, ARB), F(xm, ARB)
            cols.append([float(((ARB.const(a) - b) / (2 * h)).mid()) for a, b in zip(fp, fm)])
        return np.array(cols).T
    finally:
        ctx.prec = old


def arb_derivative(f, x, prec=320, step_bits=80):
    """f'(x) of a scalar f(x, M) by an Arb central difference (as arb_jacobian)."""
    return float(arb_jacobian(lambda v, M: [f(v[0], M)], [x], prec, step_bits)[0, 0])
