"""The test collection of sd.integrate (docs/design/integrate.md §7.1, decision C6).

Every integrand is written once, for a namespace M of elementary functions, so that the same
code runs

  - in double precision on Python floats (MATH: the math module, IEEE results where math
    raises), as quad's family calls it (one float at a time),
  - on numpy arrays (NP), as tanhsinh and nsum call it (elementwise, any shape),
  - in Arb balls (Acb(analytic): python-flint's acb with acb.integral's `analytic` flag), for
    the references of tools/integrate_ref.py.

The constants of an integrand (alpha, the peak's centre pi/4, the upper limit pi) are the doubles
in the code, and so is every limit, break point and weight parameter: a reference is the exact
integral of the problem as the double code poses it (the double pi is 1.2e-16 short of pi, which
moves cos(2^a sin x) on [0, pi] by about that much; sqrt(tan t) on [0, pi/2] by 1.6e-8).

Collections (an id is "<collection>/<name>[/<parameter>]"):
  book       the QUADPACK book's three test families (Piessens, de Doncker-Kapenga, Ueberhuber,
             Kahaner 1983): x^a log(1/x) on [0, 1], a = -0.9 ... 2.0 (30); the peak
             4^-a / ((x - pi/4)^2 + 16^-a) on [0, 1], a = 0 ... 20 (21); cos(2^a sin x) on
             [0, pi], a = 0 ... 10 (11)
  quadpack   the classic example of each QUADPACK routine: QAGS log(x)/sqrt(x), QAGP
             x^3 log|(x^2 - 1)(x^2 - 2)| with break points 1 and sqrt 2, QAGI log(x)/(1 + 100 x^2),
             QAWO log(x) sin(10 pi x), QAWF x^-1/2 cos(pi x / 2), QAWS log(x)/(1 + log(x)^2)^2
             with the log weight, QAWC 1/(5x^3 + 6) with the Cauchy weight at 0
  scipy      scipy 1.18.1's tests: test_quadpack.py's one-dimensional cases (typical, indefinite,
             singular with break points, the weighted ones, reversed limits) and
             test_tanhsinh.py's f1-f15 (Bailey, Jeyabalan, Li 2005, section 6) and its
             integral transforms of the normal density
  hard       integrals the probes of docs/design/integrate.md §7.4 found hard: endpoint
             singularities, an interior kink and singularities (with and without break points),
             fast oscillation, an infinite range, a narrow peak
The multiple integrals (dblquad, tplquad, nquad, cubature) are in MULTI, the vector-valued
ones (quad_vec) in VECTOR, the series (nsum) in SERIES and the samples (trapezoid, ...) in
SAMPLES. tests/integrate/test_problems.py checks the transcriptions: the three forms agree at
points, and every reference agrees with scipy's quad where quad converges.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np

INF = math.inf


# ---- the namespaces


def _ieee(fn, *args):
    """math's function with IEEE results where it raises: overflow to inf, domain errors to NaN
    (log(0) to -inf), as numpy and C give them."""
    try:
        return fn(*args)
    except OverflowError:
        return math.inf
    except ValueError:
        if fn is math.log and args[0] == 0:
            return -math.inf
        return math.nan


class _Math:
    """Python floats and the math module (quad's calls)."""

    name = "math"
    pi = math.pi

    @staticmethod
    def exp(x):
        return _ieee(math.exp, x)

    @staticmethod
    def log(x):
        return _ieee(math.log, x)

    @staticmethod
    def sqrt(x):
        return _ieee(math.sqrt, x)

    @staticmethod
    def sin(x):
        return math.sin(x)

    @staticmethod
    def cos(x):
        return math.cos(x)

    @staticmethod
    def tan(x):
        return math.tan(x)

    @staticmethod
    def atan(x):
        return math.atan(x)

    @staticmethod
    def pow(x, y):
        return _ieee(math.pow, x, y)

    @staticmethod
    def abs(x):
        return abs(x)

    @staticmethod
    def where(c, a, b):
        return a if c else b

    @staticmethod
    def const(v):
        return float(v)


class _Np:
    """numpy arrays (tanhsinh's and nsum's calls); the caller sets np.errstate."""

    name = "numpy"
    pi = math.pi
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    tan = staticmethod(np.tan)
    atan = staticmethod(np.arctan)
    pow = staticmethod(np.power)
    abs = staticmethod(np.abs)
    where = staticmethod(np.where)

    @staticmethod
    def const(v):
        return float(v)


class Acb:
    """Arb balls for acb.integral: a function that is not analytic on the ball returns a
    non-finite ball when `analytic` (acb.integral's flag) is set; doubles enter exactly."""

    name = "acb"

    def __init__(self, analytic=False):
        self.analytic = bool(analytic)

    @property
    def pi(self):
        from flint import acb

        return acb(math.pi)  # the double pi, as the code has it

    @staticmethod
    def _a(x):
        from flint import acb

        return x if isinstance(x, acb) else acb(x)

    def exp(self, x):
        return self._a(x).exp()

    def log(self, x):
        return self._a(x).log(analytic=self.analytic)

    def sqrt(self, x):
        return self._a(x).sqrt(analytic=self.analytic)

    def sin(self, x):
        return self._a(x).sin()

    def cos(self, x):
        return self._a(x).cos()

    def tan(self, x):
        return self._a(x).tan()

    def atan(self, x):
        return self._a(x).atan()

    def pow(self, x, y):
        y = float(y)
        if y == int(y) and abs(y) < 64:
            return self._a(x) ** int(y)
        return self._a(x).pow(self._a(y), analytic=self.analytic)

    def abs(self, x):
        return self._a(x).real_abs(analytic=self.analytic)

    def where(self, c, a, b):
        raise TypeError("a piecewise integrand has no Arb form here (its reference is a closed form)")

    def const(self, v):
        return self._a(v)


MATH = _Math()
NP = _Np()


# ---- the weights of quad (scipy's `weight`, `wvar`), applied to an Arb or float x


def weight_value(weight, wvar, a, b, x, M):
    """w(x) of quad's weight (docs/design/integrate.md §3.1) in the namespace M."""
    if weight is None:
        return M.const(1.0)
    if weight == "cos":
        return M.cos(M.const(wvar) * x)
    if weight == "sin":
        return M.sin(M.const(wvar) * x)
    if weight == "cauchy":
        return 1 / (x - M.const(wvar))
    al, be = wvar
    w = M.pow(x - M.const(a), al) * M.pow(M.const(b) - x, be)
    if weight in ("alg-loga", "alg-log"):
        w = w * M.log(x - M.const(a))
    if weight in ("alg-logb", "alg-log"):
        w = w * M.log(M.const(b) - x)
    return w


# ---- the problems


@dataclasses.dataclass(frozen=True)
class Integral:
    """One integral: integrand f (without the weight) on [a, b], with quad's options.

    expr(x, M): the integrand in the namespace M. exact(): the closed form as an arb, at the
    context's precision, for the doubles of the problem (None: no closed form; the reference is
    numerical). pieces: for the Arb references of an integrand that acb.integral cannot take as
    it is (an endpoint singularity, an infinite range), a list of (g, lo, hi) with g(t, M) on
    finite [lo, hi] whose sum is the integral, and tail: a rigorous bound on what the pieces
    leave out (a truncated infinite range). numeric: acb.integral can take expr(x)*w(x) on [a, b]
    as it is (finite, analytic on a neighbourhood of [a, b]).
    """

    id: str
    expr: Callable
    a: float
    b: float
    exact: Callable | None = None
    points: tuple = ()
    weight: str | None = None
    wvar: object = None
    pieces: tuple = ()
    tail: float = 0.0
    numeric: bool = False
    source: str = ""
    tanhsinh: bool = True  # tanhsinh's records include it (no weight; of a pair with and without
    # break points, the one without: tanhsinh has no break points and meets the singularity inside)
    note: str = ""

    @property
    def collection(self):
        return self.id.split("/")[0]

    @property
    def family(self):
        return "/".join(self.id.split("/")[:2])

    def f(self, x):
        """The integrand at a Python float (quad's call)."""
        return self.expr(x, MATH)

    def f_np(self, x):
        """The integrand on a numpy array (tanhsinh's call)."""
        return self.expr(np.asarray(x, dtype=np.float64), NP)

    def full(self, x, M):
        """The integrand times quad's weight (what is integrated)."""
        return self.expr(x, M) * weight_value(self.weight, self.wvar, self.a, self.b, x, M)


def _arb(v):
    from flint import arb

    return arb(v)


def _pi():
    from flint import arb

    return arb.pi()


# -- book: the QUADPACK book's three families

BOOK_XLOG_ALPHAS = tuple(float(f"{-0.9 + 0.1 * k:.1f}") for k in range(30))  # -0.9 ... 2.0
BOOK_PEAK_ALPHAS = tuple(float(k) for k in range(21))
BOOK_OSC_ALPHAS = tuple(float(k) for k in range(11))
PEAK_CENTRE = math.pi / 4  # the double


def _xlog(al):
    return Integral(
        id=f"book/xlog/{al:+.1f}",
        expr=lambda x, M, al=al: M.pow(x, al) * M.log(1 / x),
        a=0.0,
        b=1.0,
        exact=lambda al=al: 1 / (1 + _arb(al)) ** 2,
        source="QUADPACK book: x^a log(1/x) on [0, 1] = 1/(1 + a)^2",
    )


def _peak(al):
    s, s2 = 4.0**-al, 16.0**-al  # exact powers of two

    def exact(s=s):
        c = _arb(PEAK_CENTRE)
        return ((1 - c) / _arb(s)).atan() + (c / _arb(s)).atan()

    return Integral(
        id=f"book/peak/{int(al)}",
        expr=lambda x, M, s=s, s2=s2: (
            M.const(s) / ((x - M.const(PEAK_CENTRE)) * (x - M.const(PEAK_CENTRE)) + M.const(s2))
        ),
        a=0.0,
        b=1.0,
        exact=exact,
        numeric=al <= 6,
        source="QUADPACK book: 4^-a / ((x - pi/4)^2 + 16^-a) on [0, 1] = atan((1 - c) 4^a) + atan(c 4^a), c = pi/4",
    )


def _osc(al):
    w = 2.0**al
    return Integral(
        id=f"book/osc/{int(al)}",
        expr=lambda x, M, w=w: M.cos(M.const(w) * M.sin(x)),
        a=0.0,
        b=math.pi,
        exact=None,
        numeric=True,
        source="QUADPACK book: cos(2^a sin x) on [0, pi] = pi J0(2^a) (for the exact pi; the double b is numeric)",
        note="closed form pi J0(2^a) holds for b = pi exactly",
    )


def book_osc_closed(al):
    """pi J0(2^a): the family's closed form for the exact upper limit pi (test_problems checks the
    numerical reference for the double pi against it)."""
    return _pi() * _arb(2.0**al).bessel_j(0)


BOOK = (
    tuple(_xlog(a) for a in BOOK_XLOG_ALPHAS)
    + tuple(_peak(a) for a in BOOK_PEAK_ALPHAS)
    + tuple(_osc(a) for a in BOOK_OSC_ALPHAS)
)


# -- quadpack: the classic example of each routine

SQRT2 = math.sqrt(2.0)
QAWO_OMEGA = 10 * math.pi
QAWF_OMEGA = math.pi / 2


def _euler():
    from flint import arb

    return arb.const_euler()


def _qaws_exact():
    one = _arb(1)
    e = one.ci() * one.sin() + (_pi() / 2 - one.si()) * one.cos()  # int_0^inf e^-u / (1 + u^2) du
    return (e - 1) / 2


def _qaws_piece(t, M):
    # x = e^-u: int_0^1 log(x) / (1 + log(x)^2)^2 dx = -int_0^inf u e^-u / (1 + u^2)^2 du
    d = 1 + t * t
    return -t * M.exp(-t) / (d * d)


QAWS_T = 90.0  # the substitution's truncation; the tail is below e^-T / T^3 (4e-45)


def _log0(x, M):
    # log(x), and 0 at x = 0: QAWO's Clenshaw-Curtis rule evaluates f at the ends (the routine's
    # example defines f(0) = 0, as its test suites do)
    if M is MATH:
        return 0.0 if x == 0 else M.log(x)
    return M.log(x)


def _rsqrt0(x, M):
    # x^-1/2, and 0 at x = 0 (QAWF's first cycle evaluates f at 0)
    if M is MATH:
        return 0.0 if x == 0 else 1 / M.sqrt(x)
    return 1 / M.sqrt(x)


QUADPACK = (
    Integral(
        id="quadpack/qags",
        expr=lambda x, M: M.log(x) / M.sqrt(x),
        a=0.0,
        b=1.0,
        exact=lambda: _arb(-4),
        source="QAGS: log(x)/sqrt(x) on [0, 1] = -4",
    ),
    Integral(
        id="quadpack/qagp",
        expr=lambda x, M: x * x * x * M.log(M.abs((x * x - 1) * (x * x - 2))),
        a=0.0,
        b=3.0,
        points=(1.0, SQRT2),
        exact=lambda: 61 * _arb(2).log() + _arb(77) / 4 * _arb(7).log() - 27,
        source="QAGP: x^3 log|(x^2 - 1)(x^2 - 2)| on [0, 3], break points 1 and sqrt 2, = 61 log 2 + 77/4 log 7 - 27",
    ),
    Integral(
        id="quadpack/qagi",
        expr=lambda x, M: M.log(x) / (1 + 100 * x * x),
        a=0.0,
        b=INF,
        exact=lambda: -_pi() * _arb(10).log() / 20,
        source="QAGI: log(x)/(1 + 100 x^2) on [0, inf) = -pi log(10)/20",
    ),
    Integral(
        id="quadpack/qawo",
        expr=_log0,
        a=0.0,
        b=1.0,
        weight="sin",
        wvar=QAWO_OMEGA,
        exact=lambda: -(_euler() + _arb(QAWO_OMEGA).log() - _arb(QAWO_OMEGA).ci()) / _arb(QAWO_OMEGA),
        source="QAWO: log(x) sin(w x) on [0, 1], w = 10 pi, = -(gamma + log w - Ci(w))/w",
        tanhsinh=False,
    ),
    Integral(
        id="quadpack/qawf",
        expr=_rsqrt0,
        a=0.0,
        b=INF,
        weight="cos",
        wvar=QAWF_OMEGA,
        exact=lambda: (_pi() / (2 * _arb(QAWF_OMEGA))).sqrt(),
        source="QAWF: x^-1/2 cos(w x) on [0, inf), w = pi/2, = sqrt(pi/(2w)) (= 1 for the exact pi/2)",
        tanhsinh=False,
    ),
    Integral(
        id="quadpack/qaws",
        expr=lambda x, M: 1 / ((1 + M.log(x) * M.log(x)) * (1 + M.log(x) * M.log(x))),
        a=0.0,
        b=1.0,
        weight="alg-loga",
        wvar=(0.0, 0.0),
        exact=_qaws_exact,
        pieces=((_qaws_piece, 0.0, QAWS_T),),
        tail=math.exp(-QAWS_T) / QAWS_T**3,
        source="QAWS: log(x)/(1 + log(x)^2)^2 on [0, 1] = (Ci(1) sin 1 + (pi/2 - Si(1)) cos 1 - 1)/2",
        tanhsinh=False,
    ),
    Integral(
        id="quadpack/qawc",
        expr=lambda x, M: 1 / (5 * x * x * x + 6),
        a=-1.0,
        b=5.0,
        weight="cauchy",
        wvar=0.0,
        exact=lambda: (_arb(125) / 631).log() / 18,
        source="QAWC: the principal value of 1/(x (5x^3 + 6)) on [-1, 5] = log(125/631)/18",
        tanhsinh=False,
    ),
)


# -- scipy 1.18.1's tests


def _scipy_singular(x, M):
    if M is NP:
        return np.where((0 < x) & (x < 2.5), np.sin(x), np.where((2.5 <= x) & (x <= 5.0), np.exp(-x), 0.0))
    if 0 < x < 2.5:
        return M.sin(x)
    if 2.5 <= x <= 5.0:
        return M.exp(-x)
    return 0.0


SIN_OMEGA = 2.0**3.4
ALG_SHIFT = 2.0**-1.5
CAUCHY_S, CAUCHY_Q = 2.0**-0.4, 4.0**-0.4  # scipy's 2**-a and 4**-a, two doubles (q != s*s)


def _scipy_cauchy_exact():
    s, q = _arb(CAUCHY_S), _arb(CAUCHY_Q)
    r = q.sqrt()
    return s / (1 + q) * ((_arb(3) / 2).log() - ((16 + q) / (1 + q)).log() / 2 - ((4 / r).atan() + (1 / r).atan()) / r)


NORM_C = 1 / (2 * math.pi) ** 0.5  # test_tanhsinh's norm_pdf: 1/(2 pi)**0.5 * exp(-x**2/2)


def _norm_pdf(x, M):
    return M.const(NORM_C) * M.exp(-x * x / 2)


def _norm_exact(lo, hi):
    def exact(lo=lo, hi=hi):
        from flint import arb

        def cdf(v):
            if v == INF:
                return arb(1)
            if v == -INF:
                return arb(0)
            return (1 + (arb(v) / arb(2).sqrt()).erf()) / 2

        # the density's double constant c: the integral of c exp(-x^2/2) is c sqrt(2 pi) times Phi's
        return arb(NORM_C) * (2 * arb.pi()).sqrt() * (cdf(hi) - cdf(lo))

    return exact


SCIPY_QUAD = (
    Integral(
        id="scipy/quad/typical-bessel",
        expr=lambda x, M: M.cos(2 * x - M.const(1.8) * M.sin(x)) / M.const(math.pi),
        a=0.0,
        b=math.pi,
        numeric=True,
        source="test_quadpack TestQuad.test_typical: cos(n x - z sin x)/pi on [0, pi], n = 2, z = 1.8 (J_2(1.8))",
    ),
    Integral(
        id="scipy/quad/indefinite",
        expr=lambda x, M: -M.exp(-x) * M.log(x),
        a=0.0,
        b=INF,
        exact=_euler,
        source="test_quadpack TestQuad.test_indefinite: -exp(-x) log(x) on [0, inf) = Euler's gamma",
    ),
    Integral(
        id="scipy/quad/singular",
        expr=_scipy_singular,
        a=0.0,
        b=10.0,
        points=(2.5, 5.0),
        exact=lambda: 1 - _arb(2.5).cos() + _arb(-2.5).exp() - _arb(-5).exp(),
        source="test_quadpack TestQuad.test_singular: sin x on (0, 2.5), e^-x on [2.5, 5], 0 after; points 2.5, 5",
        tanhsinh=False,
    ),
    Integral(
        id="scipy/quad/sin-weighted-finite",
        expr=lambda x, M: M.exp(20 * (x - 1)),
        a=0.0,
        b=1.0,
        weight="sin",
        wvar=SIN_OMEGA,
        exact=lambda: (
            (20 * _arb(SIN_OMEGA).sin() - _arb(SIN_OMEGA) * _arb(SIN_OMEGA).cos() + _arb(SIN_OMEGA) * _arb(-20).exp())
            / (400 + _arb(SIN_OMEGA) ** 2)
        ),
        numeric=True,
        source="test_quadpack TestQuad.test_sine_weighted_finite: exp(20 (x - 1)) sin(w x) on [0, 1], w = 2^3.4",
        tanhsinh=False,
    ),
    Integral(
        id="scipy/quad/sin-weighted-infinite",
        expr=lambda x, M: M.exp(-x * 4.0),
        a=0.0,
        b=INF,
        weight="sin",
        wvar=3.0,
        exact=lambda: _arb(3) / 25,
        source="test_quadpack TestQuad.test_sine_weighted_infinite: exp(-4x) sin(3x) on [0, inf) = 3/25",
        tanhsinh=False,
    ),
    Integral(
        id="scipy/quad/cos-weighted-infinite",
        expr=lambda x, M: M.exp(x * 2.5),
        a=-INF,
        b=0.0,
        weight="cos",
        wvar=2.3,
        exact=lambda: _arb(2.5) / (_arb(2.5) ** 2 + _arb(2.3) ** 2),
        source="test_quadpack TestQuad.test_cosine_weighted_infinite: exp(2.5 x) cos(2.3 x) on (-inf, 0]",
        tanhsinh=False,
    ),
    Integral(
        id="scipy/quad/alg-weight",
        expr=lambda x, M: 1 / (1 + x + M.const(ALG_SHIFT)),
        a=-1.0,
        b=1.0,
        weight="alg",
        wvar=(-0.5, -0.5),
        exact=lambda: _pi() / ((1 + _arb(ALG_SHIFT)) ** 2 - 1).sqrt(),
        source="test_quadpack TestQuad.test_algebraic_log_weight: 1/(1 + x + 2^-1.5) with (1 + x)^-1/2 (1 - x)^-1/2",
        tanhsinh=False,
    ),
    Integral(
        id="scipy/quad/cauchy",
        expr=lambda x, M: M.const(CAUCHY_S) / ((x - 1) * (x - 1) + M.const(CAUCHY_Q)),
        a=0.0,
        b=5.0,
        weight="cauchy",
        wvar=2.0,
        exact=_scipy_cauchy_exact,
        source="test_quadpack TestQuad.test_cauchypv_weight: 2^-a/((x - 1)^2 + 4^-a) / (x - 2) on [0, 5], a = 0.4",
        tanhsinh=False,
    ),
    Integral(
        id="scipy/quad/sin",
        expr=lambda x, M: M.sin(x),
        a=0.0,
        b=5.0,
        exact=lambda: 1 - _arb(5).cos(),
        numeric=True,
        source="test_quadpack TestCtypesQuad.test_typical: sin on [0, 5]",
    ),
    Integral(
        id="scipy/quad/cos",
        expr=lambda x, M: M.cos(x),
        a=0.0,
        b=5.0,
        exact=lambda: _arb(5).sin(),
        numeric=True,
        source="test_quadpack TestCtypesQuad.test_typical: cos on [0, 5]",
    ),
    Integral(
        id="scipy/quad/tan",
        expr=lambda x, M: M.tan(x),
        a=0.0,
        b=1.0,
        exact=lambda: -(_arb(1).cos().log()),
        numeric=True,
        source="test_quadpack TestCtypesQuad.test_typical: tan on [0, 1]",
    ),
    Integral(
        id="scipy/quad/reversed-exp",
        expr=lambda x, M: 2 * M.exp(-3 * x),
        a=INF,
        b=0.0,
        exact=lambda: -_arb(2) / 3,
        source="test_quadpack TestQuad.test_b_less_than_a: p exp(-q x) from inf to 0, p = 2, q = 3",
    ),
    Integral(
        id="scipy/quad/reversed-gauss",
        expr=lambda x, M: M.exp(-x * x / 2 / 2) / M.sqrt(M.const(2.0 * 2)),
        a=INF,
        b=-INF,
        exact=lambda: -_pi().sqrt(),
        source="test_quadpack TestQuad.test_b_less_than_a_2: exp(-x^2/(2s))/sqrt(2s) from inf to -inf, s = 2",
    ),
    Integral(
        id="scipy/quad/reversed-alg",
        expr=lambda x, M: M.const(1.0) + 0 * x,
        a=1.0,
        b=0.0,
        weight="alg",
        wvar=(0.0, 0.0),
        exact=lambda: _arb(-1),
        source="test_quadpack TestQuad.test_b_less_than_a_3: 1 with weight 'alg' (0, 0) from 1 to 0",
        tanhsinh=False,
    ),
)

HALF_PI = math.pi / 2  # the double, below pi/2
GAUSS_CUT = 12.0  # where the substitutions' Gaussian tails are cut


def _f9_exact():
    # log(cos t) on [0, b], b the double below pi/2: -pi log(2)/2 minus the sliver on [b, pi/2],
    # int_0^d log(sin u) du = d log d - d + E, |E| <= d^3/15 (log(sin u / u) >= -u^2/5 on (0, 1))
    from flint import arb

    d = arb.pi() / 2 - arb(HALF_PI)
    sliver = d * d.log() - d
    v = -arb.pi() * arb(2).log() / 2 - sliver
    return v + arb(0, 1) * (d**3 / 15)


def _f10_exact():
    # sqrt(tan t) on [0, b]: pi sqrt(2)/2 minus int_0^d sqrt(cot u) du = 2 sqrt(d) - r,
    # 0 <= r <= d^(5/2)/5 (1 - sqrt(u cot u) <= u^2/2 on (0, 1))
    from flint import arb

    d = arb.pi() / 2 - arb(HALF_PI)
    v = arb.pi() * arb(2).sqrt() / 2 - 2 * d.sqrt()
    return v + arb(0, 1) * (d ** (arb(5) / 2) / 5)


def _f7_exact():
    from flint import arb

    return 2 * arb.pi().sqrt() * (arb(3) / 4).gamma() / (arb(1) / 4).gamma()


SCIPY_TANHSINH = (
    Integral("scipy/tanhsinh/f1", lambda t, M: t * M.log(1 + t), 0.0, 1.0, exact=lambda: _arb(1) / 4, numeric=True,
             source="test_tanhsinh f1 (Bailey, Jeyabalan, Li 2005): t log(1 + t) on [0, 1] = 1/4"),
    Integral("scipy/tanhsinh/f2", lambda t, M: t * t * M.atan(t), 0.0, 1.0,
             exact=lambda: (_pi() - 2 + 2 * _arb(2).log()) / 12, numeric=True,
             source="test_tanhsinh f2: t^2 atan(t) on [0, 1] = (pi - 2 + 2 log 2)/12"),
    Integral("scipy/tanhsinh/f3", lambda t, M: M.exp(t) * M.cos(t), 0.0, HALF_PI, numeric=True,
             source="test_tanhsinh f3: e^t cos(t) on [0, pi/2] = (e^(pi/2) - 1)/2 for the exact pi/2 (numeric here)"),
    Integral("scipy/tanhsinh/f4", lambda t, M: M.atan(M.sqrt(2 + t * t)) / ((1 + t * t) * M.sqrt(2 + t * t)), 0.0, 1.0,
             exact=lambda: 5 * _pi() ** 2 / 96, numeric=True,
             source="test_tanhsinh f4: atan(sqrt(2 + t^2)) / ((1 + t^2) sqrt(2 + t^2)) on [0, 1] = 5 pi^2/96"),
    Integral("scipy/tanhsinh/f5", lambda t, M: M.sqrt(t) * M.log(t), 0.0, 1.0, exact=lambda: _arb(-4) / 9,
             source="test_tanhsinh f5: sqrt(t) log(t) on [0, 1] = -4/9"),
    Integral("scipy/tanhsinh/f6", lambda t, M: M.sqrt(1 - t * t), 0.0, 1.0, exact=lambda: _pi() / 4, numeric=True,
             source="test_tanhsinh f6: sqrt(1 - t^2) on [0, 1] = pi/4"),
    Integral("scipy/tanhsinh/f7", lambda t, M: M.sqrt(t) / M.sqrt(1 - t * t), 0.0, 1.0, exact=_f7_exact,
             source="test_tanhsinh f7: sqrt(t)/sqrt(1 - t^2) on [0, 1] = 2 sqrt(pi) Gamma(3/4)/Gamma(1/4)"),
    Integral("scipy/tanhsinh/f8", lambda t, M: M.log(t) * M.log(t), 0.0, 1.0, exact=lambda: _arb(2),
             source="test_tanhsinh f8: log(t)^2 on [0, 1] = 2"),
    Integral("scipy/tanhsinh/f9", lambda t, M: M.log(M.cos(t)), 0.0, HALF_PI, exact=_f9_exact,
             source="test_tanhsinh f9: log(cos t) on [0, pi/2] = -pi log(2)/2, less the sliver past the double pi/2"),
    Integral("scipy/tanhsinh/f10", lambda t, M: M.sqrt(M.tan(t)), 0.0, HALF_PI, exact=_f10_exact,
             source="test_tanhsinh f10: sqrt(tan t) on [0, pi/2] = pi sqrt(2)/2, less the sliver (1.6e-8)"),
    Integral("scipy/tanhsinh/f11", lambda t, M: 1 / (1 + t * t), 0.0, INF, exact=lambda: _pi() / 2,
             source="test_tanhsinh f11: 1/(1 + t^2) on [0, inf) = pi/2"),
    Integral("scipy/tanhsinh/f12", lambda t, M: M.exp(-t) / M.sqrt(t), 0.0, INF, exact=lambda: _pi().sqrt(),
             pieces=((lambda u, M: 2 * M.exp(-u * u), 0.0, GAUSS_CUT),), tail=math.exp(-GAUSS_CUT**2) / GAUSS_CUT,
             source="test_tanhsinh f12: e^-t / sqrt(t) on [0, inf) = sqrt(pi) (t = u^2: 2 e^(-u^2), cut at u = 12)"),
    Integral("scipy/tanhsinh/f13", lambda t, M: M.exp(-t * t / 2), 0.0, INF, exact=lambda: (_pi() / 2).sqrt(),
             source="test_tanhsinh f13: e^(-t^2/2) on [0, inf) = sqrt(pi/2)"),
    Integral("scipy/tanhsinh/f14", lambda t, M: M.exp(-t) * M.cos(t), 0.0, INF, exact=lambda: _arb(1) / 2,
             source="test_tanhsinh f14: e^-t cos(t) on [0, inf) = 1/2"),
    Integral("scipy/tanhsinh/f15", lambda t, M: M.sin(t) / t, 0.0, INF, exact=lambda: _pi() / 2,
             source="test_tanhsinh f15: sin(t)/t on [0, inf) = pi/2"),
    Integral("scipy/tanhsinh/norm/0,inf", _norm_pdf, 0.0, INF, exact=_norm_exact(0.0, INF),
             source="test_tanhsinh test_integral_transforms: the normal density on [0, inf)"),
    Integral("scipy/tanhsinh/norm/-inf,0", _norm_pdf, -INF, 0.0, exact=_norm_exact(-INF, 0.0),
             source="test_tanhsinh test_integral_transforms: the normal density on (-inf, 0]"),
    Integral("scipy/tanhsinh/norm/-inf,inf", _norm_pdf, -INF, INF, exact=_norm_exact(-INF, INF),
             source="test_tanhsinh test_integral_transforms: the normal density on (-inf, inf)"),
    Integral("scipy/tanhsinh/norm/inf,-inf", _norm_pdf, INF, -INF, exact=lambda: -_norm_exact(-INF, INF)(),
             source="test_tanhsinh test_integral_transforms: the normal density from inf to -inf"),
    Integral("scipy/tanhsinh/norm/1,-1", _norm_pdf, 1.0, -1.0, exact=lambda: -_norm_exact(-1.0, 1.0)(),
             source="test_tanhsinh test_integral_transforms: the normal density from 1 to -1"),
)  # fmt: skip


# -- hard integrals (docs/design/integrate.md §7.1, §7.4)

THIRD = 1.0 / 3.0
NARROW_Q = 1e-6
SIN_RECIP_A = 0.01


def _interior(al, pts):
    def exact(al=al):
        c, p = _arb(THIRD), _arb(al) + 1
        return (c**p + (1 - c) ** p) / p

    tag = "" if not pts else "/points"
    return Integral(
        id=f"hard/interior/{al:+.2f}{tag}",
        expr=lambda x, M, al=al: M.pow(M.abs(x - M.const(THIRD)), al),
        a=0.0,
        b=1.0,
        points=(THIRD,) if pts else (),
        exact=exact,
        source="|x - 1/3|^a on [0, 1] = (c^(a+1) + (1 - c)^(a+1))/(a + 1), c the double 1/3"
        + ("; break point at c" if pts else ""),
        tanhsinh=not pts,
    )


def _sin_recip_exact():
    from flint import arb

    def F(x):
        x = arb(x)
        return x * (1 / x).sin() - (1 / x).ci()

    return F(1.0) - F(SIN_RECIP_A)


HARD = (
    Integral("hard/rsqrt", lambda x, M: 1 / M.sqrt(x), 0.0, 1.0, exact=lambda: _arb(2),
             pieces=((lambda t, M: M.const(2.0) + 0 * t, 0.0, 1.0),),
             source="x^-1/2 on [0, 1] = 2 (x = t^2: 2 on [0, 1])"),
    Integral("hard/pow-0.9", lambda x, M: M.pow(x, -0.9), 0.0, 1.0, exact=lambda: 1 / (1 + _arb(-0.9)),
             pieces=((lambda t, M: 1 / (1 + M.const(-0.9)) + 0 * t, 0.0, 1.0),),
             source="x^-0.9 on [0, 1] = 1/(1 + a) for the double a = -0.9 (x = t^(1/(1+a)): a constant)"),
    Integral("hard/rsqrt-log", lambda x, M: M.log(x) / M.sqrt(x), 0.0, 1.0, exact=lambda: _arb(-4),
             source="x^-1/2 log(x) on [0, 1] = -4 (the QAGS example with tanhsinh's records)", tanhsinh=False),
    Integral("hard/cos100", lambda x, M: M.cos(100 * x), 0.0, 1.0, exact=lambda: _arb(100).sin() / 100, numeric=True,
             source="cos(100 x) on [0, 1] = sin(100)/100"),
    Integral("hard/sin-recip", lambda x, M: M.sin(1 / x), SIN_RECIP_A, 1.0, exact=_sin_recip_exact, numeric=True,
             source="sin(1/x) on [0.01, 1] = [x sin(1/x) - Ci(1/x)] from 0.01 to 1"),
    Integral("hard/gauss", lambda x, M: M.exp(-x * x), -INF, INF, exact=lambda: _pi().sqrt(),
             pieces=((lambda x, M: M.exp(-x * x), -GAUSS_CUT, GAUSS_CUT),), tail=math.exp(-GAUSS_CUT**2) / GAUSS_CUT,
             source="e^(-x^2) on (-inf, inf) = sqrt(pi) (cut at |x| = 12: the rest is below e^-144/12)"),
    Integral("hard/narrow", lambda x, M: 1 / (x * x + M.const(NARROW_Q)), -1.0, 1.0,
             exact=lambda: 2 * (1 / _arb(NARROW_Q).sqrt()).atan() / _arb(NARROW_Q).sqrt(), numeric=True,
             source="1/(x^2 + q) on [-1, 1], q = 1e-6 (the double), = 2 atan(1/sqrt q)/sqrt q"),
) + tuple(_interior(al, pts) for al in (-0.5, 0.5, 1.0) for pts in (False, True))  # fmt: skip

INTEGRALS = BOOK + QUADPACK + SCIPY_QUAD + SCIPY_TANHSINH + HARD
BY_ID = {p.id: p for p in INTEGRALS}
if len(BY_ID) != len(INTEGRALS):
    raise RuntimeError("duplicate integral ids")


def get(pid):
    return BY_ID[pid]


# ---- batches: a family as one call of tanhsinh with its parameter as an `args` array


@dataclasses.dataclass(frozen=True)
class Batch:
    """A family of INTEGRALS as one batch: expr(x, p, M) with the parameter p per element, the
    members' ids f"{family}/{tag(p)}"."""

    family: str
    expr: Callable
    a: float
    b: float
    params: tuple
    tag: Callable

    def f_np(self, x, p):
        return self.expr(np.asarray(x, dtype=np.float64), np.asarray(p, dtype=np.float64), NP)

    def ids(self):
        return [f"{self.family}/{self.tag(p)}" for p in self.params]


BATCHES = (
    Batch("book/xlog", lambda x, al, M: M.pow(x, al) * M.log(1 / x), 0.0, 1.0, BOOK_XLOG_ALPHAS, lambda p: f"{p:+.1f}"),
    Batch("book/peak", lambda x, al, M: M.pow(4.0, -al) / ((x - M.const(PEAK_CENTRE)) * (x - M.const(PEAK_CENTRE))
                                                           + M.pow(16.0, -al)),
          0.0, 1.0, BOOK_PEAK_ALPHAS, lambda p: f"{int(p)}"),
    Batch("book/osc", lambda x, al, M: M.cos(M.pow(2.0, al) * M.sin(x)), 0.0, math.pi, BOOK_OSC_ALPHAS,
          lambda p: f"{int(p)}"),
)  # fmt: skip


# ---- multiple integrals (dblquad, tplquad, nquad, cubature)


@dataclasses.dataclass(frozen=True)
class Multiple:
    """A multiple integral. func(*coords) in scipy's argument order (the innermost variable first:
    dblquad's func(y, x)); limits as scipy's dblquad/tplquad take them (numbers or functions of
    the outer variables); exact() the closed form as an arb. kind: 'dblquad' or 'tplquad'.
    box: (a, b) per variable in nquad's / cubature's order when the region is a box (cubature's
    records), else None."""

    id: str
    kind: str
    func: Callable
    a: float
    b: float
    limits: tuple
    exact: Callable
    box: tuple | None = None
    source: str = ""


def _erf1():
    return _arb(1).erf()


MULTI = (
    Multiple("multi/dbl-linear", "dblquad", lambda y, x: x + y, 1.0, 2.0, (lambda x: x, lambda x: 2 * x),
             exact=lambda: _arb(5) / 6 * (_arb(2) ** 3 - 1),
             source="test_quadpack TestDblquad.test_double_integral: x + y, x in [1, 2], y in [x, 2x]"),
    Multiple("multi/dbl-args", "dblquad", lambda y, x: y + x + 1 + 2, 1.0, 2.0, (lambda x: x, lambda x: 2 * x),
             exact=lambda: _arb(35) / 6 + _arb(9) / 2,
             source="test_quadpack TestDblquad.test_double_integral2: x0 + x1 + t0 + t1 (t = 1, 2), y in [x, 2x]"),
    Multiple("multi/dbl-const-limits", "dblquad", lambda y, x: y + x + 1 + 2, 1.0, 2.0, (1.0, 2.0),
             exact=lambda: _arb(6), box=((1.0, 2.0), (1.0, 2.0)),
             source="test_quadpack TestDblquad.test_double_integral3: x0 + x1 + 3 on [1, 2]^2"),
    Multiple("multi/dbl-gauss-square", "dblquad", lambda y, x: np.exp(-(x * x + y * y)), 0.0, 1.0, (0.0, 1.0),
             exact=lambda: (_pi().sqrt() / 2 * _erf1()) ** 2, box=((0.0, 1.0), (0.0, 1.0)),
             source="exp(-(x^2 + y^2)) on the unit square (docs/design/integrate.md §3.2's probe)"),
    Multiple("multi/dbl-gauss-plane", "dblquad", lambda y, x: np.exp(-(x * x + y * y)), -INF, INF, (-INF, INF),
             exact=_pi, box=((-INF, INF), (-INF, INF)), source="exp(-(x^2 + y^2)) on the plane = pi"),
    Multiple("multi/tpl-linear", "tplquad", lambda z, y, x: (x + y + z) * 2.0, 1.0, 2.0,
             (lambda x: x, lambda x: 2 * x, lambda x, y: x - y, lambda x, y: x + y),
             exact=lambda: 2 * _arb(8) / 3 * (_arb(2) ** 4 - 1),
             source="test_quadpack TestTplquad.test_triple_integral: (x + y + z) t, t = 2"),
    Multiple("multi/tpl-gauss-cube", "tplquad", lambda z, y, x: np.exp(-(x * x + y * y + z * z)), 0.0, 1.0,
             (0.0, 1.0, 0.0, 1.0), exact=lambda: (_pi().sqrt() / 2 * _erf1()) ** 3,
             box=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), source="exp(-(x^2 + y^2 + z^2)) on the unit cube"),
)  # fmt: skip


# ---- vector-valued integrands (quad_vec)


@dataclasses.dataclass(frozen=True)
class Vector:
    id: str
    func: Callable  # x (float) -> numpy array
    a: float
    b: float
    exact: Callable  # -> list of arb
    source: str = ""


VECTOR = (
    Vector("vector/exp-sin-square", lambda x: np.array([math.exp(x), math.sin(x), x * x]), 0.0, 1.0,
           exact=lambda: [_arb(1).exp() - 1, 1 - _arb(1).cos(), _arb(1) / 3],
           source="[e^x, sin x, x^2] on [0, 1]"),
    Vector("vector/gauss-moments", lambda x: np.array([math.exp(-x * x), x * x * math.exp(-x * x)]), -INF, INF,
           exact=lambda: [_pi().sqrt(), _pi().sqrt() / 2], source="[e^(-x^2), x^2 e^(-x^2)] on (-inf, inf)"),
)  # fmt: skip


# ---- series (nsum): sum of f(a + k step) for a <= a + k step <= b


@dataclasses.dataclass(frozen=True)
class Series:
    """sum_k f(a + k step) over a <= a + k step <= b. expr(k, M) as Integral's; exact() the closed
    form (an arb), or terms(K) -> (partial sum of the first K terms, a rigorous bound on the rest)
    for a direct reference."""

    id: str
    expr: Callable
    a: float
    b: float
    step: float = 1.0
    exact: Callable | None = None
    source: str = ""

    def f_np(self, k):
        return self.expr(np.asarray(k, dtype=np.float64), NP)


def _zeta(p):
    return lambda p=p: _arb(p).zeta()


def _two_sided_gauss():
    # sum over all integers of e^-k^2 = 1 + 2 sum_{k>=1} e^-k^2; the rest after K = 40 is below
    # 2 e^-1681 / (1 - e^-81)
    from flint import arb

    s = arb(1)
    for k in range(1, 41):
        s += 2 * arb(-(k * k)).exp()
    return s + arb(0, 1) * 4 * arb(-1681).exp()


SERIES = (
    Series("series/zeta/2", lambda k, M: 1 / (k * k), 1.0, INF, exact=_zeta(2.0),
           source="test_tanhsinh TestNSum f1: 1/k^2 from 1 = zeta(2)"),
    Series("series/zeta/3", lambda k, M: 1 / (k * k * k), 1.0, INF, exact=_zeta(3.0), source="1/k^3 from 1 = zeta(3)"),
    Series("series/zeta/1.5", lambda k, M: M.pow(k, -1.5), 1.0, INF, exact=_zeta(1.5),
           source="k^-1.5 from 1 = zeta(1.5) (slow: the integral bound carries it)"),
    Series("series/harmonic/2/1000", lambda k, M: 1 / (k * k), 1.0, 1000.0,
           exact=lambda: _arb(2).zeta() - _arb(2).zeta(_arb(1001)),
           source="test_tanhsinh TestNSum f3: the generalized harmonic number H(1000, 2)"),
    Series("series/half-step", lambda x, M: 1 / (x * x), 1.0, INF, step=0.5, exact=lambda: 4 * (_arb(2).zeta() - 1),
           source="1/x^2 at x = 1, 1.5, 2, ... = 4 (zeta(2) - 1)"),
    Series("series/lorentz", lambda k, M: 1 / (k * k + 1), 0.0, INF,
           exact=lambda: (1 + _pi() / _pi().tanh()) / 2, source="1/(k^2 + 1) from 0 = (1 + pi coth pi)/2"),
    Series("series/geometric", lambda k, M: M.exp(-k / 10), 0.0, INF, exact=lambda: 1 / (1 - (_arb(-1) / 10).exp()),
           source="e^(-k/10) from 0 = 1/(1 - e^-0.1)"),
    Series("series/gauss-two-sided", lambda k, M: M.exp(-k * k), -INF, INF, exact=_two_sided_gauss,
           source="e^(-k^2) over all integers (theta_3(0, e^-1))"),
    Series("series/zeta-left", lambda k, M: 1 / (k * k), -INF, -1.0, exact=_zeta(2.0),
           source="1/k^2 from -inf to -1 = zeta(2)"),
)  # fmt: skip
SERIES_BY_ID = {s.id: s for s in SERIES}


# ---- samples (the sample rules trapezoid, simpson, romb and the cumulative ones)


def samples():
    """{id: (y, x or None, dx)}: numpy samples for the sample rules, built from fixed formulas.
    Their references (each rule's formula computed exactly) are tests/integrate/ref/samples.json
    (tools/integrate_ref.py samples, milestone C3)."""
    out = {}
    for n in (2, 3, 4, 5, 17, 33, 4097):
        x = np.linspace(0.0, 1.0, n)
        out[f"samples/exp/uniform/{n}"] = (np.exp(x), None, 1.0 / (n - 1))
    for n in (3, 4, 10, 101):
        x = (np.arange(n, dtype=np.float64) / (n - 1)) ** 2
        out[f"samples/exp/squares/{n}"] = (np.exp(x), x, 1.0)
    g = np.linspace(0.0, 2.0, 9)
    out["samples/2d/rows"] = (np.vstack([np.sin(g), np.cos(g), g * g]), g, 1.0)
    return out
