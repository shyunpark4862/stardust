"""quad(vectorized=True) (docs/api/integrate.md §2, docs/design/integrate.md C127): the points of one
rule application in one call of func as a 1-D float64 sd.ndarray, the same points in the same
order as the scalar calls, and a result equal to vectorized=False's in every field, bit for bit;
the returned values read and checked as tanhsinh's and nsum's."""

import math
import warnings

import numpy as np
import pytest

import stardust as sd

from . import problems as P

I = sd.integrate  # noqa: E741

# the points of one rule application: QK21, QK15 (an infinite half line, the weighted Kronrod
# rule), QK15 on the real line (both halves), the 25-point Clenshaw-Curtis rules
RULE_SIZES = {15, 21, 25, 30}

FIELDS = (
    "integral",
    "error",
    "converged",
    "status",
    "message",
    "nfev",
    "n_intervals",
    "intervals",
    "interval_integrals",
    "interval_errors",
    "n_cycles",
    "cycle_integrals",
    "cycle_errors",
    "cycle_status",
)


def bits(v):
    if isinstance(v, sd.ndarray):
        a = np.asarray(v)
        return (a.dtype.str, a.shape, a.tobytes())
    if isinstance(v, float):
        return np.float64(v).tobytes()
    if isinstance(v, complex):
        return np.complex128(v).tobytes()
    return v


def same_result(r, s):
    """Every field of two quad results, bit for bit (the parts of a complex one too)."""
    for name in FIELDS:
        has = hasattr(r, name)
        assert has == hasattr(s, name), name
        if has:
            assert bits(getattr(r, name)) == bits(getattr(s, name)), name
    if hasattr(r, "real"):
        same_result(r.real, s.real)
        same_result(r.imag, s.imag)


def both(g, a, b, value=float, dtype=sd.float64, **kw):
    """quad with the scalar g and with g applied point by point to the array: the two results, the
    scalar points, the array calls' points (one list per call)."""
    scalar, calls = [], []

    def fs(x):
        scalar.append(x)
        return g(x)

    def fv(xs):
        assert isinstance(xs, sd.ndarray) and xs.dtype == sd.float64 and xs.ndim == 1
        pts = xs.tolist()
        calls.append(pts)
        return sd.asarray([value(g(t)) for t in pts], dtype=dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = I.quad(fs, a, b, **kw)
        s = I.quad(fv, a, b, vectorized=True, **kw)
    return r, s, scalar, calls


def check_same(g, a, b, **kw):
    r, s, scalar, calls = both(g, a, b, **kw)
    same_result(r, s)
    flat = [t for c in calls for t in c]
    assert np.array_equal(np.array(flat).view(np.int64), np.array(scalar, dtype=np.float64).view(np.int64))
    assert all(len(c) in RULE_SIZES for c in calls)  # one call per rule application
    if not kw.get("complex_func"):
        assert s.nfev == len(flat)  # nfev counts points, not calls
    return r, s, calls


@pytest.mark.parametrize("p", P.INTEGRALS, ids=lambda p: p.id)
def test_every_problem_is_the_scalar_result(p):
    # QAGS, QAGI (half lines and the real line), QAGP, QAWO, QAWF, QAWS, QAWC; converged, limit
    # reached (1) and divergent (5) among them
    kw = {}
    if p.points:
        kw["points"] = list(p.points)
    if p.weight is not None:
        kw.update(weight=p.weight, wvar=p.wvar)
    check_same(p.f, p.a, p.b, **kw)


@pytest.mark.parametrize(
    ("g", "a", "b", "kw", "status"),
    [
        (lambda x: abs(math.sin(200.0 * x)), 0.0, 10.0, {}, 1),  # the limit of subintervals
        (lambda x: 1.0 / x if x else math.inf, 0.0, 1.0, {"limit": 10000}, 3),  # bad behaviour
        (lambda x: math.nan if x > 0.5 else 1.0, 0.0, 1.0, {}, 2),  # NaN from func: roundoff
        (lambda x: math.inf, 0.0, 1.0, {}, -3),  # inf from func: not finite
        (lambda x: math.exp(x), 1.0, 0.0, {}, 0),  # reversed limits
        (lambda x: math.exp(-x * x), -math.inf, math.inf, {}, 0),
        (lambda x: math.exp(x), -math.inf, 0.0, {}, 0),
        (lambda x: math.exp(x), -math.inf, 0.0, {"weight": "sin", "wvar": 3.0}, 0),  # QAWF on (-inf, b]
        (lambda x: math.exp(-x), 0.0, math.inf, {"weight": "cos", "wvar": 3.0}, 0),
        (lambda x: math.exp(-x * x), 0.0, 1.0, {"weight": "alg-loga", "wvar": (-0.5, 0.5)}, 0),
        (lambda x: 1 / (1 + x * x), -1.0, 2.0, {"weight": "cauchy", "wvar": 0.5}, 0),
        (lambda x: math.exp(x), 0.0, 1.0, {"weight": "sin", "wvar": 1.0, "limit": 1}, 1),
    ],
)
def test_statuses_and_routes(g, a, b, kw, status):
    r, _, _ = check_same(g, a, b, **kw)
    assert r.status == status


def test_empty_range_does_not_call():
    r, s, scalar, calls = both(lambda x: 1.0, 1.0, 1.0)
    same_result(r, s)
    assert scalar == [] and calls == []


@pytest.mark.parametrize(
    ("a", "b"), [(0.0, 2.0), (0.0, math.inf), (-math.inf, math.inf)], ids=["finite", "half", "line"]
)
def test_complex_func(a, b):
    def g(x):
        return complex(math.exp(-x * x), math.sin(3.0 * x) * math.exp(-abs(x)))

    r, s, scalar, calls = both(g, a, b, value=complex, dtype=sd.complex128, complex_func=True)
    same_result(r, s)
    # two quads (the real part, then the imaginary part), each calling func: scipy's two passes
    assert sum(len(c) for c in calls) == len(scalar) == r.nfev
    # a real array is the real part (the imaginary part 0), as a float from the scalar func
    r2, s2, _, _ = both(lambda x: math.exp(-x * x), a, b, complex_func=True)
    same_result(r2, s2)


def test_values_accepted_like_tanhsinh():
    # numpy arrays and lists are read as tanhsinh's f values; int arrays become float64
    def f(x):
        return np.exp(-(np.asarray(x) ** 2))

    want = I.quad(lambda x: sd.asarray(f(x)), 0.0, 1.0, vectorized=True)
    for conv in (sd.asarray, lambda v: v.tolist(), lambda v: v):
        got = I.quad(lambda x, conv=conv: conv(f(x)), 0.0, 1.0, vectorized=True)
        same_result(got, want)
    r = I.quad(lambda x: sd.ones(x.shape[0], dtype=sd.int64), 0.0, 2.0, vectorized=True)
    assert r.integral == 2.0


def test_errors_are_tanhsinhs():
    shape = r"^quad\(\): 'func' must return shape \(21,\) like its argument, not "
    with pytest.raises(ValueError, match=shape + r"\(20,\)$"):
        I.quad(lambda x: x[:20], 0.0, 1.0, vectorized=True)
    with pytest.raises(ValueError, match=shape + "a scalar$"):
        I.quad(lambda x: 1.0, 0.0, 1.0, vectorized=True)
    with pytest.raises(TypeError, match=r"^quad\(\): 'func' must return real values, not a complex128 array$"):
        I.quad(lambda x: x * 1j, 0.0, 1.0, vectorized=True)
    with pytest.raises(TypeError, match=r"^quad\(\): 'func' must return real values, not a bool array$"):
        I.quad(lambda x: x > 0.5, 0.0, 1.0, vectorized=True)
    with pytest.raises(TypeError, match=r"^quad\(\): 'func' must return an sd.ndarray or a sequence of numbers"):
        I.quad(lambda x: None, 0.0, 1.0, vectorized=True)
    with pytest.raises(ValueError, match=shape + r"\(21, 1\)$"):
        I.quad(lambda x: sd.reshape(x, (21, 1)), 0.0, 1.0, vectorized=True)
    with pytest.raises(ValueError, match=r"^quad\(\): 'func' must return shape \(21,\) like its argument"):
        I.quad(lambda x: x[:3], 0.0, 1.0, vectorized=True, complex_func=True)
    with pytest.raises(TypeError, match=r"^quad\(\): 'func' must return numbers, not a bool array$"):
        I.quad(lambda x: x > 0.5, 0.0, 1.0, vectorized=True, complex_func=True)
    # the same check of its value as the other bool options
    with pytest.raises(TypeError, match=r"^quad\(\): 'vectorized' must be True or False, not int$"):
        I.quad(math.exp, 0.0, 1.0, vectorized=1)
    with pytest.raises(TypeError, match=r"^quad\(\): 'complex_func' must be True or False, not int$"):
        I.quad(math.exp, 0.0, 1.0, complex_func=1)


def test_func_raising_propagates():
    class Boom(Exception):
        pass

    n = [0]

    def f(x):
        n[0] += 1
        if n[0] == 3:
            raise Boom("third call")
        return 1.0 / ((x - 0.3) * (x - 0.3) + 1e-4)

    with pytest.raises(Boom, match="third call"):
        I.quad(f, 0.0, 1.0, vectorized=True)
    assert n[0] == 3


def test_one_call_per_rule():
    n = []
    r = I.quad(lambda x: n.append(x.shape[0]) or sd.exp(-x * x), -math.inf, math.inf, vectorized=True)
    assert n and set(n) == {30} and sum(n) == r.nfev == 270
