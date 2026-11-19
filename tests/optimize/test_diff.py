"""sd.optimize.approx_fprime and check_grad (docs/api/optimize.md §2, docs/design/optimize.md §3.1):
scipy's forward differences to the bit with scipy's epsilon (tests/optimize/data/scipy_approx_fprime.json),
the accurate 5-point differences against the analytic derivatives of the Moré-Garbow-Hillstrom
cases, check_grad, the shapes and the errors."""

import json
import math
import pathlib

import numpy as np
import pytest

import stardust as sd

from . import mgh

REF = json.loads((pathlib.Path(__file__).parent / "data" / "scipy_approx_fprime.json").read_text())
CASES = [c for c in mgh.CASES if c.factor == 1.0]
SQRT_EPS = 2.0**-26  # scipy's default epsilon, sqrt(u)


def fns(c):
    f = lambda x: c.f(x.tolist())  # noqa: E731
    F = lambda x: sd.asarray(c.fun(x.tolist()))  # noqa: E731
    J = np.array(c.jac(c.x0()), dtype=float).reshape(c.m, c.n)
    g = 2.0 * J.T @ np.array(c.fun(c.x0()))
    return f, F, J, g


def grad_of(c):
    def g(x):
        xl = x.tolist()
        J = np.array(c.jac(xl), dtype=float).reshape(c.m, c.n)
        return sd.asarray((2.0 * J.T @ np.array(c.fun(xl))).tolist())

    return g


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.mark.parametrize("c", CASES, ids=str)
def test_forward_differences_are_scipys_to_the_bit(c):
    f, F, _, _ = fns(c)
    rec = REF[c.id]["approx_fprime"]
    x0 = c.x0()
    assert sd.optimize.approx_fprime(x0, f, epsilon=SQRT_EPS).tolist() == rec["grad"]
    assert sd.reshape(sd.optimize.approx_fprime(x0, F, epsilon=SQRT_EPS), (-1,)).tolist() == rec["jac"]
    # n steps: the same
    assert sd.optimize.approx_fprime(x0, f, epsilon=[SQRT_EPS] * c.n).tolist() == rec["grad"]


@pytest.mark.parametrize("c", CASES, ids=str)
def test_accurate_differences_against_the_analytic_derivatives(c):
    f, F, J, g = fns(c)
    x0 = c.x0()
    # worst over the 47 cases: 1.2e-11 (Jacobian), 6.4e-11 (gradient), normwise
    assert rel(sd.optimize.approx_fprime(x0, F).tolist(), J) <= 5e-11
    assert rel(sd.optimize.approx_fprime(x0, f).tolist(), g) <= 2e-10
    # scipy's forward differences are good to about sqrt(u)
    assert rel(sd.optimize.approx_fprime(x0, F, epsilon=SQRT_EPS).tolist(), J) <= 2e-6


def test_check_grad_is_scipys():
    for c in CASES:
        f, _, _, _ = fns(c)
        ref = REF[c.id]["approx_fprime"]["check_grad"]
        v = sd.optimize.check_grad(f, grad_of(c), c.x0(), epsilon=SQRT_EPS)
        assert abs(v - ref) <= 1e-12 * ref, c.id
        # the accurate differences: far smaller
        acc = sd.optimize.check_grad(f, grad_of(c), c.x0())
        gnorm = float(np.linalg.norm(fns(c)[3]))
        assert acc <= 2e-10 * gnorm, c.id


def test_check_grad_random_direction():
    c = mgh.BY_ID["mgh19-n11-m65-x1"]
    f = lambda x: c.f(x.tolist())  # noqa: E731
    a = sd.optimize.check_grad(f, grad_of(c), c.x0(), direction="random", rng=sd.random.default_rng(3))
    b = sd.optimize.check_grad(f, grad_of(c), c.x0(), direction="random", rng=sd.random.default_rng(3))
    assert a == b
    # the definition: |g . v - d/dw f(x0 + w v)| with v standard normal from rng
    v = sd.stats.Normal().sample((c.n,), rng=sd.random.default_rng(3))
    x0 = sd.asarray(c.x0())
    d = sd.optimize.approx_fprime([0.0], lambda w: f(x0 + float(w[0]) * v))
    assert a == abs(float(sd.linalg.vecdot(grad_of(c)(x0), v)) - float(d[0]))
    g = float(np.linalg.norm(fns(c)[3]))
    assert a <= 1e-9 * g
    # a wrong gradient shows
    bad = lambda x: grad_of(c)(x) * 1.001  # noqa: E731
    assert sd.optimize.check_grad(f, bad, c.x0(), direction="random", rng=sd.random.default_rng(3)) > 1e-4 * g


def test_shapes():
    x0 = [1.0, 2.0, 3.0]
    g = sd.optimize.approx_fprime(x0, lambda x: float(sd.sum(x * x)))
    assert g.shape == (3,) and g.dtype == sd.float64
    assert rel(g.tolist(), [2.0, 4.0, 6.0]) <= 1e-12
    J = sd.optimize.approx_fprime(x0, lambda x: x * x)
    assert J.shape == (3, 3)
    assert rel(J.tolist(), np.diag([2.0, 4.0, 6.0])) <= 1e-12
    # one value is still a vector function: (1, n)
    assert sd.optimize.approx_fprime(x0, lambda x: sd.sum(x * x, keepdims=True)).shape == (1, 3)
    # numpy results and Python sequences are taken; f gets a new float64 sd.ndarray
    seen = []

    def f(x):
        seen.append(x)
        x[0] = 99.0  # changing it does not change the differences
        return np.array([1.0, 2.0])

    assert sd.optimize.approx_fprime(np.array(x0), f).shape == (2, 3)
    assert all(isinstance(x, sd.ndarray) and x.dtype == sd.float64 for x in seen)
    assert sd.optimize.approx_fprime([1, 2], lambda x: [x[0] * x[1]]).shape == (1, 2)


def test_one_sided_where_f_is_not_finite():
    # log(x) at x = 1e-3: the central stencil's x - 2h is fine; at x near a pole on one side
    f = lambda x: sd.where(x > 1.0, math.nan, x * x)  # noqa: E731
    J = sd.optimize.approx_fprime([1.0], f)
    assert abs(float(J[0, 0]) - 2.0) <= 1e-9


def test_errors():
    ok = lambda x: x  # noqa: E731
    cases = [
        (lambda: sd.optimize.approx_fprime([1.0], 3), TypeError, "approx_fprime(): 'f' must be callable, not int"),
        (
            lambda: sd.optimize.approx_fprime([[1.0]], ok),
            ValueError,
            "approx_fprime(): 'xk' must be 1-dimensional, not shape (1, 1)",
        ),
        (lambda: sd.optimize.approx_fprime([], ok), ValueError, "approx_fprime(): 'xk' is empty (no variables)"),
        (lambda: sd.optimize.approx_fprime([math.inf], ok), ValueError, "approx_fprime(): 'xk' must be finite"),
        (
            lambda: sd.optimize.approx_fprime([1j], ok),
            TypeError,
            "approx_fprime(): 'xk' must be real, not a complex128 array",
        ),
        (
            lambda: sd.optimize.approx_fprime(1.0, ok),
            ValueError,
            "approx_fprime(): 'xk' must be 1-dimensional, not a scalar",
        ),
        (
            lambda: sd.optimize.approx_fprime([1.0], ok, epsilon=0.0),
            ValueError,
            "approx_fprime(): 'epsilon' must be finite and > 0",
        ),
        (
            lambda: sd.optimize.approx_fprime([1.0], ok, epsilon=[1e-8, 1e-8]),
            ValueError,
            "approx_fprime(): 'epsilon' must have 1 values like 'xk', not 2",
        ),
        (
            lambda: sd.optimize.approx_fprime([1.0], ok, epsilon="x"),
            TypeError,
            "approx_fprime(): 'epsilon' must be a number or a sequence of numbers, not str",
        ),
        (
            lambda: sd.optimize.approx_fprime([1.0], lambda x: sd.reshape(x, (1, 1))),
            ValueError,
            "approx_fprime(): 'f' must return a real number or a 1-dimensional array, not shape (1, 1)",
        ),
        (
            lambda: sd.optimize.approx_fprime([1.0], lambda x: "a"),
            TypeError,
            "approx_fprime(): 'f' must return a real number or a 1-dimensional array, not str",
        ),
        (
            lambda: sd.optimize.approx_fprime([1.0], lambda x: x * 1j),
            TypeError,
            "approx_fprime(): 'f' must return real values, not a complex128 array",
        ),
        (
            lambda: sd.optimize.approx_fprime([1.0], lambda x: math.nan),
            ValueError,
            "approx_fprime(): 'f' is not finite at the point",
        ),
        (
            lambda: sd.optimize.approx_fprime([1.0, 2.0], lambda x: sd.ones((2 if x[0] == 1.0 else 3,))),
            ValueError,
            "approx_fprime(): 'f' must return 2 values as at the first point, not shape (3,)",
        ),
        (
            lambda: sd.optimize.check_grad(ok, ok, [1.0], direction="x"),
            ValueError,
            "check_grad(): 'direction' must be 'all' or 'random', not 'x'",
        ),
        (
            lambda: sd.optimize.check_grad(ok, ok, [1.0], direction="random"),
            TypeError,
            "check_grad(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType",
        ),
        (
            lambda: sd.optimize.check_grad(ok, ok, [1.0], rng=sd.random.default_rng(1)),
            ValueError,
            "check_grad(): 'rng' is only used when direction='random'",
        ),
        (
            lambda: sd.optimize.check_grad(lambda x: x, ok, [1.0, 2.0]),
            ValueError,
            "check_grad(): 'func' must return a real number, not shape (2,)",
        ),
        (
            lambda: sd.optimize.check_grad(lambda x: 1.0, lambda x: [1.0], [1.0, 2.0]),
            ValueError,
            "check_grad(): 'grad' must return 2 values (one per variable), not shape (1,)",
        ),
        (
            lambda: sd.optimize.check_grad(lambda x: 1.0, None, [1.0]),
            TypeError,
            "check_grad(): 'grad' must be callable, not NoneType",
        ),
    ]
    for fn, typ, msg in cases:
        with pytest.raises(typ) as e:
            fn()
        assert str(e.value) == msg


def test_user_errors_pass_through():
    def f(x):
        raise KeyError("mine")

    with pytest.raises(KeyError, match="mine"):
        sd.optimize.approx_fprime([1.0], f)
