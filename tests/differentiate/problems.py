"""The test collection of sd.differentiate (docs/design/integrate.md §7.1).

  ELEMENTARY   eleven elementary functions, each at nine points of its domain, written once for a
               namespace M (NP: numpy arrays, as derivative calls the function; ARB: Arb balls for
               the references), with the first and second derivatives in closed form (Arb)
  EDGES        the cases docs/design/integrate.md §7.4 found: log at 1e-3 and sqrt at 1e-6 (the
               default initial step 0.5 leaves the domain: status -3), x^3 at 0 (a zero
               derivative: the default atol never stops it, status -2), exp at 30 (a large value),
               with the options that make them work (a small initial_step, step_direction)
  MGH          the Jacobian of the residuals F and the Hessian of f = sum F_i^2 of the
               More-Garbow-Hillstrom problems with factor 1 (tests/optimize/mgh.py) at x0; the
               references are forward-mode jets over Arb (tests/differentiate/jets.py)

A point is a double; a reference is the exact derivative of the function at that double.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np


class _Np:
    name = "numpy"
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    tan = staticmethod(np.tan)
    atan = staticmethod(np.arctan)
    tanh = staticmethod(np.tanh)


class _Arb:
    name = "arb"

    @staticmethod
    def _a(x):
        from flint import arb

        return x if isinstance(x, arb) else arb(x)

    def exp(self, x):
        return self._a(x).exp()

    def log(self, x):
        return self._a(x).log()

    def sqrt(self, x):
        return self._a(x).sqrt()

    def sin(self, x):
        return self._a(x).sin()

    def cos(self, x):
        return self._a(x).cos()

    def tan(self, x):
        return self._a(x).tan()

    def atan(self, x):
        return self._a(x).atan()

    def tanh(self, x):
        return self._a(x).tanh()


NP = _Np()
ARB = _Arb()


@dataclasses.dataclass(frozen=True)
class Function:
    """f(x, M) with f'(x) and f''(x) in closed form (d1, d2: arb -> arb) and nine points."""

    name: str
    expr: Callable
    d1: Callable
    d2: Callable
    points: tuple

    def f_np(self, x):
        return self.expr(np.asarray(x, dtype=np.float64), NP)


def _sec2(x):
    return 1 / (x.cos() ** 2)


ELEMENTARY = (
    Function("exp", lambda x, M: M.exp(x), lambda x: x.exp(), lambda x: x.exp(),
             (-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0)),
    Function("log", lambda x, M: M.log(x), lambda x: 1 / x, lambda x: -1 / (x * x),
             (0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, 100.0, 1e4)),
    Function("sin", lambda x, M: M.sin(x), lambda x: x.cos(), lambda x: -x.sin(),
             (-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0)),
    Function("cos", lambda x, M: M.cos(x), lambda x: -x.sin(), lambda x: -x.cos(),
             (-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0)),
    Function("tan", lambda x, M: M.tan(x), _sec2, lambda x: 2 * x.tan() * _sec2(x),
             (-0.9, -0.5, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.2)),
    Function("sqrt", lambda x, M: M.sqrt(x), lambda x: 1 / (2 * x.sqrt()), lambda x: -1 / (4 * x * x.sqrt()),
             (0.6, 0.75, 1.0, 1.5, 2.0, 4.0, 10.0, 100.0, 1e4)),
    Function("atan", lambda x, M: M.atan(x), lambda x: 1 / (1 + x * x), lambda x: -2 * x / (1 + x * x) ** 2,
             (-5.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 10.0)),
    Function("tanh", lambda x, M: M.tanh(x), lambda x: 1 - x.tanh() ** 2,
             lambda x: -2 * x.tanh() * (1 - x.tanh() ** 2), (-3.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0)),
    Function("cube", lambda x, M: x * x * x, lambda x: 3 * x * x, lambda x: 6 * x,
             (-2.0, -1.0, -0.5, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)),
    Function("recip", lambda x, M: 1 / x, lambda x: -1 / (x * x), lambda x: 2 / (x * x * x),
             (-3.0, -1.0, 0.6, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0)),
    Function("gauss", lambda x, M: M.exp(-x * x), lambda x: -2 * x * (-x * x).exp(),
             lambda x: (4 * x * x - 2) * (-x * x).exp(), (-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0)),
)  # fmt: skip
BY_NAME = {f.name: f for f in ELEMENTARY}


@dataclasses.dataclass(frozen=True)
class Edge:
    """A hard point of one of the ELEMENTARY functions, with derivative()'s options."""

    id: str
    function: str
    x: float
    options: dict
    note: str = ""


EDGES = (
    Edge("log/1e-3/default", "log", 1e-3, {}, "x - 0.5 < 0: NaN in the first iteration (status -3)"),
    Edge("log/1e-3/step", "log", 1e-3, {"initial_step": 1e-4}, "a step below x"),
    Edge("log/1e-3/right", "log", 1e-3, {"step_direction": 1}, "the one-sided stencil to the right"),
    Edge("sqrt/1e-6/default", "sqrt", 1e-6, {}, "x - 0.5 < 0: NaN (status -3)"),
    Edge("sqrt/1e-6/right", "sqrt", 1e-6, {"step_direction": 1, "initial_step": 1e-7}, "one-sided, small step"),
    Edge("cube/0/default", "cube", 0.0, {}, "f'(0) = 0: the default atol never stops it (status -2)"),
    Edge("cube/0/atol", "cube", 0.0, {"atol": 1e-12}, "an absolute tolerance stops it"),
    Edge("exp/30/default", "exp", 30.0, {}, "a large value"),
    Edge("tan/1.5/default", "tan", 1.5, {}, "x + 0.5 is past pi/2"),
    Edge("tan/1.5/step", "tan", 1.5, {"initial_step": 1e-3}, "a step inside the period"),
    Edge("exp/1/order-2", "exp", 1.0, {"order": 2}, "the lowest order"),
    Edge("exp/1/order-12", "exp", 1.0, {"order": 12}, "a high order"),
    Edge("exp/1/factor-1.5", "exp", 1.0, {"step_factor": 1.5}, "a smaller step factor"),
    Edge("exp/1/left", "exp", 1.0, {"step_direction": -1}, "the one-sided stencil to the left"),
    Edge("exp/1/factor-3", "exp", 1.0, {"step_factor": 3.0}, "a step factor above 2"),
    Edge("exp/1/factor-0.5", "exp", 1.0, {"step_factor": 0.5}, "a step factor below 1: the steps grow (status -1)"),
    Edge("exp/1/rtol", "exp", 1.0, {"rtol": 1e-12}, "a tighter relative tolerance"),
)
# scipy records past the one-point calls (calculus review): a 2-D batch of each function's nine
# points (3 x 3) and a parametric function with args broadcast against x
ND_SHAPE = (3, 3)
ARGS = (("args/exp-scaled", lambda x, c: np.exp(c * x), 1.0, (np.array([0.5, 1.0, 2.0, -1.0]),)),)
EDGES_BY_ID = {e.id: e for e in EDGES}


def mgh_cases():
    """The More-Garbow-Hillstrom cases with factor 1 (tests/optimize/mgh.py), the Jacobian's
    collection; the Hessian's is the ones with n <= 10 (each Hessian entry is a derivative of a
    derivative, and the large n make the record slow)."""
    from tests.optimize import mgh

    return [c for c in mgh.CASES if c.factor == 1.0]


HESSIAN_MAX_N = 10


def mgh_f_np(c):
    """f = sum F_i^2 of an MGH case for scipy's hessian: x is (n, ...) (the first axis the
    variables, scipy's contract), the result (...)."""
    from tests.optimize.ad import FLOAT

    def f(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty(x.shape[1:])
        for idx in np.ndindex(*x.shape[1:]):
            F = c.fun([float(v) for v in x[(slice(None), *idx)]], M=FLOAT)
            out[idx] = math.fsum(float(r) * float(r) for r in F)
        return out

    return f


def mgh_F_np(c):
    """F of an MGH case for scipy's jacobian: x is (n, ...), the result (m, ...)."""
    from tests.optimize.ad import FLOAT

    def F(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty((c.m, *x.shape[1:]))
        for idx in np.ndindex(*x.shape[1:]):
            out[(slice(None), *idx)] = [float(r) for r in c.fun([float(v) for v in x[(slice(None), *idx)]], M=FLOAT)]
        return out

    return F
