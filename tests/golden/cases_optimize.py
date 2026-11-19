"""Golden cases for sd.optimize (tools/golden.py has the contract).

least_squares on the 110 Moré-Garbow-Hillstrom cases (tests/optimize/mgh.py) with 'trf'
(analytic, '2-point' and '3-point' Jacobians; boxed around x0) and 'lm' (analytic, '2-point'),
plus diff_step, max_nfev, tolerances and the documented errors; approx_fprime and check_grad
at the problems' starting points. A result is stored whole (every public field). New
functions of later milestones need cases here when they land.
"""

import math

import numpy as np

from tests.optimize import mgh


def cases(L):
    sd = L.sd
    O = sd.optimize

    def fns(c):
        F = lambda x: sd.asarray(c.fun(x.tolist()))  # noqa: E731
        J = lambda x: sd.asarray(c.jac(x.tolist()))  # noqa: E731
        return F, J

    def box(c):
        x0 = c.x0()
        return [v - 0.5 * (abs(v) + 1.0) for v in x0], [v + 0.5 * (abs(v) + 1.0) for v in x0]

    def lsq(c, method, jac, bounded=False, **kw):
        def run():
            F, J = fns(c)
            if bounded:
                kw["bounds"] = box(c)
            with np.errstate(all="ignore"):
                return O.least_squares(F, c.x0(), jac=J if jac == "analytic" else jac, method=method, **kw)

        return run

    for c in mgh.CASES:
        for jac in ("analytic", "2-point", "3-point"):
            yield f"least_squares {c.id} trf {jac}", lsq(c, "trf", jac)
        for jac in ("analytic", "2-point"):
            yield f"least_squares {c.id} trf {jac} box", lsq(c, "trf", jac, bounded=True)
            if c.m >= c.n:
                yield f"least_squares {c.id} lm {jac}", lsq(c, "lm", jac)
    c = mgh.BY_ID["mgh01-n2-m2-x1"]
    for method in ("trf", "lm"):
        for ds in (1e-3, 1e-10):
            yield f"least_squares {method} diff_step={ds}", lsq(c, method, "2-point", diff_step=ds)
        yield f"least_squares {method} max_nfev=5", lsq(c, method, "2-point", max_nfev=5)
        yield f"least_squares {method} loose", lsq(c, method, "analytic", ftol=1e-3, xtol=1e-3, gtol=1e-3)
        yield f"least_squares {method} tol zero", lsq(c, method, "analytic", ftol=0.0, xtol=0.0, gtol=1e-15)
    rosen = lambda x: sd.asarray([10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]])  # noqa: E731
    yield "least_squares trf bounds scalar", lambda: O.least_squares(rosen, [-1.2, 1.0], bounds=(-2.0, 0.5))
    yield "least_squares trf bounds scalar inside", lambda: O.least_squares(rosen, [-1.2, 0.3], bounds=(-2.0, 0.5))
    yield "least_squares trf bounds inf", lambda: O.least_squares(rosen, [-1.2, 1.0], bounds=(-math.inf, math.inf))
    yield "least_squares trf x0 on bound", lambda: O.least_squares(rosen, [0.5, 0.3], bounds=(-2.0, 0.5))
    yield "least_squares m < n trf", lambda: O.least_squares(lambda x: sd.asarray([x[0] + x[1] - 1.0]), [0.0, 0.0])
    yield "least_squares nonfinite start", lambda: O.least_squares(lambda x: x * math.nan, [1.0, 2.0])
    yield "least_squares sd array x0", lambda: O.least_squares(rosen, sd.asarray([-1.2, 1.0]))
    yield "least_squares repr", lambda: repr(O.least_squares(rosen, [-1.2, 1.0]))
    yield "err least_squares lm bounds", lambda: O.least_squares(rosen, [-1.2, 1.0], method="lm", bounds=(-2.0, 2.0))
    yield (
        "err least_squares lm m < n",
        lambda: O.least_squares(lambda x: sd.asarray([x[0] + x[1]]), [0.0, 0.0], method="lm"),
    )
    yield "err least_squares method", lambda: O.least_squares(rosen, [-1.2, 1.0], method="dogbox")
    yield "err least_squares x0 outside", lambda: O.least_squares(rosen, [3.0, 1.0], bounds=(-2.0, 2.0))
    yield "err least_squares jac str", lambda: O.least_squares(rosen, [-1.2, 1.0], jac="cs")
    yield "err least_squares ftol", lambda: O.least_squares(rosen, [-1.2, 1.0], ftol=-1.0)
    yield "err least_squares max_nfev float", lambda: O.least_squares(rosen, [-1.2, 1.0], max_nfev=5.0)
    yield "err least_squares fun shape", lambda: O.least_squares(lambda x: sd.zeros((2, 2)), [-1.2, 1.0])
    yield "err least_squares x0 2d", lambda: O.least_squares(rosen, [[1.0, 2.0]])

    starts = [c for c in mgh.CASES if c.factor == 1.0]
    for c in starts:
        f = lambda x, c=c: c.f(x.tolist())  # noqa: E731
        F, J = fns(c)
        g = lambda x, c=c: sd.asarray(  # noqa: E731
            (2.0 * np.array(c.jac(x.tolist()), dtype=float).reshape(c.m, c.n).T @ np.array(c.fun(x.tolist()))).tolist()
        )
        yield f"approx_fprime {c.id} f", lambda c=c, f=f: O.approx_fprime(c.x0(), f)
        yield f"approx_fprime {c.id} F", lambda c=c, F=F: O.approx_fprime(c.x0(), F)
        yield f"approx_fprime {c.id} F eps", lambda c=c, F=F: O.approx_fprime(c.x0(), F, epsilon=1e-6)
        yield f"check_grad {c.id}", lambda c=c, f=f, g=g: O.check_grad(f, g, c.x0())
        yield (
            f"check_grad {c.id} random",
            lambda c=c, f=f, g=g: O.check_grad(f, g, c.x0(), direction="random", rng=sd.random.default_rng(3)),
        )
    yield "approx_fprime scalar x", lambda: O.approx_fprime(1.5, lambda x: x * x)
    yield (
        "approx_fprime epsilon array",
        lambda: O.approx_fprime([1.0, 2.0], lambda x: x[0] * x[1], epsilon=sd.asarray([1e-4, 1e-6])),
    )
    yield "err approx_fprime epsilon", lambda: O.approx_fprime([1.0], lambda x: x[0], epsilon=-1.0)
    yield "err check_grad direction", lambda: O.check_grad(lambda x: x[0], lambda x: x, [1.0], direction="x")
    yield "err check_grad rng", lambda: O.check_grad(lambda x: x[0], lambda x: x, [1.0], direction="random")
    yield "OptimizeResult fields", lambda: O.OptimizeResult(x=sd.asarray([1.0]), converged=True, nfev=3)
    yield "err OptimizeResult positional", lambda: O.OptimizeResult(1)
