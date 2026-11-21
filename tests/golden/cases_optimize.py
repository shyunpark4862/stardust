"""Golden cases for sd.optimize (tools/golden.py has the contract).

Every function of sd.optimize.__all__ on the test problems of tests/optimize:
  - least_squares on the 110 Moré-Garbow-Hillstrom cases (tests/optimize/mgh.py) with 'trf'
    (analytic, '2-point' and '3-point' Jacobians; boxed around x0) and 'lm' (analytic, '2-point'),
    plus diff_step, max_nfev, tolerances and the documented errors; approx_fprime and check_grad
    at the problems' starting points;
  - minimize on the MGH problems from their starting points with every method (analytic and
    difference gradients, a box around x0, Hessians for the Newton methods where n is small),
    on Rosenbrock, the maximum-entropy problem and L-BFGS-B's driver1 (tests/optimize/functions.py)
    with the options of each method, and the errors;
  - root on the HYBRD/HYBRJ driver systems (tests/optimize/systems.py) with 'hybr' and 'lm';
  - root_scalar, find_root and bracket_root on the 204 scalar root cases, minimize_scalar,
    find_minimum and bracket_minimum on the 57 scalar minimization cases, both also batched;
  - lsq_linear and nnls on the dense problems of functions.py and larger random ones.
A result is stored whole (every public field); summary() has cases of its own (O119).
"""

import math

import numpy as np

from tests.optimize import functions as fn
from tests.optimize import mgh, systems


def encode(x, L, enc):
    """An OptimizeResult: the fields its method gives, and the names of those it does not
    (reading one raises AttributeError; "OptimizeResult absent field" stores that error)."""
    if type(x) is not L.sd.optimize.OptimizeResult:
        return None
    attrs, absent = {}, []
    for name in sorted(n for n in dir(x) if not n.startswith("_") and n != "summary"):  # fields, not the method
        try:
            attrs[name] = enc(getattr(x, name))
        except AttributeError:
            absent.append(name)
    return {"obj": f"{type(x).__module__}.{type(x).__qualname__}", "attrs": attrs, "absent": absent}


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
    for x0 in (3e-6, 1e-5, 1e-4):  # the start above 2^-36 of the column grows (C112)
        yield f"approx_fprime exp x0={x0:g}", lambda x0=x0: O.approx_fprime([x0], lambda x: sd.exp(x))
    yield (
        "approx_fprime epsilon array",
        lambda: O.approx_fprime([1.0, 2.0], lambda x: x[0] * x[1], epsilon=sd.asarray([1e-4, 1e-6])),
    )
    yield "err approx_fprime epsilon", lambda: O.approx_fprime([1.0], lambda x: x[0], epsilon=-1.0)
    yield "err check_grad direction", lambda: O.check_grad(lambda x: x[0], lambda x: x, [1.0], direction="x")
    yield "err check_grad rng", lambda: O.check_grad(lambda x: x[0], lambda x: x, [1.0], direction="random")
    yield "OptimizeResult fields", lambda: O.OptimizeResult(x=sd.asarray([1.0]), converged=True, nfev=3)
    yield "err OptimizeResult positional", lambda: O.OptimizeResult(1)
    yield "OptimizeResult absent field", lambda: O.root_scalar(lambda x: x - 1.0, bracket=(0.0, 3.0)).hess_inv
    yield from more_cases(L)
    yield from summaries(L)


# ---------------------------------------------------------------- minimize, root, scalar, elementwise, linear


def _scalar(g):
    """A scalar function of the problems' kind as the scalar solvers call it (a float in, a float out)."""

    def f(x):
        with np.errstate(all="ignore"):
            return float(g(np.float64(x)))

    return f


def _vec(sd, g):
    """A problem's scalar function element by element on an sd.ndarray (the elementwise solvers)."""

    def f(x):
        with np.errstate(all="ignore"):
            return sd.asarray([float(g(np.float64(t))) for t in x.tolist()])

    return f


def more_cases(L):
    sd = L.sd
    O = sd.optimize
    yield from minimize_cases(sd, O)
    yield from root_cases(sd, O)
    yield from scalar_cases(sd, O)
    yield from elementwise_cases(sd, O)
    yield from linear_cases(L, sd, O)


def minimize_cases(sd, O):
    def mgh_fns(c):
        F, J, f, g = mgh.as_numpy(c)
        fx = lambda x: float(f(x.tolist()))  # noqa: E731
        gx = lambda x: sd.asarray(g(x.tolist()).tolist())  # noqa: E731
        hx = lambda x: sd.asarray(mgh.hessian(c, x.tolist()).tolist())  # noqa: E731
        return fx, gx, hx

    def box(x0):
        return [v - 0.5 * (abs(v) + 1.0) for v in x0], [v + 0.5 * (abs(v) + 1.0) for v in x0]

    def run(fun, x0, **kw):
        def go():
            with np.errstate(all="ignore"):
                return O.minimize(fun, x0, **kw)

        return go

    starts = [c for c in mgh.CASES if c.factor == 1.0]
    for c in starts:
        f, g, h = mgh_fns(c)
        x0 = c.x0()
        yield f"minimize {c.id} bfgs", run(f, x0, method="bfgs", jac=g)
        yield f"minimize {c.id} bfgs 2-point", run(f, x0, method="bfgs", jac="2-point")
        yield f"minimize {c.id} l-bfgs-b", run(f, x0, method="l-bfgs-b", jac=g)
        yield f"minimize {c.id} l-bfgs-b box", run(f, x0, method="l-bfgs-b", jac=g, bounds=box(x0))
        yield f"minimize {c.id} l-bfgs-b eps box", run(f, x0, method="l-bfgs-b", bounds=box(x0))
        yield f"minimize {c.id} nelder-mead", run(f, x0, method="nelder-mead", options={"max_nfev": 400})
        if c.n <= 6:
            yield f"minimize {c.id} newton-cg hess", run(f, x0, method="newton-cg", jac=g, hess=h)
            yield f"minimize {c.id} newton-cg", run(f, x0, method="newton-cg", jac=g)
            yield f"minimize {c.id} trust-ncg", run(f, x0, method="trust-ncg", jac=g, hess=h)
            yield f"minimize {c.id} trust-exact", run(f, x0, method="trust-exact", jac=g, hess=h)
    rosen = lambda x: fn.rosen(x.tolist())  # noqa: E731
    rosen_g = lambda x: fn.rosen_der(x.tolist())  # noqa: E731
    rosen_h = lambda x: fn.rosen_hess(x.tolist())  # noqa: E731
    rosen_hp = lambda x, p: fn.rosen_hess_prod(x.tolist(), p.tolist())  # noqa: E731
    rosen_fg = lambda x: (rosen(x), rosen_g(x))  # noqa: E731
    for n, x0 in fn.ROSEN_X0.items():
        lo, hi = [v - 0.5 for v in x0], [v + 2.0 for v in x0]
        for method, kw in (
            ("bfgs", dict(jac=rosen_g)),
            ("bfgs", dict(jac=rosen_fg, fun_is_fg=True)),
            ("bfgs", dict()),
            ("bfgs", dict(jac="3-point")),
            ("bfgs", dict(jac=rosen_g, options={"norm": 2.0, "xrtol": 1e-6})),
            ("bfgs", dict(jac=rosen_g, options={"max_iter": 3})),
            ("bfgs", dict(jac=rosen_g, options={"c1": 0.3, "c2": 0.4})),
            ("bfgs", dict(jac=rosen_g, tol=1e-12)),
            ("l-bfgs-b", dict(jac=rosen_g)),
            ("l-bfgs-b", dict(jac=rosen_g, options={"maxcor": 3})),
            ("l-bfgs-b", dict(jac=rosen_g, options={"max_ls": 2})),
            ("l-bfgs-b", dict(jac=rosen_g, options={"max_nfev": 7})),
            ("l-bfgs-b", dict(jac=rosen_g, bounds=(lo, hi))),
            ("l-bfgs-b", dict(jac="2-point", bounds=(lo, hi))),
            ("l-bfgs-b", dict(jac=rosen_g, bounds=(x0, x0))),
            ("l-bfgs-b", dict(bounds=([lo[0]] + x0[1:], [hi[0]] + x0[1:]))),
            ("nelder-mead", dict()),
            ("nelder-mead", dict(options={"adaptive": True})),
            ("nelder-mead", dict(options={"max_iter": 30})),
            ("nelder-mead", dict(bounds=(lo, hi), tol=1e-8)),
            ("newton-cg", dict(jac=rosen_g, hess=rosen_h)),
            ("newton-cg", dict(jac=rosen_g, hessp=rosen_hp)),
            ("newton-cg", dict(jac=rosen_g)),
            ("newton-cg", dict(jac=rosen_g, hess="2-point")),
            ("trust-ncg", dict(jac=rosen_g, hess=rosen_h)),
            ("trust-ncg", dict(jac=rosen_g, hessp=rosen_hp)),
            ("trust-exact", dict(jac=rosen_g, hess=rosen_h)),
            ("trust-exact", dict(jac=rosen_g, hess=rosen_h, options={"max_iter": 2})),
        ):
            kw = dict(kw)
            fun = rosen
            if kw.pop("fun_is_fg", False):
                fun, kw["jac"] = rosen_fg, True
            tag = " ".join(f"{k}={'fn' if callable(v) else v}" for k, v in sorted(kw.items()))
            yield f"minimize rosen{n} {method} {tag}", run(fun, x0, method=method, **kw)
        yield f"minimize rosen{n} default", run(rosen, x0)
        yield f"minimize rosen{n} default bounds", run(rosen, x0, bounds=(lo, hi))
    maxent = lambda x: fn.maxent(x.tolist())  # noqa: E731
    maxent_g = lambda x: fn.maxent_der(x.tolist())  # noqa: E731
    for method in ("bfgs", "l-bfgs-b", "newton-cg"):
        yield f"minimize maxent {method}", run(maxent, fn.MAXENT_X0, method=method, jac=maxent_g)
    yield "minimize maxent nelder-mead", run(maxent, fn.MAXENT_X0, method="nelder-mead")
    d1 = lambda x: fn.lbfgsb_driver1(x.tolist())  # noqa: E731
    d1_g = lambda x: fn.lbfgsb_driver1_der(x.tolist())  # noqa: E731
    d1_bounds = ([b[0] for b in fn.LBFGSB_DRIVER1_BOUNDS], [b[1] for b in fn.LBFGSB_DRIVER1_BOUNDS])
    yield "minimize driver1 l-bfgs-b", run(d1, fn.LBFGSB_DRIVER1_X0, method="l-bfgs-b", jac=d1_g, bounds=d1_bounds)
    yield (
        "minimize driver1 l-bfgs-b maxcor 5",
        run(d1, fn.LBFGSB_DRIVER1_X0, method="l-bfgs-b", jac=d1_g, bounds=d1_bounds, options={"maxcor": 5}),
    )
    yield (
        "minimize driver1 l-bfgs-b 2-point",
        run(d1, fn.LBFGSB_DRIVER1_X0, method="l-bfgs-b", jac="2-point", bounds=d1_bounds),
    )
    yield "minimize driver1 nelder-mead", run(d1, fn.LBFGSB_DRIVER1_X0, method="nelder-mead", bounds=d1_bounds)
    yield "err minimize x0 scalar", run(lambda x: (x[0] - 2.0) ** 2, 5.0, method="bfgs")
    yield "minimize sd x0", run(rosen, sd.asarray([-1.2, 1.0]), method="bfgs", jac=rosen_g)
    yield "minimize repr", lambda: repr(O.minimize(rosen, [-1.2, 1.0], jac=rosen_g))
    yield "minimize nonfinite at start", run(lambda x: math.nan, [1.0, 2.0], method="bfgs")
    yield "minimize inf later", run(lambda x: math.inf if x[0] > 0.5 else (x[0] - 1.0) ** 2, [0.0], method="l-bfgs-b")
    q = lambda x: (x[0] - 1.0) ** 2 + 2.0 * (x[1] + 0.5) ** 2  # noqa: E731
    errs = {
        "method case": dict(method="BFGS"),
        "method unknown": dict(method="cg"),
        "option unknown": dict(method="bfgs", options={"xatol": 1.0}),
        "option scipy name": dict(method="l-bfgs-b", options={"maxiter": 3}),
        "options list": dict(method="bfgs", options=[1]),
        "tol and option": dict(method="bfgs", tol=1e-6, options={"gtol": 1e-6}),
        "bounds bfgs": dict(method="bfgs", bounds=(-1.0, 1.0)),
        "bounds pairs": dict(method="l-bfgs-b", bounds=[(0, 1), (0, 1), (0, 1)]),
        "bounds reversed": dict(method="l-bfgs-b", bounds=(1.0, -1.0)),
        "x0 outside": dict(method="l-bfgs-b", bounds=(2.0, 3.0)),
        "jac nelder-mead": dict(method="nelder-mead", jac=lambda x: [0.0, 0.0]),
        "jac cs": dict(method="bfgs", jac="cs"),
        "newton-cg no jac": dict(method="newton-cg"),
        "trust-ncg no hess": dict(method="trust-ncg", jac=lambda x: [0.0, 0.0]),
        "trust-exact hessp": dict(method="trust-exact", jac=lambda x: [0.0, 0.0], hessp=lambda x, p: p),
        "hess bfgs": dict(method="bfgs", hess=lambda x: [[1.0, 0.0], [0.0, 1.0]]),
        "max_iter float": dict(method="bfgs", options={"max_iter": 3.0}),
        "adaptive int": dict(method="nelder-mead", options={"adaptive": 1}),
    }
    for nm, kw in errs.items():
        yield f"err minimize {nm}", run(q, [0.0, 0.0], **kw)
    yield "err minimize x0 2d", run(q, [[0.0, 0.0]])
    yield "err minimize x0 empty", run(q, [])
    yield "err minimize fun not callable", run(3, [0.0, 0.0])
    yield "err minimize fun vector", run(lambda x: [1.0, 2.0], [0.0, 0.0])


def root_cases(sd, O):
    def fns(c):
        F, J = systems.as_numpy(c)
        f = lambda x: F(np.asarray(x.tolist()))  # noqa: E731
        j = lambda x: J(np.asarray(x.tolist()))  # noqa: E731
        return f, j

    def run(f, x0, **kw):
        def go():
            with np.errstate(all="ignore"):
                return O.root(f, x0, **kw)

        return go

    for c in systems.CASES:
        f, j = fns(c)
        for method in ("hybr", "lm"):
            yield f"root {c.id} {method}", run(f, c.x0(), method=method)
            yield f"root {c.id} {method} jac", run(f, c.x0(), method=method, jac=j)
    c = systems.CASES[0]
    f, j = fns(c)
    for method, opts in (
        ("hybr", {"max_nfev": 5}),
        ("hybr", {"factor": 1.0}),
        ("hybr", {"band": (1, 1)}),
        ("hybr", {"diag": [1.0, 2.0]}),
        ("lm", {"max_nfev": 5}),
        ("lm", {"ftol": 1e-4, "xtol": 1e-4}),
        ("lm", {"factor": 10.0}),
    ):
        yield f"root {c.id} {method} {sorted(opts)}", run(f, c.x0(), method=method, options=opts)
    for method in ("hybr", "lm"):
        yield f"root tol {method}", run(f, c.x0(), method=method, tol=1e-4)
        yield f"root jac=True {method}", run(lambda x: (f(x), j(x)), c.x0(), method=method, jac=True)
        yield f"root one variable {method}", run(lambda x: [x[0] ** 3 - 8.0], [1.0], method=method)
        yield f"root at the root {method}", run(lambda x: [x[0] - 2.0, x[1]], [2.0, 0.0], method=method)
        yield f"root nonfinite {method}", run(lambda x: [math.nan, x[1]], [1.0, 0.0], method=method)
    yield "root default method", run(f, c.x0())
    yield "root repr", lambda: repr(O.root(f, c.x0()))
    yield "err root method case", run(f, c.x0(), method="HYBR")
    yield "err root method unknown", run(f, c.x0(), method="broyden1")
    yield "err root x0 empty", run(f, [])
    yield "err root x0 nan", run(f, [0.0, math.nan])
    yield "err root x0 complex", run(f, [1j, 0.0])
    yield "err root jac False", run(f, c.x0(), jac=False)
    yield "err root jac str", run(f, c.x0(), jac="2-point")
    yield "err root option scipy name", run(f, c.x0(), method="lm", options={"maxiter": 3})
    yield "err root fun not callable", run(3, [0.0, 0.0])
    yield "err root fun shape", run(lambda x: [x[0]], [0.0, 0.0])


BRACKETING = ("brentq", "toms748", "bisect")


def scalar_cases(sd, O):
    def root_run(c, m):
        def go():
            kw = {"method": m.split("/")[0]}
            if m in BRACKETING:
                kw["bracket"] = c.bracket
            else:
                kw["x0"] = c.x0
                if m in ("newton", "halley"):
                    kw["fprime"] = _scalar(c.fprime)
                if m == "halley":
                    kw["fprime2"] = _scalar(c.fprime2)
            return O.root_scalar(_scalar(c.f), **kw)

        return go

    for c in fn.ROOT_CASES:
        methods = list(BRACKETING)
        if c.x0 is not None:
            methods += ["secant", "newton/fd"] + (["newton"] if c.fprime else []) + (["halley"] if c.fprime2 else [])
        for m in methods:
            yield f"root_scalar {c.id} {m}", root_run(c, m)
    yield from root_scalar_options(O, fn.ROOTS_BY_ID["original.01"])
    for c in fn.MIN_CASES:
        f = _scalar(c.f)
        yield f"minimize_scalar {c.id} brent", lambda c=c, f=f: O.minimize_scalar(f, bracket=c.bracket, method="brent")
        yield (
            f"minimize_scalar {c.id} golden",
            lambda c=c, f=f: O.minimize_scalar(f, bracket=c.bracket, method="golden"),
        )
        yield (
            f"minimize_scalar {c.id} bounded",
            lambda c=c, f=f: O.minimize_scalar(f, bounds=c.bounds, method="bounded"),
        )
    q = lambda x: (x - 1.5) ** 2 - 0.8  # noqa: E731
    yield "minimize_scalar default", lambda: O.minimize_scalar(q)
    yield "minimize_scalar default bounds", lambda: O.minimize_scalar(q, bounds=(0.0, 1.0))
    yield "minimize_scalar bracket two", lambda: O.minimize_scalar(q, bracket=(-3.0, -2.0))
    yield "minimize_scalar tol", lambda: O.minimize_scalar(q, tol=1e-3)
    yield "minimize_scalar brent max_iter", lambda: O.minimize_scalar(q, options={"max_iter": 3})
    yield "minimize_scalar golden max_iter", lambda: O.minimize_scalar(q, method="golden", options={"max_iter": 3})
    yield "minimize_scalar bounded xatol", lambda: O.minimize_scalar(q, bounds=(0.0, 4.0), options={"xatol": 1e-2})
    yield "minimize_scalar bounded equal", lambda: O.minimize_scalar(q, bounds=(2.0, 2.0))
    yield "minimize_scalar nonfinite", lambda: O.minimize_scalar(lambda x: math.nan)
    yield "minimize_scalar repr", lambda: repr(O.minimize_scalar(q))
    yield "err minimize_scalar bounds brent", lambda: O.minimize_scalar(q, bounds=(0.0, 1.0), method="brent")
    yield "err minimize_scalar bracket bounded", lambda: O.minimize_scalar(q, bracket=(0.0, 1.0), method="bounded")
    yield "err minimize_scalar bounds reversed", lambda: O.minimize_scalar(q, bounds=(1.0, 0.0))
    yield "err minimize_scalar bracket order", lambda: O.minimize_scalar(q, bracket=(0.0, 2.0, 1.0))
    yield "err minimize_scalar method", lambda: O.minimize_scalar(q, method="Brent")
    yield "err minimize_scalar option", lambda: O.minimize_scalar(q, options={"xatol": 1.0})


def root_scalar_options(O, c):
    f = _scalar(c.f)
    for m in BRACKETING:
        yield f"root_scalar {m} xtol", lambda m=m: O.root_scalar(f, bracket=c.bracket, method=m, xtol=1e-3)
        yield f"root_scalar {m} rtol", lambda m=m: O.root_scalar(f, bracket=c.bracket, method=m, rtol=1e-6)
        yield f"root_scalar {m} max_iter=2", lambda m=m: O.root_scalar(f, bracket=c.bracket, method=m, max_iter=2)
        yield f"root_scalar {m} reversed", lambda m=m: O.root_scalar(f, bracket=c.bracket[::-1], method=m)
        yield f"root_scalar {m} root at end", lambda m=m: O.root_scalar(lambda x: x - 1.0, bracket=(1.0, 2.0), method=m)
        yield (
            f"err root_scalar {m} no sign change",
            lambda m=m: O.root_scalar(lambda x: x * x + 1.0, bracket=(0, 1), method=m),
        )
        yield f"err root_scalar {m} x0", lambda m=m: O.root_scalar(f, bracket=c.bracket, x0=0.6, method=m)
    yield "root_scalar toms748 k", lambda: O.root_scalar(f, bracket=c.bracket, method="toms748", options={"k": 2})
    yield "root_scalar default bracket", lambda: O.root_scalar(f, bracket=c.bracket)
    yield "root_scalar default x0", lambda: O.root_scalar(f, x0=0.6)
    yield "root_scalar default x0 x1", lambda: O.root_scalar(f, x0=0.6, x1=0.7)
    yield "root_scalar default fprime", lambda: O.root_scalar(f, x0=0.6, fprime=_scalar(c.fprime))
    yield "root_scalar zero derivative", lambda: O.root_scalar(lambda x: x * x + 1.0, x0=0.0, fprime=lambda x: 2.0 * x)
    yield "root_scalar repr", lambda: repr(O.root_scalar(f, bracket=c.bracket))
    yield "err root_scalar nothing", lambda: O.root_scalar(f)
    yield "err root_scalar method case", lambda: O.root_scalar(f, bracket=c.bracket, method="Brentq")
    yield "err root_scalar method unknown", lambda: O.root_scalar(f, bracket=c.bracket, method="ridder")
    yield "err root_scalar toms748 order", lambda: O.root_scalar(f, bracket=(2.0, 0.0), method="toms748")
    yield "err root_scalar newton no x0", lambda: O.root_scalar(f, method="newton")
    yield "err root_scalar bracket inf", lambda: O.root_scalar(f, bracket=(0.0, math.inf))
    yield "err root_scalar rtol small", lambda: O.root_scalar(f, bracket=c.bracket, rtol=1e-17)
    yield "err root_scalar k big", lambda: O.root_scalar(f, bracket=c.bracket, method="toms748", options={"k": 101})
    yield "err root_scalar option", lambda: O.root_scalar(f, bracket=c.bracket, options={"k": 2})


def elementwise_cases(sd, O):
    for c in fn.ROOT_CASES:
        f = _vec(sd, c.f)
        x0 = c.x0 if c.x0 is not None else c.bracket[0]
        yield f"find_root {c.id}", lambda c=c, f=f: O.find_root(f, c.bracket)
        yield f"bracket_root {c.id}", lambda f=f, x0=x0: O.bracket_root(f, x0)
    for c in fn.MIN_CASES:
        f = _vec(sd, c.f)
        if len(c.bracket) == 3:
            yield f"find_minimum {c.id}", lambda c=c, f=f: O.find_minimum(f, c.bracket)
        yield f"bracket_minimum {c.id}", lambda c=c, f=f: O.bracket_minimum(f, c.bracket[0])
    # batches: one call over many elements, each with its own parameter (args)
    cube = lambda x, a: x * x * x - a  # noqa: E731
    para = lambda x, a: (x - a) * (x - a) + 1.0  # noqa: E731
    for n in (1, 7, 10_000):
        a = sd.asarray(np.linspace(-5.0, 30.0, n).tolist())
        yield f"find_root batch {n}", lambda a=a: O.find_root(cube, (-10.0, 10.0), args=(a,))
        yield f"bracket_root batch {n}", lambda a=a: O.bracket_root(cube, 0.5, args=(a,))
        yield f"find_minimum batch {n}", lambda a=a: O.find_minimum(para, (-40.0, 0.0, 40.0), args=(a,))
        yield f"bracket_minimum batch {n}", lambda a=a: O.bracket_minimum(para, 0.0, args=(a,))
    shaped = sd.asarray(np.linspace(1.0, 8.0, 12).reshape(3, 4).tolist())
    yield "find_root batch 2d", lambda: O.find_root(cube, (0.0, 3.0), args=(shaped,))
    yield "find_root bracket arrays", lambda: O.find_root(cube, (sd.zeros(4), sd.full(4, 3.0)), args=(shaped[0],))
    yield "find_root tolerances", lambda: O.find_root(cube, (0.0, 3.0), args=(2.0,), xatol=1e-3, frtol=1e-6)
    yield "find_root max_iter", lambda: O.find_root(cube, (0.0, 3.0), args=(2.0,), max_iter=2)
    yield "find_root no sign change", lambda: O.find_root(para, (0.0, 3.0), args=(1.0,))
    yield "find_root nan", lambda: O.find_root(lambda x: x * math.nan, (0.0, 3.0))
    yield "bracket_root limits", lambda: O.bracket_root(cube, 0.5, args=(100.0,), xmin=-2.0, xmax=3.0)
    yield "bracket_root factor", lambda: O.bracket_root(cube, 0.5, 1.0, args=(20.0,), factor=3.0)
    yield "find_minimum tolerances", lambda: O.find_minimum(para, (-4.0, 0.0, 4.0), args=(1.0,), xatol=1e-3, fatol=1e-6)
    yield "bracket_minimum limits", lambda: O.bracket_minimum(para, 0.0, args=(50.0,), xmin=-1.0, xmax=10.0)
    yield "bracket_minimum xl0 xr0", lambda: O.bracket_minimum(para, 0.0, xl0=-0.5, xr0=0.5, args=(3.0,))
    yield "err find_root bracket", lambda: O.find_root(cube, (0.0,))
    yield "err find_minimum bracket order", lambda: O.find_minimum(para, (1.0, 0.0, 2.0))
    yield "err find_root not callable", lambda: O.find_root(3, (0.0, 1.0))
    yield "err bracket_root factor", lambda: O.bracket_root(cube, 0.5, factor=0.5)
    yield "err find_root max_iter", lambda: O.find_root(cube, (0.0, 3.0), args=(2.0,), max_iter=-1)
    yield "err find_minimum xatol", lambda: O.find_minimum(para, (-4.0, 0.0, 4.0), xatol=-1.0)


def linear_cases(L, sd, O):
    for pid, (A, b, bounds) in fn.LSQ_LINEAR.items():
        for method in ("trf", "bvls"):
            yield (
                f"lsq_linear {pid} {method}",
                lambda A=A, b=b, bounds=bounds, m=method: O.lsq_linear(A, b, bounds=bounds, method=m),
            )
        yield f"nnls {pid}", lambda A=A, b=b: O.nnls(A, b)
    r = L.rng(17)
    for m, n in ((200, 30), (2000, 50), (40, 60)):
        A = L.arr(r.standard_normal((m, n)))
        b = L.arr(r.standard_normal(m))
        box = ([-0.1] * n, [0.1] * n)
        for method in ("trf", "bvls"):
            yield f"lsq_linear random {m}x{n} {method}", lambda A=A, b=b, m_=method: O.lsq_linear(A, b, method=m_)
            yield (
                f"lsq_linear random {m}x{n} {method} box",
                lambda A=A, b=b, box=box, m_=method: O.lsq_linear(A, b, bounds=box, method=m_),
            )
        yield f"nnls random {m}x{n}", lambda A=A, b=b: O.nnls(A, b)
    A, b, bounds = fn.LSQ_LINEAR["cos.30x8.box"]
    yield "lsq_linear tol", lambda: O.lsq_linear(A, b, bounds=bounds, tol=1e-4)
    yield "lsq_linear max_iter", lambda: O.lsq_linear(A, b, bounds=bounds, max_iter=1)
    yield "lsq_linear bvls max_iter", lambda: O.lsq_linear(A, b, bounds=bounds, method="bvls", max_iter=1)
    yield "lsq_linear scalar bounds", lambda: O.lsq_linear(A, b, bounds=(-0.25, 0.25))
    yield "err lsq_linear A nan", lambda: O.lsq_linear([[1.0, math.nan], [0.0, 1.0]], [1.0, 2.0])
    yield "nnls max_iter", lambda: O.nnls(A, b, max_iter=1)
    yield "nnls zero", lambda: O.nnls([[1.0, 0.0], [0.0, 1.0]], [-1.0, -2.0])
    yield "err lsq_linear method", lambda: O.lsq_linear(A, b, method="dogbox")
    yield "err lsq_linear shapes", lambda: O.lsq_linear(A, b[:-1])
    yield "err lsq_linear bounds", lambda: O.lsq_linear(A, b, bounds=(1.0, -1.0))
    yield "err lsq_linear bounds pairs", lambda: O.lsq_linear(A, b, bounds=[(0, 1)] * 8)
    yield "err lsq_linear A 1d", lambda: O.lsq_linear(b, b)
    yield "err nnls shapes", lambda: O.nnls(A, b[:-1])
    yield "err nnls max_iter", lambda: O.nnls(A, b, max_iter=-1)
    yield "err nnls A 3d", lambda: O.nnls([[[1.0]]], [1.0])


def summaries(L):
    """summary() (O119) of each kind of result once: minimize (a long x, cut), least squares
    (cost, optimality), the linear least squares, a system's root (x with f(x)), the scalar
    functions (x a number) and the batched ones (one row per element, the brackets' ends)."""
    sd = L.sd
    O = sd.optimize
    quad = lambda x: float(sd.sum((x - 1.0) ** 2))  # noqa: E731
    rosen = lambda x: float(sd.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))  # noqa: E731
    c = sd.asarray([1.0, 2.0, 3.0])
    calls = {
        "minimize bfgs": lambda: O.minimize(rosen, sd.asarray([1.3, 0.7, 0.8, 1.9, 1.2]), method="bfgs"),
        "minimize nelder-mead long x": lambda: O.minimize(quad, sd.zeros((25,)), method="nelder-mead"),
        "minimize l-bfgs-b": lambda: O.minimize(rosen, sd.asarray([-1.2, 1.0]), method="l-bfgs-b"),
        "least_squares": lambda: O.least_squares(
            lambda x: sd.asarray([x[0] - 1.0, 10.0 * (x[1] - x[0] ** 2)]), sd.asarray([2.0, 2.0])
        ),
        "lsq_linear": lambda: O.lsq_linear(
            sd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), sd.asarray([1.0, 2.0, 2.0])
        ),
        "nnls": lambda: O.nnls(sd.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), sd.asarray([2.0, 1.0, -1.0])),
        "root": lambda: O.root(
            lambda x: sd.asarray([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]),
            sd.asarray([0.0, 0.0]),
        ),
        "root_scalar": lambda: O.root_scalar(lambda x: x**3 - 2.0, bracket=(0.0, 2.0)),
        "minimize_scalar": lambda: O.minimize_scalar(lambda x: (x - 0.7) ** 2),
        "find_root batch": lambda: O.find_root(lambda x, c: x * x - c, (sd.zeros((3,)), sd.full((3,), 5.0)), args=(c,)),
        "bracket_root batch": lambda: O.bracket_root(lambda x, c: x * x - c, sd.full((3,), 0.5), args=(c,)),
        "find_minimum batch": lambda: O.find_minimum(
            lambda x, c: (x - c) ** 2, (sd.zeros((3,)), sd.full((3,), 2.0), sd.full((3,), 5.0)), args=(c,)
        ),
        "bracket_minimum batch": lambda: O.bracket_minimum(lambda x, c: (x - c) ** 2, sd.zeros((3,)), args=(c,)),
        "OptimizeResult made": lambda: O.OptimizeResult(x=sd.asarray([1.0]), converged=True, nfev=3),
    }
    for name, call in calls.items():
        yield f"summary {name}", lambda call=call: call().summary()
