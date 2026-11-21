"""Golden cases for sd.differentiate (tools/golden.py has the contract).

The result class (fields, read-only rules, repr) and, since milestone C2, derivative, jacobian and
hessian on the collection of tests/differentiate/problems.py: derivative at every point of the
eleven elementary functions (alone and as a batch) and on the edge cases, jacobian of the residuals
and hessian of the sum of squares of the More-Garbow-Hillstrom cases at x0, the options, the call
forms and the errors. The functions are sd's own operations (the problems' expressions on sd
arrays), so the results do not depend on numpy.
"""

import math


class _Sd:
    """The problems' namespace M on sd arrays."""

    def __init__(self, sd):
        for name in ("exp", "log", "sqrt", "sin", "cos", "tan", "atan", "tanh"):
            setattr(self, name, getattr(sd, name))


def _mgh_vector(sd, c, scalar):
    """F (or sum F_i^2) of an MGH case on (n, N): each column through the case's float formula."""
    from tests.optimize.ad import FLOAT

    def f(x):
        cols = sd.matrix_transpose(x).tolist()
        rows = [[float(r) for r in c.fun(col, M=FLOAT)] for col in cols]
        if scalar:
            return sd.asarray([math.fsum(r * r for r in row) for row in rows])
        return sd.matrix_transpose(sd.asarray(rows))

    return f


def cases(L):
    sd = L.sd
    R = sd.differentiate.DifferentiateResult
    yield (
        "DifferentiateResult fields",
        lambda: R(df=2.7182818284590127, error=4.5e-14, converged=True, status=0, n_iter=2, nfev=11, x=1.0),
    )
    yield (
        "DifferentiateResult batch",
        lambda: R(batch=(2,), df=sd.asarray([1.0, 2.0]), error=sd.zeros(2), converged=sd.asarray([True, False]),
                  status=sd.asarray([0, -2]), n_iter=sd.asarray([2, 10]), nfev=sd.asarray([11, 27])),
    )  # fmt: skip
    yield (
        "DifferentiateResult hessian-like",
        lambda: R(ddf=sd.eye(2), error=sd.zeros((2, 2)), converged=sd.ones((2, 2)) > 0),
    )
    yield "DifferentiateResult repr", lambda: repr(R(df=float("inf"), converged=False, status=-3))
    yield "err DifferentiateResult positional", lambda: R(1)
    yield "err DifferentiateResult unknown field", lambda: R(nit=3)
    yield "err DifferentiateResult absent field", lambda: R(df=1.0).ddf
    yield "err DifferentiateResult read-only", lambda: setattr(R(df=1.0), "df", 2.0)
    yield "err DifferentiateResult unpack", lambda: tuple(R(df=1.0, error=0.0))
    yield from derivative_cases(L)
    yield from multivariate_cases(L)


def derivative_cases(L):
    from tests.differentiate import problems as P

    sd = L.sd
    D = sd.differentiate
    M = _Sd(sd)
    for f in P.ELEMENTARY:
        g = lambda x, f=f: f.expr(x, M)  # noqa: E731
        for x in f.points:
            yield f"derivative {f.name} {x!r}", lambda g=g, x=x: D.derivative(g, x)
        yield f"derivative {f.name} batch", lambda g=g, f=f: D.derivative(g, sd.asarray(list(f.points)))
    for e in P.EDGES:
        f = P.BY_NAME[e.function]
        g = lambda x, f=f: f.expr(x, M)  # noqa: E731
        yield f"derivative edge {e.id}", lambda g=g, e=e: D.derivative(g, e.x, **e.options)
    a = sd.linspace(0.5, 2.0, 12)
    sa = lambda x, a: sd.sin(a * x)  # noqa: E731
    yield "derivative args", lambda: D.derivative(sa, 1.0, args=(a,))
    yield "derivative 2d", lambda: D.derivative(sa, sd.reshape(sd.linspace(-1.0, 1.0, 12), (3, 4)), args=(2.0,))
    yield (
        "derivative 40001",
        lambda: D.derivative(sa, sd.linspace(-3.0, 3.0, 40001), args=(sd.linspace(0.5, 2.0, 40001),)),
    )
    yield "derivative steps", lambda: D.derivative(sd.exp, [1.0] * 5, initial_step=[0.5, 0.1, 0.0, -1.0, math.nan])
    yield (
        "derivative directions",
        lambda: D.derivative(sd.log, 0.01, step_direction=[-1, 0, 1, math.nan], initial_step=1e-3),
    )
    yield "derivative tolerances", lambda: D.derivative(sd.exp, 1.0, atol=1e-3, rtol=0.0)
    yield "derivative max_iter", lambda: D.derivative(sd.exp, 1.0, max_iter=1)
    yield "derivative order 3", lambda: D.derivative(sd.exp, 1.0, order=3)
    yield "derivative order 20", lambda: D.derivative(sd.exp, 1.0, order=20)  # past the cap (C118)
    yield "derivative order 18", lambda: D.derivative(sd.exp, 1.0, order=18)
    yield "derivative step_factor 3", lambda: D.derivative(sd.cos, 1.0, step_factor=3.0)
    yield "derivative step_factor 0.5", lambda: D.derivative(sd.exp, 1.0, step_factor=0.5)
    yield "derivative empty", lambda: D.derivative(sd.exp, [])
    yield "derivative nan value", lambda: D.derivative(lambda x: sd.where(x == 1.0, math.nan, sd.exp(x)), 1.0)
    yield "derivative non-finite x", lambda: D.derivative(sd.exp, [math.nan, math.inf, 1.0])
    yield "derivative order 1", lambda: D.derivative(sd.exp, 1.0, order=1)
    yield "err derivative complex value", lambda: D.derivative(lambda x: x * 1j, 1.0)
    yield "err derivative step_factor inf", lambda: D.derivative(sd.exp, 1.0, step_factor=math.inf)
    yield "derivative summary", lambda: D.derivative(sd.exp, 1.0).summary()
    yield "derivative summary batch", lambda: D.derivative(sd.exp, [1.0, 2.0, 30.0]).summary()
    yield "err derivative tolerances", lambda: D.derivative(sd.exp, 1.0, tolerances={"atol": 1.0})
    yield "err derivative step_factor 1", lambda: D.derivative(sd.exp, 1.0, step_factor=1.0)
    yield "err derivative order", lambda: D.derivative(sd.exp, 1.0, order=0)
    yield "err derivative max_iter", lambda: D.derivative(sd.exp, 1.0, max_iter=0)
    yield "err derivative atol", lambda: D.derivative(sd.exp, 1.0, atol=-1.0)
    yield "err derivative broadcast", lambda: D.derivative(sd.exp, [1.0, 2.0], initial_step=[1.0, 2.0, 3.0])
    yield "err derivative scalar value", lambda: D.derivative(lambda x: 1.0, 1.0)
    yield "err derivative shape", lambda: D.derivative(lambda x: x if x.ndim == 1 else x[:, :1], [1.0, 2.0])
    yield "err derivative bool value", lambda: D.derivative(lambda x: x > 0, 1.0)
    yield "err derivative not callable", lambda: D.derivative(1.0, 1.0)


def multivariate_cases(L):
    from tests.differentiate import problems as P

    sd = L.sd
    D = sd.differentiate
    for c in P.mgh_cases():
        x0 = [float(v) for v in c.x0()]
        yield f"jacobian {c.id}", lambda c=c, x0=x0: D.jacobian(_mgh_vector(sd, c, False), x0)
        if c.n <= P.HESSIAN_MAX_N:
            yield f"hessian {c.id}", lambda c=c, x0=x0: D.hessian(_mgh_vector(sd, c, True), x0)
    rosen = lambda x: sd.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2, axis=0)  # noqa: E731
    F = lambda x: sd.stack([sd.log(x[0]), sd.exp(x[1]) * x[0]])  # noqa: E731
    yield "jacobian gradient", lambda: D.jacobian(rosen, [1.2, 0.8, 1.1])
    yield "jacobian options", lambda: D.jacobian(F, [1e-3, 1.0], initial_step=[1e-4, 0.5], step_direction=[1, 0])
    yield "jacobian max_iter", lambda: D.jacobian(F, [1.0, 1.0], max_iter=1, order=4, step_factor=3.0)
    yield "jacobian nan column", lambda: D.jacobian(F, [1e-3, 1.0])
    yield "hessian rosen", lambda: D.hessian(rosen, [1.2, 0.8, 1.1])
    yield "hessian steps", lambda: D.hessian(lambda x: sd.log(x[0]) * x[1], [1e-3, 2.0], initial_step=[1e-4, 0.5])
    yield "hessian rtol 0", lambda: D.hessian(lambda x: x[0] ** 2 * x[1], [1.0, 2.0], rtol=0.0, atol=1e-6)
    yield "hessian summary", lambda: D.hessian(rosen, [1.2, 0.8, 1.1]).summary()
    G = lambda x: sd.stack([x[0] ** 2, x[0] * x[1]])  # noqa: E731
    yield "jacobian non-finite x", lambda: D.jacobian(G, [1.0, math.inf])
    yield "hessian non-finite x", lambda: D.hessian(lambda x: x[0] ** 2 * x[1] + sd.sin(x[1]), [1.0, math.nan])
    yield (
        "hessian options",
        lambda: D.hessian(lambda x: x[0] ** 2 * x[1] + sd.sin(x[1]), [1.0, 2.0], max_iter=1, order=4),
    )
    yield "err jacobian complex value", lambda: D.jacobian(lambda x: x * 1j, [1.0, 2.0])
    yield "err hessian steps", lambda: D.hessian(rosen, [1.0, 2.0], initial_step=[0.5, 0.5, 0.5])
    yield "err hessian x scalar", lambda: D.hessian(rosen, 1.0)
    yield "err jacobian x scalar", lambda: D.jacobian(rosen, 1.0)
    yield "err jacobian x 2d", lambda: D.jacobian(rosen, [[1.0, 2.0]])
    yield "err jacobian empty", lambda: D.jacobian(rosen, [])
    yield "err jacobian steps", lambda: D.jacobian(rosen, [1.0, 2.0], initial_step=[1.0, 2.0, 3.0])
    yield "err jacobian args", lambda: D.jacobian(rosen, [1.0, 2.0], args=(1.0,))
    yield "err jacobian first shape", lambda: D.jacobian(lambda x: sd.ones((3, 3)), [1.0, 2.0])
    yield "err jacobian later shape", lambda: D.jacobian(lambda x: x if x.shape[1] == 2 else x[:1], [1.0, 2.0])
    yield "err hessian not scalar", lambda: D.hessian(lambda x: x, [1.0, 2.0])
    yield "err hessian rtol", lambda: D.hessian(rosen, [1.0, 1.0], rtol=1e-15)
    yield "err hessian step_direction", lambda: D.hessian(rosen, [1.0, 1.0], step_direction=1)
