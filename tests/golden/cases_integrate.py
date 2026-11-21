"""Golden cases for sd.integrate (tools/golden.py has the contract).

The result class (milestone C0: fields, read-only rules, unpacking of the quad family's results,
C10, and repr) and, since milestone C3, the sample rules (trapezoid, simpson, romb,
cumulative_trapezoid, cumulative_simpson) on the sample sets of tests/integrate/problems.py,
fixed_quad on its smooth integrals and newton_cotes, with the options, the input forms and the
errors. Since milestone C4: quad on every integral of the collection without a weight (the
problems' integrands on Python floats and the math module), its options, complex_func, the
multiple integrals (dblquad, tplquad, nquad) and the errors; a result's summary() is part of it.
The samples and integrands are sd's own operations (the problems' formulas on sd arrays) or the
math module, so the results do not depend on numpy. Each milestone adds the cases of the
functions it makes public (docs/design/integrate.md §8).
"""

import math


def encode(x, L, enc):
    """An IntegrateResult: the fields it gives, the names of those it does not, and whether it
    unpacks (and into what)."""
    if type(x) is not L.sd.integrate.IntegrateResult:
        return None
    attrs, absent = {}, []
    for name in sorted(n for n in dir(x) if not n.startswith("_")):
        try:
            v = getattr(x, name)
        except AttributeError:
            absent.append(name)
            continue
        attrs[name] = enc(v() if callable(v) and not isinstance(v, type(x)) else v)  # summary()
    try:
        pair = enc(tuple(x))
    except TypeError as e:
        pair = {"error": "TypeError", "msg": str(e)}
    return {"obj": f"{type(x).__module__}.{type(x).__qualname__}", "attrs": attrs, "absent": absent,
            "unpack": pair, "repr": repr(x)}  # fmt: skip


def cases(L):
    sd = L.sd
    R = sd.integrate.IntegrateResult
    yield "IntegrateResult fields", lambda: R(integral=0.5, error=1e-15, converged=True, status=0, nfev=21)
    yield (
        "IntegrateResult quad-like",
        lambda: R._make(integral=0.5, error=1e-15, converged=True, status=0, message="ok", nfev=21, unpack=True),
    )
    yield "IntegrateResult unpack", lambda: tuple(R._make(integral=2.0, error=3.0, unpack=True))
    yield "IntegrateResult getitem", lambda: R._make(integral=2.0, error=3.0, unpack=True)[1]
    yield "err IntegrateResult no unpack", lambda: tuple(R(integral=2.0, error=3.0))
    yield (
        "IntegrateResult batch",
        lambda: R(batch=(3,), integral=sd.asarray([1.0, 2.0, 3.0]), error=sd.zeros(3),
                  converged=sd.asarray([True, False, True]), nfev=sd.asarray([67, 16387, 131])),
    )  # fmt: skip
    yield "IntegrateResult series", lambda: R(sum=1.6449340668482264, error=7.4e-9, converged=True, nfev=8561)
    yield "IntegrateResult repr", lambda: repr(R(integral=float("nan"), converged=False))
    yield "err IntegrateResult positional", lambda: R(1)
    yield "err IntegrateResult unknown field", lambda: R(estimate=1.0)
    yield "err IntegrateResult absent field", lambda: R(integral=1.0).error
    yield "err IntegrateResult read-only", lambda: setattr(R(integral=1.0), "integral", 2.0)
    yield (
        "err IntegrateResult array read-only",
        lambda: R(batch=(2,), integral=sd.zeros(2)).integral.__setitem__(0, 1.0),
    )
    yield from sample_cases(L)
    yield from fixed_cases(L)
    yield from quad_cases(L)
    yield from multiple_cases(L)


def _sample_sets(sd):
    """problems.samples()'s sets, built from the same formulas on sd arrays: {id: (y, x, dx)}."""
    out = {}
    for n in (2, 3, 4, 5, 17, 33, 4097):
        out[f"exp/uniform/{n}"] = (sd.exp(sd.linspace(0.0, 1.0, n)), None, 1.0 / (n - 1))
    for n in (3, 4, 10, 101):
        x = (sd.arange(n, dtype=sd.float64) / (n - 1)) ** 2
        out[f"exp/squares/{n}"] = (sd.exp(x), x, 1.0)
    g = sd.linspace(0.0, 2.0, 9)
    out["2d/rows"] = (sd.stack([sd.sin(g), sd.cos(g), g * g]), g, 1.0)
    return out


def sample_cases(L):
    sd = L.sd
    I = sd.integrate  # noqa: E741
    for sid, (y, x, dx) in _sample_sets(sd).items():
        kw = {"x": x} if x is not None else {"dx": dx}
        for rule in ("trapezoid", "simpson", "cumulative_trapezoid", "cumulative_simpson"):
            yield f"{rule} {sid}", lambda rule=rule, y=y, kw=kw: getattr(I, rule)(y, **kw)
        n = y.shape[-1] - 1
        if x is None and n & (n - 1) == 0:
            yield f"romb {sid}", lambda y=y, dx=dx: I.romb(y, dx=dx)
    y = sd.reshape(sd.sin(sd.linspace(0.0, 3.0, 60)), (3, 4, 5))
    for rule in ("trapezoid", "simpson", "cumulative_trapezoid", "cumulative_simpson"):
        for axis in (0, 1, -1):
            yield f"{rule} 3d axis {axis}", lambda rule=rule, axis=axis: getattr(I, rule)(y, dx=0.5, axis=axis)
    yield "romb 3d axis 2", lambda: I.romb(sd.reshape(sd.cos(sd.linspace(0.0, 3.0, 45)), (3, 3, 5)), axis=2)
    big = sd.sin(sd.linspace(0.0, 50.0, 2**17 + 1))
    xb = sd.linspace(0.0, 50.0, 2**17 + 1) ** 1.1
    yield "trapezoid long", lambda: I.trapezoid(big, xb)
    yield "simpson long", lambda: I.simpson(big, xb)
    yield "romb long", lambda: I.romb(big, dx=1e-3)
    yield "cumulative_trapezoid long", lambda: I.cumulative_trapezoid(big, xb)
    yield "cumulative_simpson long", lambda: I.cumulative_simpson(big, xb)
    many = sd.reshape(sd.cos(sd.linspace(0.0, 9.0, 64 * 1025)), (64, 1025))
    yield "simpson many lanes", lambda: I.simpson(many, dx=0.1)
    yield "cumulative_simpson many lanes", lambda: I.cumulative_simpson(many, sd.linspace(0.0, 1.0, 1025))
    yield "trapezoid complex", lambda: I.trapezoid(sd.asarray([1.0, 2j, 3.0, 4.0 - 1j]))
    yield "simpson complex", lambda: I.simpson(sd.asarray([1.0, 2j, 3.0, 4.0 - 1j]), [0.0, 0.5, 2.0, 2.5])
    yield "romb complex", lambda: I.romb(sd.asarray([1.0, 2j, 3.0, 4.0 - 1j, 1j]))
    yield "cumulative_simpson complex", lambda: I.cumulative_simpson(sd.asarray([1.0, 2j, 3.0, 4.0]), initial=1j)
    yield "trapezoid list", lambda: I.trapezoid([1, 2, 4, 8], [0.0, 1.0, 3.0, 4.0])
    yield "trapezoid bool", lambda: I.trapezoid([True, False, True])
    yield "trapezoid series", lambda: I.trapezoid(sd.frame.Series([1.0, None, 3.0]))
    yield "trapezoid empty", lambda: I.trapezoid([])
    yield "trapezoid empty lanes", lambda: I.trapezoid(sd.zeros((0, 3)))
    yield "simpson one", lambda: I.simpson([2.0])
    yield "simpson two", lambda: I.simpson([2.0, 3.0], dx=0.5)
    yield "simpson even", lambda: I.simpson([0.0, 1.0, 4.0, 9.0, 16.0, 25.0], [0.0, 1.0, 2.5, 3.0, 4.5, 5.0])
    yield "simpson zero spacing", lambda: I.simpson([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, 2.0])
    yield "trapezoid nan", lambda: I.trapezoid([1.0, math.nan, 2.0])
    yield "trapezoid x broadcast", lambda: I.trapezoid(sd.ones((2, 3)), sd.asarray([[0.0, 1.0, 3.0]]))
    yield "trapezoid dx per lane", lambda: I.trapezoid(sd.ones((2, 3)), dx=[[0.5], [2.0]])
    yield "cumulative_trapezoid initial", lambda: I.cumulative_trapezoid([1.0, 2.0, 3.0], initial=0)
    yield "cumulative_trapezoid one", lambda: I.cumulative_trapezoid([1.0])
    yield "cumulative_simpson initial lanes", lambda: I.cumulative_simpson(sd.ones((2, 4)), initial=[[1.0], [2.0]])
    yield "cumulative_simpson two", lambda: I.cumulative_simpson([1.0, 3.0], initial=1.0)
    yield "err trapezoid scalar", lambda: I.trapezoid(1.0)
    yield "err trapezoid str", lambda: I.trapezoid("abc")
    yield "err trapezoid x length", lambda: I.trapezoid([1.0, 2.0, 3.0], [0.0, 1.0])
    yield "err trapezoid x ndim", lambda: I.trapezoid(sd.ones((2, 3)), sd.ones((1, 2, 3)))
    yield "err trapezoid x broadcast", lambda: I.trapezoid(sd.ones((2, 3)), sd.ones((3, 3)))
    yield "err trapezoid x complex", lambda: I.trapezoid([1.0, 2.0], [0.0, 1j])
    yield "err trapezoid dx with x", lambda: I.trapezoid([1.0, 2.0], [0.0, 1.0], dx=2.0)
    yield "err trapezoid axis", lambda: I.trapezoid([1.0, 2.0], axis=1)
    yield "err trapezoid axis float", lambda: I.trapezoid([1.0, 2.0], axis=0.0)
    yield "err simpson dx bool", lambda: I.simpson([1.0, 2.0], dx=True)
    yield "err simpson dx shape", lambda: I.simpson(sd.ones((2, 3)), dx=[1.0, 2.0])
    yield "err romb samples", lambda: I.romb([1.0, 2.0, 3.0, 4.0])
    yield "err romb show", lambda: I.romb([1.0, 2.0], show=True)
    yield "err cumulative_trapezoid empty", lambda: I.cumulative_trapezoid([])
    yield "err cumulative_trapezoid initial", lambda: I.cumulative_trapezoid([1.0, 2.0], initial=1.0)
    yield "err cumulative_simpson x", lambda: I.cumulative_simpson([1.0, 2.0, 3.0], [0.0, 2.0, 1.0])
    yield "err cumulative_simpson initial complex", lambda: I.cumulative_simpson([1.0, 2.0, 3.0], initial=1j)
    yield "err cumulative_simpson positional", lambda: I.cumulative_simpson([1.0, 2.0], None, 1.0)


def fixed_cases(L):
    sd = L.sd
    I = sd.integrate  # noqa: E741
    for n in (1, 2, 5, 20, 100, 101, 300):
        yield f"fixed_quad exp n={n}", lambda n=n: I.fixed_quad(sd.exp, 0.0, 1.0, n=n)
    peak = lambda x: 4.0**-3 / ((x - math.pi / 4) ** 2 + 16.0**-3)  # noqa: E731
    yield "fixed_quad peak", lambda: I.fixed_quad(peak, 0.0, 1.0, n=50)
    yield "fixed_quad reversed", lambda: I.fixed_quad(sd.sin, math.pi, 0.0, n=8)
    yield "fixed_quad empty interval", lambda: I.fixed_quad(sd.exp, 2.0, 2.0)
    yield "fixed_quad vector", lambda: I.fixed_quad(lambda x: sd.stack([sd.exp(x), x * x, sd.cos(x)]), -1.0, 2.0, n=7)
    yield "fixed_quad list", lambda: I.fixed_quad(lambda x: [float(t) ** 2 for t in x.tolist()], 0.0, 1.0, n=3)
    yield "err fixed_quad infinite", lambda: I.fixed_quad(sd.exp, 0.0, math.inf)
    yield "err fixed_quad n", lambda: I.fixed_quad(sd.exp, 0.0, 1.0, n=0)
    yield "err fixed_quad scalar value", lambda: I.fixed_quad(lambda x: 1.0, 0.0, 1.0)
    yield "err fixed_quad shape", lambda: I.fixed_quad(lambda x: x[1:], 0.0, 1.0)
    yield "err fixed_quad args", lambda: I.fixed_quad(sd.exp, 0.0, 1.0, args=())
    for n in (1, 2, 4, 8, 14, 15, 20):
        yield f"newton_cotes {n}", lambda n=n: I.newton_cotes(n)
    yield "newton_cotes positions", lambda: I.newton_cotes([0.0, 0.5, 2.0])
    yield "newton_cotes equal", lambda: I.newton_cotes([0.0, 0.7, 3.0, 3.0], equal=True)
    yield "err newton_cotes 0", lambda: I.newton_cotes(0)
    yield "err newton_cotes float", lambda: I.newton_cotes(3.0)
    yield "err newton_cotes ends", lambda: I.newton_cotes([0.0, 1.0, 3.0])
    yield "err newton_cotes equal", lambda: I.newton_cotes(3, equal=1)


def quad_cases(L):
    from tests.integrate import problems as P

    sd = L.sd
    I = sd.integrate  # noqa: E741
    for p in P.INTEGRALS:
        if p.weight is None:
            kw = {"points": list(p.points)} if p.points else {}
            yield f"quad {p.id}", lambda p=p, kw=kw: I.quad(p.f, p.a, p.b, **kw)
    g = lambda x: math.exp(-x * x)  # noqa: E731
    root = lambda x: math.sqrt(x)  # noqa: E731
    yield "quad limit 1", lambda: I.quad(root, 0.0, 1.0, limit=1)
    yield "quad limit 3", lambda: I.quad(lambda x: math.log(x) / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0, limit=3)
    yield "quad limit 200", lambda: I.quad(lambda x: math.cos(2**10 * math.sin(x)), 0.0, math.pi, limit=200)
    yield "quad atol 0", lambda: I.quad(root, 0.0, 1.0, atol=0.0, rtol=1e-13)
    yield "quad rtol 0", lambda: I.quad(math.exp, 0.0, 10.0, atol=1e-2, rtol=0.0)
    yield "quad reversed", lambda: I.quad(math.exp, 1.0, 0.0)
    yield "quad reversed infinite", lambda: I.quad(g, math.inf, -math.inf)
    yield "quad left half line", lambda: I.quad(math.exp, -math.inf, 0.5)
    yield "quad empty", lambda: I.quad(math.exp, 2.0, 2.0)
    yield "quad empty infinite", lambda: I.quad(math.exp, math.inf, math.inf)
    yield "quad points reversed", lambda: I.quad(lambda x: abs(x - 0.25) ** 0.5, 1.0, 0.0, points=[0.25, 0.25, 7.0])
    yield "quad points empty", lambda: I.quad(lambda x: abs(x - 0.3), 0.0, 1.0, points=[])
    yield "quad points array", lambda: I.quad(lambda x: abs(x - 0.3), 0.0, 1.0, points=sd.asarray([0.3, 0.6]))
    yield "quad nan", lambda: I.quad(lambda x: math.nan if x > 0.5 else 1.0, 0.0, 1.0)
    yield "quad inf", lambda: I.quad(lambda x: math.inf if x > 0.9 else x, 0.0, 1.0)
    yield "quad int limits", lambda: I.quad(math.exp, 0, 1)
    yield "quad 0-d value", lambda: I.quad(lambda x: sd.asarray(x * x), 0.0, 1.0)
    yield "quad complex", lambda: I.quad(lambda x: complex(math.cos(3 * x), math.sqrt(x)), 0.0, 2.0, complex_func=True)
    yield "quad complex real values", lambda: I.quad(lambda x: x, 0.0, 1.0, complex_func=True)
    yield "quad unpack", lambda: tuple(I.quad(math.exp, 0.0, 1.0))
    yield (
        "quad summary peak",
        lambda: I.quad(lambda x: 4.0**-12 / ((x - math.pi / 4) ** 2 + 16.0**-12), 0.0, 1.0).summary(),
    )
    yield "err quad func", lambda: I.quad(1.0, 0, 1)
    yield "err quad nan limit", lambda: I.quad(math.exp, math.nan, 1)
    yield "err quad epsabs", lambda: I.quad(math.exp, 0, 1, epsabs=1e-3)
    yield "err quad args", lambda: I.quad(math.exp, 0, 1, args=(1,))
    yield "err quad atol", lambda: I.quad(math.exp, 0, 1, atol=-1.0)
    yield "err quad tolerances", lambda: I.quad(math.exp, 0, 1, atol=0.0, rtol=1e-15)
    yield "err quad limit", lambda: I.quad(math.exp, 0, 1, limit=0)
    yield "err quad limit float", lambda: I.quad(math.exp, 0, 1, limit=2.0)
    yield "err quad points scalar", lambda: I.quad(math.exp, 0, 1, points=0.5)
    yield "err quad points nan", lambda: I.quad(math.exp, 0, 1, points=[math.nan])
    yield "err quad points infinite", lambda: I.quad(math.exp, 0, math.inf, points=[1.0])
    yield "err quad points limit", lambda: I.quad(math.exp, 0, 1, points=[0.1, 0.2, 0.3], limit=3)
    yield "err quad weight", lambda: I.quad(math.exp, 0, 1, weight="cos", wvar=1.0)
    yield "err quad weight name", lambda: I.quad(math.exp, 0, 1, weight="tan")
    yield "err quad wvar", lambda: I.quad(math.exp, 0, 1, wvar=1.0)
    yield "err quad max_p1", lambda: I.quad(math.exp, 0, 1, max_p1=10)
    yield "err quad complex_func", lambda: I.quad(math.exp, 0, 1, complex_func=1)
    yield "err quad shape", lambda: I.quad(lambda x: [1.0, 2.0], 0, 1)
    yield "err quad complex value", lambda: I.quad(lambda x: 1j, 0, 1)
    yield "err quad bool value", lambda: I.quad(lambda x: True, 0, 1)


# problems.MULTI with the math module (its integrands use numpy's exp)
_MULTI = (
    ("dbl-linear", "dblquad", lambda y, x: x + y, 1.0, 2.0, (lambda x: x, lambda x: 2 * x)),
    ("dbl-const-limits", "dblquad", lambda y, x: y + x + 1 + 2, 1.0, 2.0, (1.0, 2.0)),
    ("dbl-gauss-square", "dblquad", lambda y, x: math.exp(-(x * x + y * y)), 0.0, 1.0, (0.0, 1.0)),
    ("dbl-gauss-plane", "dblquad", lambda y, x: math.exp(-(x * x + y * y)), -math.inf, math.inf,
     (-math.inf, math.inf)),
    ("tpl-linear", "tplquad", lambda z, y, x: (x + y + z) * 2.0, 1.0, 2.0,
     (lambda x: x, lambda x: 2 * x, lambda x, y: x - y, lambda x, y: x + y)),
    ("tpl-gauss-cube", "tplquad", lambda z, y, x: math.exp(-(x * x + y * y + z * z)), 0.0, 1.0,
     (0.0, 1.0, 0.0, 1.0)),
)  # fmt: skip


def multiple_cases(L):
    sd = L.sd
    I = sd.integrate  # noqa: E741
    for name, kind, f, a, b, lims in _MULTI:
        yield f"{kind} {name}", lambda kind=kind, f=f, a=a, b=b, lims=lims: getattr(I, kind)(f, a, b, *lims)
    yield "dblquad tolerances", lambda: I.dblquad(lambda y, x: math.sqrt(abs(x - y)), 0.0, 1.0, 0.0, 1.0, rtol=1e-4)
    yield "nquad box", lambda: I.nquad(lambda x, y: math.exp(-(x * x + y * y)), [[0.0, 1.0], [0.0, 1.0]])
    yield "nquad error max", lambda: I.nquad(lambda y, x: math.sqrt(abs(x - y)), [[0, 1], [0, 1]])

    def func(x0, x1, x2, x3):
        return x0**2 + x1 * x2 - x3**3 + math.sin(x0) + (1 if (x0 - 0.2 * x3 - 0.5 - 0.25 * x1 > 0) else 0)

    yield (
        "nquad functions",
        lambda: I.nquad(func, [[0, 1], [-1, 1], [0.13, 0.8], [-0.15, 1]],
                        opts=[lambda *a: {"points": [0.2 * a[2] + 0.5 + 0.25 * a[0]]}, {}, {}, {}]),
    )  # fmt: skip
    yield (
        "nquad range functions",
        lambda: I.nquad(lambda x, y, z: math.cos(x * y) * z, [[0, 1], lambda z: [z, 2 * z + 0.5], [0, 2]],
                        opts=[lambda y, z: {"rtol": 1e-10, "limit": 20}, {}, {}]),
    )  # fmt: skip
    yield "nquad inner limit", lambda: I.nquad(lambda y, x: math.cos(2**10 * math.sin(y)) + x, [[0, math.pi], [0, 1]])
    yield "nquad summary", lambda: I.nquad(lambda x: x, [[0, 1]]).summary()
    yield "err dblquad hfun nan", lambda: I.dblquad(lambda y, x: 1.0, 0, 1, 0, lambda x: math.nan)
    yield "err dblquad gfun", lambda: I.dblquad(lambda y, x: 1.0, 0, 1, "0", 1)
    yield "err dblquad complex", lambda: I.dblquad(lambda y, x: 1j, 0, 1, 0, 1)
    yield "err tplquad rfun", lambda: I.tplquad(lambda z, y, x: 1.0, 0, 1, 0, 1, 0, lambda x, y: [1, 2])
    yield "err nquad empty", lambda: I.nquad(lambda x: x, [])
    yield "err nquad pair", lambda: I.nquad(lambda x: x, [[0, 1, 2]])
    yield "err nquad opts key", lambda: I.nquad(lambda x: x, [[0, 1]], opts={"epsabs": 1e-3})
    yield "err nquad opts length", lambda: I.nquad(lambda x, y: x, [[0, 1], [0, 1]], opts=[{}])
    yield "err nquad range function", lambda: I.nquad(lambda x, y: x, [[0, 1], lambda: 3.0])
    yield "err nquad opts function", lambda: I.nquad(lambda x, y: x, [[0, 1], [0, 1]], opts=[lambda y: 3, {}])
    yield "err nquad weight", lambda: I.nquad(lambda x: x, [[0, 1]], opts={"weight": "sin", "wvar": 2.0})
