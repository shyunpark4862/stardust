"""Golden cases for sd.integrate (tools/golden.py has the contract).

The result class (milestone C0: fields, read-only rules, unpacking of the quad family's results,
C10, and repr) and, since milestone C3, the sample rules (trapezoid, simpson, romb,
cumulative_trapezoid, cumulative_simpson) on the sample sets of tests/integrate/problems.py,
fixed_quad on its smooth integrals and newton_cotes, with the options, the input forms and the
errors. Since milestone C4: quad on every integral of the collection without a weight (the
problems' integrands on Python floats and the math module), its options, complex_func, the
multiple integrals (dblquad, tplquad, nquad) and the errors; a result's summary() is part of it.
Since milestone C5: quad's weights (QAWO, QAWF, QAWS, QAWC) on the collection's weighted integrals
and the sweep of tests/integrate/weights.py, with max_p1, limlst, the frequency 0, the left half
line, complex_func, nquad's weighted levels and the errors.
Since milestone C6: tanhsinh and nsum (the elementwise family) on integrands and series of sd's own
operations, with the levels, log=True, the limits' transforms, batches with args, nsum's branches
and the errors.
Since milestone C7: quad_vec (scalar and array values, the rules, norms, infinite ranges, points,
limits, statuses) and cubature (the three rules in one to four variables, array values, infinite
limits, points, flipped limits, max_subdivisions, status 3) with their summaries and errors.
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
    yield from elementwise_cases(L)
    yield from adaptive_cases(L)


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
    yield "err fixed_quad n too large", lambda: I.fixed_quad(sd.exp, 0.0, 1.0, n=10001)  # C109
    yield "err newton_cotes rn too large", lambda: I.newton_cotes(257)  # C110
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
    yield "err newton_cotes nan position", lambda: I.newton_cotes([0.0, math.nan, 2.0])
    yield "newton_cotes 64", lambda: I.newton_cotes(64)
    yield "newton_cotes positions 6", lambda: I.newton_cotes([0.0, 0.1, 0.2, 3.0, 4.5, 5.0])


def quad_cases(L):
    from tests.integrate import problems as P

    sd = L.sd
    I = sd.integrate  # noqa: E741
    for p in P.INTEGRALS:
        if p.weight is None:
            kw = {"points": list(p.points)} if p.points else {}
            yield f"quad {p.id}", lambda p=p, kw=kw: I.quad(p.f, p.a, p.b, **kw)
    yield from weight_cases(L)
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
    yield "dblquad inf", lambda: I.dblquad(lambda y, x: math.inf, 0.0, 1.0, 0.0, 1.0)  # status -3 (C119)
    # status 3 on each of QAGS, QAGI, QAGP and status 4 (docs/api/integrate.md §2)
    inv = lambda x: 1.0 / x if x else math.inf  # noqa: E731
    yield "quad status 3 QAGS", lambda: I.quad(inv, 0.0, 1.0, limit=10000)
    yield "quad status 3 QAGI", lambda: I.quad(inv, 0.0, math.inf, limit=10000)
    yield (
        "quad status 3 QAGP",
        lambda: I.quad(lambda x: 1.0 / abs(x) if x else math.inf, -1.0, 1.0, limit=10000, points=[0.0]),
    )
    yield "quad status 4", lambda: I.quad(lambda x: x**-0.999 / (1.0 + x) if x else math.inf, 0.0, math.inf)
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
    yield "err quad weight name", lambda: I.quad(math.exp, 0, 1, weight="tan")
    yield "err quad wvar", lambda: I.quad(math.exp, 0, 1, wvar=1.0)
    yield "err quad max_p1", lambda: I.quad(math.exp, 0, 1, max_p1=10)
    yield "err quad complex_func", lambda: I.quad(math.exp, 0, 1, complex_func=1)
    yield "err quad shape", lambda: I.quad(lambda x: [1.0, 2.0], 0, 1)
    yield "err quad complex value", lambda: I.quad(lambda x: 1j, 0, 1)
    yield "err quad bool value", lambda: I.quad(lambda x: True, 0, 1)


def weight_cases(L):
    from tests.integrate import problems as P
    from tests.integrate import weights as W

    sd = L.sd
    I = sd.integrate  # noqa: E741
    for p in [p for p in P.INTEGRALS if p.weight is not None] + list(W.SWEEP):
        yield f"quad {p.id}", lambda p=p: I.quad(p.f, p.a, p.b, weight=p.weight, wvar=p.wvar)
    log0 = lambda x: math.log(x) if x > 0 else 0.0  # noqa: E731
    rsqrt = lambda x: 1 / math.sqrt(x) if x > 0 else 0.0  # noqa: E731
    yield "quad cos max_p1 2", lambda: I.quad(log0, 0.0, 1.0, weight="cos", wvar=300.0, max_p1=2)
    yield "quad sin max_p1 1", lambda: I.quad(log0, 0.0, 1.0, weight="sin", wvar=300.0, max_p1=1)
    yield "quad cos limit 1", lambda: I.quad(math.exp, 0.0, 1.0, weight="cos", wvar=1.0, limit=1)
    yield "quad cos small interval", lambda: I.quad(lambda x: 1.0 if x < 0.37 else 0.5, -3.3, 3.7, weight="sin",
                                                    wvar=300.0, atol=1e-14, rtol=1e-14)  # fmt: skip
    yield "quad qawf limlst 3", lambda: I.quad(rsqrt, 0.0, math.inf, weight="cos", wvar=1.0, limlst=3)
    yield "quad qawf limit 3", lambda: I.quad(rsqrt, 0.0, math.inf, weight="sin", wvar=2.0, limit=3)
    yield "quad qawf zero frequency", lambda: I.quad(lambda x: math.exp(-x), 1.0, math.inf, weight="cos", wvar=0.0)
    yield "quad qawf zero sine", lambda: I.quad(lambda x: math.exp(-x), 1.0, math.inf, weight="sin", wvar=0.0)
    yield "quad qawf left sine", lambda: I.quad(math.exp, -math.inf, 1.0, weight="sin", wvar=2.5)
    yield "quad qawf reversed", lambda: I.quad(rsqrt, math.inf, 0.0, weight="sin", wvar=2.0)
    # a large frequency: the Clenshaw-Curtis estimate's rounding floor (C123)
    yield "quad qawf omega 2^40", lambda: I.quad(lambda x: math.exp(-x), 0.0, math.inf, weight="cos", wvar=2.0**40)
    yield "quad qawf summary", lambda: I.quad(rsqrt, 0.0, math.inf, weight="cos", wvar=1.0).summary()
    yield "quad qaws limit 2", lambda: I.quad(math.exp, 0.0, 1.0, weight="alg-log", wvar=(-0.5, 0.5), limit=2)
    yield "quad qaws reversed", lambda: I.quad(math.exp, 1.0, -1.0, weight="alg-logb", wvar=(0.5, -0.25))
    yield "quad qawc outside", lambda: I.quad(math.exp, 0.0, 1.0, weight="cauchy", wvar=3.0)
    yield "quad weight empty", lambda: I.quad(math.exp, 1.0, 1.0, weight="cauchy", wvar=1.0)
    yield "quad weight complex", lambda: I.quad(lambda x: complex(math.exp(-x), x), 0.0, 1.0, weight="cos", wvar=3.0,
                                                complex_func=True)  # fmt: skip
    yield "quad weight complex qawf", lambda: I.quad(lambda x: complex(math.exp(-x), math.exp(-2 * x)), 0.0,
                                                     math.inf, weight="sin", wvar=2.0, complex_func=True)  # fmt: skip
    yield (
        "quad weight complex summary",
        lambda: I.quad(
            lambda x: complex(math.exp(-x), 1.0), 0.0, math.inf, weight="cos", wvar=1.0, complex_func=True
        ).summary(),
    )
    yield "nquad weights", lambda: I.nquad(lambda x, y: math.exp(-x) * (1 + y), [[0, math.inf], [0, 1]],
                                           opts=[lambda y: {"weight": "cos", "wvar": 1 + y},
                                                 {"weight": "alg", "wvar": (0.5, -0.5)}])  # fmt: skip
    yield "err quad wvar missing", lambda: I.quad(math.exp, 0, 1, weight="cos")
    yield "err quad wvar pair", lambda: I.quad(math.exp, 0, 1, weight="alg", wvar=0.5)
    yield "err quad wvar alpha", lambda: I.quad(math.exp, 0, 1, weight="alg", wvar=(-1.0, 0.0))
    yield "err quad wvar infinite", lambda: I.quad(math.exp, 0, 1, weight="sin", wvar=math.inf)
    yield "err quad weight points", lambda: I.quad(math.exp, 0, 1, weight="cos", wvar=1.0, points=[0.5])
    yield "err quad qaws limit", lambda: I.quad(math.exp, 0, 1, weight="alg", wvar=(0, 0), limit=1)
    yield "err quad limlst finite", lambda: I.quad(math.exp, 0, 1, weight="cos", wvar=1.0, limlst=10)
    yield "err quad weight whole line", lambda: I.quad(math.exp, -math.inf, math.inf, weight="cos", wvar=1.0)
    yield "err quad alg infinite", lambda: I.quad(math.exp, 0, math.inf, weight="alg", wvar=(0, 0))
    yield "err quad cauchy end", lambda: I.quad(math.exp, 0, 1, weight="cauchy", wvar=1)
    yield "err quad qawf atol", lambda: I.quad(math.exp, 0, math.inf, weight="sin", wvar=1.0, atol=0.0, rtol=1e-10)
    yield "err quad qawf rtol", lambda: I.quad(math.exp, -math.inf, 0, weight="cos", wvar=1.0, rtol=1e-10)
    yield "err quad max_p1 range", lambda: I.quad(math.exp, 0, 1, max_p1=2**31)


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
    yield "nquad weight", lambda: I.nquad(lambda x: x, [[0, 1]], opts={"weight": "sin", "wvar": 2.0})


def elementwise_cases(L):
    sd = L.sd
    I = sd.integrate  # noqa: E741
    inf = math.inf

    def gauss(x):
        return sd.exp(-x * x)

    yield "tanhsinh exp", lambda: I.tanhsinh(sd.exp, 0.0, 1.0)
    yield "tanhsinh gauss real line", lambda: I.tanhsinh(gauss, -inf, inf)
    yield "tanhsinh gauss left", lambda: I.tanhsinh(gauss, -inf, 0.5)
    yield "tanhsinh gauss right", lambda: I.tanhsinh(gauss, -0.5, inf)
    yield "tanhsinh reversed", lambda: I.tanhsinh(sd.cos, 2.0, -1.0)
    yield "tanhsinh equal", lambda: I.tanhsinh(sd.cos, 2.0, 2.0)
    yield "tanhsinh singular end", lambda: I.tanhsinh(lambda x: sd.log(x) * sd.log(x), 0.0, 1.0)
    yield "tanhsinh peak", lambda: I.tanhsinh(lambda x: 1 / ((x - 0.3) ** 2 + 1e-4), 0.0, 1.0)
    yield "tanhsinh levels", lambda: I.tanhsinh(lambda x: sd.sqrt(x), 0.0, 1.0, min_level=0, max_level=4)
    yield "tanhsinh min_level 5", lambda: I.tanhsinh(sd.sin, 0.0, 3.0, min_level=5)
    yield "tanhsinh tolerances", lambda: I.tanhsinh(sd.cos, 0.0, 10.0, atol=1e-6, rtol=0.0)
    yield "tanhsinh log", lambda: I.tanhsinh(lambda x: -x * x, 20.0, 30.0, log=True)
    yield "tanhsinh log real line", lambda: I.tanhsinh(lambda x: -x * x, -inf, inf, log=True, rtol=-30.0)
    yield "tanhsinh nan", lambda: I.tanhsinh(sd.exp, [math.nan, 0.0], [1.0, math.nan])
    yield (
        "tanhsinh batch",
        lambda: I.tanhsinh(
            lambda x, c: sd.cos(c * sd.sin(x)), 0.0, math.pi, args=(sd.asarray([1.0, 4.0, 16.0, 64.0]),)
        ),
    )
    yield (
        "tanhsinh batch 2d",
        lambda: I.tanhsinh(lambda x, p: x**p, 0.0, sd.asarray([[1.0], [2.0]]), args=(sd.asarray([0.5, -0.5, 2.0]),)),
    )
    yield "tanhsinh summary", lambda: I.tanhsinh(gauss, -inf, inf).summary()
    yield "tanhsinh batch summary", lambda: I.tanhsinh(sd.exp, 0.0, sd.asarray([1.0, 2.0])).summary()
    yield "tanhsinh empty", lambda: I.tanhsinh(sd.exp, [], 1.0)
    big = sd.linspace(0.5, 3.0, 40001)
    yield "tanhsinh large batch", lambda: I.tanhsinh(lambda x, c: sd.exp(-c * x * x), 0.0, 2.0, args=(big,))
    yield "err tanhsinh max_level", lambda: I.tanhsinh(sd.exp, 0.0, 1.0, max_level=17)
    yield "err tanhsinh log reversed", lambda: I.tanhsinh(lambda x: -x, 1.0, 0.0, log=True)
    yield "err tanhsinh rtol", lambda: I.tanhsinh(sd.exp, 0.0, 1.0, rtol=math.nan)
    yield "err tanhsinh value shape", lambda: I.tanhsinh(lambda x: sd.ravel(x), 0.0, 1.0)
    yield "err tanhsinh complex", lambda: I.tanhsinh(lambda x: x * 1j, 0.0, 1.0)
    yield "err tanhsinh keyword", lambda: I.tanhsinh(sd.exp, 0.0, 1.0, maxlevel=3)
    yield "nsum zeta 2", lambda: I.nsum(lambda k: 1 / (k * k), 1.0, inf)
    yield "nsum direct", lambda: I.nsum(lambda k: 1 / (k * k), 1.0, 1000.0)
    yield "nsum finite many terms", lambda: I.nsum(lambda k: 1 / (k * k), 1.0, 1e7)
    yield "nsum step", lambda: I.nsum(lambda x: 1 / x - 1 / (x + 1), 1.0, inf, step=2.0)
    yield "nsum left", lambda: I.nsum(lambda k: 1 / (k * k), -inf, -1.0)
    yield "nsum both sides", lambda: I.nsum(lambda k: sd.exp(-k * k / 10), -inf, inf)
    yield "nsum log", lambda: I.nsum(lambda k: -2 * sd.log(k), 1.0, inf, log=True)
    yield "nsum log both sides", lambda: I.nsum(lambda k: -k * k, -inf, inf, log=True)
    yield "nsum max_terms", lambda: I.nsum(lambda k: 1 / k**1.01, 1.0, inf, max_terms=100)
    yield "nsum divergent", lambda: I.nsum(lambda k: 1 / k, 1.0, inf)
    yield "nsum positions overflow", lambda: I.nsum(lambda k: 1 / (k * k), [1.0, 1.0], [inf, inf], step=[1e306, 1.0])
    yield "nsum empty", lambda: I.nsum(lambda k: 1 / (k * k), [], 1.0)
    yield "err tanhsinh bool end", lambda: I.tanhsinh(sd.exp, True, 1.0)
    yield "err nsum log rtol", lambda: I.nsum(lambda k: -2 * sd.log(k), 1.0, inf, log=True, rtol=math.nan)
    yield (
        "nsum invalid",
        lambda: I.nsum(lambda k: 1 / (k * k), [1.0, 10.0, math.nan], [10.0, 1.0, 5.0], step=[-1.0, 1.0, 1.0]),
    )
    yield (
        "nsum batch",
        lambda: I.nsum(lambda k, c: sd.exp(-c * k * k), sd.asarray([1.0, -inf, 0.0, -inf, 3.0]),
                       sd.asarray([inf, 0.0, 50.0, inf, 2.0]), args=(sd.asarray([0.1, 0.2, 0.3, 0.01, 1.0]),)),
    )  # fmt: skip
    yield "nsum summary", lambda: I.nsum(lambda k: 1 / (k * k), 1.0, inf).summary()
    yield "err nsum max_terms", lambda: I.nsum(sd.exp, 0.0, 1.0, max_terms=2**24 + 1)
    yield "err nsum max_terms float", lambda: I.nsum(sd.exp, 0.0, 1.0, max_terms=1e3)
    yield "err nsum tolerances", lambda: I.nsum(sd.exp, 0.0, 1.0, tolerances={"rtol": 1e-3})


def adaptive_cases(L):
    sd = L.sd
    I = sd.integrate  # noqa: E741
    inf = math.inf

    def three(x):
        return sd.asarray([math.exp(x), math.sin(x), x * x])

    def peaks(x):
        return sd.asarray([1 / (1e-4 + (x - 0.3) ** 2), math.sqrt(abs(x - 0.7))])

    def gauss(X):
        return sd.exp(-sd.sum(X * X, axis=-1))

    def sing(X):
        return 1 / sd.sqrt(sd.abs(X[:, 0] - 0.3) + sd.abs(X[:, 1] - 0.6))

    yield "quad_vec three", lambda: I.quad_vec(three, 0.0, 1.0)
    yield "quad_vec scalar", lambda: I.quad_vec(lambda x: math.exp(-x * x), 0.0, 3.0)
    yield "quad_vec real line", lambda: I.quad_vec(lambda x: sd.asarray([math.exp(-x * x), 1 / (1 + x * x)]), -inf, inf)
    yield "quad_vec half line", lambda: I.quad_vec(lambda x: math.exp(-x), 0.0, inf)
    yield "quad_vec left half line", lambda: I.quad_vec(lambda x: math.exp(x), -inf, 0.0)
    yield "quad_vec reversed real line", lambda: I.quad_vec(lambda x: math.exp(-x * x), inf, -inf)
    yield "quad_vec same infinities", lambda: I.quad_vec(lambda x: 1.0, inf, inf)
    yield "quad_vec empty range", lambda: I.quad_vec(three, 2.0, 2.0)
    yield "quad_vec reversed", lambda: I.quad_vec(three, 1.0, 0.0)
    yield "quad_vec peaks", lambda: I.quad_vec(peaks, 0.0, 1.0)
    yield "quad_vec peaks max", lambda: I.quad_vec(peaks, 0.0, 1.0, norm="max")
    yield "quad_vec gk15", lambda: I.quad_vec(peaks, 0.0, 1.0, quadrature="gk15")
    yield "quad_vec trapezoid", lambda: I.quad_vec(three, 0.0, 1.0, quadrature="trapezoid", rtol=1e-6)
    yield "quad_vec trapezoid real line", lambda: I.quad_vec(lambda x: math.exp(-x * x), -inf, inf,
                                                             quadrature="trapezoid", rtol=1e-6)  # fmt: skip
    yield "quad_vec points", lambda: I.quad_vec(peaks, 0.0, 1.0, points=[0.3, 0.7, 2.0])
    yield "quad_vec points real line", lambda: I.quad_vec(lambda x: math.exp(-abs(x - 1)), -inf, inf, points=[1.0])
    yield "quad_vec limit", lambda: I.quad_vec(lambda x: math.sin(1 / x) if x else 0.0, 0.0, 1.0, limit=40)
    yield "quad_vec limit 1", lambda: I.quad_vec(three, 0.0, 1.0, limit=1)
    yield "quad_vec rounding", lambda: I.quad_vec(lambda x: [1.0 + 1e-17 * x, 2.0], 0.0, 1.0, rtol=1e-17, atol=0.0)
    yield "quad_vec nan", lambda: I.quad_vec(lambda x: math.log(x - 0.5) if x > 0.5 else math.nan, 0.0, 1.0)
    yield "quad_vec 2-D value", lambda: I.quad_vec(lambda x: sd.asarray([[math.exp(x), x], [x * x, 1.0]]), 0.0, 1.0)
    k = sd.linspace(0.0, 30.0, 1000)
    yield "quad_vec 1000 values", lambda: I.quad_vec(lambda x: sd.cos(k * x), 0.0, 2.0)
    yield "quad_vec summary", lambda: I.quad_vec(three, 0.0, 1.0).summary()
    yield "quad_vec scalar summary", lambda: I.quad_vec(lambda x: x * x, 0.0, 1.0).summary()
    yield "err quad_vec shape", lambda: I.quad_vec(lambda x: [1.0] if x < 0.5 else [1.0, 2.0], 0.0, 1.0)
    yield "err quad_vec complex", lambda: I.quad_vec(lambda x: 1j, 0.0, 1.0)
    yield "err quad_vec empty value", lambda: I.quad_vec(lambda x: [], 0.0, 1.0)
    yield "err quad_vec norm", lambda: I.quad_vec(three, 0.0, 1.0, norm="fro")
    cache = L.np.linspace(0.0, 1.0, 50000)  # 8 x 50000-byte values: past scipy's 100e6-byte cache (C98)
    yield (
        "quad_vec past the cache",
        lambda: I.quad_vec(lambda x: L.np.sqrt(L.np.abs(x - cache)), 0.0, 1.0, rtol=1e-6, limit=400),
    )
    yield "err quad_vec quadrature", lambda: I.quad_vec(three, 0.0, 1.0, quadrature="gk31")
    yield "err quad_vec limit", lambda: I.quad_vec(three, 0.0, 1.0, limit=0)
    yield "err quad_vec nan limit", lambda: I.quad_vec(three, 0.0, math.nan)
    yield "err quad_vec epsabs", lambda: I.quad_vec(three, 0.0, 1.0, epsabs=1e-3)
    yield "err quad_vec workers", lambda: I.quad_vec(three, 0.0, 1.0, workers=2)

    yield "cubature gk21 2-D", lambda: I.cubature(gauss, [0.0, 0.0], [1.0, 1.0])
    yield "cubature gk15 3-D", lambda: I.cubature(gauss, [0.0, 0.0, 0.0], [1.0, 1.0, 2.0], rule="gk15")
    yield "err cubature gk21 8-D", lambda: I.cubature(lambda X: X[:, 0], [0.0] * 8, [1.0] * 8)  # C117
    yield "cubature genz-malik 2-D", lambda: I.cubature(gauss, [0.0, 0.0], [1.0, 1.0], rule="genz-malik")
    yield "cubature genz-malik 4-D", lambda: I.cubature(gauss, [0.0] * 4, [1.0] * 4, rule="genz-malik", rtol=1e-6)
    yield "cubature 1-D", lambda: I.cubature(lambda X: sd.cos(X[:, 0]), [0.0], [3.0])
    yield "cubature vector", lambda: I.cubature(lambda X: sd.stack([gauss(X), X[:, 0] * X[:, 1]], axis=1), [0.0, 0.0],
                                                [2.0, 1.0], rule="genz-malik")  # fmt: skip
    yield (
        "cubature elementwise",
        lambda: I.cubature(lambda X: sd.exp(-X * X), [0.0, 0.0], [2.0, 1.0], rule="genz-malik"),
    )
    yield "cubature plane", lambda: I.cubature(gauss, [-inf, -inf], [inf, inf])
    yield "cubature plane genz-malik", lambda: I.cubature(gauss, [-inf, -inf], [inf, inf], rule="genz-malik")
    yield "cubature half plane", lambda: I.cubature(gauss, [0.0, -1.0], [inf, 1.0], rule="genz-malik")
    yield "cubature left half line", lambda: I.cubature(lambda X: sd.exp(X[:, 0]), [-inf], [0.0])
    yield "cubature flipped", lambda: I.cubature(gauss, [1.0, 0.0], [0.0, 1.0], rule="genz-malik")
    yield "cubature points", lambda: I.cubature(sing, [0.0, 0.0], [1.0, 1.0], rule="genz-malik",
                                                points=[[0.3, 0.6], [0.5, 0.5]], max_subdivisions=200)  # fmt: skip
    yield "cubature points plane", lambda: I.cubature(gauss, [-inf, -inf], [inf, inf], rule="genz-malik",
                                                      points=[[1.0, 1.0]])  # fmt: skip
    yield "cubature max_subdivisions", lambda: I.cubature(sing, [0.0, 0.0], [1.0, 1.0], rule="genz-malik",
                                                          max_subdivisions=7)  # fmt: skip
    yield "cubature zero width", lambda: I.cubature(gauss, [1.0, 0.0], [1.0, 1.0], rule="genz-malik")
    yield "cubature zero width split", lambda: I.cubature(gauss, [1.0, 0.0], [1.0, 1.0], points=[[1.0, 0.5]])
    yield "cubature nan", lambda: I.cubature(lambda X: sd.log(X[:, 0] - 0.5), [0.0, 0.0], [1.0, 1.0], rule="genz-malik")
    yield "cubature summary", lambda: I.cubature(gauss, [0.0, 0.0], [1.0, 1.0], rule="genz-malik").summary()
    yield "cubature vector summary", lambda: I.cubature(lambda X: sd.stack([gauss(X), X[:, 0]], axis=1), [0.0, 0.0],
                                                        [1.0, 1.0], rule="genz-malik").summary()  # fmt: skip
    yield "err cubature lengths", lambda: I.cubature(gauss, [0.0], [1.0, 2.0])
    yield "err cubature alias", lambda: I.cubature(gauss, [0.0, 0.0], [1.0, 1.0], rule="gauss-kronrod")
    yield "err cubature genz-malik 1-D", lambda: I.cubature(gauss, [0.0], [1.0], rule="genz-malik")
    yield "err cubature max_subdivisions", lambda: I.cubature(gauss, [0.0], [1.0], max_subdivisions=0)
    yield "err cubature same infinity", lambda: I.cubature(gauss, [inf], [inf])
    yield "err cubature point length", lambda: I.cubature(gauss, [0.0], [1.0], points=[[0.5, 0.5]])
    yield "err cubature nan point", lambda: I.cubature(gauss, [0.0, 0.0], [1.0, 1.0], points=[[0.5, math.nan]])
    yield "err cubature value", lambda: I.cubature(lambda X: 1.0, [0.0], [1.0])
    yield "err cubature value shape", lambda: I.cubature(lambda X: X[:3, 0], [0.0], [1.0])
    yield "err cubature args", lambda: I.cubature(gauss, [0.0], [1.0], args=())
