"""The cases of quad_vec and cubature (milestone C7; docs/design/integrate.md §3.3, §3.4, §7.13):
the live comparisons with scipy 1.18.1 outside the records, the collection's runs measured against
the Arb references, and the runs whose error estimate is short of the true error.

QUAD_VEC_LIVE and CUBATURE_LIVE are (name, f, a, b, options) with f on Python floats (quad_vec) or
numpy arrays (cubature). The honesty runs (quad_vec_runs, cubature_runs) are the collection's
integrals without a weight (quad_vec on Python floats with quad's break points; cubature in one
variable with the break points as points, rules gk21 and gk15), the three book families as one
vector-valued integrand each (quad_vec with both norms; cubature with both rules, element by
element), the vector-valued pair of problems.VECTOR and the boxes of problems.MULTI (every rule).
The contract of a converged run (bench/accuracy/integrate.py, test_quad_vec.py, test_cubature.py):
the true error within the error estimate plus the value's rounding, |value - ref| <= error +
8 eps |ref| (cubature's |K - G| estimate has no rounding floor), but for the runs of the SHORT sets,
whose estimate misses (scipy's too: the same bits).
"""

import math

import numpy as np

from . import problems as P
from . import refdata

INF = math.inf
EPS = 2.0**-52


def _nan_safe(p):
    def f(x):
        try:
            return p.f(x)
        except ZeroDivisionError, ValueError, OverflowError:
            return math.nan

    return f


QUAD_VEC_LIVE = [
    ("scalar", lambda x: math.exp(-x * x), 0.0, 3.0, {}),
    ("scalar real line", lambda x: math.exp(-x * x), -INF, INF, {}),
    ("half line", lambda x: math.exp(-x), 0.0, INF, {}),
    ("left half line", lambda x: math.exp(x), -INF, 0.0, {}),
    ("reversed half line", lambda x: math.exp(-x), INF, 1.0, {}),
    ("reversed real line", lambda x: np.array([math.exp(-x * x), 1 / (1 + x * x)]), INF, -INF, {}),
    ("same infinities", lambda x: np.array([x, 1.0]), INF, INF, {}),
    ("empty range", lambda x: np.array([x, 1.0]), 2.0, 2.0, {}),
    ("reversed", lambda x: x, 2.0, 1.0, {}),
    ("peaks", lambda x: np.array([1 / (1e-4 + (x - 0.3) ** 2), math.sqrt(abs(x - 0.7))]), 0.0, 1.0, {}),
    ("peaks max", lambda x: np.array([1 / (1e-4 + (x - 0.3) ** 2), math.sqrt(abs(x - 0.7))]), 0.0, 1.0,
     {"norm": "max"}),
    ("trapezoid", lambda x: np.array([math.exp(x), math.cos(3 * x)]), 0.0, 2.0, {"quadrature": "trapezoid"}),
    ("trapezoid scalar", lambda x: math.sqrt(x), 0.0, 1.0, {"quadrature": "trapezoid", "rtol": 1e-6}),
    ("trapezoid real line", lambda x: math.exp(-x * x), -INF, INF, {"quadrature": "trapezoid", "rtol": 1e-6}),
    ("gk15", lambda x: np.array([math.exp(x), math.cos(30 * x)]), 0.0, 2.0, {"quadrature": "gk15"}),
    ("gk21 half line", lambda x: math.exp(-x) * math.cos(x), 0.0, INF, {"quadrature": "gk21"}),
    ("points", lambda x: np.array([abs(x - 1 / 3), abs(x - 0.5) ** 0.5]), 0.0, 1.0, {"points": [1 / 3, 0.5, 2.0, 0.5]}),
    ("points real line", lambda x: np.array([math.exp(-abs(x - 1)), 1.0 / (1 + x**4)]), -INF, INF,
     {"points": [1.0, -2.0, 0.0]}),
    ("points half line", lambda x: math.exp(-abs(x - 1)), 0.0, INF, {"points": [1.0]}),
    ("limit 40", lambda x: math.sin(1 / x) if x else 0.0, 0.0, 1.0, {"limit": 40}),
    ("limit 1", lambda x: math.exp(x), 0.0, 1.0, {"limit": 1}),
    ("tolerances 0", lambda x: np.array([math.exp(x), x**0.5]), 0.0, 1.0, {"rtol": 0.0, "atol": 0.0, "limit": 300}),
    ("atol", lambda x: np.array([math.exp(x), x**0.5]), 0.0, 1.0, {"rtol": 0.0, "atol": 1e-6}),
    ("NaN", lambda x: np.array([math.log(x - 0.5) if x > 0.5 else math.nan, 1.0]), 0.0, 1.0, {}),
    ("2-D value", lambda x: np.array([[math.exp(x), x], [x * x, math.cos(x)]]), 0.0, 1.0, {}),
    ("1000 values", lambda x: np.cos(np.arange(1000) * x) * math.exp(-x), 0.0, 3.0, {}),
    ("oscillation", lambda x: np.array([math.cos(200 * x), math.sin(150 * x)]), 0.0, 3.0, {}),
    ("rounding", lambda x: np.array([1.0 + 1e-17 * x, 2.0]), 0.0, 1.0, {"rtol": 1e-17, "atol": 0.0}),
    ("int values", lambda x: np.array([1, 2]), 0.0, 1.0, {}),
    ("numpy scalar", lambda x: np.float64(math.exp(x)), 0.0, 1.0, {}),
]  # fmt: skip


def _g(X):
    return np.exp(-np.sum(X * X, axis=-1))


def _vec(X):
    return np.stack([np.exp(-np.sum(X * X, axis=-1)), np.cos(np.sum(X, axis=-1)), X[:, 0] ** 2], axis=1)


def _vec_f(X):
    return np.array([np.exp(-np.sum(X * X, axis=-1)), np.cos(np.sum(X, axis=-1)), X[:, 0] ** 2]).T


def _sing(X):
    return 1 / np.sqrt(np.abs(X[:, 0] - 0.3) + np.abs(X[:, 1] - 0.6))


def _four(X):
    return np.stack(
        [np.exp(-np.sum(X * X, axis=-1)), np.cos(np.sum(X, axis=-1)), X[:, 0] ** 2, np.sin(X[:, 1])], axis=1
    )


CUBATURE_LIVE = [
    ("gauss gk21 2-D", _g, [0, 0], [1, 1], {}),
    ("gauss gk15 3-D", _g, [0, 0, 0], [1, 1, 2], {"rule": "gk15"}),
    ("gauss genz-malik 3-D", _g, [0, 0, 0], [1, 1, 2], {"rule": "genz-malik"}),
    ("gauss genz-malik 4-D", _g, [0, 0, 0, 0], [1, 1, 1, 1], {"rule": "genz-malik", "rtol": 1e-6}),
    ("gauss 1-D", _g, [0], [3], {}),
    ("vector C order", _vec, [0, 0], [2, 1], {"rule": "genz-malik"}),
    ("vector Fortran order", _vec_f, [0, 0], [2, 1], {"rule": "genz-malik"}),
    ("vector C order gk15", _vec, [0, 0], [2, 1], {"rule": "gk15"}),
    ("vector Fortran order gk15", _vec_f, [0, 0], [2, 1], {"rule": "gk15"}),
    ("one column", lambda X: np.exp(-X[:, :1] ** 2), [0, 0], [2, 1], {"rule": "genz-malik"}),
    ("3-D value", lambda X: _g(X)[:, None, None] * np.arange(1, 7).reshape(1, 2, 3), [0, 0], [2, 1],
     {"rule": "genz-malik"}),
    ("elementwise", lambda X: np.exp(-X * X) * np.cos(3 * X), [0, 0], [2, 1], {"rule": "genz-malik"}),
    ("elementwise gk15", lambda X: np.exp(-X * X) * np.cos(3 * X), [0, 0], [2, 1], {"rule": "gk15"}),
    ("plane", _g, [-INF, -INF], [INF, INF], {}),
    ("plane genz-malik", _g, [-INF, -INF], [INF, INF], {"rule": "genz-malik"}),
    ("half plane", _g, [0, -1], [INF, 1], {"rule": "genz-malik"}),
    ("half line", lambda X: np.exp(-X[:, 0]), [1.0], [INF], {}),
    ("real line", _g, [-INF], [INF], {}),
    ("flipped", _g, [1, 0], [0, 1], {"rule": "genz-malik"}),
    ("both flipped", _vec, [1, 2], [0, 0], {"rule": "genz-malik"}),
    ("flipped infinite", _g, [INF, 0], [-INF, 1], {}),
    ("points", _sing, [0, 0], [1, 1], {"rule": "genz-malik", "points": [[0.3, 0.6], [0.5, 0.5], [0.0, 0.2]],
                                        "max_subdivisions": 200}),
    ("point 1-D", lambda X: np.sin(X[:, 0]) / X[:, 0], [-1], [1], {"points": [[0.0]]}),
    ("point real line", lambda X: np.exp(-X[:, 0] ** 2), [-INF], [INF], {"points": [[1.0]]}),
    ("point half line", lambda X: np.exp(-X[:, 0]), [0.0], [INF], {"points": [[1.0]]}),
    ("point plane strip", _g, [-INF, -1], [INF, 1], {"rule": "genz-malik", "points": [[1.0, 0.5]]}),
    ("point on an edge", _sing, [0, 0], [1, 1], {"rule": "genz-malik", "points": [[0.0, 0.6]], "max_subdivisions": 50}),
    ("max_subdivisions", _sing, [0, 0], [1, 1], {"rule": "genz-malik", "max_subdivisions": 7}),
    ("zero width", _g, [1, 0], [1, 1], {"rule": "genz-malik"}),
    ("zero width split", _g, [1.0, 0], [1.0, 1], {"points": [[1.0, 0.5]]}),
    ("atol", _g, [0, 0], [5, 5], {"rule": "genz-malik", "atol": 1e-3, "rtol": 0.0}),
    ("peak", lambda X: 1 / (1e-3 + np.sum((X - 0.37) ** 2, axis=-1)), [0, 0], [1, 1], {"rule": "gk15", "rtol": 1e-6}),
    ("reversed columns", lambda X: _four(X)[:, ::-1], [0, 0], [2, 1], {"rule": "genz-malik", "max_subdivisions": 20}),
    ("strided rows", lambda X: np.repeat(_four(X), 2, axis=0)[::2], [0, 0], [2, 1], {"rule": "gk15",
                                                                                      "max_subdivisions": 20}),
    ("broadcast", lambda X: np.broadcast_to(np.exp(-X[:, :1] ** 2), (X.shape[0], 3)), [0, 0], [2, 1],
     {"rule": "genz-malik", "max_subdivisions": 20}),
    ("Fortran 3-D", lambda X: np.asfortranarray(_four(X).reshape(-1, 2, 2)), [0, 0], [2, 1],
     {"rule": "genz-malik", "max_subdivisions": 20}),
    ("transposed 3-D", lambda X: np.transpose(_four(X).reshape(-1, 2, 2), (0, 2, 1)), [0, 0], [2, 1],
     {"rule": "gk15", "max_subdivisions": 20}),
    ("int values", lambda X: (X[:, :2] * 10).astype(np.int64), [0, 0], [2, 1], {"rule": "genz-malik",
                                                                               "max_subdivisions": 20}),
    ("the argument", lambda X: X, [0, 0, 0], [2, 1, 1], {"rule": "genz-malik", "max_subdivisions": 20}),
]  # fmt: skip

# The converged runs whose true error is past error + 8 eps |ref| (the same in scipy)
QUAD_VEC_SHORT = {"book/peak/15", "book/peak/16", "book/peak/17", "book/peak/18", "scipy/tanhsinh/f10"}
CUBATURE_SHORT = {
    "gk21": {"book/xlog/-0.9", "book/xlog/-0.8", "book/xlog/-0.7", "book/peak/17", "book/peak/19",
             "scipy/tanhsinh/f12", "hard/pow-0.9", "hard/interior/-0.50"},
    "gk15": {"book/xlog/-0.9", "book/xlog/-0.8", "book/xlog/-0.7", "book/peak/17", "book/peak/18", "hard/pow-0.9"},
    "batch gk21": {"book/xlog/-0.9", "book/peak/14", "book/peak/19"},
    "batch gk15": {"book/xlog/-0.9", "book/peak/15", "book/peak/16", "book/peak/17", "book/peak/18"},
}  # fmt: skip


def _err(v, r):
    hi, lo = r
    return abs((v - hi) - lo)


def quad_vec_runs(wrap):
    """quad_vec of a library (wrap(f, a, b, **kw) -> (value, error,
    status)) on the collection: [(region, id, value(s), error, status, refs)]."""
    ref = refdata.integrals()
    out = []
    for p in P.INTEGRALS:
        if p.weight is not None:
            continue
        kw = {"points": list(p.points)} if p.points else {}
        v, e, s = wrap(_nan_safe(p), p.a, p.b, **kw)
        out.append(("collection", p.id, [v], e, s, [ref[p.id]["value"]]))
    for bt in P.BATCHES:
        params = np.array(bt.params)
        refs = [ref[i]["value"] for i in bt.ids()]
        for norm in ("2", "max"):

            def f(x, bt=bt):
                with np.errstate(all="ignore"):
                    return bt.f_np(np.full(len(params), x), params)

            v, e, s = wrap(f, bt.a, bt.b, norm=norm)
            out.append((f"families, norm {norm}", bt.family, list(v), e, s, refs))
    vref = refdata.vector()
    for vv in P.VECTOR:
        v, e, s = wrap(vv.func, vv.a, vv.b)
        out.append(("vector", vv.id, list(v), e, s, vref[vv.id]["value"]))
    return out


def quad_vec_contract(run):
    """(true error / (error + 8 eps |ref|) in the run's norm, short): the 2-norm of the elements'
    errors (the max-norm for a max-norm run)."""
    region, pid, v, e, s, refs = run
    errs = [_err(x, r) for x, r in zip(v, refs)]
    mags = [abs(r[0]) for r in refs]
    if "max" in region:
        true, size = max(errs), max(mags)
    else:
        true, size = math.sqrt(sum(t * t for t in errs)), math.sqrt(sum(t * t for t in mags))
    bound = e + 8 * EPS * size
    return (true / bound if bound else (0.0 if true == 0 else math.inf)), pid in QUAD_VEC_SHORT


def cubature_runs(cub):
    """cubature of a library (cub(f, a, b, **kw) -> (values, errors, status), f on numpy arrays) on
    the collection, one row per value: [(region, id, value, error, status, ref)]."""
    ref = refdata.integrals()
    out = []
    for rule in ("gk21", "gk15"):
        for p in P.INTEGRALS:
            if p.weight is not None:
                continue
            kw = {"points": [[t] for t in p.points]} if p.points else {}

            def f(X, p=p):
                with np.errstate(all="ignore"):
                    return p.f_np(X[:, 0])

            v, e, s = cub(f, [p.a], [p.b], rule=rule, **kw)
            out.append((rule, p.id, v[0], e[0], s, ref[p.id]["value"]))
        for bt in P.BATCHES:
            params = np.array(bt.params)

            def f(X, bt=bt):
                with np.errstate(all="ignore"):
                    return bt.f_np(X[:, :1], params[None, :])

            v, e, s = cub(f, [bt.a], [bt.b], rule=rule)
            for i, pid in enumerate(bt.ids()):
                out.append((f"batch {rule}", pid, v[i], e[i], s, ref[pid]["value"]))
    mref = refdata.multiple()
    for m in P.MULTI:
        if m.box is None:
            continue
        for rule in ("gk21", "gk15", "genz-malik"):

            def f(X, m=m):
                return m.func(*[X[:, i] for i in range(X.shape[1])])

            v, e, s = cub(f, [b[0] for b in m.box], [b[1] for b in m.box], rule=rule)
            out.append((f"multiple {rule}", m.id, v[0], e[0], s, mref[m.id]["value"]))
    return out


def cubature_contract(run):
    region, pid, v, e, s, r = run
    true = _err(v, r)
    bound = e + 8 * EPS * abs(r[0])
    return (true / bound if bound else (0.0 if true == 0 else math.inf)), pid in CUBATURE_SHORT.get(region, ())
