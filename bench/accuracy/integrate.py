"""sd.integrate's accuracy sweeps (docs/design/integrate.md §4, §7.5): one per function family on
the reference families of tests/integrate/ref (tools/integrate_ref.py).

Since milestone C3:
  samples     the sample rules on the sample sets of tests/integrate/problems.py against each
              rule's formula computed exactly from the same doubles (ref/samples.json). Contract:
              the error within c eps S, S the terms' size (sum |d_i| (|y_i| + |y_i+1|)), c =
              ceil(log2 n) + 10 for a sum (the pairwise tree), k + 10 for the k-th running value
              (a scan); bound 1. The relative error is recorded per rule.
  fixed_quad  n-point Gauss-Legendre (n = 5, 20, 100) on the collection's finite smooth integrals
              without a weight, against the Arb integral (ref/integrals.json): recorded only (the
              rule's own error, not a contract). Its nodes and weights are correctly rounded
              (tests/integrate/test_fixed.py) and its sum is within the pairwise bound.
Since milestone C4:
  quad        quad (QAGS, QAGI, QAGP) on the collection's integrals without a weight, one region
              per collection (book, quadpack, scipy, hard), against the Arb integral: the relative
              error of every run, and the contract of a converged run: the true error within the
              error estimate (|value - ref| / error <= 1), but for the three runs where scipy's
              QUADPACK misses too (tests/integrate/test_reference_data.QUAD_SHORT: the narrow peak
              a = 19, 20 that the first rule does not see, f10's extrapolation to the exact pi/2),
              which are recorded as "short" and left out of the contract; "converged" counts.
  multiple    dblquad and tplquad on problems.MULTI against the Arb integral (relative error;
              the contract as quad's, with the error nquad gives: the largest of every quad's).
Since milestone C5:
  quad_weights  quad with a weight (QAWO, QAWF, QAWS, QAWC), one region per routine, on the
              collection's ten weighted integrals and the sweep of tests/integrate/weights.py (99,
              ref/weights.json), against the Arb integral: the relative error, and the contract of a
              converged run: the true error within the error estimate plus the value's rounding,
              |value - ref| / (error + 8 eps |ref|) <= 1 (a Clenshaw-Curtis estimate has no rounding
              floor: it is 0 on a polynomial of QAWC, and dqc25f's moments are good to about 2e3
              ulp); "short" counts the runs past the estimate alone, "converged" the converged.
scipy.integrate 1.18.1 is the comparator: the same formulas (its sums in numpy's order) and its
roots_legendre; its QUADPACK for quad (the same steps: equal to the bit, tests/integrate/
test_quadpack.py).

Since milestone C6:
  tanhsinh    tanhsinh on the collection's integrals of its records (problems.Integral.tanhsinh),
              one region per collection, against the Arb integral: the relative error, and the
              contract of a converged run as quad's (the true error within the error estimate), but
              for the eight runs where scipy's tanhsinh is short too (test_reference_data.
              TANHSINH_SHORT: the moderate peaks a = 0, 2, 4, cos(2^a sin x) a = 6, 7, 9, 10 and f9's
              sliver past the double pi/2), recorded as "short"; "converged" counts.
  nsum        nsum on problems.SERIES against the Arb sums: the relative error and the same contract.
scipy.integrate's tanhsinh and nsum are the comparator (the same steps: equal to the bit,
tests/integrate/test_tanhsinh.py).

Since milestone C7:
  quad_vec    quad_vec on the collection's integrals without a weight (Python floats, quad's break
              points), the three book families as one vector-valued integrand each (norms '2' and
              'max') and problems.VECTOR (tests/integrate/adaptive.quad_vec_runs), against the Arb
              integrals: the relative error (of the run's norm), and the contract of a converged run:
              the true error within the error estimate plus the value's rounding, |value - ref| <=
              error + 8 eps |ref| in the run's norm, but for adaptive.QUAD_VEC_SHORT (scipy's too),
              recorded as "short"; "converged" counts.
  cubature    cubature on the collection's integrals in one variable and on the book families as
              vector values (rules gk21, gk15) and on the boxes of problems.MULTI (every rule),
              element by element: the relative error and the same contract (cubature's |K - G|
              estimate has no rounding floor), but for adaptive.CUBATURE_SHORT.
scipy.integrate's quad_vec and cubature are the comparator (the same steps: equal to the bit,
tests/integrate/test_quad_vec.py, test_cubature.py; cubature but on (-inf, b], where scipy's
transform integrates over [-b, inf)).
"""

from __future__ import annotations

import math

import numpy as np

from .common import Custom, compare_libs, region

EPS = 2.0**-52
RULES = ("trapezoid", "simpson", "romb", "cumulative_trapezoid", "cumulative_simpson")


def _lib(lib):
    if lib == "sd":
        import stardust as sd

        return sd.integrate, sd.asarray, lambda v: np.asarray(v.tolist() if isinstance(v, sd.ndarray) else v)
    if lib == "scipy":
        import scipy.integrate as si

        return si, np.asarray, np.asarray
    return None


def _samples(lib="sd"):
    m = _lib(lib)
    if m is None:
        return None
    I, wrap, back = m
    from tests.integrate import problems as P
    from tests.integrate import refdata

    ref = refdata.samples()
    out = {}
    for rule in RULES:
        where, rel, contract = [], [], []
        for sid, (y, x, dx) in P.samples().items():
            e = ref[sid]
            if rule not in e:
                continue
            kw = {"x": wrap(x)} if x is not None else {"dx": dx}
            got = back(getattr(I, rule)(wrap(y), **kw)).astype(float)
            r = np.asarray(e[rule], dtype=float)
            err = np.abs((got - r[..., 0]) - r[..., 1])
            d = np.abs(np.diff(x)) if x is not None else np.full(y.shape[-1] - 1, abs(dx))
            a = np.abs(y)
            scale = np.maximum(np.cumsum(d * (a[..., 1:] + a[..., :-1]), axis=-1), 1e-300)
            if rule.startswith("cumulative"):
                bound = (np.arange(1, err.shape[-1] + 1) + 10) * EPS * scale
            else:
                bound = (math.ceil(math.log2(max(y.shape[-1], 2))) + 10) * EPS * scale[..., -1]
            where.append(sid)
            rel.append(float(np.max(err / np.maximum(np.abs(r[..., 0]), 1e-300))))
            contract.append(float(np.max(err / bound)))
        out[rule] = region(where, rel=np.array(rel), contract=np.array(contract), bound=1.0)
    return out


def _fixed_quad(lib="sd"):
    m = _lib(lib)
    if m is None:
        return None
    I, wrap, _ = m
    from tests.integrate import problems as P
    from tests.integrate import refdata

    ref = refdata.integrals()
    out = {}
    for n in (5, 20, 100):
        where, rel = [], []
        for p in P.INTEGRALS:
            if not (p.numeric and p.weight is None and math.isfinite(p.a) and math.isfinite(p.b)):
                continue
            with np.errstate(all="ignore"):
                v = I.fixed_quad(lambda t, p=p: wrap(p.f_np(np.asarray(t.tolist() if hasattr(t, "tolist") else t))),
                                 p.a, p.b, n=n)  # fmt: skip
            v = float(v[0] if isinstance(v, tuple) else v)
            hi, lo = ref[p.id]["value"]
            err = abs((v - hi) - lo)
            where.append(p.id)
            rel.append(err / abs(hi) if hi else err)
        out[f"n={n}"] = region(where, rel=np.array(rel))
    return out


def _quad_lib(lib):
    if lib == "sd":
        import stardust as sd

        def q(f, a, b, **kw):
            r = sd.integrate.quad(f, a, b, **kw)
            return r.integral, r.error, r.status

        return q
    if lib == "scipy":
        import warnings

        from scipy import integrate

        def q(f, a, b, **kw):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = integrate.quad(f, a, b, full_output=1, **kw)
            return r[0], r[1], 0 if len(r) == 3 else 1

        return q
    return None


def _quad(lib="sd"):
    q = _quad_lib(lib)
    if q is None:
        return None
    from tests.integrate import problems as P
    from tests.integrate import refdata
    from tests.integrate.test_reference_data import QUAD_SHORT

    ref = refdata.integrals()
    out = {}
    for coll in ("book", "quadpack", "scipy", "hard"):
        where, rel, contract, short, conv = [], [], [], 0, 0
        for p in P.INTEGRALS:
            if p.weight is not None or p.collection != coll:
                continue
            kw = {"points": list(p.points)} if p.points else {}
            v, e, status = q(p.f, p.a, p.b, **kw)
            hi, lo = ref[p.id]["value"]
            err = abs((v - hi) - lo)
            where.append(p.id)
            rel.append(err / abs(hi) if hi else err)
            if status == 0:
                conv += 1
                if p.id in QUAD_SHORT:
                    short += 1
                    contract.append(0.0)
                else:
                    contract.append(err / e if e > 0 else (0.0 if err == 0 else math.inf))
            else:
                contract.append(0.0)
        out[coll] = region(where, rel=np.array(rel), contract=np.array(contract), bound=1.0, converged=np.array([conv]),
                           short=np.array([short]))  # fmt: skip
    return out


def _multiple(lib="sd"):
    if lib == "sd":
        import stardust as sd

        I = sd.integrate  # noqa: E741
    elif lib == "scipy":
        import scipy.integrate as I  # noqa: N812
    else:
        return None
    from tests.integrate import problems as P
    from tests.integrate import refdata

    ref = refdata.multiple()
    where, rel, contract = [], [], []
    for m in P.MULTI:
        f = (lambda m: lambda *xs: float(m.func(*xs)))(m)
        v, e = tuple(getattr(I, m.kind)(f, m.a, m.b, *m.limits))[:2]
        hi, lo = ref[m.id]["value"]
        err = abs((v - hi) - lo)
        where.append(m.id)
        rel.append(err / abs(hi))
        contract.append(err / e if e > 0 else (0.0 if err == 0 else math.inf))
    return {"dblquad, tplquad": region(where, rel=np.array(rel), contract=np.array(contract), bound=1.0)}


def _elementwise_lib(lib):
    """tanhsinh(f, a, b) and nsum(f, a, b, step) of a library on a numpy integrand: (value, error,
    status)."""
    if lib == "sd":
        import stardust as sd

        def wrap(g):
            def f(x):
                with np.errstate(all="ignore"):
                    return sd.asarray(g(np.asarray(x)))

            return f

        def t(g, a, b):
            r = sd.integrate.tanhsinh(wrap(g), a, b)
            return r.integral, r.error, r.status

        def n(g, a, b, step):
            r = sd.integrate.nsum(wrap(g), a, b, step=step)
            return r.sum, r.error, r.status

        return t, n
    if lib == "scipy":
        import warnings

        from scipy import integrate

        def quiet(call):
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                return call()

        def t(g, a, b):
            r = quiet(lambda: integrate.tanhsinh(g, a, b))
            return float(r.integral), float(r.error), int(r.status)

        def n(g, a, b, step):
            r = quiet(lambda: integrate.nsum(g, a, b, step=step))
            return float(r.sum), float(r.error), int(r.status)

        return t, n
    return None


def _contract_region(where, runs, short_ids):
    """region() of (id, value, error, status, [hi, lo]) runs: the relative error and the contract
    of the converged ones (true error / error estimate) but for short_ids (recorded as "short")."""
    rel, contract, short, conv = [], [], 0, 0
    for pid, v, e, status, (hi, lo) in runs:
        err = abs((v - hi) - lo)
        rel.append(err / abs(hi) if hi else err)
        if status == 0:
            conv += 1
            if pid in short_ids:
                short += 1
                contract.append(0.0)
            else:
                contract.append(err / e if e > 0 else (0.0 if err == 0 else math.inf))
        else:
            contract.append(0.0)
    return region(where, rel=np.array(rel), contract=np.array(contract), bound=1.0, converged=np.array([conv]),
                  short=np.array([short]))  # fmt: skip


def _tanhsinh(lib="sd"):
    fns = _elementwise_lib(lib)
    if fns is None:
        return None
    from tests.integrate import problems as P
    from tests.integrate import refdata
    from tests.integrate.test_reference_data import TANHSINH_SHORT

    ref = refdata.integrals()
    out = {}
    for coll in ("book", "quadpack", "scipy", "hard"):
        ps = [p for p in P.INTEGRALS if p.tanhsinh and p.collection == coll]
        runs = [(p.id, *fns[0](p.f_np, p.a, p.b), ref[p.id]["value"]) for p in ps]
        out[coll] = _contract_region([p.id for p in ps], runs, TANHSINH_SHORT)
    return out


def _nsum(lib="sd"):
    fns = _elementwise_lib(lib)
    if fns is None:
        return None
    from tests.integrate import problems as P
    from tests.integrate import refdata

    ref = refdata.series()
    runs = [(s.id, *fns[1](s.f_np, s.a, s.b, s.step), ref[s.id]["value"]) for s in P.SERIES]
    return {"series": _contract_region([s.id for s in P.SERIES], runs, set())}


def _quad_weights(lib="sd"):
    q = _quad_lib(lib)
    if q is None:
        return None
    from tests.integrate import problems as P
    from tests.integrate import refdata
    from tests.integrate import weights as W

    refs = {**refdata.integrals(), **refdata.weights()}
    runs = [p for p in P.INTEGRALS if p.weight is not None] + list(W.SWEEP)
    out = {}
    for routine in ("qawo", "qawf", "qaws", "qawc"):
        where, rel, contract, short, conv = [], [], [], 0, 0
        for p in runs:
            if W.routine(p) != routine:
                continue
            v, e, status = q(p.f, p.a, p.b, weight=p.weight, wvar=p.wvar)
            hi, lo = refs[p.id]["value"]
            err = abs((v - hi) - lo)
            where.append(p.id)
            rel.append(err / abs(hi) if hi else err)
            if status == 0:
                conv += 1
                short += err > e
                contract.append(err / (e + 8 * EPS * abs(hi)) if e or hi else (0.0 if err == 0 else math.inf))
            else:
                contract.append(0.0)
        out[routine] = region(where, rel=np.array(rel), contract=np.array(contract), bound=1.0,
                              converged=np.array([conv]), short=np.array([short]))  # fmt: skip
    return out


def _adaptive_libs(lib):
    """(quad_vec wrap, cubature wrap) of a library for the runners of tests/integrate/adaptive.py."""
    if lib == "sd":
        import stardust as sd

        def qv(f, a, b, **kw):
            r = sd.integrate.quad_vec(f, a, b, **kw)
            v = r.integral.tolist() if isinstance(r.integral, sd.ndarray) else r.integral
            return v, r.error, r.status

        def cub(f, a, b, **kw):
            r = sd.integrate.cubature(lambda X: sd.asarray(f(np.asarray(X.tolist()))), a, b, **kw)
            v = r.integral.tolist() if isinstance(r.integral, sd.ndarray) else [r.integral]
            e = r.error.tolist() if isinstance(r.error, sd.ndarray) else [r.error]
            return v, e, r.status

        return qv, cub
    if lib == "scipy":
        import warnings

        from scipy import integrate

        def qv(f, a, b, **kw):
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                y, err, info = integrate.quad_vec(f, a, b, full_output=True, **kw)
            return (np.asarray(y).tolist() if np.ndim(y) else float(y)), float(err), int(info.status)

        def cub(f, a, b, **kw):
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                r = integrate.cubature(f, np.asarray(a, dtype=float), np.asarray(b, dtype=float), **kw)
            ok = r.status == "converged" and bool(np.all(np.isfinite(r.estimate)))
            return np.atleast_1d(r.estimate).tolist(), np.atleast_1d(r.error).tolist(), 0 if ok else 1

        return qv, cub
    return None


def _adaptive_region(rows, contract):
    out = {}
    for row in rows:
        out.setdefault(row[0], []).append(row)
    regions = {}
    for name, rs in out.items():
        rel, con, short, conv = [], [], 0, 0
        for row in rs:
            ratio, known = contract(row)
            vals = row[2] if isinstance(row[2], list) else [row[2]]
            refs = row[5] if isinstance(row[2], list) else [row[5]]
            rel.append(max(abs((v - r[0]) - r[1]) / abs(r[0]) if r[0] else abs(v - r[0]) for v, r in zip(vals, refs)))
            if row[4] == 0:
                conv += 1
                if known:
                    short += 1
                    con.append(0.0)
                else:
                    con.append(ratio)
            else:
                con.append(0.0)
        regions[name] = region([r[1] for r in rs], rel=np.array(rel), contract=np.array(con), bound=1.0,
                               converged=np.array([conv]), short=np.array([short]))  # fmt: skip
    return regions


def _quad_vec(lib="sd"):
    fns = _adaptive_libs(lib)
    if fns is None:
        return None
    from tests.integrate import adaptive as A

    return _adaptive_region(A.quad_vec_runs(fns[0]), A.quad_vec_contract)


def _cubature(lib="sd"):
    fns = _adaptive_libs(lib)
    if fns is None:
        return None
    from tests.integrate import adaptive as A

    return _adaptive_region(A.cubature_runs(fns[1]), A.cubature_contract)


def sweeps():
    specs = [("samples", _samples), ("fixed_quad", _fixed_quad), ("quad", _quad), ("multiple", _multiple),
             ("quad_weights", _quad_weights), ("tanhsinh", _tanhsinh), ("nsum", _nsum), ("quad_vec", _quad_vec), ("cubature", _cubature)]  # fmt: skip
    return [Custom("integrate", name, fn, compare_libs(fn, ("scipy",), seconds=120)) for name, fn in specs]
