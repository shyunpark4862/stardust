"""The optimize area: what an iterative method's result can be checked against.

Contract (docs/design/optimize.md §4: the stopping criterion's own meaning), only where the
result alone can check it:
  * the bracketing root finders at the default tolerances: the root is within
    xtol + rtol |x*| of the Arb root (tests/optimize/data/scalar_roots.json, tools/optimize_ref.py),
    or f is exactly 0 there in double (tests/optimize/test_reference_data.py);
  * the gradient methods that report convergence by the gradient test: ||grad f(x)||_inf <= gtol
    (BFGS, trust-ncg, trust-exact; L-BFGS-B when it stopped on the projected gradient), with
    the exact gradient of the MGH problem (tests/optimize/mgh.py).
Recorded only (the baseline judges): the root's distance to the Arb root in ULP at xtol = 0
and of find_root; the gradient and f - f* (MGH's KNOWN optima) of every minimize method; the
parameters' relative error against NIST StRD's certified values (least_squares); ||F||_inf at
the result (root); the KKT residual (lsq_linear, nnls); approx_fprime's error against the
Arb derivatives of tests/optimize/ref/derivatives.npz; and how many runs did not converge.
Every reference is a file; at measurement time only residuals at sd's own results (the
gradient, F, the KKT conditions) are evaluated, in double, from the problems' formulas.
"""

from __future__ import annotations

import math

import numpy as np

from .common import Custom, compare_libs, region

EPS = 2.0**-52
BRACKETING = ("brentq", "toms748", "bisect")
TOL0 = {  # xtol = 0: as far as rtol and max_iter let the method go (docs/api/optimize.md §2)
    "brentq": {"xtol": 0.0, "max_iter": 200},
    "toms748": {"xtol": 0.0, "rtol": 2.0**-52, "max_iter": 200},  # its smallest rtol
    "bisect": {"xtol": 0.0, "max_iter": 1100},
}


class _Res:
    """x, converged and message, whatever the library calls them."""

    def __init__(self, r, lib):
        self.r = r
        if lib == "sd":
            self.x, self.converged, self.message = r.x, r.converged, getattr(r, "message", "")
        elif hasattr(r, "root"):  # scipy's RootResults
            self.x, self.converged, self.message = r.root, r.converged, r.flag
        else:
            self.x, self.converged, self.message = r.x, r.success, getattr(r, "message", "")
        self.active_mask = getattr(r, "active_mask", None)


def _opt(lib):
    if lib == "sd":
        import stardust as sd

        return sd.optimize
    import scipy.optimize as so

    return so


def _arb_roots():
    import json
    import pathlib

    from tests.optimize import functions

    p = pathlib.Path(functions.__file__).parent / "data" / "scalar_roots.json"
    ref = json.loads(p.read_text())
    out = {}
    for c in functions.ROOT_CASES:
        r = ref[c.id]
        out[c.id] = (c, *((r["x"], r["x"]) if r["kind"] == "exact" else (r["a"], r["b"])))
    return out


def _dist_ulp(x, lo, hi):
    """|x - [lo, hi]| in ULP of the root (0 inside)."""
    if lo <= x <= hi:
        return 0.0, 0.0
    d = min(abs(x - lo), abs(x - hi))
    return d, d / math.ulp(max(abs(lo), abs(hi), 2.0**-1074))


def _zero_in_double(c, x):
    with np.errstate(all="ignore"):
        return float(c.f(np.float64(x))) == 0.0


def _root_scalar(method, lib="sd"):
    O = _opt(lib)

    out = {}
    for label, kw in (("default", {}), ("xtol=0", TOL0[method])):
        if label == "xtol=0" and lib != "sd":  # scipy's root_scalar takes no xtol = 0 (it must be > 0)
            continue
        where, ulp, ratio, unconv, zwhere, zabs = [], [], [], 0, [], []
        for cid, (c, lo, hi) in _arb_roots().items():
            with np.errstate(all="ignore"):
                try:
                    kw_ = kw if lib == "sd" else {{"max_iter": "maxiter"}.get(k, k): v for k, v in kw.items()}
                    r = O.root_scalar(lambda x, c=c: c.f(np.float64(x)), method=method, bracket=c.bracket, **kw_)
                    r = _Res(r, lib)
                except ValueError, ArithmeticError:
                    unconv += 1
                    continue
            if not r.converged:
                unconv += 1
                continue
            x = float(r.x)
            d, u = _dist_ulp(x, lo, hi)
            zero = _zero_in_double(c, x)
            if label == "default":
                where.append(cid)
                ratio.append(0.0 if zero else d / (2e-12 + 4 * 2.220446049250313e-16 * abs(x)))
            elif lo == hi == 0.0:  # a root at 0 has no ULP: its absolute distance
                zwhere.append(cid)
                zabs.append(0.0 if zero else d)
            else:
                where.append(cid)
                ulp.append(0.0 if zero else u)
        if label == "default":
            e = region(where, contract=np.array(ratio), bound=1.0)
        else:
            e = region(where, ulp=np.array(ulp))
        e["unconverged"] = unconv
        out[label] = e
        if zwhere:
            out[f"{label}, root 0"] = region(zwhere, abs_err=np.array(zabs))
    return out


def _vec(c, lib="sd"):
    """c's function on an sd.ndarray, element by element in doubles (tests/optimize/test_elementwise.py)."""
    import stardust as sd

    def f(x):
        with np.errstate(all="ignore"):
            v = [float(c.f(np.float64(t))) for t in np.asarray(x).ravel().tolist()]
            return sd.asarray(v) if lib == "sd" else np.asarray(v).reshape(np.shape(x))

    return f


def _find_root(lib="sd"):
    if lib == "sd":
        import stardust as sd

        find_root = sd.optimize.find_root
    else:
        from scipy.optimize.elementwise import find_root

    roots = _arb_roots()
    where, ulp, unconv, zwhere, zabs = [], [], 0, [], []
    for cid, (c, lo, hi) in roots.items():
        with np.errstate(all="ignore"):
            r = _Res(find_root(_vec(c, lib), c.bracket), lib)
        if not bool(np.asarray(r.converged)):
            unconv += 1
            continue
        x = float(np.asarray(r.x))
        zero = _zero_in_double(c, x)
        if lo == hi == 0.0:
            zwhere.append(cid)
            zabs.append(0.0 if zero else _dist_ulp(x, lo, hi)[0])
            continue
        where.append(cid)
        ulp.append(0.0 if zero else _dist_ulp(x, lo, hi)[1])
    e = region(where, ulp=np.array(ulp))
    e["unconverged"] = unconv
    return {"default": e, "root 0": region(zwhere, abs_err=np.array(zabs))}


# ------------------------------------------------------------------------------ minimize

# the default gtol and the norm the method's test uses (docs/api/optimize.md §5)
GTOL = {"bfgs": (1e-5, np.inf), "trust-ncg": (1e-4, 2), "trust-exact": (1e-4, 2), "l-bfgs-b": (1e-5, np.inf)}
SMALL = ("newton-cg", "trust-ncg", "trust-exact")  # with the exact Hessian: n <= 10 only


def _known(c):
    from tests.optimize import mgh

    k = mgh.KNOWN.get((c.number, c.n, c.m))
    return None if k is None else min(k)


def _minimize(method, lib="sd"):
    O = _opt(lib)
    from tests.optimize import mgh

    where, grad, excess, ratio, rwhere, unconv = [], [], [], [], [], 0
    for c in mgh.CASES:
        if method in SMALL and c.n > 10:
            continue
        _, _, f, g = mgh.as_numpy(c)
        kw = {"jac": g} if method != "nelder-mead" else {}
        if method in ("trust-ncg", "trust-exact"):
            kw["hess"] = lambda x, c=c: np.asarray(mgh.hessian(c, [float(v) for v in x]), dtype=float)
        with np.errstate(all="ignore"):
            try:
                r = _Res(O.minimize(f, c.x0(), method=method, **kw), lib)
            except ValueError, ArithmeticError, np.linalg.LinAlgError:
                unconv += 1
                continue
        x = np.asarray(r.x, dtype=float)
        with np.errstate(all="ignore"):
            gx = float(np.max(np.abs(g(x))))
            fx = float(f(x))
        if not bool(r.converged):  # counted; its point is not a minimum to measure
            unconv += 1
            continue
        where.append(c.id)
        grad.append(gx)
        k = _known(c)
        excess.append(max(0.0, fx - k) / max(1.0, k) if k is not None and math.isfinite(fx) else math.nan)
        gradient_stop = method in ("bfgs", "trust-ncg", "trust-exact") or "PROJECTED GRADIENT" in str(r.message).upper()
        if bool(r.converged) and method in GTOL and gradient_stop:
            gtol, order = GTOL[method]
            with np.errstate(all="ignore"):
                gn = float(np.linalg.norm(g(x), order))
            rwhere.append(c.id)
            ratio.append(gn / gtol)
    excess = np.array(excess)
    out = {}
    e = region(where, grad=np.array(grad))  # the converged runs: a local minimum may not be MGH's f*
    known = ~np.isnan(excess)
    e["excess_max"] = float(excess[known].max()) if known.any() else None
    e["unconverged"] = unconv
    out["mgh"] = e
    if ratio:
        out["gradient stop"] = region(
            rwhere, grad=np.array(ratio) * GTOL[method][0], contract=np.array(ratio), bound=1.0
        )
    return out


# ------------------------------------------------------------------------------ the rest


def _least_squares(method, lib="sd"):
    O = _opt(lib)
    from tests.regression import nist_nls as N

    where, rel, unconv = [], [], 0
    for name in N.PROBLEMS:
        d = N.load(name)
        x, y = N.data(d, np)
        f, jac = N.model(name, np)
        cert = np.array([float(v) for v in d["params"]])
        for s in ("start1", "start2"):
            b0 = np.array([float(v) for v in d[s]])
            with np.errstate(all="ignore"):
                try:
                    r = _Res(O.least_squares(lambda b: f(x, *b) - y, b0, jac=lambda b: jac(x, *b), method=method), lib)
                except ValueError, ArithmeticError:
                    unconv += 1
                    continue
            if not bool(r.converged):
                unconv += 1
            b = np.asarray(r.x, dtype=float)
            where.append(f"{name} {s}")
            rel.append(float(np.max(np.abs(b - cert) / np.abs(cert))))
    rel, where = np.array(rel), np.array(where)
    ok = rel <= 1e-6
    e = region(where[ok], rel=rel[ok])
    e["unconverged"] = unconv
    return {"certified minimum": e, "elsewhere": region(where[~ok], rel=rel[~ok])}


def _root(method, lib="sd"):
    O = _opt(lib)
    from tests.optimize import systems

    where, fn, unconv = [], [], 0
    for c in systems.CASES:
        F, J = systems.as_numpy(c)[:2]
        with np.errstate(all="ignore"):
            try:
                r = _Res(O.root(F, c.x0(), method=method, jac=J), lib)
            except ValueError, ArithmeticError:
                unconv += 1
                continue
        if not bool(r.converged):
            unconv += 1
            continue
        where.append(c.id)
        with np.errstate(all="ignore"):
            fn.append(float(np.max(np.abs(F(np.asarray(r.x, dtype=float))))))
    e = region(where, fnorm=np.array(fn))
    e["unconverged"] = unconv
    return {"converged": e}


def _linear(which, lib="sd"):
    O = _opt(lib)
    from tests.optimize import functions

    where, kkt = [], []
    for cid, (A, b, bounds) in functions.LSQ_LINEAR.items():
        a, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
        n = a.shape[1]
        if which == "nnls":
            r = O.nnls(a, b)
            x = np.asarray(r.x if lib == "sd" else r[0], dtype=float)
            lb, ub = np.zeros(n), np.full(n, np.inf)
        else:
            lb, ub = (
                (np.asarray(v, dtype=float) for v in bounds) if bounds else (np.full(n, -np.inf), np.full(n, np.inf))
            )
            r = O.lsq_linear(a, b, bounds=(lb, ub))
            x, act = np.asarray(r.x, dtype=float), np.asarray(r.active_mask)
        gr = a.T @ (a @ x - b)  # the gradient of 1/2 ||Ax - b||^2
        # KKT: the gradient is 0 on free variables, >= 0 at a lower bound, <= 0 at an upper one
        at_lo, at_hi = (x <= lb, x >= ub) if which == "nnls" else (act == -1, act == 1)
        viol = np.where(at_lo, np.maximum(0.0, -gr), np.where(at_hi, np.maximum(0.0, gr), np.abs(gr)))
        scale = np.abs(a).sum(axis=0).max() * (np.abs(a).sum(axis=1).max() * np.abs(x).max(initial=0) + np.abs(b).max())
        where.append(cid)
        kkt.append(float(viol.max(initial=0.0) / (EPS * max(scale, 1e-300))))
    return {"cases": region(where, kkt=np.array(kkt))}


def _approx_fprime(lib="sd"):
    """The default (accurate) differences against the analytic derivatives at x0 of the MGH
    cases with factor 1 (tests/optimize/ref/derivatives.npz, tools/optimize_accuracy_ref.py:
    Arb), normwise max |error| / max |derivative| (tests/optimize/test_diff.py's measure)."""
    import pathlib

    import stardust as sd
    from tests.optimize import mgh

    z = np.load(pathlib.Path(mgh.__file__).parent / "ref" / "derivatives.npz")
    where, rg, rj = [], [], []
    i = 0
    while f"g{i}_id" in z.files:
        c = mgh.BY_ID[str(z[f"g{i}_id"])]
        x0 = z[f"g{i}_x0"]
        f = lambda x, c=c: c.f(x.tolist())  # noqa: E731
        F = (
            (lambda x, c=c: sd.asarray(c.fun(x.tolist())))
            if lib == "sd"
            else (lambda x, c=c: np.asarray(c.fun(x.tolist())))
        )  # noqa: E731
        for out, fn, key in ((rg, f, "g"), (rj, F, "J")):
            got = np.asarray(_opt(lib).approx_fprime(np.asarray(x0, dtype=float), fn), dtype=float)
            hi, lo = z[f"g{i}_{key}_hi"], z[f"g{i}_{key}_lo"]
            out.append(float(np.abs((got - hi) - lo).max() / max(np.abs(hi).max(), 1e-300)))  # as test_diff.rel
        where.append(c.id)
        i += 1
    return {"gradient": region(where, norm_rel=np.array(rg)), "jacobian": region(where, norm_rel=np.array(rj))}


def sweeps():
    """Every sweep compares with scipy.optimize (the same methods by name). R's optimizers would
    need each test function written again in R: not compared."""
    specs = [(f"root_scalar.{m}", lambda lib="sd", m=m: _root_scalar(m, lib)) for m in BRACKETING]
    specs.append(("find_root", _find_root))
    for m in ("bfgs", "l-bfgs-b", "newton-cg", "trust-ncg", "trust-exact", "nelder-mead"):
        specs.append((f"minimize.{m}", lambda lib="sd", m=m: _minimize(m, lib)))
    for m in ("trf", "lm"):
        specs.append((f"least_squares.{m}", lambda lib="sd", m=m: _least_squares(m, lib)))
    for m in ("hybr", "lm"):
        specs.append((f"root.{m}", lambda lib="sd", m=m: _root(m, lib)))
    specs += [
        ("nnls", lambda lib="sd": _linear("nnls", lib)),
        ("lsq_linear", lambda lib="sd": _linear("lsq_linear", lib)),
        ("approx_fprime", _approx_fprime),
    ]
    return [Custom("optimize", name, fn, compare_libs(fn, ("scipy",), seconds=300)) for name, fn in specs]
