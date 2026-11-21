"""sd.differentiate's accuracy sweeps (docs/design/integrate.md §4, §7.5): one per function, on the
reference families of tests/differentiate/ref (tools/differentiate_ref.py: Arb).

Contract (the stopping test's meaning, which the collection meets): a converged derivative is
within atol + rtol |df| of the exact derivative (contract = the largest |df - f'| / (atol + rtol
|df|), bound 1; derivative at the eleven elementary functions' nine points and the edge cases with
their options). The Jacobians, the gradients of sum F_i^2 (jacobian's scalar f; sweep "gradient", added in the
calculus review) and the Hessians of the More-Garbow-Hillstrom cases at x0 are measured
normwise: the largest error of a converged entry over rtol times the largest exact entry
(contract, bound 1), because the relative test means nothing at a near-zero entry.
Recorded only: the relative error (derivative), the honesty of the error estimate (how many
converged results have a true error above their estimate), and how many did not converge.
scipy.differentiate 1.18.1 is the comparator: the same method and points (C5), its weights
solved by LAPACK. R's numDeriv (grad, jacobian, hessian; method "Richardson" at its defaults) is a
second one: another method for the same quantity, called on the same Python functions through
reticulate (bench/speed times the same calls). It has no options of sd's (the edge cases with
options are left out), no convergence test (every run counts as converged) and no error estimate
(no honesty count).
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region

TINY = 2.2250738585072014e-308
RTOL = 2.0**-26


def _module(lib):
    if lib == "sd":
        import stardust as sd

        return sd.differentiate, sd.asarray
    if lib == "scipy":
        import scipy.differentiate as sdf

        return sdf, np.asarray
    return None


def _r_prelude():
    """bench/accuracy/optimize.py's R prelude (reticulate on this Python), numDeriv and the problems."""
    from .optimize import _r_prelude as base

    return base() + (
        'if (!exists(".sdd", envir = globalenv())) { suppressMessages(library(numDeriv)); '
        'py_run_string("import tests.differentiate.problems as _sdP; '
        '_sd_mgh = {c.id: c for c in _sdP.mgh_cases()}"); '
        'assign(".sdd", list(P = import("tests.differentiate.problems")), envir = globalenv()) }; '
    )


def _r_derivative():
    """numDeriv::grad on the elementary functions' points (no options: no edge cases)."""
    from tests.differentiate import problems as P
    from tests.differentiate import refdata

    from . import rsession

    ref = refdata.derivatives()
    where, rel, contract = [], [], []
    for f in P.ELEMENTARY:
        expr = _r_prelude() + (
            f'fn <- .sdd$P$BY_NAME[["{f.name}"]]; g <- function(t) as.numeric(fn$f_np(t)); '
            "sapply(X, function(x) tryCatch(grad(g, x), error = function(e) NaN))"
        )
        vals = rsession.evaluate_named(expr, {"X": np.asarray([float(x) for x in f.points])})
        for x, df in zip(f.points, vals):
            hi, lo = ref[f"{f.name}/{float(x)!r}"]["df"]
            err = abs((float(df) - hi) - lo)
            where.append(f"{f.name}/{x!r}")
            rel.append(err / abs(hi) if hi else err)
            contract.append(err / (TINY + RTOL * abs(float(df))))
    return {"elementary": region(where, rel=np.array(rel), contract=np.array(contract), bound=1.0)}


def _r_multivariate(kind):
    """numDeriv::jacobian (of F) or hessian (of f) on the MGH cases at x0."""
    from tests.differentiate import problems as P
    from tests.differentiate import refdata

    from . import rsession

    ref = refdata.mgh()
    where, contract = [], []
    for c in P.mgh_cases():
        v = ref[c.id][kind]
        if v is None:
            continue
        fn = "mgh_F_np" if kind == "jacobian" else "mgh_f_np"
        rcall = {"gradient": "grad"}.get(kind, kind)
        expr = _r_prelude() + (
            f"h <- py_eval(\"_sdP.{fn}(_sd_mgh['{c.id}'])\"); g <- function(x) as.numeric(h(np_array(as.numeric(x)))); "
            f"M <- tryCatch({rcall}(g, X0), error = function(e) NULL); "
            "if (is.null(M)) NaN else as.numeric(t(M))"
        )
        x0 = np.array([float(t) for t in c.x0()])
        got = rsession.evaluate_named(expr, {"X0": x0})
        hi, lo = v
        if got.size != np.size(hi):
            continue
        got = got.reshape(np.shape(hi))
        err = np.abs((got - hi) - lo)
        where.append(c.id)
        contract.append(float(err.max() / (RTOL * max(np.abs(hi).max(), TINY))))
    return {"mgh": region(where, contract=np.array(contract), bound=1.0)}


def _options(lib, opts):
    """sd's option names, as scipy's (tolerances, maxiter) for scipy."""
    opts = dict(opts)
    if lib == "scipy":
        tol = {k: opts.pop(k) for k in ("atol", "rtol") if k in opts}
        if tol:
            opts["tolerances"] = tol
        if "max_iter" in opts:
            opts["maxiter"] = opts.pop("max_iter")
    return opts


def _derivative(lib="sd"):
    if lib == "R":
        return _r_derivative()
    m = _module(lib)
    if m is None:
        return None
    D, wrap = m
    from tests.differentiate import problems as P
    from tests.differentiate import refdata

    ref = refdata.derivatives()
    runs = [(f"{f.name}/{x!r}", f, x, {}) for f in P.ELEMENTARY for x in f.points]
    runs += [(f"edge/{e.id}", P.BY_NAME[e.function], e.x, e.options) for e in P.EDGES]
    where, rel, contract, unconv, dishonest = [], [], [], 0, 0
    for key, f, x, opts in runs:
        with np.errstate(all="ignore"):
            r = D.derivative(lambda t, f=f: wrap(f.f_np(np.asarray(t))), x, **_options(lib, opts))
        status = int(np.asarray(r.status))
        if status != 0:
            unconv += 1
            continue
        df, est = float(np.asarray(r.df)), float(np.asarray(r.error))
        hi, lo = ref[f"{f.name}/{float(x)!r}"]["df"]
        err = abs((df - hi) - lo)
        where.append(key)
        rel.append(err / abs(hi) if hi else err)
        contract.append(err / (opts.get("atol", TINY) + opts.get("rtol", RTOL) * abs(df)))
        dishonest += err > est
    e = region(where, rel=np.array(rel), contract=np.array(contract), bound=1.0)
    e["unconverged"] = unconv
    e["error_below_true"] = int(dishonest)
    return {"elementary": e}


def _multivariate(kind, lib="sd"):
    if lib == "R":
        return _r_multivariate(kind)
    m = _module(lib)
    if m is None:
        return None
    D, wrap = m
    from tests.differentiate import problems as P
    from tests.differentiate import refdata

    ref = refdata.mgh()
    where, contract, entries, unconv, dishonest = [], [], 0, 0, 0
    for c in P.mgh_cases():
        v = ref[c.id][kind]
        if v is None:
            continue
        x0 = np.array([float(t) for t in c.x0()])
        fn = P.mgh_F_np(c) if kind == "jacobian" else P.mgh_f_np(c)
        call = D.hessian if kind == "hessian" else D.jacobian  # gradient: jacobian of the scalar sum F_i^2
        with np.errstate(all="ignore"):
            r = call(lambda x, fn=fn: wrap(fn(np.asarray(x))), wrap(x0))
        got = np.asarray(r.ddf if kind == "hessian" else r.df, dtype=float)
        hi, lo = v
        ok = np.asarray(r.status) == 0
        err = np.abs((got - hi) - lo)
        entries += int(ok.sum())
        unconv += int((~ok).sum())
        dishonest += int((ok & (err > np.asarray(r.error))).sum())
        where.append(c.id)
        contract.append(float(err[ok].max(initial=0.0) / (RTOL * max(np.abs(hi).max(), TINY))))
    e = region(where, contract=np.array(contract), bound=1.0)
    e.update(entries=entries, unconverged=unconv, error_below_true=dishonest)
    return {"mgh": e}


def sweeps():
    specs = [
        ("derivative", _derivative),
        ("jacobian", lambda lib="sd": _multivariate("jacobian", lib)),
        ("hessian", lambda lib="sd": _multivariate("hessian", lib)),
        ("gradient", lambda lib="sd": _multivariate("gradient", lib)),
    ]
    return [Custom("differentiate", name, fn, compare_libs(fn, ("scipy", "R"), seconds=300)) for name, fn in specs]
