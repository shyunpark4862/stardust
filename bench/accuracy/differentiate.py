"""sd.differentiate's accuracy sweeps (docs/design/integrate.md §4, §7.5): one per function, on the
reference families of tests/differentiate/ref (tools/differentiate_ref.py: Arb).

Contract (the stopping test's meaning, which the collection meets): a converged derivative is
within atol + rtol |df| of the exact derivative (contract = the largest |df - f'| / (atol + rtol
|df|), bound 1; derivative at the eleven elementary functions' nine points and the edge cases with
their options). The Jacobians and Hessians of the More-Garbow-Hillstrom cases at x0 are measured
normwise: the largest error of a converged entry over rtol times the largest exact entry
(contract, bound 1), because the relative test means nothing at a near-zero entry.
Recorded only: the relative error (derivative), the honesty of the error estimate (how many
converged results have a true error above their estimate), and how many did not converge.
scipy.differentiate 1.18.1 is the comparator: the same method and points (C5), its weights
solved by LAPACK.
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
        with np.errstate(all="ignore"):
            r = getattr(D, kind)(lambda x, fn=fn: wrap(fn(np.asarray(x))), wrap(x0))
        got = np.asarray(r.df if kind == "jacobian" else r.ddf, dtype=float)
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
    ]
    return [Custom("differentiate", name, fn, compare_libs(fn, ("scipy",), seconds=300)) for name, fn in specs]
