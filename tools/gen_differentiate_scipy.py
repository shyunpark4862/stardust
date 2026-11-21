"""scipy 1.18.1's results and call orders on the differentiate test collection
(docs/design/integrate.md §7.3, decision C5): scipy.differentiate's three functions with scipy's
defaults on the problems of tests/differentiate/problems.py. Writes tests/differentiate/data/:

  scipy_derivative.json  derivative on each ELEMENTARY function at each of its points, one call
                         per point ("<name>/<x!r>") and the nine points as one batch
                         ("batch/<name>"), and on the EDGES with their options ("edge/<id>"; sd's
                         option names, given to scipy as tolerances and maxiter):
                         df, error, converged (scipy's success), status, n_iter (scipy's nit),
                         nfev, x, and the calls
                         and (calculus review) the nine points as a 3 x 3 batch ("nd/<name>")
                         and problems.ARGS with their args ("args/<id>")
  scipy_jacobian.json    jacobian of the residuals F of each MGH case with factor 1 at x0: df
                         (m x n), error, converged, status, n_iter, nfev (per entry), the calls
  scipy_hessian.json     hessian of f = sum F_i^2 of the MGH cases with n <= 10 at x0: ddf, error,
                         converged, status, nfev (per entry), the calls

The calls are recorded as in tools/gen_integrate_scipy.py: the shapes in order (run-length
encoded), the number of calls and of points, and the sha256 of every point's float64 bytes in
call order. Floats are written with repr (NaN and inf as JSON's NaN/Infinity). Up to three
processes (maxtasksperchild=4) share the MGH cases.

Run: .venv/bin/python tools/gen_differentiate_scipy.py               # every file
     .venv/bin/python tools/gen_differentiate_scipy.py derivative    # scipy_derivative.json only
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
DATA = ROOT / "tests" / "differentiate" / "data"
SCIPY = "1.18.1"


def _result(r, value="df"):
    from gen_integrate_scipy import _jsonable

    out = {value: _jsonable(getattr(r, value)), "error": _jsonable(r.error), "converged": _jsonable(r.success),
           "status": _jsonable(r.status), "nfev": _jsonable(r.nfev)}  # fmt: skip
    if hasattr(r, "nit"):
        out["n_iter"] = _jsonable(r.nit)
    if hasattr(r, "x"):
        out["x"] = _jsonable(r.x)
    return out


def derivative_run(fn, x, options, args=()):
    from gen_integrate_scipy import Calls
    from scipy import differentiate

    calls = Calls()
    g = fn.f_np if hasattr(fn, "f_np") else fn

    def f(t, *a):
        calls.add(t)
        return g(t, *a)

    opts = dict(options)  # sd's names (atol, rtol, max_iter) as scipy's (tolerances, maxiter)
    tol = {k: opts.pop(k) for k in ("atol", "rtol") if k in opts}
    if tol:
        opts["tolerances"] = tol
    if "max_iter" in opts:
        opts["maxiter"] = opts.pop("max_iter")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = differentiate.derivative(f, x, args=args, **opts)
    return {**_result(r), "calls": calls.record()}


def _jacobian_task(i):
    from gen_integrate_scipy import Calls
    from scipy import differentiate

    from tests.differentiate import problems as P

    c = P.mgh_cases()[i]
    x0 = np.array([float(v) for v in c.x0()])
    F = P.mgh_F_np(c)
    calls = Calls()

    def f(x):
        calls.add(x)
        return F(x)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = differentiate.jacobian(f, x0)
    return c.id, {**_result(r), "calls": calls.record()}


def _hessian_task(i):
    from gen_integrate_scipy import Calls
    from scipy import differentiate

    from tests.differentiate import problems as P

    c = P.mgh_cases()[i]
    x0 = np.array([float(v) for v in c.x0()])
    fs = P.mgh_f_np(c)
    calls = Calls()

    def f(x):
        calls.add(x)
        return fs(x)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = differentiate.hessian(f, x0)
    return c.id, {**_result(r, "ddf"), "calls": calls.record()}


def write(name, rows):
    import scipy
    from gen_integrate_scipy import _jsonable

    lines = [f'"scipy": {json.dumps(scipy.__version__)}']
    lines += [f"{json.dumps(k)}: {json.dumps(_jsonable(v), separators=(',', ':'))}" for k, v in rows]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"{len(rows)} records -> {(DATA / name).relative_to(ROOT)}")


def main():
    import scipy

    from tests.differentiate import problems as P

    assert scipy.__version__ == SCIPY, scipy.__version__
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for fn in P.ELEMENTARY:
        for x in fn.points:
            rows.append((f"{fn.name}/{x!r}", derivative_run(fn, x, {})))
        rows.append((f"batch/{fn.name}", derivative_run(fn, np.array(fn.points), {})))
    for e in P.EDGES:
        rows.append((f"edge/{e.id}", derivative_run(P.BY_NAME[e.function], e.x, e.options)))
    for fn in P.ELEMENTARY:
        rows.append((f"nd/{fn.name}", derivative_run(fn, np.array(fn.points).reshape(P.ND_SHAPE), {})))
    for aid, g, x, args in P.ARGS:
        rows.append((aid, derivative_run(g, x, {}, args=args)))
    write("scipy_derivative.json", rows)
    if sys.argv[1:] == ["derivative"]:
        return
    cases = P.mgh_cases()
    with mp.get_context("spawn").Pool(3, maxtasksperchild=4) as pool:
        write("scipy_jacobian.json", pool.map(_jacobian_task, range(len(cases))))
        idx = [i for i, c in enumerate(cases) if c.n <= P.HESSIAN_MAX_N]
        write("scipy_hessian.json", pool.map(_hessian_task, idx))


if __name__ == "__main__":
    main()
