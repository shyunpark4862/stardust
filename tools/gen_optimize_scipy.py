"""scipy 1.18.1's results on the optimize test collection (docs/design/optimize.md §7, decision
O13): every method of §1.2 with scipy's defaults, on the problems of tests/optimize/mgh.py,
systems.py and functions.py. Writes tests/optimize/data/scipy_*.json:

  scipy_least_squares.json  least_squares 'trf' and 'lm', analytic and '2-point' Jacobians, on
                            the 110 Moré-Garbow-Hillstrom cases, and 'trf' with a box around x0
  scipy_minimize.json       minimize: Nelder-Mead, BFGS, L-BFGS-B (analytic gradient and
                            scipy's differences), Newton-CG, trust-ncg, trust-exact (exact
                            Hessians by ad.Dual2) on f = sum F_i^2 of the 110 cases, and on
                            Rosenbrock, the maximum-entropy problem and L-BFGS-B's driver1
  scipy_root.json           root 'hybr' and 'lm', with and without the Jacobian, on the 55
                            MINPACK HYBRD/HYBRJ systems
  scipy_root_scalar.json    root_scalar: brentq, toms748, bisect on every bracket; newton
                            (with fprime, and with scipy's difference), halley, secant from x0
  scipy_minimize_scalar.json  brent, golden (bracket) and bounded (bounds)
  scipy_elementwise.json    find_root, bracket_root, find_minimum, bracket_minimum, with scipy's
                            tolerances and with the papers' (whose counts they reproduce)
  scipy_lsq_linear.json     lsq_linear 'trf' and 'bvls', nnls
  scipy_approx_fprime.json  approx_fprime (forward differences, scipy's epsilon) and check_grad
                            at each case's x0

Each run records what scipy returns (x, fun or cost, nfev, njev, nhev, nit, status, success,
message), or the exception it raises. Floats are written with repr (NaN and inf as JSON's
NaN/Infinity, which Python reads back). The runs are independent; up to 8 processes.

Run: uv run --no-sync python tools/gen_optimize_scipy.py
"""

import json
import math
import multiprocessing as mp
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tests" / "optimize" / "data"


def _f(v):
    return [float(t) for t in np.ravel(v)]


def _res(r, *, vector_fun=False):
    """The fields of a scipy OptimizeResult that the reference keeps."""
    out = {}
    for k in (
        "x",
        "fun",
        "cost",
        "optimality",
        "nfev",
        "njev",
        "nhev",
        "nit",
        "status",
        "success",
        "message",
        "active_mask",
    ):
        if k not in r:
            continue
        v = r[k]
        if v is None:
            out[k] = None
        elif k in ("x", "active_mask") or (k == "fun" and vector_fun):
            out[k] = _f(v) if k == "x" or k == "fun" else [int(t) for t in np.ravel(v)]
        elif k in ("nfev", "njev", "nhev", "nit", "status"):
            out[k] = int(v)
        elif k == "success":
            out[k] = bool(v)
        elif k == "message":
            out[k] = str(v)
        else:
            out[k] = float(v)
    return out


def _guard(fn):
    def run(*a):
        np.seterr(all="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return fn(*a)
            except Exception as e:  # the reference records scipy's errors too
                return {"error": f"{type(e).__name__}: {e}"}

    return run


# ---- least_squares


def _lsq_task(args):
    from scipy.optimize import least_squares

    from tests.optimize import mgh

    cid, method, jac, box = args
    c = mgh.BY_ID[cid]
    F, J, _, _ = mgh.as_numpy(c)
    x0 = np.array(c.x0())
    kw = {}
    if box:
        kw["bounds"] = (x0 - 0.5 * (np.abs(x0) + 1.0), x0 + 0.5 * (np.abs(x0) + 1.0))

    def go():
        r = least_squares(F, x0, jac=J if jac == "analytic" else "2-point", method=method, **kw)
        return _res(r, vector_fun=True)

    return (cid + ("-box" if box else ""), f"{method}/{jac}"), _guard(go)()


def least_squares_tasks():
    from tests.optimize import mgh

    tasks = [(c.id, m, j, False) for c in mgh.CASES for m in ("trf", "lm") for j in ("analytic", "2-point")]
    tasks += [(c.id, "trf", j, True) for c in mgh.CASES if c.factor == 1.0 for j in ("analytic", "2-point")]
    return tasks


# ---- minimize


def _min_problem(pid):
    """(f, g, h, x0, bounds) of a minimize problem id."""
    from tests.optimize import functions as fn
    from tests.optimize import mgh

    if pid.startswith("mgh"):
        c = mgh.BY_ID[pid]
        _, _, f, g = mgh.as_numpy(c)
        return f, g, (lambda x: mgh.hessian(c, x)), np.array(c.x0()), None
    if pid.startswith("rosen"):
        n = int(pid.split("-n")[1])
        return (
            lambda x: fn.rosen([np.float64(v) for v in x]),
            lambda x: np.array(fn.rosen_der([np.float64(v) for v in x])),
            lambda x: np.array(fn.rosen_hess([np.float64(v) for v in x])),
            np.array(fn.ROSEN_X0[n]),
            None,
        )
    if pid == "maxent":
        from tests.optimize.ad import value_grad_hess

        return (
            lambda x: fn.maxent([np.float64(v) for v in x]),
            lambda x: np.array(fn.maxent_der([np.float64(v) for v in x])),
            lambda x: value_grad_hess(fn.maxent, x)[2],
            np.array(fn.MAXENT_X0),
            None,
        )
    if pid == "lbfgsb_driver1":
        from tests.optimize.ad import value_grad_hess

        return (
            lambda x: fn.lbfgsb_driver1([np.float64(v) for v in x]),
            lambda x: np.array(fn.lbfgsb_driver1_der([np.float64(v) for v in x])),
            lambda x: value_grad_hess(fn.lbfgsb_driver1, x)[2],
            np.array(fn.LBFGSB_DRIVER1_X0),
            fn.LBFGSB_DRIVER1_BOUNDS,
        )
    raise ValueError(pid)


MIN_METHODS = ("Nelder-Mead", "BFGS", "BFGS/fd", "L-BFGS-B", "L-BFGS-B/fd", "Newton-CG", "trust-ncg", "trust-exact")


def _min_task(args):
    from scipy.optimize import minimize

    pid, method = args
    f, g, h, x0, bounds = _min_problem(pid)
    meth, _, fd = method.partition("/")

    def go():
        kw = {}
        if meth in ("BFGS", "L-BFGS-B", "Newton-CG", "trust-ncg", "trust-exact") and not fd:
            kw["jac"] = g
        if meth in ("Newton-CG", "trust-ncg", "trust-exact"):
            kw["hess"] = h
        if bounds is not None and meth in ("Nelder-Mead", "L-BFGS-B"):
            kw["bounds"] = bounds
        r = minimize(f, x0, method=meth, **kw)
        return _res(r)

    return (pid, method), _guard(go)()


def minimize_tasks():
    from tests.optimize import mgh

    pids = [c.id for c in mgh.CASES] + ["rosen-n2", "rosen-n5", "maxent", "lbfgsb_driver1"]
    out = []
    for p in pids:
        for m in MIN_METHODS:
            if p == "lbfgsb_driver1" and m not in ("Nelder-Mead", "L-BFGS-B", "L-BFGS-B/fd"):
                continue  # driver1's problem has bounds
            out.append((p, m))
    return out


# ---- root


def _root_task(args):
    from scipy.optimize import root

    from tests.optimize import systems

    cid, method, jac = args
    c = systems.BY_ID[cid]
    F, J = systems.as_numpy(c)

    def go():
        r = root(F, np.array(c.x0()), method=method, jac=J if jac == "analytic" else None)
        out = _res(r, vector_fun=True)
        return out

    return (cid, f"{method}/{jac}"), _guard(go)()


def root_tasks():
    from tests.optimize import systems

    return [(c.id, m, j) for c in systems.CASES for m in ("hybr", "lm") for j in ("analytic", "none")]


# ---- root_scalar


def _scalar_root_task(args):
    from scipy.optimize import root_scalar

    from tests.optimize import functions as fn

    cid, method = args
    c = fn.ROOTS_BY_ID[cid]

    def f(x):
        return float(c.f(np.float64(x)))

    def go():
        kw = {"method": method.split("/")[0]}
        if method in ("brentq", "toms748", "bisect"):
            kw["bracket"] = c.bracket
        else:
            kw["x0"] = c.x0
            if method == "newton":
                kw["fprime"] = lambda x: float(c.fprime(np.float64(x)))
            if method == "halley":
                kw["fprime"] = lambda x: float(c.fprime(np.float64(x)))
                kw["fprime2"] = lambda x: float(c.fprime2(np.float64(x)))
        r = root_scalar(f, **kw)
        return {
            "root": float(r.root),
            "iterations": int(r.iterations) if r.iterations == r.iterations else None,
            "function_calls": int(r.function_calls),
            "converged": bool(r.converged),
            "flag": str(r.flag),
        }

    return (cid, method), _guard(go)()


def root_scalar_tasks():
    from tests.optimize import functions as fn

    out = []
    for c in fn.ROOT_CASES:
        out += [(c.id, m) for m in ("brentq", "toms748", "bisect")]
        if c.x0 is not None:
            out.append((c.id, "secant"))
            out.append((c.id, "newton/fd"))
            if c.fprime is not None:
                out.append((c.id, "newton"))
            if c.fprime2 is not None:
                out.append((c.id, "halley"))
    return out


# ---- minimize_scalar


def _scalar_min_task(args):
    from scipy.optimize import minimize_scalar

    from tests.optimize import functions as fn

    cid, method = args
    c = fn.MINS_BY_ID[cid]

    def f(x):
        return float(c.f(np.float64(x)))

    def go():
        if method == "bounded":
            r = minimize_scalar(f, bounds=c.bounds, method="bounded")
        else:
            r = minimize_scalar(f, bracket=c.bracket, method=method)
        return _res(r)

    return (cid, method), _guard(go)()


def minimize_scalar_tasks():
    from tests.optimize import functions as fn

    return [(c.id, m) for c in fn.MIN_CASES for m in ("brent", "golden", "bounded")]


# ---- elementwise


def _vec(c):
    def f(x):
        x = np.asarray(x, dtype=float)
        return np.array([float(c.f(np.float64(t))) for t in np.ravel(x)]).reshape(x.shape)

    return f


def _ew(r):
    """The fields of an elementwise result (one element)."""
    out = {}
    for k in ("x", "f_x", "nfev", "nit", "status", "success"):
        if k in r:
            v = r[k]
            out[k] = bool(v) if k == "success" else (int(v) if k in ("nfev", "nit", "status") else float(v))
    if "bracket" in r:
        out["bracket"] = [float(v) for v in r["bracket"]]
        out["f_bracket"] = [float(v) for v in r["f_bracket"]]
    return out


def _elementwise_task(args):
    from scipy.optimize.elementwise import bracket_minimum, bracket_root, find_minimum, find_root

    from tests.optimize import functions as fn

    kind, cid, tol = args

    def go():
        if kind in ("find_root", "bracket_root"):
            c = fn.ROOTS_BY_ID[cid]
            f = _vec(c)
            if kind == "find_root":
                t = dict(xrtol=4e-10, xatol=1e-5) if tol == "paper" else None
                return _ew(find_root(f, c.bracket, tolerances=t))
            return _ew(bracket_root(f, c.x0 if c.x0 is not None else c.bracket[0]))
        c = fn.MINS_BY_ID[cid]
        f = _vec(c)
        if kind == "find_minimum":
            t = dict(xatol=1e-4, fatol=1e-6, xrtol=1e-16, frtol=1e-16) if tol == "paper" else None
            return _ew(find_minimum(f, c.bracket, tolerances=t))
        return _ew(bracket_minimum(f, c.bracket[0]))

    return (cid, kind + ("/paper" if tol == "paper" else "")), _guard(go)()


def elementwise_tasks():
    from tests.optimize import functions as fn

    out = []
    for c in fn.ROOT_CASES:
        out.append(("find_root", c.id, "scipy"))
        out.append(("bracket_root", c.id, "scipy"))
        if c.nfev_paper is not None:
            out.append(("find_root", c.id, "paper"))
    for c in fn.MIN_CASES:
        out.append(("find_minimum", c.id, "scipy"))
        out.append(("bracket_minimum", c.id, "scipy"))
        if c.nit_paper is not None:
            out.append(("find_minimum", c.id, "paper"))
    return out


# ---- lsq_linear, nnls


def _lsq_linear_task(args):
    from scipy.optimize import lsq_linear, nnls

    from tests.optimize import functions as fn

    pid, method = args
    A, b, bounds = fn.LSQ_LINEAR[pid]
    A, b = np.array(A), np.array(b)

    def go():
        if method == "nnls":
            x, rnorm = nnls(A, b)
            return {"x": _f(x), "rnorm": float(rnorm)}
        kw = {} if bounds is None else {"bounds": (np.array(bounds[0]), np.array(bounds[1]))}
        r = lsq_linear(A, b, method=method, **kw)
        return _res(r, vector_fun=True)

    return (pid, method), _guard(go)()


def lsq_linear_tasks():
    from tests.optimize import functions as fn

    return [(p, m) for p in fn.LSQ_LINEAR for m in ("trf", "bvls", "nnls")]


# ---- approx_fprime, check_grad


def _fprime_task(cid):
    from scipy.optimize import approx_fprime, check_grad

    from tests.optimize import mgh

    c = mgh.BY_ID[cid]
    F, J, f, g = mgh.as_numpy(c)
    x0 = np.array(c.x0())

    def go():
        return {
            "grad": _f(approx_fprime(x0, f)),
            "jac": _f(approx_fprime(x0, F)),
            "check_grad": float(check_grad(f, g, x0)),
        }

    return (cid, "approx_fprime"), _guard(go)()


def fprime_tasks():
    from tests.optimize import mgh

    return [c.id for c in mgh.CASES if c.factor == 1.0]


def write(name, rows):
    """{id: {method: result}} as JSON, one id per line."""
    import scipy

    table = {}
    for (cid, method), res in rows:
        table.setdefault(cid, {})[method] = res
    lines = [f'"scipy": {json.dumps(scipy.__version__)}']
    lines += [f"{json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}" for k, v in table.items()]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    return len(rows)


def main():
    import scipy

    assert scipy.__version__ == "1.18.1", scipy.__version__
    jobs = [
        ("scipy_least_squares.json", _lsq_task, least_squares_tasks()),
        ("scipy_minimize.json", _min_task, minimize_tasks()),
        ("scipy_root.json", _root_task, root_tasks()),
        ("scipy_root_scalar.json", _scalar_root_task, root_scalar_tasks()),
        ("scipy_minimize_scalar.json", _scalar_min_task, minimize_scalar_tasks()),
        ("scipy_elementwise.json", _elementwise_task, elementwise_tasks()),
        ("scipy_lsq_linear.json", _lsq_linear_task, lsq_linear_tasks()),
        ("scipy_approx_fprime.json", _fprime_task, fprime_tasks()),
    ]
    with mp.get_context("spawn").Pool(8, maxtasksperchild=200) as pool:
        for name, fn, tasks in jobs:
            rows = pool.map(fn, tasks, chunksize=4)
            print(name, write(name, rows))


if __name__ == "__main__":
    main()
