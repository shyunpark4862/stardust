"""scipy 1.18.1's results and call orders on the integrate test collection
(docs/design/integrate.md §7.3, decision C5): every function of §1.2 with scipy's defaults on the
problems of tests/integrate/problems.py. Writes tests/integrate/data/:

  scipy_quad.json       quad on every one-dimensional integral (INTEGRALS) with its break points
                        and weight: integral, error, nfev (infodict's neval), status (QUADPACK's
                        ier, read from scipy's own dispatch), message (scipy's text, None for 0),
                        and infodict's well-defined entries (last, the interval lists cut to last,
                        iord to QUADPACK's kmax, QAWO's moments to momcom; QAWF's cycle lists cut
                        to lst; past those the arrays hold uninitialised memory)
  scipy_tanhsinh.json   tanhsinh on the integrals without a weight (of a pair with and without break
                        points, the one without; problems.Integral.tanhsinh), one call each,
                        and each family of problems.BATCHES as one batch (the parameter as args):
                        integral, error, converged (scipy's success), status, nfev, max_level
                        (scipy's maxlevel), and the calls
  scipy_nsum.json       nsum on problems.SERIES: sum, error, converged, status, nfev, and the calls
  scipy_multiple.json   dblquad and tplquad on problems.MULTI (integral, error, the calls), nquad on
                        the same ranges (its neval), and cubature on those with a box (rules gk21,
                        gk15 and genz-malik: integral, error, status, subdivisions, the calls)
  scipy_vector.json     quad_vec on problems.VECTOR (integral, error, nfev, status, intervals)
  scipy_samples.json    trapezoid, simpson, romb (2^k + 1 points), cumulative_trapezoid (initial None
                        and 0) and cumulative_simpson on problems.samples() (a cumulative result of
                        more than 200 values as its sha256, shape and five values at each end);
                        fixed_quad (n = 5, 20, 100) on the finite smooth integrals; newton_cotes(N)
                        for N = 1 ... 14
  scipy_calls.npz       quad's points in call order, one float64 array per integral ("quad/<id>")

The calls of a function that passes arrays (tanhsinh, nsum, dblquad, ...) are recorded as their
shapes in order and the sha256 of every point's float64 bytes in call order (C order within a
call; the shapes run-length encoded as [shape, times], or their digest past 64 runs), with the
number of calls and of points: a test compares its own call order's digest without storing every point
(tanhsinh calls up to 16,387 points per integral). Floats are written with repr (NaN and inf as
JSON's NaN/Infinity, which Python reads back); arrays as lists.

Run: .venv/bin/python tools/gen_integrate_scipy.py
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tests" / "integrate" / "data"
SCIPY = "1.18.1"


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return [_jsonable(e) for e in v.tolist()] if v.ndim else _jsonable(v.item())
    if isinstance(v, (list, tuple)):
        return [_jsonable(e) for e in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(e) for k, e in v.items()}
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


class Calls:
    """The points a function is called with, in order: shapes and the sha256 of the bytes."""

    def __init__(self):
        self.shapes, self.h, self.count = [], hashlib.sha256(), 0
        self.points = []  # floats, kept only where the caller asks (quad)

    def add(self, x, keep=False):
        a = np.array(x, dtype=np.float64, order="C")  # a copy that keeps a 0-d call 0-d
        self.shapes.append(list(a.shape))
        self.h.update(a.tobytes())
        self.count += a.size
        if keep:
            self.points.extend(a.ravel().tolist())

    def record(self):
        """shapes run-length encoded: [[shape, times], ...] in call order."""
        runs = []
        for sh in self.shapes:
            if runs and runs[-1][0] == sh:
                runs[-1][1] += 1
            else:
                runs.append([sh, 1])
        out = {"calls": len(self.shapes), "count": self.count, "sha256": self.h.hexdigest()}
        if len(runs) <= 64:
            out["shapes"] = runs
        else:  # cubature's many region batches: the runs' digest only
            out["shapes_sha256"] = hashlib.sha256(json.dumps(runs, separators=(",", ":")).encode()).hexdigest()
        return out


def digest_of(points_in_order):
    """The sha256 of a sequence of float64 arrays' bytes (Calls.record's digest)."""
    h = hashlib.sha256()
    for a in points_in_order:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


# ---- quad


def _quad_kwargs(p):
    kw = {}
    if p.points:
        kw["points"] = list(p.points)
    if p.weight is not None:
        kw["weight"], kw["wvar"] = p.weight, p.wvar
    return kw


def _ier(p):
    """QUADPACK's ier of quad(p) from scipy's own dispatch (quad returns only the message)."""
    from scipy.integrate import _quadpack_py as Q

    a, b = min(p.a, p.b), max(p.a, p.b)
    eps = 1.49e-8
    if p.weight is None:
        r = Q._quad(p.f, a, b, (), 1, eps, eps, 50, list(p.points) if p.points else None)
    else:
        r = Q._quad_weight(p.f, a, b, (), 1, eps, eps, 50, 50, 50, p.weight, p.wvar, None)
    return int(r[-1])


def _cut(info, limit=50):
    """infodict's well-defined entries: the interval lists to `last` (QAWF's cycle lists to `lst`),
    iord to QUADPACK's kmax (last while last <= limit/2 + 2, else limit + 1 - last; past it the
    array is uninitialised memory), QAGP's ndin to its first intervals, chebmo to its momcom
    columns."""
    out = {}
    n = info.get("last", info.get("lst"))
    for k, v in info.items():
        if k == "iord" and n is not None:
            kmax = int(n) if int(n) <= limit // 2 + 2 else limit + 1 - int(n)
            out[k] = _jsonable(np.asarray(v)[:kmax])
        elif k == "ndin":  # one per first interval: len(pts) - 1 of its npts2 slots
            out[k] = _jsonable(np.asarray(v)[: len(info["pts"]) - 1])
        elif k == "chebmo":
            out[k] = _jsonable(np.asarray(v)[:, : int(info.get("momcom", 0))])  # scipy's (25, maxp1)
        elif isinstance(v, np.ndarray) and v.ndim == 1 and n is not None and v.size > n and k != "pts":
            out[k] = _jsonable(v[: int(n)])
        else:
            out[k] = _jsonable(v)
    return out


def quad_run(p):
    from scipy import integrate

    calls = Calls()

    def f(x):
        calls.add(x, keep=True)
        return p.f(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = integrate.quad(f, p.a, p.b, full_output=1, **_quad_kwargs(p))
    integral, error, info = r[0], r[1], r[2]
    msg = r[3] if len(r) > 3 else None
    ier = _ier(p) if p.a != p.b else 0
    out = {
        "integral": float(integral),
        "error": float(error),
        "nfev": int(info["neval"]) if "neval" in info else calls.count,
        "status": ier,
        "message": msg,
        "info": _cut({k: v for k, v in info.items() if k != "neval"}),
        "calls": calls.count,
    }
    if len(r) > 4:
        out["explain"] = _jsonable(r[4])
    return out, np.array(calls.points, dtype=np.float64)


# ---- tanhsinh, nsum


def _elementwise_result(r, value="integral"):
    out = {
        value: _jsonable(getattr(r, value)),
        "error": _jsonable(r.error),
        "converged": _jsonable(r.success),
        "status": _jsonable(r.status),
        "nfev": _jsonable(r.nfev),
    }
    if hasattr(r, "maxlevel"):
        out["max_level"] = _jsonable(r.maxlevel)
    return out


def tanhsinh_run(p):
    from scipy import integrate

    calls = Calls()

    def f(x):
        calls.add(x)
        return p.f_np(x)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = integrate.tanhsinh(f, p.a, p.b)
    return {**_elementwise_result(r), "calls": calls.record()}


def tanhsinh_batch_run(bt):
    from scipy import integrate

    calls = Calls()
    params = np.array(bt.params, dtype=np.float64)

    def f(x, p):
        calls.add(x)
        return bt.f_np(x, p)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = integrate.tanhsinh(f, bt.a, bt.b, args=(params,))
    return {"ids": bt.ids(), "params": list(bt.params), **_elementwise_result(r), "calls": calls.record()}


def nsum_run(s):
    from scipy import integrate

    calls = Calls()

    def f(k):
        calls.add(k)
        return s.f_np(k)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = integrate.nsum(f, s.a, s.b, step=s.step)
    return {**_elementwise_result(r, "sum"), "calls": calls.record()}


# ---- multiple and vector integrals


def multiple_run(m):
    from scipy import integrate

    calls = Calls()

    def func(*xs):
        calls.add(np.array(xs, dtype=np.float64))
        return float(m.func(*xs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if m.kind == "dblquad":
            r = integrate.dblquad(func, m.a, m.b, *m.limits)
        else:
            r = integrate.tplquad(func, m.a, m.b, *m.limits)
    out = {m.kind: {"integral": float(r[0]), "error": float(r[1]), "nfev": len(calls.shapes),
                    "calls": calls.record()}}  # fmt: skip
    if m.box is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = integrate.nquad(lambda *xs: float(m.func(*xs)), [list(b) for b in m.box], full_output=True)
        out["nquad"] = {"integral": float(n[0]), "error": float(n[1]), "nfev": int(n[2]["neval"])}
        a = [lo for lo, _ in m.box]
        b = [hi for _, hi in m.box]
        for rule in ("gk21", "gk15", "genz-malik"):
            cc = Calls()

            def g(X, cc=cc):
                cc.add(X)
                return m.func(*[X[:, i] for i in range(X.shape[1])])

            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = integrate.cubature(g, a, b, rule=rule)
            out[f"cubature/{rule}"] = {
                "integral": float(c.estimate), "error": float(c.error), "status": c.status,
                "subdivisions": int(c.subdivisions), "nfev": cc.count // len(a), "calls": cc.record(),
            }  # fmt: skip
    return out


def vector_run(v):
    from scipy import integrate

    calls = Calls()

    def f(x):
        calls.add(x)
        return v.func(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, err, info = integrate.quad_vec(f, v.a, v.b, full_output=True)
    return {
        "integral": _jsonable(y), "error": float(err), "nfev": int(info.neval), "status": int(info.status),
        "converged": bool(info.success), "intervals": int(len(info.intervals)), "calls": calls.record(),
    }  # fmt: skip


# ---- samples, fixed_quad, newton_cotes


def _big(v, limit=200):
    """An array result: whole up to `limit` values, else its sha256 (float64 bytes), size, ends."""
    a = np.asarray(v, dtype=np.float64)
    if a.size <= limit:
        return _jsonable(a)
    return {"sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(), "shape": list(a.shape),
            "head": _jsonable(a.ravel()[:5]), "tail": _jsonable(a.ravel()[-5:])}  # fmt: skip


def samples_runs():
    from scipy import integrate

    from tests.integrate import problems as P

    out = {}
    for sid, (y, x, dx) in P.samples().items():
        kw = {"x": x} if x is not None else {"dx": dx}
        r = {
            "trapezoid": _jsonable(integrate.trapezoid(y, **kw)),
            "simpson": _jsonable(integrate.simpson(y, **kw)),
            "cumulative_trapezoid": _big(integrate.cumulative_trapezoid(y, **kw)),
            "cumulative_trapezoid/initial=0": _big(integrate.cumulative_trapezoid(y, initial=0, **kw)),
        }
        if y.shape[-1] >= 3:
            r["cumulative_simpson"] = _big(integrate.cumulative_simpson(y, **kw))
        n = y.shape[-1] - 1
        if x is None and n >= 1 and n & (n - 1) == 0:
            r["romb"] = _jsonable(integrate.romb(y, dx=dx))
        out[sid] = r
    fixed = {}
    for p in P.INTEGRALS:
        if not (p.numeric and p.weight is None and math.isfinite(p.a) and math.isfinite(p.b)):
            continue
        for n in (5, 20, 100):
            calls = Calls()

            def f(x, p=p, calls=calls):
                calls.add(x)
                return p.f_np(x)

            v, none = integrate.fixed_quad(f, p.a, p.b, n=n)
            fixed[f"{p.id}/n={n}"] = {"integral": float(v), "second": none, "calls": calls.record()}
    nc = {}
    for N in range(1, 15):
        an, B = integrate.newton_cotes(N, 1)
        nc[str(N)] = {"an": _jsonable(an), "B": float(B)}
    return out, fixed, nc


# ---- writing


def write(name, rows):
    import scipy

    lines = [f'"scipy": {json.dumps(scipy.__version__)}']
    lines += [f"{json.dumps(k)}: {json.dumps(_jsonable(v), separators=(',', ':'))}" for k, v in rows]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"{len(rows)} records -> {(DATA / name).relative_to(ROOT)}")


def main():
    import scipy

    from tests.integrate import problems as P

    assert scipy.__version__ == SCIPY, scipy.__version__
    DATA.mkdir(parents=True, exist_ok=True)
    rows, arrays = [], {}
    for p in P.INTEGRALS:
        res, pts = quad_run(p)
        rows.append((p.id, res))
        arrays[f"quad/{p.id}"] = pts
    write("scipy_quad.json", rows)
    np.savez_compressed(DATA / "scipy_calls.npz", **arrays)
    print(f"{len(arrays)} call orders -> {(DATA / 'scipy_calls.npz').relative_to(ROOT)}")
    rows = [(p.id, tanhsinh_run(p)) for p in P.INTEGRALS if p.tanhsinh]
    rows += [(f"batch/{bt.family}", tanhsinh_batch_run(bt)) for bt in P.BATCHES]
    write("scipy_tanhsinh.json", rows)
    write("scipy_nsum.json", [(s.id, nsum_run(s)) for s in P.SERIES])
    write("scipy_multiple.json", [(m.id, multiple_run(m)) for m in P.MULTI])
    write("scipy_vector.json", [(v.id, vector_run(v)) for v in P.VECTOR])
    samples, fixed, nc = samples_runs()
    write("scipy_samples.json", [("samples", samples), ("fixed_quad", fixed), ("newton_cotes", nc)])


if __name__ == "__main__":
    main()
