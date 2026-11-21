"""The result object of every sd.optimize function (docs/api/optimize.md §1.3,
docs/design/optimize.md §2.2).

One class for all methods. Its fields are the names of the other areas' results (O4:
converged, n_iter, nfev, njev; regression's NLSResult): scipy's success and nit are not
aliased. A method gives the fields it has; reading one it does not give raises
AttributeError (so hasattr() tells). Every field is read-only (regression G98) and an array
field is a read-only array, the same at each access without a copy (API.md §2): writing into
it raises, and .copy() gives an array to change. The fields' machinery is the base it shares
with sd.integrate's and sd.differentiate's results (stardust._results._FieldResult).
"""

import math

from .. import _calls, _core
from .. import _summary as _S
from .._results import _FieldResult, _Lazy  # noqa: F401  (_Lazy: minimize's import)

# The field names, in the order repr and the documentation list them. Scalar problems give
# Python numbers; a batched scalar function (find_root, ...) gives arrays of the batch shape.
FIELDS = (
    # the solution
    "x",
    "fun",
    "jac",
    "hess",
    "hess_inv",
    "grad",
    "cost",
    "optimality",
    "active_mask",
    # batched scalar functions: the final bracket and the function there
    "bracket",
    "f_bracket",
    # Nelder-Mead
    "final_simplex",
    # how it stopped
    "converged",
    "status",
    "message",
    # counts
    "n_iter",
    "nfev",
    "njev",
    "nhev",
    # the method that ran
    "method",
)


class OptimizeResult(_FieldResult):
    """The result of a solver in sd.optimize: x, fun, converged, status, message, n_iter, nfev,
    njev and the method's other fields (docs/api/optimize.md §1.3)."""

    __slots__ = ()
    _FIELDS = FIELDS
    _ABSENT = "its method does not give it"

    def __init__(self, *, batch=(), **fields):
        self._set(batch, fields, True)  # the caller's arrays are copied once

    @_calls.method("summary")
    def summary(self):
        """The result as a text table (O119): the method, convergence, status and message, the
        objective (fun, or cost for least squares) and the counts, then x (cut at both ends when
        long). A batched scalar function: one row per element of the batch."""
        return _summary(self)

    def __repr__(self):
        f = self._fields
        if self._batch:
            conv = f.get("converged")
            n = 1
            for s in self._batch:
                n *= s
            done = f" {int(_core.count_nonzero(conv))} of {n} converged" if conv is not None else ""
            return (
                f"OptimizeResult(batch shape {self._batch},{done})"
                if done
                else f"OptimizeResult(batch shape {self._batch})"
            )
        parts = []
        if "method" in f:
            parts.append(f"method={f['method']!r}")
        if "converged" in f:
            parts.append(f"converged={f['converged']}")
        if "cost" in f:
            parts.append(f"cost={_num(f['cost'])}")
        elif "fun" in f and not isinstance(f["fun"], _core.ndarray):
            parts.append(f"fun={_num(f['fun'])}")
        for k in ("n_iter", "nfev"):
            if k in f:
                parts.append(f"{k}={f[k]}")
        if "x" in f:
            parts.append(f"x={f['x']}")
        return "OptimizeResult(" + ", ".join(parts) + ")"


_COUNTS = (("n_iter", "No. Iterations:"), ("nfev", "Function Evals:"), ("njev", "Jacobian Evals:"),
           ("nhev", "Hessian Evals:"))  # fmt: skip


def _summary(r):
    f = r._fields
    left, right = [], []
    if "method" in f:
        left.append(("Method:", str(f["method"])))
    if r._batch:
        n = _S.lanes(r._batch)
        left.append(("Batch shape:", _S.shape_text(r._batch)))
        if "converged" in f:
            right.append(("Converged:", f"{int(_core.count_nonzero(f['converged']))} of {n}"))
        cols = []
        for k, fmt in (("x", lambda v: _S.forg(v, 4)), ("fun", _S.forg), ("bracket", lambda v: _S.forg(v, 4)),
                       ("converged", _S.flag), ("status", str), ("n_iter", str), ("nfev", str)):  # fmt: skip
            v = f.get(k)
            if isinstance(v, _core.ndarray) and v.size == n:
                cols.append((k, _S.flat(v, n), fmt))
            elif isinstance(v, tuple) and all(isinstance(e, _core.ndarray) and e.size == n for e in v):
                cols += [(f"{k}[{j}]", _S.flat(e, n), fmt) for j, e in enumerate(v)]  # the bracket's ends
        table = _S.Table([c[0] for c in cols], _S.lane_rows(r._batch, [(g, fm) for _, g, fm in cols]))
        return _S.render("Optimization Results", header=(left, right), tables=[table])
    if "converged" in f:
        left.append(("Converged:", str(f["converged"])))
    if "status" in f:
        left.append(("Status:", str(f["status"])))
    fun = f.get("fun")
    if "cost" in f:
        right.append(("Cost:", f"{float(f['cost']):#.6g}"))
    elif fun is not None and not isinstance(fun, _core.ndarray):
        right.append(("Function value:", f"{float(fun):#.6g}"))
    if "optimality" in f:
        right.append(("Optimality:", f"{float(f['optimality']):#.4g}"))
    for k, label in _COUNTS:
        if k in f:
            right.append((label, str(f[k])))
    wide = [("Message:", str(f["message"]))] if "message" in f else []
    tables = []
    x = f.get("x")
    if isinstance(x, _core.ndarray):
        xs = _core.reshape(x, (x.size,))
        cols = ["x"]
        resid = isinstance(fun, _core.ndarray) and fun.size == x.size and "cost" not in f  # root's f(x)
        fs = _core.reshape(fun, (fun.size,)) if resid else None
        if resid:
            cols.append("fun")
        rows = []
        for i in _S.shown(x.size):
            if i is None:
                rows.append(None)
                continue
            cells = [_S.forg(xs[i], 4)] + ([_S.forg(fs[i])] if resid else [])
            rows.append((f"x[{i}]", cells))
        tables.append(_S.Table(cols, rows))
    elif x is not None:
        tables.append(_S.Table(["x"], [("x", [_S.forg(x, 4)])]))
    return _S.render("Optimization Results", header=(left, right), wide_rows=wide, tables=tables)


def _num(v):
    try:
        v = float(v)
    except TypeError, ValueError:
        return repr(v)
    return f"{v:.6g}" if math.isfinite(v) else repr(v)
