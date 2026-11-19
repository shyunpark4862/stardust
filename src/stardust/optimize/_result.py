"""The result object of every sd.optimize function (docs/api/optimize.md §1.3,
docs/design/optimize.md §2.2).

One class for all methods. Its fields are the names of the other areas' results (O4:
converged, n_iter, nfev, njev; regression's NLSResult): scipy's success and nit are not
aliased. A method gives the fields it has; reading one it does not give raises
AttributeError (so hasattr() tells). Every field is read-only (regression G98) and an array
field is a new copy at each access (G113): changing what was returned does not change the
result.
"""

import math

from .. import _core

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
_FIELD_SET = frozenset(FIELDS)


def _fresh(v):
    """An array field as the result gives it: a new array (or a tuple of new arrays)."""
    if isinstance(v, _core.ndarray):
        return v.copy()
    if isinstance(v, tuple):
        return tuple(_fresh(e) for e in v)
    return v


def _field(name):
    def get(self):
        try:
            v = self._fields[name]
        except KeyError:
            raise AttributeError(
                f"'OptimizeResult' object has no attribute '{name}' (its method does not give it)"
            ) from None
        return _fresh(v)

    get.__name__ = name
    return property(get, doc=f"The result's '{name}' (read-only; arrays are new copies).")


class OptimizeResult:
    """The result of a solver in sd.optimize: x, fun, converged, status, message, n_iter, nfev,
    njev and the method's other fields (docs/api/optimize.md §1.3)."""

    __slots__ = ("_fields", "_batch")

    def __init__(self, *, batch=(), **fields):
        unknown = sorted(set(fields) - _FIELD_SET)
        if unknown:
            raise TypeError(f"OptimizeResult(): unexpected keyword argument {unknown[0]!r}")
        stored = {}
        for k in FIELDS:  # a fixed order, whatever the caller's
            if k in fields:
                stored[k] = _fresh(fields[k])  # the result owns its arrays
        self._fields = stored
        self._batch = tuple(batch)

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


def _num(v):
    try:
        v = float(v)
    except TypeError, ValueError:
        return repr(v)
    return f"{v:.6g}" if math.isfinite(v) else repr(v)


for _name in FIELDS:
    setattr(OptimizeResult, _name, _field(_name))
del _name
