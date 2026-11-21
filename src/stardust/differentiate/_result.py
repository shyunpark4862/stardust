"""The result object of sd.differentiate (docs/api/differentiate.md §1.3; decisions C9, C13 and
C20 in docs/design/integrate.md §1.1).

One class for derivative, jacobian and hessian, on the base it shares with sd.optimize's
OptimizeResult and sd.integrate's IntegrateResult (stardust._results._FieldResult): a function
gives the fields it has and reading another raises AttributeError, every field is read-only, an
array field is a read-only array given without a copy (O103). The values are scipy's df and
ddf; the counts are the other areas' n_iter and nfev (C4). It does not unpack.

summary() is the shared box (API.md §2.3, C20): one derivative's value, error estimate, status
and counts in the header; a batch (derivative on an array, every jacobian and hessian, whose
entries are computed and stopped one by one: C36) one row per element.
"""

from .. import _calls, _core
from .. import _summary as _S
from .._results import _FieldResult

# The field names, in the order repr and docs/api/differentiate.md §1.3 list them.
FIELDS = ("df", "ddf", "error", "converged", "status", "n_iter", "nfev", "x")
_REPR = ("df", "ddf", "error", "converged", "n_iter", "nfev")


class DifferentiateResult(_FieldResult):
    """The result of an sd.differentiate function: df (ddf for hessian), error, converged,
    status, n_iter, nfev and the function's other fields (docs/api/differentiate.md §1.3)."""

    __slots__ = ()
    _FIELDS = FIELDS

    def __init__(self, *, batch=(), **fields):
        self._set(batch, fields, True)  # the caller's arrays are copied once

    @_calls.method("summary")
    def summary(self):
        """The result as a text table (C20): the derivative, its error estimate, convergence,
        status and counts; a batch (an array x, a Jacobian, a Hessian) one row per element."""
        return _summary(self)

    def __repr__(self):
        return self._fields_repr(_REPR)


_TITLE = "Differentiation Results"
# the columns of a batch's table: field, formatter
_COLUMNS = (
    ("x", lambda v: _S.forg(v, 4)),
    ("df", lambda v: _S.forg(v, 6)),
    ("ddf", lambda v: _S.forg(v, 6)),
    ("error", _S.forg),
    ("converged", _S.flag),
    ("status", str),
    ("n_iter", str),
    ("nfev", str),
)


def _summary(r):
    f = r._fields
    left, right = [], []
    if r._batch:
        n = _S.lanes(r._batch)
        left.append(("Batch shape:", _S.shape_text(r._batch)))
        if "converged" in f:
            conv = f["converged"]
            done = int(_core.count_nonzero(conv)) if isinstance(conv, _core.ndarray) else int(bool(conv)) * n
            right.append(("Converged:", f"{done} of {n}"))
        cols = [(k, _S.flat(f[k], n), fmt) for k, fmt in _COLUMNS if k in f]
        table = _S.Table([c[0] for c in cols], _S.lane_rows(r._batch, [(g, fm) for _, g, fm in cols]))
        return _S.render(_TITLE, header=(left, right), tables=[table])
    for k, label in (("converged", "Converged:"), ("status", "Status:")):
        if k in f:
            left.append((label, str(f[k])))
    if "x" in f and not isinstance(f["x"], _core.ndarray):
        left.append(("Point:", f"{float(f['x']):#.6g}"))
    for k, label in (("df", "Derivative:"), ("ddf", "Second derivative:")):
        if k in f and not isinstance(f[k], _core.ndarray):
            right.append((label, f"{float(f[k]):#.6g}"))
    if "error" in f and not isinstance(f["error"], _core.ndarray):
        right.append(("Error estimate:", f"{float(f['error']):#.4g}"))
    for k, label in (("n_iter", "No. Iterations:"), ("nfev", "Function Evals:")):
        if k in f and not isinstance(f[k], _core.ndarray):
            right.append((label, str(f[k])))
    return _S.render(_TITLE, header=(left, right))
