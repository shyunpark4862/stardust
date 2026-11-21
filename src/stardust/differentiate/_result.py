"""The result object of sd.differentiate (docs/api/differentiate.md §1.3; decisions C9 and C13 in
docs/design/integrate.md §1.1).

One class for derivative, jacobian and hessian, on the base it shares with sd.optimize's
OptimizeResult and sd.integrate's IntegrateResult (stardust._results._FieldResult): a function
gives the fields it has and reading another raises AttributeError, every field is read-only, an
array field is a read-only array given without a copy (O103). The values are scipy's df and
ddf; the counts are the other areas' n_iter and nfev (C4). It does not unpack.
"""

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

    def __repr__(self):
        return self._fields_repr(_REPR)
