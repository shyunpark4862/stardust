"""The result object of sd.integrate (docs/api/integrate.md §1.3; decisions C9, C10 and C13 in
docs/design/integrate.md §1.1).

One class for every function, on the base it shares with sd.optimize's OptimizeResult and
sd.differentiate's DifferentiateResult (stardust._results._FieldResult): a function gives the
fields it has and reading another raises AttributeError, every field is read-only, an array
field is a read-only array given without a copy (O103). The value is `integral` everywhere
(`sum` for a series), the error estimate `error`, the count `nfev` (C13).

The quad family's results (quad, dblquad, tplquad, nquad, quad_vec) unpack into (integral,
error) like scipy's (y, abserr) tuples (C10, htest H7): a, b = r, r[0], r[1] and len(r) == 2.
The function marks that when it makes the result (_make(unpack=True)); the other results
(tanhsinh, nsum, cubature) do not unpack and are read by name.
"""

from .._results import _FieldResult

# The field names, in the order repr and docs/api/integrate.md §1.3 list them.
FIELDS = (
    "integral",
    "sum",
    "error",
    "converged",
    "status",
    "message",
    "nfev",
    "max_level",
    "intervals",
    "interval_integrals",
    "interval_errors",
    "n_intervals",
    "subdivisions",
)
_REPR = ("integral", "sum", "error", "converged", "nfev")
_PAIR = ("integral", "error")


class IntegrateResult(_FieldResult):
    """The result of an sd.integrate function: integral (sum for nsum), error, converged,
    status, nfev and the function's other fields (docs/api/integrate.md §1.3)."""

    __slots__ = ("_unpack",)
    _FIELDS = FIELDS

    def __init__(self, *, batch=(), **fields):
        self._set(batch, fields, True)  # the caller's arrays are copied once
        self._unpack = False

    @classmethod
    def _make(cls, *, batch=(), unpack=False, **fields):
        """The functions' constructor (no copy of their new arrays); unpack: the quad family's
        results, which unpack into (integral, error) (C10)."""
        r = super()._make(batch=batch, **fields)
        r._unpack = bool(unpack) and all(k in r._fields for k in _PAIR)
        return r

    def _pair(self):
        if not self._unpack:
            raise TypeError(
                "this IntegrateResult does not unpack (only the quad family's results do): read its fields by name"
            )
        return (self.integral, self.error)

    def __iter__(self):
        return iter(self._pair())

    def __len__(self):
        return len(self._pair())

    def __getitem__(self, i):
        return self._pair()[i]

    def __bool__(self):  # a result is a value, true whether it unpacks or not (len() may raise)
        return True

    def __repr__(self):
        return self._fields_repr(_REPR)
