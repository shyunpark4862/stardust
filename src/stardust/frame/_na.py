"""The missing value (docs/api/frame.md §1.4): one singleton, ``sd.frame.NA``, for every dtype.

NA in an operation behaves as it does in a one-row Series (DESIGN.md §10.1
#5): `NA op x` is `Series([x])`'s operation with NA on that side, unwrapped.
So arithmetic and comparisons give NA, `|`, `&` with bools follow Kleene
logic (NA | True is True, NA & False is False), and what the Series rejects
(NA - "a", NA & 1.5) is the same TypeError. NA with NA is NA for every
operator, == included: NA is not known to equal anything, itself included.
Test for NA with `x is NA`.
"""

import sys


class NAType:
    _instance = None
    __slots__ = ()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<NA>"

    def __bool__(self):
        raise TypeError("the truth value of NA is ambiguous")

    def __reduce__(self):  # pickling keeps the singleton
        return "NA"

    # == does not say whether two objects are the same: NA keeps the identity
    # hash, so dict keys and sets of NA work (they look up by identity first).
    __hash__ = object.__hash__
    __array_ufunc__ = None  # numpy scalars defer to our reflected operators


def _operand(value):
    """The one-row Column of a scalar operand of NA, or None for a type the
    Series operators do not take (Python then raises its own TypeError)."""
    from . import _core

    np = sys.modules.get("numpy")
    if isinstance(value, (bool, int, float, str)) or (np is not None and isinstance(value, np.generic)):
        return _core.Column.from_list([value], NA, None)  # as Series([value]): OverflowError past int64
    return None


def _binary(op, reflected):
    from . import _ops

    def method(self, other):
        from . import _core

        if other is NA:
            return NA  # every operator: there is some dtype where it is defined, and the result is NA
        b = _operand(other)
        if b is None:
            if op in _ops.SYMBOLS and not _defers(other):
                raise _ops.not_supported(op, self, other)
            return NotImplemented  # a Series or DataFrame answers with its reflected method
        a = _core.Column.from_list([None], NA, b.dtype)  # NA takes the other side's dtype, as in a Series
        if reflected:
            a, b = b, a
        return _core.binary_op(op, a, b, 0).to_list(NA)[0]

    return method


def _defers(other):
    from ._frame import DataFrame
    from ._series import Series

    return isinstance(other, (Series, DataFrame))


def _unary(self):
    return NA  # -NA, +NA, abs(NA), ~NA


def _attach():
    from . import _ops

    for name, op in {**_ops.ARITH, **_ops.BITWISE}.items():
        setattr(NAType, f"__{name}__", _binary(op, False))
        setattr(NAType, f"__r{name}__", _binary(op, True))
    for name in _ops.COMPARE:  # Python reflects comparisons itself (1 < NA is NA > 1)
        setattr(NAType, f"__{name}__", _binary(name, False))
    for name in ("neg", "pos", "abs", "invert"):
        setattr(NAType, f"__{name}__", _unary)


NA = NAType()
