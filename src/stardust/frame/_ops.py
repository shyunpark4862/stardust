"""Element-wise operators for Series and DataFrame (docs/api/frame.md §3).

The operator methods are generated here and attached to the classes, so
Series and DataFrame share one definition of what each operator does.
Only operators exist; there are no add/sub/... methods (docs/api/frame.md §1.1).
"""

import sys

from . import _core
from ._errors import type_name
from ._na import NA

Column = _core.Column

# Python operator -> kernel name. The reflected forms (1 - s) swap operands;
# comparisons need none, since Python itself turns `1 < s` into `s > 1`.
ARITH = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "truediv": "truediv",
    "floordiv": "floordiv",
    "mod": "mod",
    "pow": "pow",
}
# & | ^ are Kleene logic on bool and bitwise on int64; << >> take int64 only
# (docs/api/frame.md §3.4). The kernel picks by dtype.
BITWISE = {"and": "and", "or": "or", "xor": "xor", "lshift": "lshift", "rshift": "rshift"}
COMPARE = ("eq", "ne", "lt", "le", "gt", "ge")


def is_operand_scalar(value):
    """Whether an operator takes `value` as a scalar operand. numpy complex,
    datetime64, timedelta64, void and bytes_ scalars are not, as Python's
    complex is not: the operator then raises Python's TypeError."""
    np = sys.modules.get("numpy")
    if value is NA or isinstance(value, (bool, int, float, str)):
        return True
    return (
        np is not None
        and isinstance(value, np.generic)
        and not isinstance(value, (np.complexfloating, np.datetime64, np.timedelta64, np.void, np.bytes_))
    )


def scalar_column(value, like):
    """A one-row Column for a scalar operand. NA takes the dtype of the other
    side (`like`), so the result has the dtype the operation would have had."""
    if value is NA:
        return Column.from_list([None], NA, like)
    if like == "float64" and isinstance(value, int) and not isinstance(value, bool) and not -(2**63) <= value < 2**63:
        return Column.from_list([value], NA, "float64")  # an int outside int64 becomes a float, as in the array area
    return Column.from_list([value], NA, None)  # bool, int64 (OverflowError outside it), float64 or string


def apply(op, a, b, scalar_side):
    return _core.binary_op(op, a, b, scalar_side)


SYMBOLS = {"eq": "==", "ne": "!="}
# The symbols of Python's own TypeError for an operand no side takes.
OPERATOR_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "** or pow()",
    "and": "&",
    "or": "|",
    "xor": "^",
    "lshift": "<<",
    "rshift": ">>",
}
ORDER_SYMBOLS = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def not_supported(op, a, b):
    """== and != with an operand the other operators reject (a list, a numpy
    array, None, ...): a TypeError, as < gives, not Python's fallback to
    identity (DESIGN.md §10.1 #1, as in the array area)."""
    return TypeError(f"'{SYMBOLS[op]}' not supported between instances of {type_name(a)!r} and {type_name(b)!r}")


def is_rejected_numpy_scalar(value):
    """A numpy scalar the operators do not take (complex, datetime64, ...)."""
    np = sys.modules.get("numpy")
    return np is not None and isinstance(value, np.generic) and not is_operand_scalar(value)


def python_error(op, a, b):
    """Python's TypeError for `a op b` when neither side takes the other: what `s + 1j` gives.
    A numpy scalar would otherwise answer through its ufunc ("operand 'Series' does not support
    ufuncs"), since Series and DataFrame set __array_ufunc__ = None."""
    if op in ORDER_SYMBOLS:
        return TypeError(
            f"'{ORDER_SYMBOLS[op]}' not supported between instances of {type_name(a)!r} and {type_name(b)!r}"
        )
    return TypeError(f"unsupported operand type(s) for {OPERATOR_SYMBOLS[op]}: {type_name(a)!r} and {type_name(b)!r}")


def make_binary(op, reflected, defer):
    """The method for operator `op`. `defer`: the types to leave the operation
    to (NotImplemented), so their reflected method answers."""
    if op in SYMBOLS:

        def method(self, other):
            result = self._binary(other, op, reflected)
            if result is NotImplemented and not isinstance(other, defer()):
                raise not_supported(op, self, other)
            return result
    elif reflected:

        def method(self, other):
            return self._binary(other, op, reflected)
    else:

        def method(self, other):
            result = self._binary(other, op, reflected)
            if result is NotImplemented and is_rejected_numpy_scalar(other):
                raise python_error(op, self, other)
            return result

    return method


def attach(cls, defer):
    """The operator methods of Series or DataFrame."""
    for name, op in {**ARITH, **BITWISE}.items():
        setattr(cls, f"__{name}__", make_binary(op, False, defer))
        setattr(cls, f"__r{name}__", make_binary(op, True, defer))
    for name in COMPARE:
        setattr(cls, f"__{name}__", make_binary(name, False, defer))
