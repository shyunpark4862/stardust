"""Element-wise operators for Series and DataFrame (API.md §12).

The operator methods are generated here and attached to the classes, so
Series and DataFrame share one definition of what each operator does.
Only operators exist; there are no add/sub/... methods (API.md §11 C7).
"""

import sys

from . import _core
from ._core import Column
from ._na import NA

# Python operator -> kernel name. The reflected forms (1 - s) swap operands;
# comparisons need none, since Python itself turns `1 < s` into `s > 1`.
ARITH = {"add": "add", "sub": "sub", "mul": "mul", "truediv": "truediv", "floordiv": "floordiv", "mod": "mod", "pow": "pow"}
LOGIC = {"and": "and", "or": "or", "xor": "xor"}
COMPARE = ("eq", "ne", "lt", "le", "gt", "ge")


def is_operand_scalar(value):
    np = sys.modules.get("numpy")
    return value is NA or isinstance(value, (bool, int, float, str)) or (np is not None and isinstance(value, np.generic))


def scalar_column(value, like):
    """A one-row Column for a scalar operand. NA takes the dtype of the other
    side (`like`), so the result has the dtype the operation would have had."""
    if value is NA:
        return Column.from_list([None], NA, like)
    return Column.from_list([value], NA, None)  # bool, int64 (range-checked), float64 or string


def apply(op, a, b, scalar_side):
    return _core.binary_op(op, a, b, scalar_side)
