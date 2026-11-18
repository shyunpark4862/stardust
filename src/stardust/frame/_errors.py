"""Error messages name the public function the user called (API.md §2.1),
and the argument checks shared by several functions.

A helper shared by several public functions takes the function's name as
`func` and builds its messages with named(); the kernels in _core take it as
their `func=` argument. Operators and indexing pass None: their messages keep
Python's own form, without a name.

Every option of one kind goes through one check here, so the same mistake
gets the same exception and message everywhere (API.md §2, §2.1): bool
options (check_bool), int arguments (check_int), real numbers (check_real),
string options (check_choice) and lists of column names (name_list).
"""

import operator

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def named(func, msg):
    """"func(): msg", or msg alone when func is None."""
    return msg if func is None else f"{func}(): {msg}"


def type_name(value):
    """The short type name a message shows for a value of the wrong type (`not list`, `not ndarray`)."""
    return type(value).__name__


def check_bool(value, what, func):
    """A bool option takes True or False only, as in the array area (API.md §2):
    1, None and numpy.bool_ are TypeErrors."""
    if value is not True and value is not False:
        raise TypeError(named(func, f"'{what}' must be True or False, not {type_name(value)}"))
    return value


def as_index(value):
    """`value` as an int by operator.index semantics, or None when it is not
    one: a Python int or anything with __index__ (numpy.int64, ...); bool and
    numpy.bool_ are not ints (API.md §2, the array's index_value in bind/pyutil.cpp)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    try:
        return operator.index(value)  # numpy.bool_ and sd.ndarray refuse
    except TypeError:
        return None


def check_int(value, what, func):
    """An int argument (head(n), nrows, RangeIndex's start/stop/step) by the
    array area's rules (docs/api/array.md §1.3): an int by operator.index
    (numpy integers too); bool and every other type are TypeErrors, an int
    outside int64 is an OverflowError."""
    v = as_index(value)
    if v is None:
        raise TypeError(named(func, f"'{what}' must be an int, not {type_name(value)}"))
    if not _INT64_MIN <= v <= _INT64_MAX:
        raise OverflowError(named(func, f"'{what}' is out of range"))
    return v


def check_real(value, what, func):
    """A real-number argument (correction=): a float, or an int as check_int
    reads it (numpy integers too), as float; TypeError otherwise. An int
    beyond the double range is an OverflowError naming the argument, as the
    array area's (var(): 'correction' is out of range, API.md §2)."""
    if isinstance(value, float):
        return value
    v = as_index(value)
    if v is None:
        raise TypeError(named(func, f"'{what}' must be an int or a float, not {type_name(value)}"))
    try:
        return float(v)
    except OverflowError:
        raise OverflowError(named(func, f"'{what}' is out of range")) from None


def _one_of(choices):
    quoted = [repr(c) for c in choices]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]


def check_choice(value, what, choices, func):
    """A string option (how=, keep=, join=, na_position=, ...): a str that is
    not a str is a TypeError, a str that is not one of `choices` a ValueError,
    as in linalg (API.md §2)."""
    if not isinstance(value, str):
        raise TypeError(named(func, f"'{what}' must be {_one_of(choices)}, not {type_name(value)}"))
    if value not in choices:
        raise ValueError(named(func, f"'{what}' must be {_one_of(choices)}, not {value!r}"))
    return value


def name_list(value, what, func):
    """A column-name argument (`by`, `subset`, `on`, ...): one name, or a list or
    tuple of them, as a list. Other iterables (generators included) are
    TypeErrors, as in the array area, whose list arguments take a list or a
    tuple (sd.concat, axis=)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(named(func, f"'{what}' must be a column name or a list of them, not {type_name(value)}"))
