"""Error messages name the public function the user called (API.md §2.1),
and the argument checks shared by several functions.

A helper shared by several public functions takes the function's name as
`func` and builds its messages with named(); the kernels in _core take it as
their `func=` argument. Operators and indexing pass None: their messages keep
Python's own form, without a name."""


def named(func, msg):
    """"func(): msg", or msg alone when func is None."""
    return msg if func is None else f"{func}(): {msg}"


def name_list(value, what, func):
    """A column-name argument (`by`, `subset`, `on`, ...): one name or an iterable of them, as a list."""
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        raise TypeError(f"{func}(): '{what}' must be a column name or a list of them, not {type(value).__name__}") from None
