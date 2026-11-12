"""The dtype names (docs/design/frame.md §2.1) and the spellings accepted for them (docs/api/frame.md §1.2)."""

DTYPES = ("bool", "int64", "float64", "string")

_PYTHON_TYPES = {bool: "bool", int: "int64", float: "float64", str: "string"}


def resolve(dtype):
    """The dtype name for `dtype`: one of DTYPES, or the Python type bool, int, float or str."""
    if isinstance(dtype, str):
        if dtype in DTYPES:
            return dtype
    elif isinstance(dtype, type) and dtype in _PYTHON_TYPES:
        return _PYTHON_TYPES[dtype]
    raise TypeError(
        f"unsupported dtype {dtype!r}; use 'bool', 'int64', 'float64', 'string' or bool, int, float, str"
    )
