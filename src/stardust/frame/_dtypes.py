"""The frame's dtypes (docs/design/frame.md §2.1) and how a dtype argument is
read (docs/api/frame.md §1.2): the library's dtype objects (sd.bool,
sd.int64, sd.float64, sd.string) or their names, as in the array area
(API.md §1). Columns keep the name; the public `.dtype` is the object."""

from .. import _core as _sd_core

DTYPES = ("bool", "int64", "float64", "string")

OBJECTS = {name: getattr(_sd_core, name) for name in DTYPES}
_NAMES = {obj: name for name, obj in OBJECTS.items()}


def resolve(dtype):
    """The dtype name for `dtype`: a frame dtype object or its name."""
    if isinstance(dtype, str):
        if dtype in DTYPES:
            return dtype
    elif type(dtype) is type(OBJECTS["bool"]) and dtype in _NAMES:
        return _NAMES[dtype]
    raise TypeError(
        f"unsupported dtype {dtype!r}; use sd.bool, sd.int64, sd.float64, sd.string or their names"
    )


def obj(name):
    """The dtype object for a column's dtype name."""
    return OBJECTS[name]
