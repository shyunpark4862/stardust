"""The frame's dtypes (docs/design/frame.md §2.1) and how a dtype argument is
read (docs/api/frame.md §1.2): the library's dtype objects (sd.bool,
sd.int64, sd.float64, sd.string) or their names, as in the array area
(API.md §1). Columns keep the name; the public `.dtype` is the object."""

from .. import _core as _sd_core
from ._errors import named, type_name

DTYPES = ("bool", "int64", "float64", "string")

OBJECTS = {name: getattr(_sd_core, name) for name in DTYPES}
_NAMES = {obj: name for name, obj in OBJECTS.items()}


def resolve(dtype, func):
    """The dtype name for `dtype`: a frame dtype object or its name. `func` is
    the public function, for the message (as the array area's: a string or a
    type shown as itself, anything else by its type)."""
    if isinstance(dtype, str):
        if dtype in DTYPES:
            return dtype
    elif type(dtype) is type(OBJECTS["bool"]) and dtype in _NAMES:
        return _NAMES[dtype]
    if type(dtype) is type(OBJECTS["bool"]):
        what = f"sd.{dtype}"  # a dtype the frame does not have: sd.complex128
    else:
        # a string as itself, a type by its full name (numpy.int64: not to be read as sd.int64;
        # builtins bare), else the value's type: the array area's wording (bind/pyutil.cpp)
        if isinstance(dtype, str):
            what = repr(dtype)
        elif isinstance(dtype, type):
            what = dtype.__qualname__ if dtype.__module__ == "builtins" else f"{dtype.__module__}.{dtype.__qualname__}"
        else:
            what = type_name(dtype)
    raise TypeError(
        named(func, f"'dtype' must be sd.bool, sd.int64, sd.float64 or sd.string (or their names), not {what}")
    )


def obj(name):
    """The dtype object for a column's dtype name."""
    return OBJECTS[name]


def common(names):
    """The dtype name that dtype names `names` promote to: bool < int64 <
    float64, as in the array area; None when a string meets another dtype
    (there is no object dtype). One rule for the whole frame: the C++
    promote_columns, which list inference uses too (DESIGN.md §10.1 #2)."""
    names = list(dict.fromkeys(names))
    return names[0] if len(names) == 1 else _sd_core.frame.promote_types(names)
