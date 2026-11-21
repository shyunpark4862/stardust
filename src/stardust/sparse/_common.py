"""Argument checks and messages shared by the sd.sparse classes (API.md §2.1, docs/api/sparse.md §1).

The dtype, copy and array arguments follow the array area's rules; where the array area already
checks an argument, its message is reused with the caller's name in front ("csr_array(): ...")."""

import numbers
import operator
import sys

from .. import _core as _sd

_core = _sd.sparse

FORMATS = ("coo", "csr", "csc", "dok")


def type_name(v):
    return _sd._type_name(v)  # numpy's types keep their module (API.md §2.1)


def renamed(func, call, *args, **kwargs):
    """call(*args, **kwargs), an error of the array area ("asarray(): ...") renamed "func(): ..."."""
    try:
        return call(*args, **kwargs)
    except (TypeError, ValueError, OverflowError) as e:
        head, sep, rest = str(e).partition("(): ")
        if sep and head.isidentifier():
            raise type(e)(f"{func}(): {rest}") from None
        raise


def allocation_error(func, e):
    """e renamed "func(): ..." when it is an allocation failure of an array kernel (MemoryError
    "cannot allocate N bytes: ...", ValueError "array is too big"), with or without the kernel's
    name in front; else None. The user called func, not the kernel (API.md §2.1)."""
    if type(e) not in (MemoryError, ValueError):
        return None
    msg = str(e)
    head, sep, rest = msg.partition("(): ")
    if not (sep and head.isidentifier()):
        rest = msg
    if rest.startswith("cannot allocate ") if type(e) is MemoryError else rest == "array is too big":
        return type(e)(f"{func}(): {rest}")
    return None


def sized(func, call, *args, **kwargs):
    """call(*args, **kwargs), an allocation failure inside named func (allocation_error)."""
    try:
        return call(*args, **kwargs)
    except (MemoryError, ValueError) as e:
        r = allocation_error(func, e)
        if r is None:
            raise
        raise r from None


def int64_arg(v, func, name):
    """An int argument (is_int's rule) as a Python int in the int64 range: OverflowError
    "func(): 'name' is out of range" outside it, as the dense functions (API.md §2)."""
    if not is_int(v):
        raise TypeError(f"{func}(): '{name}' must be an int, not {type_name(v)}")
    v = operator.index(v)
    if not -(2**63) <= v <= 2**63 - 1:
        raise OverflowError(f"{func}(): '{name}' is out of range")
    return v


def check_shape_range(dims, func):
    """OverflowError "func(): the result's shape is out of range" when a computed shape has a
    length past int64 (a product or a sum of the inputs' lengths)."""
    if any(d > 2**63 - 1 for d in dims):
        raise OverflowError(f"{func}(): the result's shape is out of range")


def copy_arg(copy, func):
    if copy is None or copy is True or copy is False:
        return copy
    raise TypeError(f"{func}(): 'copy' must be True, False or None, not {type_name(copy)}")


def dtype_arg(dtype, func):
    """None, or one of the four array dtypes (an object or its name; API.md §1)."""
    if dtype is None:
        return None
    return renamed(func, lambda: _sd.empty(0, dtype=dtype).dtype)


def is_int(v):
    """An int by operator.index's rule; bool and numpy.bool are not ints (API.md §2)."""
    if isinstance(v, bool) or type(v).__name__ == "bool" and type(v).__module__ == "numpy":
        return False
    try:
        operator.index(v)
    except TypeError:
        return False
    return True


def is_scalar(v):
    return isinstance(v, numbers.Number)


def shape_arg(shape, func, ndims=(1, 2), what="shape"):
    """A shape: an int or a tuple of ints, none negative, of 1 or 2 dimensions (`ndims`)."""
    if is_int(shape):
        dims = (shape,)
    elif isinstance(shape, tuple):
        bad = next((d for d in shape if not is_int(d)), None)
        if bad is not None:
            raise TypeError(
                f"{func}(): '{what}' must be an int or a tuple of ints, not a tuple holding {type_name(bad)}"
            )
        dims = shape
    else:
        raise TypeError(f"{func}(): '{what}' must be an int or a tuple of ints, not {type_name(shape)}")
    dims = tuple(operator.index(d) for d in dims)
    if any(d > 2**63 - 1 for d in dims):
        raise OverflowError(f"{func}(): '{what}' is out of range")
    if any(d < 0 for d in dims):
        raise ValueError(f"{func}(): negative dimensions are not allowed")
    check_ndim(len(dims), func, ndims, what=f"'{what}'")
    return dims


def check_ndim(nd, func, ndims, what="the array"):
    if nd not in ndims:
        need = "2 dimensions" if ndims == (2,) else "1 or 2 dimensions"
        raise ValueError(f"{func}(): {what} must have {need}, got {nd}")


def check_copy_false(copy, func, reason):
    if copy is False:
        raise ValueError(f"{func}(): copy=False, but {reason} requires a copy")


def contiguous(x):
    return x._flags()["c_contiguous"]


def is_scipy_sparse(x):
    sp = sys.modules.get("scipy.sparse")
    return sp is not None and sp.issparse(x)


def as_dense(arg, func):
    """A dense input (an sd.ndarray, a buffer such as a numpy array, a nested list or tuple, a
    range) as an sd.ndarray, by asarray's rules."""
    if isinstance(arg, (str, bytes)) or is_scalar(arg) or arg is None or isinstance(arg, dict):
        raise TypeError(
            f"{func}(): 'arg' must be a sparse array, an sd.ndarray, a numpy array, a sequence of numbers "
            f"or a shape tuple, not {type_name(arg)}"
        )
    if not isinstance(arg, (_sd.ndarray, list, tuple, range)) and not _has_buffer(arg):
        raise TypeError(
            f"{func}(): 'arg' must be a sparse array, an sd.ndarray, a numpy array, a sequence of numbers "
            f"or a shape tuple, not {type_name(arg)}"
        )
    return renamed(func, _sd.asarray, arg)


def _has_buffer(x):
    try:
        memoryview(x).release()
    except TypeError:
        return False
    return True


def data_array(v, func, dtype, copy, what="data"):
    """The values of a (data, ...) tuple: a 1-D contiguous sd.ndarray by asarray's rules (dtype=
    converts explicitly; copy as asarray's)."""
    if is_scalar(v) or v is None or isinstance(v, (str, bytes, dict)) or _is_sparse(v):
        raise TypeError(
            f"{func}(): '{what}' must be an sd.ndarray, a numpy array or a sequence of numbers, not {type_name(v)}"
        )
    x = renamed(func, _sd.asarray, v, dtype=dtype, copy=copy)
    if x.ndim != 1:
        raise ValueError(f"{func}(): '{what}' must have 1 dimension, got {x.ndim}")
    if not contiguous(x):
        check_copy_false(copy, func, f"a non-contiguous '{what}'")
        x = x.copy()
    return x


def index_array(v, func, what):
    """An index array: 1-D int64 by asarray's rules (int32 and the other integer buffers widen;
    an empty sequence is int64)."""
    if is_scalar(v) or v is None or isinstance(v, (str, bytes, dict)) or _is_sparse(v):
        raise TypeError(
            f"{func}(): '{what}' must be an sd.ndarray, a numpy array or a sequence of ints, not {type_name(v)}"
        )
    x = renamed(func, _sd.asarray, v)
    if x.ndim != 1:
        raise ValueError(f"{func}(): '{what}' must have 1 dimension, got {x.ndim}")
    if x.dtype != _sd.int64:
        if x.shape[0] == 0:
            return _sd.empty(0, dtype=_sd.int64)
        raise TypeError(f"{func}(): '{what}' must hold integers, not {x.dtype}")
    if not contiguous(x):
        x = x.copy()
    return x


def _is_sparse(v):
    from ._base import sparray

    return isinstance(v, sparray) or is_scipy_sparse(v)


def writable(x):
    """x for a caller to keep: a writable sd.ndarray (index arrays are read-only)."""
    return x.copy() if not x._flags()["writeable"] else x


def new_values(value, old):
    """`A.data = value`: an sd.ndarray of the same shape as the old values (any of the four dtypes;
    a non-contiguous one is copied)."""
    if value is old:
        return old
    if not isinstance(value, _sd.ndarray):
        raise TypeError(f".data: the values must be an sd.ndarray, not {type_name(value)}")
    if tuple(value.shape) != tuple(old.shape):
        raise ValueError(f".data: the values must have shape {tuple(old.shape)}, got {tuple(value.shape)}")
    return value if contiguous(value) else value.copy()
