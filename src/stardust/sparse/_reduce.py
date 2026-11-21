"""The reductions of sd.sparse: sum, mean, max, min, nanmax, nanmin, argmax, argmin (docs/api/sparse.md
§3.3, docs/design/sparse.md §2.4, §2.6, §3.3).

  - axis=None gives a Python scalar (keepdims: a dense array of 1s in the shape), an axis a dense
    sd.ndarray (keepdims as the array's). A 1-D array's axis 0 is the whole array.
  - The sums follow docs/design/sparse.md §2.6: the whole sum is the pairwise sum of the stored
    values in C order (sd.sum of a CSR's data), an axis sum adds each line in index order (native
    kernels). The values are those of the array itself: a COO array's repeated positions are
    summed first (in stored order), whatever the format the bits are the same.
  - The means of bool and int64 sum exactly and divide once (as the array's mean).
  - max/min propagate NaN, nanmax/nanmin skip it; the implicit zeros take part unless
    explicit=True. argmax/argmin count the implicit zeros as positions: ties go to the first
    position in C order, a NaN wins at its first position (the array's rules).
  - The dtypes and the errors for a dtype are the array's (sum of bool is int64, max of complex is
    a TypeError).
"""

from .. import _core as _sd
from . import _common as C
from ._common import _core
from ._ops import one, oriented


def _flag(v, func, what):
    if v is True or v is False:
        return v
    raise TypeError(f"{func}(): '{what}' must be True or False, not {C.type_name(v)}")


def _axis(axis, nd, func):
    """None (the whole array) or 0/1 of a 2-D array."""
    if axis is None:
        return None
    if not C.is_int(axis):
        raise TypeError(f"{func}(): 'axis' must be an int or None, not {C.type_name(axis)}")
    a = axis.__index__()
    if not -nd <= a < nd:
        raise ValueError(f"{func}(): axis {a} is out of bounds for an array of dimension {nd}")
    a %= nd
    return None if nd == 1 else a


def _row_major_values(A):
    """The values of A in C order, repeated COO positions summed: what the whole sum adds."""
    fmt = A.format
    if fmt == "csr":
        return A._data
    if fmt == "csc":
        return A._swap(None, "sum")._data
    if fmt == "coo":
        return A._summed()[2]
    return A._parts()[2]


def _scalar_result(value, dtype, nd, keepdims):
    if not keepdims:
        return value
    return _sd.asarray([value], dtype=dtype).reshape((1,) * nd)


def _keep(result, A, axis, keepdims):
    if not keepdims:
        return result
    shape = list(A.shape)
    shape[axis] = 1
    return result.reshape(tuple(shape))


def _axis_lines(A, axis, func):
    """The compressed array whose lines run along `axis` and whether they are its own lines (True)
    or its positions (False): axis 1 reduces each row, axis 0 each column. A CSR or CSC is used as
    it is (the sums scatter along positions); other formats become CSR."""
    cs = A if A.format in ("csr", "csc") else A._tocsr(None, func)
    gather = (axis == 1) == cs._by_rows
    return cs, gather


def sum_(A, axis, dtype, keepdims):
    func = "sum"
    keepdims = _flag(keepdims, func, "keepdims")
    dtype = C.dtype_arg(dtype, func)
    nd = A.ndim
    axis = _axis(axis, nd, func)
    res = _sd.sum(_sd.zeros((1, 1), dtype=A.dtype), axis=0, dtype=dtype).dtype  # the array's rule and errors
    if axis is None:
        values = _row_major_values(A)
        total = _sd.sum(values) if dtype is None else _sd.sum(values, dtype=dtype)
        return _scalar_result(total, res, nd, keepdims)
    cs, gather = _axis_lines(A, axis, func)
    data = cs._data
    if dtype is not None and data.dtype != res:
        data = _sd.astype(data, res)
    lines, positions = cs._lines()
    out = _core._line_sum(
        cs._indptr, cs._indices, data, n_minor=positions, scatter=not gather, track=gather or cs._long_across()
    )
    if out.dtype != res:
        out = _sd.astype(out, res)
    return _keep(out, A, axis, keepdims)


def mean(A, axis, keepdims):
    func = "mean"
    keepdims = _flag(keepdims, func, "keepdims")
    nd = A.ndim
    axis = _axis(axis, nd, func)
    complex_ = A.dtype == _sd.complex128
    res = _sd.complex128 if complex_ else _sd.float64
    if axis is None:
        values = _row_major_values(A)
        n = A.size
        if A.dtype in (_sd.bool, _sd.int64):
            m = _core._exact_mean(values, n)
        else:
            s = _sd.sum(values)
            if n == 0:
                m = complex(float("nan"), float("nan")) if complex_ else float("nan")
            elif complex_:
                m = complex(s.real / n, s.imag / n)
            else:
                m = s / n
        return _scalar_result(m, res, nd, keepdims)
    cs, gather = _axis_lines(A, axis, func)
    lines, positions = cs._lines()
    count = A.shape[axis]
    if count == 0:  # 0 / 0 for every output, as the array's mean
        out = _sd.full(A.shape[1 - axis], float("nan"), dtype=res)
        return _keep(out, A, axis, keepdims)
    out = _core._line_sum(
        cs._indptr, cs._indices, cs._data, n_minor=positions, scatter=not gather, count=count,
        track=gather or cs._long_across(),
    )  # fmt: skip
    return _keep(out, A, axis, keepdims)


def _empty_error(func):
    raise ValueError(f"{func}(): undefined for an empty array (or along an empty axis)")


def minmax(A, func, axis, keepdims, explicit):
    keepdims = _flag(keepdims, func, "keepdims")
    explicit = _flag(explicit, func, "explicit")
    nd = A.ndim
    axis = _axis(axis, nd, func)
    getattr(_sd, func)(one(A.dtype))  # the array's dtype errors ("max(): not supported for complex128")
    if axis is None:
        if A.size == 0:
            _empty_error(func)
        values = _row_major_values(A)
        bounds = _sd.asarray([0, values.shape[0]])
        m = _core._line_minmax(bounds, values, A.size, op=func, explicit=explicit).tolist()[0]
        return _scalar_result(m, A.dtype, nd, keepdims)
    if A.shape[axis] == 0:
        _empty_error(func)
    cs = oriented(A, axis == 1, func)
    out = _core._line_minmax(cs._indptr, cs._data, A.shape[axis], op=func, explicit=explicit)
    return _keep(out, A, axis, keepdims)


def argminmax(A, func, axis, keepdims, explicit):
    keepdims = _flag(keepdims, func, "keepdims")
    explicit = _flag(explicit, func, "explicit")
    nd = A.ndim
    axis = _axis(axis, nd, func)
    getattr(_sd, func)(one(A.dtype))  # the array's dtype errors
    if axis is None:
        if A.size == 0:
            _empty_error(func)
        cs = oriented(A, True, func)
        if cs._data.shape[0] == 0:
            if explicit:
                raise ValueError(f"{func}(): the array stores no values (explicit=True)")
            pos = 0
        else:
            pos = _core._flat_arg(cs._indptr, cs._indices, cs._data, A.shape[-1], op=func, explicit=explicit)
        return _scalar_result(pos, _sd.int64, nd, keepdims)
    if A.shape[axis] == 0:
        _empty_error(func)
    cs = oriented(A, axis == 1, func)
    out = _core._line_arg(cs._indptr, cs._indices, cs._data, A.shape[axis], op=func, explicit=explicit)
    return _keep(out, A, axis, keepdims)
