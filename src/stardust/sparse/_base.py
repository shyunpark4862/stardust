"""sparray: what the four formats share (docs/api/sparse.md §2, docs/design/sparse.md §2.5).

A sparse array is a Python class whose fields are sd.ndarrays (COO: data and coords; CSR and CSC:
data, indices and indptr) or, for DOK, a native hash map. The index arrays are read-only: CSR and
CSC are always canonical, and changing an index would break that (docs/design/sparse.md §2.3).
`data` is writable. Every computation is one call of the native core (stardust._core.sparse).

Each format implements the conversions (_tocoo, _tocsr, _tocsc, _todok with the copy rule of
the array area: None copies only when it must, True always, False never), _transpose, _copy,
_with_data, _resize and the counts; the methods here put the checks and the shared rules
around them. The operators, the element-wise functions and the products are in _ops.py, the
reductions in _reduce.py (milestone SP2).
"""

import math
import operator

from .. import _core as _sd
from . import _common as C
from . import _index as I
from . import _ops as O
from . import _reduce as R
from ._common import _core


class sparray:
    """The base class of coo_array, csr_array, csc_array and dok_array (isinstance checks only)."""

    __slots__ = ("_shape",)
    format = None

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "sparray(): the base class is not constructed directly; use coo_array, csr_array, csc_array or dok_array"
        )

    # ---- protocol ----

    __array_ufunc__ = None  # numpy defers to our operators instead of converting us
    __iter__ = None  # no row iteration: iter(A) is a TypeError

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            f"a sparse array is not converted to a dense array implicitly; use A.toarray() ({type(self).__name__})"
        )

    def __len__(self):
        raise TypeError("the length of a sparse array is ambiguous; use A.nnz or A.shape[0]")

    def __bool__(self):
        raise TypeError("the truth value of a sparse array is ambiguous")

    def __repr__(self):
        n = self.nnz
        return (
            f"<{type(self).__name__} of shape {self._shape}, {self.dtype}, {n} stored element{'' if n == 1 else 's'}>"
        )

    __str__ = __repr__

    # ---- attributes ----

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return math.prod(self._shape)

    @property
    def T(self):
        if len(self._shape) != 2:
            raise ValueError(f".T: the array must have 2 dimensions, got {len(self._shape)}")
        return self._transpose()

    @property
    def mT(self):
        if len(self._shape) != 2:
            raise ValueError(f".mT: the array must have at least 2 dimensions, got {len(self._shape)}")
        return self._transpose()

    @property
    def real(self):
        if self.dtype == _sd.complex128:
            return self._with_data(_contiguous(_sd.real(self._values())))
        return self._with_data(self._values(), share=True)

    @property
    def imag(self):
        return self._with_data(_contiguous(_sd.imag(self._values())))

    # ---- conversions ----

    def toarray(self):
        """The dense array: a new C-ordered sd.ndarray, zeros plus the stored values (a stored -0.0
        is +0.0 there, as in scipy; duplicates of a COO array add up in stored order)."""
        return self._toarray()

    def tocoo(self, *, copy=None):
        return self._convert("coo", C.copy_arg(copy, "tocoo"), "tocoo")

    def tocsr(self, *, copy=None):
        return self._convert("csr", C.copy_arg(copy, "tocsr"), "tocsr")

    def tocsc(self, *, copy=None):
        return self._convert("csc", C.copy_arg(copy, "tocsc"), "tocsc")

    def todok(self, *, copy=None):
        return self._convert("dok", C.copy_arg(copy, "todok"), "todok")

    def asformat(self, format, /, *, copy=None):
        copy = C.copy_arg(copy, "asformat")
        if format is None:
            return self._copy() if copy else self
        if not isinstance(format, str):
            raise TypeError(
                f"asformat(): 'format' must be 'coo', 'csr', 'csc', 'dok' or None, not {C.type_name(format)}"
            )
        if format not in C.FORMATS:
            raise ValueError(f"asformat(): 'format' must be 'coo', 'csr', 'csc', 'dok' or None, not {format!r}")
        return self._convert(format, copy, "asformat")

    def _convert(self, fmt, copy, func):
        if fmt == "csc" and len(self._shape) != 2:
            raise ValueError(f"{func}(): a csc_array has 2 dimensions, the array has {len(self._shape)}")
        if fmt == self.format:
            return self._copy() if copy else self
        try:
            return getattr(self, "_to" + fmt)(copy, func)
        except (MemoryError, ValueError) as e:  # a kernel's allocation failure, named by the caller
            r = C.allocation_error(func, e)
            if r is None:
                raise
            raise r from None

    def to_scipy(self, *, copy=None):
        """The scipy.sparse array of the same format (csr_array, ...). The values and index arrays
        are numpy views of this array's memory (the index views read-only) unless copy=True; a
        dok_array is always copied (copy=False is then a ValueError)."""
        from ._interop import to_scipy

        return to_scipy(self, C.copy_arg(copy, "to_scipy"))

    def astype(self, dtype, /, *, copy=True):
        copy = C.copy_arg(copy, "astype")
        target = _sd.astype(_sd.empty(0, dtype=self.dtype), dtype, copy=copy).dtype  # the array's checks
        if target == self.dtype:
            return self._copy() if copy else self
        return self._astype(target)

    def _astype(self, dtype):
        return self._with_data(_sd.astype(self._values(), dtype))

    def copy(self):
        """A copy whose values share no memory with this array (the read-only index arrays may be
        shared: they never change)."""
        return self._copy()

    def transpose(self):
        """The transpose (.T) of a 2-D array; a 1-D array itself."""
        if len(self._shape) == 1:
            return self
        return self._transpose()

    def conj(self):
        """The complex conjugate, a new array (a copy for a real dtype)."""
        if self.dtype == _sd.complex128:
            return self._with_data(_sd.conj(self._values()))
        return self._copy()

    def reshape(self, shape, /, *, copy=None):
        """The same values in C order in a new shape (one entry may be -1): a coo_array (scipy's
        result for every format), or the array itself when the shape is unchanged."""
        copy = C.copy_arg(copy, "reshape")
        new = _reshape_target(shape, self._shape)
        if new == self._shape:
            return self._copy() if copy else self
        coo = self._convert("coo", copy, "reshape")
        major, minor = coo._split()
        rows, cols = _core._reshape_coords(major, minor, self._shape, new)
        coords = (rows, cols) if rows is not None else (cols,)
        from ._coo import coo_array

        return coo_array._new(new, coords, coo._data, True if coo._canonical is True else None)

    def resize(self, shape, /):
        """Changes the shape in place: entries outside it are dropped, new rows and columns are
        empty. The number of dimensions stays (1-D stays 1-D)."""
        new = C.shape_arg(shape, "resize")
        if len(new) != len(self._shape):
            raise ValueError(f"resize(): the shape must have {len(self._shape)} dimensions, got {len(new)}")
        C.sized("resize", self._resize, new)

    def nonzero(self):
        """The positions of the nonzero values: one int64 array per axis, in row-major order (a
        coo_array: in stored order, repeated positions as stored)."""
        coo = self._nonzero_coo()
        major, minor = coo._split()
        kept = _core._filter_coo(major, minor, coo._data, 2**63 - 1, 2**63 - 1, drop_zeros=True)
        if kept is not None:
            major, minor = kept[0], kept[1]
        if major is None:
            return (C.writable(minor),)
        return (C.writable(major), C.writable(minor))

    def _nonzero_coo(self):
        return self._convert("coo", None, "nonzero")

    def count_nonzero(self, *, axis=None, keepdims=False):
        """How many values are nonzero (duplicates of a COO array summed first): a Python int, or
        per index of the other axis an int64 sd.ndarray; keepdims as sd.count_nonzero's (and the
        other reductions', SP114): the reduced axis kept with length 1 (all of them for axis=None,
        a dense int64 array)."""
        if keepdims is not True and keepdims is not False:
            raise TypeError(f"count_nonzero(): 'keepdims' must be True or False, not {C.type_name(keepdims)}")
        nd = len(self._shape)
        if axis is not None:
            if not C.is_int(axis):
                raise TypeError(f"count_nonzero(): 'axis' must be an int or None, not {C.type_name(axis)}")
            a = axis.__index__()
            if not -nd <= a < nd:
                raise ValueError(f"count_nonzero(): axis {a} is out of bounds for an array of dimension {nd}")
            axis = a % nd
            if nd == 1:
                axis = None
        r = self._count_nonzero(axis)
        if not keepdims:
            return r
        if axis is None:
            return _sd.full((1,) * nd, r, dtype=_sd.int64)
        shape = list(self._shape)
        shape[axis] = 1
        return r.reshape(tuple(shape))

    # ---- indexing (docs/api/sparse.md §4.1; the rules in _index.py) ----

    def __getitem__(self, key):
        return I.getitem(self, key)

    def __setitem__(self, key, value):
        I.setitem(self, key, value)

    def diagonal(self, *, k=0):
        """The k-th diagonal (elements A[i, i + k]) as a dense 1-D sd.ndarray: the values of
        A.toarray() there."""
        return I.diagonal(self, k)

    def trace(self, *, k=0):
        """The sum of the k-th diagonal (a Python scalar): sd.sum of diagonal(k=k)."""
        return I.trace(self, k)

    def setdiag(self, values, /, *, k=0):
        """Sets the k-th diagonal in place: a number for all of it, or a 1-D array for its first
        len(values) elements (longer values are cut)."""
        I.setdiag(self, values, k)

    # ---- operators (docs/api/sparse.md §3.1; the rules in _ops.py) ----

    __hash__ = None  # arrays are mutable containers

    def _add_sub(self, other, op, reflected):
        if O.is_sparse(other):
            return O.sparse_sparse(self, other, op)
        if O.is_number(other):
            if other != 0 or other != other:
                raise ValueError(_nonzero_scalar_message(op, reflected))
            f = getattr(_sd, op)
            return O.with_values(self, (lambda v: f(other, v)) if reflected else (lambda v: f(v, other)), dedupe=False)
        if isinstance(other, _sd.ndarray):
            return O.dense_result(self, other, getattr(_sd, op), reflected)
        return self._other(other, "+" if op == "add" else "-", reflected)

    def _other(self, other, symbol, reflected=False):
        """An operand we do not take: NotImplemented (Python raises TypeError), or that TypeError
        raised here for a numpy array, whose own operator would answer with numpy's message."""
        if O.is_numpy_array(other):
            raise O.refused(other, self, symbol) if reflected else O.refused(self, other, symbol)
        return NotImplemented

    def __add__(self, other):
        return self._add_sub(other, "add", False)

    def __radd__(self, other):
        return self._add_sub(other, "add", True)

    def __sub__(self, other):
        return self._add_sub(other, "subtract", False)

    def __rsub__(self, other):
        return self._add_sub(other, "subtract", True)

    def _mul(self, other, reflected, func=None):
        # an operator calls the array's operator (its errors carry no function name: API.md §2),
        # the method A.multiply the array's function multiply (named "multiply(): ...")
        mul = _sd.multiply if func else operator.mul
        if O.is_number(other):
            f = (lambda v: mul(other, v)) if reflected else (lambda v: mul(v, other))
            return O.with_values(self, f, dedupe=False)
        if isinstance(other, _sd.ndarray):
            return O.dense_pattern_op(self, other, mul, reflected, func)
        if O.is_sparse(other):
            return (
                O.sparse_sparse(other, self, "multiply", func)
                if reflected
                else O.sparse_sparse(self, other, "multiply", func)
            )
        return NotImplemented if func else self._other(other, "*", reflected)

    def __mul__(self, other):
        return self._mul(other, False)

    def __rmul__(self, other):
        return self._mul(other, True)

    def __truediv__(self, other):
        if O.is_number(other):
            return O.with_values(self, lambda v: v / other, dedupe=False)
        if isinstance(other, _sd.ndarray):
            return O.dense_pattern_op(self, other, operator.truediv)
        if O.is_sparse(other):  # dense: 0 / 0 is NaN where neither stores a value
            O.check_same_shape(self, other)
            return self.toarray() / other.toarray()
        return self._other(other, "/")

    def __pow__(self, other):
        if not O.is_number(other):
            return self._other(other, "** or pow()")
        return self._power(other, None)

    def __matmul__(self, other):
        if isinstance(other, _sd.ndarray):
            return O.matmul_dense(self, other)
        if O.is_sparse(other):
            return O.matmul_sparse(self, other)
        return self._other(other, "@")

    def __rmatmul__(self, other):
        if isinstance(other, _sd.ndarray):
            return O.rmatmul_dense(self, other)
        return self._other(other, "@", True)

    def _compare(self, other, op):
        if O.is_sparse(other):
            return O.sparse_sparse(self, other, op)
        if O.is_number(other):
            return O.scalar_compare(self, other, op)
        if isinstance(other, _sd.ndarray):
            return O.dense_result(self, other, O._OPERATORS[op])
        return NotImplemented  # a numpy array answers with Python's "'<' not supported ..." itself

    def __eq__(self, other):
        r = self._compare(other, "equal")
        return _equality(self, other, r, "==") if r is NotImplemented else r

    def __ne__(self, other):
        r = self._compare(other, "not_equal")
        return _equality(self, other, r, "!=") if r is NotImplemented else r

    def __lt__(self, other):
        return self._compare(other, "less")

    def __le__(self, other):
        return self._compare(other, "less_equal")

    def __gt__(self, other):
        return self._compare(other, "greater")

    def __ge__(self, other):
        return self._compare(other, "greater_equal")

    def __neg__(self):
        return O.with_values(self, _sd.negative, dedupe=False)

    def __abs__(self):
        return O.with_values(self, _sd.abs, dedupe=True, dok_csr=True)

    # ---- element-wise methods (docs/api/sparse.md §3.2) ----

    def multiply(self, other, /):
        """Element-wise product with a sparse array, a dense sd.ndarray (broadcast to this shape) or
        a number: a sparse array (the format of this one for a dense or a number)."""
        r = self._mul(other, False, "multiply")
        if r is NotImplemented:
            raise TypeError(_operand_message("multiply", other))
        return r

    def maximum(self, other, /):
        """Element-wise maximum with a sparse array, a dense sd.ndarray or a number."""
        return self._extreme(other, "maximum")

    def minimum(self, other, /):
        """Element-wise minimum with a sparse array, a dense sd.ndarray or a number."""
        return self._extreme(other, "minimum")

    def _extreme(self, other, op):
        if O.is_sparse(other):
            return O.sparse_sparse(self, other, op, op)
        if O.is_number(other):
            return O.scalar_extreme(self, other, op, op)
        if isinstance(other, _sd.ndarray):
            return O.dense_result(self, other, getattr(_sd, op), func=op)
        raise TypeError(_operand_message(op, other))

    def power(self, n, /):
        """Element-wise power x ** n of the stored values, n > 0 (the implicit zeros stay 0)."""
        return self._power(n, "power")

    def _power(self, n, func):
        head = f"{func}(): " if func else ""
        if not O.is_number(n) or isinstance(n, complex) or type(n).__name__.startswith("complex"):
            raise TypeError(f"{head}'n' must be a real number, not {C.type_name(n)}")
        if n == 0:
            raise ValueError(
                f"{head}the power 0 is not supported: the result would not be sparse (0 ** 0 is 1); "
                "use A.toarray() ** 0"
            )
        if not n > 0:
            raise ValueError(f"{head}'n' must be positive: 0 ** {n} is not 0, the result would not be sparse")
        if func:
            f = lambda v: C.renamed(func, _sd.pow, v, n)  # noqa: E731  (the method's name, not pow's)
        else:
            f = lambda v: v**n  # noqa: E731  (the operator: no function name)
        return O.with_values(self, f, dedupe=True, dok_csr=True)

    # ---- reductions (docs/api/sparse.md §3.3) ----

    def sum(self, *, axis=None, dtype=None, keepdims=False):
        """The sum of all values (a Python scalar) or along an axis (a dense sd.ndarray)."""
        return R.sum_(self, axis, dtype, keepdims)

    def mean(self, *, axis=None, keepdims=False):
        """The mean over all elements, implicit zeros included (a Python scalar), or along an axis."""
        return R.mean(self, axis, keepdims)

    def max(self, *, axis=None, keepdims=False, explicit=False):
        """The largest value, the implicit zeros included unless explicit (NaN propagates)."""
        return R.minmax(self, "max", axis, keepdims, explicit)

    def min(self, *, axis=None, keepdims=False, explicit=False):
        """The smallest value, the implicit zeros included unless explicit (NaN propagates)."""
        return R.minmax(self, "min", axis, keepdims, explicit)

    def nanmax(self, *, axis=None, keepdims=False, explicit=False):
        """The largest value, NaN skipped."""
        return R.minmax(self, "nanmax", axis, keepdims, explicit)

    def nanmin(self, *, axis=None, keepdims=False, explicit=False):
        """The smallest value, NaN skipped."""
        return R.minmax(self, "nanmin", axis, keepdims, explicit)

    def argmax(self, *, axis=None, keepdims=False, explicit=False):
        """The position of the largest value (flat C order for axis=None): the first of ties, the
        first NaN if any."""
        return R.argminmax(self, "argmax", axis, keepdims, explicit)

    def argmin(self, *, axis=None, keepdims=False, explicit=False):
        """The position of the smallest value (flat C order for axis=None): the first of ties, the
        first NaN if any."""
        return R.argminmax(self, "argmin", axis, keepdims, explicit)


def _make_function(name):
    def method(self):
        return O.function(self, name)

    method.__name__ = method.__qualname__ = name
    method.__doc__ = (
        f"Element-wise {name} of the stored values, bit for bit the array's function on them (f(0) = 0, so "
        "the implicit zeros stay 0); a COO array's repeated positions are summed first."
    )
    return method


for _name in O.FUNCTIONS:
    setattr(sparray, _name, _make_function(_name))
del _name


def _nonzero_scalar_message(op, reflected):
    if op == "add":
        return "adding a nonzero scalar to a sparse array is not supported; use A.toarray() + c"
    if reflected:
        return "subtracting a sparse array from a nonzero scalar is not supported; use c - A.toarray()"
    return "subtracting a nonzero scalar from a sparse array is not supported; use A.toarray() - c"


def _operand_message(func, other):
    return f"{func}(): 'other' must be a sparse array, an sd.ndarray or a number, not {C.type_name(other)}"


def _equality(self, other, refused, symbol):
    """== and != of an operand we do not take: the other side's reflected comparison, else a
    TypeError (Python's fallback to identity would quietly give False; the array's rule)."""
    reflected = getattr(type(other), "__eq__" if symbol == "==" else "__ne__", None)
    if reflected is not None:
        r = reflected(other, self)
        if r is not NotImplemented:
            return r
    raise TypeError(f"'{symbol}' not supported between instances of '{type(self).__name__}' and '{C.type_name(other)}'")


def _contiguous(x):
    return x if C.contiguous(x) else x.copy()


def _reshape_target(shape, old):
    """The new shape of reshape: ints, at most one -1, 1 or 2 dimensions, the same size."""
    if C.is_int(shape):
        dims = (operator.index(shape),)
    elif isinstance(shape, tuple) and all(C.is_int(d) for d in shape):
        dims = tuple(operator.index(d) for d in shape)
    else:
        what = (
            f"a tuple holding {C.type_name(next(d for d in shape if not C.is_int(d)))}"
            if isinstance(shape, tuple)
            else C.type_name(shape)
        )
        raise TypeError(f"reshape(): 'shape' must be an int or a tuple of ints, not {what}")
    C.check_ndim(len(dims), "reshape", (1, 2), what="'shape'")
    if any(d > 2**63 - 1 or d < -(2**63) for d in dims):
        raise OverflowError("reshape(): 'shape' is out of range")
    size = math.prod(old)
    unknown = [i for i, d in enumerate(dims) if d == -1]
    if len(unknown) > 1:
        raise ValueError("reshape(): can only specify one unknown dimension")
    if any(d < -1 for d in dims):
        raise ValueError("reshape(): negative dimensions are not allowed")
    if unknown:
        rest = math.prod(d for d in dims if d != -1)
        if rest == 0 or size % rest:
            raise ValueError(f"reshape(): cannot reshape array of size {size} into shape {dims}")
        dims = tuple(size // rest if d == -1 else d for d in dims)
        if any(d > 2**63 - 1 for d in dims):  # a 1-D shape of more than 2**63 - 1 elements
            raise OverflowError("reshape(): 'shape' is out of range")
    if math.prod(dims) != size:
        raise ValueError(f"reshape(): cannot reshape array of size {size} into shape {dims}")
    return dims
