"""csr_array and csc_array: compressed rows and compressed columns (docs/api/sparse.md §2).

`indptr` (read-only) gives where each line starts in `indices` (read-only) and `data`: a line is
a row of CSR, a column of CSC, and a 1-D csr_array is one line. The arrays are always canonical
(docs/design/sparse.md §2.3): within a line the indices increase strictly. The transpose of one
is the other over the same three arrays.
"""

from .. import _calls
from . import _common as C
from ._base import sparray
from ._common import _core

_LINE = 4096  # L of docs/design/sparse.md §2.6 (csrc/sparse/arith.hpp kLine)


class _compressed(sparray):
    __slots__ = ("_indptr", "_indices", "_data", "_long")
    _by_rows = True

    @classmethod
    def _new(cls, shape, indptr, indices, data, long=None):
        obj = cls.__new__(cls)
        obj._shape = shape
        obj._indptr = indptr
        obj._indices = indices
        obj._data = data
        obj._long = long
        return obj

    def _take(self, other):
        self._shape, self._indptr, self._indices, self._data, self._long = (
            other._shape, other._indptr, other._indices, other._data, other._long
        )  # fmt: skip

    def _long_across(self):
        """Whether some position (an index of the other axis) holds more than L = 4096 entries: a
        sum scattered along it is then cut into pieces (docs/design/sparse.md §2.6), and the
        scatter must count. Found once and kept (the index arrays never change)."""
        if self._long is None:
            lines, positions = self._lines()
            self._long = lines > _LINE and _core._minor_over(self._indices, positions, _LINE)
        return self._long

    # ---- attributes ----

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):  # A.data *= 2 assigns the array it changed in place
        self._data = C.new_values(value, self._data)

    @property
    def indices(self):
        return self._indices

    @property
    def indptr(self):
        return self._indptr

    @property
    def nnz(self):
        return self._data.shape[0]

    @property
    def dtype(self):
        return self._data.dtype

    def _values(self):
        return self._data

    def _lines(self):
        """(lines, positions per line): (rows, cols) of CSR, (cols, rows) of CSC, (1, n) of 1-D."""
        if len(self._shape) == 1:
            return 1, self._shape[0]
        return (self._shape[0], self._shape[1]) if self._by_rows else (self._shape[1], self._shape[0])

    # ---- per format ----

    def _with_data(self, data, share=False):
        return type(self)._new(self._shape, self._indptr, self._indices, data, self._long)

    def _copy(self):
        return type(self)._new(self._shape, self._indptr, self._indices, self._data.copy(), self._long)

    def _transpose(self):
        other = csc_array if self._by_rows else csr_array
        return other._new(self._shape[::-1], self._indptr, self._indices, self._data, self._long)

    def _coo_parts(self, copy):
        """(coords, data, canonical) of tocoo(): the major index of every entry expanded."""
        lines, _ = self._lines()
        data = self._data.copy() if copy else self._data
        if len(self._shape) == 1:
            return (self._indices,), data, True
        major = _core._expand(self._indptr, lines, self.nnz)
        if self._by_rows:
            return (major, self._indices), data, True
        return (self._indices, major), data, None

    def _tocoo(self, copy, func):
        from ._coo import coo_array

        coords, data, canonical = self._coo_parts(copy)
        return coo_array._new(self._shape, coords, data, canonical)

    def _swap(self, copy, func):
        """The other compressed format of the same array (new arrays)."""
        C.check_copy_false(copy, func, f"converting {self.format} to {'csc' if self._by_rows else 'csr'}")
        lines, positions = self._lines()
        indptr, indices, data = _core._transpose(lines, positions, self._indptr, self._indices, self._data)
        other = csc_array if self._by_rows else csr_array
        return other._new(self._shape, indptr, indices, data)

    def _todok(self, copy, func):
        from ._dok import dok_array

        C.check_copy_false(copy, func, "a dok_array")
        coords, data, _ = self._coo_parts(False)
        major, minor = (coords[0], coords[1]) if len(coords) == 2 else (None, coords[0])
        return dok_array._from_coo(self._shape, major, minor, data)

    def _toarray(self):
        return _core._to_dense(self._shape, self._indptr, self._indices, self._data, by_rows=self._by_rows)

    def _resize(self, shape):
        old_lines, _ = self._lines()
        if len(shape) == 1:
            lines, positions = 1, shape[0]
        else:
            lines, positions = (shape[0], shape[1]) if self._by_rows else (shape[1], shape[0])
        kept = _core._filter_compressed(
            old_lines, self._indptr, self._indices, self._data, lines, positions, drop_zeros=False
        )
        if kept is not None:
            self._indptr, self._indices, self._data = kept
            self._long = None
        self._shape = shape

    def _count_nonzero(self, axis):
        if axis is None:
            return _core._count_nonzero(self._data)
        lines, positions = self._lines()
        per_line = (axis == 1) == self._by_rows  # CSR rows are lines; CSC columns are
        if per_line:
            return _core._rowcount_nonzero(self._indptr, self._data)
        return _core._bincount_nonzero(self._indices, self._data, positions)

    # ---- in place ----

    def eliminate_zeros(self):
        """Removes the stored zeros (and -0.0), in place."""
        lines, positions = self._lines()
        kept = _core._filter_compressed(
            lines, self._indptr, self._indices, self._data, lines, positions, drop_zeros=True
        )
        if kept is not None:
            self._indptr, self._indices, self._data = kept
            self._long = None


class csr_array(_compressed):
    __slots__ = ()
    format = "csr"
    _by_rows = True

    @_calls.method("csr_array")
    def __init__(self, arg, /, *, shape=None, dtype=None, copy=None):
        from ._interop import build

        self._take(build("csr", arg, shape, dtype, copy, "csr_array"))

    def _tocsr(self, copy, func):
        return self._copy() if copy else self

    def _tocsc(self, copy, func):
        return self._swap(copy, func)


class csc_array(_compressed):
    __slots__ = ()
    format = "csc"
    _by_rows = False

    @_calls.method("csc_array")
    def __init__(self, arg, /, *, shape=None, dtype=None, copy=None):
        from ._interop import build

        self._take(build("csc", arg, shape, dtype, copy, "csc_array"))

    def _tocsr(self, copy, func):
        return self._swap(copy, func)

    def _tocsc(self, copy, func):
        return self._copy() if copy else self

    def _nonzero_coo(self):
        return self._swap(None, "nonzero")._tocoo(None, "nonzero")  # row-major, as scipy's
