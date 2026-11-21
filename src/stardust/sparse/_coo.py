"""coo_array: coordinates and values (docs/api/sparse.md §2).

`coords` holds one read-only int64 array per axis and `data` the values. Unlike CSR and CSC the
entries may be in any order and repeat a position (docs/design/sparse.md §2.3): a repeated
position means the sum of its values, added left to right in stored order. `has_canonical_format`
says whether the entries are in row-major order without repeats; it is found out when first asked
and kept (the index arrays never change).
"""

from .. import _calls
from .. import _core as _sd
from . import _common as C
from ._base import sparray
from ._common import _core


class coo_array(sparray):
    __slots__ = ("_coords", "_data", "_canonical")
    format = "coo"

    @_calls.method("coo_array")
    def __init__(self, arg, /, *, shape=None, dtype=None, copy=None):
        from ._interop import build

        self._take(build("coo", arg, shape, dtype, copy, "coo_array"))

    @classmethod
    def _new(cls, shape, coords, data, canonical):
        obj = cls.__new__(cls)
        obj._shape = shape
        obj._coords = coords
        obj._data = data
        obj._canonical = canonical
        return obj

    def _take(self, other):
        self._shape, self._coords, self._data, self._canonical = (
            other._shape, other._coords, other._data, other._canonical
        )  # fmt: skip

    # ---- attributes ----

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):  # A.data *= 2 assigns the array it changed in place
        self._data = C.new_values(value, self._data)

    @property
    def coords(self):
        return self._coords

    @property
    def nnz(self):
        return self._data.shape[0]

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def has_canonical_format(self):
        if self._canonical is None:
            major, minor = self._split()
            self._canonical = _core._is_canonical(major, minor)
        return self._canonical

    def _values(self):
        return self._data

    def _split(self):
        """(major, minor) for the kernels: (row, col), or (None, index) for 1-D."""
        return (self._coords[0], self._coords[1]) if len(self._coords) == 2 else (None, self._coords[0])

    # ---- per format ----

    def _with_data(self, data, share=False):
        return coo_array._new(self._shape, self._coords, data, self._canonical)

    def _copy(self):
        return coo_array._new(self._shape, self._coords, self._data.copy(), self._canonical)

    def _astype(self, dtype):
        # scipy converts, then sums the repeats (in the new dtype): the result is canonical
        out = coo_array._new(self._shape, self._coords, _sd.astype(self._data, dtype), self._canonical)
        out.sum_duplicates()
        return out

    def _transpose(self):
        return coo_array._new(self._shape[::-1], self._coords[::-1], self._data, None)

    def _summed(self):
        """(major, minor, data) in canonical order, duplicates summed (new arrays when needed)."""
        major, minor = self._split()
        if self.has_canonical_format:
            return major, minor, self._data
        n_major = self._shape[0] if major is not None else 1
        return _core._sum_duplicates(n_major, major, minor, self._data)

    def _compressed(self, by_rows, copy, func):
        major, minor = self._split()
        if not by_rows:
            major, minor = minor, major
        n_major = self._shape[0 if by_rows else 1] if major is not None else 1
        indptr, indices, out = _core._compress(n_major, major, minor, self._data, share=copy is not True)
        if copy is False and out is not self._data:
            C.check_copy_false(copy, func, "sorting or summing duplicates")
        return indptr, indices, out

    def _tocsr(self, copy, func):
        from ._compressed import csr_array

        return csr_array._new(self._shape, *self._compressed(True, copy, func))

    def _tocsc(self, copy, func):
        from ._compressed import csc_array

        return csc_array._new(self._shape, *self._compressed(False, copy, func))

    def _todok(self, copy, func):
        from ._dok import dok_array

        C.check_copy_false(copy, func, "a dok_array")
        major, minor, data = self._summed()
        return dok_array._from_coo(self._shape, major, minor, data)

    def _toarray(self):
        major, minor = self._split()
        return _core._coo_to_dense(self._shape, major, minor, self._data, distinct=self.has_canonical_format)

    def _resize(self, shape):
        major, minor = self._split()
        bounds = (shape[0], shape[1]) if len(shape) == 2 else (2**63 - 1, shape[0])
        kept = _core._filter_coo(major, minor, self._data, *bounds, drop_zeros=False)
        if kept is not None:
            self._set_entries(kept)
            if self._canonical is False:
                self._canonical = None  # dropping entries can remove the repeats
        self._shape = shape

    def _set_entries(self, kept):
        rows, cols, data = kept
        self._coords = (rows, cols) if rows is not None else (cols,)
        self._data = data

    def _count_nonzero(self, axis):
        major, minor, data = self._summed()
        if axis is None:
            return _core._count_nonzero(data)
        # axis 0 counts per column, axis 1 per row
        return _core._bincount_nonzero(minor if axis == 0 else major, data, self._shape[1 - axis])

    # ---- in place ----

    def sum_duplicates(self):
        """Sorts the entries in row-major order and sums repeated positions (left to right in
        stored order), in place."""
        if self.has_canonical_format:
            return
        self._set_entries(self._summed())
        self._canonical = True

    def eliminate_zeros(self):
        """Removes the stored zeros, in place (repeated positions are not summed first)."""
        major, minor = self._split()
        kept = _core._filter_coo(major, minor, self._data, 2**63 - 1, 2**63 - 1, drop_zeros=True)
        if kept is not None:
            self._set_entries(kept)
            if self._canonical is False:
                self._canonical = None
