"""dok_array: a dictionary of keys, for building a sparse array one entry at a time
(docs/api/sparse.md §2, docs/design/sparse.md §2.1).

The entries live in a native hash map (stardust._core.sparse._DokMap): `A[i, j] = v` and `A[i, j]`
take O(1). There are no dict methods (keys, items, get ...): read the entries with tocoo().
Setting a zero removes the entry (scipy's dok_array); a conversion keeps the stored zeros of its
input. Converting to the other formats sorts the keys, so the result does not depend on the order
the entries went in.
"""

from .. import _calls
from .. import _core as _sd
from . import _common as C
from . import _index as I
from ._base import sparray
from ._common import _core


class dok_array(sparray):
    __slots__ = ("_map",)
    format = "dok"

    @_calls.method("dok_array")
    def __init__(self, arg, /, *, shape=None, dtype=None, copy=None):
        from ._interop import build

        self._take(build("dok", arg, shape, dtype, copy, "dok_array"))

    @classmethod
    def _new(cls, shape, m):
        obj = cls.__new__(cls)
        obj._shape = shape
        obj._map = m
        return obj

    @classmethod
    def _from_coo(cls, shape, major, minor, data):
        """From canonical COO entries (distinct positions)."""
        return cls._new(shape, _core._DokMap.from_coo(data.dtype, shape, major, minor, data))

    @classmethod
    def _empty(cls, shape, dtype):
        return cls._new(shape, _core._DokMap(dtype, shape))

    def _take(self, other):
        self._shape, self._map = other._shape, other._map

    def __getitem__(self, key):
        if _element_key(key, len(self._shape)):
            return self._map.get(key)  # O(1)
        return I.getitem(self, key)

    def __setitem__(self, key, value):
        if _element_key(key, len(self._shape)):
            self._map.set(key, value)  # O(1)
            return
        I.setitem(self, key, value)

    # ---- attributes ----

    @property
    def nnz(self):
        return self._map.nnz

    @property
    def dtype(self):
        return self._map.dtype

    @property
    def real(self):
        return self._tocsr(None, "real").real  # scipy: a csr_array

    @property
    def imag(self):
        return self._tocsr(None, "imag").imag

    def conj(self):
        if self.dtype == _sd.complex128:
            return self._tocsr(None, "conj").conj()  # scipy: a csr_array
        return self._copy()

    # ---- per format ----

    def _parts(self):
        """(major or None, minor, data): the entries in row-major order (new arrays)."""
        if self._map.shape != self._shape:  # an invariant: a shared map is never resized in place
            raise RuntimeError("dok_array: the hash map's shape differs from the array's")
        return self._map.to_coo()

    def _copy(self):
        return dok_array._new(self._shape, self._map.copy())

    def _transpose(self):
        return dok_array._new(self._shape[::-1], self._map.transposed())

    def _astype(self, dtype):
        return dok_array._new(self._shape, self._map.astype(dtype))

    def _tocoo(self, copy, func):
        from ._coo import coo_array

        C.check_copy_false(copy, func, "converting dok to coo")
        major, minor, data = self._parts()
        return coo_array._new(self._shape, (major, minor) if major is not None else (minor,), data, True)

    def _tocsr(self, copy, func):
        from ._compressed import csr_array

        C.check_copy_false(copy, func, "converting dok to csr")
        if self._map.shape != self._shape:  # an invariant: a shared map is never resized in place
            raise RuntimeError("dok_array: the hash map's shape differs from the array's")
        return csr_array._new(self._shape, *self._map.to_csr())

    def _tocsc(self, copy, func):
        return self._tocsr(copy, func)._swap(None, func)

    def _toarray(self):
        major, minor, data = self._parts()
        return _core._coo_to_dense(self._shape, major, minor, data, distinct=True)

    def _resize(self, shape):
        # A new map: dok_array(D) shares D's map, and resizing one of the two must not change the
        # keys or the shape the other one sees (copy on resize; element writes stay shared until then).
        self._map = self._map.resized(shape)
        self._shape = shape

    def _count_nonzero(self, axis):
        return self._tocoo(None, "count_nonzero")._count_nonzero(axis)


def _element_key(key, nd):
    """Whether key names one element: an int per axis (not a bool)."""
    if nd == 1:
        return C.is_int(key)
    return type(key) is tuple and len(key) == 2 and C.is_int(key[0]) and C.is_int(key[1])
