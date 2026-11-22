"""RadiusResult: the result of the radius searches (docs/api/spatial.md §5, docs/design/spatial.md §2.5;
decisions SPT3, SPT17). radius_neighbors (SPT2) makes it; KDTree.query_ball_point and query_ball_tree
(SPT4) make it too, through the same constructor. The class is not exported (as sd.htest's results).

The constructor takes the CSR layout as it is and makes it read-only, without checking or copying:

    RadiusResult(indptr, indices, distances, shape)

indptr (int64, rows + 1), indices (int64) and distances (float64) are sd.ndarrays; row i's neighbours are
indices[indptr[i]:indptr[i + 1]] in increasing order, their distances at the same places; shape is
(rows, data points). The caller (a native search) guarantees the layout.
"""

from .. import _calls


@_calls.methods
class RadiusResult:
    """The neighbours within a radius, row by row, in the CSR layout (row i: indices[indptr[i]:indptr[i + 1]],
    increasing; their distances at the same places). tocsr() gives the sd.sparse.csr_array of the distances,
    with the zero distances (a point and its duplicates) as stored zeros."""

    __slots__ = ("_indptr", "_indices", "_distances", "_shape")

    def __init__(self, indptr, indices, distances, shape):
        for a in (indptr, indices, distances):
            a._set_readonly()
        self._indptr, self._indices, self._distances = indptr, indices, distances
        self._shape = (int(shape[0]), int(shape[1]))

    @property
    def indptr(self):
        """Where each row starts in indices and distances (int64, rows + 1), read-only."""
        return self._indptr

    @property
    def indices(self):
        """The neighbours' indices, row after row, increasing within a row (int64), read-only."""
        return self._indices

    @property
    def distances(self):
        """The neighbours' distances, at the places of indices (float64), read-only."""
        return self._distances

    @property
    def shape(self):
        """(rows, data points)."""
        return self._shape

    def tocsr(self):
        """The sd.sparse.csr_array (rows, data points) of the distances: a new array on every call (its data
        is writable); a zero distance is a stored zero (nnz counts it, count_nonzero() does not)."""
        from .. import sparse

        return sparse.csr_array((self._distances, self._indices, self._indptr), shape=self._shape)

    def __repr__(self):
        return f"RadiusResult(shape={self._shape}, nnz={int(self._indices.shape[0])})"
