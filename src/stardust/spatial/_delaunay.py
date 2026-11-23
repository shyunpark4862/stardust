"""Delaunay of sd.spatial (docs/api/spatial_geometry.md §3, docs/design/spatial_geometry.md §2.3, §3.3-§3.5;
milestones SPT9 (d = 2) and SPT10 (d = 3)).

A Python class holding the native triangulation (stardust._core.spatial._Delaunay, csrc/spatial/delaunay2.cpp;
the 3-D build is csrc/spatial/delaunay3.cpp), like KDTree (ARCHITECTURE §3). The points are always copied
(SPT98). The triangulation is a function of the point set (the exact predicates and the tie rule of §3.2) and
its arrays are in the canonical form of §2.2, so a pickled or copied Delaunay is rebuilt from its points into
the same object (SPT105).

The arrays the build makes (simplices, neighbors, vertex_to_simplex) are read-only attributes; the others
(convex_hull, vertex_neighbor_vertices, coplanar, transform) are made on first use and kept (SPT106, SPT132).
find_simplex answers exactly with closed simplices: the smallest simplex holding the point, -1 outside the
convex hull or for a non-finite point (SPT103).
"""

import math

from .. import _core as _sd
from .._calls import method
from . import _common as C

_core = _sd.spatial

# Bytes per input point the build holds at its peak (docs/design/spatial_geometry.md §3.11): d = 2 (SPT136) the
# copy and sort keys, the edge structure (72 per distinct point), the internal and canonical triangles; d = 3
# (SPT164) at least kPointBytes3 + kTetBytes (delaunay3.hpp) for one tetrahedron per point, the fewest a 3-D
# triangulation has. The 3-D build also counts its tetrahedra (MemoryError past physical memory's share).
_BUILD_BYTES = {2: 360, 3: 328}


def _readonly(a):
    a._set_readonly()
    return a


class Delaunay:
    """The Delaunay triangulation of `points` (n, d), d = 2 or 3, exact for every finite float64 input."""

    __slots__ = ("_tri", "_points", "_simplices", "_neighbors", "_v2s", "_lazy")

    @method("Delaunay")
    def __init__(self, points, /):
        self._build(points, -1)

    def _build(self, points, max_simplices):
        """max_simplices: the tetrahedra (ghosts included) a 3-D build may hold, -1 for physical memory's share
        (tests pass less: _limited)."""
        name = "Delaunay"
        a = C.geometry_points(points, name)
        C.check_alloc(_BUILD_BYTES[a.shape[1]] * a.shape[0], name)
        self._tri = _core._Delaunay(a, max_simplices)
        self._points = _readonly(self._tri.points())
        self._simplices = _readonly(self._tri.simplices())
        self._neighbors = _readonly(self._tri.neighbors())
        self._v2s = _readonly(self._tri.vertex_to_simplex())
        self._lazy = {}

    def _made(self, key, make):
        v = self._lazy.get(key)
        if v is None:
            v = self._lazy[key] = make()
        return v

    # ---- attributes ----

    @property
    def points(self):
        """The points (n, d) float64, read-only (a copy of the input)."""
        return self._points

    @property
    def ndim(self):
        return self._tri.ndim

    @property
    def npoints(self):
        return self._tri.n

    @property
    def nsimplex(self):
        return self._tri.nsimplex

    @property
    def simplices(self):
        """(nsimplex, d + 1) int64: positively oriented, the smallest index first, rows in lexicographic order."""
        return self._simplices

    @property
    def neighbors(self):
        """(nsimplex, d + 1) int64: neighbors[i, k] shares the facet opposite vertex k of simplex i, -1 for none."""
        return self._neighbors

    @property
    def vertex_to_simplex(self):
        """(n,) int64: the smallest simplex holding each point (an equal point: its representative's)."""
        return self._v2s

    @property
    def convex_hull(self):
        """(facets, d) int64: the boundary facets of the triangulation, outward, rows in lexicographic order."""
        return self._made("convex_hull", lambda: _readonly(self._tri.convex_hull()))

    @property
    def vertex_neighbor_vertices(self):
        """(indptr, indices) int64: the neighbours of point i are indices[indptr[i]:indptr[i + 1]], increasing."""

        def make():
            indptr, indices = self._tri.vertex_neighbor_vertices()
            return (_readonly(indptr), _readonly(indices))

        return self._made("vertex_neighbor_vertices", make)

    @property
    def coplanar(self):
        """(k, 3) int64 rows (point, simplex, vertex) of the points left out: the equal points that are not their
        representative (the smallest index with the same coordinates)."""
        return self._made("coplanar", lambda: _readonly(self._tri.coplanar()))

    @property
    def transform(self):
        """(nsimplex, d + 1, d) float64: transform[i, :d] is the inverse of T (columns: vertex k minus vertex d)
        and transform[i, d] is vertex d, so the barycentric coordinates of x are T^-1 (x - r) and 1 - their sum."""
        return self._made("transform", lambda: _readonly(self._tri.transform()))

    @property
    def min_bound(self):
        """The smallest coordinate on each axis (d,), read-only."""
        return self._made("min_bound", lambda: _readonly(_sd.min(self._points, axis=0)))

    @property
    def max_bound(self):
        """The largest coordinate on each axis (d,), read-only."""
        return self._made("max_bound", lambda: _readonly(_sd.max(self._points, axis=0)))

    def __repr__(self):
        return f"Delaunay(n={self.npoints}, d={self.ndim}, nsimplex={self.nsimplex})"

    def __reduce__(self):  # rebuilt from the points: the same triangulation (SPT105)
        return (_rebuild, (memoryview(self._points).tobytes(), self.npoints, self.ndim))

    # ---- queries ----

    @method("find_simplex")
    def find_simplex(self, xi, /):
        """The smallest simplex whose closed simplex holds each point of xi (..., d): int64 of xi's batch shape
        (an int for one point), -1 outside the convex hull and for NaN or inf coordinates."""
        name = "find_simplex"
        sd = _sd
        X = C.data(xi, name, "xi")
        d = self.ndim
        if X.ndim == 0 or X.shape[-1] != d:
            got = X.shape[-1] if X.ndim else "no axis"
            raise ValueError(f"{name}(): 'xi' must have {d} entries on its last axis, not {got}")
        batch = tuple(X.shape[:-1])
        X2 = sd.reshape(X, (math.prod(batch), d))
        if not X2._flags()["c_contiguous"]:  # copy() keeps an F-ordered array F-ordered
            c_ordered = sd.empty(tuple(X2.shape), dtype=X2.dtype)
            c_ordered[...] = X2
            X2 = c_ordered
        out = self._tri.find_simplex(X2)
        if not batch:
            return int(out[0])
        return sd.reshape(out, batch)


def _limited(points, max_simplices):
    """A Delaunay whose 3-D build may hold at most max_simplices tetrahedra (ghosts included): the MemoryError
    path for tests, without exhausting memory."""
    D = Delaunay.__new__(Delaunay)
    D._build(points, max_simplices)
    return D


def _rebuild(raw, n, d):
    """A pickled Delaunay: its points (bytes of n x d float64)."""
    sd = _sd
    flat = sd.asarray(memoryview(raw).cast("d")) if n * d else sd.zeros(0)
    return Delaunay(sd.reshape(flat, (n, d)))
