"""Voronoi of sd.spatial (docs/api/spatial_geometry.md §5, docs/design/spatial_geometry.md §2.5, §3.7; milestone
SPT12, decisions SPT110, SPT111, SPT172-SPT189).

A Python class holding the arrays of the native diagram (stardust._core.spatial._voronoi,
csrc/spatial/voronoi.cpp), computed in the constructor from the exact Delaunay triangulation, which is not kept
(SPT172; like ConvexHull, SPT140). The diagram is the true one of the point set: cospherical simplices share one
vertex and ridges without size are left out (SPT111), so the object is a function of the point set: the same
bytes whatever the thread count, and a pickled diagram is rebuilt from its points.
"""

from .. import _core as _sd
from .._calls import method
from . import _common as C

_core = _sd.spatial

# Bytes per input point the build holds at its peak before it can count (docs/design/spatial_geometry.md §3.11,
# SPT181): d = 2 Delaunay's 360 and the diagram's arrays with their chunk copies (10^6 points: 603 MB); d = 3
# the input's 200 and one tetrahedron's 352 after the build (csrc/spatial/voronoi.cpp), at least one tetrahedron
# per point. The 3-D build also counts its tetrahedra (MemoryError past physical memory's share).
_BUILD_BYTES = {2: 640, 3: 552}


def _readonly(a):
    a._set_readonly()
    return a


class Voronoi:
    """The Voronoi diagram of `points` (n, d), d = 2 or 3, exact in its structure for every finite float64 input."""

    __slots__ = ("_points", "_vertices", "_ridge_points", "_ridge_vertices", "_regions", "_point_region", "_lazy")

    @method("Voronoi")
    def __init__(self, points, /):
        self._build(points, -1)

    def _build(self, points, max_simplices):
        """max_simplices: the tetrahedra (ghosts included) a 3-D build may hold, -1 for physical memory's share
        (tests pass less: _limited)."""
        name = "Voronoi"
        a = C.geometry_points(points, name)
        C.check_alloc(_BUILD_BYTES[a.shape[1]] * a.shape[0], name)
        p, v, rp, rptr, ridx, gptr, gidx, pr = _core._voronoi(a, max_simplices)
        self._points = _readonly(p)
        self._vertices = _readonly(v)
        self._ridge_points = _readonly(rp)
        self._ridge_vertices = (_readonly(rptr), _readonly(ridx))
        self._regions = (_readonly(gptr), _readonly(gidx))
        self._point_region = _readonly(pr)
        self._lazy = {}

    @property
    def points(self):
        """The points (n, d) float64, read-only (a copy of the input)."""
        return self._points

    @property
    def vertices(self):
        """The Voronoi vertices (nvertices, d) float64: the circumcentres of the Delaunay cells, numbered in the
        order of each cell's smallest simplex."""
        return self._vertices

    @property
    def ridge_points(self):
        """(nridges, 2) int64: the pairs (i, j), i < j, whose regions share a ridge, in lexicographic order."""
        return self._ridge_points

    @property
    def ridge_vertices(self):
        """(indptr, indices) int64: ridge k's vertices are indices[indptr[k]:indptr[k + 1]], -1 first for an
        unbounded ridge; d = 3 counterclockwise seen from ridge_points[k, 0] towards ridge_points[k, 1]."""
        return self._ridge_vertices

    @property
    def regions(self):
        """(indptr, indices) int64: region i (point i's) is indices[indptr[i]:indptr[i + 1]], -1 first when it
        is unbounded; d = 2 counterclockwise, d = 3 increasing. An equal point that is not its representative
        has an empty row."""
        return self._regions

    @property
    def point_region(self):
        """(n,) int64: the region of each point (its own index; an equal point's representative's)."""
        return self._point_region

    @property
    def ndim(self):
        return self._points.shape[1]

    @property
    def npoints(self):
        return self._points.shape[0]

    @property
    def min_bound(self):
        """The smallest coordinate on each axis (d,), read-only."""
        v = self._lazy.get("min_bound")
        if v is None:
            v = self._lazy["min_bound"] = _readonly(_sd.min(self._points, axis=0))
        return v

    @property
    def max_bound(self):
        """The largest coordinate on each axis (d,), read-only."""
        v = self._lazy.get("max_bound")
        if v is None:
            v = self._lazy["max_bound"] = _readonly(_sd.max(self._points, axis=0))
        return v

    def __repr__(self):
        return f"Voronoi(n={self.npoints}, d={self.ndim}, nvertices={self._vertices.shape[0]})"

    def __reduce__(self):  # rebuilt from the points: the diagram is a function of them (SPT105)
        return (_rebuild, (memoryview(self._points).tobytes(), self.npoints, self.ndim))


def _limited(points, max_simplices):
    """A Voronoi whose 3-D build may hold at most max_simplices tetrahedra (ghosts included): the MemoryError
    path for tests, without exhausting memory."""
    V = Voronoi.__new__(Voronoi)
    V._build(points, max_simplices)
    return V


def _rebuild(raw, n, d):
    """A pickled Voronoi: its points (bytes of n x d float64)."""
    sd = _sd
    flat = sd.asarray(memoryview(raw).cast("d")) if n * d else sd.zeros(0)
    return Voronoi(sd.reshape(flat, (n, d)))
