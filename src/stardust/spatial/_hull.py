"""ConvexHull of sd.spatial (docs/api/spatial_geometry.md §4, docs/design/spatial_geometry.md §2.4, §3.6;
milestone SPT11, decisions SPT108, SPT109, SPT140-SPT153).

A Python class holding the arrays of the native hull (stardust._core.spatial._convex_hull,
csrc/spatial/hull.cpp), computed in the constructor: the hull has no queries, so nothing native is kept
(SPT140). The combinatorial answers (which points are extreme, which facets exist) are exact for the
inputs read as rationals, and the arrays are in the canonical form of §2.2, so the object is a function of
the point set: the same bytes whatever the thread count, and a pickled hull is rebuilt from its points.
"""

from .. import _core as _sd
from .._calls import method
from . import _common as C

_core = _sd.spatial


def _readonly(a):
    a._set_readonly()
    return a


class ConvexHull:
    """The convex hull of the rows of `points` (n, d), d = 2 or 3."""

    __slots__ = ("_points", "_vertices", "_simplices", "_neighbors", "_equations", "_area", "_volume", "_min", "_max")

    @method("ConvexHull")
    def __init__(self, points, /):
        a = C.geometry_points(points, "ConvexHull", check_finite=False)  # finiteness: the native scan
        p, v, s, nb, eq, area, volume, lo, hi = _core._convex_hull(a)
        self._points = _readonly(p)
        self._vertices = _readonly(v)
        self._simplices = _readonly(s)
        self._neighbors = _readonly(nb)
        self._equations = _readonly(eq)
        self._area = area
        self._volume = volume
        self._min = _readonly(lo)
        self._max = _readonly(hi)

    @property
    def points(self):
        """The points (n, d) float64, read-only (a copy of the input)."""
        return self._points

    @property
    def vertices(self):
        """The extreme points (h,) int64: d = 2 counterclockwise from the smallest index, d = 3 ascending."""
        return self._vertices

    @property
    def simplices(self):
        """The facets (nfacets, d) int64, outward, smallest index first, rows in lexicographic order."""
        return self._simplices

    @property
    def neighbors(self):
        """neighbors[i, k]: the facet across from vertex k of facet i (nfacets, d) int64."""
        return self._neighbors

    @property
    def equations(self):
        """The facets' planes (nfacets, d + 1) float64: outward unit normal and offset (normal . x + offset <= 0
        inside); the triangles of one flat facet share a row."""
        return self._equations

    @property
    def area(self):
        """d = 3: the surface area; d = 2: the perimeter (scipy's meaning)."""
        return self._area

    @property
    def volume(self):
        """d = 3: the volume; d = 2: the enclosed area (scipy's meaning)."""
        return self._volume

    @property
    def ndim(self):
        return self._points.shape[1]

    @property
    def npoints(self):
        return self._points.shape[0]

    @property
    def nsimplex(self):
        return self._simplices.shape[0]

    @property
    def min_bound(self):
        """The smallest coordinate on each axis (d,), read-only."""
        return self._min

    @property
    def max_bound(self):
        """The largest coordinate on each axis (d,), read-only."""
        return self._max

    def __repr__(self):
        return f"ConvexHull(n={self.npoints}, d={self.ndim}, nvertices={self._vertices.shape[0]})"

    def __reduce__(self):  # rebuilt from the points: the hull is a function of them (SPT105)
        return (_rebuild, (memoryview(self._points).tobytes(), self.npoints, self.ndim))


def _rebuild(raw, n, d):
    """A pickled ConvexHull: its points (bytes of n x d float64)."""
    sd = _sd
    return ConvexHull(sd.reshape(sd.asarray(memoryview(raw).cast("d")), (n, d)))
