"""Voronoi's arrays do not depend on the thread count (docs/design/spatial_geometry.md §3.10; SPT12): at 1, 3
and 10 threads every array has the same bytes, on inputs past 2^14 points (the triangulation's parallel paths,
the cluster tests and centres by simplex chunks, the regions and ridges by point chunks), cocircular and
cospherical grids (the union-find everywhere), and equal points."""

import numpy as np
import pytest

import stardust as sd

from . import geompoints as G

pytestmark = pytest.mark.usefixtures("restore_threads")


def _inputs():
    g = np.random.default_rng(43)
    grid2 = np.array([[i, j] for i in range(160) for j in range(160)], dtype=np.float64)
    grid3 = np.array([[i, j, k] for i in range(26) for j in range(26) for k in range(26)], dtype=np.float64)
    return [
        g.random((60000, 2)),
        grid2[g.permutation(len(grid2))],
        np.concatenate([g.random((20000, 2)), g.random((3000, 2))[:, :1].repeat(2, axis=1)]),
        G.get("dup2").x,
        g.random((30000, 3)),
        grid3[g.permutation(len(grid3))],
        G.get("dup3").x,
    ]


def _arrays(x):
    V = sd.spatial.Voronoi(x)
    out = [V.vertices, V.ridge_points, *V.ridge_vertices, *V.regions, V.point_region]
    return [np.asarray(a).tobytes() for a in out]


def test_one_three_ten_threads():
    xs = _inputs()
    got = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got[t] = [_arrays(x) for x in xs]
    for t in (3, 10):
        for i, (a, b) in enumerate(zip(got[1], got[t])):
            for k, (u, v) in enumerate(zip(a, b)):
                assert u == v, (t, i, k)
