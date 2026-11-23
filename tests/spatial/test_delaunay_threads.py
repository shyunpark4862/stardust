"""Delaunay's arrays do not depend on the thread count (docs/design/spatial_geometry.md §3.10; SPT9, SPT10): at
1, 3 and 10 threads every array has the same bytes, on inputs large enough for the top of the recursion (more
than 2^14 points: partitions and merges side by side), the sort's merge rounds, the canonical form's passes
and find_simplex's blocks; for d = 3 the BRIO keys and sort too (the insertion itself is one thread)."""

import numpy as np
import pytest

import stardust as sd

from . import geompoints as G

pytestmark = pytest.mark.usefixtures("restore_threads")


def _inputs():
    g = np.random.default_rng(29)
    grid = np.array([[i, j] for i in range(200) for j in range(200)], dtype=np.float64)
    return [
        g.random((70000, 2)),
        g.standard_normal((40000, 2)),
        grid[g.permutation(len(grid))],  # every cell cocircular: the tie rule everywhere
        np.concatenate([g.random((30000, 2)), g.random((5000, 2))[:, :1].repeat(2, axis=1)]),
        G.get("dup2").x,
    ]


def _arrays(x):
    D = sd.spatial.Delaunay(x)
    q = np.random.default_rng(31).random((20000, 2)) * (x.max(0) - x.min(0)) + x.min(0)
    q = np.concatenate([q, x[:3000]])
    out = [D.simplices, D.neighbors, D.vertex_to_simplex, D.convex_hull, D.coplanar, D.transform,
           *D.vertex_neighbor_vertices, D.find_simplex(q)]  # fmt: skip
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


def _inputs3():
    """d = 3 (SPT10): more than 2^14 points for the parallel sort, BRIO keys, canonical form and queries; a
    shuffled grid (every cell cospherical) and equal points."""
    g = np.random.default_rng(37)
    grid = np.array([[i, j, k] for i in range(28) for j in range(28) for k in range(28)], dtype=np.float64)
    return [
        g.random((40000, 3)),
        g.standard_normal((20000, 3)),
        grid[g.permutation(len(grid))],
        np.concatenate([g.random((18000, 3)), g.random((2000, 3))[:, :1].repeat(3, axis=1)]),
        G.get("dup3").x,
    ]


def _arrays3(x):
    D = sd.spatial.Delaunay(x)
    q = np.random.default_rng(41).random((20000, 3)) * (x.max(0) - x.min(0)) + x.min(0)
    q = np.concatenate([q, x[:3000]])
    out = [D.simplices, D.neighbors, D.vertex_to_simplex, D.convex_hull, D.coplanar, D.transform,
           *D.vertex_neighbor_vertices, D.find_simplex(q)]  # fmt: skip
    return [np.asarray(a).tobytes() for a in out]


def test_one_three_ten_threads_3d():
    xs = _inputs3()
    got = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got[t] = [_arrays3(x) for x in xs]
    for t in (3, 10):
        for i, (a, b) in enumerate(zip(got[1], got[t])):
            for k, (u, v) in enumerate(zip(a, b)):
                assert u == v, (t, i, k)
