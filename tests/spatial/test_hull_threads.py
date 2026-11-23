"""ConvexHull does not depend on the thread count (docs/design/spatial_geometry.md §3.10; milestone SPT11): at 1,
3 and 10 threads every array, area and volume have the same bytes, on point sets large enough to split the
scans, the pre-filter, the coplanar tests and the sums into many tasks."""

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import geompoints as G

pytestmark = pytest.mark.usefixtures("restore_threads")


def _sets():
    g = np.random.default_rng(20261123)
    u = np.linspace(0.0, 2 * np.pi, 60_000, endpoint=False)
    z = g.uniform(-1.0, 1.0, 40_000)
    t = g.uniform(0.0, 2 * np.pi, 40_000)
    r = np.sqrt(1.0 - z * z)
    grid = np.array(np.meshgrid(*[np.arange(40.0)] * 3)).reshape(3, -1).T  # 64,000 points, flat facets
    return {
        "uniform2": g.random((200_000, 2)),
        "circle2": np.c_[np.cos(u), np.sin(u)],
        "uniform3": g.random((200_000, 3)),
        "sphere3": np.c_[r * np.cos(t), r * np.sin(t), z],
        "grid3": grid[g.permutation(len(grid))],
        "far3": 1e8 + g.random((100_000, 3)),
        "cube": G.get("cube/5").x,
    }


def _result(x):
    h = sd.spatial.ConvexHull(sd.asarray(x))
    return (h.vertices, h.simplices, h.neighbors, h.equations, h.area, h.volume, h.min_bound, h.max_bound)


@pytest.mark.parametrize("name", list(_sets()))
def test_same_bytes_at_1_3_and_10_threads(name):
    x = _sets()[name]
    out = []
    for n in (1, 3, 10):
        sd.set_num_threads(n)
        out.append(_result(x))
    assert bits_equal(out[0], out[1]) and bits_equal(out[0], out[2])
