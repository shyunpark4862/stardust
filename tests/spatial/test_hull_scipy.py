"""ConvexHull against scipy 1.18.1 (Qhull) on points in general position (docs/design/spatial_geometry.md §7.3;
milestone SPT11): the same extreme points and facets, and area, volume and equations within the §4 bounds plus
scipy's own error. Degenerate data are compared through scipy's records only (test_geometry_records.py)."""

import numpy as np
import pytest
import scipy.spatial as S

import stardust as sd

from . import geompoints as G
from .conftest import LIVE

pytestmark = LIVE
IDS = [i for i in G.ids("random")] + ["ulp2/200", "ulp3/200", "polygon2/200"]


@pytest.mark.parametrize("id_", IDS)
def test_matches_scipy(id_):
    x = G.get(id_).x
    h, q = sd.spatial.ConvexHull(sd.asarray(x)), S.ConvexHull(x)
    v = np.asarray(h.vertices)
    assert sorted(v) == sorted(q.vertices)
    if x.shape[1] == 2:  # both counterclockwise: the same cycle
        k = list(q.vertices).index(v[0])
        assert list(v) == list(np.roll(q.vertices, -k))
    s, eq = np.asarray(h.simplices), np.asarray(h.equations)
    mine = {frozenset(map(int, f)): e for f, e in zip(s, eq)}
    theirs = {frozenset(map(int, f)): e for f, e in zip(q.simplices, q.equations)}
    assert mine.keys() == theirs.keys()
    for f, e in mine.items():
        np.testing.assert_allclose(e, theirs[f], rtol=0, atol=1e-13 * max(1.0, np.abs(x).max()))
    np.testing.assert_allclose([h.area, h.volume], [q.area, q.volume], rtol=1e-13)
