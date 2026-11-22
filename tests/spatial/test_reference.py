"""sd.spatial's distances against the Arb references of tests/spatial/ref (docs/design/spatial.md §4):
every point of every method within the contract (tests/spatial/refdata.py), chebyshev and the
unweighted counts correctly rounded."""

import math

import numpy as np
import pytest

import stardust as sd

from . import refdata

S = sd.spatial
DIST = refdata.ref("distances")
COUNT = refdata.ref("counting")
HAV = refdata.ref("haversine")


def _worst(family, r, method):
    m = r.methods[method]
    worst, where = 0.0, None
    for j, k in enumerate(m.index):
        x = refdata.inputs(family, int(k))
        got = refdata.call(S, method, x["u"], x["v"], x.get("w"), x.get("V"), x.get("VI"))
        assert isinstance(got, float)
        u = refdata.point_units(family, method, int(k), got, j)
        if u > worst:
            worst, where = u, (str(r.inputs["set"][k]), int(r.inputs["pair"][k]), got, float(m.hi[j]))
    return worst, where


@pytest.mark.parametrize("method", sorted(DIST.methods))
def test_distances_within_the_contract(method):
    worst, where = _worst("distances", DIST, method)
    assert worst <= 1.0, (method, worst, where)


@pytest.mark.parametrize("method", ["chebyshev", "w_chebyshev"])
def test_chebyshev_is_correctly_rounded(method):
    m = DIST.methods[method]
    for j, k in enumerate(m.index):
        x = refdata.inputs("distances", int(k))
        got = refdata.call(S, method, x["u"], x["v"], x.get("w"))
        assert got == m.hi[j], (str(DIST.inputs["set"][k]), got, m.hi[j])


@pytest.mark.parametrize("method", sorted(COUNT.methods))
def test_counting_within_the_contract(method):
    worst, where = _worst("counting", COUNT, method)
    assert worst <= 1.0, (method, worst, where)
    if not method.startswith("w_"):  # exact counts, one division: correctly rounded
        m = COUNT.methods[method]
        for j, k in enumerate(m.index):
            x = refdata.inputs("counting", int(k))
            assert getattr(S, method)(x["u"], x["v"]) == m.hi[j]


def test_haversine_within_the_contract_up_to_the_antipode():
    worst, where = _worst("haversine", HAV, "haversine")
    assert worst <= 1.0, (worst, where)
    m = HAV.methods["haversine"]
    for j, k in enumerate(m.index):
        if "antipode" in str(HAV.inputs["set"][k]):
            x = refdata.inputs("haversine", int(k))
            got = S.haversine(x["u"], x["v"])
            assert abs(got - m.hi[j]) <= 4 * math.ulp(m.hi[j])


def test_cdist_gives_the_reference_pairs_the_same_bits():
    """The pairs of the references through cdist (the tile kernel) are the distance functions' values."""
    for method in ("euclidean", "sqeuclidean", "cityblock", "canberra", "braycurtis", "cosine", "w_euclidean"):
        m = DIST.methods[method]
        ks = [int(k) for k in m.index if len(DIST.inputs["u"][k]) == 33][:8]
        for k in ks:
            x = refdata.inputs("distances", k)
            kw = {"w": x["w"]} if method.startswith("w_") else {}
            name = method[2:] if method.startswith("w_") else method
            c = S.cdist(np.vstack([x["u"]] * 5), np.vstack([x["v"]] * 9), metric=name, **kw)
            one = refdata.call(S, method, x["u"], x["v"], x.get("w"))
            assert all(c[i, j] == one or (math.isnan(one) and math.isnan(c[i, j])) for i in range(5) for j in range(9))
