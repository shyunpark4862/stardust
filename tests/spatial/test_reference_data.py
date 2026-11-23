"""The reference and comparison data of sd.spatial (docs/design/spatial.md §7): every file belongs to
the test points as they are (digests), the references pass checks of their own (exact rationals on
the sums, the nearest-double test of the square roots, the lattice tables against exact double
arithmetic), and the recorded scipy 1.18.1, scikit-learn 1.9.1 and R results agree with the
references and show the behaviours the design departs from. The recorded files are compared, not
the installed libraries, so the tests do not change with them."""

import math
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd

from . import points as P
from . import refdata

REFS = {f: refdata.ref(f) for f in refdata.FAMILIES}
SCIPY = refdata.data("scipy")
SKLEARN = refdata.data("sklearn")
R = refdata.data("r")


def fr(v) -> Fraction:
    return Fraction(float(v))


def dd_exact(hi, lo) -> Fraction:
    return fr(hi) + fr(lo)


# ---- the files belong to the points ---------------------------------------------------------


def test_every_family_is_there_with_its_provenance():
    assert refdata.names() == sorted(refdata.FAMILIES)
    for f, r in REFS.items():
        assert r.meta["arb"].startswith("python-flint 0.9.0"), f
        assert r.meta["tool"] == "tools/spatial_ref.py"
        assert r.meta["target_bits"] >= 106
        assert r.meta["n_requested"] == r.points, f  # no point was dropped
        for m in r.methods.values():
            assert m.dropped.size == 0


def _inputs(family, k):
    r = REFS[family]
    return {name: (v[k] if not isinstance(v, list) else v[k]) for name, v in r.inputs.items()}


def test_points_are_the_generators_points():
    """Every point's inputs are the ones points.py makes now (and its digest says so)."""
    edges = {eid: (u, v) for eid, u, v in P.edge_pairs()}
    havs = {hid: (u, v) for hid, u, v in P.haversine_pairs()}
    for family in ("distances", "counting", "haversine"):
        r = REFS[family]
        for k in range(r.points):
            sid, pair = str(r.inputs["set"][k]), int(r.inputs["pair"][k])
            u, v = r.inputs["u"][k], r.inputs["v"][k]
            if sid.startswith("edge/"):
                want = edges[sid[5:]]
            elif sid.startswith("haversine/"):
                want = havs[sid[10:]]
            else:
                s = P.get(sid)
                want = (s.x[pair], s.x[pair + 1])
            assert np.array_equal(u, want[0]) and np.array_equal(v, want[1]), (family, sid, pair)
            dtype = str(r.inputs["dtype"][k]) if family == "counting" else str(u.dtype)
            assert dtype == str(np.asarray(want[0]).dtype), (family, sid)
    for family in ("neighbors", "sphere"):
        r = REFS[family]
        for k in range(r.points):
            s = P.get(str(r.inputs["set"][k]))
            assert np.array_equal(r.inputs["query"][k], s.queries[int(r.inputs["pair"][k])])
            assert r.inputs["digest"][k] == P.digest(s.queries[int(r.inputs["pair"][k])])
            for m in r.methods.values():
                assert len(m.hi[k]) == s.m, (family, k)


def test_geometry_points_are_the_generators_points():
    """The geometry families (SPT187) belong to geompoints.py's sets and queries as they are now."""
    from . import geompoints as G

    for family in ("geom_measure", "voronoi_vertices"):
        r = REFS[family]
        for k in range(r.points):
            x = G.get(str(r.inputs["set"][k])).x
            assert np.array_equal(r.inputs["x"][k], x) and r.inputs["digest"][k] == P.digest(x), (family, k)
    r = REFS["barycentric"]
    for k in range(r.points):
        q = G.queries(str(r.inputs["set"][k]))[int(r.inputs["pair"][k])]
        assert np.array_equal(r.inputs["query"][k], q) and r.inputs["digest"][k] == P.digest(q), k
    assert {str(s) for s in REFS["geom_measure"].inputs["set"]} == {
        i for i in G.IDS if not {"invalid", "large"} & G.get(i).tags
    }


def test_geometry_volumes_are_exact_rationals():
    """volume is a rational: hi is it rounded once (lo the rest), for the sets whose volume is finite."""
    from . import exactgeom as X

    r = REFS["geom_measure"]
    m = r.methods["volume"]
    for j, k in enumerate(m.index):
        x = r.inputs["x"][k]
        if x.shape[0] > 200 or not math.isfinite(m.hi[j]):
            continue
        h = sd.spatial.ConvexHull(x)
        P_ = X.fpoints(x)
        v, f = np.asarray(h.vertices).tolist(), np.asarray(h.simplices).tolist()
        if x.shape[1] == 2:
            fan = (X.orient_value(P_[v[0]], P_[v[i]], P_[v[i + 1]]) for i in range(1, len(v) - 1))
            exact = sum(fan, Fraction(0)) / 2
        else:
            exact = sum((X.orient_value(P_[v[0]], *(P_[i] for i in t)) for t in f), Fraction(0)) / 6
        assert fr(m.hi[j]) == Fraction(float(exact)), k


def test_every_point_set_is_covered():
    sets = {str(s) for f in ("distances", "counting") for s in REFS[f].inputs["set"]}
    for sid in P.ids():
        if P.get(sid).m >= 2 and "sphere" not in P.get(sid).tags:
            assert sid in sets, sid
    assert set(P.ids("neighbors")) == {str(s) for s in REFS["neighbors"].inputs["set"]}
    assert set(P.ids("sphere")) == {str(s) for s in REFS["sphere"].inputs["set"]}


# ---- the references check themselves --------------------------------------------------------


def _method(family, name):
    r = REFS[family]
    m = r.methods[name]
    return r, m


def test_exact_sums_are_the_rational_sums():
    r = REFS["distances"]
    for name, f in (("sqeuclidean", lambda z: sum((x * x for x in z), Fraction(0))),
                    ("cityblock", lambda z: sum((abs(x) for x in z), Fraction(0))),
                    ("chebyshev", lambda z: max((abs(x) for x in z), default=Fraction(0)))):  # fmt: skip
        m = r.methods[name]
        for j, k in enumerate(m.index[::3]):
            u, v = r.inputs["u"][k], r.inputs["v"][k]
            exact = f([fr(a) - fr(b) for a, b in zip(u, v)])
            hi, lo = m.hi[3 * j], m.lo[3 * j]
            if math.isinf(hi):
                assert exact > Fraction(np.finfo(np.float64).max), (name, k)
                continue
            assert fr(hi) == Fraction(float(exact)), (name, k)  # hi is the sum rounded once
            assert abs(dd_exact(hi, lo) - exact) <= abs(exact) * Fraction(1, 2**104) + Fraction(1, 2**1074), (name, k)


def test_square_roots_are_nearest_doubles():
    """euclidean's hi is the double nearest sqrt(S), S the exact sum of squares: (hi -+ ulp/2)^2
    bracket S."""
    r = REFS["distances"]
    m = r.methods["euclidean"]
    for j, k in enumerate(m.index):
        u, v = r.inputs["u"][k], r.inputs["v"][k]
        s = sum(((fr(a) - fr(b)) ** 2 for a, b in zip(u, v)), Fraction(0))
        hi = m.hi[j]
        if s == 0:
            assert hi == 0
            continue
        half = Fraction(math.ulp(hi)) / 2
        assert (fr(hi) - half) ** 2 <= s <= (fr(hi) + half) ** 2, k


def test_lattice_tables_are_exact_double_arithmetic():
    """On the lattice sets every difference and sum is exact in doubles, so numpy's sqrt (correctly
    rounded) of the double sum is the reference's hi: the data of §4 neighbours 2."""
    r = REFS["neighbors"]
    for k in range(r.points):
        s = P.get(str(r.inputs["set"][k]))
        if "lattice" not in s.tags:
            continue
        z = s.x - r.inputs["query"][k]
        assert np.array_equal(np.sqrt((z * z).sum(axis=1)), r.methods["p2"].hi[k]), s.id
        assert np.array_equal(np.abs(z).sum(axis=1), r.methods["p1"].hi[k])
        assert np.array_equal(np.abs(z).max(axis=1), r.methods["pinf"].hi[k])
        assert not np.any(r.methods["p1"].lo[k]) and not np.any(r.methods["pinf"].lo[k])


def test_lattice_sets_have_many_exact_ties():
    r = REFS["neighbors"]
    ties = 0
    for k in range(r.points):
        if "lattice" in P.get(str(r.inputs["set"][k])).tags:
            d = r.methods["p2"].hi[k]
            ties += d.size - np.unique(d).size
    assert ties > 10000


def test_shell_distances_step_by_one_ulp():
    r = REFS["neighbors"]
    k = next(i for i in range(r.points) if r.inputs["set"][i] == "shell/3" and r.inputs["pair"][i] == 0)
    d = np.unique(r.methods["p2"].hi[k])
    assert d.size == 16 and np.all(np.diff(d) == 2.0**-52)


def test_haversine_bounds_and_antipodes():
    r, m = _method("haversine", "haversine")
    for j, k in enumerate(m.index):
        assert 0 <= m.hi[j] <= math.pi
        if str(r.inputs["set"][k]) == "haversine/antipode0":
            assert m.hi[j] > math.pi - 1e-6


def test_counting_is_exact():
    r = REFS["counting"]
    m = r.methods["hamming"]
    for j, k in enumerate(m.index):
        u, v = r.inputs["u"][k], r.inputs["v"][k]
        assert dd_exact(m.hi[j], m.lo[j]) == Fraction(int(np.sum(u != v)), u.size) or m.lo[j] != 0
        assert m.hi[j] == float(Fraction(int(np.sum(u != v)), u.size))


# ---- the other libraries' records -----------------------------------------------------------


def test_records_have_provenance():
    assert SCIPY["_provenance"]["scipy"] == "1.18.1"
    assert SKLEARN["_provenance"]["sklearn"] == "1.9.1"
    assert R["_provenance"]["R"].startswith("R version 4.6.2")


def test_scipy_edge_records_belong_to_the_edge_pairs():
    for eid, u, v in P.edge_pairs():
        assert SCIPY[f"distance/euclidean/{eid}"]["digest"] == P.digest(u, v)


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra", "braycurtis"])
def test_scipy_distances_agree_with_the_references_on_ordinary_pairs(metric):
    r = REFS["distances"]
    m = r.methods[metric]
    edges = {f"edge/{eid}" for eid, _, _ in P.edge_pairs()}
    for j, k in enumerate(m.index):
        sid = str(r.inputs["set"][k])
        if sid not in edges or sid in ("edge/huge", "edge/tiny", "edge/subnormal"):
            continue
        rec = SCIPY[f"distance/{metric}/{sid[5:]}"]
        if sid == "edge/empty" and metric == "chebyshev":  # scipy's function: max() of nothing
            assert rec["error"] == "ValueError" and SCIPY[f"pdist/{metric}/empty"]["value"] == 0.0
            continue
        assert "value" in rec, (metric, sid)
        assert rec["value"] == pytest.approx(m.hi[j], rel=1e-15, abs=1e-300), (metric, sid)


def test_scipy_behaviours_the_design_departs_from():
    """docs/design/spatial.md §2.6 and docs/api/spatial.md §1.5: what scipy does there."""
    assert SCIPY["cdist/chebyshev_nan"]["value"] == 5.0  # skips the NaN
    assert SCIPY["canberra/nan_function"]["value"] == pytest.approx(1 / 3)  # nansum
    assert math.isnan(SCIPY["canberra/nan_pdist"]["value"])  # propagates
    assert SCIPY["pdist/minkowski_p-1"]["value"] > 0 and math.isinf(SCIPY["pdist/minkowski_p0"]["value"])
    assert SCIPY["minkowski/p-1"]["error"] == "ValueError"
    assert math.isinf(SCIPY["minkowski/p2000"]["value"]) and math.isinf(SCIPY["euclidean/1e200"]["value"])
    assert SCIPY["pdist/complex"] == {"error": "ValueError", "msg": "Unsupported dtype complex128"}
    unused = SCIPY["pdist/unused_p"]
    assert unused["error"] == "TypeError" and "incompatible function arguments" in unused["msg"]
    assert SCIPY["pdist/negative_V"]["value"] == 0.0 and math.isinf(SCIPY["pdist/zero_V"]["value"])
    assert math.isnan(SCIPY["pdist/nan_w"]["value"]) and math.isinf(SCIPY["pdist/inf_w"]["value"])
    assert SCIPY["hamming/int_2^60"]["value"] == 0.0 or SCIPY["hamming/int_2^60"]["value"] == 1.0
    assert SCIPY["cosine/batch"]["error"] == "ValueError"
    assert SCIPY["squareform/nan"]["msg"] == "Distance matrix 'X' must be symmetric."
    assert SCIPY["squareform/empty"]["value"]["shape"] == [1, 1]
    assert SCIPY["kdtree/k_true"]["value"] is not None and "value" in SCIPY["kdtree/k_float"]
    assert SCIPY["kdtree/k_list_zero"] == {"crash": -11}
    assert SCIPY["kdtree/r_negative_same_as_1"]["value"] is True and SCIPY["kdtree/r_nan"]["value"] == []
    assert SCIPY["kdtree/dub_negative_same_as_1"]["value"] is True
    assert SCIPY["kdtree/leafsize_float"]["value"] == 2
    assert SCIPY["kdtree/m0_query"]["error"] == "IndexError"
    assert SCIPY["kdtree/ball_point_one"]["value"] == "list" and SCIPY["kdtree/query_pairs_type"]["value"] == "set"
    assert SCIPY["kdtree/sdm_zero_dok"]["value"] == [4, 0] and SCIPY["kdtree/sdm_zero_coo"]["value"] == 4
    assert SCIPY["kdtree/tie_order"]["value"]["array"] == [1, 2, 3, 4, 5, 0]
    assert SCIPY["order/cdist_sequential"]["same"] == SCIPY["order/cdist_sequential"]["pairs"]
    assert SCIPY["order/kdtree_vs_cdist"]["differ"]["3000"] > 100
    assert 300 < SCIPY["kdtree/boundary_square_vs_root"]["differ"] < 600


def test_scipy_neighbours_agree_with_the_references_where_there_are_no_ties():
    r = REFS["neighbors"]
    for sid in P.ids("neighbors"):
        rec = SCIPY[f"knn/{sid}"]
        s = P.get(sid)
        assert rec["digest"] == P.set_digest(s)
        rows = [k for k in range(r.points) if r.inputs["set"][k] == sid]
        for q, k in enumerate(rows):
            d = r.methods["p2"].hi[k] + r.methods["p2"].lo[k]
            order = np.lexsort((np.arange(d.size), d))[: rec["k"]]
            if np.unique(np.round(d[order], 12)).size == rec["k"] and d[order[-1]] < np.sort(d)[rec["k"]] - 1e-9:
                assert sorted(rec["indices"][q]) == sorted(order.tolist()), (sid, q)


def test_sklearn_expanded_form_loses_the_far_cluster():
    """§3.2: on far/1e8 the expanded form gives distance 0 for every query and no right neighbour."""
    rec = SKLEARN["argmin_min/far/1e8"]
    r = REFS["neighbors"]
    rows = [k for k in range(r.points) if r.inputs["set"][k] == "far/1e8"]
    true = [int(np.argmin(r.methods["p2"].hi[k])) for k in rows]
    assert rec["distances"] == [0.0] * len(rows)
    assert sum(a == b for a, b in zip(rec["indices"], true)) == 0
    assert SKLEARN["euclidean_distances/1e8"]["value"] == 0.0
    assert SKLEARN["nn/tie_order"]["value"]["array"] == [3, 4, 1, 5, 2, 0]
    assert SKLEARN["nn/k_gt_n"]["error"] == "ValueError"


def test_sklearn_haversine_agrees_with_the_references_away_from_the_antipode():
    r, m = _method("haversine", "haversine")
    for j, k in enumerate(m.index):
        sid = str(r.inputs["set"][k])
        rec = SKLEARN[f"haversine/{sid[10:]}"]
        if "antipode" in sid or sid.endswith("poles"):
            continue
        assert rec["value"] == pytest.approx(m.hi[j], rel=1e-12, abs=1e-15), sid


def test_r_dist_agrees_on_the_lattice_and_rescales_canberra():
    s = P.get("lattice/int3d")
    rec = R["dist/lattice/int3d/euclidean"]
    assert rec["digest"] == P.set_digest(s)
    x = s.x[:20]
    i, j = np.triu_indices(20, 1)
    # R's dist is column-major lower triangle: pairs (i, j) with i > j, by column j
    want = [math.sqrt(float(((x[a] - x[b]) ** 2).sum())) for b in range(20) for a in range(b + 1, 20)]
    assert rec["values"] == want
    assert R["dist/edge/canberra_0/0"]["values"] == [3.0]  # (0.5 + 1) * 4 / 2: the 0/0 terms dropped and rescaled
    assert math.isinf(R["dist/edge/euclidean_1e200"]["values"][0])
