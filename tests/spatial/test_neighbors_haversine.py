"""kneighbors and radius_neighbors with metric='haversine' through the KD tree on the unit sphere
(docs/design/spatial.md §3.4, decisions SPT83-SPT88): the tree selects candidates by the chord between the
points on the sphere and the haversine pair kernel decides, so the answer is the definition (cdist's row
sorted by (d, j), the bits of cdist) on the sets where rounding could break the monotone relation between
the chord and the haversine: near-identical points, antipodes, the poles, the dateline, exactly equal
distances, shells of distances 1 ULP apart, duplicates, longitudes in [0, 2 pi) and [-8, 8]; on hypothesis'
clusters anywhere on the sphere; outside the domain of the margin (the brute force answers); at 1, 3 and
10 threads. Against the Arb references (tests/spatial/ref/sphere.npz, docs/design/spatial.md §4 neighbours
3): the k nearest and the points within r are the true ones within the haversine contract."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .._bits import bits_equal
from . import points as P
from . import refdata
from .test_neighbors import check_knn, check_radius

S = sd.spatial
N = S._neighbors
CORE = sd._core.spatial
INF = math.inf
LAT = 1.5707963267948966  # fl(pi / 2), the largest latitude of the domain
PATHS = (0, 1, 3)


def _sphere(g, n, lon=(-math.pi, math.pi)):
    return np.column_stack([np.arcsin(g.uniform(-1.0, 1.0, n)), g.uniform(*lon, n)])


def _check(A, B, ks=(1, 5, 40), radii=None, ub=None):
    """Every k and radius against the definition on the brute force, the tree (forced) and the automatic
    path; the native search must answer (the points are in its domain)."""
    assert CORE._sphere_kneighbors(sd.asarray(A), sd.asarray(B), 1, INF) is not None
    for k in ks:
        check_knn(A, B, k, "haversine", paths=PATHS)
    D = np.asarray(S.cdist(A, B, metric="haversine"))
    finite = np.sort(D.ravel())
    if radii is None:
        radii = [0.0, float(finite[len(finite) // 50]), float(finite[len(finite) // 3])]
    for r in radii:
        check_radius(A, B, r, "haversine", paths=PATHS)
    for u in ub if ub is not None else [float(finite[len(finite) // 10])]:
        check_knn(A, B, 7, "haversine", ub=u, paths=PATHS)


def test_uniform_and_the_longitude_ranges():
    g = np.random.default_rng(1)
    _check(_sphere(g, 60), _sphere(g, 900))
    _check(_sphere(g, 60, (0.0, 2.0 * math.pi)), _sphere(g, 900, (0.0, 2.0 * math.pi)))
    _check(_sphere(g, 60, (-8.0, 8.0)), _sphere(g, 900, (-8.0, 8.0)))  # the domain's whole longitude range
    # the same places written with longitudes 2 pi apart
    B = _sphere(g, 700)
    B2 = B.copy()
    B2[::2, 1] += np.where(B2[::2, 1] < 0.0, 2.0 * math.pi, -2.0 * math.pi)
    _check(B[:50], B2)


def test_near_identical_points_and_duplicates():
    g = np.random.default_rng(2)
    c = _sphere(g, 8)
    for spread in (1e-15, 1e-12, 1e-9, 1e-6):
        B = np.repeat(c, 120, axis=0) + g.normal(0.0, spread, (960, 2))
        B[840:960:2] = B[0:120:2]  # duplicates
        A = np.repeat(c, 6, axis=0) + g.normal(0.0, spread, (48, 2))
        _check(A, B, ks=(1, 6, 50))
    # points that differ in the last bit of a coordinate, and exact copies of the query
    base = _sphere(g, 5)
    B = np.vstack([np.column_stack([np.nextafter(base[:, 0], s * 2.0), base[:, 1]]) for s in (-1, 1)]
                  + [np.column_stack([base[:, 0], np.nextafter(base[:, 1], s * 9.0)]) for s in (-1, 1)]
                  + [base, base, _sphere(g, 300)])  # fmt: skip
    _check(base, B, ks=(1, 3, 6, 7))


def test_antipodes():
    g = np.random.default_rng(3)
    A = _sphere(g, 30)
    anti = np.column_stack([-A[:, 0], np.where(A[:, 1] > 0, A[:, 1] - math.pi, A[:, 1] + math.pi)])
    B = np.vstack([anti + g.normal(0.0, s, anti.shape) for s in (0.0, 1e-14, 1e-10, 1e-6, 1e-3)])
    B[:, 0] = np.clip(B[:, 0], -LAT, LAT)
    B = np.vstack([B, _sphere(g, 200)])
    # the farthest points: k up to every point, radii just below pi
    _check(A, B, ks=(1, 10, len(B)), radii=[3.0, math.pi - 1e-9, math.nextafter(math.pi, 0.0), math.pi])


def test_poles_and_the_dateline():
    g = np.random.default_rng(4)
    pole = np.column_stack([np.repeat([LAT, -LAT, LAT * (1 - 1e-16), -LAT * (1 - 1e-12)], 40), g.uniform(-8, 8, 160)])
    B = np.vstack([pole, _sphere(g, 300)])
    A = np.vstack([pole[::9], [[LAT, 0.0], [-LAT, 3.0], [LAT - 1e-8, -2.0]]])
    _check(A, B, ks=(1, 20, 45))
    side = g.uniform(0.0, 1e-7, 200)
    lat = g.uniform(-1.4, 1.4, 200)
    B = np.vstack([np.column_stack([lat, math.pi - side]), np.column_stack([lat, -math.pi + side[::-1]]),
                   np.column_stack([lat, math.pi + side])])  # fmt: skip
    A = np.column_stack([lat[::10], np.full(20, math.pi)])
    _check(A, B, ks=(1, 4, 30))


def test_exactly_equal_distances():
    # mirror images about the query's meridian and parallel at the equator: h is the same expression, so
    # the distances are the same double; the order among them is the index
    g = np.random.default_rng(5)
    q = np.array([[0.0, 0.0], [0.0, 1.0]])
    pts = []
    for a, b in g.uniform(0.0, 0.3, (40, 2)):
        pts += [[0.0, b], [0.0, -b], [a, 0.0], [-a, 0.0]]
    B = np.array(pts)
    B = np.vstack([B, B[::-1], B + [0.0, 1.0]])
    D = np.asarray(S.cdist(q, B, metric="haversine"))
    assert len(np.unique(D[0])) < D.shape[1] // 2  # many exact ties
    _check(q, B, ks=(1, 2, 3, 4, 17, 64), radii=[float(D[0, 5]), float(D[0, 100])], ub=[float(D[0, 7])])


def test_one_ulp_shells():
    # points whose distances to the query are a few ULPs apart; k and r cut through the shell
    g = np.random.default_rng(6)
    q = np.array([[0.4, 1.1]])
    rows = []
    for d0 in (1e-7, 0.01, 1.0, 3.0):
        brg = g.uniform(0.0, 2.0 * math.pi, 1)[0]
        f = math.asin(math.sin(0.4) * math.cos(d0) + math.cos(0.4) * math.sin(d0) * math.cos(brg))
        lam = 1.1 + math.atan2(math.sin(brg) * math.sin(d0) * math.cos(0.4), math.cos(d0) - math.sin(0.4) * math.sin(f))
        for i in range(-6, 7):
            for j in range(-6, 7):
                rows.append([f + i * math.ulp(f), lam + j * math.ulp(lam)])
    B = np.array(rows)
    D = np.asarray(S.cdist(q, B, metric="haversine"))[0]
    order = np.lexsort((np.arange(len(D)), D))
    radii = []
    for c in (80, 250, 420, 600):
        r = float(D[order[c]])
        radii += [r, math.nextafter(r, 0.0), math.nextafter(r, INF)]
    _check(q, B, ks=(1, 84, 169, 300, 500), radii=radii, ub=[float(D[order[c]]) for c in (84, 400)])


def test_shapes_and_fill():
    g = np.random.default_rng(7)
    A, B = _sphere(g, 20), _sphere(g, 150)
    check_knn(A, B, 400, "haversine", paths=PATHS)  # k > mB: (inf, mB)
    check_knn(A, B[:0], 3, "haversine", paths=PATHS)
    check_knn(A[:0], B, 3, "haversine", paths=PATHS)
    check_radius(A[:0], B, 0.1, "haversine", paths=PATHS)
    check_radius(A, B, np.linspace(0.0, 2.0, 20), "haversine", paths=PATHS)  # a radius per row
    check_radius(A, B, INF, "haversine", paths=PATHS)
    check_knn(A, B, 5, "haversine", ub=0.0, paths=PATHS)


def test_outside_the_domain_takes_the_brute_force():
    g = np.random.default_rng(8)
    A, B = _sphere(g, 30), _sphere(g, 400)
    for bad in ([1.6, 0.0], [0.0, 8.5], [0.0, -1e6], [-2.0, 0.0]):
        B2 = np.vstack([B, [bad]])
        assert CORE._sphere_kneighbors(sd.asarray(A), sd.asarray(B2), 3, INF) is None
        assert CORE._sphere_radius(sd.asarray(B2), sd.asarray(B), sd.full(401, 0.1), False) is None
        check_knn(A, B2, 5, "haversine", paths=PATHS)
        check_radius(A, B2, 0.2, "haversine", paths=PATHS)
    degrees = np.degrees(B)  # a common mistake: still the definition
    check_knn(np.degrees(A), degrees, 4, "haversine", paths=PATHS)


@pytest.mark.parametrize("sid", P.ids("sphere"))
def test_arb_references_true_neighbours_within_the_contract(sid):
    ref = refdata.ref("sphere")
    s = P.get(sid)
    rows = [k for k in range(ref.points) if ref.inputs["set"][k] == sid]
    Q = np.array([ref.inputs["query"][k] for k in rows])
    eta = refdata.C_HAVERSINE * refdata.U
    m = ref.methods["haversine"]
    for path in (1, 3):
        got = N._knn("kneighbors", Q, s.x, 9, "haversine", None, None, None, None, INF, path)
        d, i = np.asarray(got.distances), np.asarray(got.indices)
        for q, k in enumerate(rows):
            true = m.hi[k] + m.lo[k]
            dk = d[q, -1]
            mask = np.zeros(true.size, bool)
            mask[i[q]] = True
            assert np.all(true[mask] <= dk * (1 + eta) + refdata.ETA), (sid, q, path)
            assert np.all(true[~mask] >= dk * (1 - eta) - refdata.ETA), (sid, q, path)
            for j, v in zip(i[q], d[q]):
                assert abs(v - true[j]) <= refdata.bound("haversine", true[j], n=2), (sid, q, j)
        r = float(np.median(d[:, 4]))
        res = N._radius("radius_neighbors", Q, s.x, r, "haversine", None, None, None, None, False, path)
        ip, ix = np.asarray(res.indptr), np.asarray(res.indices)
        for q, k in enumerate(rows):
            true = m.hi[k] + m.lo[k]
            mask = np.zeros(true.size, bool)
            mask[ix[ip[q] : ip[q + 1]]] = True
            assert np.all(true[mask] <= r * (1 + eta) + refdata.ETA), (sid, q, path)
            assert np.all(true[~mask] >= r * (1 - eta) - refdata.ETA), (sid, q, path)


@st.composite
def _clusters(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    g = np.random.default_rng(seed)
    nc = draw(st.integers(1, 5))
    centers = np.column_stack([
        g.choice([0.0, 1.0, -1.0, LAT, -LAT], nc) * g.uniform(0.0, 1.0, nc) ** draw(st.sampled_from([0.0, 1.0])),
        g.choice([0.0, math.pi, -math.pi, 2.0 * math.pi], nc) + g.uniform(-0.5, 0.5, nc),
    ])  # fmt: skip
    spread = 10.0 ** draw(st.floats(-15.0, 0.0))
    mb = draw(st.integers(1, 400))
    B = centers[g.integers(0, nc, mb)] + g.normal(0.0, spread, (mb, 2))
    if draw(st.booleans()):
        B = np.round(B, draw(st.integers(1, 12)))  # ties
    if draw(st.booleans()):
        B[g.integers(0, mb, mb // 3)] = B[g.integers(0, mb, mb // 3)]  # duplicates
    ma = draw(st.integers(1, 40))
    A = np.vstack(
        [
            B[g.integers(0, mb, ma // 2)],
            centers[g.integers(0, nc, ma - ma // 2)] + g.normal(0.0, spread, (ma - ma // 2, 2)),
        ]
    )
    A[:, 0], B[:, 0] = np.clip(A[:, 0], -LAT, LAT), np.clip(B[:, 0], -LAT, LAT)
    return A, B, draw(st.integers(1, 60)), draw(st.floats(0.0, 1.0))


@settings(deadline=None)
@given(_clusters())
def test_hypothesis_clusters(case):
    A, B, k, q = case
    check_knn(A, B, k, "haversine", paths=(1, 3))
    D = np.asarray(S.cdist(A, B, metric="haversine"))
    r = float(np.quantile(D, q))
    check_radius(A, B, r, "haversine", paths=(1, 3))
    check_knn(A, B, k, "haversine", ub=r, paths=(1, 3))


@pytest.mark.usefixtures("restore_threads")
def test_one_three_ten_threads():
    g = np.random.default_rng(9)
    A, B = _sphere(g, 3000), _sphere(g, 20000)
    B[5000:6000] = B[:1000]  # ties
    got = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        out = list(S.kneighbors(A, B, k=8, metric="haversine"))
        out += list(N._knn("kneighbors", A[:40], B, 3, "haversine", None, None, None, None, INF, 3))
        r = S.radius_neighbors(A, B, 0.02, metric="haversine")
        out += [
            r.indptr,
            r.indices,
            r.distances,
            S.radius_neighbors(A, B, 0.05, metric="haversine", return_length=True),
        ]
        got[t] = out
    for t in (3, 10):
        for i, (x, y) in enumerate(zip(got[1], got[t])):
            assert bits_equal(x, y), (t, i)
