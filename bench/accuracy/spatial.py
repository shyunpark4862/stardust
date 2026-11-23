"""sd.spatial's accuracy sweeps: one per reference family of tests/spatial/ref (tools/spatial_ref.py) whose
functions are public. SPT1 brings three:

  distances  every method of distances.npz (euclidean, sqeuclidean, cityblock, chebyshev, minkowski_p0.5,
             p1.5, p3, p7.5, seuclidean, mahalanobis, cosine, correlation, canberra, braycurtis and the
             weighted w_<metric>) as a region: one point per pair, the distance function of the method
             (sd.spatial.euclidean(u, v), ...; pdist and cdist give the same bits, the one pair kernel);
  counting   hamming, jaccard and their weighted forms on the bool and int64 pairs of counting.npz;
  haversine  haversine on haversine.npz (random pairs, antipodes, poles, identical and close points).

The contract is tests/spatial/refdata.py's (docs/design/spatial.md §4; the tests judge with the same
functions), in units of its bound. The comparators run live on the same pairs: scipy 1.18.1
(scipy.spatial.distance's functions of the same names), scikit-learn 1.9.1 (euclidean_distances,
also squared, manhattan_distances, cosine_distances and haversine_distances: its expanded and
1 - cos forms) and R 4.6.2's stats::dist (euclidean, manhattan, maximum, minkowski, canberra on the
pairs of nonnegative, normal coordinates without 0/0 terms: R's terms are |x - y| / |x + y|, and it
drops the terms whose denominator is below DBL_MIN and rescales for them; binary for the counting jaccard).

SPT2 adds

  neighbors  the k = 5 nearest of every query of neighbors.npz (euclidean, cityblock, chebyshev: the
             methods p2, p1, pinf) by sd.spatial.kneighbors, one region per metric and point set, one point
             per query. The contract (docs/design/spatial.md §4 neighbours 2 and 3): on the lattice sets the
             answer is the Arb order (RN(d), index) with the distances RN(d) (0 when it is, inf when not);
             elsewhere every returned point's true distance is at most d_(k)(1 + eta) and every other one's
             at least d_(k)(1 - eta) (units: the largest excess over d_(k) in eta d_(k)). ulp: the returned
             k-th distance against the reference at its index. Comparators: scipy's KDTree.query and
             scikit-learn's NearestNeighbors(algorithm='brute') with the same metric.

SPT3 adds

  kdtree     the same points and judgement through sd.spatial.KDTree(X).query(Q, k=5, p=1, 2, inf), with
             scipy's and scikit-learn's KDTree.query as the comparators.

SPT4 adds

  ball       sd.spatial.KDTree(X).query_ball_point(q, r, p=) with r the 5th smallest reference distance
             RN(d) of the query (a boundary: points at exactly r are in), one region per metric and point
             set. The contract: on the lattice sets the points are those with RN(d) <= r and their
             distances RN(d); elsewhere every returned point's true distance is at most r(1 + eta) and
             every other one's at least r(1 - eta) (units: the largest excess over r in eta r). ulp: the
             largest returned distance against its reference. Comparators: scipy's
             KDTree.query_ball_point and scikit-learn's KDTree.query_radius (points only; their
             distances are not compared).

The haversine tree path (SPT83) adds

  sphere     the k = 5 nearest of every query of sphere.npz (the (latitude, longitude) sets of tag sphere:
             uniform, near-identical clusters, antipodes, the poles, the dateline, exact ties) by
             sd.spatial.kneighbors(metric='haversine') (the KD tree on the unit sphere), one region per set,
             judged as `neighbors` on general data with eta = 20u (the haversine contract). Comparator:
             scikit-learn's BallTree(metric='haversine').query.

The computational geometry (docs/design/spatial_geometry.md §4, §7.5; SPT13, decision SPT187) adds four, on the
point sets of tests/spatial/geompoints.py the classes take, with the references of geom_measure.npz,
voronoi_vertices.npz and barycentric.npz:

  geom_measure      ConvexHull's area and volume (d = 2 perimeter and enclosed area, d = 3 surface area and
                    volume), one region per measure and d, one point per set. The contract: relative error
                    gamma_(s+4) (perimeter), gamma_(s+2) (area), gamma_(s+6) (surface), gamma_(s+3) (volume), s the
                    additions of the 4,096-term partial sums over sd's own terms, 2^-1074 per term near the
                    subnormals; an overflowing measure is inf. Comparator: scipy's ConvexHull (Qhull).
  voronoi_vertices  Voronoi's vertices, one region per set, one point per vertex coordinate: |v^_k - v_k| <=
                    u |v_k| + 4 u |v_k - a_k| (a the first vertex of the cell's smallest simplex) + 2^-1073.
                    Comparator: scipy's Voronoi, each of its vertices against the nearest reference vertex (ulp
                    only: its numbering is Qhull's).
  barycentric       Delaunay.transform at the query points of geompoints.queries, one region per set: the
                    contract is transform's (every entry of T^-1 within 4u of the exact one, relative; an entry
                    that overflows or underflows to 0 is out of the contract and counted in excluded); ulp: those
                    entries; abs: the barycentric coordinates T^-1 (x - r), 1 - their sum computed the way scipy's
                    documentation shows. Comparator: scipy's Delaunay, the coordinates only (its simplex must
                    hold the same vertices, else the query is excluded).
  delaunay_exact    §4 item 1 (every simplex positive, the simplices tile the hull, every distinct point a
                    vertex, empty spheres) checked exactly per set: contract 0 when it holds, inf when not (the
                    worst names the set). By exactgeom.check_delaunay (d = 2, and d = 3 up to 250 rows: the
                    tests' split), else check_delaunay3_large.
                    Comparator: scipy's Delaunay (its simplices turned positive and its equal points read as their
                    representative; a flat simplex is a failure).
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region, ulps

FAMILIES = ("distances", "counting", "haversine")
NEIGHBOR_K = 5
_NN_METHODS = (("p2", "euclidean", 2.0), ("p1", "cityblock", 1.0), ("pinf", "chebyshev", np.inf))
_REFUSED = object()

# sklearn's functions for a method of the references (None: sklearn has no such distance)
_SKLEARN = {
    "euclidean": ("euclidean_distances", {}),
    "sqeuclidean": ("euclidean_distances", {"squared": True}),
    "cityblock": ("manhattan_distances", {}),
    "cosine": ("cosine_distances", {}),
    "haversine": ("haversine_distances", {}),
}
# R's stats::dist method for a method of the references
_R = {
    "euclidean": "euclidean",
    "cityblock": "manhattan",
    "chebyshev": "maximum",
    "canberra": "canberra",
    "minkowski_p0.5": "minkowski",
    "minkowski_p1.5": "minkowski",
    "minkowski_p3": "minkowski",
    "minkowski_p7.5": "minkowski",
    "jaccard": "binary",
}


def _points(family, method):
    from tests.spatial import refdata

    r = refdata.ref(family)
    m = r.methods[method]
    for j, k in enumerate(m.index):
        yield j, int(k), refdata.inputs(family, int(k)), f"{r.inputs['set'][k]}#{int(r.inputs['pair'][k])}"


def _value(lib, method, x):
    """lib's distance at one point (None: lib has no such distance)."""
    import warnings

    from tests.spatial import refdata

    if lib == "sd":
        import stardust as sd

        return refdata.call(sd.spatial, method, x["u"], x["v"], x.get("w"), x.get("V"), x.get("VI"))
    if lib == "scipy":
        import scipy.spatial.distance as D

        if method == "haversine":
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return float(refdata.call(D, method, x["u"], x["v"], x.get("w"), x.get("V"), x.get("VI")))
            except ValueError:  # scipy refuses (its chebyshev of no columns): left out, counted in "excluded"
                return _REFUSED
    if lib == "sklearn":
        if method not in _SKLEARN:
            return None
        import sklearn.metrics.pairwise as K

        name, kw = _SKLEARN[method]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u, v = np.atleast_2d(x["u"]).astype(np.float64), np.atleast_2d(x["v"]).astype(np.float64)
            if u.shape[1] == 0:
                return None
            return float(getattr(K, name)(u, v, **kw)[0, 0])
    return None


def _regions(family, lib):
    from tests.spatial import refdata

    if lib == "R":
        return _r_regions(family)
    out = {}
    for method in sorted(refdata.ref(family).methods):
        where, err, rel, ul = [], [], [], []
        m = refdata.ref(family).methods[method]
        refused = 0
        for j, k, x, at in _points(family, method):
            got = _value(lib, method, x)
            if got is None:
                continue
            if got is _REFUSED:
                refused += 1
                continue
            hi, lo = float(m.hi[j]), float(m.lo[j])
            where.append(at)
            err.append(refdata.point_units(family, method, k, got, j))
            with np.errstate(all="ignore"):
                rel.append(abs((got - hi) - lo) / abs(hi) if hi else abs(got - hi))
            ul.append(float(ulps(got, hi)))
        if where:
            out[method] = region(where, ulp=np.array(ul), rel=np.array(rel), contract=np.array(err), bound=1.0,
                                 excluded=refused if lib != "sd" else None)  # fmt: skip
    return out or None


def _r_regions(family):
    """R's stats::dist on the pairs of the methods it has (one R call per method)."""
    from tests.spatial import refdata

    from . import rsession

    out = {}
    for method in sorted(refdata.ref(family).methods):
        if method not in _R:
            continue
        m = refdata.ref(family).methods[method]
        pts = []
        for j, k, x, at in _points(family, method):
            u, v = np.asarray(x["u"], dtype=np.float64), np.asarray(x["v"], dtype=np.float64)
            if u.size == 0 or not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
                continue
            tiny = np.any((np.abs(np.concatenate([u, v])) < 2.2250738585072014e-308) & (np.concatenate([u, v]) != 0))
            if method == "canberra" and (np.any((u == 0) & (v == 0)) or np.any(u < 0) or np.any(v < 0) or tiny):
                continue  # R's terms: |x - y| / |x + y| (scipy's for x, y >= 0), terms below DBL_MIN dropped
            pts.append((j, k, at, u, v))
        if not pts:
            continue
        p = float(method[len("minkowski_p") :]) if method.startswith("minkowski") else 2.0
        expr = (
            "{ off <- c(0, cumsum(N)); sapply(seq_along(N), function(k) { i <- (off[k] + 1):off[k + 1]; "
            f"as.numeric(dist(rbind(U[i], V[i]), method = '{_R[method]}', p = {p!r})) }}) }}"
        )
        vals = rsession.evaluate_named(
            expr,
            {"U": np.concatenate([q[3] for q in pts]), "V": np.concatenate([q[4] for q in pts]),
             "N": np.array([q[3].size for q in pts], dtype=np.float64)},
        )  # fmt: skip
        where, err, rel, ul = [], [], [], []
        for (j, k, at, _, _), got in zip(pts, vals):
            got = float(got)
            hi, lo = float(m.hi[j]), float(m.lo[j])
            where.append(at)
            err.append(refdata.point_units(family, method, k, got, j))
            with np.errstate(all="ignore"):
                rel.append(abs((got - hi) - lo) / abs(hi) if hi else abs(got - hi))
            ul.append(float(ulps(got, hi)))
        out[method] = region(where, ulp=np.array(ul), rel=np.array(rel), contract=np.array(err), bound=1.0)
    return out or None


def _knn(lib, metric, p, Q, X):
    """lib's k nearest (distances, indices) of the rows of Q in X (None: lib has no such search)."""
    k = NEIGHBOR_K
    if lib == "sd":
        import stardust as sd

        d, i = sd.spatial.kneighbors(Q, X, k=k, metric=metric)
        return np.asarray(d), np.asarray(i)
    if lib == "scipy":
        import scipy.spatial as S

        return S.KDTree(X).query(Q, k=k, p=p)
    if lib == "sklearn":
        from sklearn.neighbors import NearestNeighbors

        name = {"euclidean": "euclidean", "cityblock": "manhattan", "chebyshev": "chebyshev"}[metric]
        return NearestNeighbors(n_neighbors=k, algorithm="brute", metric=name).fit(X).kneighbors(Q)
    return None


def _tree_knn(lib, metric, p, Q, X):
    """The `kdtree` sweep (SPT3): the k nearest by a KD tree, sd.spatial.KDTree(X).query(Q, k=, p=) against
    scipy's and scikit-learn's KDTree.query (the same judgement as `neighbors`)."""
    k = NEIGHBOR_K
    if lib == "sd":
        import stardust as sd

        d, i = sd.spatial.KDTree(X).query(Q, k=k, p=p)
        return np.asarray(d), np.asarray(i)
    if lib == "scipy":
        import scipy.spatial as S

        return S.KDTree(X).query(Q, k=k, p=p)
    if lib == "sklearn":
        from sklearn.neighbors import KDTree

        name = {"euclidean": "euclidean", "cityblock": "manhattan", "chebyshev": "chebyshev"}[metric]
        return KDTree(X, metric=name).query(Q, k=k)
    return None


def _neighbor_regions(lib, knn=None):
    from tests.spatial import points as P
    from tests.spatial import refdata

    r = refdata.ref("neighbors")
    out = {}
    for method, metric, p in _NN_METHODS:
        m = r.methods[method]
        for sid in P.ids("neighbors"):
            rows = [k for k in range(r.points) if r.inputs["set"][k] == sid]
            s = P.get(sid)
            Q = np.array([r.inputs["query"][k] for k in rows])
            got = (knn or _knn)(lib, metric, p, Q, s.x)
            if got is None:
                continue
            d, i = np.asarray(got[0], dtype=np.float64), np.asarray(got[1], dtype=np.int64)
            lattice = "lattice" in s.tags
            eta = max(refdata.bound(metric, 1.0, n=s.n) - refdata.ETA, 2.0**-52)
            where, units, ul = [], [], []
            for q, k in enumerate(rows):
                hi, lo = m.hi[k], m.lo[k]
                true = hi + lo
                dk = d[q, -1]
                if lattice:
                    want = np.lexsort((np.arange(hi.size), hi))[:NEIGHBOR_K]
                    ok = np.array_equal(i[q], want) and np.array_equal(d[q], hi[want])
                    units.append(0.0 if ok else np.inf)
                else:
                    mask = np.zeros(true.size, bool)
                    mask[i[q]] = True
                    over = max(
                        float(np.max(true[mask] - dk, initial=0.0)), float(np.max(dk - true[~mask], initial=0.0))
                    )
                    units.append(over / (eta * dk) if dk > 0 else (0.0 if over == 0 else np.inf))
                ul.append(float(ulps(dk, hi[i[q, -1]])))
                where.append(f"{sid}#{int(r.inputs['pair'][k])}")
            out[f"{metric}/{sid}"] = region(where, ulp=np.array(ul), contract=np.array(units), bound=1.0)
    return out or None


def _ball(lib, metric, p, q, X, r):
    """lib's points within r of q (indices, distances or None), or None when lib has no such search."""
    if lib == "sd":
        import stardust as sd

        res = sd.spatial.KDTree(X).query_ball_point(q, r, p=p)
        return np.asarray(res.indices, dtype=np.int64), np.asarray(res.distances)
    if lib == "scipy":
        import scipy.spatial as S

        return np.sort(np.asarray(S.KDTree(X).query_ball_point(q, r, p=p), dtype=np.int64)), None
    if lib == "sklearn":
        from sklearn.neighbors import KDTree

        name = {"euclidean": "euclidean", "cityblock": "manhattan", "chebyshev": "chebyshev"}[metric]
        return np.sort(KDTree(X, metric=name).query_radius(q[None, :], r)[0].astype(np.int64)), None
    return None


def _ball_regions(lib):
    """The `ball` sweep (SPT4): query_ball_point at a boundary radius, judged as the docstring says."""
    from tests.spatial import points as P
    from tests.spatial import refdata

    r = refdata.ref("neighbors")
    out = {}
    for method, metric, p in _NN_METHODS:
        m = r.methods[method]
        for sid in P.ids("neighbors"):
            s = P.get(sid)
            lattice = "lattice" in s.tags
            eta = max(refdata.bound(metric, 1.0, n=s.n) - refdata.ETA, 2.0**-52)
            where, units, ul = [], [], []
            for k in (k for k in range(r.points) if r.inputs["set"][k] == sid):
                hi, lo = m.hi[k], m.lo[k]
                true = hi + lo
                rad = float(np.sort(hi)[NEIGHBOR_K - 1])
                got = _ball(lib, metric, p, np.asarray(r.inputs["query"][k], dtype=np.float64), s.x, rad)
                if got is None:
                    break
                idx, dist = got
                if lattice:
                    want = np.nonzero(hi <= rad)[0]
                    ok = np.array_equal(idx, want) and (dist is None or np.array_equal(dist, hi[want]))
                    units.append(0.0 if ok else np.inf)
                else:
                    mask = np.zeros(true.size, bool)
                    mask[idx] = True
                    over = max(
                        float(np.max(true[mask] - rad, initial=0.0)), float(np.max(rad - true[~mask], initial=0.0))
                    )
                    units.append(over / (eta * rad) if rad > 0 else (0.0 if over == 0 else np.inf))
                if dist is not None and dist.size:
                    j = int(np.argmax(dist))
                    ul.append(float(ulps(dist[j], hi[idx[j]])))
                else:
                    ul.append(0.0)
                where.append(f"{sid}#{int(r.inputs['pair'][k])}")
            if where:
                out[f"{metric}/{sid}"] = region(where, ulp=np.array(ul), contract=np.array(units), bound=1.0)
    return out or None


def _sphere_regions(lib):
    """The `sphere` sweep: the haversine k nearest against sphere.npz, judged as the docstring says."""
    from tests.spatial import points as P
    from tests.spatial import refdata

    r = refdata.ref("sphere")
    m = r.methods["haversine"]
    eta = refdata.C_HAVERSINE * refdata.U
    out = {}
    for sid in P.ids("sphere"):
        rows = [k for k in range(r.points) if r.inputs["set"][k] == sid]
        s = P.get(sid)
        Q = np.array([r.inputs["query"][k] for k in rows])
        if lib == "sd":
            import stardust as sd

            d, i = sd.spatial.kneighbors(Q, s.x, k=NEIGHBOR_K, metric="haversine")
        elif lib == "sklearn":
            from sklearn.neighbors import BallTree

            d, i = BallTree(s.x, metric="haversine").query(Q, k=NEIGHBOR_K)
        else:
            return None
        d, i = np.asarray(d, dtype=np.float64), np.asarray(i, dtype=np.int64)
        where, units, ul = [], [], []
        for q, k in enumerate(rows):
            hi, lo = m.hi[k], m.lo[k]
            true = hi + lo
            dk = d[q, -1]
            mask = np.zeros(true.size, bool)
            mask[i[q]] = True
            over = max(float(np.max(true[mask] - dk, initial=0.0)), float(np.max(dk - true[~mask], initial=0.0)))
            units.append(over / (eta * dk) if dk > 0 else (0.0 if over == 0 else np.inf))
            ul.append(float(ulps(dk, hi[i[q, -1]])))
            where.append(f"{sid}#{int(r.inputs['pair'][k])}")
        out[sid] = region(where, ulp=np.array(ul), contract=np.array(units), bound=1.0)
    return out


# ---------------------------------------------------------------- computational geometry (SPT187)

_U = 2.0**-53
_TINY = 2.0**-1074


def _gamma(k):
    return k * _U / (1 - k * _U)


def _sum_depth(n):
    """The most additions a term passes in the sum of 4,096-term partial sums (§4's s)."""
    return min(n, 4096) + -(-n // 4096) - 2


def _sd_points(x):
    import stardust as sd

    return sd.asarray(np.ascontiguousarray(x, dtype=np.float64))


def _exact_err(got, hi, lo):
    """|got - (hi + lo)| in Fractions (inf when got is not finite and the reference is, 0 when both are the same inf)."""
    from fractions import Fraction

    if not np.isfinite(got) or not np.isfinite(hi):
        return 0.0 if got == hi else np.inf
    return abs(Fraction(float(got)) - Fraction(float(hi)) - Fraction(float(lo)))


def _geom_measure_regions(lib):
    from fractions import Fraction

    from tests.spatial import refdata

    r = refdata.ref("geom_measure")
    names = {(2, "area"): "perimeter", (2, "volume"): "area", (3, "area"): "surface", (3, "volume"): "volume"}
    rows = {}
    for k in range(r.points):
        sid, x = str(r.inputs["set"][k]), r.inputs["x"][k]
        d = x.shape[1]
        if lib == "sd":
            import stardust as sd

            h = sd.spatial.ConvexHull(_sd_points(x))
            got = {"area": h.area, "volume": h.volume}
            nf, nv = h.nsimplex, int(np.asarray(h.vertices).size)
        elif lib == "scipy":
            from scipy.spatial import ConvexHull

            try:
                h = ConvexHull(x)
                got = {"area": float(h.area), "volume": float(h.volume)}
            except Exception:  # Qhull refuses the set: left out
                got = {"area": np.nan, "volume": np.nan}
            nf = nv = 0
        else:
            return None
        for m in ("area", "volume"):
            j = int(np.nonzero(r.methods[m].index == k)[0][0])
            hi, lo = float(r.methods[m].hi[j]), float(r.methods[m].lo[j])
            g = float(got[m])
            e = rows.setdefault(names[(d, m)], {"where": [], "ulp": [], "rel": [], "units": [], "excl": 0})
            if not np.isfinite(g) and np.isfinite(hi):
                e["excl"] += lib != "sd"
                if lib != "sd":
                    continue
            err = _exact_err(g, hi, lo)
            ref = Fraction(hi) + Fraction(lo) if np.isfinite(hi) else None
            e["where"].append(sid)
            e["ulp"].append(float(ulps(g, hi)))
            e["rel"].append(float(err / abs(ref)) if ref else (0.0 if err == 0 else np.inf))
            if lib == "sd":
                terms = {("area", 2): nf, ("volume", 2): nv - 2, ("area", 3): nf, ("volume", 3): nf}[(m, d)]
                extra = {("area", 2): 4, ("volume", 2): 2, ("area", 3): 6, ("volume", 3): 3}[(m, d)]
                if ref is None:
                    e["units"].append(0.0 if err == 0 else np.inf)
                else:
                    bound = Fraction(_gamma(_sum_depth(terms) + extra)) * abs(ref) + terms * Fraction(_TINY)
                    e["units"].append(float(err / bound) if bound else (0.0 if err == 0 else np.inf))
    out = {}
    for name, e in rows.items():
        c = np.array(e["units"]) if lib == "sd" else None
        out[name] = region(e["where"], ulp=np.array(e["ulp"]), rel=np.array(e["rel"]), contract=c,
                           bound=1.0 if lib == "sd" else None, excluded=None if lib == "sd" else e["excl"])  # fmt: skip
    return out


def _voronoi_regions(lib):
    from fractions import Fraction

    from tests.spatial import refdata

    r = refdata.ref("voronoi_vertices")
    m = r.methods["vertex"]
    out = {}
    for j, k in enumerate(m.index):
        sid, x = str(r.inputs["set"][k]), r.inputs["x"][k]
        d = x.shape[1]
        hi, lo = np.asarray(m.hi[j]).reshape(-1, d), np.asarray(m.lo[j]).reshape(-1, d)
        rows = np.asarray(m.fields["simplex"][j]).reshape(-1, d + 1)
        where, ul, units = [], [], []
        excluded = 0
        if lib == "sd":
            import stardust as sd

            V = np.asarray(sd.spatial.Voronoi(_sd_points(x)).vertices)
            if V.shape != hi.shape:  # a different number of vertices: the numbering broke
                out[sid] = region([f"{sid}#count"], ulp=np.array([np.inf]), contract=np.array([np.inf]), bound=1.0)
                continue
            for v in range(len(V)):
                a = x[rows[v, 0]]
                for c in range(d):
                    got, h, low = float(V[v, c]), float(hi[v, c]), float(lo[v, c])
                    ref = Fraction(h) + Fraction(low) if np.isfinite(h) else None
                    err = _exact_err(got, h, low)
                    if ref is None or not np.isfinite(got):
                        u = 0.0 if got == h else np.inf
                    else:
                        bound = Fraction(_U) * (abs(ref) + 4 * abs(ref - Fraction(float(a[c])))) + 2 * Fraction(_TINY)
                        u = float(err / bound) if bound else (0.0 if err == 0 else np.inf)
                    where.append(f"{sid}#v{v}")
                    ul.append(float(ulps(got, h)))
                    units.append(u)
            out[sid] = region(where, ulp=np.array(ul), contract=np.array(units), bound=1.0)
        elif lib == "scipy":
            from scipy.spatial import Voronoi, cKDTree

            try:
                V = np.asarray(Voronoi(x).vertices)
            except Exception:
                continue
            ok = np.isfinite(hi).all(axis=1)
            if not ok.any():
                continue
            fin = np.isfinite(V).all(axis=1)
            dist, near = np.full(len(V), np.inf), np.zeros(len(V), dtype=np.int64)
            with np.errstate(all="ignore"):
                dist[fin], near[fin] = cKDTree(hi[ok]).query(V[fin])
            ref = hi[ok][near]
            for v in range(len(V)):
                if not np.isfinite(dist[v]):
                    excluded += 1
                    continue
                where.append(f"{sid}#v{v}")
                ul.append(float(np.max(ulps(V[v], ref[v]))))
            if where:
                out[sid] = region(where, ulp=np.array(ul), excluded=excluded)
        else:
            return None
    return out


def _barycentric_regions(lib):
    from fractions import Fraction

    from tests.spatial import geompoints as G
    from tests.spatial import refdata

    r = refdata.ref("barycentric")
    inv, bar = r.methods["inverse"], r.methods["barycentric"]
    by = {}
    for j, k in enumerate(inv.index):
        by.setdefault(str(r.inputs["set"][k]), []).append((j, int(k)))
    out = {}
    for sid, items in by.items():
        x = G.get(sid).x
        d = x.shape[1]
        Q = np.array([r.inputs["query"][k] for _, k in items])
        where, ul, units, ab = [], [], [], []
        excluded = 0
        if lib == "sd":
            import stardust as sd

            D = sd.spatial.Delaunay(_sd_points(x))
            T, S = np.asarray(D.transform), np.asarray(D.simplices)
            found = np.asarray(D.find_simplex(_sd_points(Q)))
        elif lib == "scipy":
            from scipy.spatial import Delaunay

            try:
                D = Delaunay(x)
            except Exception:
                continue
            T, S = D.transform, D.simplices
            found = D.find_simplex(Q)
        else:
            return None
        for (j, k), q, t in zip(items, Q, found):
            row = np.asarray(inv.fields["simplex"][j])
            if t < 0 or sorted(S[t].tolist()) != sorted(row.tolist()):
                if lib == "sd":
                    where.append(f"{sid}#{r.inputs['pair'][k]}")
                    ul.append(np.inf), units.append(np.inf), ab.append(np.inf)
                else:
                    excluded += 1
                continue
            with np.errstate(all="ignore"):
                c = T[t, :d].dot(q - T[t, d])
                lam = np.append(c, 1 - c.sum())
            pos = {int(v): i for i, v in enumerate(S[t])}
            lam = lam[[pos[int(v)] for v in row]]  # the reference's (canonical) vertex order
            bh, bl = np.asarray(bar.hi[j]), np.asarray(bar.lo[j])
            errs = [float(_exact_err(a, h, lo)) for a, h, lo in zip(lam, bh, bl)]
            if not all(np.isfinite(errs)):
                excluded += 1
                continue
            where.append(f"{sid}#{r.inputs['pair'][k]}")
            ab.append(max(errs))
            if lib == "sd":
                ih, il = np.asarray(inv.hi[j]), np.asarray(inv.lo[j])
                got = T[t, :d].ravel()
                worst_u, worst_ulp = 0.0, 0.0
                for g, h, lo in zip(got, ih, il):
                    ref = Fraction(float(h)) + Fraction(float(lo)) if np.isfinite(h) else None
                    if ref is None or not np.isfinite(g) or (g == 0 and ref != 0):
                        excluded += 1  # overflow or underflow of the result: outside the contract
                        continue
                    worst_u = max(
                        worst_u, float(abs(Fraction(float(g)) - ref) / (4 * Fraction(_U) * abs(ref))) if ref else 0.0
                    )
                    worst_ulp = max(worst_ulp, float(ulps(g, h)))
                units.append(worst_u)
                ul.append(worst_ulp)
        if not where:
            continue
        if lib == "sd":
            out[sid] = region(where, ulp=np.array(ul), contract=np.array(units), bound=1.0, abs_err=np.array(ab),
                              excluded=excluded)  # fmt: skip
        else:
            out[sid] = region(where, abs_err=np.array(ab), excluded=excluded)
    return out


def _delaunay_exact_regions(lib):
    """§4 item 1 per set: 0 (holds) or inf, one region per d."""
    from tests.spatial import exactgeom as X
    from tests.spatial import geompoints as G

    ids = [i for i in G.IDS if not {"invalid", "large"} & G.get(i).tags]
    rows = {}
    for sid in ids:
        x = G.get(sid).x
        d = x.shape[1]
        try:
            if lib == "sd":
                import stardust as sd

                D = sd.spatial.Delaunay(_sd_points(x))
                S, NB = np.asarray(D.simplices), np.asarray(D.neighbors)
            elif lib == "scipy":
                from scipy.spatial import Delaunay

                D = Delaunay(x)
                S, NB = D.simplices.astype(np.int64), D.neighbors.astype(np.int64)
                rep = np.array(X.representatives(X.fpoints(x)), dtype=np.int64)
                S = rep[S]  # scipy's equal points read as their representative
                for t in range(len(S)):  # positive rows: swap the last two vertices (and their neighbours)
                    o = X.orient(*X.fpoints(x[S[t]]))
                    if o < 0:
                        S[t, [-2, -1]] = S[t, [-1, -2]]
                        NB[t, [-2, -1]] = NB[t, [-1, -2]]
            else:
                return None
            if d == 2 or len(x) <= 250:  # the float filter of the large checker needs no overflow or underflow
                X.check_delaunay(x, S, neighbors=NB, exhaustive=60)
            else:
                X.check_delaunay3_large(x, S, NB, perturbed=False)
            units = 0.0
        except AssertionError:
            units = np.inf
        except Exception:  # the library refuses the set (Qhull)
            if lib == "sd":
                raise
            continue
        e = rows.setdefault(f"d{d}", ([], []))
        e[0].append(sid)
        e[1].append(units)
    return {k: region(w, contract=np.array(u), bound=1.0) for k, (w, u) in rows.items()}


def sweeps():
    out = []
    for fam in FAMILIES:

        def run(fam=fam):
            return _regions(fam, "sd")

        def measure(lib, fam=fam):
            return _regions(fam, lib)

        out.append(Custom("spatial", fam, run, compare_libs(measure, ("scipy", "sklearn", "R"), seconds=120.0)))
    out.append(Custom("spatial", "neighbors", lambda: _neighbor_regions("sd"),
                      compare_libs(_neighbor_regions, ("scipy", "sklearn"), seconds=120.0)))  # fmt: skip
    out.append(Custom("spatial", "kdtree", lambda: _neighbor_regions("sd", _tree_knn),
                      compare_libs(lambda lib: _neighbor_regions(lib, _tree_knn), ("scipy", "sklearn"), seconds=120.0)))  # fmt: skip
    out.append(Custom("spatial", "ball", lambda: _ball_regions("sd"),
                      compare_libs(_ball_regions, ("scipy", "sklearn"), seconds=120.0)))  # fmt: skip
    out.append(Custom("spatial", "sphere", lambda: _sphere_regions("sd"),
                      compare_libs(_sphere_regions, ("sklearn",), seconds=120.0)))  # fmt: skip
    for name, fn in (("geom_measure", _geom_measure_regions), ("voronoi_vertices", _voronoi_regions),
                     ("barycentric", _barycentric_regions), ("delaunay_exact", _delaunay_exact_regions)):  # fmt: skip
        out.append(Custom("spatial", name, lambda fn=fn: fn("sd"), compare_libs(fn, ("scipy",), seconds=600.0)))
    return out
