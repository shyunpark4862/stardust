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
    return out
