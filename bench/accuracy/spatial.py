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
`neighbors` joins with SPT2 and SPT3.
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region, ulps

FAMILIES = ("distances", "counting", "haversine")
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


def sweeps():
    out = []
    for fam in FAMILIES:

        def run(fam=fam):
            return _regions(fam, "sd")

        def measure(lib, fam=fam):
            return _regions(fam, lib)

        out.append(Custom("spatial", fam, run, compare_libs(measure, ("scipy", "sklearn", "R"), seconds=120.0)))
    return out
