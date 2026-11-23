"""sd.spatial against scipy.spatial 1.18.1, scikit-learn 1.9.1 and R 4.6.2's stats::dist
(docs/design/spatial.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the
sd.spatial names it needs, and cases() keeps a spec once its names are public (SPT1 brings the
distances, SPT2 kneighbors and radius_neighbors, SPT3 KDTree and query, SPT4 the radius, pair and
count queries). The sizes are those of docs/design/spatial.md §5, where SPT0 timed the comparators.
scipy's distance and tree kernels run on one thread; the tree queries are also compared with
scipy's workers=-1, since sd runs on its pool by default. scikit-learn's expanded-form functions
use the BLAS's threads (threads="follow"). R's dist runs in the one R session.

Data: standard normal points (distances, one-call searches) and uniform points in [0, 1)^m (trees),
numpy arrays made in the setup from fixed seeds; sd gets them as sd.ndarray, outside the timing.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..registry import Case, Check, Ref, RRef


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]
    size: str
    setup: Callable[[], dict]
    sd: Callable
    refs: tuple = ()
    quick: bool = False
    note: str = ""
    check: Check = dataclasses.field(default_factory=Check)


def _normal(ma, mb, n, seed=20261122):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, ma, mb, n]))
        a, b = g.standard_normal((ma, n)), g.standard_normal((mb, n))
        return {"a": a, "b": b, "sa": sd.asarray(a), "sb": sd.asarray(b), "r_a": a.ravel(order="F"), "r_m": float(ma),
                "r_n": float(n)}  # fmt: skip

    return setup


def _geo(ma, mb, seed=20261123):
    """(latitude, longitude) in radians, uniform on the sphere: XA (ma rows) and XB (mb rows)."""

    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, ma, mb]))
        a, b = (np.column_stack([np.arcsin(g.uniform(-1.0, 1.0, m)), g.uniform(-np.pi, np.pi, m)]) for m in (ma, mb))
        return {"a": a, "b": b, "sa": sd.asarray(a), "sb": sd.asarray(b), "r": 0.01}

    return setup


def _bool(ma, mb, n, seed=20261124):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, ma, mb, n]))
        a, b = g.random((ma, n)) < 0.3, g.random((mb, n)) < 0.3
        return {"a": a, "b": b, "sa": sd.asarray(a), "sb": sd.asarray(b)}

    return setup


def _latlon(m, seed=20261125):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, m]))
        a = np.column_stack([g.uniform(-1.5, 1.5, m), g.uniform(-3.1, 3.1, m)])
        return {"a": a, "sa": sd.asarray(a)}

    return setup


def _uniform(npts, m, nq, seed=20261123):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, npts, m, nq]))
        x, q = g.random((npts, m)), g.random((nq, m))
        return {"x": x, "q": q, "sx": sd.asarray(x), "sq": sd.asarray(q)}

    return setup


def _tree_setup(npts, m, nq):
    """The uniform points with the trees built in the setup (the query cases time the query only)."""
    base = _uniform(npts, m, nq)

    def setup():
        import scipy.spatial as S
        from sklearn.neighbors import KDTree as SKDTree

        import stardust as sd

        d = base()
        d["T"], d["ST"], d["sT"] = S.KDTree(d["x"]), SKDTree(d["x"]), sd.spatial.KDTree(d["sx"])
        return d

    return setup


def _sp():
    import stardust as sd

    return sd.spatial


def _scipy_d():
    import scipy.spatial.distance as D

    return D


R_DIST = "dist(matrix(r_a, nrow = r_m), method = '{}')"
R_DATA = {"r_a": "r_a", "r_m": "r_m"}
BALL_R = (10 / 10**6 * 3 / (4 * np.pi)) ** (1 / 3)  # about 10 points of 10^6 uniform ones in the unit cube
PAIR_R = (10 / 10**5 * 3 / (4 * np.pi)) ** (1 / 3)  # about 10 points of 10^5 uniform ones
COUNT_R = np.linspace(0.001, 0.05, 20)  # count_neighbors' 20 radii (docs/design/spatial.md §7.5)


def _distance_specs():
    out = []
    for ma, n in ((1000, 3), (1000, 64), (4000, 3), (4000, 64), (4000, 512)):
        size = f"({ma}, {n})"
        refs = (
            Ref("scipy", lambda d: lambda: _scipy_d().cdist(d["a"], d["b"])),
            Ref("sklearn", lambda d: (lambda f: lambda: f(d["a"], d["b"]))(_sk("euclidean_distances")), threads="follow",
                differs="the expanded form (cancellation; docs/design/spatial.md §3.2)"),
        )  # fmt: skip
        out.append(Spec("SPT1", "distance", "cdist euclidean", ("cdist",), size, _normal(ma, ma, n),
                        lambda d: lambda: _sp().cdist(d["sa"], d["sb"]), refs, quick=(ma, n) == (1000, 64)))  # fmt: skip
    for metric in ("cityblock", "chebyshev", "minkowski", "cosine", "correlation", "canberra", "braycurtis"):
        kw = {"p": 3.0} if metric == "minkowski" else {}
        out.append(Spec("SPT1", "distance", f"cdist {metric}", ("cdist",), "(1000, 64)", _normal(1000, 1000, 64),
                        lambda d, m=metric, kw=kw: lambda: _sp().cdist(d["sa"], d["sb"], metric=m, **kw),
                        (Ref("scipy", lambda d, m=metric, kw=kw: lambda: _scipy_d().cdist(d["a"], d["b"], m, **kw)),)))  # fmt: skip
    for metric, kw in (("seuclidean", {}), ("mahalanobis", {})):
        n = 16 if metric == "mahalanobis" else 64
        out.append(Spec("SPT1", "distance", f"cdist {metric}", ("cdist",), f"(1000, {n})", _normal(1000, 1000, n),
                        lambda d, m=metric: lambda: _sp().cdist(d["sa"], d["sb"], metric=m),
                        (Ref("scipy", lambda d, m=metric: lambda: _scipy_d().cdist(d["a"], d["b"], m)),),
                        note="default V / VI from the data"))  # fmt: skip
    out.append(Spec("SPT1", "distance", "cdist hamming bool", ("cdist",), "(1000, 256)", _bool(1000, 1000, 256),
                    lambda d: lambda: _sp().cdist(d["sa"], d["sb"], metric="hamming"),
                    (Ref("scipy", lambda d: lambda: _scipy_d().cdist(d["a"], d["b"], "hamming")),),
                    note="bit-packed rows (SPT23)"))  # fmt: skip
    out.append(Spec("SPT1", "distance", "cdist haversine", ("cdist",), "(2000, 2)", _latlon(2000),
                    lambda d: lambda: _sp().cdist(d["sa"], d["sa"], metric="haversine"),
                    (Ref("sklearn", lambda d: (lambda f: lambda: f(d["a"]))(_sk("haversine_distances")),
                         threads="follow"),)))  # fmt: skip
    for n in (3, 64):
        out.append(Spec("SPT1", "distance", "pdist cosine", ("pdist",), f"(2000, {n})", _normal(2000, 1, n),
                        lambda d: lambda: _sp().pdist(d["sa"], metric="cosine"),
                        (Ref("scipy", lambda d: lambda: _scipy_d().pdist(d["a"], "cosine")),)))  # fmt: skip
    for n in (3, 64):
        refs = [Ref("scipy", lambda d: lambda: _scipy_d().pdist(d["a"]))]
        refs.append(RRef("R", R_DIST.format("euclidean"), data=R_DATA))
        out.append(Spec("SPT1", "distance", "pdist euclidean", ("pdist",), f"(2000, {n})", _normal(2000, 1, n),
                        lambda d: lambda: _sp().pdist(d["sa"]), tuple(refs), quick=n == 3))  # fmt: skip
    return out


def _sk(name):
    import sklearn.metrics.pairwise as K

    return getattr(K, name)


def _neighbor_specs():
    out = []
    for n in (3, 64, 512):
        refs = (
            Ref("sklearn", lambda d: (lambda f: lambda: f(d["a"], d["b"]))(_sk("pairwise_distances_argmin_min")),
                threads="follow", differs="the expanded form (cancellation; docs/design/spatial.md §3.2)"),
        )  # fmt: skip
        out.append(Spec("SPT2", "neighbors", "kneighbors k=1", ("kneighbors",), f"(4000, {n})", _normal(4000, 4000, n),
                        lambda d: lambda: _sp().kneighbors(d["sa"], d["sb"]), refs, quick=n == 64))  # fmt: skip
        out.append(Spec("SPT2", "neighbors", "kneighbors k=10", ("kneighbors",), f"(4000, {n})", _normal(4000, 4000, n),
                        lambda d: lambda: _sp().kneighbors(d["sa"], d["sb"], k=10),
                        (Ref("sklearn brute", _nn_brute(10), threads="follow"),)))  # fmt: skip
    for n in (3, 64):
        out.append(Spec("SPT2", "neighbors", "radius_neighbors", ("radius_neighbors",), f"(4000, {n})",
                        _with_radius(_normal(4000, 4000, n)), lambda d: lambda: _sp().radius_neighbors(d["sa"], d["sb"], d["r"]),
                        (Ref("sklearn brute", _nn_radius(), threads="follow"),),
                        note="r: about 10 neighbours per row", check=RADIUS_CHECK))  # fmt: skip
    out.append(Spec("SPT2", "neighbors", "kneighbors k=5 20 rows", ("kneighbors",), "(1000000, 3)",
                    _normal(20, 10**6, 3), lambda d: lambda: _sp().kneighbors(d["sa"], d["sb"], k=5),
                    (Ref("sklearn brute", _nn_brute(5), threads="follow"),), note="20 query rows; XB cut into segments"))  # fmt: skip
    out.append(Spec("SPT2", "neighbors", "kneighbors k=5 cosine", ("kneighbors",), "(2000, 64)",
                    _normal(2000, 2000, 64), lambda d: lambda: _sp().kneighbors(d["sa"], d["sb"], k=5, metric="cosine"),
                    (Ref("sklearn brute", _nn_brute(5, "cosine"), threads="follow",
                         differs="1 - cos (cancellation)"),), note="direct path"))  # fmt: skip
    out.append(Spec("SPT6", "neighbors", "kneighbors k=5 haversine", ("kneighbors",), "(100000, 2)", _geo(10**4, 10**5),
                    lambda d: lambda: _sp().kneighbors(d["sa"], d["sb"], k=5, metric="haversine"),
                    (Ref("sklearn BallTree", _ball_tree(5)),),
                    note="10^4 query rows; the KD tree on the unit sphere (SPT83); BallTree built in the call"))  # fmt: skip
    out.append(Spec("SPT6", "neighbors", "radius_neighbors haversine", ("radius_neighbors",), "(100000, 2)",
                    _geo(10**4, 10**5), lambda d: lambda: _sp().radius_neighbors(d["sa"], d["sb"], d["r"], metric="haversine"),
                    (Ref("sklearn BallTree", _ball_tree(None)),),
                    note="10^4 query rows, r = 0.01 (about 2.5 neighbours per row); BallTree built in the call",
                    check=RADIUS_CHECK))  # fmt: skip
    return out


def _ball_tree(k):
    """scikit-learn's BallTree with metric='haversine', built in the timed call as sd builds its tree: the k
    nearest, or (k None) the points within d["r"] as (distances, indices)."""

    def make(d):
        from sklearn.neighbors import BallTree

        if k is not None:
            return lambda: BallTree(d["b"], metric="haversine").query(d["a"], k=k)
        return lambda: BallTree(d["b"], metric="haversine").query_radius(d["a"], d["r"], return_distance=True)[::-1]

    return make


def _with_radius(base, per_row=10):
    """The setup with r: the distance of about per_row neighbours, from the first rows' sorted distances."""

    def setup():
        d = base()
        a, b = d["a"][:50], d["b"]
        dist = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        d["r"] = float(np.median(np.sort(dist, axis=1)[:, per_row - 1]))
        return d

    return setup


def _radius_sd(r):
    """RadiusResult as (indices, distances) row after row (the rows' indices increasing)."""
    return np.concatenate([np.asarray(r.indices, dtype=np.float64), np.asarray(r.distances)])


def _radius_ref(label, v):
    """sklearn's (distances, indices) object arrays, each row sorted by index as sd's."""
    dist, ind = v
    order = [np.argsort(i, kind="stable") for i in ind]
    idx = np.concatenate([i[o] for i, o in zip(ind, order)]).astype(np.float64)
    return np.concatenate([idx, np.concatenate([d[o] for d, o in zip(dist, order)])])


RADIUS_CHECK = Check(sd=_radius_sd, ref=_radius_ref)


def _nn_radius():
    def make(d):
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(radius=d["r"], algorithm="brute").fit(d["b"])
        return lambda: nn.radius_neighbors(d["a"])

    return make


def _nn_brute(k, metric="euclidean"):
    def make(d):
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric).fit(d["b"])
        return lambda: nn.kneighbors(d["a"])

    return make


def _tree_range(t):
    """A built tree as what the smoke check compares: (n, m) and the data's range per axis."""
    if hasattr(t, "get_arrays"):  # scikit-learn's KDTree: its data
        x = np.asarray(t.get_arrays()[0])
        return np.concatenate([x.shape, x.min(axis=0), x.max(axis=0)]).astype(float)
    return np.concatenate([[t.n, t.m], np.asarray(t.mins), np.asarray(t.maxes)]).astype(float)


BUILD_CHECK = Check(sd=_tree_range, ref=lambda label, t: _tree_range(t), rtol=0.0, atol=0.0)


def _tree_specs():
    out = []
    for npts in (10**5, 10**6):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT3", "tree", "KDTree build m=3", ("KDTree",), f"10^{exp} points", _uniform(npts, 3, 1),
                        lambda d: lambda: _sp().KDTree(d["sx"]),
                        (Ref("scipy", lambda d: (lambda S: lambda: S.KDTree(d["x"]))(_scipy_s())),
                         Ref("sklearn", lambda d: (lambda K: lambda: K(d["x"]))(_sk_tree()))),
                        check=BUILD_CHECK))  # fmt: skip
    for m, npts in ((3, 10**6), (16, 10**5)):
        for k in (1, 10):
            refs = (
                Ref("scipy", lambda d, k=k: lambda: d["T"].query(d["q"], k=k)),
                Ref("scipy workers=-1", lambda d, k=k: lambda: d["T"].query(d["q"], k=k, workers=-1), lib="scipy"),
                Ref("sklearn", lambda d, k=k: lambda: d["ST"].query(d["q"], k=k)),
            )
            out.append(Spec("SPT3", "tree", f"query m={m} k={k}", ("KDTree",), "10^4 points", _tree_setup(npts, m, 10**4),
                            lambda d, k=k: lambda: d["sT"].query(d["sq"], k=k), refs, quick=(m, k) == (3, 10)))  # fmt: skip
    out.append(Spec("SPT4", "tree", "query_ball_point m=3", ("KDTree.query_ball_point",), "10^4 points",
                    _tree_setup(10**6, 3, 10**4),
                    lambda d: lambda: d["sT"].query_ball_point(d["sq"], BALL_R),
                    (Ref("scipy", lambda d: lambda: d["T"].query_ball_point(d["q"], BALL_R)),
                     Ref("scipy workers=-1", lambda d: lambda: d["T"].query_ball_point(d["q"], BALL_R, workers=-1),
                         lib="scipy")), note="mean about 10 points per query", check=ROWS_CHECK))  # fmt: skip
    out.append(Spec("SPT4", "tree", "query_ball_point return_length m=3", ("KDTree.query_ball_point",), "10^4 points",
                    _tree_setup(10**6, 3, 10**4),
                    lambda d: lambda: d["sT"].query_ball_point(d["sq"], BALL_R, return_length=True),
                    (Ref("scipy", lambda d: lambda: d["T"].query_ball_point(d["q"], BALL_R, return_length=True)),
                     Ref("scipy workers=-1", lambda d: lambda: d["T"].query_ball_point(d["q"], BALL_R, workers=-1,
                                                                                       return_length=True),
                         lib="scipy")), note="the counts only"))  # fmt: skip
    pairs = _pair_setup(10**5, 3)
    out.append(Spec("SPT4", "tree", "query_ball_tree m=3", ("KDTree.query_ball_tree",), "10^5 points", pairs,
                    lambda d: lambda: d["sT"].query_ball_tree(d["sU"], PAIR_R),
                    (Ref("scipy", lambda d: lambda: d["T"].query_ball_tree(d["U"], PAIR_R)),),
                    note="two trees of 10^5 points, about 10 neighbours each", check=ROWS_CHECK))  # fmt: skip
    out.append(Spec("SPT4", "tree", "query_pairs m=3", ("KDTree.query_pairs",), "10^5 points", pairs,
                    lambda d: lambda: d["sT"].query_pairs(PAIR_R),
                    (Ref("scipy ndarray", lambda d: lambda: d["T"].query_pairs(PAIR_R, output_type="ndarray")),
                     Ref("scipy set", lambda d: lambda: d["T"].query_pairs(PAIR_R))),
                    note="about 5 pairs per point; scipy's default is a set", check=PAIRS_CHECK))  # fmt: skip
    out.append(Spec("SPT4", "tree", "count_neighbors m=3", ("KDTree.count_neighbors",), "10^5 points", pairs,
                    lambda d: lambda: d["sT"].count_neighbors(d["sU"], PAIR_R),
                    (Ref("scipy", lambda d: lambda: d["T"].count_neighbors(d["U"], PAIR_R)),), quick=True,
                    check=Check(rtol=0.0, atol=0.0)))  # fmt: skip
    out.append(Spec("SPT4", "tree", "count_neighbors 20 radii m=3", ("KDTree.count_neighbors",), "10^5 points", pairs,
                    lambda d: lambda: d["sT"].count_neighbors(d["sU"], COUNT_R),
                    (Ref("scipy", lambda d: lambda: d["T"].count_neighbors(d["U"], COUNT_R)),),
                    check=Check(rtol=0.0, atol=0.0)))  # fmt: skip
    out.append(Spec("SPT4", "tree", "count_neighbors 20 radii weighted m=3", ("KDTree.count_neighbors",),
                    "10^5 points", pairs,
                    lambda d: lambda: d["sT"].count_neighbors(d["sU"], COUNT_R, weights=(d["sw"], d["sv"])),
                    (Ref("scipy", lambda d: lambda: d["T"].count_neighbors(d["U"], COUNT_R,
                                                                           weights=(d["w"], d["v"]))),),
                    note="weights in [0, 1)", check=Check(rtol=1e-12)))  # fmt: skip
    out.append(Spec("SPT4", "tree", "sparse_distance_matrix m=3", ("KDTree.sparse_distance_matrix",), "10^5 points",
                    pairs, lambda d: lambda: d["sT"].sparse_distance_matrix(d["sU"], PAIR_R),
                    (Ref("scipy coo_array", lambda d: lambda: d["T"].sparse_distance_matrix(d["U"], PAIR_R,
                                                                                           output_type="coo_array")),
                     Ref("scipy dok_matrix", lambda d: lambda: d["T"].sparse_distance_matrix(d["U"], PAIR_R))),
                    note="sd gives a csr_array; scipy's default is a dok_matrix", check=SPARSE_CHECK))  # fmt: skip
    return out


def _pair_setup(npts, m, seed=20261126):
    """Two sets of uniform points with their trees built in the setup, and weights."""

    def setup():
        import scipy.spatial as S

        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, npts, m]))
        x, y, w, v = g.random((npts, m)), g.random((npts, m)), g.random(npts), g.random(npts)
        return {"x": x, "y": y, "w": w, "v": v, "T": S.KDTree(x), "U": S.KDTree(y), "sw": sd.asarray(w),
                "sv": sd.asarray(v), "sT": sd.spatial.KDTree(sd.asarray(x)),
                "sU": sd.spatial.KDTree(sd.asarray(y))}  # fmt: skip

    return setup


def _rows_sd(r):
    """RadiusResult as its indices row after row (increasing within a row)."""
    return np.asarray(r.indices, dtype=np.float64)


def _rows_ref(label, v):
    """scipy's lists (a list or an object array of lists) as sd's rows: each row sorted."""
    return np.concatenate([np.sort(np.asarray(row, dtype=np.float64)) for row in v] or [np.zeros(0)])


def _pairs_ref(label, v):
    a = np.array(sorted(v), dtype=np.int64).reshape(-1, 2) if isinstance(v, set) else np.asarray(v)
    return a[np.lexsort((a[:, 1], a[:, 0]))].astype(np.float64).ravel()


def _sparse(m):
    """A sparse matrix (sd's or scipy's) as its CSR indices and data, rows sorted."""
    if hasattr(m, "tocsr") and type(m).__module__.startswith("scipy"):
        c = m.tocsr()
        c.sort_indices()
        return np.concatenate([c.indices.astype(np.float64), c.data])
    return np.concatenate([np.asarray(m.indices, dtype=np.float64), np.asarray(m.data)])


ROWS_CHECK = Check(sd=_rows_sd, ref=_rows_ref, rtol=0.0, atol=0.0)
PAIRS_CHECK = Check(sd=lambda a: np.asarray(a, dtype=np.float64).ravel(), ref=_pairs_ref, rtol=0.0, atol=0.0)
SPARSE_CHECK = Check(sd=_sparse, ref=lambda label, m: _sparse(m), rtol=1e-14)


def _scipy_s():
    import scipy.spatial as S

    return S


def _sk_tree():
    from sklearn.neighbors import KDTree

    return KDTree


# ---- Delaunay (SPT9: d = 2; docs/design/spatial_geometry.md §5) ----


def _plane(kind, npts, seed=20261127):
    """2-D points: 'uniform' in [0, 1)^2, 'normal', 'grid' (an integer grid of side npts, shuffled: every cell
    cocircular), 'circle' (computed points of the unit circle)."""

    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, npts, len(kind)]))
        if kind == "uniform":
            x = g.random((npts, 2))
        elif kind == "normal":
            x = g.standard_normal((npts, 2))
        elif kind == "grid":
            i, j = np.divmod(np.arange(npts * npts), npts)
            x = np.column_stack([i, j]).astype(np.float64)[g.permutation(npts * npts)]
        else:
            a = 2 * np.pi * np.arange(npts) / npts
            x = np.column_stack([np.cos(a), np.sin(a)])
        q = g.random((len(x), 2)) * (x.max(0) - x.min(0)) + x.min(0)
        return {"x": x, "sx": sd.asarray(x), "q": q, "sq": sd.asarray(q)}

    return setup


def _built(kind, npts):
    base = _plane(kind, npts)

    def setup():
        import scipy.spatial as S

        import stardust as sd

        d = base()
        d["T"], d["sT"] = S.Delaunay(d["x"]), sd.spatial.Delaunay(d["sx"])
        return d

    return setup


def _simplex_set(s):
    """The simplices as sorted rows in lexicographic order (sd's canonical rows or scipy's)."""
    a = np.sort(np.asarray(s.simplices, dtype=np.int64), axis=1)
    return a[np.lexsort(a.T[::-1])].astype(np.float64).ravel()


SIMPLICES_CHECK = Check(sd=_simplex_set, ref=lambda label, s: _simplex_set(s), rtol=0.0, atol=0.0)
NSIMPLEX_CHECK = Check(sd=lambda s: float(s.nsimplex), ref=lambda label, s: float(s.nsimplex), rtol=0.0, atol=0.0)


def _vnv(v):
    """vertex_neighbor_vertices as its rows, each sorted (scipy's rows have no order)."""
    indptr, indices = (np.asarray(a) for a in v)
    return np.concatenate([np.sort(indices[indptr[i] : indptr[i + 1]]) for i in range(len(indptr) - 1)]).astype(float)


def _scipy_vnv(d):
    T = d["T"]

    def run():
        T._vertex_neighbor_vertices = None  # scipy keeps it after the first use
        return T.vertex_neighbor_vertices

    return run


def _geometry_specs():
    out = []
    for npts in (10**3, 10**4, 10**5, 10**6):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT9", "delaunay", "Delaunay 2-D uniform", ("Delaunay",), f"10^{exp} points",
                        _plane("uniform", npts), lambda d: lambda: _sp().Delaunay(d["sx"]),
                        (Ref("scipy", lambda d: (lambda S: lambda: S.Delaunay(d["x"]))(_scipy_s())),),
                        quick=npts == 10**5, note="scipy (Qhull) on one thread" if npts == 10**6 else "",
                        check=SIMPLICES_CHECK))  # fmt: skip
    for kind, npts, size, note, check in (
        ("normal", 10**5, "10^5 points", "", SIMPLICES_CHECK),
        ("grid", 316, "99856 points", "every cell cocircular: the tie rule decides", NSIMPLEX_CHECK),
        ("circle", 10**4, "10^4 points", "points of the unit circle", NSIMPLEX_CHECK),
    ):
        out.append(Spec("SPT9", "delaunay", f"Delaunay 2-D {kind}", ("Delaunay",), size, _plane(kind, npts),
                        lambda d: lambda: _sp().Delaunay(d["sx"]),
                        (Ref("scipy", lambda d: (lambda S: lambda: S.Delaunay(d["x"]))(_scipy_s())),), note=note,
                        check=check))  # fmt: skip
    for npts in (10**3, 10**4, 10**5):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT9", "delaunay", "Delaunay.find_simplex 2-D", ("Delaunay.find_simplex",),
                        f"10^{exp} points", _built("uniform", npts), lambda d: lambda: d["sT"].find_simplex(d["sq"]),
                        (Ref("scipy", lambda d: lambda: d["T"].find_simplex(d["q"])),),
                        note="as many uniform query points as points",
                        check=Check(skip="the simplex numbers differ (sd's canonical order)")))  # fmt: skip
    for npts in (10**4, 10**5, 10**6):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT9", "delaunay", "Delaunay.vertex_neighbor_vertices 2-D",
                        ("Delaunay.vertex_neighbor_vertices",), f"10^{exp} points", _built("uniform", npts),
                        lambda d: lambda: d["sT"]._tri.vertex_neighbor_vertices(), (Ref("scipy", _scipy_vnv),),
                        note="both are made on first use and kept: timed uncached",
                        check=Check(sd=_vnv, ref=lambda label, v: _vnv(v), rtol=0.0, atol=0.0)))  # fmt: skip
    return out


# ---- ConvexHull (SPT11; docs/design/spatial_geometry.md §5) ----


def _hull_points(d, kind, npts, seed=20261128):
    """d-D points: 'uniform' in [0, 1)^d, or 'sphere' (every point extreme: the unit circle's computed points
    for d = 2, normalized normal vectors for d = 3)."""

    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, d, npts, len(kind)]))
        if kind == "uniform":
            x = g.random((npts, d))
        elif d == 2:
            a = 2 * np.pi * np.arange(npts) / npts
            x = np.column_stack([np.cos(a), np.sin(a)])
        else:
            x = g.standard_normal((npts, 3))
            x /= np.linalg.norm(x, axis=1)[:, None]
        return {"x": x, "sx": sd.asarray(x)}

    return setup


def _hull_measure(h):
    """A hull as what the smoke check compares: the extreme points (sorted), area and volume."""
    return np.concatenate([np.sort(np.asarray(h.vertices, dtype=np.float64)), [h.area, h.volume]])


HULL_CHECK = Check(sd=_hull_measure, ref=lambda label, h: _hull_measure(h), rtol=1e-12)


def _hull_specs():
    out = []
    for d, kind, sizes in ((2, "uniform", (10**4, 10**5, 10**6)), (3, "uniform", (10**4, 10**5, 10**6)),
                           (2, "sphere", (10**5,)), (3, "sphere", (10**5,))):  # fmt: skip
        for npts in sizes:
            exp = int(round(np.log10(npts)))
            name = f"ConvexHull {d}-D {'uniform' if kind == 'uniform' else 'circle' if d == 2 else 'sphere'}"
            out.append(Spec("SPT11", "hull", name, ("ConvexHull",), f"10^{exp} points", _hull_points(d, kind, npts),
                            lambda d: lambda: _sp().ConvexHull(d["sx"]),
                            (Ref("scipy", lambda d: (lambda S: lambda: S.ConvexHull(d["x"]))(_scipy_s())),),
                            quick=(kind, npts) == ("uniform", 10**5),
                            note="every point extreme" if kind == "sphere" else "", check=HULL_CHECK))  # fmt: skip
    return out


# ---- Delaunay for d = 3 (SPT10; docs/design/spatial_geometry.md §5) ----


def _space(kind, npts, seed=20261129):
    """3-D points: 'uniform' in [0, 1)^3 or 'grid' (an integer grid of side npts, shuffled: every cell
    cospherical); query points uniform in the box, as many as points."""

    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, npts, len(kind)]))
        if kind == "uniform":
            x = g.random((npts, 3))
        else:
            i, r = np.divmod(np.arange(npts**3), npts * npts)
            j, k = np.divmod(r, npts)
            x = np.column_stack([i, j, k]).astype(np.float64)[g.permutation(npts**3)]
        q = g.random((len(x), 3)) * (x.max(0) - x.min(0)) + x.min(0)
        return {"x": x, "sx": sd.asarray(x), "q": q, "sq": sd.asarray(q)}

    return setup


def _built3(npts):
    base = _space("uniform", npts)

    def setup():
        import scipy.spatial as S

        import stardust as sd

        d = base()
        d["T"], d["sT"] = S.Delaunay(d["x"]), sd.spatial.Delaunay(d["sx"])
        return d

    return setup


def _delaunay3_specs():
    out = []
    delaunay = Ref("scipy", lambda d: (lambda S: lambda: S.Delaunay(d["x"]))(_scipy_s()))
    for npts in (10**3, 10**4, 10**5):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT10", "delaunay", "Delaunay 3-D uniform", ("Delaunay",), f"10^{exp} points",
                        _space("uniform", npts), lambda d: lambda: _sp().Delaunay(d["sx"]), (delaunay,),
                        quick=npts == 10**4, note="sd inserts on one thread (SPT107)" if npts == 10**5 else "",
                        check=SIMPLICES_CHECK))  # fmt: skip
    out.append(Spec("SPT10", "delaunay", "Delaunay 3-D grid", ("Delaunay",), "32768 points", _space("grid", 32),
                    lambda d: lambda: _sp().Delaunay(d["sx"]), (delaunay,),
                    note="every cell cospherical: the tie rule decides",
                    check=Check(skip="the number of tetrahedra of cospherical points is not fixed (a cube: 5 or 6)")))  # fmt: skip
    for npts in (10**3, 10**4, 10**5):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT10", "delaunay", "Delaunay.find_simplex 3-D", ("Delaunay.find_simplex",),
                        f"10^{exp} points", _built3(npts), lambda d: lambda: d["sT"].find_simplex(d["sq"]),
                        (Ref("scipy", lambda d: lambda: d["T"].find_simplex(d["q"])),),
                        note="as many uniform query points as points",
                        check=Check(skip="the simplex numbers differ (sd's canonical order)")))  # fmt: skip
    for npts in (10**4, 10**5):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT10", "delaunay", "Delaunay.vertex_neighbor_vertices 3-D",
                        ("Delaunay.vertex_neighbor_vertices",), f"10^{exp} points", _built3(npts),
                        lambda d: lambda: d["sT"]._tri.vertex_neighbor_vertices(), (Ref("scipy", _scipy_vnv),),
                        note="both are made on first use and kept: timed uncached",
                        check=Check(sd=_vnv, ref=lambda label, v: _vnv(v), rtol=0.0, atol=0.0)))  # fmt: skip
    return out


# ---- Voronoi (SPT12; docs/design/spatial_geometry.md §5) ----


def _voronoi_vertices(v):
    """A diagram as what the smoke check compares: its vertices in lexicographic order (scipy's numbering is
    Qhull's; the 3-D ridges differ by the unbounded ones scipy leaves out)."""
    a = np.asarray(v.vertices, dtype=np.float64)
    return a[np.lexsort(a.T[::-1])].ravel()


VORONOI_CHECK = Check(sd=_voronoi_vertices, ref=lambda label, v: _voronoi_vertices(v), rtol=1e-9, atol=1e-9)


def _voronoi_specs():
    out = []
    voronoi = Ref("scipy", lambda d: (lambda S: lambda: S.Voronoi(d["x"]))(_scipy_s()))
    for npts in (10**3, 10**4, 10**5):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT12", "voronoi", "Voronoi 2-D uniform", ("Voronoi",), f"10^{exp} points",
                        _plane("uniform", npts), lambda d: lambda: _sp().Voronoi(d["sx"]), (voronoi,),
                        quick=npts == 10**5, check=VORONOI_CHECK))  # fmt: skip
    out.append(Spec("SPT12", "voronoi", "Voronoi 2-D grid", ("Voronoi",), "99856 points", _plane("grid", 316),
                    lambda d: lambda: _sp().Voronoi(d["sx"]), (voronoi,),
                    note="every cell cocircular: one vertex per cell", check=VORONOI_CHECK))  # fmt: skip
    for npts in (10**3, 10**4, 10**5):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT12", "voronoi", "Voronoi 3-D uniform", ("Voronoi",), f"10^{exp} points",
                        _space("uniform", npts), lambda d: lambda: _sp().Voronoi(d["sx"]), (voronoi,),
                        quick=npts == 10**4, note="sd inserts on one thread (SPT107)" if npts == 10**5 else "",
                        check=VORONOI_CHECK))  # fmt: skip
    out.append(Spec("SPT12", "voronoi", "Voronoi 3-D grid", ("Voronoi",), "32768 points", _space("grid", 32),
                    lambda d: lambda: _sp().Voronoi(d["sx"]), (voronoi,),
                    note="every cell cospherical: one vertex per cell", check=VORONOI_CHECK))  # fmt: skip
    return out


def _large_specs():
    """SPT13: the 10^6 rows of §5 that scipy is not run on (its memory passes 2 GB or it takes minutes)."""
    return [
        Spec("SPT13", "delaunay", "Delaunay 3-D uniform", ("Delaunay",), "10^6 points", _space("uniform", 10**6),
             lambda d: lambda: _sp().Delaunay(d["sx"]), note="scipy not run (about 4.6 GB at 10^6)"),
        Spec("SPT13", "delaunay", "Delaunay.find_simplex 2-D", ("Delaunay.find_simplex",), "10^6 points",
             _built_sd("uniform", 10**6), lambda d: lambda: d["sT"].find_simplex(d["sq"]),
             note="scipy not run (42 s); as many uniform query points as points"),
        Spec("SPT13", "voronoi", "Voronoi 2-D uniform", ("Voronoi",), "10^6 points", _plane("uniform", 10**6),
             lambda d: lambda: _sp().Voronoi(d["sx"]), note="scipy not run (its memory passes 2 GB)"),
    ]  # fmt: skip


def _built_sd(kind, npts):
    base = _plane(kind, npts)

    def setup():
        import stardust as sd

        d = base()
        d["sT"] = sd.spatial.Delaunay(d["sx"])
        return d

    return setup


SPECS = (_distance_specs() + _neighbor_specs() + _tree_specs() + _geometry_specs() + _hull_specs()
         + _delaunay3_specs() + _voronoi_specs() + _large_specs())  # fmt: skip


def _public(names) -> bool:
    import stardust as sd

    def public(name):  # "KDTree.query_ball_point": a method of a public class (SPT4 adds methods to SPT3's class)
        cls, _, attr = name.partition(".")
        return cls in sd.spatial.__all__ and (not attr or hasattr(getattr(sd.spatial, cls), attr))

    return all(public(n) for n in names)


def cases() -> list[Case]:
    return [
        Case(area="spatial", group=s.group, name=s.name, size=s.size, setup=s.setup, sd=s.sd, refs=s.refs,
             quick=s.quick, note=s.note, check=s.check)
        for s in SPECS
        if _public(s.needs)
    ]  # fmt: skip


TITLES = {
    "distance": "거리 (`pdist`, `cdist`)",
    "neighbors": "한 번에 끝나는 이웃 찾기",
    "tree": "KD 트리",
    "delaunay": "삼각분할 (`Delaunay`)",
    "hull": "볼록 껍질 (`ConvexHull`)",
    "voronoi": "보로노이 (`Voronoi`)",
}
