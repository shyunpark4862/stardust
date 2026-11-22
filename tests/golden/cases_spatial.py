"""Golden cases for sd.spatial (tools/golden.py has the contract).

The distances (milestone SPT1, docs/design/spatial.md §8): pdist, cdist and every distance function for
every metric, weighted and not, on small sets with the edge values (NaN, inf, zeros, constant rows, huge
and tiny coordinates that take the careful paths) and on sets large enough that pdist and cdist split into
many tasks at 3 and 10 threads (the tile kernel's row groups and panel chunks, the scalar drivers of
minkowski, mahalanobis and haversine, the bit-packed bool rows); the default V and VI; the batch shapes of
the distance functions; hamming and jaccard on bool, int64 and float64; squareform both ways with every
dtype; the wrong calls (their errors are results). kneighbors and radius_neighbors (milestone SPT2): every
metric, the GEMM pre-selection's sizes, a far cluster, ranks, the upper bound, RadiusResult and its tocsr().
KDTree and query (milestone SPT3): the build's attributes and node counts, every p, k as an int and as
ranks, the upper bound, duplicates and ties, n = 0 and m = 0, a set large enough for the parallel build
and query, the wrong calls. The radius, pair and count queries (milestone SPT4): query_ball_point (one
radius, per-point radii, return_length), query_ball_tree, query_pairs, count_neighbors (one and many radii,
binned, weighted in every form) and sparse_distance_matrix for every p, boundary radii on a lattice,
duplicates, huge and tiny coordinates, empty trees, sets large enough for many tasks at 3 and 10 threads
(the weighted sums' task order), the wrong calls.
The inputs come from fixed literals and L.rng(seed); the largest result is 1.8 * 10^5 doubles.
"""

import math

import numpy as np

NAN, INF = math.nan, math.inf
METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "seuclidean", "mahalanobis", "cosine",
           "correlation", "canberra", "braycurtis", "hamming", "jaccard")  # fmt: skip
WEIGHTLESS = ("seuclidean", "mahalanobis")


def encode(x, L, enc):
    """RadiusResult (SPT2) as its fields and its tocsr(); a csr_array as its CSR arrays (sd.sparse's own
    golden form)."""
    S = L.sd.sparse
    if isinstance(x, S.sparray):
        return {"sparse": type(x).__name__, "shape": list(x.shape), "dtype": str(x.dtype), "nnz": x.nnz,
                "indptr": enc(x.indptr), "indices": enc(x.indices), "data": enc(x.data)}  # fmt: skip
    if type(x).__name__ == "RadiusResult":
        return {"RadiusResult": list(x.shape), "repr": repr(x), "indptr": enc(x.indptr), "indices": enc(x.indices),
                "distances": enc(x.distances), "tocsr": encode(x.tocsr(), L, enc)}  # fmt: skip
    return None


def _data(L):
    r = L.rng(701)
    edge = np.array([
        [1.0, -2.0, 3.0, 0.5],
        [0.5, 2.0, -1.0, 0.5],
        [NAN, 1.0, 0.0, 2.0],
        [INF, 1.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 3.0],
        [0.0, 1.0, 0.0, -2.0],
        [1e200, -1e200, 3e199, 0.0],
        [1e-200, 3e-200, -2e-200, 0.0],
        [5e-324, 0.0, 1e-320, 0.0],
    ])  # fmt: skip
    small = np.round(r.standard_normal((9, 5)) * 4) / 4
    big = r.standard_normal((600, 40))
    big[:, ::4] = np.round(big[:, ::4])
    other = r.standard_normal((300, 40))
    return {
        "edge": edge,
        "small": small,
        "big": big,
        "other": other,
        "w4": np.array([0.5, 0.0, 2.0, 1.0]),
        "w5": r.uniform(0.0, 2.0, 5),
        "w40": r.uniform(0.0, 2.0, 40),
        "V5": r.uniform(0.5, 2.0, 5),
        "bool": r.random((400, 150)) < 0.3,
        "int": r.integers(-2, 3, (50, 20)),
        "int60": np.array([[2**60, 1, 0], [2**60 + 1, 1, 0], [0, 0, 0]], dtype=np.int64),
        "hav": np.column_stack([r.uniform(-1.5, 1.5, 500), r.uniform(-3.1, 3.1, 500)]),
        "antipode": np.array([[0.4, -1.0], [-0.4, -1.0 + math.pi], [-0.4 + 1e-9, -1.0 + math.pi - 1e-9]]),
    }


def _kw(metric, n):
    if metric == "minkowski":
        return {"p": 3.0}
    if metric == "seuclidean":
        return {"V": np.linspace(0.5, 2.0, n)}
    if metric == "mahalanobis":
        a = np.arange(n * n, dtype=float).reshape(n, n) % 7 - 3
        return {"VI": a @ a.T + np.eye(n)}
    return {}


def cases(L):
    S = L.sd.spatial
    d = _data(L)
    a = L.arr

    # ---- pdist and cdist, every metric
    for metric in METRICS:
        for name in ("edge", "small"):
            x = d[name]
            kw = _kw(metric, x.shape[1])
            yield f"pdist {metric} {name}", lambda x=x, metric=metric, kw=kw: S.pdist(a(x), metric=metric, **kw)
            yield (f"cdist {metric} {name}",
                   lambda x=x, metric=metric, kw=kw: S.cdist(a(x), a(x[::-1][:7]), metric=metric, **kw))  # fmt: skip
            if metric not in WEIGHTLESS:
                w = d["w4"] if name == "edge" else d["w5"]
                yield (f"pdist {metric} {name} w",
                       lambda x=x, metric=metric, kw=kw, w=w: S.pdist(a(x), metric=metric, w=a(w), **kw))  # fmt: skip
        kw = _kw(metric, 40)
        yield f"pdist {metric} big", lambda metric=metric, kw=kw: S.pdist(a(d["big"]), metric=metric, **kw)
        yield (f"cdist {metric} big",
               lambda metric=metric, kw=kw: S.cdist(a(d["big"][:200]), a(d["other"]), metric=metric, **kw))  # fmt: skip
        if metric not in WEIGHTLESS:
            yield (f"cdist {metric} big w",
                   lambda metric=metric, kw=kw: S.cdist(a(d["big"][:200]), a(d["other"]), metric=metric,
                                                        w=a(d["w40"]), **kw))  # fmt: skip
    for p in (0.5, 1, 1.5, 2, 7.5, INF):
        yield f"pdist minkowski p={p}", lambda p=p: S.pdist(a(d["edge"]), metric="minkowski", p=p)
        yield f"minkowski p={p}", lambda p=p: S.minkowski(a(d["small"][:4]), a(d["small"][4:8]), p=p)
    yield "pdist minkowski p=2000", lambda: S.pdist(a(np.array([[0.0, 0.0], [3.0, 4.0]])), metric="minkowski", p=2000)
    yield "pdist seuclidean default V", lambda: S.pdist(a(d["small"]), metric="seuclidean")
    yield "cdist seuclidean default V", lambda: S.cdist(a(d["small"]), a(d["small"][:3] + 1), metric="seuclidean")
    yield "pdist mahalanobis default VI", lambda: S.pdist(a(d["big"][:60, :5]), metric="mahalanobis")
    yield "pdist mahalanobis default VI NaN", lambda: S.pdist(a(d["edge"][:, :2]), metric="mahalanobis")
    yield "pdist mahalanobis indefinite", lambda: S.pdist(a(d["small"][:, :2]), metric="mahalanobis",
                                                          VI=a(np.array([[1.0, 0.0], [0.0, -1.0]])))  # fmt: skip
    yield "pdist no columns", lambda: S.pdist(a(np.zeros((3, 0))), metric="cosine")
    yield "pdist one row", lambda: S.pdist(a(np.zeros((1, 3))))
    yield "cdist empty side", lambda: S.cdist(a(np.zeros((0, 3))), a(np.zeros((2, 3))))

    # ---- haversine
    yield "pdist haversine", lambda: S.pdist(a(d["hav"]), metric="haversine")
    yield "cdist haversine antipode", lambda: S.cdist(a(d["antipode"]), a(d["antipode"]), metric="haversine")
    yield "haversine batch", lambda: S.haversine(a(d["hav"][:10]), a(d["hav"][10:20]))

    # ---- hamming and jaccard on their dtypes (SPT23)
    for metric in ("hamming", "jaccard"):
        yield f"pdist {metric} bool", lambda metric=metric: S.pdist(a(d["bool"]), metric=metric)
        yield (f"cdist {metric} bool w",
               lambda metric=metric: S.cdist(a(d["bool"][:40]), a(d["bool"][40:80]), metric=metric,
                                             w=a(np.linspace(0.0, 1.0, 150))))  # fmt: skip
        yield f"pdist {metric} int64", lambda metric=metric: S.pdist(a(d["int"]), metric=metric)
        yield f"pdist {metric} int64 2^60", lambda metric=metric: S.pdist(a(d["int60"]), metric=metric)
        yield f"{metric} bool int64", lambda metric=metric: getattr(S, metric)(
            a(np.array([True, False, True])), a(np.array([1, 0, 2])))  # fmt: skip

    # ---- the distance functions
    for metric in METRICS:
        fn = getattr(S, metric)
        x, y = d["small"][:4], d["small"][4:8]
        if metric == "seuclidean":
            yield "seuclidean", lambda x=x, y=y: S.seuclidean(a(x), a(y), a(d["V5"]))
            yield "seuclidean scalar", lambda: S.seuclidean([1.0, 2.0], [0.0, 0.0], [2.0, 0.5])
            continue
        if metric == "mahalanobis":
            VI = _kw("mahalanobis", 5)["VI"]
            yield "mahalanobis", lambda x=x, y=y, VI=VI: S.mahalanobis(a(x), a(y), a(VI))
            continue
        yield f"{metric} batch", lambda fn=fn, x=x, y=y: fn(a(x), a(y))
        yield f"{metric} broadcast", lambda fn=fn, x=x, y=y: fn(a(x.reshape(4, 1, 5)), a(y))
        yield f"{metric} scalar", lambda fn=fn: fn([1.0, 2.0, 0.0], [0.5, -1.0, 0.0])
        yield f"{metric} w", lambda fn=fn, x=x, y=y: fn(a(x), a(y), w=a(d["w5"]))
        yield f"{metric} edge rows", lambda fn=fn: fn(a(d["edge"][:5]), a(d["edge"][5:]))
    yield "correlation centered=False", lambda: S.correlation([1.0, 2.0, 4.0], [2.0, 0.0, 1.0], centered=False)

    # ---- squareform
    for dt in (np.bool_, np.int64, np.float64):
        v = (np.arange(15) % 3).astype(dt)
        yield f"squareform tomatrix {np.dtype(dt).name}", lambda v=v: S.squareform(a(v))
        yield f"squareform round trip {np.dtype(dt).name}", lambda v=v: S.squareform(S.squareform(a(v)))
    yield "squareform pdist", lambda: S.squareform(S.pdist(a(d["big"][:300])))
    yield "squareform empty", lambda: S.squareform(a(np.zeros(0)))
    yield "squareform 1x1", lambda: S.squareform(a(np.zeros((1, 1))))
    yield "squareform checks=False", lambda: S.squareform(a(np.array([[0.0, 1.0], [2.0, 0.0]])), checks=False)
    yield "squareform force", lambda: S.squareform(a(np.array([3.0])), force="tomatrix")

    # ---- wrong calls
    x = a(d["small"])
    yield "error metric alias", lambda: S.pdist(x, metric="e")
    yield "error metric positional", lambda: S.pdist(x, "cityblock")
    yield "error p unused", lambda: S.pdist(x, p=3)
    yield "error p zero", lambda: S.pdist(x, metric="minkowski", p=0)
    yield "error w nan", lambda: S.pdist(x, w=[NAN, 1, 1, 1, 1])
    yield "error w length", lambda: S.pdist(x, w=[1.0])
    yield "error w seuclidean", lambda: S.pdist(x, metric="seuclidean", w=[1.0] * 5)
    yield "error V negative", lambda: S.pdist(x, metric="seuclidean", V=[-1.0] * 5)
    yield "error VI shape", lambda: S.pdist(x, metric="mahalanobis", VI=np.eye(2))
    yield "error VI few rows", lambda: S.pdist(x[:3], metric="mahalanobis")
    yield "error X 1-D", lambda: S.pdist(a(np.zeros(3)))
    yield "error columns", lambda: S.cdist(x, a(np.zeros((2, 3))))
    yield "error complex", lambda: S.pdist(a(np.array([[1j, 0], [0, 1]])))
    yield "error haversine columns", lambda: S.pdist(x, metric="haversine")
    yield "error last axis", lambda: S.euclidean([1.0, 2.0], [1.0])
    yield "error broadcast", lambda: S.cosine(a(np.ones((2, 3))), a(np.ones((4, 3))))
    yield "error squareform asym", lambda: S.squareform(a(np.array([[0.0, 1.0], [2.0, 0.0]])))
    yield "error squareform nan", lambda: S.squareform(a(np.array([[0.0, NAN], [NAN, 0.0]])))
    yield "error squareform length", lambda: S.squareform(a(np.zeros(4)))
    yield "error memory", lambda: S.pdist(a(np.zeros((2**22, 0))))

    yield from _neighbor_cases(L)
    yield from _kdtree_cases(L)
    yield from _kdtree_radius_cases(L)


def _neighbor_cases(L):
    """kneighbors and radius_neighbors (milestone SPT2): every metric on small sets, the GEMM path's sizes
    (n >= 12, XB in several blocks, a far-from-origin cluster, duplicates), k as ranks, the upper bound,
    padding, RadiusResult with its tocsr(), the counts, the wrong calls."""
    S = L.sd.spatial
    a = L.arr
    r = L.rng(702)
    small = np.round(r.standard_normal((30, 5)) * 4) / 4
    q = small[::3] + 0.125
    big, other = r.standard_normal((300, 40)), r.standard_normal((2600, 40))
    other[1000:1100] = other[:100]  # duplicates
    far, farq = 1e8 + r.random((2500, 16)), 1e8 + r.random((150, 16))
    unit = ("cosine", "correlation", "hamming", "jaccard", "braycurtis", "canberra")  # distances of order 1
    for metric in METRICS:
        kw = _kw(metric, 5)
        rad = 0.4 if metric in unit else 2.0
        yield (f"kneighbors {metric}",
               lambda metric=metric, kw=kw: S.kneighbors(a(q), a(small), k=4, metric=metric, **kw))  # fmt: skip
        yield (f"radius_neighbors {metric}",
               lambda metric=metric, kw=kw, rad=rad: S.radius_neighbors(a(q), a(small), rad, metric=metric,
                                                                        **kw))  # fmt: skip
    yield "kneighbors haversine", lambda: S.kneighbors(a(np.array([[0.1, 0.2], [-1.0, 3.0]])),
                                                       a(_data(L)["hav"][:200]), k=3, metric="haversine")  # fmt: skip
    yield "kneighbors w", lambda: S.kneighbors(a(q), a(small), k=3, w=a(np.array([1.0, 0.0, 2.0, 0.5, 1.0])))
    yield "kneighbors k=1", lambda: S.kneighbors(a(q), a(small))
    yield "kneighbors ranks", lambda: S.kneighbors(a(q), a(small), k=[3, 1, 1, 40])
    yield "kneighbors k > mB", lambda: S.kneighbors(a(q), a(small[:4]), k=6)
    yield "kneighbors upper bound", lambda: S.kneighbors(a(q), a(small), k=5, distance_upper_bound=1.0)
    yield "kneighbors empty XB", lambda: S.kneighbors(a(q), a(np.zeros((0, 5))), k=2)
    yield "kneighbors gemm k=1", lambda: S.kneighbors(a(big), a(other))
    yield "kneighbors gemm k=10", lambda: S.kneighbors(a(big), a(other), k=10)
    yield "kneighbors gemm sqeuclidean", lambda: S.kneighbors(a(big[:50]), a(other), k=3, metric="sqeuclidean")
    yield "kneighbors gemm one row", lambda: S.kneighbors(a(big[:1]), a(other), k=7)
    yield "kneighbors far cluster", lambda: S.kneighbors(a(farq), a(far), k=5)
    yield "kneighbors itself", lambda: S.kneighbors(a(other[:300]), a(other[:300]), k=3)
    yield "radius_neighbors gemm", lambda: S.radius_neighbors(a(big[:60]), a(other), 7.0)
    yield "radius_neighbors far cluster", lambda: S.radius_neighbors(a(farq), a(far), 0.35)
    yield "radius_neighbors r array", lambda: S.radius_neighbors(a(q), a(small), a(np.linspace(0.0, 3.0, len(q))))
    yield "radius_neighbors zeros stored", lambda: S.radius_neighbors(a(small), a(small), 0.0).tocsr()
    yield "radius_neighbors return_length", lambda: S.radius_neighbors(a(big), a(other), 7.5, return_length=True)
    x = a(small)
    yield "error kneighbors nan", lambda: S.kneighbors(a(np.array([[NAN, 0, 0, 0, 0]])), x)
    yield "error kneighbors k=0", lambda: S.kneighbors(x, x, k=0)
    yield "error kneighbors k positional", lambda: S.kneighbors(x, x, 3)
    yield "error kneighbors upper bound", lambda: S.kneighbors(x, x, distance_upper_bound=-1.0)
    yield "error radius_neighbors r", lambda: S.radius_neighbors(x, x, -1.0)
    yield "error radius_neighbors r shape", lambda: S.radius_neighbors(x, x, [1.0, 2.0])


def _kdtree_cases(L):
    """KDTree and query (milestone SPT3): the attributes, every p (1, 2, inf, 1.5, 3), k as an int and as
    ranks, the strict upper bound, the fill, duplicates and ties, n = 0 and m = 0, huge and tiny
    coordinates (the careful paths), 4 * 10^4 points and 5000 queries (the parallel build and query at 3
    and 10 threads), the wrong calls."""
    S = L.sd.spatial
    a = L.arr
    r = L.rng(703)
    small = np.round(r.standard_normal((40, 3)) * 2) / 2  # many equal coordinates and ties
    q = np.vstack([small[:5], np.round(r.standard_normal((6, 3)) * 2) / 2 + 0.25])
    dup = np.repeat(r.integers(-2, 3, (8, 2)).astype(float), 5, axis=0)
    big = r.standard_normal((40_000, 3))
    big[::6] = np.round(big[::6], 1)
    bigq = r.standard_normal((5000, 3))

    def attrs(t):
        return (t.n, t.m, t.leafsize, t.size, t.mins, t.maxes, t.data, repr(t))

    yield "KDTree attributes", lambda: attrs(S.KDTree(a(small), leafsize=4))
    yield "KDTree attributes big", lambda: attrs(S.KDTree(a(big)))
    yield "KDTree attributes empty", lambda: attrs(S.KDTree(a(np.zeros((0, 3)))))
    yield "KDTree attributes no columns", lambda: attrs(S.KDTree(a(np.zeros((5, 0)))))
    for p in (1, 2, INF, 1.5, 3):
        yield f"query p={p}", lambda p=p: S.KDTree(a(small), leafsize=3).query(a(q), k=6, p=p)
        yield f"query dup p={p}", lambda p=p: S.KDTree(a(dup)).query(a(dup[::7] + 0.5), k=12, p=p)
        yield f"query big p={p}", lambda p=p: S.KDTree(a(big)).query(a(bigq), k=4 if p == 2 else 2, p=p)
    yield "query one point", lambda: S.KDTree(a(small)).query(a(q[0]))
    yield "query one point k=3", lambda: S.KDTree(a(small)).query(a(q[0]), k=3)
    yield "query ranks", lambda: S.KDTree(a(small)).query(a(q), k=[3, 1, 1, 50])
    yield "query batch", lambda: S.KDTree(a(small)).query(a(q[:6].reshape(2, 3, 3)), k=[2])
    yield "query no points", lambda: S.KDTree(a(small)).query(a(np.zeros((0, 3))), k=2)
    yield "query upper bound", lambda: S.KDTree(a(small)).query(a(q), k=8, distance_upper_bound=1.0)
    yield "query upper bound 0", lambda: S.KDTree(a(small)).query(a(small[:4]), k=3, distance_upper_bound=0.0)
    yield "query k > n", lambda: S.KDTree(a(small[:3])).query(a(q[:2]), k=5)
    yield "query empty tree", lambda: S.KDTree(a(np.zeros((0, 3)))).query(a(q[:2]), k=2)
    yield "query no columns", lambda: S.KDTree(a(np.zeros((5, 0)))).query(a(np.zeros((2, 0))), k=3)
    yield "query huge", lambda: S.KDTree(a(small * 1e200)).query(a(q * 1e200), k=3)
    yield "query tiny", lambda: S.KDTree(a(small * 1e-200)).query(a(q * 1e-200), k=3)
    yield "query leafsize 1", lambda: S.KDTree(a(small), leafsize=1).query(a(q), k=5)

    # ---- wrong calls
    t = S.KDTree(a(small))
    yield "error KDTree leafsize positional", lambda: S.KDTree(a(small), 5)
    yield "error KDTree leafsize float", lambda: S.KDTree(a(small), leafsize=2.5)
    yield "error KDTree leafsize 0", lambda: S.KDTree(a(small), leafsize=0)
    yield "error KDTree 1-D", lambda: S.KDTree(a(np.zeros(3)))
    yield "error KDTree nan", lambda: S.KDTree(a(np.array([[NAN, 0.0]])))
    yield "error query k positional", lambda: t.query(a(q), 3)
    yield "error query k 0", lambda: t.query(a(q), k=0)
    yield "error query k [0]", lambda: t.query(a(q), k=[0])
    yield "error query k bool", lambda: t.query(a(q), k=True)
    yield "error query k float", lambda: t.query(a(q), k=2.0)
    yield "error query p", lambda: t.query(a(q), p=0.5)
    yield "error query bound", lambda: t.query(a(q), distance_upper_bound=-1.0)
    yield "error query columns", lambda: t.query(a(np.zeros((2, 2))))
    yield "error query inf", lambda: t.query(a(np.array([INF, 0.0, 0.0])))


def _kdtree_radius_cases(L):
    """The radius, pair and count queries (milestone SPT4): every p, the boundary (radii equal to lattice
    distances), per-point radii, return_length, duplicates, huge and tiny coordinates, n = 0, 2 * 10^4 and
    1.5 * 10^4 points (many tasks of the dual-tree walk at 3 and 10 threads; the weighted sums are added in
    task order), the weight forms, the wrong calls."""
    S = L.sd.spatial
    a = L.arr
    r = L.rng(705)
    grid = np.array(np.meshgrid(*[np.arange(-3.0, 4.0)] * 2)).reshape(2, -1).T  # lattice distances: ties
    small = np.round(r.standard_normal((40, 3)) * 2) / 2
    other = np.vstack([small[:6], np.round(r.standard_normal((30, 3)) * 2) / 2 + 0.25])
    q = np.vstack([small[:5], np.round(r.standard_normal((6, 3)) * 2) / 2 + 0.25])
    big = r.standard_normal((20_000, 3))
    big[::6] = np.round(big[::6], 1)
    bigo = r.standard_normal((15_000, 3))
    bigo[:1000] = big[:1000]
    bigq = r.standard_normal((3000, 3))
    w = r.standard_normal(40)
    wo = r.random(36)
    wbig, wbigo = r.random(20_000), r.standard_normal(15_000)
    rs = [0.0, 0.5, 1.0, 1.5, 3.0, INF]

    def tree(x, **kw):
        return S.KDTree(a(x), **kw)

    for p in (1, 2, INF, 1.5, 3):
        yield f"query_ball_point p={p}", lambda p=p: tree(small, leafsize=3).query_ball_point(a(q), 1.0, p=p)
        yield (
            f"query_ball_point lattice p={p}",
            lambda p=p: tree(grid).query_ball_point(a(np.array([[0.0, 0.0], [0.5, 0.5], [3.0, -3.0]])), 2.0, p=p),
        )
        yield f"query_ball_tree p={p}", lambda p=p: tree(small).query_ball_tree(tree(other), 1.0, p=p)
        yield f"query_pairs p={p}", lambda p=p: tree(small, leafsize=4).query_pairs(1.0, p=p)
        yield f"query_pairs lattice p={p}", lambda p=p: tree(grid).query_pairs(2.0, p=p)
        yield f"count_neighbors p={p}", lambda p=p: tree(small).count_neighbors(tree(other), rs, p=p)
        yield f"count_neighbors self p={p}", lambda p=p: tree(grid).count_neighbors(tree(grid), 2.0, p=p)
        yield f"sparse_distance_matrix p={p}", lambda p=p: tree(small).sparse_distance_matrix(tree(other), 1.0, p=p)
        yield f"query_ball_point big p={p}", lambda p=p: tree(big).query_ball_point(a(bigq), 0.1, p=p)
        yield f"query_pairs big p={p}", lambda p=p: tree(big).query_pairs(0.02, p=p)
        yield (
            f"count_neighbors big weighted p={p}",
            lambda p=p: tree(big).count_neighbors(
                tree(bigo), np.linspace(0.0, 0.2, 9), p=p, weights=(a(wbig), a(wbigo))
            ),
        )
    yield "query_ball_point radii", lambda: tree(small).query_ball_point(a(q), a(np.linspace(0.0, 2.0, 11)))
    yield (
        "query_ball_point batch",
        lambda: tree(small).query_ball_point(a(q[:6].reshape(2, 3, 3)), a(np.array([[0.5], [1.5]]))),
    )
    yield "query_ball_point one point", lambda: tree(small).query_ball_point(a(q[0]), 1.5)
    yield "query_ball_point length", lambda: tree(small).query_ball_point(a(q), 1.5, return_length=True)
    yield "query_ball_point length one", lambda: tree(small).query_ball_point(a(q[0]), 1.5, return_length=True)
    yield "query_ball_point length big", lambda: tree(big).query_ball_point(a(bigq), 0.3, return_length=True)
    yield "query_ball_point inf", lambda: tree(small[:8]).query_ball_point(a(q[:3]), INF)
    yield "query_ball_point huge", lambda: tree(small * 1e200).query_ball_point(a(q * 1e200), 1e200)
    yield "query_ball_point tiny", lambda: tree(small * 1e-200).query_ball_point(a(q * 1e-200), 1e-200)
    yield "query_ball_point empty tree", lambda: tree(np.zeros((0, 3))).query_ball_point(a(q[:2]), 1.0)
    yield "query_ball_tree self", lambda: tree(small).query_ball_tree(tree(small), 0.5)
    yield "query_ball_tree big", lambda: tree(big).query_ball_tree(tree(bigo), 0.03)
    yield "query_pairs empty", lambda: tree(np.zeros((0, 3))).query_pairs(1.0)
    yield "query_pairs no columns", lambda: tree(np.zeros((4, 0))).query_pairs(0.0)
    yield "count_neighbors one radius", lambda: tree(small).count_neighbors(tree(other), 1.0)
    yield "count_neighbors binned", lambda: tree(small).count_neighbors(tree(other), rs, cumulative=False)
    yield "count_neighbors unsorted", lambda: tree(small).count_neighbors(tree(other), [1.5, 0.5, 1.0, 0.5])
    yield "count_neighbors weights tuple", lambda: tree(small).count_neighbors(tree(other), rs, weights=(a(w), a(wo)))
    yield (
        "count_neighbors weights one side",
        lambda: tree(small).count_neighbors(tree(other), 1.0, weights=(None, a(wo))),
    )
    yield (
        "count_neighbors weights none none",
        lambda: tree(small).count_neighbors(tree(other), 1.0, weights=(None, None)),
    )
    yield (
        "count_neighbors weights self",
        lambda: (lambda t: t.count_neighbors(t, rs, weights=a(w), cumulative=False))(tree(small)),
    )
    yield "count_neighbors empty r", lambda: tree(small).count_neighbors(tree(other), a(np.zeros(0)))
    yield "count_neighbors big many radii", lambda: tree(big).count_neighbors(tree(bigo), np.linspace(0.0, 0.5, 200))
    yield "sparse_distance_matrix zeros", lambda: tree(small).sparse_distance_matrix(tree(small), 0.0)
    yield "sparse_distance_matrix big", lambda: tree(big).sparse_distance_matrix(tree(bigo), 0.03)

    # ---- wrong calls
    t = tree(small)
    yield "error query_ball_point r negative", lambda: t.query_ball_point(a(q), -1.0)
    yield "error query_ball_point r nan", lambda: t.query_ball_point(a(q), NAN)
    yield "error query_ball_point r shape", lambda: t.query_ball_point(a(q), [1.0, 2.0])
    yield "error query_ball_point return_length", lambda: t.query_ball_point(a(q), 1.0, return_length=1)
    yield "error query_ball_point x", lambda: t.query_ball_point(a(np.zeros((2, 2))), 1.0)
    yield "error query_ball_tree other", lambda: t.query_ball_tree(a(small), 1.0)
    yield "error query_ball_tree m", lambda: t.query_ball_tree(tree(np.zeros((3, 2))), 1.0)
    yield "error query_pairs r", lambda: t.query_pairs(-1.0)
    yield "error query_pairs positional p", lambda: t.query_pairs(1.0, 2.0)
    yield "error count_neighbors r", lambda: t.count_neighbors(t, [0.5, NAN])
    yield "error count_neighbors r 2-D", lambda: t.count_neighbors(t, [[0.5]])
    yield "error count_neighbors binned", lambda: t.count_neighbors(t, [1.0, 0.5], cumulative=False)
    yield "error count_neighbors weights", lambda: t.count_neighbors(tree(other), 1.0, weights=a(w))
    yield "error count_neighbors weights length", lambda: t.count_neighbors(t, 1.0, weights=(a(w[:5]), None))
    yield "error count_neighbors weights nan", lambda: t.count_neighbors(t, 1.0, weights=a(np.full(40, NAN)))
    yield "error sparse_distance_matrix max_distance", lambda: t.sparse_distance_matrix(t, NAN)
