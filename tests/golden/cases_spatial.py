"""Golden cases for sd.spatial (tools/golden.py has the contract).

The distances (milestone SPT1, docs/design/spatial.md §8): pdist, cdist and every distance function for
every metric, weighted and not, on small sets with the edge values (NaN, inf, zeros, constant rows, huge
and tiny coordinates that take the careful paths) and on sets large enough that pdist and cdist split into
many tasks at 3 and 10 threads (the tile kernel's row groups and panel chunks, the scalar drivers of
minkowski, mahalanobis and haversine, the bit-packed bool rows); the default V and VI; the batch shapes of
the distance functions; hamming and jaccard on bool, int64 and float64; squareform both ways with every
dtype; the wrong calls (their errors are results). The inputs come from fixed literals and L.rng(seed); the
largest result is 1.8 * 10^5 doubles.
"""

import math

import numpy as np

NAN, INF = math.nan, math.inf
METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "seuclidean", "mahalanobis", "cosine",
           "correlation", "canberra", "braycurtis", "hamming", "jaccard")  # fmt: skip
WEIGHTLESS = ("seuclidean", "mahalanobis")


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
