"""Golden cases for sd.cluster (tools/golden.py has the contract).

The cases come with the milestones that make the names public (docs/design/cluster.md §8). The hierarchy (CL1
milestone): the point sets of tests/cluster/samples.py's kinds rebuilt from literals and L.rng (separated and
overlapping blobs, an integer lattice and its shuffle, an evenly spaced line, repeated and all-equal points, a
hexagon and its centre, n = 2 and 3, collinear points, the narrow, shifted, huge and subnormal scales, the inversion
triangles) times every method and metric and 'precomputed', optimal_ordering, nan_policy; divisive; the scipy tools
on those trees (fcluster's criteria, cut_tree, cophenet and its correlation, leaves_list, optimal_leaf_ordering,
dendrogram's options, inconsistent, maxdists, maxinconsts, maxRstat, leaders, the predicates); HierarchyResult's
cut and summary; the wrong calls (their errors are results); and cases of thousands of points that split into tasks
at 3 and 10 threads (one worker below 1 GB).

The partitions (CL2 milestone): kmeans, minibatch_kmeans, bisecting_kmeans, kmedoids (PAM and CLARA, three
metrics and 'precomputed') and fuzzy_cmeans on those point sets and their options, the results' predict, transform
and predict_proba, nan_policy, the wrong calls, and cases of tens of thousands of points.

The density methods (CL3 milestone): dbscan, hdbscan and optics on those point sets, each option's values (eps,
min_samples, weights; min_cluster_size, cluster_selection_method, cluster_selection_epsilon, max_cluster_size,
allow_single_cluster, alpha; max_eps, xi, predecessor_correction, cluster_method), four metrics and 'precomputed',
the results' dbscan_clustering, extract_xi and extract_dbscan, nan_policy, the wrong calls, and cases of tens of
thousands of points.

The plan for the rest (docs/design/cluster.md §7.5): every public function on the representatives of
tests/cluster/samples.py times each option's values (covariance_type, reg_covar; average_method, beta), the
results' methods (predict, sample with its rng), NaN under each nan_policy and the same points in two orders
(decision CL7). Results are the same at 1, 3 and 10 threads (decision CL9), so no case is thread dependent.
"""

import math

NAN, INF = math.nan, math.inf
METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")

# Public names no case calls (tools/golden.py): the reason each needs none.
EXCLUDED = {
    "DendrogramResult": "a namedtuple class: dendrogram's cases store it whole",
    "CophenetResult": "a namedtuple class: cophenet_correlation's cases store it whole",
}


def _sets(L):
    """(name, X) of the hierarchy's point sets, rebuilt from literals and L.rng."""
    np = L.np
    r = L.rng(20261126)
    out = []
    c = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    out.append(("blobs60", np.vstack([c[j] + r.standard_normal((20, 2)) for j in range(3)])[r.permutation(60)]))
    out.append(("overlap45", np.vstack([c[j] / 3.0 + r.standard_normal((15, 2)) for j in range(3)])))
    out.append(("p5_40", r.standard_normal((40, 5))))
    lat = np.array([[i, j] for i in range(4) for j in range(4)], float)
    out.append(("lattice16", lat))
    out.append(("lattice16_shuffled", lat[r.permutation(16)]))
    out.append(("line7", np.arange(7.0)[:, None]))
    out.append(("dups20", np.repeat(np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]]), 5, axis=0)))
    out.append(("allsame6", np.full((6, 2), 1.25)))
    hexa = np.round(np.array([[math.cos(math.pi * j / 3), math.sin(math.pi * j / 3)] for j in range(6)]), 12)
    out.append(("hexagon7", np.vstack([[[0.0, 0.0]], hexa])))
    out.append(("two", np.array([[0.25, 0.25], [1.75, 0.75]])))
    out.append(("three", np.array([[0.0], [1.0], [3.5]])))
    t = r.random(25)
    out.append(("collinear25", np.column_stack([t, 2.0 * t + 1.0])))
    b = np.vstack([c[j] + r.standard_normal((10, 2)) for j in range(3)])
    out += [("narrow30", b * 1e-8), ("shifted30", b + 1e8), ("huge30", b * 1e150)]
    out.append(("subnormal30", np.round(b * 64) * 2.0**-1068))
    out.append(("inversion4", np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.8], [8.0, 0.0]])))
    out.append(("inversion30", r.standard_normal((30, 2))))
    return out


def _linkage_cases(L):
    sd, np = L.sd, L.np
    a = L.arr
    cl = sd.cluster
    sets = _sets(L)
    for name, X in sets:
        for m in METHODS:
            yield f"linkage {name} {m}", lambda X=X, m=m: cl.linkage(a(X), method=m)
        yield (
            f"linkage {name} average optimal_ordering",
            lambda X=X: cl.linkage(a(X), method="average", optimal_ordering=True),
        )
        yield f"agglomerative {name} ward", lambda X=X: cl.agglomerative(a(X), method="ward")
        yield f"divisive {name}", lambda X=X: cl.divisive(a(X))
    X = sets[0][1]
    for metric in ("cityblock", "chebyshev", "cosine", "minkowski", "sqeuclidean"):
        for m in ("single", "complete", "average", "weighted"):
            yield (
                f"linkage blobs60 {m} metric={metric}",
                lambda m=m, metric=metric: cl.linkage(a(X), method=m, metric=metric),
            )
        yield f"divisive blobs60 metric={metric}", lambda metric=metric: cl.divisive(a(X), metric=metric)
    y = sd.spatial.pdist(a(X))
    for m in METHODS:
        yield f"linkage condensed {m}", lambda m=m: cl.linkage(y, method=m)
        yield (
            f"agglomerative precomputed square {m}",
            lambda m=m: cl.agglomerative(sd.spatial.squareform(y), method=m, metric="precomputed"),
        )
    Xn = X.copy()
    Xn[[3, 17], 0] = NAN
    for pol in ("raise", "omit"):
        yield (
            f"agglomerative nan_policy={pol}",
            lambda pol=pol: cl.agglomerative(a(Xn), method="average", nan_policy=pol),
        )
        yield f"agglomerative nan_policy={pol} cut", lambda pol=pol: cl.agglomerative(a(Xn), nan_policy=pol).cut(k=3)
        yield f"divisive nan_policy={pol}", lambda pol=pol: cl.divisive(a(Xn), nan_policy=pol)
    yield "linkage list input", lambda: cl.linkage(X[:10].tolist(), method="complete")
    yield "linkage int input", lambda: cl.linkage(np.arange(12).reshape(6, 2), method="ward")
    yield "agglomerative 1-D", lambda: cl.agglomerative(a(np.array([0.0, 1.0, 5.0, 6.5, 20.0])), method="median")
    for kw in (
        {"method": "Ward"}, {"method": "ward", "metric": "cityblock"}, {"metric": "manhattan"},
        {"optimal_ordering": 1}, {"method": 3},
    ):  # fmt: skip
        yield f"linkage error {sorted(kw)}={[kw[k] for k in sorted(kw)]!r}", lambda kw=kw: cl.linkage(a(X), **kw)
    for bad, label in ((np.zeros((1, 2)), "one point"), (np.array([1.0, 2.0]), "bad condensed"),
                       (np.array([[0.0], [INF]]), "inf"), (Xn, "nan"), (np.zeros((2, 2, 2)), "3-D")):  # fmt: skip
        yield f"linkage error {label}", lambda bad=bad: cl.linkage(a(bad))
    yield "agglomerative error nan_policy", lambda: cl.agglomerative(a(X), nan_policy="drop")
    yield "linkage error positional method", lambda: cl.linkage(a(X), "ward")
    # large enough to split into tasks
    big = r_big = L.rng(5).standard_normal((6000, 3))
    for m in METHODS:
        yield f"linkage 6000 {m}", lambda m=m: cl.linkage(a(big), method=m)
    yield "linkage 30000 single", lambda: cl.linkage(a(L.rng(6).standard_normal((30000, 2))))
    yield "divisive 1500", lambda: cl.divisive(a(r_big[:1500]))
    yield "cophenet 6000", lambda: cl.cophenet(cl.linkage(a(big), method="average"))
    yield "optimal_leaf_ordering 400", lambda: cl.linkage(a(big[:400]), method="ward", optimal_ordering=True)


def _tool_cases(L):
    sd, np = L.sd, L.np
    a = L.arr
    cl = sd.cluster
    trees = {}
    for name, X in _sets(L):
        if name in ("blobs60", "lattice16", "line7", "inversion4", "dups20", "three"):
            for m in ("single", "average", "ward", "centroid"):
                trees[f"{name} {m}"] = (cl.linkage(a(X), method=m), X)
    for tname, (Z, X) in trees.items():
        n = X.shape[0]
        yield f"leaves_list {tname}", lambda Z=Z: cl.leaves_list(Z)
        yield f"cophenet {tname}", lambda Z=Z: cl.cophenet(Z)
        yield f"cophenet_correlation {tname}", lambda Z=Z, X=X: cl.cophenet_correlation(Z, sd.spatial.pdist(a(X)))
        yield f"optimal_leaf_ordering {tname}", lambda Z=Z, X=X: cl.optimal_leaf_ordering(Z, a(X))
        yield f"is_monotonic {tname}", lambda Z=Z: cl.is_monotonic(Z)
        yield f"is_valid_linkage {tname}", lambda Z=Z: cl.is_valid_linkage(Z)
        yield f"num_obs_linkage {tname}", lambda Z=Z: cl.num_obs_linkage(Z)
        yield f"maxdists {tname}", lambda Z=Z: cl.maxdists(Z)
        for d in (0, 1, 2, 3):
            yield f"inconsistent {tname} d={d}", lambda Z=Z, d=d: cl.inconsistent(Z, d=d)
        R = cl.inconsistent(Z)
        yield f"maxinconsts {tname}", lambda Z=Z, R=R: cl.maxinconsts(Z, R)
        for i in range(4):
            yield f"maxRstat {tname} i={i}", lambda Z=Z, R=R, i=i: cl.maxRstat(Z, R, i)
        for t in (0.0, 0.7, 1.15, 3.0):
            yield f"fcluster {tname} inconsistent t={t}", lambda Z=Z, t=t: cl.fcluster(Z, t)
            yield f"fcluster {tname} distance t={t}", lambda Z=Z, t=t: cl.fcluster(Z, t, criterion="distance")
            yield (
                f"fcluster {tname} monocrit t={t}",
                lambda Z=Z, t=t: cl.fcluster(Z, t, criterion="monocrit", monocrit=cl.maxdists(Z)),
            )
        for k in sorted({1, 2, 3, n - 1, n, n + 3}):
            if k < 1:
                continue
            yield f"fcluster {tname} maxclust t={k}", lambda Z=Z, k=k: cl.fcluster(Z, k, criterion="maxclust")
            yield (
                f"fcluster {tname} maxclust_monocrit t={k}",
                lambda Z=Z, k=k: cl.fcluster(Z, k, criterion="maxclust_monocrit", monocrit=cl.maxdists(Z)),
            )
            if k <= n:
                yield f"cut_tree {tname} n_clusters={k}", lambda Z=Z, k=k: cl.cut_tree(Z, n_clusters=k)
            T = cl.fcluster(Z, k, criterion="maxclust")
            yield f"leaders {tname} k={k}", lambda Z=Z, T=T: cl.leaders(Z, T)
        yield f"cut_tree {tname} full", lambda Z=Z: cl.cut_tree(Z)
        yield f"cut_tree {tname} n_clusters=[1, 2]", lambda Z=Z: cl.cut_tree(Z, n_clusters=[1, 2])
        for h in (0.5, 1.0, 2.5):
            yield f"cut_tree {tname} height={h}", lambda Z=Z, h=h: cl.cut_tree(Z, height=h)
        for kw in (
            {}, {"truncate_mode": "lastp", "p": 3}, {"truncate_mode": "level", "p": 1}, {"count_sort": "ascending"},
            {"distance_sort": True}, {"color_threshold": 0.0}, {"color_threshold": 1e300},
            {"show_leaf_counts": False, "truncate_mode": "lastp", "p": 2}, {"labels": [f"p{i}" for i in range(n)]},
            {"orientation": "left"},
        ):  # fmt: skip
            yield f"dendrogram {tname} {kw}", lambda Z=Z, kw=kw: cl.dendrogram(Z, **kw)
    Z, X = trees["blobs60 ward"]
    r = cl.agglomerative(a(X), method="ward")
    for k in (1, 2, 3, 60):
        yield f"HierarchyResult cut k={k}", lambda k=k: r.cut(k=k)
    for h in (0.0, 2.0, 1e9):
        yield f"HierarchyResult cut height={h}", lambda h=h: r.cut(height=h)
    yield "HierarchyResult dendrogram", lambda: r.dendrogram(truncate_mode="lastp", p=5)
    ri = cl.agglomerative(a(trees["inversion4 centroid"][1]), method="centroid")
    yield "HierarchyResult inversion cut k=3", lambda: ri.cut(k=3)
    yield "HierarchyResult error inversion cut height", lambda: ri.cut(height=1.9)
    yield "HierarchyResult error cut nothing", lambda: r.cut()
    yield "HierarchyResult error cut both", lambda: r.cut(k=2, height=1.0)
    yield "HierarchyResult error cut k=0", lambda: r.cut(k=0)
    yield "is_isomorphic same", lambda: cl.is_isomorphic([1, 1, 2, 3], [5, 5, 1, 0])
    yield "is_isomorphic different", lambda: cl.is_isomorphic([1, 1, 2, 3], [5, 5, 5, 0])
    yield "is_isomorphic strings", lambda: cl.is_isomorphic(["a", "a", "b"], [2, 2, 7])
    bad = np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 3.0]])
    for label, Zb in (
        ("count", np.array([[0.0, 1.0, 1.0, 3.0]])), ("twice", np.array([[0.0, 0.0, 1.0, 2.0]])),
        ("before formed", np.array([[0.0, 2.0, 1.0, 2.0]])), ("negative", np.array([[0.0, 1.0, -1.0, 2.0]])),
        ("nan height", np.array([[0.0, 1.0, NAN, 2.0]])), ("fraction", np.array([[0.5, 1.0, 1.0, 2.0]])),
        ("empty", np.zeros((0, 4))), ("3 columns", np.zeros((2, 3))),
    ):  # fmt: skip
        yield f"is_valid_linkage {label}", lambda Zb=Zb: cl.is_valid_linkage(a(Zb))
        yield f"leaves_list error {label}", lambda Zb=Zb: cl.leaves_list(a(Zb))
    yield "fcluster error t=0 maxclust", lambda: cl.fcluster(a(bad), 0, criterion="maxclust")
    yield "fcluster error criterion", lambda: cl.fcluster(a(bad), 1.0, criterion="dist")
    yield "fcluster error monocrit missing", lambda: cl.fcluster(a(bad), 1.0, criterion="monocrit")
    yield (
        "fcluster error monocrit decreasing",
        lambda: cl.fcluster(a(bad), 1.0, criterion="monocrit", monocrit=a(np.array([2.0, 1.0]))),
    )
    yield "fcluster error R", lambda: cl.fcluster(a(bad), 1.0, criterion="distance", R=a(np.zeros((2, 4))))
    yield "leaders error not a subtree", lambda: cl.leaders(a(bad), a(np.array([1, 2, 1])))
    yield "cut_tree error both", lambda: cl.cut_tree(a(bad), n_clusters=2, height=1.0)
    yield "cophenet_correlation error length", lambda: cl.cophenet_correlation(a(bad), a(np.ones(2)))
    yield "dendrogram error labels", lambda: cl.dendrogram(a(bad), labels=["a"])
    yield "dendrogram error orientation", lambda: cl.dendrogram(a(bad), orientation="up")
    yield "inconsistent error d", lambda: cl.inconsistent(a(bad), d=-1)
    yield "maxRstat error i", lambda: cl.maxRstat(a(bad), cl.inconsistent(a(bad)), 4)


def _partition_cases(L):
    """Milestone CL2: kmeans, minibatch_kmeans, bisecting_kmeans, kmedoids and fuzzy_cmeans on the hierarchy's
    point sets and their options, the results' methods, NaN, the wrong calls and large cases."""
    sd, np = L.sd, L.np
    a = L.arr
    cl = sd.cluster

    def g(seed=0):
        return sd.random.default_rng(seed)

    sets = dict(_sets(L))
    for name in ("blobs60", "overlap45", "p5_40", "lattice16", "lattice16_shuffled", "line7", "dups20", "hexagon7",
                 "three", "collinear25", "narrow30", "shifted30", "huge30", "subnormal30"):  # fmt: skip
        X = sets[name]
        k = 2 if name in ("three", "line7") else 3
        yield f"kmeans {name}", lambda X=X, k=k: cl.kmeans(a(X), k, rng=g())
        yield f"kmeans {name} random n_init=4", lambda X=X, k=k: cl.kmeans(a(X), k, rng=g(1), init="random", n_init=4)
        yield f"kmeans {name} init array tol=0", lambda X=X, k=k: cl.kmeans(a(X), k, rng=g(), init=a(X[:k]), tol=0.0)
        yield f"minibatch_kmeans {name}", lambda X=X, k=k: cl.minibatch_kmeans(a(X), k, rng=g(2), batch_size=16)
        yield f"bisecting_kmeans {name}", lambda X=X, k=k: cl.bisecting_kmeans(a(X), k, rng=g(3))
        yield f"kmedoids {name}", lambda X=X, k=k: cl.kmedoids(a(X), k)
        yield f"fuzzy_cmeans {name}", lambda X=X, k=k: cl.fuzzy_cmeans(a(X), k, rng=g(4))
    X = sets["blobs60"]
    Xe = np.array([[0.0, 0.0], [3.0, 3.0], [100.0, -50.0]])
    r = cl.kmeans(a(X), 3, rng=g())
    yield "KMeansResult predict", lambda: r.predict(a(Xe))
    yield "KMeansResult transform", lambda: r.transform(a(Xe))
    yield "KMeansResult error predict columns", lambda: r.predict(a(np.zeros((2, 3))))
    w = np.linspace(0.0, 2.0, len(X))
    yield "kmeans weights", lambda: cl.kmeans(a(X), 3, rng=g(), weights=a(w))
    yield "kmeans k=1", lambda: cl.kmeans(a(X), 1, rng=g())
    yield "kmeans max_iter=1", lambda: cl.kmeans(a(X), 3, rng=g(), max_iter=1)
    yield "kmeans empty cluster", lambda: cl.kmeans(a(np.array([[0.0], [0.1], [10.0], [10.1]])), 3, rng=g(),
                                                    init=a(np.array([[0.0], [10.0], [100.0]])))  # fmt: skip
    yield "minibatch_kmeans options", lambda: cl.minibatch_kmeans(
        a(X), 3, rng=g(), init="random", batch_size=8, max_iter=5, max_no_improvement=None, tol=1e-3,
        init_size=10, reassignment_ratio=0.5, weights=a(w),
    )  # fmt: skip
    rb = cl.bisecting_kmeans(a(X), 4, rng=g(), strategy="largest_cluster", init="k-means++", n_init=2)
    yield "bisecting_kmeans largest_cluster", lambda: rb
    yield "BisectingKMeansResult predict", lambda: rb.predict(a(Xe))
    for metric in ("euclidean", "cityblock", "chebyshev"):
        yield f"kmedoids metric={metric}", lambda metric=metric: cl.kmedoids(a(X), 3, metric=metric)
    y = sd.spatial.pdist(a(X))
    yield "kmedoids precomputed", lambda: cl.kmedoids(y, 3, metric="precomputed")
    yield "kmedoids max_iter=0", lambda: cl.kmedoids(a(X), 3, max_iter=0)
    yield "kmedoids clara", lambda: cl.kmedoids(a(X), 3, method="clara", rng=g(5), samples=3, sample_size=20)
    rm = cl.kmedoids(a(X), 3)
    yield "KMedoidsResult predict", lambda: rm.predict(a(Xe))
    for m in (1.5, 2.0, 3.0):
        yield f"fuzzy_cmeans m={m}", lambda m=m: cl.fuzzy_cmeans(a(X), 3, rng=g(), m=m)
    rf = cl.fuzzy_cmeans(a(X), 3, rng=g(), init=a(X[:3]), tol=1e-8)
    yield "FuzzyCMeansResult predict_proba", lambda: rf.predict_proba(a(Xe))
    yield "FuzzyCMeansResult predict", lambda: rf.predict(a(Xe))
    Xn = X.copy()
    Xn[[3, 17], 0] = NAN
    for pol in ("raise", "omit"):
        yield f"kmeans nan_policy={pol}", lambda pol=pol: cl.kmeans(a(Xn), 3, rng=g(), nan_policy=pol)
        yield f"kmedoids nan_policy={pol}", lambda pol=pol: cl.kmedoids(a(Xn), 3, nan_policy=pol)
        yield f"fuzzy_cmeans nan_policy={pol}", lambda pol=pol: cl.fuzzy_cmeans(a(Xn), 3, rng=g(), nan_policy=pol)
    yield "kmeans error few distinct", lambda: cl.kmeans(a(np.zeros((5, 2))), 2, rng=g())
    yield "kmeans error no rng", lambda: cl.kmeans(a(X), 2)
    yield "kmeans error init", lambda: cl.kmeans(a(X), 2, rng=g(), init="kmeans++")
    yield "kmeans error n_init", lambda: cl.kmeans(a(X), 2, rng=g(), init=a(X[:2]), n_init=2)
    yield "kmedoids error rng with pam", lambda: cl.kmedoids(a(X), 2, rng=g())
    yield "kmedoids error clara sample_size", lambda: cl.kmedoids(a(X), 2, method="clara", rng=g(), sample_size=2)
    yield "fuzzy_cmeans error m", lambda: cl.fuzzy_cmeans(a(X), 2, rng=g(), m=1.0)
    # large enough to split into tasks
    rr = L.rng(7)
    big = np.vstack([rr.normal(c, 1.0, (10000, 5)) for c in (0.0, 4.0, 8.0)])
    yield "kmeans 30000", lambda: cl.kmeans(a(big), 8, rng=g())
    yield "kmeans 30000 p=40 k=30", lambda: cl.kmeans(a(rr.standard_normal((20000, 40))), 30, rng=g())
    yield "minibatch_kmeans 30000", lambda: cl.minibatch_kmeans(a(big), 8, rng=g())
    yield "bisecting_kmeans 30000", lambda: cl.bisecting_kmeans(a(big), 8, rng=g())
    yield "fuzzy_cmeans 30000", lambda: cl.fuzzy_cmeans(a(big), 4, rng=g())
    yield "kmedoids 2000", lambda: cl.kmedoids(a(big[::15]), 5)
    yield "kmedoids clara 30000", lambda: cl.kmedoids(a(big), 5, method="clara", rng=g())


def _density_cases(L):
    """Milestone CL3: dbscan, hdbscan and optics on the hierarchy's point sets and their options, the results'
    methods (dbscan_clustering, extract_xi, extract_dbscan), the metrics and 'precomputed', NaN, the wrong calls and
    large cases."""
    sd, np = L.sd, L.np
    a = L.arr
    cl = sd.cluster
    sets = dict(_sets(L))
    names = ("blobs60", "overlap45", "p5_40", "lattice16", "lattice16_shuffled", "line7", "dups20", "allsame6",
             "hexagon7", "two", "three", "collinear25", "narrow30", "shifted30", "huge30", "subnormal30")  # fmt: skip
    for name in names:
        X = sets[name]
        sc = float(np.max(np.abs(X - np.median(X, axis=0)))) or 1.0
        eps = 0.3 * sc if name in ("narrow30", "shifted30", "huge30", "subnormal30") else 1.0
        ms = min(3, len(X))
        m2 = max(2, ms)
        yield f"dbscan {name}", lambda X=X, eps=eps, ms=ms: cl.dbscan(a(X), eps=eps, min_samples=ms)
        yield f"hdbscan {name}", lambda X=X, ms=ms, m2=m2: cl.hdbscan(a(X), min_cluster_size=m2, min_samples=ms)
        yield f"optics {name}", lambda X=X, m2=m2: cl.optics(a(X), min_samples=m2)
    X = sets["blobs60"]
    for eps in (0.3, 1.0, 3.0):
        for ms in (1, 4, 10):
            yield (
                f"dbscan blobs60 eps={eps} min_samples={ms}",
                lambda eps=eps, ms=ms: cl.dbscan(a(X), eps=eps, min_samples=ms),
            )
    w = np.linspace(0.0, 2.0, len(X))
    yield "dbscan weights", lambda: cl.dbscan(a(X), eps=1.0, weights=a(w))
    yield "DBSCANResult summary", lambda: cl.dbscan(a(X), eps=1.0).summary()
    for metric in ("cityblock", "chebyshev", "cosine", "minkowski"):
        yield f"dbscan metric={metric}", lambda metric=metric: cl.dbscan(a(X), eps=0.8, metric=metric)
        yield f"hdbscan metric={metric}", lambda metric=metric: cl.hdbscan(a(X), metric=metric)
        yield f"optics metric={metric}", lambda metric=metric: cl.optics(a(X), metric=metric)
    y = sd.spatial.pdist(a(X))
    yield "dbscan precomputed", lambda: cl.dbscan(y, eps=1.0, metric="precomputed")
    yield "hdbscan precomputed square", lambda: cl.hdbscan(sd.spatial.squareform(y), metric="precomputed")
    yield "optics precomputed", lambda: cl.optics(y, metric="precomputed")
    for kw in ({"cluster_selection_method": "leaf"}, {"min_cluster_size": 10}, {"min_samples": 2},
               {"allow_single_cluster": True}, {"cluster_selection_epsilon": 1.5}, {"max_cluster_size": 15},
               {"alpha": 0.5}, {"cluster_selection_method": "leaf", "cluster_selection_epsilon": 1.0},
               {"allow_single_cluster": True, "min_cluster_size": 40}):  # fmt: skip
        yield f"hdbscan {sorted(kw.items())!r}", lambda kw=kw: cl.hdbscan(a(X), **kw)
    rh = cl.hdbscan(a(X))
    for cut in (0.3, 0.8, 2.0):
        yield f"HDBSCANResult dbscan_clustering {cut}", lambda cut=cut: rh.dbscan_clustering(cut)
    yield "HDBSCANResult dbscan_clustering min_cluster_size", lambda: rh.dbscan_clustering(0.8, min_cluster_size=2)
    for kw in ({"max_eps": 1.0}, {"min_samples": 0.1}, {"xi": 0.2}, {"predecessor_correction": False},
               {"min_cluster_size": 10}, {"cluster_method": "dbscan"}, {"cluster_method": "dbscan", "eps": 0.7},
               {"cluster_method": "dbscan", "max_eps": 2.0, "eps": 1.0}):  # fmt: skip
        yield f"optics {sorted(kw.items())!r}", lambda kw=kw: cl.optics(a(X), **kw)
    ro = cl.optics(a(X))
    for xi in (0.01, 0.1, 0.4):
        yield f"OPTICSResult extract_xi {xi}", lambda xi=xi: ro.extract_xi(xi=xi)
    yield "OPTICSResult extract_xi options", lambda: ro.extract_xi(min_samples=3, min_cluster_size=0.2,
                                                                   predecessor_correction=False)  # fmt: skip
    for eps in (0.2, 0.6, INF):
        yield f"OPTICSResult extract_dbscan {eps}", lambda eps=eps: ro.extract_dbscan(eps)
    Xn = X.copy()
    Xn[[3, 17], 0] = NAN
    for pol in ("raise", "omit"):
        yield f"dbscan nan_policy={pol}", lambda pol=pol: cl.dbscan(a(Xn), eps=1.0, nan_policy=pol)
        yield f"hdbscan nan_policy={pol}", lambda pol=pol: cl.hdbscan(a(Xn), nan_policy=pol)
        yield f"optics nan_policy={pol}", lambda pol=pol: cl.optics(a(Xn), nan_policy=pol)
    yield (
        "hdbscan nan_policy=omit dbscan_clustering",
        lambda: cl.hdbscan(a(Xn), nan_policy="omit").dbscan_clustering(1.0),
    )
    yield "dbscan error eps", lambda: cl.dbscan(a(X), eps=0.0)
    yield "dbscan error positional eps", lambda: cl.dbscan(a(X), 0.5)
    yield "dbscan error weights", lambda: cl.dbscan(a(X), weights=a(-w))
    yield "hdbscan error min_cluster_size", lambda: cl.hdbscan(a(X), min_cluster_size=1)
    yield "hdbscan error min_samples", lambda: cl.hdbscan(a(X), min_samples=100)
    yield "hdbscan error copy", lambda: cl.hdbscan(a(X), copy=True)
    yield "hdbscan error one point", lambda: cl.hdbscan(a(X[:1]))
    yield "optics error min_samples", lambda: cl.optics(a(X), min_samples=1)
    yield "optics error eps above max_eps", lambda: cl.optics(a(X), cluster_method="dbscan", max_eps=1.0, eps=2.0)
    yield "OPTICSResult extract_dbscan error", lambda: ro.extract_dbscan(-1.0)
    # large enough to split into tasks
    rr = L.rng(11)
    big = np.vstack([rr.normal(c, 0.5, (10000, 2)) for c in (0.0, 2.0, 4.0)] + [rr.uniform(-2, 6, (2000, 2))])
    yield "dbscan 32000", lambda: cl.dbscan(a(big), eps=0.05)
    yield "dbscan 32000 weights", lambda: cl.dbscan(a(big), eps=0.05, weights=a(rr.uniform(0.0, 2.0, len(big))))
    yield "hdbscan 32000", lambda: cl.hdbscan(a(big))
    yield "hdbscan 32000 p=5", lambda: cl.hdbscan(a(rr.standard_normal((8000, 5))), min_cluster_size=20)
    yield "optics 12000", lambda: cl.optics(a(big[::3]))
    yield "optics 12000 max_eps", lambda: cl.optics(a(big[::3]), max_eps=0.3, cluster_method="dbscan", eps=0.1)


def cases(L):
    yield from _linkage_cases(L)
    yield from _tool_cases(L)
    yield from _partition_cases(L)
    yield from _density_cases(L)
