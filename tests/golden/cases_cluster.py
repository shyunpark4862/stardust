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

The plan for the rest (docs/design/cluster.md §7.5): every public function on the representatives of
tests/cluster/samples.py times each option's values (init, n_init, max_iter, tol; eps, min_samples,
min_cluster_size, cluster_selection_method; covariance_type, reg_covar; average_method, beta), the results' methods
(predict, transform, sample with its rng, extract_xi, extract_dbscan, dbscan_clustering), NaN under each
nan_policy and the same points in two orders (decision CL7). Results are the same at 1, 3 and 10 threads (decision
CL9), so no case is thread dependent.
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


def cases(L):
    yield from _linkage_cases(L)
    yield from _tool_cases(L)
