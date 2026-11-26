"""sd.cluster against scipy 1.18.1, scikit-learn 1.9.1 and R 4.6.2's stats and cluster (docs/design/cluster.md §5,
§7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the sd.cluster names
it needs, and cases() keeps a spec once its names are public (CL1 the hierarchy, CL2 k-means and k-medoids, CL3
the density methods, CL4 the mixtures, CL5 the indices, the gap statistic and Gower's distance, CL6 affinity
propagation). The sizes are those of docs/design/cluster.md §5, where
milestone CL0 timed the comparators (tools/cluster_probe.py time). The matrix-free hierarchy at n = 10^5 has no
comparator: scipy and R would need the 40 GB condensed matrix. k-means and the mixture start every library from
the same centres (the data's first k rows), so the smoke run compares their fixed points; scikit-learn runs on its
OpenMP threads (it follows the thread count), R and scipy on one.

Data: Gaussian points from fixed seeds, made in the setup as numpy arrays; sd gets them as sd.ndarray outside the
timing.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..registry import Case, Check, Ref, RRef, count


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
    check: Check = Check()


def _points(n, p, seed=20261124):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n, p]))
        X = g.standard_normal((n, p))
        return {"X": X, "sX": sd.asarray(X), "r_x": np.ascontiguousarray(X.T.ravel()), "dims": np.array([n, p], float)}

    return setup


def _blobs(n, p, k, seed=20261125):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n, p, k]))
        centers = 4.0 * g.standard_normal((k, p))
        X = centers[g.integers(0, k, n)] + g.standard_normal((n, p))
        init = X[:k].copy()
        lab = g.integers(0, k, n)
        return {
            "X": X,
            "sX": sd.asarray(X),
            "init": init,
            "sinit": sd.asarray(init),
            "labels": lab,
            "r_lab": lab + 1.0,
            "r_x": np.ascontiguousarray(X.T.ravel()),
            "dims": np.array([n, p], float),
        }

    return setup


def _labelings(n, seed=20261126):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n]))
        a = g.integers(0, 20, n)
        b = np.where(g.random(n) < 0.9, a, g.integers(0, 25, n))
        return {"a": a, "b": b, "sa": sd.asarray(a), "sb": sd.asarray(b)}

    return setup


def _cl():
    import stardust as sd

    return sd.cluster


def _sch():
    import scipy.cluster.hierarchy as sch

    return sch


R_X = {"r_x": "xv", "dims": "dims"}
R_MATRIX = "x <- matrix(xv, nrow = dims[1])"


def _hierarchy_specs():
    out = []
    for method in ("single", "complete", "average", "weighted", "ward", "centroid", "median"):
        refs = [Ref("scipy", lambda d, m=method: lambda: _sch().linkage(d["X"], m))]
        if method in ("complete", "average"):
            refs.append(RRef("R", f"hclust(dist(x), '{method}')$height", data=R_X, setup=R_MATRIX))
        out.append(
            Spec(
                "CL1",
                "hierarchy",
                f"linkage {method} p=5",
                ("linkage",),
                count(5000),
                _points(5000, 5),
                lambda d, m=method: lambda: _cl().linkage(d["sX"], method=m),
                tuple(refs),
                quick=method in ("single", "ward"),
                check=Check(sd=_heights, ref=_ref_heights, rtol=1e-12),
            )
        )
    for method in ("single", "ward"):
        out.append(
            Spec(
                "CL1",
                "hierarchy",
                f"linkage {method} p=5 matrix-free",
                ("linkage",),
                count(10**5),
                _points(10**5, 5),
                lambda d, m=method: lambda: _cl().linkage(d["sX"], method=m),
                note="no comparator: scipy and R need the 40 GB condensed matrix",
            )
        )
    out.append(
        Spec(
            "CL1",
            "hierarchy",
            "divisive p=5",
            ("divisive",),
            count(1000),
            _points(1000, 5),
            lambda d: lambda: _cl().divisive(d["sX"]).linkage,
            (
                RRef(
                    "R cluster",
                    "cluster::diana(x)$height",
                    data=R_X,
                    setup=R_MATRIX,
                    packages=("cluster",),
                    differs="R's diana gives its banner (the heights in leaf order), not the merges",
                ),
            ),
        )
    )
    out.append(
        Spec(
            "CL1",
            "hierarchy",
            "cophenet",
            ("cophenet",),
            count(5000),
            lambda: _with_tree(5000, 5),
            lambda d: lambda: _cl().cophenet(d["sZ"]),
            (Ref("scipy", lambda d: lambda: _sch().cophenet(d["Z"])),),
            check=Check(rtol=1e-12),
        )
    )
    out.append(
        Spec(
            "CL1",
            "hierarchy",
            "optimal_leaf_ordering",
            ("optimal_leaf_ordering",),
            count(1000),
            lambda: _with_tree(1000, 5, distances=True),
            lambda d: lambda: _cl().optimal_leaf_ordering(d["sZ"], d["sy"]),
            (
                Ref(
                    "scipy",
                    lambda d: lambda: _sch().optimal_leaf_ordering(d["Z"], d["y"]),
                    differs="scipy 1.18.1's order is not always the optimal one (decision CL53)",
                ),
            ),
        )
    )
    return out


def _heights(Z):
    """The sorted heights of a linkage matrix (R gives its heights alone, in its merge order)."""
    return np.sort(np.asarray(Z)[:, 2])


def _ref_heights(label, v):
    v = np.asarray(v)
    return np.sort(v[:, 2] if v.ndim == 2 else v)


def _with_tree(n, p, distances=False):
    import stardust as sd

    d = _points(n, p)()
    Z = _sch().linkage(d["X"], "average")
    d.update({"Z": Z, "sZ": sd.asarray(Z)})
    if distances:
        import scipy.spatial.distance as ssd

        y = ssd.pdist(d["X"])
        d.update({"y": y, "sy": sd.asarray(y)})
    return d


def _kmeans_specs():
    refs = (
        Ref(
            "sklearn",
            lambda d: (
                lambda: (
                    __import__("sklearn.cluster", fromlist=["KMeans"])
                    .KMeans(10, init=d["init"], n_init=1, max_iter=300, tol=1e-4)
                    .fit(d["X"])
                )
            ),
            threads="follow",
        ),
        RRef(
            "R Lloyd",
            "kmeans(x, x[1:10, ], iter.max = 300, algorithm = 'Lloyd')$tot.withinss",
            data=R_X,
            setup=R_MATRIX,
            differs="R's Lloyd has no tol: it runs until no label changes (117 iterations here, sd's tol 1e-4 stops at 11)",
        ),
        RRef(
            "R",
            "kmeans(x, x[1:10, ], iter.max = 300)$tot.withinss",
            data=R_X,
            setup=R_MATRIX,
            differs="R's default Hartigan-Wong moves single points: another local minimum",
        ),
    )
    return [
        Spec(
            "CL2",
            "partition",
            "kmeans p=10 k=10",
            ("kmeans",),
            count(10**5),
            _blobs(10**5, 10, 10),
            lambda d: (
                lambda: _cl().kmeans(d["sX"], 10, rng=__import__("stardust").random.default_rng(0), init=d["sinit"])
            ),
            refs,
            quick=True,
            check=Check(
                sd=lambda r: float(r.inertia),
                ref=lambda label, r: float(r.inertia_) if hasattr(r, "inertia_") else float(np.asarray(r).ravel()[0]),
                rtol=1e-9,
            ),
        )
    ]


def _partition_specs():
    """minibatch_kmeans, bisecting_kmeans, kmedoids (PAM and CLARA) and fuzzy_cmeans (CL2): random starts on every
    side, so the smoke run checks the size of the objective only loosely (or not at all)."""

    def rng():
        return __import__("stardust").random.default_rng(0)

    loose = Check(sd=lambda r: float(r.inertia), ref=lambda label, r: float(r.inertia_), rtol=0.05)
    return [
        Spec(
            "CL2",
            "partition",
            "minibatch_kmeans p=10 k=10",
            ("minibatch_kmeans",),
            count(10**5),
            _blobs(10**5, 10, 10),
            lambda d: lambda: _cl().minibatch_kmeans(d["sX"], 10, rng=rng()),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: (
                            __import__("sklearn.cluster", fromlist=["MiniBatchKMeans"])
                            .MiniBatchKMeans(10, random_state=0)
                            .fit(d["X"])
                        )
                    ),
                    threads="follow",
                ),
            ),
            check=loose,
        ),
        Spec(
            "CL2",
            "partition",
            "bisecting_kmeans p=10 k=10",
            ("bisecting_kmeans",),
            count(10**5),
            _blobs(10**5, 10, 10),
            lambda d: lambda: _cl().bisecting_kmeans(d["sX"], 10, rng=rng()),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: (
                            __import__("sklearn.cluster", fromlist=["BisectingKMeans"])
                            .BisectingKMeans(10, random_state=0)
                            .fit(d["X"])
                        )
                    ),
                    threads="follow",
                ),
            ),
            check=Check(skip="random starts on both sides: bisections differ"),
        ),
        Spec(
            "CL2",
            "partition",
            "kmedoids pam p=5 k=10",
            ("kmedoids",),
            count(2000),
            _blobs(2000, 5, 10),
            lambda d: lambda: _cl().kmedoids(d["sX"], 10),
            (RRef("R", "pam(x, 10)$objective[2]", data=R_X, setup="library(cluster); " + R_MATRIX),),
            check=Check(
                sd=lambda r: float(r.objective), ref=lambda label, r: float(np.asarray(r).ravel()[0]), rtol=1e-9
            ),
        ),
        Spec(
            "CL2",
            "partition",
            "kmedoids clara p=5 k=10",
            ("kmedoids",),
            count(10**5),
            _blobs(10**5, 5, 10),
            lambda d: lambda: _cl().kmedoids(d["sX"], 10, method="clara", rng=rng()),
            (
                RRef(
                    "R",
                    "clara(x, 10)$objective",
                    data=R_X,
                    setup="library(cluster); " + R_MATRIX,
                    differs="R's own sampler: other samples",
                ),
            ),  # fmt: skip
            check=Check(
                sd=lambda r: float(r.objective), ref=lambda label, r: float(np.asarray(r).ravel()[0]), rtol=0.05
            ),
        ),
        Spec(
            "CL2",
            "partition",
            "fuzzy_cmeans p=10 k=10",
            ("fuzzy_cmeans",),
            count(10**5),
            _blobs(10**5, 10, 10),
            lambda d: lambda: _cl().fuzzy_cmeans(d["sX"], 10, rng=rng(), init=d["sinit"]),
            note="no comparator: e1071 is not installed and R's fanny minimizes another objective on n x n distances",
        ),
    ]


def _density_specs():
    return [
        Spec(
            "CL3",
            "density",
            "dbscan p=2 eps=0.05",
            ("dbscan",),
            "20000",
            _points(2 * 10**4, 2),
            lambda d: lambda: _cl().dbscan(d["sX"], eps=0.05, min_samples=5),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: (
                            __import__("sklearn.cluster", fromlist=["DBSCAN"])
                            .DBSCAN(eps=0.05, min_samples=5)
                            .fit(d["X"])
                        )
                    ),
                    differs="border points join the first cluster that reaches them (sd: the nearest core point's)",
                ),
            ),
            quick=True,
            check=Check(
                sd=lambda r: np.array([r.n_clusters, int((np.asarray(r.labels) < 0).sum())], float),
                ref=lambda label, r: np.array([len(set(r.labels_.tolist()) - {-1}), int((r.labels_ < 0).sum())], float),
                rtol=0.0,
                atol=0.0,
            ),
        ),
        Spec(
            "CL3",
            "density",
            "hdbscan p=2",
            ("hdbscan",),
            "20000",
            _points(2 * 10**4, 2),
            lambda d: lambda: _cl().hdbscan(d["sX"]),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: __import__("sklearn.cluster", fromlist=["HDBSCAN"]).HDBSCAN(copy=True).fit(d["X"])
                    ),
                ),
            ),
            check=Check(
                skip="equal mutual reachabilities: scikit-learn's tree follows its Prim's order (tests compare at "
                "min_samples = 1)"
            ),
        ),
        Spec(
            "CL3",
            "density",
            "optics p=2",
            ("optics",),
            "5000",
            _points(5000, 2),
            lambda d: lambda: _cl().optics(d["sX"]),
            (
                Ref(
                    "sklearn", lambda d: lambda: __import__("sklearn.cluster", fromlist=["OPTICS"]).OPTICS().fit(d["X"])
                ),
            ),
            note="scikit-learn starts from the first input point, sd from the first in key order (CL7): other orders",
            check=Check(sd=lambda r: np.asarray(r.core_distances), ref=lambda label, r: r.core_distances_, rtol=1e-12),
        ),
    ]


def _gm_start(k):
    """The same start for every library: equal weights, the data's first k rows as means, unit precisions."""

    def kw(d, sd_side, ctype):
        p = d["X"].shape[1]
        w = np.full(k, 1.0 / k)
        m = d["X"][:k].copy()
        P = {"full": np.repeat(np.eye(p)[None], k, axis=0), "tied": np.eye(p), "diag": np.ones((k, p)),
             "spherical": np.ones(k)}[ctype]  # fmt: skip
        if sd_side:
            import stardust as sd

            return {"weights_init": sd.asarray(w), "means_init": sd.asarray(m), "precisions_init": sd.asarray(P)}
        return {"weights_init": w, "means_init": m, "precisions_init": P}

    return kw


def _mixture_specs():
    def sk_gm():
        return __import__("sklearn.mixture", fromlist=["GaussianMixture"]).GaussianMixture

    def rng():
        return __import__("stardust").random.default_rng(0)

    start = _gm_start(5)
    out = [
        Spec(
            "CL4",
            "mixture",
            "gaussian_mixture full p=10 k=5",
            ("gaussian_mixture",),
            "20000",
            _blobs(2 * 10**4, 10, 5),
            lambda d: lambda: _cl().gaussian_mixture(d["sX"], 5, rng=rng()),
            (Ref("sklearn", lambda d: lambda: sk_gm()(5, random_state=0).fit(d["X"]), threads="follow"),),
            quick=True,
            note="each library's own k-means start; the blobs are well separated, so both reach one fixed point",
            check=Check(sd=lambda r: float(r.lower_bound), ref=lambda label, r: float(r.lower_bound_), rtol=1e-9),
        )
    ]
    for ctype in ("full", "tied", "diag", "spherical"):
        out.append(
            Spec(
                "CL4",
                "mixture",
                f"gaussian_mixture {ctype} p=10 k=5 same start",
                ("gaussian_mixture",),
                "20000",
                _blobs(2 * 10**4, 10, 5),
                lambda d, t=ctype: (
                    lambda: _cl().gaussian_mixture(d["sX"], 5, rng=rng(), covariance_type=t, **start(d, True, t))
                ),
                (
                    Ref(
                        "sklearn",
                        lambda d, t=ctype: lambda: sk_gm()(5, covariance_type=t, **start(d, False, t)).fit(d["X"]),
                        threads="follow",
                    ),
                ),
                note="the same start (weights, means, precisions) on both sides: the same EM iterations",
                check=Check(
                    sd=lambda r: np.array([r.lower_bound, r.n_iter], float),
                    ref=lambda label, r: np.array([r.lower_bound_, r.n_iter_], float),
                    rtol=1e-10,
                ),
            )
        )
    return out


def _metrics_specs():
    from_sk = lambda name: lambda d: lambda: getattr(__import__("sklearn.metrics", fromlist=[name]), name)  # noqa: E731

    def sk(name):
        return getattr(__import__("sklearn.metrics", fromlist=[name]), name)

    out = [
        Spec(
            "CL5",
            "metrics",
            "silhouette_score p=10 k=5",
            ("silhouette_score",),
            "20000",
            _blobs(2 * 10**4, 10, 5),
            lambda d: lambda: _cl().silhouette_score(d["sX"], d["labels"]),
            (
                Ref(
                    "sklearn", lambda d: lambda: from_sk("silhouette_score")(d)()(d["X"], d["labels"]), threads="follow"
                ),
                RRef(
                    "R cluster",
                    "mean(cluster::silhouette(lab, dist(x))[, 3])",
                    data={**R_X, "r_lab": "lab"},
                    setup=R_MATRIX,
                    packages=("cluster",),
                    differs="R needs the 1.6 GB dist object (made in the timing)",
                ),
            ),
            quick=True,
            check=Check(rtol=1e-12),
        ),
        Spec(
            "CL5",
            "metrics",
            "calinski_harabasz_score p=10 k=5",
            ("calinski_harabasz_score",),
            count(10**5),
            _blobs(10**5, 10, 5),
            lambda d: lambda: _cl().calinski_harabasz_score(d["sX"], d["labels"]),
            (Ref("sklearn", lambda d: lambda: sk("calinski_harabasz_score")(d["X"], d["labels"])),),
            check=Check(rtol=1e-12),
        ),
        Spec(
            "CL5",
            "metrics",
            "davies_bouldin_score p=10 k=5",
            ("davies_bouldin_score",),
            count(10**5),
            _blobs(10**5, 10, 5),
            lambda d: lambda: _cl().davies_bouldin_score(d["sX"], d["labels"]),
            (Ref("sklearn", lambda d: lambda: sk("davies_bouldin_score")(d["X"], d["labels"])),),
            check=Check(rtol=1e-12),
        ),
        Spec(
            "CL5",
            "metrics",
            "adjusted_rand_score k=20",
            ("adjusted_rand_score",),
            count(10**6),
            _labelings(10**6),
            lambda d: lambda: _cl().adjusted_rand_score(d["sa"], d["sb"]),
            (Ref("sklearn", lambda d: lambda: from_sk("adjusted_rand_score")(d)()(d["a"], d["b"])),),
            quick=True,
            check=Check(rtol=1e-15),
        ),
        Spec(
            "CL5",
            "metrics",
            "normalized_mutual_info_score k=20",
            ("normalized_mutual_info_score",),
            count(10**6),
            _labelings(10**6),
            lambda d: lambda: _cl().normalized_mutual_info_score(d["sa"], d["sb"]),
            (Ref("sklearn", lambda d: lambda: sk("normalized_mutual_info_score")(d["a"], d["b"])),),
            check=Check(rtol=1e-13),
        ),
        Spec(
            "CL5",
            "metrics",
            "adjusted_mutual_info_score k=20",
            ("adjusted_mutual_info_score",),
            count(10**5),
            _labelings(10**5),
            lambda d: lambda: _cl().adjusted_mutual_info_score(d["sa"], d["sb"]),
            (Ref("sklearn", lambda d: lambda: sk("adjusted_mutual_info_score")(d["a"], d["b"])),),
            quick=True,
            note="the expected MI by the hypergeometric ratio recurrence (scikit-learn: lgamma at every cell count)",
            check=Check(rtol=1e-9),
        ),
        Spec(
            "CL5",
            "metrics",
            "gap_statistic kmeans p=4 k_max=8 B=50",
            ("gap_statistic",),
            count(1000),
            _blobs(1000, 4, 5),
            lambda d: (
                lambda: _cl().gap_statistic(
                    d["sX"], 8, rng=__import__("stardust").random.default_rng(0), n_reference=50
                )
            ),  # fmt: skip
            (
                RRef(
                    "R cluster",
                    "set.seed(1); g <- cluster::clusGap(x, kmeans, K.max = 8, B = 50, verbose = FALSE); "
                    "cluster::maxSE(g$Tab[, 'gap'], g$Tab[, 'SE.sim'])",
                    data=R_X,
                    setup=R_MATRIX,
                    packages=("cluster",),
                    differs="R's kmeans is Hartigan-Wong with one random start, its reference sets another stream",
                ),
            ),
            quick=True,
            check=Check(
                sd=lambda r: float(r.k), ref=lambda label, r: float(np.asarray(r).ravel()[0]), rtol=0.0, atol=0.0
            ),  # fmt: skip
        ),
        Spec(
            "CL5",
            "metrics",
            "gap_statistic pam p=2 k_max=6 B=20",
            ("gap_statistic",),
            "300",
            _blobs(300, 2, 3),
            lambda d: (
                lambda: _cl().gap_statistic(
                    d["sX"], 6, rng=__import__("stardust").random.default_rng(0), n_reference=20, method="pam"
                )
            ),  # fmt: skip
            (
                RRef(
                    "R cluster",
                    "set.seed(1); cluster::clusGap(x, cluster::pam, K.max = 6, B = 20, verbose = FALSE)$Tab[, 'logW']",
                    data=R_X,
                    setup=R_MATRIX,
                    packages=("cluster",),
                ),
            ),
            check=Check(sd=lambda r: np.asarray(r.log_w), ref=lambda label, r: np.asarray(r).ravel(), rtol=1e-12),
        ),
        Spec(
            "CL5",
            "metrics",
            "gower_distance p=6",
            ("gower_distance",),
            "3000",
            _mixed(3000),
            lambda d: lambda: _cl().gower_distance(d["frame"]),
            (
                RRef(
                    "R cluster",
                    "as.vector(cluster::daisy(df, metric = 'gower', type = list(symm = 'c')))",
                    data=_MIXED_R,
                    setup=_MIXED_SETUP,
                    packages=("cluster",),
                ),
            ),
            quick=True,
            check=Check(rtol=1e-14),
        ),
    ]
    return out


def _affinity_specs():
    def sk_ap():
        return __import__("sklearn.cluster", fromlist=["AffinityPropagation"]).AffinityPropagation

    return [
        Spec(
            "CL6",
            "affinity",
            "affinity_propagation p=2",
            ("affinity_propagation",),
            "2000",
            _blobs(2000, 2, 10),
            lambda d: lambda: _cl().affinity_propagation(d["sX"]),
            (Ref("sklearn", lambda d: lambda: sk_ap()(random_state=0).fit(d["X"])),),
            quick=True,
            note="both run the 200 iterations (no convergence); scikit-learn's noise gives other exemplars",
            check=Check(skip="scikit-learn adds random noise to the similarities"),
        ),
        Spec(
            "CL6",
            "affinity",
            "affinity_propagation p=10 preference",
            ("affinity_propagation",),
            "3000",
            _blobs(3000, 10, 6),
            lambda d: lambda: _cl().affinity_propagation(d["sX"], preference=-2000.0, damping=0.9),
            (Ref("sklearn", lambda d: lambda: sk_ap()(preference=-2000.0, damping=0.9, random_state=0).fit(d["X"])),),
            check=Check(
                sd=lambda r: float(r.n_clusters),
                ref=lambda label, r: float(len(r.cluster_centers_indices_)),
                rtol=0.0,
                atol=0.0,
            ),  # fmt: skip
        ),
    ]


_MIXED_R = {"r_a": "a", "r_b": "b", "r_c": "c", "r_d": "d", "r_e": "e", "r_f": "f"}
_MIXED_SETUP = ("df <- data.frame(a = a, b = factor(b), c = c == 1, d = d, e = factor(e), f = f); "
                "df$a[is.nan(df$a)] <- NA")  # fmt: skip


def _mixed(n, seed=20261127):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n]))
        a = g.normal(size=n)
        a[g.random(n) < 0.05] = np.nan
        b = g.integers(0, 4, n)
        c = g.integers(0, 2, n)
        d = g.exponential(size=n)
        e = g.integers(0, 3, n)
        f = g.uniform(size=n)
        frame = sd.frame.DataFrame({"a": [None if np.isnan(v) else float(v) for v in a],
                                    "b": [f"b{v}" for v in b], "c": [bool(v) for v in c], "d": d.tolist(),
                                    "e": [f"e{v}" for v in e], "f": f.tolist()})  # fmt: skip
        return {"frame": frame, "r_a": a, "r_b": b.astype(float), "r_c": c.astype(float), "r_d": d,
                "r_e": e.astype(float), "r_f": f}  # fmt: skip

    return setup


def _larger_specs():
    """CL7: the kernels' larger shapes against the comparators (docs/design/cluster.md §5): k-means where the filter
    (CL49, CL95) carries the scans, the full mixture where the E step's quadratic form grows as p^2 (§9's 40th
    question), the density methods at p = 10 (the KD tree's last dimensions: CL72) and OPTICS at n = 2 x 10^4
    (scikit-learn's Python loop over the steps takes minutes there, so no comparator)."""

    def rng():
        return __import__("stardust").random.default_rng(0)

    start = _gm_start(5)
    return [
        Spec(
            "CL7",
            "partition",
            "kmeans p=50 k=100",
            ("kmeans",),
            count(10**5),
            _blobs(10**5, 50, 100),
            lambda d: lambda: _cl().kmeans(d["sX"], 100, rng=rng(), init=_first(d["sX"], 100)),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: (
                            __import__("sklearn.cluster", fromlist=["KMeans"])
                            .KMeans(100, init=d["X"][:100].copy(), n_init=1, max_iter=300, tol=1e-4)
                            .fit(d["X"])
                        )
                    ),
                    threads="follow",
                ),
            ),
            note="the same start (the data's first 100 rows)",
            check=Check(sd=lambda r: float(r.inertia), ref=lambda label, r: float(r.inertia_), rtol=1e-9),
        ),
        Spec(
            "CL7",
            "mixture",
            "gaussian_mixture full p=50 k=5 same start",
            ("gaussian_mixture",),
            "20000",
            _blobs(2 * 10**4, 50, 5),
            lambda d: (
                lambda: _cl().gaussian_mixture(d["sX"], 5, rng=rng(), covariance_type="full", **start(d, True, "full"))
            ),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: (
                            __import__("sklearn.mixture", fromlist=["GaussianMixture"])
                            .GaussianMixture(5, covariance_type="full", **start(d, False, "full"))
                            .fit(d["X"])
                        )
                    ),
                    threads="follow",
                ),
            ),
            note="the same start on both sides: the same EM iterations",
            check=Check(
                sd=lambda r: np.array([r.lower_bound, r.n_iter], float),
                ref=lambda label, r: np.array([r.lower_bound_, r.n_iter_], float),
                rtol=1e-10,
            ),
        ),
        Spec(
            "CL7",
            "density",
            "dbscan p=10 eps=0.9",
            ("dbscan",),
            "20000",
            _points(2 * 10**4, 10),
            lambda d: lambda: _cl().dbscan(d["sX"], eps=0.9, min_samples=5),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: (
                            __import__("sklearn.cluster", fromlist=["DBSCAN"])
                            .DBSCAN(eps=0.9, min_samples=5)
                            .fit(d["X"])
                        )
                    ),
                    differs="border points join the first cluster that reaches them (sd: the nearest core point's)",
                ),
            ),
            check=Check(skip="border points: the clusters' sizes differ by the points scikit-learn gives away"),
        ),
        Spec(
            "CL7",
            "density",
            "hdbscan p=10",
            ("hdbscan",),
            "20000",
            _points(2 * 10**4, 10),
            lambda d: lambda: _cl().hdbscan(d["sX"]),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: __import__("sklearn.cluster", fromlist=["HDBSCAN"]).HDBSCAN(copy=True).fit(d["X"])
                    ),
                ),
            ),
            check=Check(skip="equal mutual reachabilities: scikit-learn's tree follows its Prim's order"),
        ),
        Spec(
            "CL7",
            "density",
            "optics p=2",
            ("optics",),
            "20000",
            _points(2 * 10**4, 2),
            lambda d: lambda: _cl().optics(d["sX"]),
            note="no comparator: scikit-learn's step loop in Python takes minutes at this size",
        ),
    ]


def _first(sX, k):
    import stardust as sd

    return sd.asarray(np.asarray(sX)[:k].copy())


SPECS = (
    _hierarchy_specs()
    + _kmeans_specs()
    + _partition_specs()
    + _density_specs()
    + _mixture_specs()
    + _metrics_specs()
    + _affinity_specs()
    + _larger_specs()
)


def _public(names) -> bool:
    import stardust as sd

    return all(n in sd.cluster.__all__ for n in names)


def _loaded(setup):
    """The setup, after loading the comparators' libraries: threadpoolctl limits the OpenMP pools it finds when a
    measurement starts, so scikit-learn imported lazily inside the first measurement of a worker ran its OpenMP
    threads unlimited at "1 thread" (k-means 14.5 ms against 30.3 ms on one thread, milestone CL7)."""

    def run():
        import scipy.cluster.hierarchy  # noqa: F401
        import sklearn.cluster  # noqa: F401
        import sklearn.metrics  # noqa: F401
        import sklearn.mixture  # noqa: F401

        return setup()

    return run


def cases() -> list[Case]:
    return [
        Case(
            area="cluster",
            group=s.group,
            name=s.name,
            size=s.size,
            setup=_loaded(s.setup),
            sd=s.sd,
            refs=s.refs,
            quick=s.quick,
            note=s.note,
            check=s.check,
        )
        for s in SPECS
        if _public(s.needs)
    ]


TITLES = {
    "hierarchy": "계층 군집",
    "partition": "분할 (`kmeans`, `minibatch_kmeans`, `bisecting_kmeans`, `kmedoids`, `fuzzy_cmeans`)",
    "density": "밀도 군집",
    "mixture": "가우스 혼합",
    "metrics": "지표 (`gap_statistic`, `gower_distance` 포함)",
    "affinity": "Affinity Propagation",
}
