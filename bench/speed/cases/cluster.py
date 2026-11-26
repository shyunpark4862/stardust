"""sd.cluster against scipy 1.18.1, scikit-learn 1.9.1 and R 4.6.2's stats and cluster (docs/design/cluster.md §5,
§7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the sd.cluster names
it needs, and cases() keeps a spec once its names are public (CL1 the hierarchy, CL2 k-means and k-medoids, CL3
the density methods, CL4 the mixtures, CL5 the indices). The sizes are those of docs/design/cluster.md §5, where
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
        return {
            "X": X,
            "sX": sd.asarray(X),
            "init": init,
            "sinit": sd.asarray(init),
            "labels": g.integers(0, k, n),
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
    for method in ("single", "complete", "average", "ward", "centroid"):
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
                check=Check(rtol=1e-12),
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
    return out


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
            check=Check(skip="the labels of equal partitions are compared by the tests"),
        ),
    ]


def _mixture_specs():
    return [
        Spec(
            "CL4",
            "mixture",
            "gaussian_mixture full p=10 k=5",
            ("gaussian_mixture",),
            "20000",
            _blobs(2 * 10**4, 10, 5),
            lambda d: lambda: _cl().gaussian_mixture(d["sX"], 5, rng=__import__("stardust").random.default_rng(0)),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: (
                            __import__("sklearn.mixture", fromlist=["GaussianMixture"])
                            .GaussianMixture(5, random_state=0)
                            .fit(d["X"])
                        )
                    ),
                    threads="follow",
                ),
            ),
            check=Check(skip="different random starts"),
        )
    ]


def _metrics_specs():
    from_sk = lambda name: lambda d: lambda: getattr(__import__("sklearn.metrics", fromlist=[name]), name)  # noqa: E731
    return [
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
            ),
            quick=True,
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
            check=Check(rtol=1e-15),
        ),
    ]


SPECS = _hierarchy_specs() + _kmeans_specs() + _density_specs() + _mixture_specs() + _metrics_specs()


def _public(names) -> bool:
    import stardust as sd

    return all(n in sd.cluster.__all__ for n in names)


def cases() -> list[Case]:
    return [
        Case(
            area="cluster",
            group=s.group,
            name=s.name,
            size=s.size,
            setup=s.setup,
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
    "partition": "분할 (`kmeans`)",
    "density": "밀도 군집",
    "mixture": "가우스 혼합",
    "metrics": "지표",
}
